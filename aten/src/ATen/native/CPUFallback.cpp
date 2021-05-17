#include <ATen/native/CPUFallback.h>

#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <ATen/Functions.h>
#include <torch/library.h>

namespace at { namespace native {

// convenience helper for converting tensors to cpu

std::vector<at::Tensor> to_cpu(const at::TensorList& tensors) {
    // We can't just call at::to_cpu() on the entire list of Tensors
    // Because it will break on undefined tensors. Separate out undefined tensors first.
    std::vector<at::Tensor> cpu_tensors(tensors.size());
    std::vector<at::Tensor> valid_tensors;
    std::vector<bool> to_translate(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const at::Tensor& tensor = tensors[i];
        if (tensor.defined()) {
            to_translate[i] = true;
            valid_tensors.push_back(tensor);
        } else {
            cpu_tensors[i] = tensor;
        }
    }
    auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
    for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
        if (to_translate[i]) {
            cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
        }
    }
  return cpu_tensors;
}


void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  std::vector<at::Tensor> tensor_args;
  std::vector<int> tensor_args_indices;

  // Step 1: Convert all non-CPU tensor inputs into CPU tensors
  // and put them on the stack at the correct indices.
  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    } else if (ivalue.isTensorList()) {
      // I'm worried about the ivalue storing a reference to the std::vector temporary that I create.
      // To be extra safe I'm converting to a c10::List, IValue will take ownership of the items in the list.
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(to_cpu(ivalue.toTensorList().vec())));
      (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
    }
  }
  auto cpu_tensors = to_cpu(tensor_args);

  for (auto i = 0; i < tensor_args_indices.size(); ++i) {
    //auto cpu_ivalue = c10::IValue(cpu_tensors[i]);
    auto idx = tensor_args_indices[i];
    //(*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
    (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);
  }

  // Step 2: Call the underlying CPU implementation of the operator
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

  // Step 3: We need to take special care to handle mutable aliases properly:
  // If any input tensors are mutable aliases, we need to
  // directly copy the updated data on the CPU tensors back to the original inputs.
  for (int64_t i = 0; i < tensor_args_indices.size(); ++i) {
    auto tensor_idx = tensor_args_indices[i];
    const auto& alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info.has_value() && alias_info.value().isWrite()) {
      at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
    }
  }

  // Step 4: Convert any CPU output tensors back to the original input device.
  // For mutable alias'd outputs, we also need to take special care
  // to move the ORIGINAL input tensor back onto the stack, in place of
  // the temporary CPU output tensor that we created.
  //
  // Also note that we are incapable of handling immutable alises properly.
  // Why?
  // Schemas with an immutable alias'd tensor outputs correspond to view operators.
  // For example, the `view_as` schema from native_functions.yaml:
  // `view_as(Tensor(a) self, Tensor other) -> Tensor(a)`
  // We can't handle these ops properly, because view ops are supposed to return
  // a NEW tensor that shares the SAME storage as the original tensor.
  // However, the new tensor that we created cannot share the same storage,
  // since it lives on CPU and the original tensor lives on a different device.
  // Because of that, we treat immutable aliases the same way that we treat non-aliases:
  // as a fresh tensor that has entirely new storage.
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  for (int64_t idx = 0; idx < returns.size(); ++idx) {
    if (returns[idx].isTensor()) {
      const auto& return_tens = returns[idx].toTensor();
      if (return_tens.defined()) {
        const auto& alias_info = schema_returns[idx].alias_info();
        if (alias_info.has_value() && alias_info.value().isWrite()) {
          // mutable alias case: move the input ivalue directly onto the stack
          // in place of the existing cpu output tensor.
          bool found_alias = false;
          for (int64_t i = 0; i < tensor_args_indices.size(); ++i) {
            auto input_tensor_idx = tensor_args_indices[i];
            const auto& input_tensor = cpu_tensors[i];
            const auto& input_alias_info = schema_args[input_tensor_idx].alias_info();
            if (input_tensor.defined() && alias_info == input_alias_info) {
              // We've found the original input tensor that aliases with the current output.
              // Wrap it in an IValue and put it directly on the stack.
              (*stack)[returns_begin + idx] = c10::IValue(tensor_args[i]);
              found_alias = true;
              break;
            }
          }
          TORCH_CHECK(found_alias, "The operator ", op.schema().operator_name(), " appears to have invalid alias information. ",
                      "Found a return tensor argument with a mismatched mutable alias: ", schema_returns[idx]);
        } else {
          // copy case: copy the cpu output tensor to the original device.
          auto tgt_device = tensor_args[0].device();
          (*stack)[returns_begin + idx] = c10::IValue(returns[idx].toTensor().to(tgt_device));
        }
      }
    }
  }
}

} // namespace native
} // namespace at
