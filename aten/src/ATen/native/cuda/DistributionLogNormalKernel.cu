#include <ATen/AccumulateType.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <functional>
#include <utility>

#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#include <THC/THCGeneral.h>
#include <THC/THCApply.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace at {
namespace native {

void log_normal_kernel(
    TensorIteratorBase& iter,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(
      gen, cuda::detail::getDefaultCUDAGenerator());
  at::native::templates::cuda::log_normal_kernel(iter, mean, std, generator);
}

REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel);

} // namespace native
} // namespace at
