#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

// Public C API for NCCL Xfer's copy/staging reshard primitive. The API is
// provided by an installed NCCL Xfer shared library (libnccl_Xfer.so). This
// file builds dispatcher stubs when the header/library are unavailable.
#if defined(NCCL_HAS_XFER_COPY_API)
#include "nccl_Xfer.h"
#endif

namespace {

#if defined(NCCL_HAS_XFER_COPY_API)

using c10d::getNcclErrorDetailStr;
using c10d::ncclGetErrorWithVersion;

ncclDataType_t to_nccl_dtype(at::ScalarType st) {
  return c10d::getNcclDataType(st);
}

int64_t checked_numel(at::IntArrayRef shape, const char* name) {
  int64_t numel = 1;
  for (const auto d : c10::irange(shape.size())) {
    TORCH_CHECK(
        shape[d] >= 0,
        "dtensor._xg_redistribute: ",
        name,
        " must be non-negative, got ",
        shape);
    TORCH_CHECK(
        numel <= std::numeric_limits<int64_t>::max() /
                std::max<int64_t>(shape[d], 1),
        "dtensor._xg_redistribute: ",
        name,
        " overflows int64_t: ",
        shape);
    numel *= shape[d];
  }
  return numel;
}

bool has_tensor(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor->defined();
}

const at::Tensor& checked_tensor(
    const std::optional<at::Tensor>& tensor,
    at::IntArrayRef local_shape,
    const char* name) {
  TORCH_CHECK(
      has_tensor(tensor), "dtensor._xg_redistribute: ", name, " tensor is absent");
  const auto& value = *tensor;
  TORCH_CHECK(
      value.is_cuda(), "dtensor._xg_redistribute: ", name, " tensor must be CUDA");
  TORCH_CHECK(
      value.is_contiguous(),
      "dtensor._xg_redistribute: ",
      name,
      " tensor must be contiguous");
  const int64_t expected_numel = checked_numel(local_shape, name);
  TORCH_CHECK(
      value.numel() == expected_numel,
      "dtensor._xg_redistribute: ",
      name,
      " tensor has ",
      value.numel(),
      " elements but ",
      name,
      "_local_shape=",
      local_shape,
      " describes ",
      expected_numel,
      " elements");
  return value;
}

void fill_mesh(
    ncclXferReshardMesh_t& mesh,
    at::IntArrayRef dims,
    int64_t start_rank,
    at::IntArrayRef placement,
    const char* name) {
  TORCH_CHECK(
      dims.size() == 2,
      "dtensor._xg_redistribute: ",
      name,
      "_mesh_dims must have length 2, got ",
      dims.size());
  TORCH_CHECK(
      placement.size() == 2,
      "dtensor._xg_redistribute: ",
      name,
      "_placements must have length 2, got ",
      placement.size());
  TORCH_CHECK(
      dims[0] > 0 && dims[1] > 0,
      "dtensor._xg_redistribute: ",
      name,
      "_mesh_dims entries must be positive, got ",
      dims);
  mesh.dims[0] = static_cast<int>(dims[0]);
  mesh.dims[1] = static_cast<int>(dims[1]);
  mesh.start_rank = static_cast<int>(start_rank);
  mesh.placement[0] = static_cast<int>(placement[0]);
  mesh.placement[1] = static_cast<int>(placement[1]);
}

void fill_local_shape(
    size_t out[NCCLXFER_RESHARD_MAX_TENSOR_DIMS],
    at::IntArrayRef local_shape,
    bool is_role) {
  for (const auto d : c10::irange(local_shape.size())) {
    out[d] = is_role ? static_cast<size_t>(local_shape[d]) : 0;
  }
}

#endif // NCCL_HAS_XFER_COPY_API

void nccl_xg_redistribute_init() {
#if defined(NCCL_HAS_XFER_COPY_API)
  C10D_NCCL_CHECK(
      ncclXferReshardInit(/*config=*/nullptr),
      "ncclXferReshardInit failed in dtensor._xg_redistribute");
#else
  TORCH_CHECK(
      false,
      "torch.distributed.tensor.init_xfer requires NCCL Xfer copy/staging API "
      "(nccl_Xfer.h and libnccl_Xfer.so). NCCL_HAS_XFER_COPY_API was not "
      "defined at build time.");
#endif
}

void nccl_xg_redistribute_finalize() {
#if defined(NCCL_HAS_XFER_COPY_API)
  auto rc = ncclXferReshardFinalize();
  if (rc != ncclSuccess) {
    LOG(WARNING) << "ncclXferReshardFinalize returned " << rc
                 << " -- dtensor._xg_redistribute resources may not be fully released.";
  }
#else
  // Stub when the copy/staging API isn't compiled in -- nothing to release.
#endif
}

void nccl_xg_redistribute(
    const std::optional<at::Tensor>& src_tensor,
    const std::optional<at::Tensor>& dst_tensor,
    at::IntArrayRef src_local_shape,
    at::IntArrayRef src_mesh_dims,
    int64_t src_mesh_start_rank,
    at::IntArrayRef src_placement,
    at::IntArrayRef dst_local_shape,
    at::IntArrayRef dst_mesh_dims,
    int64_t dst_mesh_start_rank,
    at::IntArrayRef dst_placement,
    int64_t comm_ptr) {
#if defined(NCCL_HAS_XFER_COPY_API)
  const int64_t ndims = static_cast<int64_t>(src_local_shape.size());
  TORCH_CHECK(
      ndims >= 1 && ndims <= NCCLXFER_RESHARD_MAX_TENSOR_DIMS,
      "dtensor._xg_redistribute: tensor rank must be in [1, ",
      NCCLXFER_RESHARD_MAX_TENSOR_DIMS,
      "], got ",
      ndims);
  TORCH_CHECK(
      static_cast<int64_t>(dst_local_shape.size()) == ndims,
      "dtensor._xg_redistribute: dst_local_shape rank (",
      dst_local_shape.size(),
      ") must match src_local_shape rank (",
      ndims,
      ")");
  TORCH_CHECK(
      comm_ptr != 0,
      "dtensor._xg_redistribute: comm must be a non-null ncclComm_t pointer");

  const bool is_src_role = has_tensor(src_tensor);
  const bool is_dst_role = has_tensor(dst_tensor);
  TORCH_CHECK(
      is_src_role || is_dst_role,
      "dtensor._xg_redistribute: at least one of src or dst tensor must be present");

  checked_numel(src_local_shape, "src_local_shape");
  checked_numel(dst_local_shape, "dst_local_shape");

  const at::Tensor* device_tensor = nullptr;
  if (is_src_role) {
    device_tensor = &checked_tensor(src_tensor, src_local_shape, "src");
  }
  if (is_dst_role) {
    const auto& dst = checked_tensor(dst_tensor, dst_local_shape, "dst");
    if (device_tensor == nullptr) {
      device_tensor = &dst;
    } else {
      TORCH_CHECK(
          device_tensor->scalar_type() == dst.scalar_type(),
          "dtensor._xg_redistribute: src dtype ",
          device_tensor->scalar_type(),
          " must match dst dtype ",
          dst.scalar_type());
      TORCH_CHECK(
          device_tensor->device() == dst.device(),
          "dtensor._xg_redistribute: src device ",
          device_tensor->device(),
          " must match dst device ",
          dst.device());
    }
  }
  TORCH_INTERNAL_ASSERT(device_tensor != nullptr);

  c10::cuda::CUDAGuard guard(device_tensor->device());
  const auto stream =
      at::cuda::getCurrentCUDAStream(device_tensor->device().index());
  auto comm = reinterpret_cast<ncclComm_t>(static_cast<uintptr_t>(comm_ptr));

  ncclXferReshardMesh_t src_mesh{};
  ncclXferReshardMesh_t dst_mesh{};
  fill_mesh(src_mesh, src_mesh_dims, src_mesh_start_rank, src_placement, "src");
  fill_mesh(dst_mesh, dst_mesh_dims, dst_mesh_start_rank, dst_placement, "dst");

  ncclXferDistTensor_t src{};
  ncclXferDistTensor_t dst{};
  const auto dtype = to_nccl_dtype(device_tensor->scalar_type());
  src.ndims = static_cast<int>(ndims);
  src.dtype = dtype;
  src.mesh = &src_mesh;
  src.data_ptr = is_src_role ? src_tensor->data_ptr() : nullptr;
  fill_local_shape(src.local_shape, src_local_shape, is_src_role);

  dst.ndims = static_cast<int>(ndims);
  dst.dtype = dtype;
  dst.mesh = &dst_mesh;
  dst.data_ptr = is_dst_role ? dst_tensor->data_ptr() : nullptr;
  fill_local_shape(dst.local_shape, dst_local_shape, is_dst_role);

  C10D_NCCL_CHECK(
      ncclXferReshard(comm, &src, &dst, stream),
      "ncclXferReshard failed in dtensor._xg_redistribute");
#else
  TORCH_CHECK(
      false,
      "torch.distributed.tensor._xg_redistribute requires NCCL Xfer copy/staging API "
      "(nccl_Xfer.h and libnccl_Xfer.so). NCCL_HAS_XFER_COPY_API was not "
      "defined at build time.");
#endif
}

TORCH_LIBRARY_FRAGMENT(dtensor_xfer, m) {
  m.def("init() -> ()");
  m.def("finalize() -> ()");
  m.def(
      "reshard(Tensor? src, Tensor(a!)? dst, int[] src_local_shape, int[] src_mesh_dims, int src_mesh_start_rank, int[] src_placement, int[] dst_local_shape, int[] dst_mesh_dims, int dst_mesh_start_rank, int[] dst_placement, int comm) -> ()");
}

TORCH_LIBRARY_IMPL(dtensor_xfer, CompositeExplicitAutograd, m) {
  m.impl("init", nccl_xg_redistribute_init);
  m.impl("finalize", nccl_xg_redistribute_finalize);
  m.impl("reshard", nccl_xg_redistribute);
}

} // namespace
