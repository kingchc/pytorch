#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// Public C API for the user-window NCCL reshard primitive (originated in
// the nccl-rl repository's `user-window-api-only` branch; expected to land
// upstream in NCCL).  Gated behind `NCCL_HAS_RESHARD_API` so the binding
// can be staged ahead of the upstream merge: define the macro at build
// time once the NCCL distribution exposes `nccl_reshard.h` and the
// `ncclReshardUserWindow` symbol.
#if defined(NCCL_HAS_RESHARD_API)
#if !defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT)
#error "NCCL_HAS_RESHARD_API requires NCCL_HAS_SYMMEM_DEVICE_SUPPORT (NCCL >= 2.28)"
#endif
#include "nccl_reshard.h"
#endif

#include <mutex>

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

#if defined(NCCL_HAS_RESHARD_API)

namespace {

ncclDataType_t to_nccl_dtype(at::ScalarType st) {
  switch (st) {
    case at::kByte:     return ncclUint8;
    case at::kChar:     return ncclInt8;
    case at::kFloat8_e4m3fn: return ncclFloat8e4m3;
    case at::kFloat8_e5m2:   return ncclFloat8e5m2;
    case at::kHalf:     return ncclFloat16;
    case at::kBFloat16: return ncclBfloat16;
    case at::kInt:      return ncclInt32;
    case at::kLong:     return ncclInt64;
    case at::kFloat:    return ncclFloat32;
    case at::kDouble:   return ncclFloat64;
    default:
      TORCH_CHECK(
          false,
          "nccl_mxn_cast: unsupported dtype ",
          st,
          " (supported: uint8/int8/float8_e4m3fn/float8_e5m2/"
          "float16/bfloat16/int32/int64/float32/float64)");
  }
}

void fill_mesh(
    ::ncclReshardMesh_t& m,
    at::IntArrayRef dims,
    int64_t start_rank,
    at::IntArrayRef placement) {
  TORCH_CHECK(
      dims.size() == 2,
      "nccl_mxn_cast: mesh_dims must have length 2, got ",
      dims.size());
  TORCH_CHECK(
      placement.size() == 2,
      "nccl_mxn_cast: placement must have length 2, got ",
      placement.size());
  m.dims[0] = static_cast<int>(dims[0]);
  m.dims[1] = static_cast<int>(dims[1]);
  m.start_rank = static_cast<int>(start_rank);
  m.placement[0] = static_cast<int>(placement[0]);
  m.placement[1] = static_cast<int>(placement[1]);
}

} // namespace

void nccl_mxn_cast(
    at::Tensor& buf,
    at::IntArrayRef src_local_shape,
    at::IntArrayRef src_mesh_dims,
    int64_t src_mesh_start_rank,
    at::IntArrayRef src_placement,
    at::IntArrayRef dst_local_shape,
    at::IntArrayRef dst_mesh_dims,
    int64_t dst_mesh_start_rank,
    at::IntArrayRef dst_placement,
    const std::string& group_name) {
  TORCH_CHECK(buf.is_cuda(), "nccl_mxn_cast: buf must be a CUDA tensor");
  TORCH_CHECK(buf.is_contiguous(), "nccl_mxn_cast: buf must be contiguous");
  const int ndims = static_cast<int>(src_local_shape.size());
  TORCH_CHECK(
      ndims >= 1 && ndims <= NCCL_RESHARD_MAX_TENSOR_DIMS,
      "nccl_mxn_cast: ndims must be in [1, ",
      NCCL_RESHARD_MAX_TENSOR_DIMS,
      "], got ",
      ndims);
  TORCH_CHECK(
      static_cast<int>(dst_local_shape.size()) == ndims,
      "nccl_mxn_cast: dst_local_shape rank (",
      dst_local_shape.size(),
      ") must match src_local_shape rank (",
      ndims,
      ")");

  // A rank participates as "src" iff its src_local_shape is fully positive,
  // and as "dst" iff its dst_local_shape is fully positive. Zero-shape on a
  // side signals the rank is absent there; ncclDistTensor_t::data_ptr is
  // then NULL, matching the user-window API contract for non-participating
  // ranks (cf. PyTorch DTensor's size-0 local tensor).
  int64_t src_numel = 1;
  int64_t dst_numel = 1;
  for (int d = 0; d < ndims; ++d) {
    TORCH_CHECK(
        src_local_shape[d] >= 0 && dst_local_shape[d] >= 0,
        "nccl_mxn_cast: local shapes must be non-negative; got src=",
        src_local_shape,
        ", dst=",
        dst_local_shape);
    src_numel *= src_local_shape[d];
    dst_numel *= dst_local_shape[d];
  }
  const bool is_src_role = src_numel > 0;
  const bool is_dst_role = dst_numel > 0;
  TORCH_CHECK(
      is_src_role || is_dst_role,
      "nccl_mxn_cast: at least one of src_local_shape or dst_local_shape "
      "must be non-empty; got src=",
      src_local_shape,
      ", dst=",
      dst_local_shape);

  int64_t required_numel = 0;
  if (is_src_role) {
    required_numel = std::max(required_numel, src_numel);
  }
  if (is_dst_role) {
    required_numel = std::max(required_numel, dst_numel);
  }
  TORCH_CHECK(
      buf.numel() >= required_numel,
      "nccl_mxn_cast: buf.numel() (",
      buf.numel(),
      ") must be >= max(src_numel, dst_numel) = ",
      required_numel,
      " (src_numel=",
      src_numel,
      ", dst_numel=",
      dst_numel,
      ")");

  // The buffer must live in NCCL symmetric memory; ncclReshardUserWindow
  // operates on the registered ncclWindow_t.
  auto symm_mem = c10d::symmetric_memory::rendezvous(buf, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "nccl_mxn_cast: buf must be allocated via NCCL symmetric memory "
      "(use symm_mem.empty with NCCL backend)");
  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(
      nccl_hdl != nullptr,
      "nccl_mxn_cast: requires NCCL symmetric memory backend");

  c10::cuda::CUDAGuard guard(buf.device());
  const auto device_index = buf.device().index();
  auto caller_stream = at::cuda::getCurrentCUDAStream(device_index);
  auto stream = caller_stream;
  const auto raw_caller_stream = caller_stream.stream();
  const bool use_side_stream =
      caller_stream == at::cuda::getDefaultCUDAStream(device_index) ||
      raw_caller_stream == nullptr ||
      raw_caller_stream == cudaStreamLegacy ||
      raw_caller_stream == cudaStreamPerThread;
  at::cuda::CUDAEvent side_stream_start(cudaEventDisableTiming);
  at::cuda::CUDAEvent side_stream_done(cudaEventDisableTiming);
  at::Tensor buf_stash;

  if (use_side_stream) {
    // ncclReshardUserWindow runs default-stream callers on a library-owned
    // side stream that does not wait for prior default-stream work.  Bridge
    // the caller stream ourselves so the op preserves PyTorch stream ordering.
    stream = at::cuda::getStreamFromPool(
        /*isHighPriority=*/false, device_index);
    buf_stash = buf;
    side_stream_start.record(caller_stream);
    side_stream_start.block(stream);
  }

  auto& manager =
      c10d::symmetric_memory::NCCLDevCommManager::get(buf.device());
  ncclComm_t comm = manager.get_comm(group_name);
  ncclWindow_t window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "nccl_mxn_cast: NCCL window is null");

  // ncclReshardInit is documented as idempotent.  Pass nullptr for the
  // config to use library defaults; expose a config knob later if a
  // caller needs to override maxCta.
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    C10D_NCCL_CHECK(
        ncclReshardInit(/*config=*/nullptr),
        "ncclReshardInit failed in nccl_mxn_cast");
  });

  ::ncclReshardMesh_t src_mesh{};
  ::ncclReshardMesh_t dst_mesh{};
  fill_mesh(src_mesh, src_mesh_dims, src_mesh_start_rank, src_placement);
  fill_mesh(dst_mesh, dst_mesh_dims, dst_mesh_start_rank, dst_placement);

  // Pack ncclDistTensor_t descriptors. Both descriptors are required on
  // every rank — they each carry one side's mesh, which the library reads
  // everywhere to compute who-talks-to-whom. data_ptr=NULL signals the
  // rank does not own a tile on that side.
  ::ncclDistTensor_t src{};
  ::ncclDistTensor_t dst{};
  const ncclDataType_t dtype = to_nccl_dtype(buf.scalar_type());
  src.ndims = ndims;
  src.dtype = dtype;
  src.mesh = &src_mesh;
  src.data_ptr = is_src_role ? buf.data_ptr() : nullptr;
  for (int d = 0; d < ndims; ++d) {
    src.local_shape[d] =
        is_src_role ? static_cast<size_t>(src_local_shape[d]) : 0;
  }
  dst.ndims = ndims;
  dst.dtype = dtype; // src.dtype must equal dst.dtype (same in-place buffer)
  dst.mesh = &dst_mesh;
  dst.data_ptr = is_dst_role ? buf.data_ptr() : nullptr;
  for (int d = 0; d < ndims; ++d) {
    dst.local_shape[d] =
        is_dst_role ? static_cast<size_t>(dst_local_shape[d]) : 0;
  }

  C10D_NCCL_CHECK(
      ::ncclReshardUserWindow(comm, window, &src, &dst, stream),
      "ncclReshardUserWindow failed in nccl_mxn_cast");

  if (use_side_stream) {
    side_stream_done.record(stream);
    side_stream_done.block(caller_stream);
    // Keep the buffer storage live until the side stream has rejoined the
    // caller stream. This mirrors ProcessGroupNCCL's avoid-recordStream path.
    (void)buf_stash;
  }
}

void nccl_mxn_cast_finalize() {
  // Best-effort release of the reshard library's internal caches and
  // transpose buffer.  Idempotent: subsequent calls are no-ops.  Safe
  // to invoke from an atexit handler — the library is documented to
  // accept Finalize without prior Init.  Errors during shutdown are
  // logged, never thrown.
  static std::once_flag fin_flag;
  std::call_once(fin_flag, []() {
    auto rc = ::ncclReshardFinalize();
    if (rc != ncclSuccess) {
      LOG(WARNING) << "ncclReshardFinalize returned " << rc
                   << " — symm_mem.mxn_cast resources may not be fully released.";
    }
  });
}

#else // !NCCL_HAS_RESHARD_API

void nccl_mxn_cast(
    at::Tensor& /*buf*/,
    at::IntArrayRef /*src_local_shape*/,
    at::IntArrayRef /*src_mesh_dims*/,
    int64_t /*src_mesh_start_rank*/,
    at::IntArrayRef /*src_placement*/,
    at::IntArrayRef /*dst_local_shape*/,
    at::IntArrayRef /*dst_mesh_dims*/,
    int64_t /*dst_mesh_start_rank*/,
    at::IntArrayRef /*dst_placement*/,
    const std::string& /*group_name*/) {
  TORCH_CHECK(
      false,
      "nccl_mxn_cast requires NCCL with the user-window reshard API "
      "(originated in the nccl-rl repository; pending upstream NCCL merge). "
      "NCCL_HAS_RESHARD_API was not defined at build time.");
}

void nccl_mxn_cast_finalize() {
  // Stub when the reshard API isn't compiled in — nothing to release.
}

#endif // NCCL_HAS_RESHARD_API

} // namespace c10d::nccl_extension
