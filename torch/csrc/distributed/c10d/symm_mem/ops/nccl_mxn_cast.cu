#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/macros.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>

// Public C API for the user-window NCCL reshard primitive. It is provided
// by an installed NCCL Xfer shared library (libnccl_xfer.so), or by NCCL
// itself once the symbols land in libnccl.so.
// Gated behind `NCCL_HAS_RESHARD_API` so the binding can build a stub when
// the header and shared-library symbols are unavailable.
#if defined(NCCL_HAS_RESHARD_API)
#if !defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT)
#error "NCCL_HAS_RESHARD_API requires NCCL_HAS_SYMMEM_DEVICE_SUPPORT (NCCL >= 2.28)"
#endif
#include "nccl_xfer.h"
#endif

#include <mutex>

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

#if defined(NCCL_HAS_RESHARD_API)

namespace {

ncclDataType_t to_nccl_dtype(at::ScalarType st) {
  return c10d::getNcclDataType(st);
}

void fill_mesh(
    ::ncclXferMesh_t& m,
    at::IntArrayRef dims,
    int64_t start_rank) {
  TORCH_CHECK(
      dims.size() == 2,
      "nccl_mxn_cast: mesh_dims must have length 2, got ",
      dims.size());
  m.dims[0] = static_cast<int>(dims[0]);
  m.dims[1] = static_cast<int>(dims[1]);
  m.startRank = static_cast<int>(start_rank);
}

void fill_placements(
    int placements[NCCLXFER_MESH_NDIMS],
    at::IntArrayRef placement) {
  TORCH_CHECK(
      placement.size() == 2,
      "nccl_mxn_cast: placement must have length 2, got ",
      placement.size());
  placements[0] = static_cast<int>(placement[0]);
  placements[1] = static_cast<int>(placement[1]);
}

std::mutex xfer_handle_mutex;
::ncclXferHandle_t xfer_handle = nullptr;
bool xfer_finalized = false;

::ncclXferHandle_t get_xfer_handle() {
  std::lock_guard<std::mutex> lock(xfer_handle_mutex);
  TORCH_CHECK(
      !xfer_finalized,
      "nccl_mxn_cast: nccl_mxn_cast_finalize() has already finalized "
      "the NCCL Xfer handle");
  if (xfer_handle == nullptr) {
    C10D_NCCL_CHECK(
        ::ncclXferInit(&xfer_handle, /*config=*/nullptr),
        "ncclXferInit failed in nccl_mxn_cast");
  }
  return xfer_handle;
}

::ncclXferHandle_t release_xfer_handle() {
  std::lock_guard<std::mutex> lock(xfer_handle_mutex);
  xfer_finalized = true;
  auto* handle = xfer_handle;
  xfer_handle = nullptr;
  return handle;
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
      ndims >= 1 && ndims <= NCCLXFER_MAX_TENSOR_DIMS,
      "nccl_mxn_cast: ndims must be in [1, ",
      NCCLXFER_MAX_TENSOR_DIMS,
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
  // side signals the rank is absent there; ncclXferDistTensor_t::dataPtr is
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

  // The buffer must live in NCCL symmetric memory; ncclXferReshardWithWindow
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
    // ncclXferReshardWithWindow runs default-stream callers on a library-owned
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

  ::ncclXferMesh_t src_mesh{};
  ::ncclXferMesh_t dst_mesh{};
  fill_mesh(src_mesh, src_mesh_dims, src_mesh_start_rank);
  fill_mesh(dst_mesh, dst_mesh_dims, dst_mesh_start_rank);

  // Pack ncclXferDistTensor_t descriptors. Both descriptors are required on
  // every rank — they each carry one side's mesh, which the library reads
  // everywhere to compute who-talks-to-whom. Placements live on the tensor
  // descriptors, while dataPtr=NULL signals the rank does not own a tile on
  // that side.
  ::ncclXferDistTensor_t src{};
  ::ncclXferDistTensor_t dst{};
  const ncclDataType_t dtype = to_nccl_dtype(buf.scalar_type());
  src.ndims = ndims;
  src.dtype = dtype;
  src.mesh = &src_mesh;
  src.dataPtr = is_src_role ? buf.data_ptr() : nullptr;
  fill_placements(src.placements, src_placement);
  for (int d = 0; d < ndims; ++d) {
    src.localShape[d] =
        is_src_role ? static_cast<size_t>(src_local_shape[d]) : 0;
  }
  dst.ndims = ndims;
  dst.dtype = dtype; // src.dtype must equal dst.dtype (same in-place buffer)
  dst.mesh = &dst_mesh;
  dst.dataPtr = is_dst_role ? buf.data_ptr() : nullptr;
  fill_placements(dst.placements, dst_placement);
  for (int d = 0; d < ndims; ++d) {
    dst.localShape[d] =
        is_dst_role ? static_cast<size_t>(dst_local_shape[d]) : 0;
  }

  C10D_NCCL_CHECK(
      ::ncclXferReshardWithWindow(
          get_xfer_handle(), comm, window, &src, &dst, stream),
      "ncclXferReshardWithWindow failed in nccl_mxn_cast");

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
  // transpose buffer. Idempotent: subsequent calls are no-ops. Call this
  // before tearing down the process group; it is not suitable for atexit
  // cleanup after NCCL communicator shutdown. Errors during shutdown are
  // logged, never thrown.
  auto* handle = release_xfer_handle();
  if (handle != nullptr) {
    auto rc = ::ncclXferFinalize(handle);
    if (rc != ncclSuccess) {
      LOG(WARNING) << "ncclXferFinalize returned " << rc
                   << " — symm_mem.mxn_cast resources may not be fully released.";
    }
  }
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
      "nccl_mxn_cast requires the user-window reshard API from NCCL Xfer "
      "(libnccl_xfer.so) or an NCCL build exporting it "
      "from libnccl.so. NCCL_HAS_RESHARD_API was not defined at build time.");
}

void nccl_mxn_cast_finalize() {
  // Stub when the reshard API isn't compiled in — nothing to release.
}

#endif // NCCL_HAS_RESHARD_API

} // namespace c10d::nccl_extension
