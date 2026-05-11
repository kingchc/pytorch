#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

namespace c10d::nccl_extension {

TORCH_API bool is_nccl_symmem_available();

TORCH_API void nccl_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_get(at::Tensor& tensor, const int64_t peer);

TORCH_API void nccl_wait_for_signal(at::Tensor& sigpad, int64_t signal);

TORCH_API void nccl_put_with_signal(
    at::Tensor& tensor,
    int64_t signal,
    int64_t peer);

// Simultaneously reduce N blocks of a 2-D input tensor from a shared symmetric
// memory buffer, routing each to a specific destination rank. Blocks are
// described by inclusive-prefix-sum offsets along `dim` (0 or 1); all blocks
// must have equal size.
TORCH_API void nccl_reduce_scatter_offset(
    const at::Tensor& input,
    at::TensorList out,
    const std::string& group_name,
    int64_t dim,
    std::optional<at::IntArrayRef> offsets,
    std::optional<at::IntArrayRef> dst_ranks,
    const std::string& red_op);

// In-place M-to-N cast (resharding) of a 1-D, 2-D, or 3-D tensor between
// two rank meshes, backed by `ncclReshardUserWindow` (originated in
// nccl-rl; pending upstream NCCL merge).  `buf` must be allocated through NCCL
// symmetric memory and sized to hold the larger of the source and
// destination local shapes.  Each mesh is described by
// `(dims[2], start_rank, placement[2])` where placement[i] is -1 for
// REPLICATE or a non-negative tensor dim index for SHARD.  A rank that
// does not own a tile on a given side passes a zero-shape on that side;
// the binding maps that to `ncclDistTensor_t::data_ptr = NULL`.
TORCH_API void nccl_mxn_cast(
    at::Tensor& buf,
    at::IntArrayRef src_local_shape,
    at::IntArrayRef src_mesh_dims,
    int64_t src_mesh_start_rank,
    at::IntArrayRef src_placement,
    at::IntArrayRef dst_local_shape,
    at::IntArrayRef dst_mesh_dims,
    int64_t dst_mesh_start_rank,
    at::IntArrayRef dst_placement,
    const std::string& group_name);

// Best-effort `ncclReshardFinalize` — releases the reshard library's
// internal caches + transpose buffer.  Idempotent and safe to call
// multiple times; used by an atexit hook in the Python wrapper so the
// release happens while CUDA + NCCL contexts are still live.
TORCH_API void nccl_mxn_cast_finalize();
} // namespace c10d::nccl_extension
