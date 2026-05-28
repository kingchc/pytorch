from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.distributed.tensor.placement_types import Partial, Placement, Replicate, Shard


_XFER_REPLICATE = -1


def _as_local_tensor(tensor: Any | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor
    local_tensor = getattr(tensor, "_local_tensor", None)
    if isinstance(local_tensor, torch.Tensor):
        return local_tensor
    raise TypeError(
        "torch.distributed.tensor._xg_redistribute expects each participating "
        "side to be a torch.Tensor or DTensor-like object with a _local_tensor"
    )


def _local_shape(
    tensor: torch.Tensor | None,
    explicit_shape: Sequence[int] | None,
    name: str,
) -> tuple[int, ...]:
    if explicit_shape is not None:
        shape = tuple(int(dim) for dim in explicit_shape)
    elif tensor is not None:
        shape = tuple(int(dim) for dim in tensor.shape)
    else:
        raise ValueError(
            f"{name}_local_shape is required when {name} tensor is absent on "
            "this rank"
        )
    if not shape:
        raise ValueError(f"{name}_local_shape must describe at least one dimension")
    if any(dim < 0 for dim in shape):
        raise ValueError(f"{name}_local_shape must be non-negative, got {shape}")
    return shape


def _mesh_tensor(mesh: Any) -> torch.Tensor:
    if isinstance(mesh, torch.Tensor):
        return mesh.detach().to(device="cpu", dtype=torch.int64)

    # DeviceMesh.mesh may require an initialized process group. Use the
    # underlying layout when present so descriptor normalization itself does not
    # depend on torch.distributed process-group state.
    layout = getattr(mesh, "_layout", None)
    rank_map = getattr(mesh, "_rank_map", None)
    if layout is not None and rank_map is not None:
        return layout.remap_to_tensor(rank_map).detach().to(
            device="cpu", dtype=torch.int64
        )

    raw_mesh = getattr(mesh, "mesh", None)
    if raw_mesh is not None:
        return torch.as_tensor(raw_mesh, device="cpu", dtype=torch.int64)

    return torch.as_tensor(mesh, device="cpu", dtype=torch.int64)


def _mesh_to_xfer(mesh: Any, name: str) -> tuple[tuple[int, int], int]:
    mesh_tensor = _mesh_tensor(mesh).contiguous()
    if mesh_tensor.ndim == 0:
        mesh_tensor = mesh_tensor.reshape(1)
    if mesh_tensor.ndim == 1:
        dims = (int(mesh_tensor.numel()), 1)
    elif mesh_tensor.ndim == 2:
        dims = (int(mesh_tensor.shape[0]), int(mesh_tensor.shape[1]))
    else:
        raise ValueError(
            f"{name}_mesh must be 1-D or 2-D for NCCL Xfer, got "
            f"{mesh_tensor.ndim}-D"
        )
    flat = [int(rank) for rank in mesh_tensor.flatten().tolist()]
    if not flat:
        raise ValueError(f"{name}_mesh must contain at least one rank")
    start_rank = flat[0]
    expected = list(range(start_rank, start_rank + len(flat)))
    if flat != expected:
        raise ValueError(
            f"{name}_mesh must be a contiguous row-major rank interval for "
            f"NCCL Xfer, got {flat}"
        )
    return dims, start_rank


def _placement_to_xfer(placement: Placement | int) -> int:
    if isinstance(placement, Replicate):
        return _XFER_REPLICATE
    if isinstance(placement, Shard):
        return int(placement.dim)
    if isinstance(placement, Partial):
        raise ValueError(
            "NCCL Xfer cross-group redistribute does not support Partial placements"
        )
    if isinstance(placement, int):
        return int(placement)
    raise TypeError(
        f"Unsupported placement for NCCL Xfer cross-group redistribute: {placement!r}"
    )


def _placements_to_xfer(
    placements: Sequence[Placement | int],
    mesh_dims: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    if len(placements) not in (1, 2):
        raise ValueError(
            f"{name}_placements must have length 1 or 2, got {len(placements)}"
        )
    normalized = [_placement_to_xfer(p) for p in placements]
    if len(normalized) == 1:
        normalized.append(_XFER_REPLICATE)
    for value in normalized:
        if value < _XFER_REPLICATE:
            raise ValueError(
                f"{name}_placements entries must be -1 or tensor dimensions, "
                f"got {tuple(normalized)}"
            )
    # A 1-D mesh is represented as (N, 1) for NCCL Xfer. The padded second
    # placement must be replicate because there is no real second mesh axis.
    if mesh_dims[1] == 1 and len(placements) == 1:
        normalized[1] = _XFER_REPLICATE
    return int(normalized[0]), int(normalized[1])


def _comm_ptr(comm: Any) -> int:
    if isinstance(comm, int):
        ptr = comm
    else:
        ptr = int(getattr(comm, "ptr"))
    if ptr == 0:
        raise ValueError("comm must be a non-null ncclComm_t pointer")
    return ptr


def _call_xg_redistribute_op(*args: Any) -> None:
    torch.ops.dtensor_xfer.reshard(*args)


def init_xfer() -> None:
    torch.ops.dtensor_xfer.init()


def finalize_xfer() -> None:
    torch.ops.dtensor_xfer.finalize()


class XferContext:
    def init(self) -> "XferContext":
        init_xfer()
        return self

    def finalize(self) -> None:
        finalize_xfer()

    def __enter__(self) -> "XferContext":
        return self.init()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.finalize()


def _xg_redistribute(
    src: Any | None,
    dst: Any | None,
    *,
    src_mesh: Any,
    src_placements: Sequence[Placement | int],
    dst_mesh: Any,
    dst_placements: Sequence[Placement | int],
    comm: Any,
    src_local_shape: Sequence[int] | None = None,
    dst_local_shape: Sequence[int] | None = None,
    stream: torch.cuda.Stream | None = None,
) -> Any | None:
    src_local = _as_local_tensor(src)
    dst_local = _as_local_tensor(dst)
    src_shape = _local_shape(src_local, src_local_shape, "src")
    dst_shape = _local_shape(dst_local, dst_local_shape, "dst")
    if len(src_shape) != len(dst_shape):
        raise ValueError(
            f"src_local_shape rank {len(src_shape)} must match "
            f"dst_local_shape rank {len(dst_shape)}"
        )

    src_mesh_dims, src_start_rank = _mesh_to_xfer(src_mesh, "src")
    dst_mesh_dims, dst_start_rank = _mesh_to_xfer(dst_mesh, "dst")
    src_xfer_placements = _placements_to_xfer(
        src_placements, src_mesh_dims, "src"
    )
    dst_xfer_placements = _placements_to_xfer(
        dst_placements, dst_mesh_dims, "dst"
    )
    comm_value = _comm_ptr(comm)

    def invoke() -> None:
        _call_xg_redistribute_op(
            src_local,
            dst_local,
            list(src_shape),
            list(src_mesh_dims),
            src_start_rank,
            list(src_xfer_placements),
            list(dst_shape),
            list(dst_mesh_dims),
            dst_start_rank,
            list(dst_xfer_placements),
            comm_value,
        )

    if stream is None:
        invoke()
    else:
        with torch.cuda.stream(stream):
            invoke()
    return dst
