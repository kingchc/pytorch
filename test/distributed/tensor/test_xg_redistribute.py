import importlib
import os
import unittest
from unittest import mock

import torch
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.common_utils import run_tests, TestCase

_xg_redistribute = importlib.import_module("torch.distributed.tensor._xg_redistribute")


class _DummyComm:
    ptr = 0x1234


class _DummyDTensor:
    def __init__(self, local_tensor):
        self._local_tensor = local_tensor


class XGRedistributeAPITest(TestCase):
    def test_comm_ptr_accepts_int_or_ptr_object(self):
        self.assertEqual(_xg_redistribute._comm_ptr(0xCAFE), 0xCAFE)
        self.assertEqual(_xg_redistribute._comm_ptr(_DummyComm()), 0x1234)
        with self.assertRaisesRegex(ValueError, "non-null"):
            _xg_redistribute._comm_ptr(0)

    def test_mesh_to_xfer_requires_contiguous_row_major_ranks(self):
        self.assertEqual(
            _xg_redistribute._mesh_to_xfer([4, 5, 6], "src"), ((3, 1), 4)
        )
        self.assertEqual(
            _xg_redistribute._mesh_to_xfer([[2, 3], [4, 5]], "dst"),
            ((2, 2), 2),
        )
        with self.assertRaisesRegex(ValueError, "contiguous row-major"):
            _xg_redistribute._mesh_to_xfer([0, 2], "src")

    def test_placements_to_xfer(self):
        self.assertEqual(
            _xg_redistribute._placements_to_xfer([Shard(0)], (2, 1), "src"),
            (0, -1),
        )
        self.assertEqual(
            _xg_redistribute._placements_to_xfer(
                [Replicate(), Shard(1)], (2, 2), "dst"
            ),
            (-1, 1),
        )
        with self.assertRaisesRegex(ValueError, "Partial"):
            _xg_redistribute._placements_to_xfer([Partial()], (2, 1), "src")

    def test_xg_redistribute_normalizes_arguments_for_dispatch_op(self):
        src = torch.arange(6).reshape(2, 3)
        dst = torch.empty(3, 2)
        calls = []

        with mock.patch.object(
            _xg_redistribute,
            "_call_xg_redistribute_op",
            side_effect=lambda *args: calls.append(args),
        ):
            result = _xg_redistribute._xg_redistribute(
                _DummyDTensor(src),
                dst,
                src_mesh=[0, 1],
                src_placements=[Shard(0)],
                dst_mesh=[[2, 3]],
                dst_placements=[Replicate()],
                comm=_DummyComm(),
            )

        self.assertIs(result, dst)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], src)
        self.assertIs(calls[0][1], dst)
        self.assertEqual(calls[0][2], [2, 3])
        self.assertEqual(calls[0][3], [2, 1])
        self.assertEqual(calls[0][4], 0)
        self.assertEqual(calls[0][5], [0, -1])
        self.assertEqual(calls[0][6], [3, 2])
        self.assertEqual(calls[0][7], [1, 2])
        self.assertEqual(calls[0][8], 2)
        self.assertEqual(calls[0][9], [-1, -1])
        self.assertEqual(calls[0][10], 0x1234)

    def test_xg_redistribute_requires_shape_for_absent_side(self):
        dst = torch.empty(4)
        with self.assertRaisesRegex(ValueError, "src_local_shape is required"):
            _xg_redistribute._xg_redistribute(
                None,
                dst,
                src_mesh=[0],
                src_placements=[Replicate()],
                dst_mesh=[1],
                dst_placements=[Replicate()],
                comm=_DummyComm(),
            )

    def test_xfer_context_calls_lifecycle_ops(self):
        calls = []
        with mock.patch.object(
            _xg_redistribute, "init_xfer", side_effect=lambda: calls.append("init")
        ), mock.patch.object(
            _xg_redistribute,
            "finalize_xfer",
            side_effect=lambda: calls.append("finalize"),
        ):
            with _xg_redistribute.XferContext():
                calls.append("body")
        self.assertEqual(calls, ["init", "body", "finalize"])


@unittest.skipIf(
    os.environ.get("RUN_NCCL_XG_REDISTRIBUTE_TEST") != "1",
    "set RUN_NCCL_XG_REDISTRIBUTE_TEST=1 and launch two local ranks",
)
class XGRedistributeFunctionalTest(TestCase):
    def _init_nccl4py_comm(self):
        try:
            import torch.distributed as dist
            from nccl.core.communicator import Communicator
            from nccl.core.utils import UniqueId, get_unique_id
        except ImportError as exc:
            self.skipTest(f"nccl4py is required for functional test: {exc}")

        required_env = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
        missing = [name for name in required_env if name not in os.environ]
        if missing:
            self.skipTest(f"missing torchrun environment variables: {missing}")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size != 2:
            self.skipTest("functional POC currently expects exactly 2 ranks")
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for NCCL Xfer functional test")
        torch.cuda.set_device(local_rank)

        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]) + 17,
            world_size=world_size,
            is_master=(rank == 0),
            use_libuv=False,
        )
        key = "nccl_xg_redistribute_uid"
        if rank == 0:
            unique_id = get_unique_id()
            store.set(key, unique_id.as_bytes)
        else:
            store.wait([key])
            unique_id = UniqueId.from_bytes(store.get(key))
        comm = Communicator.init(
            nranks=world_size,
            rank=rank,
            unique_id=unique_id,
        )
        return rank, comm

    def test_replicate_to_replicate_two_rank_copy_api(self):
        rank, comm = self._init_nccl4py_comm()
        src = (
            torch.arange(8, device="cuda", dtype=torch.float32)
            if rank == 0
            else None
        )
        dst = (
            torch.empty(8, device="cuda", dtype=torch.float32)
            if rank == 1
            else None
        )

        try:
            with _xg_redistribute.XferContext():
                _xg_redistribute._xg_redistribute(
                    src,
                    dst,
                    src_mesh=[0],
                    src_placements=[Replicate()],
                    dst_mesh=[1],
                    dst_placements=[Replicate()],
                    comm=comm,
                    src_local_shape=(8,),
                    dst_local_shape=(8,),
                )
                torch.cuda.synchronize()
                if rank == 1:
                    self.assertEqual(
                        dst.cpu(),
                        torch.arange(8, dtype=torch.float32),
                    )
        finally:
            comm.destroy()


if __name__ == "__main__":
    run_tests()
