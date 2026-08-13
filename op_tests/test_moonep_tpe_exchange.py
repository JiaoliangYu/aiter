# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Check the symmetric-heap expert histogram exchange against ``all_gather``.

``_publish_tpe`` replaced a per-layer ``dist.all_gather`` -- a full NCCL
collective costing 25% of prefill GPU time to move 12 KB.  The chain test
cannot police that replacement: the histogram only feeds the planner's
balancing decision, so an exchange that is wrong but *identically* wrong on
every rank still yields correct numbers.  This compares the matrix directly.

The negative control matters as much as the positive one.  A test that only
asserts equality proves nothing until you have seen it fail: it would pass
just as happily against an exchange that never ran, if the buffer happened to
hold the right bytes already.  So the buffer is poisoned before each exchange,
and one case deliberately drops the peer writes and requires a mismatch.

Run::

    torchrun --standalone --nproc-per-node=8 op_tests/test_moonep_tpe_exchange.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.moonep_dispatch_combine_op import (
    MoonEPDispatchCombineConfig,
    MoonEPDispatchCombineIntraNodeOp,
)

S = int(os.environ.get("T_S", "256"))
H = int(os.environ.get("T_H", "512"))
K = int(os.environ.get("T_K", "4"))
E = int(os.environ.get("T_E", "32"))


def histogram(indices: torch.Tensor, e: int) -> torch.Tensor:
    flat = indices.reshape(-1).to(torch.int64)
    tpe = torch.zeros(e, dtype=torch.int32, device=indices.device)
    tpe.scatter_add_(0, flat.clamp_min(0), (flat >= 0).to(torch.int32))
    return tpe


def reference(tpe: torch.Tensor, world: int) -> torch.Tensor:
    want = [torch.empty_like(tpe) for _ in range(world)]
    dist.all_gather(want, tpe)
    return torch.stack(want)


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    import mori
    import mori.shmem as ms
    from mori.shmem.tensor_utils import symm_mori_shmem_tensor

    cpu_group = dist.new_group(backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")

    cfg = MoonEPDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=H,
        max_num_inp_token_per_rank=S,
        num_experts_per_rank=E // world,
        num_experts_per_token=K,
    )
    op = MoonEPDispatchCombineIntraNodeOp(cfg)

    def poison() -> None:
        """Leave nothing of the previous round in the buffer.

        Without this an exchange that silently did nothing would still match
        the reference on the second call.
        """
        op._tpe_symm.fill_(-1)
        torch.cuda.synchronize(dev)
        ms.shmem_barrier_all()

    def run(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-rank seeds: with identical histograms on every rank, a broken
        # exchange that leaves each row holding the local vector would pass.
        g = torch.Generator(device="cpu").manual_seed(seed * 100 + rank)
        idx = torch.rand(S, E, generator=g).topk(K, dim=1).indices
        tpe = histogram(idx.to(torch.int32).to(dev), E)
        got = op._publish_tpe(tpe)
        torch.cuda.synchronize(dev)
        return got.clone(), reference(tpe, world)

    failures = []

    # 1. positive control
    poison()
    got, want = run(1)
    ok = torch.equal(got, want)
    failures += [] if ok else ["exchange != all_gather"]
    if rank == 0:
        print(f"[tpe] exchange matches all_gather: {'OK' if ok else 'FAIL'}", flush=True)

    # 2. a second round with different routing must overwrite, not accumulate
    #    or go stale -- the buffer is reused every layer.
    poison()
    got2, want2 = run(2)
    ok2 = torch.equal(got2, want2) and not torch.equal(got2, got)
    failures += [] if ok2 else ["second round stale or unchanged"]
    if rank == 0:
        print(f"[tpe] second round overwrites: {'OK' if ok2 else 'FAIL'}", flush=True)

    # 3. negative control: drop the peer writes. Every rank then keeps only
    #    its own row and the others stay poisoned, so the comparison MUST
    #    fail. If it passes, the positive control above proves nothing.
    if world > 1:
        poison()
        saved = op._tpe_peer
        op._tpe_peer = [symm_mori_shmem_tensor(op._tpe_symm, rank)]
        broken, want3 = run(3)
        op._tpe_peer = saved
        caught = not torch.equal(broken, want3)
        failures += [] if caught else ["negative control passed -- test is blind"]
        if rank == 0:
            print(
                f"[tpe] negative control caught: {'OK' if caught else 'FAIL'}",
                flush=True,
            )

    # 4. the in-op assertion must fire on the same breakage.
    if world > 1:
        poison()
        saved = op._tpe_peer
        op._tpe_peer = [symm_mori_shmem_tensor(op._tpe_symm, rank)]
        op._check_tpe = True
        raised = False
        try:
            run(4)
        except AssertionError:
            raised = True
        finally:
            op._check_tpe = False
            op._tpe_peer = saved
        failures += [] if raised else ["MOONEP_CHECK_TPE did not fire"]
        if rank == 0:
            print(f"[tpe] in-op check fires: {'OK' if raised else 'FAIL'}", flush=True)

    poison()
    op.close()
    dist.barrier()
    if rank == 0:
        print(f"[tpe] {'PASS' if not failures else 'FAIL ' + '; '.join(failures)}", flush=True)
    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
