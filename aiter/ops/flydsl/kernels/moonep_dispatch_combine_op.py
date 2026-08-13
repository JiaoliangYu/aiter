# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MoonEP behind the EP-backend contract ATOM already uses.

ATOM's EP backends are stateless and routing-driven::

    dispatch(input, weights, scales, indices) -> packed rows + metadata
    combine(input, weights, indices)          -> per-token output

MoonEP is plan-driven: it needs the **all-ranks** tokens-per-expert histogram
before it can decide anything, and dispatch/combine must see the *same* plan.
This class absorbs that difference so the manager above it looks like
``FlyDSLDispatchCombineIntraNodeOp``:

* ``dispatch`` derives the local histogram from ``indices``, all-gathers it,
  builds the plan on GPU (~113 us) and stashes it;
* ``combine`` reuses the stashed plan and refuses to run if none is live, which
  is the only way a caller can silently pair mismatched plans.

Deliberately *not* hidden
-------------------------
Three gaps are real and are raised as errors rather than papered over, because
each one is a silent-wrong-answer risk if guessed at:

1. **BF16 only.**  ``moonep_dispatch.py`` is BF16 by construction ("BF16 only
   for the first gfx950 milestone"); there is no quantised path.  A caller
   asking for fp8 gets ``NotImplementedError``, not a wrong-dtype kernel.
2. **Token count is fixed at construction.**  The plan, ``num_dispatch_rows``
   and every symmetric buffer derive from ``max_num_inp_token_per_rank``.  A
   batch larger than that is rejected; a smaller one is padded, which costs
   wire bytes proportional to the padding.
3. **Expert-weight migration is on**, so a rank computes its ``E/R`` home
   experts *plus* up to ``B`` migrated ones.  ``local_group_sizes`` therefore
   spans ``E/R + B`` groups, not ``E/R``; see its docstring.

Push combine gating
-------------------
Push (remote writes, 448 GB/s) beats pull (remote reads, 235 GB/s) at balanced
to typical routing but **loses at pathological routing** -- measured 1.46x /
1.33x / 0.91x at maxvio 0.32 / 9.7 / 47.  It is therefore off unless explicitly
enabled, and when enabled it is gated on the measured maxvio.  The cause of the
high-maxvio regression is not yet established, so the threshold is a measured
crossover, not a model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor
from mori.shmem.tensor_utils import symm_mori_shmem_tensor

from aiter.ops.flydsl.moonep import (
    MoonEPGpuPlanner,
    MoonEPPlanConfig,
    MoonEPReferencePlan,
)
from aiter.ops.flydsl.kernels.moonep_dispatch_op import (
    MoonEPPreplannedDispatchOp,
)

# maxvio above which push combine is slower than dedup pull.  Measured
# crossover lies between 9.7 (push 1.33x faster) and 47.0 (push 0.91x); 20 is
# the midpoint in the regime upstream itself sweeps.  Revisit once the cause of
# the regression is understood.
PUSH_MAXVIO_LIMIT = 20.0


@dataclass
class MoonEPDispatchCombineConfig:
    """Mirrors the fields ``FlyDSLDispatchCombineConfig`` exposes."""

    rank: int
    world_size: int
    hidden_dim: int
    max_num_inp_token_per_rank: int
    num_experts_per_rank: int
    num_experts_per_token: int
    # Decode plan: separate, much smaller instance that skips balancing and
    # migration. 0 disables it and runs the balanced plan in both phases.
    max_decode_token_per_rank: int = 0
    data_type: torch.dtype = torch.bfloat16
    prefetch_slots: int | None = None
    dispatch_block_num: int = 1024
    combine_block_num: int = 1024
    quant_type: str = "none"
    token_padding: int = 128
    enable_push_combine: bool = False

    @property
    def num_experts(self) -> int:
        return self.num_experts_per_rank * self.world_size


class MoonEPDispatchCombineIntraNodeOp:
    """Plan-driven MoonEP transport with a stateless-looking API."""

    def __init__(self, cfg: MoonEPDispatchCombineConfig) -> None:
        if cfg.data_type != torch.bfloat16:
            raise NotImplementedError(
                "MoonEP dispatch/combine is BF16-only (moonep_dispatch.py has "
                f"no quantised path); got data_type={cfg.data_type}. Use the "
                "'aiter' or 'mori' EP backend for fp8."
            )
        if cfg.quant_type != "none":
            raise NotImplementedError(
                f"MoonEP has no quantisation path; got quant_type="
                f"{cfg.quant_type!r}."
            )
        self.cfg = cfg
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.plan_config = MoonEPPlanConfig(
            rank=cfg.rank,
            world_size=cfg.world_size,
            num_tokens=cfg.max_num_inp_token_per_rank,
            top_k=cfg.num_experts_per_token,
            num_experts=cfg.num_experts,
            prefetch_slots=cfg.prefetch_slots,
            token_padding=cfg.token_padding,
        )
        self._planner = MoonEPGpuPlanner(self.plan_config, self.device)
        self._op = MoonEPPreplannedDispatchOp(
            self.plan_config, cfg.hidden_dim, block_num=cfg.dispatch_block_num
        )
        # Decode instance. Three things make it cheap, and all three come from
        # dropping the balancer:
        #   * no migration -> the migration groups are empty, so the experts
        #     step is one call over the home experts and the prefetch is skipped
        #   * a small num_tokens -> the plan is sized for a decode step instead
        #     of the prefill worst case
        #   * token_padding=1 -> with 48 local experts, padding each group up to
        #     128 rows costs 6144 rows to carry a few hundred. Nothing needs the
        #     alignment here: fused_moe is driven by per-row slot ids and does
        #     its own sorting, so the groups never have to be tile-aligned.
        self._decode_config = None
        self._decode_planner = None
        self._decode_op = None
        if cfg.max_decode_token_per_rank > 0:
            self._decode_config = MoonEPPlanConfig(
                rank=cfg.rank,
                world_size=cfg.world_size,
                num_tokens=cfg.max_decode_token_per_rank,
                top_k=cfg.num_experts_per_token,
                num_experts=cfg.num_experts,
                prefetch_slots=1,
                token_padding=1,
                no_migration=True,
            )
            self._decode_planner = MoonEPGpuPlanner(
                self._decode_config, self.device
            )
            self._decode_op = MoonEPPreplannedDispatchOp(
                self._decode_config,
                cfg.hidden_dim,
                block_num=cfg.dispatch_block_num,
            )
        # Global expert histogram, exchanged over the symmetric heap rather
        # than with a collective.  Upstream does this inside the planning
        # kernel (planning.py Phase A: copy_v4_remote of the local [E] into
        # meta_buf at rank*E, then a self-resetting cross-rank barrier), which
        # is why its `tokens_per_expert` argument is documented as *local*.
        # Building the [R, E] matrix with dist.all_gather instead cost a full
        # NCCL collective per MoE layer -- 1760 ms, 25% of prefill GPU time,
        # almost all of it waiting, for 12 KB of payload.  Allocated here so
        # the collective allocation order matches on every PE.
        self._tpe_symm = mori_shmem_create_tensor(
            (cfg.world_size, cfg.num_experts), torch.int32
        )
        # symm_mori_shmem_tensor returns the local tensor for our own rank and
        # preserves dtype and shape for the rest, so the list is uniform.
        self._tpe_peer = [
            symm_mori_shmem_tensor(self._tpe_symm, p)
            for p in range(cfg.world_size)
        ]
        self._check_tpe = os.environ.get("MOONEP_CHECK_TPE", "0") == "1"
        # Whichever instance the live plan belongs to.
        self._act_op = self._op
        self._act_planner = self._planner
        self._act_cfg = self.plan_config
        self._plan: MoonEPReferencePlan | None = None
        self._tpe_all = None
        self._closed = False
        # Built lazily on first combine so a dispatch-only caller pays nothing.
        self._push: Any | None = None

    # -- helpers ---------------------------------------------------------
    def _pad_tokens(self, x: torch.Tensor, n: int, cap: int) -> torch.Tensor:
        """Pad to the fixed plan size; MoonEP has no dynamic-shape support."""
        if n > cap:
            raise ValueError(
                f"MoonEP was built for at most {cap} tokens per rank but got "
                f"{n}. The plan and every symmetric buffer are sized at "
                "construction; rebuild the op with a larger "
                "max_num_inp_token_per_rank."
            )
        if n == cap:
            return x
        out = x.new_zeros((cap, *x.shape[1:]))
        out[:n] = x
        return out

    def _pad_routing(self, rows: int, k: int) -> torch.Tensor:
        """Routing for padding tokens: valid ids, spread evenly.

        The planner is fixed-shape and takes exactly ``num_tokens * top_k``
        routed entries per rank -- ``build_reference_plan`` rejects an id
        outside ``[0, E)`` outright, and the GPU planner, which does not
        validate, faults instead.  So padding rows cannot be marked idle with
        -1; they have to carry real expert ids.  Round-robin keeps them from
        piling onto one expert and skewing the very load balance the plan is
        computing.

        Their outputs are discarded (combine reads only the first ``n``
        tokens), so this is wasted compute proportional to ``cap - n``.  That
        is the price of one fixed plan size; bucketing the plan by token count
        is what removes it.
        """

        base = torch.arange(rows * k, device=self.device, dtype=torch.int32)
        return (base % self.cfg.num_experts).reshape(rows, k)

    def _publish_tpe(self, tpe: torch.Tensor) -> torch.Tensor:
        """Put this rank's ``[E]`` histogram in every rank's ``[R, E]`` view.

        Push rather than pull: each rank writes its own row into all peers,
        then one device barrier makes the whole matrix readable everywhere.
        Remote writes run at 448 GB/s against 235 GB/s for remote reads on this
        link, and a write needs no return trip.

        Ordering across layers is already covered: the next layer cannot start
        writing until every rank has passed the barrier that ``dispatch``
        enqueues after its kernel, which is downstream of this layer's reads.

        One small copy per peer is R launches per layer, a few us each, plus
        one barrier at ~250 us measured. Against the ~92 ms per prefill step
        the collective was costing (1783 ms over 1178 calls, 61 layers) that
        is worth measuring before folding the writes into the meta kernel the
        way upstream does.
        """

        rank = self.cfg.rank
        for view in self._tpe_peer:
            view[rank].copy_(tpe)
        ms.shmem_barrier_on_stream(torch.cuda.current_stream(self.device))
        if self._check_tpe:
            self._assert_matches_all_gather(tpe)
        return self._tpe_symm

    def _assert_matches_all_gather(self, tpe: torch.Tensor) -> None:
        """Compare the exchanged matrix against the collective it replaced.

        The chain test cannot catch a bad exchange on its own: the histogram
        only feeds the planner's balancing decision, so a matrix that is wrong
        but *identically* wrong on every rank still produces correct numbers.
        Only a direct comparison rules that out. Gated because it costs the
        collective plus a device sync per layer.
        """

        import torch.distributed as dist

        want = [torch.empty_like(tpe) for _ in range(self.cfg.world_size)]
        dist.all_gather(want, tpe)
        want = torch.stack(want)
        if not torch.equal(self._tpe_symm, want):
            bad = (self._tpe_symm != want).any(dim=1).nonzero().flatten()
            raise AssertionError(
                f"rank {self.cfg.rank}: symmetric-heap histogram disagrees "
                f"with all_gather on rows {bad.tolist()}"
            )

    def _build_plan(self, indices: torch.Tensor) -> MoonEPReferencePlan:
        """Local histogram -> symmetric-heap exchange -> GPU plan.

        The global histogram is genuinely needed per layer -- MoonEP balances
        against the whole group's expert load and the routing differs layer to
        layer -- so this is not something to do less often. Upstream does it
        every layer too; it just does it inside the planning kernel over the
        NVLink meta buffer instead of with a collective.
        """
        e = self.cfg.num_experts
        # Negative ids are the planner's idle mark -- padding rows here, and
        # masked-out experts from ATOM's router -- so they must not be counted.
        # scatter_add rather than bincount on a filtered view: the filter would
        # have a data-dependent shape and cost a device sync per MoE layer.
        flat = indices.reshape(-1).to(torch.int64)
        tpe = torch.zeros(e, dtype=torch.int32, device=indices.device)
        tpe.scatter_add_(
            0, flat.clamp_min(0), (flat >= 0).to(torch.int32)
        )
        tpe_all = self._publish_tpe(tpe)
        # Kept for the maxvio property; computing it here would cost a device
        # sync on every MoE layer for a number the serving path never reads.
        self._tpe_all = tpe_all
        return self._act_planner.build(indices.to(torch.int32), tpe_all)

    # -- grouped API consumed by MoonEPPrepareAndFinalize ------------------
    def dispatch_grouped(self, hidden, topk_weights, topk_ids, decode: bool = False):
        """Dispatch and hand back rows already grouped by expert.

        Returns ``(rows, row_weights, cu_seqlens)`` where ``rows`` is 2-D
        ``(NvS, H)`` -- ``FusedMoEActivationFormat.Standard`` -- ordered by
        expert group rather than by token, and ``cu_seqlens`` are the group end
        offsets over all ``E + B`` groups.
        """

        if self._closed:
            raise RuntimeError("op is closed")
        if decode and self._decode_op is None:
            raise RuntimeError(
                "decode plan requested but the op was built without one; set "
                "max_decode_token_per_rank"
            )
        self._act_op = self._decode_op if decode else self._op
        self._act_planner = self._decode_planner if decode else self._planner
        self._act_cfg = self._decode_config if decode else self.plan_config
        n = hidden.shape[0]
        cap = self._act_cfg.num_tokens
        x = self._pad_tokens(hidden, n, cap)
        w = self._pad_tokens(topk_weights, n, cap)
        idx = topk_ids.to(torch.int32)
        if n < cap:
            idx = torch.cat([idx, self._pad_routing(cap - n, idx.shape[1])])
        self._n_tokens = n
        self._plan = self._build_plan(idx)
        rows, row_weights = self._act_op.dispatch(x, w, self._plan)
        return rows, row_weights, self._plan.cu_seqlens

    def local_group_sizes(self, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Rows per group this rank executes, in plan-group order.

        ``cu_seqlens`` holds the end offset of every one of the ``E + B``
        groups, so the per-group size is its first difference.  This rank
        executes its ``E/R`` home groups **and** the ``B`` migration groups
        that hold experts pulled in from overloaded peers -- rebalancing is
        what the planner is for, so the migration groups are routinely
        non-empty and dropping them loses real tokens.  The returned vector is
        therefore ``E/R + B`` long, home groups first.

        A *third* kind of group can be non-empty: a remote expert the planner
        allocated rows for without giving this rank a prefetch slot.  That is
        not an error.  MoonEP's contract makes ``B`` a performance knob, not a
        correctness floor -- upstream's README says of ``B < E/R``: "If a rank
        ever needs more distinct remote experts than B, the group GEMM reads
        the overflow weights straight from the home rank through the symmetric
        mapping -- slightly slower, with no impact on correctness."  The
        balancer is free to ignore ``B`` precisely because of this fallback.
        ``overflow_groups()`` enumerates them and ``run_experts`` gives each
        its own call against the owner's pool.

        The returned vector therefore covers home groups, migration slots and
        overflow groups -- every row this rank executes.
        """

        sizes = cu_seqlens - torch.cat(
            [cu_seqlens.new_zeros(1), cu_seqlens[:-1]]
        )
        e = self.cfg.num_experts
        lo = self.cfg.rank * self.cfg.num_experts_per_rank
        hi = lo + self.cfg.num_experts_per_rank
        mine = torch.cat([sizes[lo:hi], sizes[e:]])
        overflow = torch.cat([sizes[:lo], sizes[hi:e]])
        return torch.cat([mine, overflow]).to(torch.int32)

    def overflow_groups(self) -> list[tuple[int, int, int]]:
        """``(expert_id, row_lo, row_hi)`` per remote group with no slot.

        Rows are laid out in group order, so each group is one contiguous
        slice -- no gather is needed, just one experts call per group against
        the owner rank's home segment.  Empty for the decode plan, which
        migrates nothing.

        This costs one device-to-host copy of ``cu_seqlens`` per layer.  It is
        folded into the same sync ``expert_call_split`` already performs.
        """

        if not self.needs_split():
            return []
        plan = self.live_plan()
        cu = plan.cu_seqlens.to("cpu", non_blocking=False).tolist()
        gei = plan.group_expert_ids.to("cpu", non_blocking=False).tolist()
        e = self.cfg.num_experts
        epn = self.cfg.num_experts_per_rank
        lo_e = self.cfg.rank * epn
        hi_e = lo_e + epn
        out: list[tuple[int, int, int]] = []
        for g in range(e):
            if lo_e <= g < hi_e:
                continue
            start = 0 if g == 0 else cu[g - 1]
            if cu[g] > start and gei[g] >= 0:
                out.append((gei[g], start, cu[g]))
        # How much of the step actually takes the slower remote-weight path.
        # Logged once per process: enough to size the effect, not enough to
        # flood a 61-layer model.
        if out and not getattr(self, "_overflow_logged", False):
            self._overflow_logged = True
            rows = sum(hi - lo for _, lo, hi in out)
            print(
                f"[MoonEP] rank {self.cfg.rank}: {len(out)} overflow groups, "
                f"{rows} rows of {cu[-1]} ({100.0 * rows / max(cu[-1], 1):.2f}%) "
                f"read weights from the owner (prefetch_slots="
                f"{self._act_cfg.prefetch_slots})",
                flush=True,
            )
        return out

    def decode_plan_available(self) -> bool:
        return self._decode_op is not None

    def needs_split(self) -> bool:
        """Whether the experts step needs the two-call home/migration split.

        The decode plan migrates nothing, so its migration groups are always
        empty: one call over the home experts covers every row, driven by
        ``valid_rows()`` on device.  That is what removes the last host sync
        from a decode step.
        """

        return not self._act_cfg.no_migration

    def expert_call_split(self) -> tuple[int, int, int, int, int]:
        """``(home_lo, home_hi, mig_lo, mig_hi, num_borrowed)``.

        aiter's quantised MoE gives a **zero** result when the weight tensor
        declares experts that no row routes to -- measured: any spare slot
        breaks it, while every expert count from 1 to 48 is exact when all of
        them are used.  So the ``E + B`` pool cannot be handed over as one
        tensor; the experts step is split into a home call over ``E/R``
        experts and a migration call over exactly the ``nb`` borrowed ones.

        The split is free of a search: rows are laid out in group order, so
        home groups occupy ``[0, cu[E-1])`` and migration groups
        ``[cu[E-1], cu[-1])``, and ``experts_to_copy`` fills its slots from 0
        up, so the used ones are a prefix.

        One device sync per MoE layer, not three: the three counts are stacked
        and fetched together, because each separate ``.item()`` is its own
        pipeline stall and in eager mode that is the whole cost.  Reaching zero
        needs both calls to start at row 0 so the host never learns the split --
        ``fused_moe`` already takes ``num_local_tokens`` as a device tensor, so
        the home call is free; the migration call would need its rows copied
        into their own buffer.

        The home call can no longer start at row 0: an overflow group for a
        lower-numbered expert sits in front of it in group order.  So the home
        segment is bounded on both sides -- ``cu[rank*epn - 1]`` to
        ``cu[(rank+1)*epn - 1]`` -- and whatever lies outside it below ``E``
        belongs to ``overflow_groups()``.
        """

        plan = self.live_plan()
        cu = plan.cu_seqlens
        e = self.cfg.num_experts
        epn = self.cfg.num_experts_per_rank
        lo_e = self.cfg.rank * epn
        sel = plan.experts_to_copy[self.cfg.rank]
        first = cu.new_zeros(1) if lo_e == 0 else cu[lo_e - 1 : lo_e]
        packed = torch.cat(
            [
                first,
                cu[lo_e + epn - 1 : lo_e + epn],
                cu[e - 1 : e],
                cu[-1:],
                (sel >= 0).sum().to(cu.dtype).reshape(1),
            ]
        )
        home_lo, home_hi, mig_lo, mig_hi, nb = packed.tolist()
        return (
            int(home_lo),
            int(home_hi),
            int(mig_lo),
            int(mig_hi),
            int(nb),
        )

    def row_slot_ids(self) -> torch.Tensor:
        """Per-row local weight-pool slot, shaped ``[NvS, 1]`` for a topk-1 MoE.

        MoonEP's rows are already grouped by expert, which is what a grouped
        GEMM wants -- but aiter's ``fused_moe`` reaches that layout through its
        own sorting pass driven by ``topk_ids``.  Handing it one id per row
        turns that pass into a no-op reordering of an already-correct order,
        and lets the whole quantised experts path stay untouched.

        Ids are local to whichever experts call the row belongs to: home rows
        index this rank's ``E/R`` experts, migration rows index the borrowed
        ones from 0.  See ``expert_call_split`` for why the two cannot share
        one weight tensor.

        ``searchsorted`` rather than ``repeat_interleave`` because the latter
        needs the row total on the host, and a device sync per MoE layer would
        cost more than this kernel.
        """

        plan = self.live_plan()
        cu = plan.cu_seqlens
        e = self.cfg.num_experts
        epn = self.cfg.num_experts_per_rank
        g = cu.numel()
        gidx = torch.arange(g, device=cu.device, dtype=torch.int32)
        slot = torch.where(gidx < e, plan.group_expert_ids % epn, gidx - e)
        rows = torch.arange(
            self._act_cfg.num_dispatch_rows, device=cu.device, dtype=torch.int32
        )
        grp = torch.searchsorted(cu.contiguous(), rows, right=True).clamp_(max=g - 1)
        return slot[grp].to(torch.int32).unsqueeze(1)

    def valid_rows(self) -> torch.Tensor:
        """Rows the experts kernel must cover, as ``num_local_tokens``.

        Everything past ``cu_seqlens[-1]`` is untouched dispatch buffer.  Kept
        on device so no MoE layer has to synchronize.
        """

        return self.live_plan().cu_seqlens[-1:].to(torch.int32)

    def get_expert_output_buffer(self) -> torch.Tensor:
        """Where the experts kernel must write; combine reads it in place.

        ``peer_expert_output_ptrs`` is mapped to this buffer
        (``moonep_dispatch_op.py:104``), so peers gather from exactly these
        bytes.  Writing results anywhere else makes combine read stale data --
        silently, since nothing checks the provenance of the rows.
        """

        return self._act_op.expert_output

    def dispatched_weights(self) -> torch.Tensor:
        """The route weight of every dispatched row, in dispatch-row order.

        One scalar per ``(token, expert)`` route, written by the dispatch
        kernel into the destination's symmetric buffer.  The combine kernel is
        built with ``apply_route_weights=False``, so it sums the top-k
        contributions unweighted and the caller owes the weighting -- see
        ``moonep_ep.py``, which multiplies ``expert_output`` by this vector
        before calling combine.
        """

        return self._act_op.recv_route_weights

    def gathered_weights(self) -> torch.Tensor:
        """Per ``(token, k)`` route weight, as combine read it back.

        Combine gathers each route's scalar from the rank that ran the expert,
        so this is ``dispatched_weights`` mapped back to the source token's
        layout -- the same shape as the caller's ``topk_weights``, and equal to
        it if the whole weight path is sound.  Valid only after a combine.
        """

        return self._act_op.gathered_route_weights

    def combine_grouped(self, fused_expert_output: torch.Tensor) -> torch.Tensor:
        """Reduce the grouped expert outputs back to per-token rows."""

        if self._closed:
            raise RuntimeError("op is closed")
        if self._plan is None:
            raise RuntimeError(
                "combine called with no live plan; MoonEP's combine must be "
                "paired with the dispatch that produced its plan."
            )
        if fused_expert_output.data_ptr() != self._act_op.expert_output.data_ptr():
            raise ValueError(
                "MoonEP combine consumes expert_output in place; have the "
                "experts kernel write into get_expert_output_buffer() rather "
                "than passing a copy."
            )
        if self.cfg.enable_push_combine and self.maxvio <= PUSH_MAXVIO_LIMIT:
            # Push wins at balanced-to-typical routing (1.46x / 1.33x at maxvio
            # 0.32 / 9.7) and loses at pathological (0.91x at 47), hence the
            # gate -- but the staging buffer and per-plan reverse-map publish
            # are not wired here yet; moonep_ep_fast.py has a working version.
            raise NotImplementedError(
                "push combine is not available through this adapter yet"
            )
        out, _gathered = self._act_op.combine(self._plan)
        return out

    def live_plan(self) -> MoonEPReferencePlan:
        if self._plan is None:
            raise RuntimeError(
                "no live plan; run_experts/combine must follow the dispatch "
                "that produced the plan"
            )
        return self._plan

    @property
    def maxvio(self) -> float:
        """Global expert-load imbalance of the last dispatched batch.

        Computed on demand: it costs a device sync, and only benchmarks read it.
        """
        if self._tpe_all is None:
            raise RuntimeError("no dispatch has run yet")
        total = self._tpe_all.sum(dim=0).double()
        return float((total.max() / total.mean() - 1.0).item())

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        self._tpe_peer = []
        mori_shmem_free_tensor(self._tpe_symm)
        self._op.close()
        if self._decode_op is not None:
            self._decode_op.close()
        self._closed = True


__all__ = [
    "MoonEPDispatchCombineConfig",
    "MoonEPDispatchCombineIntraNodeOp",
    "PUSH_MAXVIO_LIMIT",
]
