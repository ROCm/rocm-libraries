# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build spec of the ML4 decoder-layer kernel.

The placement (``plan``) and the mailbox byte offsets (``scratch``) are data
the host passes in: they come from the launcher that owns the scratch buffer,
so this family never re-derives a layout that could drift from it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from rocke.helpers.task_schedule import MAPS, Stage, TaskTable

# ``draft``: a DSpark draft layer (no indexer, a bf16 latent cache, the window
# keys in ``indices``, the dense MLP stages)
KINDS = ("rope", "rope_shared", "nope", "nope_kvs", "draft")
# stages this family emits, in program order
STAGES = (
    "qkv_a", "q_norm", "cache", "q_b", "index_q", "uk", "index_score",
    "index_select", "split", "uv", "o", "router", "ug", "down",
    "mlp_ug", "mlp_down",
)  # fmt: skip
# the selection tree's levels below the last (``index_select``)
_SELECT_LEVEL = re.compile(r"index_sel\d+")


@dataclass(frozen=True)
class Ml4LayerSpec:
    """One ML4 decoder layer (attention, indexer, MoE) as one launch.

    ``plan``: ``(stage, tasks, base)`` of every stage of the layer schedule in
    program order, ``base`` the CTA task 0 runs on. ``stages``: the stages
    this build emits (a subset of ``plan``; the others are produced by a
    previous launch or not at all). ``scratch``: ``(region, byte offset)``.
    """

    kind: str = "rope"
    rows: int = 4
    heads: int = 16
    npes: int = 1
    # expert intermediate rows of this rank's shard
    inter: int = 1536
    stages: tuple[str, ...] = ("q_norm",)
    plan: tuple[tuple[str, int, int], ...] = ()
    # task -> CTA maps by stage, "stage:map+stage:map" (``task_schedule.MAPS``;
    # unnamed stages: "affine", the reference placement). ``xcd_blocked`` puts
    # consecutive tasks on one XCD under the round-robin dispatch: split's
    # chunks of a row, up/gate's and down's tiles of the same experts share
    # that XCD's L2
    stage_maps: str = "split:xcd_blocked+ug:xcd_blocked+down:xcd_blocked"
    scratch: tuple[tuple[str, int], ...] = ()
    # tagged mailboxes in IOD-homed dual copies: every put stores both copies
    # (8 KiB apart), every poll reads the one homed at its CTA's IOD (bit p
    # of the mask word at scratch + ``numa_hdr`` names the copy homed at the
    # IOD of XCDs 0-3 on the area's 2 MiB page p). ``numa_hdr``: past every
    # region of every layout sharing the scratch; the copies start at the next
    # 2 MiB boundary past the header. Regions in plain loads / stores
    # (``ctx.NUMA_PLAIN``) keep their one copy
    mb_numa: bool = False
    numa_hdr: int = 0
    # mb_numa: the doubled regions, "a+b+..." ("": every tagged region); the
    # bulk ones (split partials, score planes) pay a second store per pair
    numa_regions: str = ""
    # mb_numa: the kinds it applies to, "rope+nope" ("": all); the others
    # build with one copy (kinds sharing a scratch can mix: the copies sit
    # past every layout's regions)
    numa_kinds: str = ""
    # symmetric (peer-visible) buffer: region byte offsets, e.g. ``attn``,
    # ``ffn`` and ``_part_stride`` (the size of one of its two parity slots)
    sym: tuple[tuple[str, int], ...] = ()
    # layer launches per decode step (the TP reduce slots alternate per launch)
    launches_per_step: int = 1
    grid: int = 256
    threads: int = 512
    layer_slots: int = 128
    # paged index plane: rows per page, block-table row stride (pages) and
    # verify rows per request (0: all rows are one request)
    index_page_size: int = 16
    index_bt_stride: int = 0
    index_request_width: int = 0
    # positions the index plane holds (sets the selection tree's shape)
    index_capacity: int = 32768
    # up/gate: every row's routed tile of a round issued, multiplied and
    # reduced together (one reduction per round instead of one per row)
    ug_batch: bool = True
    # up/gate (batched): issue round r + 1's weights right after round r's MFMAs
    ug_prefetch: bool = True
    # up/gate prefetch: rows whose round r + 1 tiles go early (0: all, which
    # holds two rounds of tiles at once: past the VGPR budget at 4 rows)
    ug_prefetch_rows: int = 2
    # up/gate (batched): issue the shared expert's round-0 tile before routing
    ug_shared_first: bool = True
    # up/gate (batched): issue round 0's routed tiles before staging the
    # activation (they need only the routing)
    ug_routed_early: bool = False
    # down: issue the shared expert's tiles before the mid wait
    dn_shared_early: bool = False
    # down: the shared expert's units and the routed ones in one software
    # pipeline (else the shared ones load and multiply batch by batch first)
    dn_pipe: bool = False
    # down: units per pipeline batch (0: 9 at <= 4 rows, else 2)
    dn_batch: int = 0
    # down: the residual (``a``) polled at task start, not after the reduction
    dn_resid_early: bool = True
    # down (dn_resid_early): the residual polled with the first mid chunk, not
    # alone before it (its wait, in order behind the prefetched weight units,
    # would drain them before the first mid poll goes out)
    dn_resid_with_mid: bool = False
    # uv: the split softmax stats and partials in one poll (one wait, not three)
    uv_batch_poll: bool = True
    # o: the residual (the layer input) loaded at task start, not after the
    # reduction
    o_resid_early: bool = True
    # down: mid blocks per poll (0: all of a wave's blocks in one poll)
    mid_chunk: int = 10
    # down: route again even when up/gate (same launch) left the routing in LDS
    dn_reroute: bool = False
    # up/gate / down: a weight tile's MXFP4 scale byte extracted from its
    # scale word at the MFMA, not right after the word's load (the extract
    # waits on the load, and the in-order vmcnt drains every load before it)
    ug_scale_late: bool = True
    dn_scale_late: bool = True
    # up/gate expert-major (stage_moe_em): the work items are the routing's
    # distinct experts (+ the shared one); CTAs take contiguous ranges of
    # (item, 8-column chunk) tiles and load each tile once for all the rows
    # that route to its expert. ``em_tiles`` tiles per reduction,
    # ``em_prefetch`` of the next batch's issued during this one's reduction
    ug_em: bool = False
    em_tiles: int = 4
    em_prefetch: int = 2
    # expert-major down (implies ug_em): each expert's group runs its down
    # tiles once its own up/gate is done, per-expert partials to ``dn_part``;
    # ``down`` sums a row's partials in slot order. x_out is re-associated
    # (equal to rounding, not bitwise, against the slot-major down). Off for
    # one row (the expert-major up/gate needs two or more)
    moe_em: bool = True
    # ug_em / moe_em: CTAs of a fixed shared-expert group (0: the shared
    # expert is one more item); em_dn_pre: the first down tile's weights
    # issued before the mid poll
    em_shared_ctas: int = 16
    em_dn_pre: int = 1
    # em_scalar: each routed CTA derives its expert and the rows' slots in
    # uniform registers (no LDS item tables, no barriers); em_early: its
    # first chunks' loads go out before the activation staging
    em_scalar: bool = True
    em_early: bool = True
    # up/gate: tasks take the slots in down's accumulation order (round 0 the
    # shared expert and slot 1, round 1 slot 2, round 2 slots 3-4), so whole
    # slots finish rounds early (default: slot 4 and the shared expert first)
    ug_slot_order: bool = False
    # down: mid polled, quantized and multiplied a slot group at a time
    # (shared + 1, 2, 3 + 4), each group's first weights issued before its
    # poll; with ug_slot_order the first groups run while up/gate's last round
    # finishes elsewhere (one row per wave: rows == 4)
    dn_slot_phases: bool = False
    # down (dn_slot_phases): weight units of the next slot group issued before
    # its mid poll (-1: one pipeline batch; fewer hold fewer VGPRs over the poll)
    dn_group_pre: int = -1
    # top-k radix digits: "12_12_8" (three passes) or "8x4" (four)
    select_radix: str = "12_12_8"
    # q_b: the index-query tiles before the task's own (else after)
    index_q_first: bool = True
    # q_b (index_q_first): prefetch the index-query tile's weights instead of
    # q_b's (those issue after the index tiles)
    iq_prefetch: bool = False
    # index_score: the first tiles' block-table lookups and cached keys load
    # before the query wait (they need no query)
    score_prefetch: bool = False
    # index_score: this launch's index keys polled once per CTA into LDS
    # (each tile otherwise polls them four times in a row)
    score_newk_lds: bool = True
    # index_score (with score_newk_lds): no index_ready wait; a CTA polls this
    # launch's index keys only before its first tile that holds one, so the
    # other tiles score once the index query and head weights land
    score_lazy_newk: bool = False
    # index_score: the request's first position loaded once per CTA, not per
    # tile (a chain of three dependent loads ahead of each tile's scoring)
    score_hoist: bool = False
    # indexer kinds: the cache task publishes the index keys, and runs the
    # latent rows (kvnew / penew, first read by split after the top-k) after
    # its CTA's score tiles instead of ahead of them
    cache_kv_late: bool = False
    # the index-query tiles left past the free CTAs as second tiles, loaded
    # with the first, on the first index_q CTAs (not after q_b's tile on q_b's)
    iq_dual: bool = True
    # emit index_score before uk (its tasks on uk's CTAs stop queuing behind
    # uk's wait for every q_b tile)
    score_before_uk: bool = False
    # index key / head-weight projections on the first q_b CTAs (idle until
    # q_norm) instead of qkv_a's, and the extra index-query tiles on the CTAs
    # the plan wraps them onto, short of the cache CTA
    idx_proj_on_qb: bool = False
    # top-k: staged keys padded one word per 32 (conflict-free thread-major reads)
    sel_pad: bool = True
    # top-k: before the ready flag, "fence" (agent release: an L2 writeback) or
    # "waitcnt" (store completion; every published value is an sc1 store)
    sel_release: str = "waitcnt"
    # top-k level 0: per-thread chunks sized to the list, flags counted and
    # recomputed from LDS instead of kept per item
    sel_lean: bool = True
    # top-k level 0: the threshold by one select over buckets sized to the
    # keys' range and a rank of the threshold bucket (radix passes for ties)
    sel_range: bool = False
    # top-k -> split: a single-list row's indices also go out as tagged pairs
    # (isel0's identity plane), which split polls directly: no ready flag, no
    # reload behind it
    sel_tagged: bool = False
    # split: stage the queries before waiting on the top-k indices
    split_q_first: bool = True
    # split (indexer kinds): this launch's rows (kvnew / penew) into LDS
    # before the top-k wait, so patching keys that are those rows reads LDS
    # (rows <= 4)
    split_newkv_lds: bool = True
    # split: after indices_ready, "fence" (agent acquire) or "none" (every
    # producer value the task reads is an sc1 load)
    split_acquire: str = "none"
    # split: the request's block table copied to LDS while the top-k runs, so
    # the slot lookup is not a dependent global load
    split_bt_lds: bool = True
    # draft kind, mlp_ug: two of a task's three tiles in flight from its
    # start (their reductions after the last tile's MFMAs), else one tile's
    # weights streaming during the previous tile's reduction
    mlp_ug_pipe: bool = False
    # draft kind, mlp_down: weight units per pipeline batch (two in flight)
    mlp_dn_batch: int = 4
    # draft kind: w_ug's 16-row tiles hold 8 gate rows and the same 8 up rows
    # (the host interleaves them), so mlp_ug publishes bf16(SiLU(gate) * up)
    # as bf16 pairs and mlp_down stages it as is (a quarter of the bytes of
    # the gate and up values as f32 pairs)
    mlp_act: bool = False
    # s_memrealtime stamps per task (start, end) into the int64 buffer at
    # parameter-table row 7, rows in ``plan`` order (the reference layout)
    timeline: bool = False
    # timeline builds: also the mid-task stamps (inputs landed, compute done):
    # True, False, or the stages that get them. They slow the stages that
    # keep weight loads in flight across them (down most), so by default only
    # the top-k and split stamp them
    tl_inner: bool | str = "index_sel0+index_select+split"
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "plan", tuple(tuple(p) for p in self.plan))
        object.__setattr__(self, "scratch", tuple(tuple(p) for p in self.scratch))
        object.__setattr__(self, "sym", tuple(tuple(p) for p in self.sym))
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not 1 <= self.rows <= 16:
            raise ValueError(f"rows must be in 1..16, got {self.rows}")
        if self.threads != 512:
            raise ValueError("the stages are written for 512 threads (8 waves)")
        if self.mb_numa and self.numa_kinds and self.kind not in self.numa_kinds.split("+"):
            object.__setattr__(self, "mb_numa", False)
        if self.rows < 2:
            object.__setattr__(self, "moe_em", False)
            object.__setattr__(self, "ug_em", False)
        if self.mb_numa:
            past = [name for name, off in self.scratch if off >= self.numa_hdr]
            if self.numa_hdr <= 0 or self.numa_hdr % 64 or past:
                raise ValueError(f"mb_numa: numa_hdr {self.numa_hdr} must be a positive "
                                 f"multiple of 64 past every region (not past: {past})")  # fmt: skip
        planned = {name for name, _, _ in self.plan}
        for s in self.stages:
            if s not in STAGES and not _SELECT_LEVEL.fullmatch(s):
                raise ValueError(f"stage {s!r} is not emitted by this family {STAGES}")
            if s not in planned:
                raise ValueError(f"stage {s!r} has no entry in plan")
            sparse_moe = s in ("index_q", "index_score", "router", "ug", "down") or s.startswith("index_sel")
            if self.draft and sparse_moe:
                raise ValueError(f"stage {s!r} is not a stage of the draft kind")
            if not self.draft and s in ("mlp_ug", "mlp_down"):
                raise ValueError(f"stage {s!r} runs in the draft kind only")

    @property
    def with_indexer(self) -> bool:
        return self.kind not in ("rope_shared", "draft")

    @property
    def draft(self) -> bool:
        return self.kind == "draft"

    @property
    def kernel_name(self) -> str:
        tl = "_tl" if self.timeline else ""
        if self.name:
            return self.name + tl
        return (
            f"ml4_layer_{self.kind}_r{self.rows}_h{self.heads}_tp{self.npes}_"
            + "_".join(self.stages)
            + tl
        )

    def offset(self, region: str) -> int:
        for name, off in self.scratch:
            if name == region:
                return int(off)
        raise KeyError(f"scratch region {region!r} not in spec")

    def sym_offset(self, region: str) -> int:
        for name, off in self.sym:
            if name == region:
                return int(off)
        raise KeyError(f"symmetric region {region!r} not in spec")

    def maps(self) -> dict[str, str]:
        """``stage_maps`` as {stage: map}; a stage this kind does not run (the
        indexer's in rope_shared) is ignored, a name that is no stage is not."""
        out = {}
        for item in filter(None, self.stage_maps.split("+")):
            stage, sep, m = item.partition(":")
            if not sep or m not in MAPS:
                raise ValueError(f"stage_maps item {item!r}: want stage:map, map in {MAPS}")
            if stage not in STAGES and not _SELECT_LEVEL.fullmatch(stage):
                raise ValueError(f"stage_maps: {stage!r} is not a stage of this family")
            out[stage] = m
        return out

    def table(self) -> TaskTable:
        maps = self.maps()
        return TaskTable(
            stages=tuple(Stage(n, t, base=b, map=maps.get(n, "affine")) for n, t, b in self.plan),
            grid=self.grid,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> Ml4LayerSpec:
        return cls(**json.loads(text))
