# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""LDS-staged gfx950 FP6 GEMM on the native 32x32x64 scaled MFMA.

``C[M, N] = A[M, K] @ B[N, K]^T``. A and B are row-major packed FP6 (E2M3 or
E3M2, four codes per three bytes), A_scale[M, K/32] and B_scale[K/32, N] hold
E8M0 bytes, and C is BF16 or FP16 with FP32 accumulation. The ``a_layout``,
``b_layout``, ``a_scale_layout`` and ``b_scale_layout`` fields select
pre-packed K-blocked layouts instead, where one stage of a tile is contiguous
in memory.

One workgroup of ``waves_m * waves_n`` wave64s computes a ``block_m x block_n``
tile; each wave owns ``(block_m / waves_m) x (block_n / waves_n)`` of 32x32
accumulators. K advances in stages of 128 values: every stage copies 96 bytes
of each tile row from global memory into one buffer of an ``stages``-deep LDS
ring with 16-byte buffer-to-LDS DMA, while the MFMAs consume an older buffer.

Fragment reads. A lane of the scaled MFMA holds one aligned run of 32 K values
(24 bytes, one E8M0 scale). The two K64 MFMAs of a stage give lane halves 0/1
K blocks 0/2 and then 3/1 of the stage, so every lane's fragment is one
16-byte-aligned ``ds_read_b128`` plus one ``ds_read_b64`` in register order,
and the two half-chunk reads of a lane are never adjacent (adjacent ones get
merged into one b128 that no fragment can use without register copies). The
products are summed in a different K order than the rows store them; the sum
is the same.

LDS layout. A stage stores each tile row as six 16-byte chunks. With
``lds_swizzle="rot16"`` rows 16..31 of every 32 rotate their chunks by one,
which makes the b128 reads conflict-free under the gfx950 bank model; the b64
reads keep a 2-way conflict because all lanes of a phase read the same chunk
half. The DMA expresses the rotation in its global source addresses.

Every optimization is a defaulted spec field so a tuner can toggle it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

from ...core.arch import ArchTarget
from ...core.arch.mfma_scale import MATRIX_FORMATS
from ...core.dtypes import normalize_dtype
from ...core.ir import (
    BF16,
    F16,
    F32,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    TemporalHint,
    Value,
    VectorType,
)
from ...helpers.spec import SignatureBuilder

_FP6_DTYPES = ("fp6e2m3", "fp6e3m2")
_OUTPUT_DTYPES = {"bf16": BF16, "fp16": F16}
STAGE_K = 128
_ROW_BYTES = STAGE_K * 6 // 8  # 96 bytes of one row per stage
_CHUNKS = _ROW_BYTES // 16  # 16-byte DMA chunks per row per stage
_ROW_DWORDS = _ROW_BYTES // 4
_WAVE = 64
_LDS_BYTES = 160 * 1024
_VMCNT_MAX = 63

LDS_SWIZZLES = ("none", "rot16")
XCD_REMAPS = ("none", "xcd_block")
SCHEDULES = ("none", "iglp0", "iglp1", "interleave")
ACC_FENCES = ("agpr", "none")
MFMA_FORMS = ("intrinsic", "asm_agpr")
ABLATIONS = ("none", "no_mfma", "no_dma", "no_lds_read", "no_scales")
FRAG_PIPELINES = ("stage", "step")
MFMA_GROUPS = (16, 8, 4, 2, 1)
MFMA_ORDERS = ("rows", "serp")
BACKENDS = ("llvm", "direct")
DMA_PROBES = ("none", "l2", "contig", "l2contig")
LAYOUTS = ("row", "kblock")
B_LAYOUTS = (*LAYOUTS, "lanes")
B_SCALE_LAYOUTS = (*LAYOUTS, "kblock4")
A_SCALE_FETCHES = ("stage", "iter", "iter2")
B_SCALE_FETCHES = ("stage", "pair")
A_SCALE_PATHS = ("direct", "lds2")
AS_READS = ("b32", "b128x4")
# a_scale_path="lds2": an epoch of eight stages holds a row's 32 scale bytes,
# two 16-byte halves, in one of two LDS slots.
_EPOCH_STAGES, _EPOCH_BYTES = 8, 32
B_PATHS = ("lds", "vgpr")
DMA_M0S = ("default", "early", "bshare", "early_bshare")
# dma_m0="bshare": a B half's DMAs per wave sit within the LDS-DMA
# instruction offset field (12 bits) of one base.
_DMA_OFFSET_MAX = 4095
CC_PASSES = ("dead_lanes", "m0_fill", "peep_waits", "peep_shifts", "ds_rebase")
CC_EXTRAS = ("lds_waw", "u16_zext", "acc_init_keep", "pin_vmem", "m0_cse")
# Short name tokens: the direct compiler names the code object's file after
# the kernel (`<name>.hsaco`, at most 255 bytes).
_CC_EXTRA_TOKENS = {"lds_waw": "w", "u16_zext": "z", "acc_init_keep": "k", "pin_vmem": "p", "m0_cse": "m"}
_DMA_M0_TOKENS = {"bshare": "bs", "early_bshare": "ebs"}
_NAME_MAX = 255 - len(".hsaco")
ACC_INITS = ("default", "early")
A_RINGS = ("stage", "line", "line2r")
FILL_SPLITS = ("none", "line1")
STEP_VMEM_ORDERS = ("reads_first", "group_late", "b_early", "refill_early", "frag_first", "frag_mid")
# step_vmem_order="frag_mid": the MFMA block after which step 1's first read pair goes.
_FRAG_MID_START = 4
DMA_PLACES = ("default", "even", "even_bfirst", "dense_bfirst", "pair_bfirst")
SYNC_PLACES = ("none", "split", "sink", "split_sink", "split_lgkm")
PROLOGUE_WAITS = ("all", "stage0")
EPILOGUES = ("default", "sa", "pk2", "ct", "ct4", "lds", "ct_lds")
# epilogue="lds" / "ct_lds": C rows a wave stages per pass, and the 16-byte
# LDS chunk an XOR swizzle permutes per row bit (rows 1 / 2 / 4 / 8 apart).
_C_PASS_ROWS = 64
_C_SWIZZLE = (4, 1, 8, 2)
K_ORDERS = ("rot", "t10")
B_RELEASES = ("full", "half")
# b_release="half": a B row holds a stage half in 48 bytes, three 16-byte chunks.
_HALF_ROW_BYTES = _ROW_BYTES // 2
# a_ring="line": three 128-byte lines per A row hold four K128 stages (384 B).
_LINE_BYTES, _LINE_CHUNKS, _LINE_SLOTS, _LINE_PERIOD = 128, 8, 3, 4
# a_ring="line2r": two line slots per A row, the 32 bytes of a window's first
# line that its second stage reads in a remnant buffer, and three B slots; the
# slots' parity flips every window, so the slot pattern repeats every 24 stages.
_L2R_SLOTS, _L2R_B_SLOTS, _L2R_PERIOD, _REMNANT_BYTES = 2, 3, 24, 32
# MFMA blocks between the remnant copy's LDS reads and its writes.
_REMNANT_GAP = 8
# AMD sched_group_barrier masks.
_SG_MFMA, _SG_VMEM_READ, _SG_DS_READ = 0x008, 0x020, 0x100
# scale_opsel: the v_perm_b32 selector of lane half h is _PERM_IDENTITY ^ h *
# _PERM_ROT16_XOR (half 1: the dword rotated by 16 bits); step s then reads
# byte _STEP_SCALE_BYTE[s] of every scale dword.
_PERM_IDENTITY, _PERM_ROT16_XOR = 0x03020100, 0x02020202
_STEP_SCALE_BYTE = (0, 3)
# k_order="t10": lane half h's A scale halfword holds K32 blocks 2h, 2h + 1.
_T10_SCALE_BYTE = (0, 1)


@dataclass(frozen=True)
class TiledBlockScaledGemmSpec:
    """One tiled FP6 GEMM configuration. Only the problem shape is required."""

    M: int
    N: int
    K: int
    name: str = "mxfp6_tiled"
    dtype_a: str = "fp6e2m3"
    dtype_b: str = "fp6e2m3"
    dtype_c: str = "bf16"
    block_m: int = 256
    block_n: int = 256
    waves_m: int = 2
    waves_n: int = 2
    # LDS ring depth in K128 stages; stages - 1 stages are in flight.
    stages: int = 2
    lds_swizzle: str = "rot16"
    # "xcd_block" groups the 32 tiles each XCD runs into a 4 x 8 block so the
    # XCD's L2 serves A rows to 8 tiles and B rows to 4.
    xcd_remap: str = "none"
    num_xcds: int = 8
    # Instruction ordering inside each stage: LLVM's own ("none"), its canned
    # MFMA patterns ("iglp0"/"iglp1"), or explicit MFMA/LDS/VMEM interleave.
    sched: str = "none"
    # Raise wave priority while the stage's MFMAs issue.
    setprio: bool = False
    # 0 leaves occupancy to LLVM; 1 allows the full 512-register budget.
    waves_per_eu: int = 1
    # AGPR budget "min,max" (accumulators live in AGPRs); "auto" leaves it to LLVM.
    agpr_alloc: str = "256,256"
    # AUX cache policy of the A/B DMA (0 = default, 1 = GLC, 2 = streaming).
    dma_cache: int = 0
    # "agpr" passes the loop-carried accumulators through an AGPR-output asm
    # at the back-edge, so the register allocator keeps them in AGPRs.
    acc_fence: str = "none"
    # "intrinsic": LLVM's scaled-MFMA intrinsic (LLVM picks the register file).
    # "asm_agpr": each K64 step's MFMAs are one inline-asm block with tied
    # AGPR accumulators and VGPR fragments and scales, ended by ``asm_tail_nop``
    # wait states because LLVM's hazard recognizer cannot see into the block.
    mfma_form: str = "asm_agpr"
    asm_tail_nop: int = 8
    # Code-object compiler the IR is shaped for. "llvm": COMGR. "direct":
    # rocKE's direct compiler, which schedules and allocates the MFMAs itself
    # and takes no inline asm, so the MFMAs are the intrinsic op, no memory
    # fences pin LDS reads (its barriers order them) and AGPRs are its choice.
    backend: str = "llvm"
    # "stage": read a whole stage's fragments, then run its MFMAs. "step": read
    # each K64 step's fragments one step ahead, during the previous step's
    # MFMAs, with the next stage's barrier between a stage's two steps.
    frag_pipeline: str = "stage"
    # MFMAs per block. Below a whole K64 step (16), the blocks keep their
    # program order and the stage's LDS reads, DMA and scale loads are placed
    # between them, so they issue while the MFMA pipe is busy; `sched` then
    # does not apply. Intrinsic blocks are held in order by scheduling
    # barriers, asm blocks by their side effects.
    mfma_group: int = 16
    # Order of a K64 step's MFMAs: "rows" walks every A row's B columns
    # forward; "serp" walks odd rows backward, so consecutive MFMAs change
    # one operand except at the step boundary.
    mfma_order: str = "rows"
    # Give every LDS buffer its own alias scope. Without scopes the compiler
    # cannot tell a fragment read from the DMA into another buffer and waits
    # for every DMA in flight before each read.
    lds_alias_scopes: bool = False
    # Global layout of the A ([M][K]) and B ([N][K]) FP6 bytes. "row" is
    # row-major. "kblock" stores each 128-wide K block of all rows together
    # ([K/128][rows][96 B]), so one stage of a tile is one contiguous block.
    # B only, "lanes" (with b_release="half"; elsewhere as "kblock"): each
    # stage half [K/128][2] holds per 32-column block the 64 lanes' bytes
    # [0:16) of the K32 block they read, lane-linear, then their [16:24):
    # lane (h, r) column 32c + r, block 2h (half 0) or 3 - 2h (half 1). The
    # 8-byte reads then cover 256 distinct bytes per pass.
    a_layout: str = "row"
    b_layout: str = "row"
    # Global layout of each operand's E8M0 scales. "row" is A_scale[M][K/32]
    # and B_scale[K/32][N] bytes. "kblock" is A_scale[K/128][M] and
    # B_scale[K/128][N] dwords holding one stage's four K32 scales, so a
    # stage's scales of a tile are contiguous. B only, "kblock4": per stage,
    # wave column group and lane (half h, r), two dwords, one per K64 step
    # s, whose byte j is the scale of column 32j + r in the K32 block step s
    # of half h reads ([K/128][N/wave_tile_n][64][2] dwords): one load per
    # stage, and the MFMA of column j selects byte j.
    a_scale_layout: str = "row"
    b_scale_layout: str = "row"
    # kblock4 B scales per stage ("stage"), or ("pair"; with lds2 A scales and
    # B released in halves on the line ring, elsewhere the name changes only)
    # one four-dword load per two stages: stages 2v and 2v + 1's dwords of a
    # lane side by side ([ceil(K/256)][N/wave_tile_n][64][2][2] dwords), the
    # pair loaded by the odd stage before it (pair 0 by the prologue).
    b_scale_fetch: str = "stage"
    # Row-major A scales: one dword per row per stage, or ("iter") one load per
    # row per loop iteration holding the next iteration's `stages` dwords, so a
    # row's cache line is touched once per iteration instead of every stage.
    # "iter2" (with scale_opsel and the line ring's step pipeline; otherwise
    # as "iter"): every other iteration, lane half h loads iteration j+1+h of
    # the row the halves share, and a permlane32 swap per dword gives both
    # halves both iterations: half the group loads and line requests.
    a_scale_fetch: str = "stage"
    # Row-major A scales through LDS instead ("lds2"; the line ring's step
    # pipeline on the direct backend, elsewhere the name changes only; it
    # replaces a_scale_fetch). Each wave copies two 32-row blocks of every
    # row's 32 scale bytes of the next epoch of eight stages, one 16-byte DMA
    # half per lane half, into the free one of two LDS slots
    # ([slot][row block][half][32 rows][16 B]); each stage reads its dword
    # per row from LDS with the next stage's fragments.
    a_scale_path: str = "direct"
    # lds2's A scale reads: "b32", one dword per row block and stage; or
    # "b128x4", one 16-byte read per row block every four stages (a row's
    # dwords of four stages are contiguous), kept in registers until the
    # next, so a read pass covers 256 distinct bytes. Elsewhere the name
    # changes only.
    as_read: str = "b32"
    # Where row-major A lives in LDS. "stage": one K128 slot per stage, as B.
    # "line": a ring of three whole 128-byte lines per row, refilled one line
    # per DMA, so every DMA reads whole cache lines (a row's K128 slice is 96
    # bytes and straddles lines); B keeps `stages` (two) K128 slots.
    # "line2r" (the step pipeline on the direct backend; elsewhere as "line"):
    # two line slots per row, line l in slot l % 2, and a remnant buffer. A
    # window of four stages reads three lines; its first stage copies, in LDS,
    # the 32 bytes of its first line that the second stage reads, so that
    # line's slot takes the window's third line after the first stage. B gets
    # three K128 slots released whole (it replaces stages and b_release), and
    # the barrier between a stage's steps is the only one.
    a_ring: str = "stage"
    # Split MFMA blocks of the stage pipeline: where a stage's fillers go.
    # "none": the step-1 reads, all of the stage's DMA and the next stage's
    # scale loads share step 0's blocks. "line1": step 0 keeps the reads, B's
    # DMA and the scale loads; A's DMA (with the line ring its refill) and the
    # per-iteration A scale loads go between step 1's blocks, the loads in the
    # iteration's second stage.
    fill_split: str = "none"
    # Select each MFMA operand's scale byte with the instruction's op_sel
    # instead of a VALU shift per operand and step. Lane half 1 keeps its
    # scale dwords rotated by 16 bits (one v_perm per dword once it has
    # landed), so both halves take byte 0 in step 0 and byte 3 in step 1.
    # Asm MFMAs write the selects as op_sel / op_sel_hi; intrinsic MFMAs carry
    # them as the `scale_sel` attribute, which the direct compiler encodes.
    scale_opsel: bool = False
    # Run the K loop one iteration short without the end-of-K clamps (next
    # stage, scale stage, line refill, A scale group start), which only the
    # last iteration needs; that iteration is emitted after the loop with
    # them. The stage ring's step pipeline ignores it.
    k_peel: bool = False
    # The line ring's step pipeline: where each step's VMEM ops go among its
    # MFMA blocks. "reads_first": round-robin after the fragment reads (step
    # 0: the next stage's scale loads, the refill, the A scale group; step 1:
    # B's DMA). "group_late": the A scale group follows B's DMA in step 1.
    # "b_early": B's DMA one per second block from step 1's first, the reads
    # spread. "refill_early": step 0's VMEM one per block from its first.
    # "frag_first": step 1's reads (the next stage's step-0 fragments) one
    # per block from its first, its A scale dwords with the first ones, then
    # B's DMA one per block, so a barrier at the end of the stage finds the
    # reads landed. "frag_mid": B's DMA one per block from step 1's first,
    # then the reads with their A scale dwords one per block from the
    # fifth, clear of the window right after the barrier where the refill
    # lands. Other pipelines ignore it.
    step_vmem_order: str = "reads_first"
    # The line ring's step pipeline with b_release="half" on the direct
    # backend: where its DMAs go among a step's MFMA blocks. "default": as
    # step_vmem_order puts them. "even": every DMA in an odd gap (the slot
    # after an odd block), so a wave's DMAs are two MFMA blocks apart. A
    # line refill goes in two halves, the first in step 1 of the stage before
    # (its slot is free from that stage's barrier), the second at the start
    # of its own step 0, then B's half; a step with no refill takes B's half
    # one per four blocks. Step 0's scale loads go to the even gaps after the
    # second, the A scale DMA late, its fragment reads to the even gaps; step
    # 1's reads stay where step_vmem_order puts them. "even_bfirst": B's
    # half ahead of the refill's. "dense_bfirst" / "pair_bfirst": as
    # "even_bfirst" (B's half first), but a step's DMAs one / two per gap in
    # consecutive gaps from gap 1, B's alone included. Other pipelines
    # ignore it.
    dma_place: str = "default"
    # With dma_place: where each barrier group (lgkmcnt(0), the landed-DMA
    # vmcnt, s_barrier) sits. "none": after the step's last MFMA block.
    # "split": the waits after block n - 2, the barrier after block n - 1.
    # "sink": the whole group after the next step's first block, whose
    # operands were read long before. "split_sink": the waits after block
    # n - 2, the barrier after the next step's first block. "split_lgkm":
    # only lgkmcnt(0) after block n - 2. With "split*" no fragment read goes
    # in a step's last two gaps. Elsewhere the name changes only.
    sync_place: str = "none"
    # The line ring's step pipeline with b_release="half": the prologue
    # waits for all of its DMA ("all") or ("stage0") issues line 1 last and
    # waits only for what stage 0 reads; Bar(1) waits for line 1. Elsewhere
    # the name changes only.
    prologue_wait: str = "all"
    # How C is stored. "default": one bf16 store per accumulator element at
    # a per-element address. "sa": the lane's part of the address once, the
    # rest uniform. "pk2": as "sa", the two rows of an element pair rounded
    # in one packed convert. "ct" (intrinsic MFMAs; elsewhere as "sa"): the
    # MFMAs take B as their first operand, so an accumulator holds C's
    # transpose and a lane four consecutive columns of a row, one 8-byte
    # store each; "ct4" pairs them across the lane halves (permlane32 swap)
    # into 16-byte stores. "lds" / "ct_lds" (a line ring on the direct
    # backend, 128-column wave tiles; elsewhere as "sa" / "ct"): after a
    # barrier that frees the ring, each wave stages 64 rows at a time in its
    # own part of it, row-major with an XOR-swizzled 16-byte chunk per row,
    # and stores whole rows, 16 bytes per lane, 4 rows per instruction.
    # "lds" writes a dword per element pair (neighbour lanes swap a row
    # through a DPP quad permute), "ct_lds" 8 bytes per lane from "ct"'s
    # accumulators.
    epilogue: str = "default"
    # The streaming (nontemporal) hint on C's stores.
    store_nt: bool = False
    # The K32 block lane half h takes in step s of a stage. "rot": 2h in
    # step 0, 3 - 2h in step 1. "t10" (the line ring's step pipeline with
    # b_release="half", kblock4 B scales, scale_opsel and lds2 A scales on
    # the direct backend): 2h + s. A lane's 48 A bytes are then contiguous:
    # step 0 reads the 16-byte chunks at 48h and 48h + 16 (the second's
    # upper 8 bytes, unused by the FP6 operand, are step 1's first), step 1
    # the 8 bytes at 48h + 24 and the chunk at 48h + 32. Its A scales are
    # the halfword at byte 2h of the stage dword (ds_read_u16), bytes 0 and
    # 1 for the two steps, with no rotation. B's half 1 holds blocks 1, 3 and kblock4's dword s blocks
    # 2h + s (`bench`). Other pipelines ignore it.
    k_order: str = "rot"
    # The line ring's step pipeline with kblock B: "half" releases each B
    # slot in two K halves instead of whole. B is then packed
    # [K/128][2][N][48 B]: half 0 holds the K32 blocks step 0 reads (0, 2),
    # half 1 those of step 1 (3, 1), each row as x[0:16] x[16:24] y[16:24]
    # y[0:16] for its blocks (x, y), so lane half h reads b128 at byte 32h and
    # b64 at byte 16 + 8h of its half. A B slot is two sub-slots; a second
    # barrier at the end of every stage frees the step-0 half, whose DMA two
    # stages ahead then issues a step earlier. Other pipelines ignore it.
    b_release: str = "full"
    # Where B's fragments come from. "lds": DMA into LDS slots, then LDS
    # reads. "vgpr" (the line ring's step pipeline on the direct backend;
    # elsewhere the name changes only): 1 x 4 waves, each streaming its own
    # 64 columns of B from global memory straight into registers, a step and
    # a half ahead of their MFMAs, with no B in LDS. B is then packed per
    # stage, 32-column block and K64 step as the 64 lanes' 16-byte pieces,
    # then (after all of those) their 8-byte pieces: lane (h, r) of step s
    # holds bytes [0:16) and [16:24) of column 32c + r's K32 block 2h (s = 0)
    # or 3 - 2h (s = 1). B scales are one dword per stage and lane of the
    # wave's column group ([K/128][N/64][64]), byte 2s + j the scale of
    # column 32j + r in that step's block. It replaces waves_m / waves_n,
    # b_layout, b_scale_layout and b_release.
    b_path: str = "lds"
    # The direct backend's M0 writes for the LDS DMA loads: "default", or
    # "early" (kernel attribute m0_early): each moves up past the MFMA before
    # it, so no s_nop is needed between it and its load. Other backends
    # ignore it. "bshare" (b_release="half", at most four DMAs per B half):
    # wave w's DMAs of a half fill the half's consecutive 1 KiB regions from
    # w times their count on, so all of them take one LDS base and one
    # voffset, the per-DMA step going into the instruction offset; with
    # kernel attribute m0_cse their M0 writes after the first drop.
    # "early_bshare" is both. Elsewhere "bshare" is the name only.
    dma_m0: str = "default"
    # Direct-backend compiler passes to enable, as kernel attributes: "none",
    # or names from CC_PASSES joined with "+" ("all" for every one). Other
    # backends ignore it.
    cc_passes: str = "none"
    # More direct-backend compiler passes, the same way, from CC_EXTRAS.
    cc_extra: str = "none"
    # The accumulators' zero initialization on the line ring's step pipeline
    # (direct backend): "default" after the prologue's wait, or "early"
    # before it (kernel attribute acc_init_keep, which emits the loop's
    # initial accumulator copies where the zeros are), so the AGPR writes
    # overlap the prologue's DMA. Elsewhere the name only.
    acc_init: str = "default"
    # Diagnostic only, results are wrong when set: every tile streams the
    # first tile's rows so the DMA hits L2 ("l2"), the DMA reads each stage's
    # rows as one contiguous block of the same bytes ("contig"), or both.
    dma_probe: str = "none"
    # Diagnostic only, results are wrong when set: drop the MFMAs ("no_mfma"),
    # the A/B DMA ("no_dma"), the fragment LDS reads ("no_lds_read") or the
    # scale loads ("no_scales") to time the rest of the pipeline. Join
    # several with "+".
    ablate: str = "none"

    def __post_init__(self) -> None:
        for field in ("dtype_a", "dtype_b", "dtype_c"):
            object.__setattr__(self, field, normalize_dtype(getattr(self, field)))
        if self.b_vgpr:
            object.__setattr__(self, "waves_m", 1)
            object.__setattr__(self, "waves_n", 4)
        if self.a_line2r:
            object.__setattr__(self, "stages", _L2R_B_SLOTS)
            object.__setattr__(self, "b_release", "full")

    @property
    def b_vgpr(self) -> bool:
        """B streams into registers (`b_path`)."""
        return self.b_path == "vgpr" and (self.frag_pipeline, self.a_ring, self.backend) == ("step", "line", "direct")

    @property
    def a_line2r(self) -> bool:
        """A in two line slots and a remnant buffer, B in three slots (`a_ring`)."""
        return self.a_ring == "line2r" and (self.frag_pipeline, self.backend) == ("step", "direct")

    @property
    def threads(self) -> int:
        return _WAVE * self.waves_m * self.waves_n

    @property
    def block_size(self) -> int:
        return self.threads

    @property
    def wave_tile_m(self) -> int:
        return self.block_m // self.waves_m

    @property
    def wave_tile_n(self) -> int:
        return self.block_n // self.waves_n

    @property
    def a_lds_scales(self) -> bool:
        """A scales come through LDS (`a_scale_path`)."""
        return self.a_scale_path != "direct" and self.a_ring in A_RINGS[1:] and (
            self.frag_pipeline, self.a_scale_layout, self.backend) == ("step", "row", "direct")

    @property
    def lds_bytes(self) -> int:
        scales = 2 * self.block_m * _EPOCH_BYTES if self.a_lds_scales else 0
        if self.a_line2r:
            a = _L2R_SLOTS * self.block_m * _LINE_BYTES + self.block_m * _REMNANT_BYTES
            return a + self.stages * self.block_n * _ROW_BYTES + scales
        if self.a_ring in A_RINGS[1:]:
            b = 0 if self.b_vgpr else self.stages * self.block_n * _ROW_BYTES
            return _LINE_SLOTS * self.block_m * _LINE_BYTES + b + scales
        return self.stages * (self.block_m + self.block_n) * _ROW_BYTES

    @property
    def unroll(self) -> int:
        """K128 stages per K-loop iteration: the ring depth, or the line
        ring's period of four stages."""
        return _LINE_PERIOD if self.a_ring in A_RINGS[1:] else self.stages

    def ablated(self, what: str) -> bool:
        return what in self.ablate.split("+")

    @property
    def b_split(self) -> bool:
        """B is released, and packed, in two K halves (`b_release`)."""
        return self.b_release == "half" and self.frag_pipeline == "step" and self.b_layout in B_LAYOUTS[1:] and (
            self.a_ring in A_RINGS[1:]) and not self.b_vgpr

    @property
    def b_lanes(self) -> bool:
        """B's stage halves are packed lane-linear (`b_layout`)."""
        return self.b_layout == "lanes" and self.b_split

    @property
    def c_staged(self) -> bool:
        """C's stores go through the line ring's LDS (`epilogue`)."""
        if self.epilogue not in ("lds", "ct_lds") or self.a_ring not in A_RINGS[1:] or self.backend != "direct":
            return False
        if self.a_line2r:
            ring = (_L2R_B_SLOTS * self.block_n * _ROW_BYTES + _L2R_SLOTS * self.block_m * _LINE_BYTES
                    + self.block_m * _REMNANT_BYTES)
        else:
            ring = _LINE_SLOTS * self.block_m * _LINE_BYTES
        return (self.wave_tile_n == 128 and self.wave_tile_m % _C_PASS_ROWS == 0
                and self.waves_m * self.waves_n * _C_PASS_ROWS * self.wave_tile_n * 2 <= ring)

    @property
    def dma_placed(self) -> bool:
        """The step pipeline's DMAs go to fixed gaps (`dma_place`)."""
        return self.dma_place != "default" and self.b_split and self.a_ring == "line" and self.backend == "direct"

    @property
    def sync_placed(self) -> bool:
        """The barrier groups move (`sync_place`)."""
        return self.sync_place != "none" and self.dma_placed

    @property
    def b_scale_pairs(self) -> bool:
        """B scales load two stages at a time (`b_scale_fetch`)."""
        return self.b_scale_fetch == "pair" and self.b_split and self.a_ring == "line" and (
            self.b_scale_layout == "kblock4" and self.a_lds_scales)

    @property
    def b_share(self) -> bool:
        """A B half's DMAs share one LDS base per wave (`dma_m0`)."""
        dmas = self.block_n * _HALF_ROW_BYTES // 16 // self.threads
        return self.dma_m0 in ("bshare", "early_bshare") and self.b_split and self.a_ring == "line" and (
            self.backend == "direct") and (dmas - 1) * _WAVE * 16 <= _DMA_OFFSET_MAX

    @property
    def k_t10(self) -> bool:
        """Lane half h takes K32 block 2h + s in step s (`k_order`)."""
        return self.k_order == "t10" and self.b_split and self.a_ring == "line" and self.backend == "direct" and (
            self.b_scale_layout == "kblock4" and self.scale_opsel and self.a_lds_scales)

    def kernel_name(self) -> str:
        a, b = (d.removeprefix("fp6") for d in (self.dtype_a, self.dtype_b))
        knobs = (
            f"s{self.stages}_{self.lds_swizzle}_{self.xcd_remap}_{self.sched}"
            f"_p{int(self.setprio)}_w{self.waves_per_eu}"
            f"_a{self.agpr_alloc.replace(',', 'x')}_c{self.dma_cache}_f{self.acc_fence}"
            f"_{self.mfma_form}{self.asm_tail_nop if self.mfma_form == 'asm_agpr' else ''}"
            + ("" if self.frag_pipeline == "stage" else f"_fp{self.frag_pipeline}")
            + ("" if self.mfma_group == 16 else f"_g{self.mfma_group}")
            + ("" if self.mfma_order == "rows" else f"_mo{self.mfma_order}")
            + ("_aa" if self.lds_alias_scopes else "")
            + ("" if (self.a_layout, self.b_layout) == ("row", "row") else f"_L{self.a_layout[0]}{self.b_layout[0]}")
            + (
                ""
                if (self.a_scale_layout, self.b_scale_layout) == ("row", "row")
                else f"_S{self.a_scale_layout[0]}{self.b_scale_layout[0]}{self.b_scale_layout[6:]}"
            )
            + ("" if self.a_scale_fetch == "stage" else f"_as{self.a_scale_fetch}")
            + ("" if self.b_scale_fetch == "stage" else f"_bs{self.b_scale_fetch}")
            + ("" if self.a_scale_path == "direct" else f"_ap{self.a_scale_path}")
            + ("" if self.as_read == "b32" else f"_ar{self.as_read}")
            + ("" if self.a_ring == "stage" else f"_A{self.a_ring}")
            + ("" if self.fill_split == "none" else f"_fs{self.fill_split}")
            + ("_so" if self.scale_opsel else "")
            + ("_kp" if self.k_peel else "")
            + ("" if self.step_vmem_order == "reads_first" else f"_vo{self.step_vmem_order}")
            + ("" if self.dma_place == "default" else f"_dp{self.dma_place}")
            + ("" if self.sync_place == "none" else f"_sp{self.sync_place}")
            + ("" if self.prologue_wait == "all" else f"_pw{self.prologue_wait}")
            + ("" if self.epilogue == "default" else f"_ep{self.epilogue}")
            + ("_snt" if self.store_nt else "")
            + ("" if self.k_order == "rot" else f"_ko{self.k_order}")
            + ("" if self.b_release == "full" else f"_br{self.b_release}")
            + ("" if self.b_path == "lds" else f"_bp{self.b_path}")
            + ("" if self.dma_m0 == "default" else f"_m0{_DMA_M0_TOKENS.get(self.dma_m0, self.dma_m0)}")
            + ("" if self.cc_passes == "none" else f"_cc{self.cc_passes.replace('+', '_')}")
            + ("" if self.cc_extra == "none" else "_cx" + (self.cc_extra if self.cc_extra == "all" else "".join(
                _CC_EXTRA_TOKENS.get(p, p) for p in self.cc_extra.split("+"))))
            + ("" if self.acc_init == "default" else f"_ai{self.acc_init}")
            + ("" if self.dma_probe == "none" else f"_D{self.dma_probe}")
            + ("" if self.backend == "llvm" else f"_{self.backend}")
            + ("" if self.ablate == "none" else f"_X{self.ablate.replace('+', '_')}")
        )
        return (
            f"{self.name}_{a}_{b}_{self.dtype_c}_M{self.M}N{self.N}K{self.K}"
            f"_t{self.block_m}x{self.block_n}_w{self.waves_m}x{self.waves_n}_{knobs}"
        )


def _pitch(spec: TiledBlockScaledGemmSpec) -> int:
    return spec.K * 3 // 4


class _Once:
    """A value emitted on first use, so deferred emitters keep the IR order."""

    def __init__(self, make: Callable[[], Value]):
        self._make, self._value = make, None

    def __call__(self) -> Value:
        if self._value is None:
            self._value = self._make()
        return self._value


class _GroupDwords:
    """An A scale group given dword by dword: `get(i, e)` is row i's dword e."""

    def __init__(self, get: Callable[[int, int], Value]):
        self.get = get


def _round_robin(*lists: Sequence) -> list:
    return [x for group in itertools.zip_longest(*lists) for x in group if x is not None]


def _vmem_per_stage(spec: TiledBlockScaledGemmSpec) -> int:
    """VMEM ops one stage issues: its share of the A/B DMA and the scales.

    The compiler orders the two freely within a stage, so a barrier's vmcnt
    counts whole stages.
    """
    dma = 0 if spec.ablated("no_dma") else (spec.block_m + spec.block_n) * _CHUNKS // spec.threads
    b_loads = 1 if spec.b_scale_layout == "kblock" else 2  # per 32 columns
    # Per-iteration A scale loads issue in one stage of each iteration only.
    a_loads = 0 if spec.a_scale_fetch != "stage" else 1
    b_vmem = 1 if spec.b_scale_layout == "kblock4" else b_loads * spec.wave_tile_n // 32
    scales = 0 if spec.ablated("no_scales") else a_loads * spec.wave_tile_m // 32 + b_vmem
    return dma + scales


def is_valid_spec(spec: TiledBlockScaledGemmSpec, arch: str = "gfx950") -> tuple[bool, str]:
    if arch != "gfx950":
        return False, "tiled scaled-MFMA GEMM requires gfx950"
    if spec.dtype_a not in _FP6_DTYPES or spec.dtype_b not in _FP6_DTYPES:
        return False, "A and B must be FP6 (fp6e2m3 or fp6e3m2)"
    if spec.dtype_c not in _OUTPUT_DTYPES:
        return False, "output must be bf16 or fp16"
    if spec.lds_swizzle not in LDS_SWIZZLES:
        return False, f"lds_swizzle must be one of {LDS_SWIZZLES}"
    if spec.xcd_remap not in XCD_REMAPS:
        return False, f"xcd_remap must be one of {XCD_REMAPS}"
    if spec.sched not in SCHEDULES:
        return False, f"sched must be one of {SCHEDULES}"
    if spec.dma_cache not in (0, 1, 2, 3):
        return False, "dma_cache must be 0..3"
    if spec.waves_per_eu not in (0, 1, 2):
        return False, "waves_per_eu must be 0, 1 or 2"
    if spec.acc_fence not in ACC_FENCES:
        return False, f"acc_fence must be one of {ACC_FENCES}"
    if spec.mfma_form not in MFMA_FORMS:
        return False, f"mfma_form must be one of {MFMA_FORMS}"
    if not 0 <= spec.asm_tail_nop <= 15:
        return False, "asm_tail_nop must be 0..15"
    if spec.ablate != "none" and not set(spec.ablate.split("+")) <= set(ABLATIONS[1:]):
        return False, f"ablate must be 'none' or a '+'-joined set of {ABLATIONS[1:]}"
    if spec.frag_pipeline not in FRAG_PIPELINES:
        return False, f"frag_pipeline must be one of {FRAG_PIPELINES}"
    if spec.dma_probe not in DMA_PROBES:
        return False, f"dma_probe must be one of {DMA_PROBES}"
    if spec.a_layout not in LAYOUTS:
        return False, f"a_layout must be one of {LAYOUTS}"
    if spec.b_layout not in B_LAYOUTS:
        return False, f"b_layout must be one of {B_LAYOUTS}"
    if spec.a_scale_layout not in LAYOUTS:
        return False, f"a_scale_layout must be one of {LAYOUTS}"
    if spec.b_scale_layout not in B_SCALE_LAYOUTS:
        return False, f"b_scale_layout must be one of {B_SCALE_LAYOUTS}"
    if spec.b_scale_layout == "kblock4" and spec.wave_tile_n > 4 * 32:
        return False, "b_scale_layout=kblock4 holds at most 4 columns per lane: wave_tile_n <= 128"
    if spec.a_scale_fetch not in A_SCALE_FETCHES:
        return False, f"a_scale_fetch must be one of {A_SCALE_FETCHES}"
    if spec.b_scale_fetch not in B_SCALE_FETCHES:
        return False, f"b_scale_fetch must be one of {B_SCALE_FETCHES}"
    if spec.a_scale_fetch != "stage":
        if spec.a_scale_layout != "row" or (spec.frag_pipeline, spec.a_ring) == ("step", "stage"):
            return False, "a_scale_fetch=iter takes row-major A scales and the stage pipeline (or the line ring)"
        if spec.K // STAGE_K < spec.unroll:
            return False, "a_scale_fetch=iter needs at least one loop iteration of K stages"
    if spec.a_scale_path not in A_SCALE_PATHS:
        return False, f"a_scale_path must be one of {A_SCALE_PATHS}"
    if spec.as_read not in AS_READS:
        return False, f"as_read must be one of {AS_READS}"
    if spec.a_lds_scales and spec.block_m // 32 % (spec.threads // _WAVE):
        return False, "a_scale_path=lds2 gives every wave the same number of 32-row blocks"
    if spec.a_ring not in A_RINGS:
        return False, f"a_ring must be one of {A_RINGS}"
    if spec.a_ring == "line" or (spec.a_ring == "line2r" and not spec.a_line2r):
        if (spec.a_layout, spec.stages, spec.dma_probe) != ("row", 2, "none"):
            return False, "a_ring=line takes row-major A, stages=2 (B's ring) and no DMA probe"
        if spec.block_m * _LINE_CHUNKS % spec.threads:
            return False, "a_ring=line needs block_m * 8 to be a multiple of the thread count"
    if spec.a_line2r:
        if (spec.a_layout, spec.dma_probe) != ("row", "none"):
            return False, "a_ring=line2r takes row-major A and no DMA probe"
        if spec.block_m % spec.threads or spec.K // STAGE_K < 2:
            return False, "a_ring=line2r copies whole rows per thread (block_m a multiple of the thread count), K >= 256"
    if spec.fill_split not in FILL_SPLITS:
        return False, f"fill_split must be one of {FILL_SPLITS}"
    if spec.step_vmem_order not in STEP_VMEM_ORDERS:
        return False, f"step_vmem_order must be one of {STEP_VMEM_ORDERS}"
    if spec.dma_place not in DMA_PLACES:
        return False, f"dma_place must be one of {DMA_PLACES}"
    if spec.dma_placed and spec.step_vmem_order not in ("reads_first", "frag_first", "frag_mid"):
        return False, "dma_place places step 1's reads as reads_first, frag_first or frag_mid does"
    if spec.sync_place not in SYNC_PLACES:
        return False, f"sync_place must be one of {SYNC_PLACES}"
    if spec.prologue_wait not in PROLOGUE_WAITS:
        return False, f"prologue_wait must be one of {PROLOGUE_WAITS}"
    if spec.epilogue not in EPILOGUES:
        return False, f"epilogue must be one of {EPILOGUES}"
    if spec.k_order not in K_ORDERS:
        return False, f"k_order must be one of {K_ORDERS}"
    if spec.b_release not in B_RELEASES:
        return False, f"b_release must be one of {B_RELEASES}"
    if spec.b_split and spec.block_n * _HALF_ROW_BYTES // 16 % spec.threads:
        return False, "b_release=half needs the 16-byte chunks of a B stage half (block_n * 3) to be a multiple of the thread count"
    if spec.b_path not in B_PATHS:
        return False, f"b_path must be one of {B_PATHS}"
    if spec.dma_m0 not in DMA_M0S:
        return False, f"dma_m0 must be one of {DMA_M0S}"
    if spec.cc_passes not in ("none", "all"):
        names = spec.cc_passes.split("+")
        if any(p not in CC_PASSES for p in names) or len(set(names)) != len(names):
            return False, f"cc_passes must be 'none', 'all' or distinct names from {CC_PASSES} joined with '+'"
    if spec.cc_extra not in ("none", "all"):
        names = spec.cc_extra.split("+")
        if any(p not in CC_EXTRAS for p in names) or len(set(names)) != len(names):
            return False, f"cc_extra must be 'none', 'all' or distinct names from {CC_EXTRAS} joined with '+'"
    if spec.acc_init not in ACC_INITS:
        return False, f"acc_init must be one of {ACC_INITS}"
    if spec.b_vgpr and spec.wave_tile_n > 64:
        return False, "b_path=vgpr takes at most two 32-column blocks per wave (one scale dword)"
    if spec.scale_opsel:
        if (spec.mfma_form, spec.backend) == ("intrinsic", "llvm"):
            return False, "scale_opsel takes asm MFMAs or the direct backend: the LLVM intrinsic call has no scale selects"
        if (spec.frag_pipeline, spec.a_ring) == ("step", "stage"):
            return False, "scale_opsel takes the stage pipeline (or the line ring)"
    if spec.mfma_group not in MFMA_GROUPS:
        return False, f"mfma_group must be one of {MFMA_GROUPS}"
    if spec.mfma_order not in MFMA_ORDERS:
        return False, f"mfma_order must be one of {MFMA_ORDERS}"
    if spec.backend not in BACKENDS:
        return False, f"backend must be one of {BACKENDS}"
    if spec.backend == "direct" and (spec.mfma_form, spec.agpr_alloc, spec.acc_fence) != ("intrinsic", "auto", "none"):
        return False, "backend='direct' takes no inline asm: mfma_form='intrinsic', agpr_alloc='auto', acc_fence='none'"
    step_mfmas = (spec.wave_tile_m // 32) * (spec.wave_tile_n // 32)
    if spec.mfma_group < step_mfmas and step_mfmas % spec.mfma_group:
        return False, "mfma_group must divide the MFMAs of a K64 step"
    if spec.mfma_form == "asm_agpr" and spec.agpr_alloc != "auto":
        lo, hi = (int(v) for v in spec.agpr_alloc.split(","))
        need = (spec.wave_tile_m // 32) * (spec.wave_tile_n // 32) * 16
        if hi < need:
            return False, f"asm_agpr keeps {need} accumulator registers in AGPRs"
    if spec.agpr_alloc != "auto":
        try:
            lo, hi = (int(v) for v in spec.agpr_alloc.split(","))
        except ValueError:
            return False, "agpr_alloc must be 'auto' or 'min,max'"
        if not 0 <= lo <= hi <= 256:
            return False, "agpr_alloc must satisfy 0 <= min <= max <= 256"
    if min(spec.M, spec.N, spec.K) <= 0:
        return False, "M, N and K must be positive"
    if spec.M % spec.block_m or spec.N % spec.block_n:
        return False, "M and N must be multiples of block_m and block_n"
    if spec.K % STAGE_K:
        return False, f"K must be a multiple of {STAGE_K}"
    if not 2 <= spec.stages <= 4 or spec.K // STAGE_K < spec.stages - 1:
        return False, "stages must be 2..4 with at least stages-1 K128 stages"
    for tile, waves in ((spec.block_m, spec.waves_m), (spec.block_n, spec.waves_n)):
        if waves <= 0 or tile % (32 * waves):
            return False, "each wave tile must be a positive multiple of 32"
    if spec.threads > 1024:
        return False, "at most 16 waves per workgroup"
    blocks = (spec.wave_tile_m // 32) * (spec.wave_tile_n // 32)
    if blocks * 16 > 256:
        return False, "a wave's accumulators exceed 256 registers per lane"
    for rows in (spec.block_m, spec.block_n):
        if rows * _CHUNKS % spec.threads:
            return False, "tile rows * 6 must be a multiple of the thread count"
    if (spec.stages - 2) * _vmem_per_stage(spec) > _VMCNT_MAX:
        return False, "too many DMA loads in flight for vmcnt"
    if spec.lds_bytes > _LDS_BYTES:
        return False, f"LDS ring needs {spec.lds_bytes} bytes (> {_LDS_BYTES})"
    if max(spec.M, spec.N) * _pitch(spec) > 0x7FFFFFFF or spec.M * spec.N > 0x7FFFFFFF:
        return False, "tensor byte offsets exceed i32"
    if spec.xcd_remap == "xcd_block":
        tiles_m, tiles_n = spec.M // spec.block_m, spec.N // spec.block_n
        per_xcd, rem = divmod(tiles_m * tiles_n, spec.num_xcds)
        if rem or per_xcd != 32 or tiles_m % 4 or tiles_n % 8:
            return False, "xcd_block needs 32 tiles per XCD in a grid of 4 x 8 blocks"
    if spec.backend == "direct" and len(spec.kernel_name()) > _NAME_MAX:
        return False, f"the kernel name is longer than {_NAME_MAX} bytes (the direct compiler's file name)"
    return True, "ok"


def tiled_block_scaled_gemm_grid(spec: TiledBlockScaledGemmSpec) -> tuple[int, int, int]:
    return (spec.M // spec.block_m) * (spec.N // spec.block_n), 1, 1


def tiled_block_scaled_gemm_signature(spec: TiledBlockScaledGemmSpec):
    signature = SignatureBuilder()
    for name, dtype in (
        ("A", "i8"),
        ("B", "i8"),
        ("A_scale", "i32"),
        ("B_scale", "i32" if spec.b_scale_layout == "kblock" else "i8"),
        ("C", spec.dtype_c),
    ):
        signature.ptr(name, dtype)
    for name in ("M", "N", "K"):
        signature.scalar(name, "i32")
    return signature.build()


def _rotation(ir: IRBuilder, spec: TiledBlockScaledGemmSpec, row: Value) -> Value:
    """Chunk rotation (0 or 1) of a tile row; depends only on row % 32."""
    if spec.lds_swizzle == "none":
        return ir.const_i32(0)
    return ir.land(ir.lshr(row, ir.const_i32(4)), ir.const_i32(1))


def _line_rotation(ir: IRBuilder, spec: TiledBlockScaledGemmSpec, row: Value) -> Value:
    """16-byte chunk rotation of a row's A lines, (row >> 1) & 7: the rows one
    b128 read pass covers then hit distinct banks. Depends only on row % 16."""
    if spec.lds_swizzle == "none":
        return ir.const_i32(0)
    return ir.land(ir.lshr(row, ir.const_i32(1)), ir.const_i32(_LINE_CHUNKS - 1))


def _remnant_swizzle(ir: IRBuilder, spec: TiledBlockScaledGemmSpec, row: Value) -> Value:
    """a_ring="line2r": which of its two 16-byte remnant chunks a row stores
    first, ((row >> 3) ^ (row >> 2)) & 1: the rows one b128 read or write pass
    covers then hit distinct banks. Depends only on row % 16."""
    if spec.lds_swizzle == "none":
        return ir.const_i32(0)
    return ir.land(ir.xor(ir.lshr(row, ir.const_i32(3)), ir.lshr(row, ir.const_i32(2))), ir.const_i32(1))


def _l2r_source(q: int, step: int, piece: str, half: int) -> tuple[int | str, int, int]:
    """a_ring="line2r": where lane half `half` finds its `piece` ("b128" or
    "b64") of `step` in stage q (0..7) of the slot pattern: line slot 0 or 1,
    or "R" (the remnant buffer); the 16-byte chunk of the line (of R); the
    byte in the chunk. Stage q reads row bytes [96 u, 96 u + 96), u = q % 4,
    of its window's lines 3n..3n+2, line 3n + j in slot (q // 4 + j) % 2;
    line 3n's last 32 bytes, which only phase 1 reads, come from R."""
    u, parity = q % _LINE_PERIOD, q // _LINE_PERIOD
    off = {(0, "b128"): 48 * half, (0, "b64"): 48 * half + 16, (1, "b64"): 72 - 48 * half,
           (1, "b128"): 80 - 48 * half}[(step, piece)]
    line, byte = divmod(_ROW_BYTES * u + off, _LINE_BYTES)
    if line == 0 and byte >= _LINE_BYTES - _REMNANT_BYTES:
        byte -= _LINE_BYTES - _REMNANT_BYTES
        return "R", byte // 16, byte % 16
    return (parity + line) % _L2R_SLOTS, byte // 16, byte % 16


def _tile_origin(ir: IRBuilder, spec: TiledBlockScaledGemmSpec) -> tuple[Value, Value]:
    tiles_n = spec.N // spec.block_n
    bid = ir.block_id_x()
    if spec.xcd_remap == "xcd_block":
        # Hardware places workgroup i on XCD i % num_xcds; give each XCD a
        # 4 x 8 block of tiles, walked row-major inside the block.
        xcd = ir.mod(bid, ir.const_i32(spec.num_xcds))
        local = ir.div(bid, ir.const_i32(spec.num_xcds))
        blocks_n = tiles_n // 8
        tm = ir.add(
            ir.mul(ir.div(xcd, ir.const_i32(blocks_n)), ir.const_i32(4)),
            ir.div(local, ir.const_i32(8)),
        )
        tn = ir.add(
            ir.mul(ir.mod(xcd, ir.const_i32(blocks_n)), ir.const_i32(8)),
            ir.mod(local, ir.const_i32(8)),
        )
    else:
        tm = ir.div(bid, ir.const_i32(tiles_n))
        tn = ir.mod(bid, ir.const_i32(tiles_n))
    return ir.mul(tm, ir.const_i32(spec.block_m)), ir.mul(tn, ir.const_i32(spec.block_n))


def build_tiled_block_scaled_gemm(
    spec: TiledBlockScaledGemmSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """Build the tiled FP6 GEMM; see the module docstring for the contract."""
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid tiled block-scaled GEMM spec for {arch}: {reason}")
    atom = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="mfma_scaled",
        a_dtype=spec.dtype_a,
        b_dtype=spec.dtype_b,
        c_dtype="fp32",
        scales=("e8m0", "e8m0", 32),
        m=32,
        n=32,
        k=64,
    )
    T = spec.threads
    S = spec.stages
    U = spec.unroll
    KT = spec.K // STAGE_K
    pitch = _pitch(spec)
    tm_w, tn_w = spec.wave_tile_m // 32, spec.wave_tile_n // 32
    passes_a = spec.block_m * _CHUNKS // T
    passes_b = spec.block_n * _CHUNKS // T
    dma_ops = passes_a + passes_b

    ir = IRBuilder(spec.kernel_name())
    ir.kernel.attrs["max_workgroup_size"] = T
    if spec.waves_per_eu:
        ir.kernel.attrs["waves_per_eu"] = (spec.waves_per_eu, spec.waves_per_eu)
    if spec.agpr_alloc != "auto":
        ir.kernel.attrs["agpr_alloc"] = spec.agpr_alloc
    if spec.lds_alias_scopes:
        ir.kernel.attrs["lds_alias_scopes"] = True
    if spec.a_ring in A_RINGS[1:] and spec.backend == "direct":
        # The ring is one allocation, read while a refill lands in another of
        # its slots; the stage barriers' waits order every DMA.
        ir.kernel.attrs["lds_dma_explicit"] = True
    if spec.dma_m0 in ("early", "early_bshare") and spec.backend == "direct":
        ir.kernel.attrs["m0_early"] = True
    if spec.cc_passes != "none" and spec.backend == "direct":
        for p in CC_PASSES if spec.cc_passes == "all" else spec.cc_passes.split("+"):
            ir.kernel.attrs[p] = True
    if spec.cc_extra != "none" and spec.backend == "direct":
        for p in CC_EXTRAS if spec.cc_extra == "all" else spec.cc_extra.split("+"):
            ir.kernel.attrs[p] = True
    b_share = spec.b_share
    if b_share:
        ir.kernel.attrs["m0_cse"] = True
    acc_early = spec.acc_init == "early" and spec.a_ring == "line" and spec.frag_pipeline == "step" and (
        spec.backend == "direct")
    if acc_early:
        ir.kernel.attrs["acc_init_keep"] = True

    A = ir.param("A", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    B = ir.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    AS = ir.param("A_scale", PtrType(I32, "global"), noalias=True, readonly=True, align=4)
    a_kblock_scales = spec.a_scale_layout == "kblock"
    b_vgpr = spec.b_vgpr
    b_kblock_scales = spec.b_scale_layout == "kblock" and not b_vgpr
    b_kblock4 = spec.b_scale_layout == "kblock4" and not b_vgpr
    b_pairs = spec.b_scale_pairs
    BS = ir.param(
        "B_scale", PtrType(I32 if b_kblock_scales or b_kblock4 or b_vgpr else I8, "global"), noalias=True,
        readonly=True, align=8 if b_kblock4 else 4 if b_kblock_scales or b_vgpr else 1,
    )
    c_ty = _OUTPUT_DTYPES[spec.dtype_c]
    C = ir.param("C", PtrType(c_ty, "global"), noalias=True, writeonly=True, align=16)
    for name in ("M", "N", "K"):
        ir.param(name, I32)

    c32 = ir.const_i32(32)
    tid = ir.thread_id_x()
    lane = ir.mod(tid, ir.const_i32(_WAVE))
    wave = ir.to_sgpr_u32(ir.div(tid, ir.const_i32(_WAVE)))
    r32 = ir.mod(lane, c32)
    half = ir.div(lane, c32)
    wm = ir.div(wave, ir.const_i32(spec.waves_n))
    wn = ir.mod(wave, ir.const_i32(spec.waves_n))
    m0, n0 = _tile_origin(ir, spec)

    # ---- DMA plan: chunk q of a stage lands at LDS byte 16*q --------------
    a_rsrc = ir.buffer_rsrc(A, ir.const_i32(spec.M * pitch))
    b_rsrc = ir.buffer_rsrc(B, ir.const_i32(spec.N * pitch))
    def addressing(layout: str, rows: int) -> tuple[int, int]:
        """(row pitch, stage step) in bytes of one operand's source."""
        if layout in B_LAYOUTS[1:] or spec.dma_probe in ("contig", "l2contig"):
            return _ROW_BYTES, rows * _ROW_BYTES
        return pitch, _ROW_BYTES

    (a_pitch, a_step), (b_pitch, b_step) = addressing(spec.a_layout, spec.M), addressing(spec.b_layout, spec.N)

    def dma_voffsets(passes: int, row_pitch: int) -> list[Value]:
        offsets = []
        for p in range(passes):
            q = ir.add(tid, ir.const_i32(p * T))
            row = ir.div(q, ir.const_i32(_CHUNKS))
            pos = ir.mod(q, ir.const_i32(_CHUNKS))
            # Inverse rotation: the chunk this LDS position holds.
            chunk = ir.mod(
                ir.sub(ir.add(pos, ir.const_i32(_CHUNKS)), _rotation(ir, spec, row)),
                ir.const_i32(_CHUNKS),
            )
            offsets.append(
                ir.add(ir.mul(row, ir.const_i32(row_pitch)), ir.mul(chunk, ir.const_i32(16)))
            )
        return offsets

    a_line = spec.a_ring in A_RINGS[1:]
    l2r = spec.a_line2r
    voff_a = [] if a_line else dma_voffsets(passes_a, a_pitch)
    same = (passes_b, b_pitch) == (passes_a, a_pitch) and not a_line
    b_split = spec.b_split
    t10 = spec.k_t10
    voff_b = [] if b_split or b_vgpr else voff_a if same else dma_voffsets(passes_b, b_pitch)
    # b_split: a stage half of the tile is contiguous; lane q of pass p copies
    # its 16 bytes q to LDS byte 16q of the sub-slot.
    half_passes = spec.block_n * _HALF_ROW_BYTES // 16 // T
    voff_half = [ir.mul(ir.add(tid, ir.const_i32(p * T)), ir.const_i32(16)) for p in range(half_passes)] if b_split else []
    a_lds = [] if a_line else [ir.smem_alloc(I32, [spec.block_m * _ROW_DWORDS], name_hint=f"a_lds{s}") for s in range(S)]
    b_lds = [] if b_vgpr or l2r else [
        ir.smem_alloc(I32, [spec.block_n * _ROW_DWORDS], name_hint=f"b_lds{s}") for s in range(S)
    ]
    # One allocation: the two halves of a wave read a fragment from different
    # lines, so its slot is per lane.
    # line2r: one allocation for B's three slots, then A's two line slots and
    # the remnant buffer R, so B's slot offsets fit the DS offset field and
    # A's per-lane slot and R offsets sit in the address registers.
    l2r_b_slot = spec.block_n * _ROW_BYTES
    l2r_a0 = _L2R_B_SLOTS * l2r_b_slot
    l2r_r0 = l2r_a0 + _L2R_SLOTS * spec.block_m * _LINE_BYTES
    if l2r:
        a_ring_lds = ir.smem_alloc(I32, [(l2r_r0 + spec.block_m * _REMNANT_BYTES) // 4], name_hint="lines_b")
    else:
        a_ring_lds = ir.smem_alloc(I32, [_LINE_SLOTS * spec.block_m * _LINE_BYTES // 4], name_hint="a_lines") if a_line else None
    wave_lds = ir.zext(ir.to_sgpr_u32(ir.mul(wave, ir.const_i32(_WAVE * 16))), I64)
    a_wave_base = [ir.smem_ptr_add(ir.smem_addr_of(x), wave_lds) for x in a_lds]
    b_wave_base = [ir.smem_ptr_add(ir.smem_addr_of(x), wave_lds) for x in b_lds]
    if l2r:
        l2r_base = ir.smem_addr_of(a_ring_lds)
        b_wave_base = [ir.smem_ptr_add(ir.smem_ptr_add(l2r_base, ir.const_i64(s * l2r_b_slot)), wave_lds) if s else
                       ir.smem_ptr_add(l2r_base, wave_lds) for s in range(_L2R_B_SLOTS)]
        a_ring_wave = ir.smem_ptr_add(ir.smem_ptr_add(l2r_base, ir.const_i64(l2r_a0)), wave_lds)
    else:
        a_ring_wave = ir.smem_ptr_add(ir.smem_addr_of(a_ring_lds), wave_lds) if a_line else None
    if spec.dma_probe in ("l2", "l2contig"):
        a_soff0 = b_soff0 = ir.const_i32(0)
    else:
        a_soff0 = ir.mul(m0, ir.const_i32(a_pitch))
        b_soff0 = ir.mul(n0, ir.const_i32(b_pitch))
    c_last = ir.const_i32(KT - 1)
    # Off while k_peel emits the steady-state loop, which stays inside K.
    clamping = [True]

    def clamp(v: Value, last: Value) -> Value:
        return ir.select(ir.cmp_lt(v, last), v, last) if clamping[0] else v

    def clamp_stage(kt: Value) -> Value:
        return clamp(kt, c_last)

    # The line ring's barrier waits: every VMEM op in program order since the
    # prologue's vmcnt(0), tagged B / A (DMA; with b_split BF / BG, B's two
    # halves), s (scale load), G (A scale group), and "|0" / "|1" where a step
    # of the step pipeline ends. A barrier waits for the last op of its `need`
    # tags (before the latest `before` mark, when given); what was issued after
    # it may stay in flight. Waits inside the K loop are tightened once the
    # body is known (its back edge enters them too).
    vmem_log: list[str] = []
    loop_waits: list[list] = []

    def logged(tag: str, result: Value | None = None) -> Value | None:
        vmem_log.append(tag)
        return result

    def in_flight(log: Sequence[str], need: Sequence[str], before: str | None, nth: int = 0) -> int:
        end = len(log)
        if before is not None:
            marks = [i for i, t in enumerate(log) if t == before]
            if len(marks) <= nth:
                return _VMCNT_MAX
            end = marks[-1 - nth]
        hits = [i for i in range(end) if log[i] in need]
        if not hits:
            return _VMCNT_MAX
        return sum(not t.startswith("|") for t in log[hits[-1] + 1 :])

    def landed_wait(need: Sequence[str] = ("B",), before: str | None = None, nth: int = 0) -> None:
        """`nth`: count to the nth-latest `before` mark (a wait emitted after
        the step whose end it guards)."""
        op = ir.s_waitcnt(vmcnt=in_flight(vmem_log, need, before, nth))
        if loop_waits:
            loop_waits[-1].append((op, len(vmem_log), need, before, nth))

    def open_loop() -> None:
        loop_waits.append([len(vmem_log)])

    def close_loop() -> None:
        start, *waits = loop_waits.pop()
        body = vmem_log[start:]
        for op, at, need, before, nth in waits:
            op.attrs["vmcnt"] = min(op.attrs["vmcnt"], in_flight(body + vmem_log[start:at], need, before, nth))

    def dma_thunks(kt: Value, buf: int, prologue: bool = False, operands: str = "ab",
                   b_tag: str = "B") -> list[Callable[[], None]]:
        """One callable per DMA instruction of a stage, A then B (of `operands`;
        B's logged as `b_tag`).

        `no_dma` keeps the prologue's DMA: with LDS never written the compiler
        folds every fragment read away and the ablation would time no reads.
        """
        if spec.ablated("no_dma") and not prologue:
            return []
        out = []
        for name, rsrc, bases, voffs, soff0, step in (
            ("a", a_rsrc, a_wave_base, voff_a, a_soff0, a_step),
            ("b", b_rsrc, b_wave_base, voff_b, b_soff0, b_step),
        ):
            if name not in operands:
                continue
            soff = _Once(lambda soff0=soff0, step=step: ir.add(soff0, ir.mul(kt, ir.const_i32(step))))
            tag = b_tag if name == "b" else "A"
            for p, voff in enumerate(voffs):

                def issue(rsrc=rsrc, base=bases[buf], voff=voff, soff=soff, p=p, tag=tag):
                    s = soff()
                    dst = ir.smem_ptr_add(base, ir.const_i64(p * T * 16)) if p else base
                    ir.async_buffer_load_lds_addr(rsrc, dst, voff, s, 4, coherency=spec.dma_cache)
                    logged(tag)

                out.append(issue)
        return out

    b_half_soff0 = ir.mul(n0, ir.const_i32(_HALF_ROW_BYTES)) if b_split else None
    half_bytes = spec.block_n * _HALF_ROW_BYTES
    if b_share:
        # Wave w's DMA p of a half fills region w * half_passes + p of the
        # half's 1 KiB regions, the bytes it holds without bshare too.
        region = _WAVE * 16
        share_lds = ir.zext(ir.to_sgpr_u32(ir.mul(wave, ir.const_i32(half_passes * region))), I64)
        share_base = [[ir.smem_ptr_add(ir.smem_addr_of(x), share_lds) if not h else
                       ir.smem_ptr_add(ir.smem_ptr_add(ir.smem_addr_of(x), ir.const_i64(h * half_bytes)), share_lds)
                       for h in range(2)] for x in b_lds]
        share_voff = ir.add(ir.mul(tid, ir.const_i32(16)), ir.mul(wave, ir.const_i32((half_passes - 1) * region)))

    def b_half_thunks(kt: Value, buf: int, h: int, prologue: bool = False) -> list[Callable[[], None]]:
        """b_split: one callable per DMA instruction of B's stage `kt`, half
        `h`, into sub-slot h of slot `buf`."""
        if spec.ablated("no_dma") and not prologue:
            return []
        soff = _Once(lambda: ir.add(b_half_soff0, ir.add(ir.mul(kt, ir.const_i32(spec.N * _ROW_BYTES)),
                                                         ir.const_i32(h * spec.N * _HALF_ROW_BYTES))))
        out = []
        for p, voff in enumerate(voff_half):

            def issue(p=p, voff=voff):
                s = soff()
                if b_share:
                    ir.async_buffer_load_lds_addr(b_rsrc, share_base[buf][h], share_voff, s, 4,
                                                  coherency=spec.dma_cache, offset=p * region)
                    logged("BG" if h else "BF")
                    return
                off = h * half_bytes + p * T * 16
                dst = ir.smem_ptr_add(b_wave_base[buf], ir.const_i64(off)) if off else b_wave_base[buf]
                ir.async_buffer_load_lds_addr(b_rsrc, dst, voff, s, 4, coherency=spec.dma_cache)
                logged("BG" if h else "BF")

            out.append(issue)
        return out

    def issue_dma(kt: Value, buf: int, prologue: bool = False) -> None:
        for issue in dma_thunks(kt, buf, prologue):
            issue()

    # ---- a_ring=line: A arrives one whole 128-byte line per row and DMA ---
    # Lane q of pass p writes LDS byte 16*q of the slot: row q // 8, LDS chunk
    # q % 8, which holds the line's chunk rotated by `_line_rotation`.
    # line2r: a slot stores row r = 32 b + 2 k + e at byte k * 8 block_m +
    # 256 b + 128 e (rows 2k, 2k+1 of every 32-row block b side by side), so a
    # 32-row block is 256 bytes further in the slots as in R and one DS offset
    # serves a read whose lanes are split between them; LDS chunk q holds row
    # 32 ((q / 16) % (block_m / 32)) + 2 (q / (block_m / 2)) + (q / 8) % 2.
    line_passes = spec.block_m * _LINE_CHUNKS // T
    voff_line, c_line_last = [], None
    if a_line:
        for p in range(line_passes):
            q = ir.add(tid, ir.const_i32(p * T))
            if l2r:
                row = ir.add(
                    ir.add(ir.mul(ir.mod(ir.div(q, ir.const_i32(16)), ir.const_i32(spec.block_m // 32)), c32),
                           ir.mul(ir.div(q, ir.const_i32(spec.block_m // 2)), ir.const_i32(2))),
                    ir.mod(ir.div(q, ir.const_i32(8)), ir.const_i32(2)))
            else:
                row = ir.lshr(q, ir.const_i32(3))
            chunk = ir.land(ir.add(ir.land(q, ir.const_i32(_LINE_CHUNKS - 1)), _line_rotation(ir, spec, row)),
                            ir.const_i32(_LINE_CHUNKS - 1))
            voff_line.append(ir.add(ir.mul(row, ir.const_i32(pitch)), ir.mul(chunk, ir.const_i32(16))))
        c_line_last = ir.const_i32(-(-3 * spec.K // (4 * _LINE_BYTES)) - 1)

    def line_thunks(line: Value, slot: int, prologue: bool = False) -> list[Callable[[], None]]:
        """One callable per DMA instruction of A line `line` (clamped to a
        row's last) into ring slot `slot`."""
        if spec.ablated("no_dma") and not prologue:
            return []
        clamped = _Once(lambda: clamp(line, c_line_last))
        soff = _Once(lambda: ir.add(a_soff0, ir.mul(clamped(), ir.const_i32(_LINE_BYTES))))
        out = []
        for p, voff in enumerate(voff_line):
            def issue(p=p, voff=voff):
                off = slot * spec.block_m * _LINE_BYTES + p * T * 16
                dst = ir.smem_ptr_add(a_ring_wave, ir.const_i64(off)) if off else a_ring_wave
                ir.async_buffer_load_lds_addr(a_rsrc, dst, voff, soff(), 4, coherency=spec.dma_cache)
                logged("A")

            out.append(issue)
        return out

    # Stage 4n+u reads bytes [96u, 96u+96) of lines 3n..3n+2 of each row. At
    # its start, after the barrier, every phase but 1 refills the line whose
    # slot the last stage freed: line 3n+c into slot c % 3, two stages before
    # its first use.
    line_refill_line = {0: 2, 2: 3, 3: 4}

    def line_refill(n_it: Value, phase: int, parity: int | None = None,
                    part: int | None = None) -> list[Callable[[], None]]:
        """The refill of stage `phase` of window `n_it`; line2r: into slot
        (n + c) % 2, n's `parity` known, as the window's step 1 issues it.
        `part` 0 / 1: only the first / second half of its passes."""
        if phase not in line_refill_line:
            return []
        c = line_refill_line[phase]
        slot = c % _LINE_SLOTS if parity is None else (parity + c) % _L2R_SLOTS
        out = line_thunks(ir.add(ir.mul(n_it, ir.const_i32(_LINE_SLOTS)), ir.const_i32(c)), slot)
        if part is None:
            return out
        return out[len(out) // 2 :] if part else out[: len(out) // 2]

    # ---- per-lane fragment and scale addressing ---------------------------
    rot = _rotation(ir, spec, r32)

    def chunk_dword(chunk: Value) -> Value:
        return ir.mul(ir.mod(ir.add(chunk, rot), ir.const_i32(_CHUNKS)), ir.const_i32(4))

    three_h = ir.mul(half, ir.const_i32(3))
    # step 0 (K block 2h): b128 at chunk 3h, b64 at the first half of chunk
    # 3h+1. step 1 (K block 3-2h): b64 at the second half of chunk 4-3h, b128
    # at chunk 5-3h.
    piece = {
        (0, "b128"): chunk_dword(three_h),
        (0, "b64"): chunk_dword(ir.add(three_h, ir.const_i32(1))),
        (1, "b64"): ir.add(chunk_dword(ir.sub(ir.const_i32(4), three_h)), ir.const_i32(2)),
        (1, "b128"): chunk_dword(ir.sub(ir.const_i32(5), three_h)),
    }
    a_row_dw = ir.mul(ir.add(ir.mul(wm, ir.const_i32(spec.wave_tile_m)), r32), ir.const_i32(_ROW_DWORDS))
    b_row_dw = ir.mul(ir.add(ir.mul(wn, ir.const_i32(spec.wave_tile_n)), r32), ir.const_i32(_ROW_DWORDS))
    a_idx = {key: ir.add(a_row_dw, v) for key, v in piece.items()}
    b_idx = {key: ir.add(b_row_dw, v) for key, v in piece.items()}
    block_dw = 32 * _ROW_DWORDS
    zero_pad = ir.vector_splat(ir.const_i32(0), 2)

    asm_mfma = spec.mfma_form == "asm_agpr"
    # epilogue="ct": the accumulators hold C's transpose.
    c_t = spec.epilogue in ("ct", "ct4", "ct_lds") and not asm_mfma
    atom_t = ArchTarget.from_gfx(arch).mma.op_for_shape(
        family="mfma_scaled", a_dtype=spec.dtype_b, b_dtype=spec.dtype_a, c_dtype="fp32",
        scales=("e8m0", "e8m0", 32), m=32, n=32, k=64,
    ) if c_t else None

    def fragment(smem: Value, idx: dict, step: int, blk: int, base_dw: int = 0) -> Value:
        if spec.ablated("no_lds_read"):
            seed = ir.add(idx[(step, "b128")], ir.const_i32(blk))
            return ir.vector_splat(seed, 6 if asm_mfma else 8)
        off = ir.const_i32(base_dw + blk * block_dw)
        wide = ir.smem_load_vN(smem, ir.add(idx[(step, "b128")], off), dtype=I32, n=4)
        narrow = ir.smem_load_vN(smem, ir.add(idx[(step, "b64")], off), dtype=I32, n=2)
        live = ir.vec_concat(wide, narrow) if step == 0 else ir.vec_concat(narrow, wide)
        # The asm operand is the six live dwords an FP6 operand occupies; the
        # intrinsic takes the zero-padded eight-dword ABI.
        return live if asm_mfma else ir.vec_concat(live, zero_pad)

    # a_ring=line: the pieces of a lane's fragment sit at row bytes 96u + o,
    # o as in `piece` (step 0: 48h and 48h+16; step 1: 72-48h and 80-48h);
    # their line, and so their slot, depend on the lane half. One dword index
    # per (phase u, step, piece), for the wave's first 32-row block.
    line_idx = {}
    if a_line and not l2r:
        line_rot = _line_rotation(ir, spec, r32)
        a_row = ir.add(ir.mul(wm, ir.const_i32(spec.wave_tile_m)), r32)
        h48 = ir.mul(half, ir.const_i32(48))
        # k_order="t10": the 16-byte chunks at 48h and 48h+16 (step 0), and
        # the 8 bytes at 48h+24 with the chunk at 48h+32 (step 1).
        piece_byte = {
            (0, "b128"): h48,
            (0, "c1"): ir.add(h48, ir.const_i32(16)),
            (1, "b64"): ir.add(h48, ir.const_i32(24)),
            (1, "b128"): ir.add(h48, ir.const_i32(32)),
        } if t10 else {
            (0, "b128"): h48,
            (0, "b64"): ir.add(h48, ir.const_i32(16)),
            (1, "b64"): ir.sub(ir.const_i32(72), h48),
            (1, "b128"): ir.sub(ir.const_i32(80), h48),
        }
        for phase in range(_LINE_PERIOD):
            for key, off in piece_byte.items():
                pos = ir.add(off, ir.const_i32(_ROW_BYTES * phase))
                w = ir.land(pos, ir.const_i32(_LINE_BYTES - 1))
                lds_chunk = ir.land(ir.sub(ir.add(ir.lshr(w, ir.const_i32(4)), ir.const_i32(_LINE_CHUNKS)), line_rot),
                                    ir.const_i32(_LINE_CHUNKS - 1))
                slot_dw = ir.mul(ir.lshr(pos, ir.const_i32(7)), ir.const_i32(spec.block_m * _LINE_BYTES // 4))
                row_dw = ir.mul(a_row, ir.const_i32(_LINE_BYTES // 4))
                in_chunk = ir.lshr(ir.land(w, ir.const_i32(15)), ir.const_i32(2))
                line_idx[(phase, *key)] = ir.add(ir.add(slot_dw, row_dw), ir.add(ir.mul(lds_chunk, ir.const_i32(4)), in_chunk))

    # line2r: R stores row r = 32 b + 8 g + m at byte g * 8 block_m + 256 b +
    # 32 m (its two chunks in the order `_remnant_swizzle` gives), the slots
    # as above. The index of a (stage q of the slot pattern, step, piece)
    # selects between the lane halves' sources (`_l2r_source`), so the region
    # offsets stay in the register and only the 32-row block's 256 bytes go to
    # the DS offset. The remnant copy: thread t copies rows t + T j; lane half
    # h reads the tail's chunk k ^ h first, which keeps its addresses
    # per-lane too.
    copy_src: list[Value] = []
    copy_dst: list[Value] = []
    if l2r:
        lo_half = ir.cmp_lt(half, ir.const_i32(1))
        cls = ir.const_i32(spec.block_m * 8)

        def row_bases(row: Value) -> tuple[Value, Value, Value, Value]:
            """(slot byte, R byte, line rotation, R swizzle) of tile row `row`."""
            blk = ir.mul(ir.lshr(row, ir.const_i32(5)), ir.const_i32(256))
            slot_row = ir.add(ir.add(ir.mul(ir.land(ir.lshr(row, ir.const_i32(1)), ir.const_i32(15)), cls), blk),
                              ir.mul(ir.land(row, ir.const_i32(1)), ir.const_i32(_LINE_BYTES)))
            r_row = ir.add(ir.add(ir.mul(ir.land(ir.lshr(row, ir.const_i32(3)), ir.const_i32(3)), cls), blk),
                           ir.mul(ir.land(row, ir.const_i32(7)), ir.const_i32(_REMNANT_BYTES)))
            return slot_row, r_row, _line_rotation(ir, spec, row), _remnant_swizzle(ir, spec, row)

        def l2r_dword(bases: tuple, src: int | str, chunk: int, byte: int) -> Value:
            slot_row, r_row, line_rot, swz = bases
            if src == "R":
                at = ir.add(r_row, ir.mul(ir.xor(swz, ir.const_i32(chunk)), ir.const_i32(16)))
                return ir.lshr(ir.add(at, ir.const_i32(l2r_r0 + byte)), ir.const_i32(2))
            pos = ir.land(ir.sub(ir.const_i32(chunk + _LINE_CHUNKS), line_rot), ir.const_i32(_LINE_CHUNKS - 1))
            at = ir.add(slot_row, ir.mul(pos, ir.const_i32(16)))
            return ir.lshr(ir.add(at, ir.const_i32(l2r_a0 + src * spec.block_m * _LINE_BYTES + byte)), ir.const_i32(2))

        bases = row_bases(ir.add(ir.mul(wm, ir.const_i32(spec.wave_tile_m)), r32))
        for q in range(_LINE_PERIOD * _L2R_SLOTS):
            for key in ((0, "b128"), (0, "b64"), (1, "b64"), (1, "b128")):
                h0, h1 = (l2r_dword(bases, *_l2r_source(q, *key, h)) for h in (0, 1))
                line_idx[(q, *key)] = ir.select(lo_half, h0, h1)
        first = _LINE_CHUNKS - _REMNANT_BYTES // 16
        for j in range(spec.block_m // T):
            rows = row_bases(ir.add(tid, ir.const_i32(j * T)))
            for k in range(_REMNANT_BYTES // 16):
                copy_src.append(ir.select(lo_half, l2r_dword(rows, 0, first + k, 0), l2r_dword(rows, 0, first + 1 - k, 0)))
                copy_dst.append(ir.select(lo_half, l2r_dword(rows, "R", k, 0), l2r_dword(rows, "R", 1 - k, 0)))

    def remnant_thunks(parity: int) -> tuple[list[Callable[[], None]], list[Callable[[], None]]]:
        """line2r: the remnant copy of a window whose first line is in slot
        `parity`: the thread's LDS reads of its rows' last 32 bytes, and the
        writes of what they read into R."""
        vals: list[Value] = [None] * len(copy_src)
        slot_dw = parity * spec.block_m * _LINE_BYTES // 4

        def read(k: int) -> None:
            vals[k] = ir.smem_load_vN(a_ring_lds, ir.add(copy_src[k], ir.const_i32(slot_dw)) if slot_dw else copy_src[k],
                                      dtype=I32, n=4)

        return ([lambda k=k: read(k) for k in range(len(vals))],
                [lambda k=k: ir.smem_store_vN(a_ring_lds, [copy_dst[k]], vals[k], 4) for k in range(len(vals))])

    line_blk_dw = 256 // 4 if l2r else 32 * _LINE_BYTES // 4

    def line_fragment(phase: int, step: int, blk: int) -> Value:
        if spec.ablated("no_lds_read"):
            seed = ir.add(line_idx[(phase, step, "b128")], ir.const_i32(blk))
            return ir.vector_splat(seed, 6 if asm_mfma else 8)
        off = ir.const_i32(blk * line_blk_dw)
        if t10 and step == 0:
            return ir.vec_concat(*(ir.smem_load_vN(a_ring_lds, ir.add(line_idx[(phase, 0, k)], off), dtype=I32, n=4)
                                   for k in ("b128", "c1")))
        wide = ir.smem_load_vN(a_ring_lds, ir.add(line_idx[(phase, step, "b128")], off), dtype=I32, n=4)
        narrow = ir.smem_load_vN(a_ring_lds, ir.add(line_idx[(phase, step, "b64")], off), dtype=I32, n=2)
        live = ir.vec_concat(wide, narrow) if step == 0 else ir.vec_concat(narrow, wide)
        return live if asm_mfma else ir.vec_concat(live, zero_pad)

    def a_fragment(buf: int, step: int, blk: int) -> Value:
        """A fragment of stage slot `buf`, or with the line ring of phase `buf`."""
        return line_fragment(buf, step, blk) if a_line else fragment(a_lds[buf], a_idx, step, blk)

    # b_split: a lane's B piece of either step sits in that step's sub-slot,
    # rows of 48 bytes (12 dwords), b128 at dword 8h and b64 at dword 4 + 2h.
    # b_lanes: per 32-column block (384 dwords), b128 at dword 4l, b64 at
    # dword 256 + 2l.
    if spec.b_lanes:
        b_lane_blk = ir.mul(wn, ir.const_i32(tn_w * 32 * _HALF_ROW_BYTES // 4))
        b_half_idx = {
            "b128": ir.add(b_lane_blk, ir.mul(lane, ir.const_i32(4))),
            "b64": ir.add(b_lane_blk, ir.add(ir.mul(lane, ir.const_i32(2)), ir.const_i32(256))),
        }
    elif b_split:
        b_half_row = ir.mul(ir.add(ir.mul(wn, ir.const_i32(spec.wave_tile_n)), r32), ir.const_i32(_HALF_ROW_BYTES // 4))
        b_half_idx = {
            "b128": ir.add(b_half_row, ir.mul(half, ir.const_i32(8))),
            "b64": ir.add(b_half_row, ir.add(ir.mul(half, ir.const_i32(2)), ir.const_i32(4))),
        }

    def b_fragment(buf: int, step: int, blk: int) -> Value:
        """B fragment of slot `buf % stages`."""
        if l2r:
            return fragment(a_ring_lds, b_idx, step, blk, base_dw=buf % S * l2r_b_slot // 4)
        if not b_split:
            return fragment(b_lds[buf % S], b_idx, step, blk)
        if spec.ablated("no_lds_read"):
            return ir.vector_splat(ir.add(b_half_idx["b128"], ir.const_i32(blk + 64 * step)), 6 if asm_mfma else 8)
        off = ir.const_i32(step * half_bytes // 4 + blk * 32 * _HALF_ROW_BYTES // 4)
        wide = ir.smem_load_vN(b_lds[buf % S], ir.add(b_half_idx["b128"], off), dtype=I32, n=4)
        narrow = ir.smem_load_vN(b_lds[buf % S], ir.add(b_half_idx["b64"], off), dtype=I32, n=2)
        live = ir.vec_concat(wide, narrow)
        return live if asm_mfma else ir.vec_concat(live, zero_pad)

    # b_path="vgpr": the 16-byte pieces of (stage t, column block c, step s)
    # start at byte ((t * N/32 + c) * 2 + s) * 1024, the 8-byte pieces at
    # b_x2 plus half that. The tile's column blocks and the step go in the
    # scalar offset, the wave's and the lane's block in the vector one.
    if b_vgpr:
        b_v_rsrc = ir.buffer_rsrc(B, ir.const_i32(KT * spec.N * _ROW_BYTES))
        b_x2 = KT * (spec.N // 32) * 2048
        b_voff = {
            (n, j): ir.add(ir.mul(lane, ir.const_i32(4 * n)), ir.mul(ir.add(ir.mul(wn, ir.const_i32(tn_w)),
                                                                            ir.const_i32(j)), ir.const_i32(512 * n)))
            for n in (4, 2) for j in range(tn_w)
        }
        b_soff0 = {
            (4, s): ir.to_sgpr_u32(ir.add(ir.mul(n0, ir.const_i32(64)), ir.const_i32(1024 * s))) for s in range(2)
        } | {
            (2, s): ir.to_sgpr_u32(ir.add(ir.mul(n0, ir.const_i32(32)), ir.const_i32(b_x2 + 512 * s)))
            for s in range(2)
        }

    def b_vgpr_thunks(kt: Value, step: int, out: list, prologue: bool = False) -> list[Callable[[], None]]:
        """b_path="vgpr": one callable per load of stage `kt`'s B fragments of
        `step` into `out` (one per column block)."""
        if spec.ablated("no_dma") and not prologue:
            return [lambda j=j: out.__setitem__(j, ir.vector_splat(ir.add(kt, ir.const_i32(64 * step + j)), 8))
                    for j in range(tn_w)]
        per_t = spec.N // 32 * 2048
        soff = {n: _Once(lambda n=n: ir.add(b_soff0[(n, step)], ir.mul(kt, ir.const_i32(per_t * n // 4))))
                for n in (4, 2)}
        wide: list[Value] = [None] * tn_w
        thunks = []
        for j in range(tn_w):
            for n in (4, 2):

                def load(j=j, n=n):
                    v = ir.buffer_load_vN(b_v_rsrc, b_voff[(n, j)], soff[n](), I32, n)
                    logged("V")
                    if n == 4:
                        wide[j] = v
                    else:
                        out[j] = ir.vec_concat(ir.vec_concat(wide[j], v), zero_pad)

                thunks.append(load)
        return thunks

    fmt = f"op_sel_hi:[0,0,0] cbsz:{MATRIX_FORMATS[spec.dtype_a]} blgp:{MATRIX_FORMATS[spec.dtype_b]}"

    def mfma_fmt(sel: tuple[int, int] | None) -> str:
        """The asm modifiers; `sel`: the A and B scale bytes (op_sel holds bit 0
        of each, op_sel_hi bit 1)."""
        if sel is None:
            return fmt
        sa, sb = sel
        return (f"op_sel:[{sa & 1},{sb & 1},0] op_sel_hi:[{sa >> 1},{sb >> 1},0] "
                f"cbsz:{MATRIX_FORMATS[spec.dtype_a]} blgp:{MATRIX_FORMATS[spec.dtype_b]}")

    pairs_all = [
        (i, j) for i in range(tm_w)
        for j in (range(tn_w - 1, -1, -1) if spec.mfma_order == "serp" and i % 2 else range(tn_w))
    ]
    grouped = spec.mfma_group < len(pairs_all)

    def mfma_block(accs, pairs, a_frag, b_frag, a_sc, b_sc, sel=None) -> list[Value]:
        """The (i, j) MFMAs of one K64 step: one asm block with AGPR
        accumulators, or with intrinsic MFMAs one `mma` per pair. `sel`: the
        scale bytes (`scale_opsel`)."""
        def pair_sel(j: int) -> tuple[int, int] | None:
            # kblock4: column j's B scale is byte j of the step's dword;
            # b_path="vgpr": byte 2s + j of the stage's.
            if sel is not None and b_vgpr:
                return sel[0], 2 * _STEP_SCALE_BYTE.index(sel[0]) + j
            return (sel[0], j) if sel is not None and b_kblock4 else sel

        if not asm_mfma:
            accs = list(accs)
            for i, j in pairs:
                k = i * tn_w + j
                if c_t:
                    attrs = None if sel is None else {"scale_sel": list(pair_sel(j))[::-1]}
                    accs[k] = ir.mma(atom_t, b_frag[j], a_frag[i], accs[k], b_sc[j], a_sc[i], attrs=attrs)
                    continue
                attrs = None if sel is None else {"scale_sel": list(pair_sel(j))}
                accs[k] = ir.mma(atom, a_frag[i], b_frag[j], accs[k], a_sc[i], b_sc[j], attrs=attrs)
            return accs
        rows = sorted({i for i, _ in pairs})
        cols = sorted({j for _, j in pairs})
        ks = [i * tn_w + j for i, j in pairs]
        n = len(pairs)
        base_a = 2 * n
        base_b = base_a + len(rows)
        base_sa = base_b + len(cols)
        base_sb = base_sa + len(rows)
        # A split block can directly follow the VALU shift of its scales,
        # which the hazard recognizer cannot see into the asm.
        lines = ["s_nop 1"] if grouped else []
        for t, (i, j) in enumerate(pairs):
            ia, jb = rows.index(i), cols.index(j)
            lines.append(
                f"v_mfma_scale_f32_32x32x64_f8f6f4 ${t}, ${base_a + ia}, ${base_b + jb}, "
                f"${n + t}, ${base_sa + ia}, ${base_sb + jb} {mfma_fmt(pair_sel(j))}"
            )
        if spec.asm_tail_nop:
            lines.append(f"s_nop {spec.asm_tail_nop}")
        constraints = ",".join(
            ["=a"] * n + [str(t) for t in range(n)] + ["v"] * (2 * (len(rows) + len(cols)))
        )
        outs = ir.inline_asm_multi(
            "\n\t".join(lines),
            constraints,
            [
                *(accs[k] for k in ks),
                *(a_frag[i] for i in rows),
                *(b_frag[j] for j in cols),
                *(a_sc[i] for i in rows),
                *(b_sc[j] for j in cols),
            ],
            result_types=[accs[k].type for k in ks],
            # The step pipeline and split blocks rely on program order: the
            # memory ops placed around a block must issue where they stand.
            sideeffect=spec.frag_pipeline == "step" or grouped,
            result_name_hint="mfma",
        )
        accs = list(accs)
        for k, v in zip(ks, outs):
            accs[k] = v
        return accs

    def mfma_cluster(accs, a_frag, b_frag, a_sc, b_sc, sel=None) -> list[Value]:
        return mfma_block(accs, pairs_all, a_frag, b_frag, a_sc, b_sc, sel)

    # Scales. Lane half h takes K32 block 2h in step 0 and 3-2h in step 1.
    # A loads one dword (4 blocks) per row per stage; B one byte per column
    # and step, or with kblock scales one dword per column like A.
    a_scale_rows = [
        ir.add(ir.add(m0, ir.mul(wm, ir.const_i32(spec.wave_tile_m))), ir.add(r32, ir.const_i32(32 * i)))
        for i in range(tm_w)
    ]
    b_scale_cols = [
        ir.add(ir.add(n0, ir.mul(wn, ir.const_i32(spec.wave_tile_n))), ir.add(r32, ir.const_i32(32 * j)))
        for j in range(tn_w)
    ]
    blk_of_step = [ir.mul(half, ir.const_i32(2)), ir.sub(ir.const_i32(3), ir.mul(half, ir.const_i32(2)))]
    a_shift = [ir.mul(b, ir.const_i32(8)) for b in blk_of_step]
    kblk_rows = ir.const_i32(KT)  # A_scale dwords per row

    as_lds_on = spec.a_lds_scales
    a_grouped = spec.a_scale_fetch != "stage" and not as_lds_on
    # kblock4: the lane's first dword in a stage's block of B scales.
    b4_lane = _Once(lambda: ir.add(
        ir.mul(ir.add(ir.div(n0, ir.const_i32(spec.wave_tile_n)), wn), ir.const_i32(2 * _WAVE)),
        ir.mul(lane, ir.const_i32(2)),
    ))
    # b_path="vgpr": the lane's dword in a stage's N dwords of B scales.
    bv_lane = _Once(lambda: ir.add(ir.add(n0, ir.mul(wn, ir.const_i32(spec.wave_tile_n))), lane))

    def scale_thunks(kt: Value) -> list[Callable[[], Value]]:
        """One callable per scale load of a stage, in `load_scales` order.

        With grouped A scales (or A scales from LDS) only B's loads are per
        stage (`load_a_group`). b_scale_pairs: the B load of stage kt's pair.
        """
        n_a = 0 if a_grouped or as_lds_on else tm_w
        if spec.ablated("no_scales"):
            if b_vgpr:
                return [lambda k=k: ir.add(kt, ir.const_i32(127 + k)) for k in range(n_a + 1)]
            if b_kblock4:
                return [lambda k=k: ir.add(kt, ir.const_i32(127 + k)) for k in range(n_a)] + [
                    lambda: ir.vector_splat(ir.add(kt, ir.const_i32(127 + n_a)), 4 if b_pairs else 2)
                ]
            n = n_a + (1 if b_kblock_scales else 2) * tn_w
            return [lambda k=k: ir.add(kt, ir.const_i32(127 + k)) for k in range(n)]

        def kblock_dword(ptr: Value, rows: int, r: Value) -> Callable[[], Value]:
            return lambda: logged("s", ir.global_load(ptr, ir.add(ir.mul(kt, ir.const_i32(rows)), r), I32, align=4))

        if a_grouped or as_lds_on:
            out = []
        elif a_kblock_scales:
            out = [kblock_dword(AS, spec.M, row) for row in a_scale_rows]
        else:
            out = [
                lambda row=row: logged("s", ir.global_load(AS, ir.add(ir.mul(row, kblk_rows), kt), I32, align=4))
                for row in a_scale_rows
            ]
        if b_vgpr:
            return out + [lambda: logged("s", ir.global_load(BS, ir.add(ir.mul(kt, ir.const_i32(spec.N)), bv_lane()),
                                                             I32, align=4))]
        if b_kblock_scales:
            return out + [kblock_dword(BS, spec.N, col) for col in b_scale_cols]
        if b_kblock4:
            per_stage = spec.N // spec.wave_tile_n * 2 * _WAVE
            if b_pairs:
                return out + [lambda: logged("s", ir.global_load_vN(
                    BS, ir.add(ir.mul(ir.lshr(kt, ir.const_i32(1)), ir.const_i32(2 * per_stage)),
                               ir.mul(b4_lane(), ir.const_i32(2))), I32, 4, align=16))]
            return out + [lambda: logged("s", ir.global_load_vN(
                BS, ir.add(ir.mul(kt, ir.const_i32(per_stage)), b4_lane()), I32, 2, align=8))]
        kblk0 = _Once(lambda: ir.mul(kt, ir.const_i32(4)))
        for col in b_scale_cols:
            for blk in blk_of_step:

                def load(col=col, blk=blk):
                    g = ir.add(kblk0(), blk)
                    byte = logged("s", ir.global_load(BS, ir.add(ir.mul(g, ir.const_i32(spec.N)), col), I8, align=1))
                    return ir.zext(byte, I32)

                out.append(load)
        return out

    def load_scales(kt: Value) -> list[Value]:
        return [load() for load in scale_thunks(kt)]

    def a_group_thunks(start: Value) -> list[Callable[[], Value]]:
        """Per A row of the lane, one callable loading the scale dwords of one
        loop iteration's stages from `start`."""
        if spec.ablated("no_scales"):
            return [lambda i=i: ir.vector_splat(ir.add(start, ir.const_i32(127 + i)), U) for i in range(tm_w)]
        return [
            lambda row=row: logged("G", ir.global_load_vN(AS, ir.add(ir.mul(row, kblk_rows), start), I32, U, align=4))
            for row in a_scale_rows
        ]

    def load_a_group(start: Value) -> list[Value]:
        return [load() for load in a_group_thunks(start)]

    # scale_opsel: lane half 1's scale dwords rotated by 16 bits (a v_perm
    # with a per-lane selector, the identity for half 0), so step 0 takes
    # byte 0 and step 1 byte 3 of every scale dword for both halves. B's
    # row-major scales are one byte per column and step: byte 0, unrotated.
    opsel = spec.scale_opsel
    a_iter2 = spec.a_scale_fetch == "iter2" and opsel and a_line and spec.frag_pipeline == "step" and a_grouped
    sel_bytes = _T10_SCALE_BYTE if t10 else _STEP_SCALE_BYTE
    step_sel = [(b, b if b_kblock_scales else 0) for b in sel_bytes] if opsel else [None, None]
    rot_sel = _Once(lambda: ir.xor(ir.const_i32(_PERM_IDENTITY), ir.mul(half, ir.const_i32(_PERM_ROT16_XOR))))

    def rotate(x: Value) -> Value:
        return ir.perm_b32(x, x, rot_sel())

    # Per-stage scale values the rotation applies to (`load_scales` order).
    n_stage_a = 0 if a_grouped or as_lds_on else tm_w
    stage_rotated = list(range(n_stage_a)) + (list(range(n_stage_a, n_stage_a + tn_w)) if b_kblock_scales else [])

    def rotate_stage_scales(vals: list[Value]) -> list[Value]:
        return [rotate(v) if k in stage_rotated else v for k, v in enumerate(vals)]

    def group_scalars(group: Sequence[Value]) -> list[Value]:
        """scale_opsel: an A scale group as its rotated dwords, row-major."""
        return [rotate(ir.vec_extract(g, e)) for g in group for e in range(U)]

    # ---- a_scale_path="lds2": epoch E (stages 8E..8E+7) of the tile's A
    # scales sits in LDS slot E % 2 as [row block][half][32 rows][16 B]. Wave
    # w copies row blocks w + n_waves * k, one 16-byte DMA each: lane (h, r)
    # half h of row r's 32 bytes. A read of row r's dword has a 16-byte lane
    # stride, and both lane halves read the same address.
    n_waves = T // _WAVE
    as_slot_bytes = spec.block_m * _EPOCH_BYTES
    # t10 with b32 reads: halfwords, each lane half reading its own.
    as_u16 = t10 and spec.as_read == "b32"
    if as_lds_on:
        as_lds = ir.smem_alloc(F16, [2 * as_slot_bytes // 2], name_hint="a_scales") if as_u16 else (
            ir.smem_alloc(I32, [2 * as_slot_bytes // 4], name_hint="a_scales"))
        as_wave = ir.smem_ptr_add(ir.smem_addr_of(as_lds), wave_lds)
        as_rsrc = ir.buffer_rsrc(AS, ir.const_i32(spec.M * KT * 4))
        as_soff0 = ir.mul(m0, ir.const_i32(KT * 4))
        as_voff = [
            ir.add(ir.mul(ir.add(ir.mul(ir.add(wave, ir.const_i32(n_waves * k)), c32), r32), ir.const_i32(KT * 4)),
                   ir.mul(half, ir.const_i32(16)))
            for k in range(spec.block_m // 32 // n_waves)
        ]
        as_read_base = ir.add(ir.mul(wm, ir.const_i32(tm_w * 256)), ir.mul(r32, ir.const_i32(4)))
        c_epoch_last = ir.const_i32((KT - 1) // _EPOCH_STAGES)

    def as_dma_thunks(epoch: Value) -> list[Callable[[], None]]:
        """lds2: one callable per scale DMA of `epoch` (clamped to the last)
        into slot `epoch % 2`."""
        if spec.ablated("no_scales"):
            return []
        src = _Once(lambda: ir.select(ir.cmp_lt(epoch, c_epoch_last), epoch, c_epoch_last))
        soff = _Once(lambda: ir.add(as_soff0, ir.mul(src(), ir.const_i32(_EPOCH_BYTES))))
        slot = _Once(lambda: ir.mul(ir.land(epoch, ir.const_i32(1)), ir.const_i32(as_slot_bytes)))
        out = []
        for k, voff in enumerate(as_voff):

            def issue(k=k, voff=voff):
                off = ir.to_sgpr_u32(ir.add(slot(), ir.const_i32(k * n_waves * _WAVE * 16)))
                ir.async_buffer_load_lds_addr(as_rsrc, ir.smem_ptr_add(as_wave, ir.zext(off, I64)), voff, soff(), 4)
                logged("AS")

            out.append(issue)
        return out

    # as_read="b128x4": per row block, the dwords of the epoch half the last
    # read covered; loop-carried.
    as_b128 = as_lds_on and spec.as_read == "b128x4" and not spec.ablated("no_scales")
    as_cache: list = [None] * tm_w

    def as_read_thunks(epoch: Value, at: int, out: list) -> list[Callable[[], None]]:
        """lds2: one callable per A row block of the lane, reading its scale
        dword of stage `at` (0..7) of `epoch` into `out`. b128x4: stages
        with `at % 4` > 0 take it from the read of stage `at` - `at % 4`."""
        slot = _Once(lambda: ir.mul(ir.land(epoch, ir.const_i32(1)), ir.const_i32(as_slot_bytes // 4)))

        def read(i: int) -> None:
            if spec.ablated("no_scales"):
                out[i] = ir.add(epoch, ir.const_i32(127 + i))
                return
            if as_b128:
                if at % 4 == 0:
                    off = ir.add(slot(), ir.const_i32(i * 256 + at // 4 * 128))
                    as_cache[i] = ir.smem_load_vN(as_lds, ir.add(as_read_base, off), dtype=I32, n=4)
                out[i] = ir.vec_extract(as_cache[i], at % 4)
                return
            off = ir.add(slot(), ir.const_i32(i * 256 + at // 4 * 128 + at % 4))
            if as_u16:
                idx = ir.add(ir.mul(ir.add(as_read_base, off), ir.const_i32(2)), half)
                # The direct backend holds 16-bit values as f16.
                out[i] = ir.vec_extract(ir.smem_load_vN(as_lds, idx, dtype=F16, n=1), 0)
                return
            out[i] = ir.vec_extract(ir.smem_load_vN(as_lds, ir.add(as_read_base, off), dtype=I32, n=1), 0)

        return [lambda i=i: read(i) for i in range(tm_w)]

    def as_fix(x: Value) -> Value:
        """scale_opsel: an `as_read_thunks` value as the MFMA takes it; t10
        widens lane half h's halfword instead of rotating. Applied after the
        last MFMA block, not at the read, so no wave waits for the read."""
        if not t10:
            return rotate(x)
        return ir.zext(ir.bitcast(x, I16), I32) if as_u16 else ir.lshr(x, ir.mul(half, ir.const_i32(16)))

    def b_scale(scales: Sequence[Value], j: int, step: int, hi: int = 0) -> Value:
        if b_vgpr:
            byte = 2 * step + j
            return scales[tm_w] if opsel or not byte else ir.lshr(scales[tm_w], ir.const_i32(8 * byte))
        if b_kblock4:
            x = ir.vec_extract(scales[tm_w], 2 * hi + step)
            return x if opsel else ir.lshr(x, ir.const_i32(8 * j)) if j else x
        if b_kblock_scales:
            return scales[tm_w + j] if opsel else ir.lshr(scales[tm_w + j], a_shift[step])
        return scales[tm_w + 2 * j + step]

    def scale_ops(scales: Sequence[Value], step: int, hi: int = 0) -> tuple[list[Value], list[Value]]:
        """`hi`: b_scale_pairs' odd stage, whose B dwords are the pair's upper two."""
        a_sc = [scales[i] if opsel else ir.lshr(scales[i], a_shift[step]) for i in range(tm_w)]
        return a_sc, [b_scale(scales, j, step, hi) for j in range(tn_w)]

    def read_thunks(buf: int, step: int, out: list[Value], b_buf: int | None = None) -> list[Callable[[], None]]:
        """One callable per fragment of a step into `out` (A, then B slots);
        B's of slot `b_buf` when given, else of `buf`.

        A and B alternate, so no two narrow reads of one buffer are adjacent
        for the load-store optimizer to merge into a read2.
        """
        b_buf = buf if b_buf is None else b_buf
        thunks = []
        for t in range(max(tm_w, tn_w)):
            if t < tm_w:
                thunks.append(lambda t=t: out.__setitem__(t, a_fragment(buf, step, t)))
            if t < tn_w and not b_vgpr:
                thunks.append(lambda t=t: out.__setitem__(tm_w + t, b_fragment(b_buf, step, t)))
        return thunks

    def sink(accs, frags, a_sc, b_sc) -> list[Value]:
        """`no_mfma`: keep the fragments and scales live without matrix work."""
        acc0 = accs[0]
        for v in [ir.vec_extract(f, 0) for f in frags] + list(a_sc) + list(b_sc):
            acc0 = ir.vec_insert(acc0, ir.fadd(ir.vec_extract(acc0, 0), ir.bitcast(v, F32)), 0)
        return [acc0, *accs[1:]]

    def step_mfma(accs, frags, a_sc, b_sc, fillers: Sequence[Callable[[], None]], sel=None,
                  slots: dict[int, list[Callable[[], None]]] | None = None,
                  head: dict[int, list[Callable[[], None]]] | None = None,
                  after0: Sequence[Callable[[], None]] = ()) -> list[Value]:
        """One K64 step in blocks of `mfma_group`, the fillers spread between
        them; `slots[b]` are placed after block b's share of the fillers,
        `head[b]` ahead of it, `after0` right after the first block (a
        barrier group the previous step left, `sync_place`)."""
        slots, head = slots or {}, head or {}
        if spec.ablated("no_mfma"):
            for fill in [*after0, *fillers, *(f for s in (head, slots) for b in sorted(s) for f in s[b])]:
                fill()
            return sink(accs, frags, a_sc, b_sc)
        g = spec.mfma_group
        blocks = [pairs_all[x : x + g] for x in range(0, len(pairs_all), g)]
        for b, pairs in enumerate(blocks):
            accs = mfma_block(accs, pairs, frags[:tm_w], frags[tm_w:], a_sc, b_sc, sel)
            for fill in after0 if b == 0 else ():
                fill()
            for fill in head.get(b, ()):
                fill()
            for fill in fillers[len(fillers) * b // len(blocks) : len(fillers) * (b + 1) // len(blocks)]:
                fill()
            for fill in slots.get(b, ()):
                fill()
            if not asm_mfma:
                # Intrinsic MFMAs carry no side effect to hold them in place:
                # fence each block with its fillers from the next.
                ir.sched_barrier(0)
        return accs

    def interleave(n_mfma: int, n_ds: int, n_vmem: int) -> None:
        ds_left, vm_left = n_ds, n_vmem
        for k in range(n_mfma):
            ir.sched_group_barrier(_SG_MFMA, 1, 0)
            take = -(-ds_left // (n_mfma - k)) if ds_left else 0
            if take:
                ir.sched_group_barrier(_SG_DS_READ, take, 0)
                ds_left -= take
            vt = -(-vm_left // (n_mfma - k)) if vm_left else 0
            if vt:
                ir.sched_group_barrier(_SG_VMEM_READ, vt, 0)
                vm_left -= vt

    split = grouped and spec.fill_split != "none"
    # The stage of a loop iteration that loads the next iteration's A scales.
    group_phase = 1 if split else 0

    def stage_dma(kt: Value, buf: int, n_it: Value | None) -> list[Callable[[], None]]:
        """The DMA a stage issues: the next ring slot's stage of A and B, or
        with the line ring B's next stage, then its line refill."""
        nxt = clamp_stage(ir.add(kt, ir.const_i32(S - 1)))
        if not a_line:
            return dma_thunks(nxt, (buf + S - 1) % S)
        return dma_thunks(nxt, (buf + S - 1) % S, operands="b") + line_refill(n_it, buf)

    def stage_dma_split(kt: Value, buf: int, n_it: Value | None) -> tuple[list, list]:
        """`stage_dma` as (B's DMA, A's DMA); with the line ring A's is the refill."""
        nxt = clamp_stage(ir.add(kt, ir.const_i32(S - 1)))
        slot = (buf + S - 1) % S
        b_dma = dma_thunks(nxt, slot, operands="b")
        return b_dma, line_refill(n_it, buf) if a_line else dma_thunks(nxt, slot, operands="a")

    def group_rotations(src: Callable[[], Sequence[Value]] | _GroupDwords, out: list,
                        last_block: bool) -> dict[int, list]:
        """scale_opsel, the iteration's last stage: the next A scale group's
        dwords rotated into `out` (row-major) as step-1 fillers. Element e of
        row i goes after block e of the row; the last element after the row's
        last MFMA, which reads the dword it replaces. `last_block`: all after
        the last block (the stage loads the group itself). `src` returns the
        group's vectors, or gives it dword by dword."""
        elem = src.get if isinstance(src, _GroupDwords) else lambda i, e: ir.vec_extract(src()[i], e)
        g, nb = spec.mfma_group, len(pairs_all) // spec.mfma_group
        slots: dict[int, list] = {}
        for i in range(tm_w):
            for e in range(U):
                pos = i * tn_w + (tn_w - 1 if e == U - 1 else min(e, tn_w - 1))
                block = nb - 1 if last_block else pos // g

                def rot(i=i, e=e):
                    out[i * U + e] = rotate(elem(i, e))

                slots.setdefault(block, []).append(rot)
        return slots

    def stage(kt: Value, buf: int, accs: Sequence[Value], scales: Sequence[Value], a_vals=None, group_start=None,
              n_it=None, rotate_group=None):
        """One K128 stage. With grouped A scales, `a_vals` are this stage's A
        scale dwords and `group_start`, when set, starts the next group's load;
        returns (accs, next stage's per-stage scales, next group or None).
        With the line ring `buf` is the stage's phase in its loop iteration
        `n_it`, and B's slot is `buf % stages`.
        `rotate_group` (scale_opsel): the next A scale group, which this stage
        rotates into its loop-carried dwords (True: the group it loads)."""
        # The barrier intrinsic does not touch memory: pin the previous stage's
        # LDS reads above it and this stage's below it, or a read can sink past
        # the barrier while another wave's DMA overwrites its buffer.
        if spec.backend == "llvm":
            ir.inline_asm("", "~{memory}", [])
        if not a_line:
            ir.s_waitcnt(vmcnt=(S - 2) * _vmem_per_stage(spec))
        elif spec.backend == "direct":
            landed_wait()
        else:
            # LLVM may reorder B's DMA with the refill, which another scope
            # holds; it also waits for the refill before any read of the ring.
            ir.s_waitcnt(vmcnt=0)
        ir.s_barrier_bare(fence="compiler")
        nxt_group = load_a_group(group_start) if group_start is not None and not split else None
        if a_vals is not None:
            scales = [*a_vals, *scales]
        if grouped:
            # Step 0 waits for its own reads; the step-1 reads, the DMA and
            # the next stage's scales issue between step 0's MFMA blocks
            # (with `fill_split`, A's DMA and the A scale group between
            # step 1's).
            if spec.setprio:
                ir.s_setprio(1)
            F, G, nxt = [None] * (tm_w + tn_w), [None] * (tm_w + tn_w), []
            for read in read_thunks(buf, 0, F):
                read()
            reads = read_thunks(buf, 1, G)
            b_dma, a_dma = stage_dma_split(kt, buf, n_it) if split else (stage_dma(kt, buf, n_it), [])
            fillers = _round_robin(
                reads,
                b_dma,
                [lambda t=t: nxt.append(t()) for t in scale_thunks(clamp_stage(ir.add(kt, ir.const_i32(1))))],
            )
            group_vals: list[Value] = []
            if split and group_start is not None:
                a_dma = a_dma + [lambda t=t: group_vals.append(t()) for t in a_group_thunks(group_start)]
            slots: dict[int, list] = {}
            rotated: list[Value] = []
            if opsel:
                if rotate_group is not None:
                    rotated = [None] * (tm_w * U)
                    own = rotate_group is True
                    src = (lambda: group_vals or nxt_group) if own else (lambda: rotate_group)
                    slots = group_rotations(src, rotated, own)
                # The next stage's scales load during step 0; rotate them last.
                last = len(pairs_all) // spec.mfma_group - 1
                slots.setdefault(last, []).extend(
                    lambda k=k: nxt.__setitem__(k, rotate(nxt[k])) for k in stage_rotated
                )
            accs = step_mfma(accs, F, *scale_ops(scales, 0), fillers, step_sel[0])
            accs = step_mfma(accs, G, *scale_ops(scales, 1), a_dma, step_sel[1], slots)
            if spec.setprio:
                ir.s_setprio(0)
            return accs, nxt, rotated or group_vals or nxt_group
        if spec.sched.startswith("iglp"):
            ir.iglp_opt(int(spec.sched[-1]))
        nxt_scales = load_scales(clamp_stage(ir.add(kt, ir.const_i32(1))))
        dma = stage_dma(kt, buf, n_it)
        for issue in dma:
            issue()
        if spec.setprio:
            ir.s_setprio(1)
        accs = list(accs)
        for step in range(2):
            a_frag = [a_fragment(buf, step, i) for i in range(tm_w)]
            b_frag = [b_fragment(buf, step, j) for j in range(tn_w)]
            a_sc, b_sc = scale_ops(scales, step)
            if spec.ablated("no_mfma"):
                # Keep the fragments and scales live without any matrix work.
                sink = [ir.vec_extract(f, 0) for f in a_frag + b_frag] + a_sc + b_sc
                acc0 = accs[0]
                for v in sink:
                    acc0 = ir.vec_insert(acc0, ir.fadd(ir.vec_extract(acc0, 0), ir.bitcast(v, F32)), 0)
                accs[0] = acc0
                continue
            accs = mfma_cluster(accs, a_frag, b_frag, a_sc, b_sc, step_sel[step])
        if spec.setprio:
            ir.s_setprio(0)
        if spec.sched == "interleave":
            n_mfma = 2 * tm_w * tn_w
            n_dma = len(dma) if a_line else dma_ops
            interleave(n_mfma, 4 * (tm_w + tn_w), n_dma + len(nxt_scales) + len(nxt_group or ()))
        if opsel:
            nxt_scales = rotate_stage_scales(nxt_scales)
            if rotate_group is not None:
                return accs, nxt_scales, group_scalars(nxt_group if rotate_group is True else rotate_group)
        return accs, nxt_scales, nxt_group

    # ---- frag_pipeline=step with the line ring ------------------------------
    # Stage s: step 0 runs on fragments read during the previous stage's step
    # 1, with stage s's step-1 reads, the next stage's scale loads, the line
    # refill and (in phase 1) the next A scale group between its blocks. Then
    # the barrier Bar(s+1): B(s+1) has landed and every wave is done reading
    # B's slot s%2 and stage s's lines. Step 1 runs on registers already read,
    # with stage s+1's step-0 reads and B(s+2)'s DMA into slot s%2 between its
    # blocks. The last stage reads one stage past K, which the clamps keep in
    # bounds, and discards it.
    step_group_phase = 1
    vmem_order = spec.step_vmem_order
    n_blocks = len(pairs_all) // min(spec.mfma_group, len(pairs_all))

    def one_per_block(thunks: Sequence[Callable[[], None]], every: int, slots: dict[int, list],
                      start: int = 0, last: int | None = None) -> dict[int, list]:
        """`thunks` into `slots`, one after every `every`-th MFMA block from
        block `start` (the rest after block `last`, by default the last), ahead
        of what the block had."""
        last = n_blocks - 1 if last is None else last
        out = {b: list(v) for b, v in slots.items()}
        for k, thunk in reversed(list(enumerate(thunks))):
            out.setdefault(min(start + k * every, last), []).insert(0, thunk)
        return out

    placed = spec.dma_placed
    sync = spec.sync_place if spec.sync_placed else "none"
    # sync_place: a step's waits after its block n - 2 (with "split_lgkm"
    # lgkmcnt only), its barrier (and the waits left) after the next step's
    # first block; reads stay out of a step's last two gaps.
    sync_early = {"split": 2, "split_sink": 2, "split_lgkm": 1}.get(sync, 0)
    sync_sink = sync in ("sink", "split_sink")
    read_last = max(n_blocks - 3, 0) if sync_early else n_blocks - 1
    # The barrier group a step left for the next step's first block.
    pending_sync: list[list[Callable[[], None]]] = [[]]

    def sync_group(waits: Sequence[Callable[[], None]], bar: Callable[[], None],
                   slots: dict[int, list]) -> list[Callable[[], None]]:
        """Place a barrier group: its first `sync_early` waits at the end of
        block n - 2's slot; returns the rest with the barrier, for the end of
        the step or (`sync_sink`) the next step's first block."""
        for w in waits[:sync_early]:
            slots.setdefault(max(n_blocks - 2, 0), []).append(w)
        return [*waits[sync_early:], bar]

    def gaps(start: int, every: int) -> list[int]:
        return list(range(max(start, 0), n_blocks, every)) or [n_blocks - 1]

    def place_fixed(thunks: Sequence[Callable[[], None]], at: Sequence[int], slots: dict[int, list]) -> dict[int, list]:
        """`thunks` after MFMA blocks `at` (the last of them takes the rest),
        behind what the blocks had."""
        for k, thunk in enumerate(thunks):
            slots.setdefault(at[min(k, len(at) - 1)], []).append(thunk)
        return slots

    def place_dmas(refill: Sequence[Callable[[], None]], b_half: Sequence[Callable[[], None]],
                   slots: dict[int, list]) -> dict[int, list]:
        """dma_place: a step's part of a refill and its B half into the odd
        gaps; B's alone one per four blocks. dense / pair: B's half then the
        refill, one / two per gap from gap 1."""
        packed = {"dense_bfirst": 1, "pair_bfirst": 2}.get(spec.dma_place)
        if packed:
            return place_fixed([*b_half, *refill], [g for g in gaps(1, 1) for _ in range(packed)], slots)
        if not refill:
            return place_fixed(b_half, gaps(1, 4), slots)
        first, second = (b_half, refill) if spec.dma_place == "even_bfirst" else (refill, b_half)
        odd = gaps(1, 2)
        place_fixed(first, odd, slots)
        return place_fixed(second, odd[len(first) :] or odd[-1:], slots)

    def step_line_stage(kt: Value, u: int, accs, F, scales, a_vals, group_start, n_it, rotate_group=None,
                        final=False, a_lds_next=None, a_dma_epoch=None, bv=None, t_static=None):
        """One stage of the line ring's step pipeline; returns (accs, the next
        stage's step-0 fragments, its per-stage scales, next group or None,
        b_path="vgpr" state or None).
        `final`: K's last stage where it is emitted once (a tail, the peeled
        or a single-trip iteration): it loads no next-stage scales, which the
        compiler would drop as dead while the barrier's wait counts them, and
        passes its own on, unread.
        lds2: `a_lds_next` = (epoch, stage in it) of the next stage, whose A
        scale dwords step 1 reads (returned in place of the group; at an
        epoch's first stage Bar(s+1) also waits for its DMA), and
        `a_dma_epoch` the epoch whose scale DMA step 0 issues.
        b_path="vgpr": `bv` = (B's step-1 fragments of this stage, B's step-0
        fragments of the next); step 0 loads the next stage's step-1 ones and
        step 1 the step-0 ones of the stage after, which the state returned
        holds. `t_static`: the stage's index where known, so loads past K,
        which nothing reads and the compiler would drop, are not emitted."""
        carried_scales = list(scales)
        if a_vals is not None:
            scales = [*a_vals, *scales]
        if spec.setprio:
            ir.s_setprio(1)
        G, nxt, group_vals = [None] * (tm_w + tn_w), [], []
        # b_split: B(s+2)'s step-0 half goes into its sub-slot first, which
        # the previous stage's end barrier freed.
        b_kt = clamp_stage(ir.add(kt, ir.const_i32(2))) if b_split else None
        h0 = b_half_thunks(b_kt, u % S, 0) if b_split else []
        # b_scale_pairs: an even stage passes its pair on to the odd one.
        hi = u % 2 if b_pairs else 0
        sc = [] if final or (b_pairs and not hi) else [
            lambda t=t: nxt.append(t()) for t in scale_thunks(clamp_stage(ir.add(kt, ir.const_i32(1))))
        ]
        if b_pairs and not hi:
            nxt = list(carried_scales)
        refill = line_refill(n_it, u, part=1 if placed else None)
        vmem = h0 + sc + refill
        group = [] if group_start is None else [
            lambda t=t: group_vals.append(t()) for t in a_group_thunks(group_start)
        ]
        if a_dma_epoch is not None:
            group = group + as_dma_thunks(a_dma_epoch)
        late_group = group if vmem_order == "group_late" else []
        vmem += [] if late_group else group
        bg_next: list[Value] = [None] * tn_w
        bf_next2: list[Value] = [None] * tn_w
        if b_vgpr:
            G[tm_w:] = bv[0]
            if not final and (t_static is None or t_static + 1 < KT):
                vmem += b_vgpr_thunks(clamp_stage(ir.add(kt, ir.const_i32(1))), 1, bg_next)
        reads = read_thunks(u, 1, G)
        e_prev, pending_sync[0] = pending_sync[0], []

        def bar_lgkm() -> None:
            ir.s_waitcnt(lgkmcnt=0)
            if spec.backend == "llvm":
                ir.inline_asm("", "~{memory}", [])

        # Bar(s+1): step 1 reads stage s+1's step-0 fragments. B(s+1) (from
        # the previous stage's step 1) and every older line must land; with
        # b_split B(s+1)'s step-0 half and the refill of stage s-1, both from
        # its step 0.
        scale_dma = ("AS",) if a_lds_next is not None and a_lds_next[1] == 0 else ()

        def bar_vm() -> None:
            if spec.backend != "direct":
                ir.s_waitcnt(vmcnt=0)
            elif spec.ablated("no_dma"):
                ir.s_waitcnt(vmcnt=_VMCNT_MAX)
            elif b_split:
                landed_wait(("BF", "A", *scale_dma), before="|0")
            elif b_vgpr:
                # Stage s+1's lines were refilled by stage s-1's step 0 at the latest.
                landed_wait(("A", *scale_dma), before="|0")
            else:
                landed_wait(("B", *scale_dma))

        def bar_sync() -> None:
            logged("|0")
            ir.s_barrier_bare(fence="compiler")

        bar_rest = [bar_lgkm, bar_vm, bar_sync]
        if placed:
            slots0 = place_dmas(refill, h0, one_per_block(reads, 2, {}, last=read_last))
            place_fixed(sc, gaps(2, 2) if refill else gaps(3, 4), slots0)
            place_fixed(group, gaps(n_blocks - 1 if refill else n_blocks - 3, 2), slots0)
            bar_rest = sync_group([bar_lgkm, bar_vm], bar_sync, slots0)
            accs = step_mfma(accs, F, *scale_ops(scales, 0, hi), [], step_sel[0], slots0, after0=e_prev)
        elif vmem_order == "refill_early":
            accs = step_mfma(accs, F, *scale_ops(scales, 0, hi), reads, step_sel[0], one_per_block(vmem, 1, {}))
        else:
            accs = step_mfma(accs, F, *scale_ops(scales, 0, hi), _round_robin(reads, vmem), step_sel[0])
        bar_next: list[Callable[[], None]] = []
        if sync_sink:
            bar_next = bar_rest
        else:
            for part in bar_rest:
                part()
        F_next = [None] * (tm_w + tn_w)
        if b_vgpr:
            F_next[tm_w:] = bv[1]
            b_dma = []
            if not final and (t_static is None or t_static + 2 < KT):
                b_dma = b_vgpr_thunks(clamp_stage(ir.add(kt, ir.const_i32(2))), 0, bf_next2)
        elif b_split:
            b_dma = b_half_thunks(b_kt, u % S, 1)
        else:
            b_dma = dma_thunks(clamp_stage(ir.add(kt, ir.const_i32(2))), u % S, operands="b")
        slots: dict[int, list] = {}
        rotated: list[Value] = []
        if opsel:
            if rotate_group is not None:
                rotated = [None] * (tm_w * U)
                own = rotate_group is True
                src = rotate_group if isinstance(rotate_group, _GroupDwords) else (
                    (lambda: group_vals) if own else (lambda: rotate_group))
                slots = group_rotations(src, rotated, own)
            last = len(pairs_all) // spec.mfma_group - 1
            if not final:
                slots.setdefault(last, []).extend(
                    lambda k=k: nxt.__setitem__(k, rotate(nxt[k])) for k in stage_rotated
                )
        frags = read_thunks((u + 1) % U, 0, F_next)
        reads, a_reads = frags, []
        a_next: list[Value] = []
        if a_lds_next is not None:
            # The next stage's A scale dwords first, rotated after the last block.
            a_next = [None] * tm_w
            a_reads = as_read_thunks(*a_lds_next, a_next)
            reads = a_reads + reads
            if opsel:
                slots.setdefault(n_blocks - 1, []).extend(
                    lambda i=i: a_next.__setitem__(i, as_fix(a_next[i])) for i in range(tm_w))

        def paired(k: int) -> Callable[[], None]:
            def read() -> None:
                for thunk in (*a_reads[k : k + 1], frags[k]):
                    thunk()

            return read

        pairs1 = [paired(k) for k in range(len(frags))]
        if placed:
            # The next stage's refill, its first half; its slot is free from Bar(s+1).
            nu = u + 1
            ra = [] if final else line_refill(n_it if nu < U else ir.add(n_it, ir.const_i32(1)), nu % U, part=0)
            if vmem_order == "reads_first":
                spread = _round_robin(reads, b_dma)
                n_rd = read_last + 1
                at = {id(t): b for b in range(n_rd)
                      for t in spread[len(spread) * b // n_rd : len(spread) * (b + 1) // n_rd]}
                held: dict[int, list] = {}
                for t in reads:
                    held.setdefault(at[id(t)], []).append(t)
            else:
                held = one_per_block(pairs1, 1, {}, start=0 if vmem_order == "frag_first" else _FRAG_MID_START,
                                     last=read_last)
            held = place_dmas(ra, b_dma, held)
            for b, extra in slots.items():
                held.setdefault(b, []).extend(extra)
            fill1, slots = [], held
        elif vmem_order == "b_early":
            fill1, slots = reads, one_per_block(b_dma, 2, slots)
        elif vmem_order in ("frag_first", "frag_mid"):
            if vmem_order == "frag_first":
                fill1, slots = [], one_per_block(pairs1 + b_dma + late_group, 1, slots)
            else:
                fill1 = []
                slots = one_per_block(pairs1, 1, one_per_block(b_dma + late_group, 1, slots), start=_FRAG_MID_START)
        else:
            fill1 = _round_robin(reads, b_dma + late_group)
        e_rest: list[Callable[[], None]] = []
        if b_split and not final:
            # Deferred past this step's end, the wait counts to the mark before.
            nth = int(sync_sink and sync_early < 2)
            e_rest = sync_group([end_lgkm, lambda: end_vm(("BG",), nth)], end_sync, slots)
        accs = step_mfma(accs, G, *scale_ops(scales, 1, hi), fill1, step_sel[1], slots, after0=bar_next)
        if spec.setprio:
            ir.s_setprio(0)
        if sync_sink:
            pending_sync[0] = e_rest
        else:
            for part in e_rest:
                part()
        logged("|1")
        b_state = (bg_next, bf_next2) if b_vgpr else None
        return accs, F_next, carried_scales if final else nxt, a_next or rotated or group_vals or None, b_state

    def end_lgkm() -> None:
        ir.s_waitcnt(lgkmcnt=0)
        if spec.backend == "llvm":
            ir.inline_asm("", "~{memory}", [])

    def end_vm(need: Sequence[str], nth: int = 0) -> None:
        if spec.backend != "direct":
            ir.s_waitcnt(vmcnt=0)
        elif spec.ablated("no_dma"):
            ir.s_waitcnt(vmcnt=_VMCNT_MAX)
        elif need:
            landed_wait(need, before="|1", nth=nth)

    def end_sync() -> None:
        ir.s_barrier_bare(fence="compiler")

    def end_barrier(need: Sequence[str] = ("BG",)) -> None:
        """b_split, E(s+1) after stage s: every wave has read stage s+1's
        step-0 fragments (the next step-0 DMA overwrites their sub-slot), and
        B(s+1)'s step-1 half, from the previous stage's step 1, has landed for
        the reads in stage s+1's step 0."""
        end_lgkm()
        end_vm(need)
        end_sync()

    def step_line2r_stage(kt: Value, u: int, p: int, accs, F, scales, a_vals, group_start, n_it, rotate_group=None,
                          final=False, a_lds_next=None, a_dma_epoch=None):
        """a_ring="line2r": stage s at place `p` of the 24-stage slot pattern
        (its window's slot parity p // 4 % 2, B's slot p % 3); returns as
        `step_line_stage`. Step 0 runs on the fragments read in stage s-1's
        step 1, with stage s's step-1 reads and the next stage's scale loads
        between its blocks, and B(s+2), one DMA per two blocks from the first,
        into the slot B(s-1) left at Bar(s); a window's first stage copies its
        remnant. Then Bar(s+1), the stage's only barrier: every wave is done
        with what stage s read, and B(s+1) and every refill so far have landed.
        Step 1 runs on registers, with the refill of the slot stage s freed
        (one DMA per block from the first, ahead of the rest) and stage s+1's
        step-0 reads."""
        q, parity = p % (_LINE_PERIOD * _L2R_SLOTS), p // _LINE_PERIOD % _L2R_SLOTS
        carried_scales = list(scales)
        if a_vals is not None:
            scales = [*a_vals, *scales]
        if spec.setprio:
            ir.s_setprio(1)
        G, nxt, group_vals = [None] * (tm_w + tn_w), [], []
        b_slot = (p + 2) % _L2R_B_SLOTS
        head = one_per_block(dma_thunks(clamp_stage(ir.add(kt, ir.const_i32(2))), b_slot, operands="b",
                                        b_tag=f"B{b_slot}"), 2, {})
        if u == 0 and not final:
            copy_reads, copy_writes = remnant_thunks(parity)
            head.setdefault(min(1, n_blocks - 1), []).extend(copy_reads)
            head.setdefault(min(1 + _REMNANT_GAP, n_blocks - 1), []).extend(copy_writes)
        vmem = [] if final else [
            lambda t=t: nxt.append(t()) for t in scale_thunks(clamp_stage(ir.add(kt, ir.const_i32(1))))
        ]
        if group_start is not None:
            vmem += [lambda t=t: group_vals.append(t()) for t in a_group_thunks(group_start)]
        if a_dma_epoch is not None:
            vmem += as_dma_thunks(a_dma_epoch)
        reads = read_thunks(q, 1, G, p % _L2R_B_SLOTS)
        accs = step_mfma(accs, F, *scale_ops(scales, 0), _round_robin(reads, vmem), step_sel[0], head=head)
        ir.s_waitcnt(lgkmcnt=0)
        scale_dma = ("AS",) if a_lds_next is not None and a_lds_next[1] == 0 else ()
        if spec.ablated("no_dma"):
            ir.s_waitcnt(vmcnt=_VMCNT_MAX)
        else:
            landed_wait((f"B{(p + 1) % _L2R_B_SLOTS}", "A", *scale_dma))
        logged("|0")
        ir.s_barrier_bare(fence="compiler")
        F_next = [None] * (tm_w + tn_w)
        slots: dict[int, list] = {}
        rotated: list[Value] = []
        if opsel:
            if rotate_group is not None:
                rotated = [None] * (tm_w * U)
                own = rotate_group is True
                src = rotate_group if isinstance(rotate_group, _GroupDwords) else (
                    (lambda: group_vals) if own else (lambda: rotate_group))
                slots = group_rotations(src, rotated, own)
            if not final:
                slots.setdefault(n_blocks - 1, []).extend(
                    lambda k=k: nxt.__setitem__(k, rotate(nxt[k])) for k in stage_rotated
                )
        reads = read_thunks((q + 1) % (_LINE_PERIOD * _L2R_SLOTS), 0, F_next, (p + 1) % _L2R_B_SLOTS)
        a_next: list[Value] = []
        if a_lds_next is not None:
            a_next = [None] * tm_w
            reads = as_read_thunks(*a_lds_next, a_next) + reads
            if opsel:
                slots.setdefault(n_blocks - 1, []).extend(
                    lambda i=i: a_next.__setitem__(i, rotate(a_next[i])) for i in range(tm_w))
        refill = one_per_block(line_refill(n_it, u, parity), 1, {})
        accs = step_mfma(accs, G, *scale_ops(scales, 1), reads, step_sel[1], slots, head=refill)
        if spec.setprio:
            ir.s_setprio(0)
        logged("|1")
        return accs, F_next, carried_scales if final else nxt, a_next or rotated or group_vals or None, None

    def step_line_pipeline(accs, scales, group) -> list[Value]:
        """The K loop of the line ring's step pipeline, after its prologue's
        DMA (lines 0 and 1, B's stage 0) and vmcnt(0)."""
        if spec.backend == "llvm":
            ir.inline_asm("", "~{memory}", [])
        ir.s_barrier_bare(fence="compiler")
        if b_split:
            # B(1)'s halves as if issued in the steps of a stage before stage 0.
            for h in range(2):
                # dma_place: stage 0's refill, its first half, with step 1's.
                early = line_refill(ir.const_i32(0), 0, part=0) if placed and h else []
                for issue in early + b_half_thunks(clamp_stage(ir.const_i32(1)), 1 % S, h, prologue=True):
                    issue()
                logged(f"|{h}")
        elif b_vgpr:
            # B's step-0 fragments of stage 1, as if from a stage before stage 0.
            bf1: list[Value] = [None] * tn_w
            if KT > 1:
                for issue in b_vgpr_thunks(ir.const_i32(1), 0, bf1, prologue=True):
                    issue()
        else:
            for issue in dma_thunks(clamp_stage(ir.const_i32(1)), 1 % S, prologue=True, operands="b",
                                    b_tag=f"B{1 % S}" if l2r else "B"):
                issue()
        F = [None] * (tm_w + tn_w)
        for read in read_thunks(0, 0, F):
            read()
        # b_path="vgpr": (B's step-1 fragments of the next stage, its step-0
        # fragments of the one after).
        b_carry = [None]
        if b_vgpr:
            bf0, bg0 = b_stage0
            F[tm_w:] = bf0
            b_carry[0] = (bg0, bf1)
        if as_lds_on:
            group = [None] * tm_w
            for read in as_read_thunks(ir.const_i32(0), 0, group):
                read()
            if opsel:
                group = [as_fix(x) for x in group]
        if b_split and sync_sink:
            # E(0) after stage 0's first block. In the loop it is the back
            # edge's E too; on entry no B half precedes the mark it counts to.
            if sync_early:
                end_lgkm()
                pending_sync[0] = [end_sync]
            else:
                pending_sync[0] = [end_lgkm, lambda: end_vm(("BG",), 1), end_sync]
        elif b_split:
            # E(0): stage 0's step-0 DMA overwrites what F_0 read.
            end_barrier(need=())
        loops, tail = divmod(KT, U)

        def swapped(group: Sequence[Value], later: list) -> _GroupDwords:
            """iter2: `group` holds iteration j+1+h in lane half h. Each dword's
            swap gives both halves iteration j+1 (returned) and j+2 (`later`)."""

            def get(i: int, e: int) -> Value:
                x = ir.vec_extract(group[i], e)
                now, later[i * U + e] = ir.permlane32_swap(x, x)
                return now

            return _GroupDwords(get)

        def iteration(it, accs, F, scales, group, last=False, parity=None, later=None, it_static=None, w=None):
            """One loop iteration (`U` stages). iter2: `parity` 0 loads the next
            two iterations' groups and returns the second's dwords; 1 loads
            none and rotates `later`. `it_static`: `it` where it is known
            (straight-line code). line2r: `w` = `it` mod 6, the iteration's
            place in the slot pattern."""
            # The peeled last iteration with no tail after it loads no next
            # group (nothing reads it; the compiler would drop the loads that
            # the barrier waits count).
            next_group = not (last and tail == 0)
            base = ir.mul(it, ir.const_i32(U))
            nxt_group, out_later = group, None
            epoch = ir.lshr(it, ir.const_i32(1)) if as_lds_on else None
            for u in range(U):
                final = not next_group and u == U - 1
                a_next = a_dma = None
                if as_lds_on:
                    # lds2: the pair body is one epoch; its first iteration's
                    # stage 1 copies the next epoch.
                    at = U * parity + u + 1
                    if not final:
                        a_next = (ir.add(epoch, ir.const_i32(1)), 0) if at == _EPOCH_STAGES else (epoch, at)
                    if parity == 0 and u == 1 and next_group:
                        a_dma = ir.add(epoch, ir.const_i32(1))
                start = None
                if a_grouped and u == step_group_phase and next_group and parity != 1:
                    start = ir.add(base, ir.const_i32(U))
                    if parity == 0:
                        # Half 1 reads two iterations ahead, also in a loop without clamps.
                        start = ir.add(start, ir.mul(half, ir.const_i32(U)))
                        start = ir.select(ir.cmp_lt(start, c_group_last), start, c_group_last)
                    else:
                        start = clamp(start, c_group_last)
                rotate_group = None
                if opsel and a_grouped and u == U - 1 and next_group:
                    if parity == 0:
                        out_later = [None] * (tm_w * U)
                        rotate_group = swapped(nxt_group, out_later)
                    elif parity == 1:
                        rotate_group = _GroupDwords(lambda i, e: later[i * U + e])
                    else:
                        rotate_group = True if u == step_group_phase else nxt_group
                kt = ir.add(base, ir.const_i32(u))
                a_vals = list(nxt_group) if as_lds_on else stage_scales(group, u)
                if l2r:
                    accs, F, scales, loaded, b_state = step_line2r_stage(
                        kt, u, U * w + u, accs, F, scales, a_vals, start, it, rotate_group, final=final,
                        a_lds_next=a_next, a_dma_epoch=a_dma,
                    )
                else:
                    accs, F, scales, loaded, b_state = step_line_stage(
                        kt, u, accs, F, scales, a_vals, start, it,
                        rotate_group, final=final, a_lds_next=a_next, a_dma_epoch=a_dma, bv=b_carry[0],
                        t_static=None if it_static is None else it_static * U + u,
                    )
                nxt_group = loaded or nxt_group
                b_carry[0] = b_state
            return accs, F, scales, nxt_group, out_later

        n_loop = loops - 1 if spec.k_peel and loops else loops
        # iter2: a two-iteration loop body (even, odd); left-over iterations
        # after it. line2r: six iterations, the 24 stages of its slot pattern.
        pairs = a_iter2 or as_lds_on
        per_body = _L2R_PERIOD // U if l2r else 2 if pairs else 1
        n_body = n_loop // per_body
        later = None
        if n_body:
            n_acc, n_f, n_sc = len(accs), len(F), len(scales)
            carried = [(f"acc{k}", a) for k, a in enumerate(accs)]
            carried += [(f"frag{k}", f) for k, f in enumerate(F)]
            carried += [(f"sc{k}", s) for k, s in enumerate(scales)]
            carried += [(f"sa{k}", g) for k, g in enumerate(group)]
            n_g = len(group)
            b_in = [] if b_carry[0] is None else [*b_carry[0][0], *b_carry[0][1]]
            carried += [(f"bv{k}", v) for k, v in enumerate(b_in)]
            n_ac = tm_w if as_b128 else 0
            carried += [(f"ac{k}", v) for k, v in enumerate(as_cache[:n_ac])]
            clamping[0] = not spec.k_peel
            # A single-trip body is straight-line code: its stage indices are known.
            one = n_body == 1
            loop = ir.scf_for_iter(ir.const_i32(0), ir.const_i32(n_body), ir.const_i32(1), carried, iv_name="kloop")
            with loop as (it, vals):
                accs, F = list(vals[:n_acc]), list(vals[n_acc : n_acc + n_f])
                scales = list(vals[n_acc + n_f : n_acc + n_f + n_sc])
                group = list(vals[n_acc + n_f + n_sc : n_acc + n_f + n_sc + n_g])
                b_vals = list(vals[n_acc + n_f + n_sc + n_g : len(vals) - n_ac])
                as_cache[:n_ac] = vals[len(vals) - n_ac :]
                if b_in:
                    b_carry[0] = (b_vals[:tn_w], b_vals[tn_w:])
                open_loop()
                if l2r:
                    first = ir.mul(it, ir.const_i32(per_body))
                    for w in range(per_body):
                        # A single-trip loop with nothing after it ends with the last iteration.
                        accs, F, scales, group, later = iteration(
                            ir.add(first, ir.const_i32(w)) if w else first, accs, F, scales, group,
                            last=n_loop == loops == per_body and w == per_body - 1,
                            parity=w % 2 if pairs else None, later=later if w % 2 else None,
                            it_static=w if one else None, w=w,
                        )
                elif pairs:
                    even = ir.mul(it, ir.const_i32(2))
                    accs, F, scales, group, later = iteration(even, accs, F, scales, group, parity=0,
                                                              it_static=0 if one else None)
                    accs, F, scales, group, _ = iteration(
                        ir.add(even, ir.const_i32(1)), accs, F, scales, group, last=n_body * 2 == loops == 2,
                        parity=1, later=later, it_static=1 if one else None,
                    )
                else:
                    # A single-trip loop with nothing after it is the last iteration.
                    accs, F, scales, group, _ = iteration(it, accs, F, scales, group, last=n_loop == loops == 1,
                                                          it_static=0 if one else None)
                close_loop()
                # Loads past K are not emitted; their slots yield what came in.
                b_out = [] if not b_in else [
                    v if v is not None else b_vals[k] for k, v in enumerate([*b_carry[0][0], *b_carry[0][1]])
                ]
                F = [v if v is not None else vals[n_acc + k] for k, v in enumerate(F)]
                ir.scf_yield(*fence(accs), *F, *scales, *group, *b_out, *as_cache[:n_ac])
            clamping[0] = True
            res = list(loop.results)
            as_cache[:n_ac] = res[len(res) - n_ac :]
            res = res[: len(res) - n_ac]
            accs, F = res[:n_acc], res[n_acc : n_acc + n_f]
            scales, group = res[n_acc + n_f : n_acc + n_f + n_sc], res[n_acc + n_f + n_sc : n_acc + n_f + n_sc + n_g]
            if b_in:
                b_res = res[n_acc + n_f + n_sc + n_g :]
                b_carry[0] = (b_res[:tn_w], b_res[tn_w:])
        for k in range(per_body * n_body, loops):
            accs, F, scales, group, later = iteration(
                ir.const_i32(k), accs, F, scales, group, last=k == loops - 1,
                parity=k % 2 if pairs else None, later=later, it_static=k, w=k % (_L2R_PERIOD // U),
            )
        n_tail = ir.const_i32(loops) if tail else None
        for u in range(tail):
            t, final = loops * U + u, u == tail - 1
            # lds2: the tail lies in an epoch already copied.
            a_next = None
            if as_lds_on and not final:
                a_next = (ir.const_i32((t + 1) // _EPOCH_STAGES), (t + 1) % _EPOCH_STAGES)
            kt = ir.const_i32(t)
            a_vals = list(group) if as_lds_on else stage_scales(group, u + U - tail)
            if l2r:
                accs, F, scales, loaded, b_carry[0] = step_line2r_stage(
                    kt, u, t % _L2R_PERIOD, accs, F, scales, a_vals, None, n_tail, final=final, a_lds_next=a_next,
                )
            else:
                accs, F, scales, loaded, b_carry[0] = step_line_stage(
                    kt, u, accs, F, scales, a_vals, None, n_tail, final=final, a_lds_next=a_next,
                    bv=b_carry[0], t_static=t,
                )
            group = loaded or group
        # The last stage's E, left for a next stage there is none of.
        for part in pending_sync[0]:
            part()
        return accs

    # ---- prologue, K loop, epilogue ---------------------------------------
    def fence(accs: Sequence[Value]) -> list[Value]:
        if spec.acc_fence == "none":
            return list(accs)
        return [ir.inline_asm("", "=a,0", [a], a.type, sideeffect=False) for a in accs]

    scales = load_scales(ir.const_i32(0))
    group = load_a_group(ir.const_i32(0)) if a_grouped else []
    if opsel:
        scales = rotate_stage_scales(scales)
        group = group_scalars(group)
    # prologue_wait="stage0": line 1 last, left in flight for Bar(1).
    line1_late = a_line and spec.frag_pipeline == "step" and b_split and spec.prologue_wait == "stage0" and (
        spec.backend == "direct")
    if a_line:
        # Lines 0 and 1 and B's stage 0; stage 0 itself refills line 2.
        for line in range(1 if line1_late else 2):
            for issue in line_thunks(ir.const_i32(line), line, prologue=True):
                issue()
        if b_vgpr:
            # B's fragments of stage 0, both steps.
            b_stage0 = ([None] * tn_w, [None] * tn_w)
            b0 = [t for s in range(2) for t in b_vgpr_thunks(ir.const_i32(0), s, b_stage0[s], prologue=True)]
        else:
            b0 = [t for h in range(2) for t in b_half_thunks(ir.const_i32(0), 0, h, prologue=True)] if b_split else (
                dma_thunks(ir.const_i32(0), 0, prologue=True, operands="b"))
        for issue in b0:
            issue()
        if as_lds_on:
            for issue in as_dma_thunks(ir.const_i32(0)):
                issue()
        if line1_late:
            for issue in line_thunks(ir.const_i32(1), 1, prologue=True):
                issue()
        if acc_early:
            accs = fence([ir.zero_vec_f32(16) for _ in range(tm_w * tn_w)])
        ir.s_waitcnt(vmcnt=line_passes if line1_late else 0)
        del vmem_log[: len(vmem_log) - line_passes if line1_late else len(vmem_log)]
    else:
        for s in range(S - 1):
            issue_dma(ir.const_i32(s), s, prologue=True)
    if spec.ablated("no_dma") and not a_line:
        ir.s_waitcnt(vmcnt=0)
    if not acc_early:
        accs = fence([ir.zero_vec_f32(16) for _ in range(tm_w * tn_w)])
    if spec.frag_pipeline == "step" and a_line:
        loops, tail = 0, 0  # step_line_pipeline below
    elif spec.frag_pipeline == "step":
        helpers = SimpleNamespace(
            fragment=fragment, mfma_cluster=mfma_cluster, load_scales=load_scales, issue_dma=issue_dma,
            clamp_stage=clamp_stage, a_lds=a_lds, b_lds=b_lds, a_idx=a_idx, b_idx=b_idx,
            a_shift=a_shift, b_scale=b_scale, grouped=grouped, read_thunks=read_thunks,
            dma_thunks=dma_thunks, scale_thunks=scale_thunks, scale_ops=scale_ops, step_mfma=step_mfma,
        )
        accs = _step_pipeline(ir, spec, accs, scales, helpers, dma_ops)
        loops, tail = 0, 0
    else:
        # The stage barriers count whole stages of VMEM ops; the prologue
        # issued only DMA after stage 0's, so wait for stage 0 here.
        if not spec.ablated("no_dma") and not a_line:
            ir.s_waitcnt(vmcnt=(S - 2) * dma_ops)
        loops, tail = divmod(KT, U)
    # Group t holds stages t*U .. t*U+U-1; the one past the loop starts at
    # KT-U to stay in bounds, so the tail reads its last `tail` dwords.
    c_group_last = ir.const_i32(KT - U) if a_grouped else None

    def stage_scales(group: Sequence[Value], elem: int) -> list[Value] | None:
        if as_lds_on:
            return list(group)
        if not a_grouped:
            return None
        if opsel:
            return [group[i * U + elem] for i in range(tm_w)]
        return [ir.vec_extract(g, elem) for g in group]

    def iteration(it: Value, accs, scales, group, last=False):
        """One loop iteration of the stage pipeline: `U` stages. The peeled
        last iteration with no tail after it loads no next A scale group:
        nothing reads it, and the compiler would drop the loads that the
        barrier waits count."""
        next_group = not (last and tail == 0)
        base = ir.mul(it, ir.const_i32(U))
        nxt_group = group
        for u in range(U):
            start = None
            if a_grouped and u == group_phase and next_group:
                start = clamp(ir.add(base, ir.const_i32(U)), c_group_last)
            rotate_group = None
            if opsel and a_grouped and u == U - 1 and next_group:
                rotate_group = True if u == group_phase else nxt_group
            accs, scales, loaded = stage(
                ir.add(base, ir.const_i32(u)), u, accs, scales, stage_scales(group, u), start, n_it=it,
                rotate_group=rotate_group,
            )
            nxt_group = loaded or nxt_group
        return accs, scales, nxt_group

    if spec.frag_pipeline == "step" and a_line:
        accs = step_line_pipeline(accs, scales, group)
    n_loop = loops - 1 if spec.k_peel and loops else loops
    if n_loop:
        n_acc, n_sc = len(accs), len(scales)
        carried = [(f"acc{k}", a) for k, a in enumerate(accs)]
        carried += [(f"sc{k}", s) for k, s in enumerate(scales)]
        carried += [(f"sa{k}", g) for k, g in enumerate(group)]
        clamping[0] = not spec.k_peel
        loop = ir.scf_for_iter(
            ir.const_i32(0), ir.const_i32(n_loop), ir.const_i32(1), carried, iv_name="kloop"
        )
        with loop as (it, vals):
            accs, scales = list(vals[:n_acc]), list(vals[n_acc : n_acc + n_sc])
            group = list(vals[n_acc + n_sc :])
            open_loop()
            # A single-trip loop with nothing after it is the last iteration;
            # it matters where a barrier counts the group loads (fill_split).
            accs, scales, nxt_group = iteration(it, accs, scales, group, last=split and n_loop == loops == 1)
            close_loop()
            ir.scf_yield(*fence(accs), *scales, *nxt_group)
        clamping[0] = True
        accs = list(loop.results[:n_acc])
        scales = list(loop.results[n_acc : n_acc + n_sc])
        group = list(loop.results[n_acc + n_sc :])
    if n_loop < loops:
        accs, scales, group = iteration(ir.const_i32(loops - 1), accs, scales, group, last=True)
    n_tail = ir.const_i32(loops) if a_line and tail else None
    for u in range(tail):
        accs, scales, _ = stage(
            ir.const_i32(loops * U + u), u, accs, scales, stage_scales(group, u + U - tail), n_it=n_tail
        )
    # The last stages re-issue clamped DMA; drain it before LDS is released.
    ir.s_waitcnt(vmcnt=0)
    if asm_mfma:
        # The hazard recognizer cannot see the asm MFMAs: wait out the last
        # results before the epilogue reads the accumulators.
        for _ in range(2):
            ir.inline_asm("s_nop 15", "", [])

    c_n = ir.const_i32(spec.N)
    hint = TemporalHint.STREAMING if spec.store_nt else TemporalHint.DEFAULT

    def put(idx: Value, val: Value, n: int = 1) -> None:
        if n == 1 and not spec.store_nt:
            ir.global_store(C, idx, val, align=2)
        else:
            ir.global_store_vN(C, idx, val, n, align=2 * n, temporal_hint=hint)

    if spec.epilogue == "default":
        row0 = ir.add(ir.add(m0, ir.mul(wm, ir.const_i32(spec.wave_tile_m))), ir.mul(half, ir.const_i32(4)))
        col0 = ir.add(ir.add(n0, ir.mul(wn, ir.const_i32(spec.wave_tile_n))), r32)
        for i in range(tm_w):
            for j in range(tn_w):
                acc = accs[i * tn_w + j]
                for slot in range(16):
                    r = ir.add(row0, ir.const_i32(32 * i + (slot // 4) * 8 + slot % 4))
                    c = ir.add(col0, ir.const_i32(32 * j))
                    put(ir.add(ir.mul(r, c_n), c), ir.cast_f32_to(ir.vec_extract(acc, slot), c_ty))
        ir.ret()
        return ir.kernel
    # The wave's origin (uniform) and the lane's offset from it, once.
    origin = ir.add(ir.mul(ir.add(m0, ir.mul(wm, ir.const_i32(spec.wave_tile_m))), c_n),
                    ir.add(n0, ir.mul(wn, ir.const_i32(spec.wave_tile_n))))
    if spec.c_staged:
        _staged_epilogue(ir, spec, accs, a_ring_lds, wave, lane, r32, half, origin, c_t, c_ty, put)
        ir.ret()
        return ir.kernel
    if c_t:
        # Lane (h, r): row r of the 32, columns 4h.. of every 8 (ct4: 8h.. of every 16).
        lane_off = ir.add(ir.mul(r32, c_n), ir.mul(half, ir.const_i32(8 if spec.epilogue == "ct4" else 4)))
    else:
        lane_off = ir.add(ir.mul(half, ir.const_i32(4 * spec.N)), r32)

    def at(off: int) -> Value:
        return ir.add(lane_off, ir.add(origin, ir.const_i32(off)))

    def rounded(acc: Value, slots: Sequence[int]) -> Value:
        return ir.vec_cast_f32_to(ir.vec_pack([ir.vec_extract(acc, s) for s in slots], F32), c_ty)

    for i in range(tm_w):
        for j in range(tn_w):
            acc = accs[i * tn_w + j]
            if c_t and spec.epilogue == "ct4":
                for p in range(2):
                    x0, x1 = (ir.bitcast(rounded(acc, range(4 * g, 4 * g + 4)), VectorType(I32, 2))
                              for g in (2 * p, 2 * p + 1))
                    lo, hi = zip(*(ir.permlane32_swap(ir.vec_extract(x0, d), ir.vec_extract(x1, d)) for d in range(2)))
                    out = ir.bitcast(ir.vec_pack([*lo, *hi], I32), VectorType(c_ty, 8))
                    put(at(32 * i * spec.N + 32 * j + 16 * p), out, 8)
            elif c_t:
                for g in range(4):
                    put(at(32 * i * spec.N + 32 * j + 8 * g), rounded(acc, range(4 * g, 4 * g + 4)), 4)
            elif spec.epilogue == "pk2":
                for s in range(0, 16, 2):
                    pair = rounded(acc, (s, s + 1))
                    for e in range(2):
                        row = 32 * i + (s // 4) * 8 + s % 4 + e
                        put(at(row * spec.N + 32 * j), ir.vec_extract(pair, e))
            else:
                for slot in range(16):
                    row = 32 * i + (slot // 4) * 8 + slot % 4
                    put(at(row * spec.N + 32 * j), ir.cast_f32_to(ir.vec_extract(acc, slot), c_ty))
    ir.ret()
    return ir.kernel


def _c_swizzle(row: int) -> int:
    """The XOR on a staged C row's 16-byte chunk index (`_C_SWIZZLE`)."""
    out = 0
    for bit, mask in enumerate(_C_SWIZZLE):
        if row >> bit & 1:
            out ^= mask
    return out


def _staged_epilogue(ir, spec, accs, ring: Value, wave: Value, lane: Value, r32: Value, half: Value,
                     origin: Value, c_t: bool, c_ty, put) -> None:
    """epilogue="lds" / "ct_lds": C through the wave's part of the line ring.

    A staged row is 64 dwords; dword d of chunk c of row r sits at
    64 r + 4 (c ^ swizzle(r)) + d. The swizzle is linear in the row's bits,
    so a lane's own row bits give a lane term and the unrolled ones a
    constant, XORed once per distinct constant. With it, a b32 write's four
    rows ("lds"), 16 rows of a b64 write ("ct_lds") and a b128 read's row
    each fall in distinct banks."""
    tn_w = spec.wave_tile_n // 32
    row_dw = spec.wave_tile_n // 2
    c = ir.const_i32

    def bit(v: Value, b: int) -> Value:
        return ir.land(ir.lshr(v, c(b)) if b else v, c(1))

    def swizzled(lane_dw: Value, cache: dict) -> Callable[[int], Value]:
        def get(k: int) -> Value:
            if k not in cache:
                cache[k] = ir.xor(lane_dw, c(k)) if k else lane_dw
            return cache[k]

        return get

    # Every wave is done reading the ring, and its DMA has landed (the vmcnt
    # wait above), before any wave writes C there.
    ir.s_waitcnt(lgkmcnt=0)
    ir.s_barrier_bare(fence="compiler")
    region = ir.mul(wave, c(_C_PASS_ROWS * row_dw))
    if c_t:
        # Lane (h, r): row r, 8 bytes at dword 2h of chunk 4j + g.
        fr = ir.add(ir.add(ir.mul(bit(r32, 0), c(_C_SWIZZLE[0])), ir.mul(bit(r32, 1), c(_C_SWIZZLE[1]))),
                    ir.add(ir.mul(bit(r32, 2), c(_C_SWIZZLE[2])), ir.mul(bit(r32, 3), c(_C_SWIZZLE[3]))))
        w_lane = ir.add(region, ir.add(ir.mul(r32, c(row_dw)), ir.mul(half, c(2))))
        w_x = swizzled(ir.mul(fr, c(4)), {})
    else:
        # Lane (h, r) with r odd swaps rows with lane r - 1: an even lane
        # keeps row R of the pair (R, R + 1) for columns (r, r + 1), an odd
        # lane row R + 1. Its dword: chunk 4j + r // 8, dword r // 2 % 4.
        odd = bit(r32, 0)
        w_lane = ir.add(region, ir.add(ir.mul(ir.add(ir.mul(half, c(4)), odd), c(row_dw)),
                                        ir.land(ir.lshr(r32, c(1)), c(3))))
        w_x = swizzled(ir.mul(ir.add(ir.lshr(r32, c(3)), ir.add(ir.mul(odd, c(_C_SWIZZLE[0])),
                                                                 ir.mul(half, c(_C_SWIZZLE[2])))), c(4)), {})
        is_odd = ir.cmp_eq(odd, c(1))
    # Reads: lane l takes chunk l % 16 of row l // 16 of every four.
    rl, cl = ir.lshr(lane, c(4)), ir.land(lane, c(15))
    r_lane = ir.add(region, ir.mul(rl, c(row_dw)))
    r_x = swizzled(ir.mul(ir.xor(cl, ir.add(ir.mul(bit(rl, 0), c(_C_SWIZZLE[0])),
                                             ir.mul(bit(rl, 1), c(_C_SWIZZLE[1])))), c(4)), {})
    g_lane = ir.add(origin, ir.add(ir.mul(rl, c(spec.N)), ir.mul(cl, c(8))))
    per_pass = _C_PASS_ROWS // 32
    for p in range(spec.wave_tile_m // _C_PASS_ROWS):
        for ii in range(per_pass):
            i = p * per_pass + ii
            for j in range(tn_w):
                acc = accs[i * tn_w + j]
                if c_t:
                    for g in range(4):
                        v = ir.vec_cast_f32_to(ir.vec_pack([ir.vec_extract(acc, s) for s in range(4 * g, 4 * g + 4)],
                                                           F32), c_ty)
                        idx = ir.add(w_x(4 * (4 * j + g)), ir.add(w_lane, c(32 * ii * row_dw)))
                        ir.smem_store_vN(ring, [idx], ir.bitcast(v, VectorType(I32, 2)), 2)
                    continue
                for s in range(0, 16, 2):
                    x0, x1 = ir.vec_extract(acc, s), ir.vec_extract(acc, s + 1)
                    got = ir.bitcast(ir.dpp_xor(ir.bitcast(ir.select(is_odd, x0, x1), I32), 1), F32)
                    lo, hi = ir.select(is_odd, got, x0), ir.select(is_odd, x1, got)
                    v = ir.vec_cast_f32_to(ir.vec_pack([lo, hi], F32), c_ty)
                    row = 32 * ii + (s // 4) * 8 + s % 4
                    k = 4 * ((4 * j) ^ _c_swizzle(row))
                    idx = ir.add(w_x(k), ir.add(w_lane, c(row * row_dw)))
                    ir.smem_store_vN(ring, [idx], ir.bitcast(v, I32), 1)
        for k in range(_C_PASS_ROWS // 4):
            idx = ir.add(r_x(4 * _c_swizzle(4 * k)), ir.add(r_lane, c(4 * k * row_dw)))
            out = ir.bitcast(ir.smem_load_vN(ring, idx, dtype=I32, n=4), VectorType(c_ty, 8))
            put(ir.add(g_lane, c((p * _C_PASS_ROWS + 4 * k) * spec.N)), out, 8)


def _step_pipeline(ir, spec, accs, sc0, h: SimpleNamespace, dma_ops: int):
    """K loop that reads each K64 step's fragments during the previous step.

    Stage k runs: read step 1 of k, MFMAs of step 0, barrier B(k+1) (stage k+1
    resident, stage k's buffer free), scales for k+2 and DMA for k+S into k's
    buffer, read step 0 of k+1, MFMAs of step 1. The final stage has no B(KT).
    With split MFMA blocks (``h.grouped``) each step's reads, and after the
    barrier the DMA and scale loads, issue between the MFMA blocks instead.
    """
    S, KT = spec.stages, spec.K // STAGE_K
    tm_w, tn_w = spec.wave_tile_m // 32, spec.wave_tile_n // 32
    n_frag = tm_w + tn_w
    group = _vmem_per_stage(spec)
    dma_ops = 0 if spec.ablated("no_dma") else dma_ops
    clamp_stage, load_scales, issue_dma = h.clamp_stage, h.load_scales, h.issue_dma

    def read_step(buf: int, step: int) -> list[Value]:
        a = [h.fragment(h.a_lds[buf], h.a_idx, step, i) for i in range(tm_w)]
        b = [h.fragment(h.b_lds[buf], h.b_idx, step, j) for j in range(tn_w)]
        return a + b

    def cluster(accs, frags, scales, step, fillers=()):
        if h.grouped:
            return h.step_mfma(accs, frags, *h.scale_ops(scales, step), list(fillers))
        a_frag, b_frag = frags[:tm_w], frags[tm_w:]
        a_sc = [ir.lshr(scales[i], h.a_shift[step]) for i in range(tm_w)]
        b_sc = [h.b_scale(scales, j, step) for j in range(tn_w)]
        if spec.ablated("no_mfma"):
            acc0 = accs[0]
            for v in [ir.vec_extract(f, 0) for f in frags] + a_sc + b_sc:
                acc0 = ir.vec_insert(acc0, ir.fadd(ir.vec_extract(acc0, 0), ir.bitcast(v, F32)), 0)
            return [acc0, *accs[1:]]
        return h.mfma_cluster(accs, a_frag, b_frag, a_sc, b_sc)

    def barrier(vmcnt: int) -> None:
        # This wave's reads of the buffer the next DMA overwrites have landed;
        # every wave has passed them, and the next stage's DMA is complete.
        ir.s_waitcnt(lgkmcnt=0)
        if spec.backend == "llvm":
            ir.inline_asm("", "~{memory}", [])
        ir.s_waitcnt(vmcnt=vmcnt)
        ir.s_barrier_bare(fence="compiler")

    def stage(k: Value, buf: int, accs, F, sc_k, sc_k1, last: bool):
        if h.grouped:
            return stage_grouped(k, buf, accs, F, sc_k, sc_k1, last)
        G = read_step(buf, 1)
        ir.sched_barrier(0)
        accs = cluster(accs, F, sc_k, 0)
        if last:
            return cluster(accs, G, sc_k, 1), None, None, None
        barrier((S - 2) * group)
        sc_k2 = load_scales(clamp_stage(ir.add(k, ir.const_i32(2))))
        issue_dma(clamp_stage(ir.add(k, ir.const_i32(S))), buf)
        F_next = read_step((buf + 1) % S, 0)
        ir.sched_barrier(0)
        return cluster(accs, G, sc_k, 1), F_next, sc_k1, sc_k2

    def stage_grouped(k: Value, buf: int, accs, F, sc_k, sc_k1, last: bool):
        G: list[Value] = [None] * n_frag
        accs = cluster(accs, F, sc_k, 0, h.read_thunks(buf, 1, G))
        if last:
            return cluster(accs, G, sc_k, 1), None, None, None
        barrier((S - 2) * group)
        F_next: list[Value] = [None] * n_frag
        sc_k2: list[Value] = []
        fillers = _round_robin(
            h.read_thunks((buf + 1) % S, 0, F_next),
            h.dma_thunks(clamp_stage(ir.add(k, ir.const_i32(S))), buf),
            [lambda t=t: sc_k2.append(t()) for t in h.scale_thunks(clamp_stage(ir.add(k, ir.const_i32(2))))],
        )
        return cluster(accs, G, sc_k, 1, fillers), F_next, sc_k1, sc_k2

    # B(0): only the DMA of stages 1..S-2 may still be in flight.
    barrier((S - 2) * dma_ops)
    sc1 = load_scales(clamp_stage(ir.const_i32(1)))
    issue_dma(clamp_stage(ir.const_i32(S - 1)), (S - 1) % S)
    F = read_step(0, 0)
    sc_k, sc_k1 = list(sc0), list(sc1)

    loops, tail = divmod(KT - 1, S)
    if loops:
        n_acc, n_f, n_s = len(accs), len(F), len(sc_k)
        carried = [(f"acc{i}", a) for i, a in enumerate(accs)]
        carried += [(f"frag{i}", f) for i, f in enumerate(F)]
        carried += [(f"sca{i}", s) for i, s in enumerate(sc_k)]
        carried += [(f"scb{i}", s) for i, s in enumerate(sc_k1)]
        loop = ir.scf_for_iter(
            ir.const_i32(0), ir.const_i32(loops), ir.const_i32(1), carried, iv_name="kstep"
        )
        with loop as (it, vals):
            vals = list(vals)
            accs = vals[:n_acc]
            F = vals[n_acc : n_acc + n_f]
            sc_k = vals[n_acc + n_f : n_acc + n_f + n_s]
            sc_k1 = vals[n_acc + n_f + n_s :]
            base = ir.mul(it, ir.const_i32(S))
            for u in range(S):
                accs, F, sc_k, sc_k1 = stage(ir.add(base, ir.const_i32(u)), u, accs, F, sc_k, sc_k1, False)
            ir.scf_yield(*accs, *F, *sc_k, *sc_k1)
        res = list(loop.results)
        accs = res[:n_acc]
        F = res[n_acc : n_acc + n_f]
        sc_k = res[n_acc + n_f : n_acc + n_f + n_s]
        sc_k1 = res[n_acc + n_f + n_s :]
    for u in range(tail):
        accs, F, sc_k, sc_k1 = stage(ir.const_i32(loops * S + u), u, accs, F, sc_k, sc_k1, False)
    accs, *_ = stage(ir.const_i32(KT - 1), (KT - 1) % S, accs, F, sc_k, sc_k1, True)
    return accs


__all__ = [
    "ABLATIONS",
    "ACC_FENCES",
    "A_SCALE_FETCHES",
    "DMA_PROBES",
    "FRAG_PIPELINES",
    "LAYOUTS",
    "LDS_SWIZZLES",
    "MFMA_FORMS",
    "MFMA_GROUPS",
    "SCHEDULES",
    "STAGE_K",
    "XCD_REMAPS",
    "TiledBlockScaledGemmSpec",
    "build_tiled_block_scaled_gemm",
    "is_valid_spec",
    "tiled_block_scaled_gemm_grid",
    "tiled_block_scaled_gemm_signature",
]
