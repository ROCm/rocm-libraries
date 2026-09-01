"""Dense flash-attention prefill kernel for gfx950 (MI355X).

Productized from the ``flash_dense_dualwave_swp`` experiment
(``kernels/gfx950/experiments/flash_dense_dualwave_swp/``). This is the shippable
step-1 pipeline with every WINNING lever baked in as always-on (no env gates):

  * **CK-1 transposed PV** — P feeds the PV MFMA in its native QK-output layout via a
    half-local V load (``pv32_v_load_paired``); the cross-half P-relayout shuffle is
    gone (~96 ``ds_bpermute`` removed). +35% over the pre-CK-1 winner.
  * **LDS bank-conflict padding on K** (``[NBUF, BN, D+8]``) — kills the 8-way conflict
    on the QK K-reads. The dominant base win (+80% over the naive baseline).
  * **native exp2_fast** (``v_exp_f32``, no overflow guard — the softmax argument is
    always <= 0) — +11.5%.
  * **full-population ``sched_group_barrier`` template** naming DS_READ/MFMA/VALU/TRANS
    per PV step.
  * **diagonal-only causal masking** — a mask-free body loop over below-diagonal KV
    tiles (~94% at Sq=8192) plus a masked diagonal tail.
  * **depth-1 cluster split** fusing exp2 into the PV MFMA loop for MFMA/VALU co-exec.
  * **partial-vmcnt software prefetch** — the per-tile K/V DMA drain is a partial
    `vmcnt` (keeps the freshest V prefetch in flight across the barrier) instead of a
    full `vmcnt(0)` serialize; raises MfmaUtil, bit-identical.
  * **PV-only `s_setprio`** — the PV MFMA cluster is bracketed at raised priority so
    it wins issue slots; paired with the prefetch this is a measured ~+3.5%.
  * **vectorized O store**.

Measured on MI355X (bf16, D=128, causal, 128/8 GQA, Sq=8192, 0 spill, err ~1.46e-3
vs SDPA). Absolute TFLOPS swing +/-25-30% with auto-clock, so only SAME-SESSION
ratios are load-bearing; one representative session, each number pinned to its
config (grid / decode / V-pad / lazy):

    default grid            (one-CTA/q-block, V-pad 32, lazy on) : ~543 TFLOPS
    persistent baseline     (qb-major,        V-pad 0,  lazy off): ~877 TFLOPS
    persistent + V-pad      (qb-major,        V-pad 32, lazy off): ~912 TFLOPS
    persistent (SHIPPED)    (hkv-major,       V-pad 32, lazy on) : ~948 TFLOPS

Clock-invariant deltas (the load-bearing part): hkv/qb ~1.04x, V-pad 0->32 ~+5%,
lazy ~+2%. Shape (batch/seqlen/heads/head_dim) is baked at build time (dense,
compile-time-sized ABI); the KV tile, occupancy hint, and persistent knobs are the
tunable parameters. Lazy online-softmax rescale (skip the O/l rescale when every
lane's tile-max is within 8 log2 of the running max) is ALWAYS-ON by default
(``lazy_rescale=True``): parity-identical (1.46e-3) and ~+2%.

Head-size / seqlen coverage:
  * ``head_size`` is 64 or 128 (bf16/fp16, MHA + GQA incl. non-power-of-2 NQK).
    D=128 uses the per-row padded LDS fast path (1 K/V row per async-DMA instr).
    D=64 packs 2 rows per instr (64 lanes x 2 bf16 = 128 elems = 2 D=64 rows),
    which rules out a per-row pad because a padded row is not contiguous with
    the next -- so K instead pads between DMA row-GROUPS (``lds_k_group_pad``):
    the group stays contiguous for the DMA while the group pitch restores the
    QK read's bank spread. V still uses the unpadded packed pitch, so a
    conflict-free transposed-V layout remains an open D=64 lever. D=128 codegen
    is byte-identical across this change (same IR hash -> same TFLOPS).
  * ``seqlen_q``/``seqlen_kv`` must be a multiple of 256 / ``block_n`` on the
    default (aligned) kernel. Non-multiple lengths are handled by a SEPARATE
    ``ragged=True`` kernel path (its own kernel_name) that pads the boundary
    tiles ON-CHIP: OOB query rows load as 0 via a bounds-checked buffer load
    (register pad), OOB keys load as 0 into LDS (LDS pad), the grid/work-item
    count is ceil'd to cover the partial last query block, and the partial O rows
    are dropped by a guarded store. Causal needs no key mask (padded ktok >=
    seqlen_kv > every real query, so causal drops them); non-causal adds a
    ktok<seqlen_kv key mask. Self-attention only. The aligned path is emitted
    byte-identically when ``ragged=False`` (no TFLOPS impact).

Experimental/negative levers from the sweep (step-2 8-cluster, K-staging, per-nsub
staging, score truncation, PV V-prefetch) are intentionally NOT carried over — see
the experiment's ``plan.md`` for their measured results.
"""

from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

from rocke.core.ir import IRBuilder, KernelDef, PtrType, BF16, F16, F32, I32, I64
from rocke.helpers.attention import mfma_32x32x16_for_dtype, pv32_v_load_paired
from rocke.helpers.schedule import MFMA, VALU, TRANS, DS_READ
from rocke.helpers.spec import kernel_name_join
from kernels.gfx950.attention_tiled_2d import _mfma_32x32_c_row, _mfma_32x32_c_col

LOG2E = 1.4426950408889634
_DTYPE_IR = {"bf16": BF16, "fp16": F16}

# Baked pipeline constants (NOT tunable knobs — these are load-bearing):
#   _BLOCK_M: query rows per CTA. The causal mask + P relayout assume 256; the
#             kernel FAULTS at other values until those hardcodes are lifted.
#   num_waves = _BLOCK_M // 32 = 8 (block = 512 threads).
#   _NBUF=2 double-buffer (NBUF=3 is a measured dead end: 256 VGPR + 58 spills).
#   _LDS_PAD=8 bf16 elements of K-row padding (the +80% bank-conflict fix).
_BLOCK_M = 256
_NBUF = 2
_LDS_PAD = 8
# _LDS_PAD_V: bf16 elements of V-row padding for the transposed PV read
#   (ds_read_b64_tr_b16). The transpose read has a stricter bank pattern than
#   K's ds_read_b128, so it needs a LARGER pad than _LDS_PAD (8): a measured
#   sweep @ GQA-8 S=8192 gives conflicts {VPAD0: 30, VPAD8: 29, VPAD16: 11,
#   VPAD32: 0} and TFLOPS {906, 901, 944, 953} -- i.e. +8 is useless here and
#   only +32 fully clears the V-read conflicts (matches flyDSL's SMEM_V_PAD).
#   Overridable via ROCKE_DENSE_VPAD for re-sweeps.
import os as _os  # noqa: E402

_LDS_PAD_V = int(_os.environ.get("ROCKE_DENSE_VPAD", "32"))
# Lazy-rescale re-anchor threshold in the log2 domain: skip the O/l rescale when
# every lane's (tile_max - running_max) <= this. exp2(8)=256 bounds P safely.
_LAZY_RESCALE_THRESHOLD = 8.0


@dataclass(frozen=True)
class AttentionDenseSpec:
    """Compile-time spec for the dense flash-attention prefill kernel.

    Functional fields (batch / seqlen / heads / head_size / causal / dtype) are baked
    into the kernel as constants — this is a dense, statically-sized ABI. ``block_n``
    and ``waves_per_eu`` are the only performance knobs; every algorithmic lever is
    always-on (see the module docstring).
    """

    # --- functional (compile-time shape) ---
    batch: int
    seqlen_q: int
    seqlen_kv: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    causal: bool = True
    dtype: str = "bf16"
    # sliding_window: left-context window W. 0 = disabled (full causal, the
    #   byte-identical always-on path). When W>0 each query token q attends to
    #   keys k in [q-W+1, q] (causal AND within-window). The valid KV region per
    #   256-row block is a slope-1 parallelogram band, so the KV loop is pruned
    #   to ~(W/block_n + block_m/block_n) tiles instead of the full causal
    #   triangle. Requires causal=True and W % block_n == 0.
    sliding_window: int = 0
    # ragged: separate kernel path for sequence lengths that are NOT a multiple
    #   of the tile geometry (seqlen_q % 256 != 0 and/or seqlen_kv % block_n != 0).
    #   Instead of host-side zero-padding (which we cannot do), the boundary tiles
    #   are padded ON-CHIP: OOB query rows load as 0 via a bounds-checked buffer
    #   load (register pad), OOB key rows load as 0 into LDS (LDS pad), the ceil'd
    #   grid covers the partial last query block, and the partial O rows are
    #   dropped by a guarded store. Causal masking already excludes the padded
    #   keys (their token index >= seqlen_kv > every real query), so causal needs
    #   NO extra key mask; non-causal adds a ktok<seqlen_kv key mask. Self-
    #   attention only (seqlen_q == seqlen_kv). 0-cost when False: the aligned
    #   kernel is emitted unchanged (byte-identical IR).
    ragged: bool = False
    # varlen: packed variable-length batch. Q/K/V/O are packed [total_tok, H, D]
    #   and per-sequence boundaries come from cu_seqlens_q/cu_seqlens_kv (int32
    #   [batch+1]) at runtime. seqlen_q/seqlen_kv are the MAX lengths (grid
    #   sizing); each sequence's length must be a multiple of block_m (q) and
    #   block_n (kv). Self-attention only (per-seq seqlen_q == seqlen_kv). Not
    #   supported with persistent. 0-cost when False (dense uniform path).
    varlen: bool = False

    # --- validated performance knobs ---
    # block_n: KV tile length. 64 (66 KB LDS, WPE-tunable) and 128 (135 KB LDS, pins
    #   the 256-VGPR cap) both match ~peak; 64 is strictly more resource-efficient.
    block_n: int = 64
    # waves_per_eu: occupancy hint. 2 is a free win (tighter allocation, still 2
    #   waves/SIMD); 3 is a measured trap (VGPR<=170 forces spills -> -20%).
    waves_per_eu: int = 2
    # lds_k_group_pad: bf16 elements of K padding between DMA row-GROUPS, on the
    #   packed head_size<128 path only (ignored at 128, which pads per row via
    #   _LDS_PAD). The async DMA writes 128//head_size rows contiguously, so a
    #   per-row pad is impossible there -- but the pad can sit BETWEEN groups and
    #   still break the QK ds_read_b128 bank pattern. A wave64 b128 read moves
    #   1024 B while LDS delivers 64 banks x 4 B = 256 B/cycle, so 4 lanes per
    #   bank is the conflict-free floor. Aggregated over all 64 lanes (4 dwords
    #   each): unpadded D=64 touches 16 of 64 banks at 16 lanes deep; a group pad
    #   of 8, 16 or 24 reaches all 64 banks at that floor; 32 falls back to 32
    #   banks at 8 deep. The whole-wave model does not separate 8/16/24, but a
    #   16-lane phase (16 x 4 dwords = one full 64-bank sweep) does: pad 8 gives
    #   16 distinct start banks, pad 16 repeats each twice -- hence 8, which is
    #   also the cheaper of the two in LDS. Must be a non-negative multiple of 8
    #   elements (16 B) because smem_load_vN stamps `align 16` on the n=8 read
    #   unconditionally, so an 8-byte-aligned pitch would keep the ds_read_b128
    #   while silently breaking its alignment contract. 0 reproduces the old
    #   unpadded layout for A/B.
    lds_k_group_pad: int = 8
    # persistent: emit the grid-stride PERSISTENT variant instead of one CTA per
    #   (query-block, head, batch). A 1-D grid of ``num_persistent`` long-lived CTAs
    #   grid-strides over the W = (seqlen_q//256)*Hq*B work items, so the per-CTA
    #   launch/dispatch + scalar setup + K/V-prime cold-start (~4.5 tile-equivalents,
    #   plan.md "CAUSAL GAP = FIXED-COST AMORTIZATION") is paid once per CU instead of
    #   once per query-block. Inner compute is byte-identical to the default path.
    #   Measured MI355X Sq=8192 causal: 512 -> 853 TFLOPS (+70%), 0 spill, err 1.46e-3.
    persistent: bool = False
    # num_persistent: number of long-lived CTAs when ``persistent``. 256 = exactly one
    #   8-wave block per CU on MI355X (256 CUs) at 2 waves/SIMD; larger oversubscribes
    #   the CUs -> a serialized 2nd block -> tail loss (304 measured -20%).
    num_persistent: int = 256
    # interleave: boustrophedon query-block ordering that reverses qb on alternating
    #   (hq,bt) planes to spread the triangular causal load across CTAs. A large-Sq
    #   lever (helps Sq>=16384) that slightly hurts small Sq; only used when persistent.
    interleave: bool = False
    # persist_decode: work-item -> (qb, hq, bt) decode for the persistent grid.
    #   "qb_major" (default): wi = qb*(Hq*B) + hq*B + bt. Balances the triangular
    #     causal load across CTAs, but every 256-CTA grid-stride phase spans ALL
    #     kv-heads at once -> large L2 footprint (measured 57% L2 hit @ GQA-8).
    #   "hkv_major": wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt, with blk
    #     folded to a low/high-paired qb. Concentrates each grid-stride phase on
    #     ~1 kv-head so the shared GQA K/V stays L2-resident across its gqa query
    #     heads (measured L2 hit 57% -> 93%, HBM misses 5.9x lower, matching the
    #     non-persistent grid). Only balances the causal triangle when each CTA
    #     grid-strides across BOTH halves of a kv-head, i.e. gqa*NQB*B >= 2*NP;
    #     otherwise each CTA gets a fixed qb (severe imbalance) -> slower.
    #   "auto" (default): hkv_major when it is balance-safe AND GQA
    #     (gqa>1 and gqa*NQB*B >= 2*num_persistent), else qb_major. Strictly >=
    #     qb_major (falls back where hkv_major would lose).
    persist_decode: str = "auto"
    # lazy_rescale: adaptive online-softmax rescale. Keep the running max as a
    #   LAZY max that only re-anchors when a tile's max exceeds it by > 8 (log2);
    #   when every lane is within 8 (wave_all vote) skip the O/l rescale entirely
    #   (a 0/1-trip scf.for compiles the skip to a wave-uniform scalar branch),
    #   cutting the per-tile VALU between the QK and PV MFMA clusters (raises
    #   MFMA utilization). P is then bounded by exp2(8)=256 (safe for fp32 accum
    #   / bf16 P) rather than <=1, so this is a numerically APPROXIMATE lever
    #   (still within bf16/fp16 tolerance). ALWAYS-ON by default (parity-identical
    #   at 1.46e-3, ~+2% TFLOPS); set False only to disable for A/B.
    lazy_rescale: bool = True
    # paged: read K/V from a paged cache [num_blocks, block_size, num_kv_heads,
    #   head_size] via block_tables indirection instead of contiguous memory.
    #   Single-sequence only in this revision. 0-cost when False (byte-identical IR).
    paged: bool = False
    # block_size: KV cache PAGE size (tokens per physical block). Distinct from
    #   block_n (the compute KV tile). Required >0 and must divide block_n when paged.
    block_size: int = 0
    # num_kv_blocks: total physical blocks in the paged cache (key_cache.shape[0]).
    #   Bounds the paged buffer resource. Required >0 when paged.
    num_kv_blocks: int = 0
    # use_sinks: A per-query-head learned scalar logit that participates in the softmax
    # denominator but has no value vector, acting as a repository for unnecessary attention mass.
    use_sinks: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_IR:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_IR)}, got {self.dtype}"
            )
        # head_size must be 64 or 128: the QK/PV MFMA tiling needs a multiple of
        # 32, and the async K/V DMA (64 lanes x 2 bf16 = 128 elems/instr) needs
        # 128 % head_size == 0 so it packs a whole number of rows per instr.
        if self.head_size not in (64, 128):
            raise ValueError(f"head_size must be 64 or 128, got {self.head_size}")
        # A group pitch that is not 16-byte aligned would keep the QK
        # ds_read_b128 while breaking its alignment contract (smem_load_vN stamps
        # `align 16` on the n=8 read unconditionally) -- wrong data or a fault,
        # silently. Checked for every head size so an unused value cannot go
        # stale and then bite when head_size changes.
        if self.lds_k_group_pad < 0 or self.lds_k_group_pad % 8 != 0:
            raise ValueError(
                "lds_k_group_pad must be a non-negative multiple of 8 bf16 "
                "elements (16 bytes) so the K group pitch stays "
                f"ds_read_b128-aligned, got {self.lds_k_group_pad}"
            )
        if self.block_n % 32 != 0:
            raise ValueError(f"block_n must be a multiple of 32, got {self.block_n}")
        if self.ragged:
            if self.seqlen_q <= 0 or self.seqlen_kv <= 0:
                raise ValueError("ragged requires positive seqlen_q/seqlen_kv")
            if self.seqlen_q != self.seqlen_kv:
                raise ValueError(
                    "ragged is self-attention only (seqlen_q == seqlen_kv), got "
                    f"{self.seqlen_q} != {self.seqlen_kv}"
                )
            if self.varlen:
                raise ValueError("ragged is not supported with varlen")
            if self.sliding_window > 0:
                raise ValueError("ragged is not supported with sliding_window")
        else:
            if self.seqlen_q % _BLOCK_M != 0:
                raise ValueError(
                    f"seqlen_q must be a multiple of {_BLOCK_M}, got {self.seqlen_q}"
                )
            if self.seqlen_kv % self.block_n != 0:
                raise ValueError(
                    f"seqlen_kv must be a multiple of block_n={self.block_n}, got {self.seqlen_kv}"
                )
        if self.num_kv_heads == 0 or self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be a positive multiple "
                f"of num_kv_heads ({self.num_kv_heads})"
            )
        if self.block_n % 32 != 0 or self.block_n <= 0:
            raise ValueError(
                f"block_n must be a positive multiple of 32, got {self.block_n}"
            )
        if self.persistent and self.num_persistent <= 0:
            raise ValueError(
                f"num_persistent must be positive, got {self.num_persistent}"
            )
        if self.persist_decode not in ("qb_major", "hkv_major", "auto"):
            raise ValueError(
                f"persist_decode must be 'qb_major', 'hkv_major', or 'auto', got "
                f"{self.persist_decode}"
            )
        if self.sliding_window < 0:
            raise ValueError(f"sliding_window must be >= 0, got {self.sliding_window}")
        if self.sliding_window > 0:
            if not self.causal:
                raise ValueError("sliding_window>0 requires causal=True")
            if self.sliding_window % self.block_n != 0:
                raise ValueError(
                    f"sliding_window ({self.sliding_window}) must be a multiple of "
                    f"block_n={self.block_n}"
                )
        if self.varlen:
            if self.persistent:
                raise ValueError("varlen is not supported with persistent=True")
            if not self.causal:
                raise ValueError("varlen requires causal=True")
        if self.waves_per_eu < 1 or self.waves_per_eu > 8:
            raise ValueError(
                f"waves_per_eu must be in [1, 8], got {self.waves_per_eu} "
                "(note: 3+ caps VGPRs at <=170 and causes spills on this kernel)"
            )
        if self.paged:
            # --- Hard layout / hardware invariants (permanent contract) ---
            if self.block_size <= 0:
                raise ValueError("paged=True requires block_size > 0")
            if self.block_size & (self.block_size - 1) != 0:
                raise ValueError(
                    f"paged block_size ({self.block_size}) must be a power of two so "
                    "the per-page div/mod lower to shift/mask (production page sizes "
                    "are 16/32); non-power-of-two is correct but pays magic-number "
                    "address math"
                )
            if self.block_n % self.block_size != 0:
                raise ValueError(
                    f"block_n ({self.block_n}) must be a multiple of page "
                    f"block_size ({self.block_size})"
                )
            rows_per_wave = (
                self.block_n // self.num_waves
            )  # per-wave K/V rows; fit 1 page
            if self.block_size < rows_per_wave or self.block_size % rows_per_wave != 0:
                raise ValueError(
                    f"paged block_size ({self.block_size}) must be >= and a multiple of "
                    f"ROWS_PER_WAVE ({rows_per_wave} = block_n/{self.num_waves}) so each "
                    f"wave's K/V rows stay within one page"
                )
            if self.num_kv_blocks <= 0:
                raise ValueError("paged=True requires num_kv_blocks > 0")
            cache_bytes = (
                self.num_kv_blocks
                * self.block_size
                * self.num_kv_heads
                * self.head_size
                * 2
            )
            if (
                cache_bytes > 2**31 - 1
            ):  # i32 byte-offset limit; i64 paging not yet impl.
                raise ValueError(
                    f"paged cache {cache_bytes} B exceeds i32 addressing (2 GiB); "
                    "i64 paging not yet implemented"
                )
            # --- Not-yet-implemented shortcuts (deferred scope, NOT a permanent
            #     contract; see the dense-paged design future-scope list). The paged
            #     mechanism is general -- these cohorts are simply unverified so far and
            #     are loud-rejected until each is validated on gfx950. Raised as
            #     ValueError so supports_*() reports "unsupported" rather than faulting.
            if self.batch != 1:
                raise ValueError("paged multi-sequence (batch>1) not yet implemented")
            if self.varlen:
                raise ValueError("paged varlen not yet implemented (single-seq only)")
            if self.persistent:
                raise ValueError(
                    "paged + persistent not yet implemented "
                    "(persistent builder is contiguous-only)"
                )
            if self.head_size != 128:
                raise ValueError("paged not yet implemented for head_size != 128")
            # forward guard: unreachable while _DTYPE_IR == {fp16, bf16} (both
            # validated); fires if a future dtype is added there before paged-validation.
            if self.dtype not in ("fp16", "bf16"):
                raise ValueError(
                    f"paged not yet implemented for dtype={self.dtype} (fp16/bf16 only)"
                )
            if self.sliding_window <= 0:
                raise ValueError(
                    "paged not yet implemented for plain-causal (sliding_window>0 only)"
                )
        if self.use_sinks and self.paged:
            raise ValueError("use_sinks is not yet supported with paged KV")
        if self.use_sinks and self.varlen:
            raise ValueError("use_sinks is not yet supported with varlen")

    @property
    def num_waves(self) -> int:
        return _BLOCK_M // 32

    @property
    def dtype_ir(self):
        return _DTYPE_IR[self.dtype]

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def resolved_persist_decode(self) -> str:
        """Resolve persist_decode='auto' to 'hkv_major' (GQA L2-locality win,
        when balance-safe) or 'qb_major'. hkv_major balances the causal triangle
        only when each CTA grid-strides across both halves of a kv-head
        (gqa*NQB*B >= 2*num_persistent) and there is GQA sharing (gqa>1)."""
        if self.persist_decode != "auto":
            return self.persist_decode
        gqa = self.num_query_heads // self.num_kv_heads
        nqb = (self.seqlen_q + _BLOCK_M - 1) // _BLOCK_M  # ceil (ragged partial)
        per_hkv = gqa * nqb * self.batch
        if gqa > 1 and per_hkv >= 2 * self.num_persistent:
            return "hkv_major"
        return "qb_major"

    def kernel_name(self) -> str:
        parts = [
            "rocke_attention_dense",
            f"d{self.head_size}",
            f"hq{self.num_query_heads}",
            f"kv{self.num_kv_heads}",
            f"bn{self.block_n}",
            self.dtype,
        ]
        # The K group pad changes the emitted layout, so it has to be part of the
        # kernel identity or two kernels that differ only in pad share a symbol
        # name (and a launcher-cache entry). Only live on the packed path, so the
        # head_size=128 name is unchanged.
        if 128 // self.head_size > 1:
            parts.append(f"kpad{self.lds_k_group_pad}")
        parts += [
            f"sq{self.seqlen_q}",
            f"sk{self.seqlen_kv}",
            "causal" if self.causal else "full",
        ]
        if self.ragged:
            parts.append("ragged")
        if self.sliding_window > 0:
            parts.append(f"swa{self.sliding_window}")
        if self.use_sinks:
            parts.append("sinks")
        if self.varlen:
            parts.append("varlen")
        if self.paged:
            parts.append(f"pgd{self.block_size}")
            parts.append(
                f"nb{self.num_kv_blocks}"
            )  # IR-live: sets the paged rsrc bound
        if self.lazy_rescale:
            parts.append("lazyrs")
        if self.persistent:
            parts.append(f"persist{self.num_persistent}")
            if self.resolved_persist_decode == "hkv_major":
                parts.append("hkvmaj")
            if self.interleave:
                parts.append("intl")
        return kernel_name_join(*parts)


def supports_attention_dense(
    spec: AttentionDenseSpec, *, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return (ok, reason). The kernel is gfx950-only and dense (no paging/bias)."""
    if arch != "gfx950":
        return False, f"attention_dense is gfx950-only (got {arch})"
    try:
        AttentionDenseSpec(**{f.name: getattr(spec, f.name) for f in spec.__dataclass_fields__.values()})  # type: ignore[attr-defined]
    except ValueError as e:
        return False, str(e)
    return True, ""


def build_attention_dense(
    spec: AttentionDenseSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """Emit the dense flash-attention prefill kernel described by ``spec``."""
    if arch != "gfx950":
        raise NotImplementedError(f"attention_dense is gfx950-only (got {arch})")

    if spec.persistent:
        return _build_attention_dense_persistent(spec)

    B = spec.batch
    Sq = spec.seqlen_q
    Skv = spec.seqlen_kv
    Hq = spec.num_query_heads
    Hkv = spec.num_kv_heads
    D = spec.head_size
    causal = spec.causal
    dtype = spec.dtype_ir

    BLOCK_M = _BLOCK_M
    WAVES = spec.num_waves
    BN = spec.block_n
    NBUF = _NBUF
    PAD = _LDS_PAD
    W = spec.sliding_window
    Wt = W // BN  # window length in KV tiles (0 when disabled)
    varlen = spec.varlen
    RAGGED = spec.ragged
    LAZY_RESCALE = spec.lazy_rescale
    use_sinks = spec.use_sinks

    K_STEPS = D // 16
    D_TILES = D // 32
    N_SUB = BN // 32
    KK_STEPS = BN // 16
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    # DMA row packing: one async_buffer_load_lds instr moves 64 lanes x 2 bf16 =
    # 128 elems. D==128 => 1 row/instr (the padded fast path, byte-identical).
    # D<128 => pack 128//D rows/instr into one contiguous LDS group; K places
    # those groups at a padded pitch, V at the plain packed pitch. Reads get the
    # correct pitch from their LDS tensor shape either way.
    ROWS_PER_INSTR = 128 // D

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    if use_sinks:
        sinks = b.param(
            "sink_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
        )
    if varlen:
        cu_q = b.param(
            "cu_seqlens_q", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        )
        cu_kv = b.param(
            "cu_seqlens_kv",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )
    block_tables = kv_lens = bt_stride = None
    if spec.paged:
        block_tables = b.param(
            "block_tables", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        )
        kv_lens = b.param(
            "kv_lens", PtrType(I32, "global"), noalias=True, readonly=True, align=4
        )
        bt_stride = b.param("block_table_stride", I32)
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))

    _exp2 = b.exp2_fast  # native v_exp_f32 (softmax arg always <= 0)

    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(8))
    neg_inf = b.const_f32(-1e30)
    if use_sinks:
        rcp_ln2 = b.const_f32(LOG2E)
        one_f = b.const_f32(1.0)

    qb = b.block_id_x()
    hq = b.block_id_y()
    bt = b.block_id_z()
    hkv = b.div(hq, b.const_i32(gqa))
    q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))

    if varlen:
        # Per-sequence token bases from cu_seqlens; early-exit q-blocks that fall
        # past this sequence's length (packed [total_tok, H, D] layout).
        q_seq0 = b.global_load_i32(cu_q, bt)
        q_seq1 = b.global_load_i32(cu_q, b.add(bt, b.const_i32(1)))
        seqlen_q_b = b.sub(q_seq1, q_seq0)
        kv_seq0 = b.global_load_i32(cu_kv, bt)
        kv_seq1 = b.global_load_i32(cu_kv, b.add(bt, b.const_i32(1)))
        seqlen_kv_b = b.sub(kv_seq1, kv_seq0)
        with b.scf_if(b.cmp_ge(b.mul(qb, b.const_i32(BLOCK_M)), seqlen_q_b)):
            b.ret()
        q_base = b.add(
            b.mul(q_seq0, b.const_i32(stride_q_tok)), b.mul(hq, b.const_i32(D))
        )
        k_base = b.add(
            b.mul(kv_seq0, b.const_i32(stride_k_tok)), b.mul(hkv, b.const_i32(D))
        )
    else:
        q_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
        k_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Skv)), b.const_i32(stride_k_tok)),
            b.mul(hkv, b.const_i32(D)),
        )

    # --- LDS allocation: PAD on K (bank-conflict fix), +PAD_V on V (transposed
    #     PV read bank-conflict pad). A per-ROW pad requires 1 row/instr (a
    #     padded row is not contiguous with the next), so the packed D<128 K
    #     loader pads between DMA row-GROUPS instead: the group stays contiguous
    #     for the multi-row DMA while the group pitch still breaks the QK read's
    #     bank pattern (see the ``lds_k_group_pad`` spec field). V keeps the
    #     unpadded packed pitch. ---
    if ROWS_PER_INSTR == 1:
        K_GROUP = 1
        K_ROWS_LDS = BN
        LDROW = D + PAD
    else:
        K_GROUP = ROWS_PER_INSTR
        K_ROWS_LDS = BN // K_GROUP
        LDROW = K_GROUP * D + spec.lds_k_group_pad
    VROW = D + _LDS_PAD_V if ROWS_PER_INSTR == 1 else D
    K_lds = b.smem_alloc(dtype, [NBUF, K_ROWS_LDS, LDROW], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [NBUF, BN, VROW], name_hint="Vlds")
    # Packed-K read decode, hoisted out of the KV loop: krow = nsub*32 + lane_m
    # with nsub*32 even, so the group/sub-row split is a shift+and on lane_m
    # plus a compile-time nsub scale.
    if K_GROUP > 1:
        k_lane_grp = b.div(lane_m, b.const_i32(K_GROUP))
        k_sub_col = b.mul(b.mod(lane_m, b.const_i32(K_GROUP)), b.const_i32(D))
    else:
        k_lane_grp = None
        k_sub_col = None

    # Q packs (B operand), scaled once by qk_scale = softmax_scale * log2(e).
    # ragged: a bounds-checked buffer load returns 0 for OOB query rows (the
    # partial last block), so padded rows are register-zero (their output is
    # dropped by the guarded store). Aligned: direct global load (unchanged IR).
    q_rsrc = b.buffer_rsrc(q, b.const_i32(B * Sq * Hq * D * 2)) if RAGGED else None
    q_tok = b.add(q_tok0, lane_m)
    q_packs = []
    for ks in range(K_STEPS):
        col = b.add(b.const_i32(ks * 16), d_base)
        addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
        if RAGGED:
            raw = b.buffer_load_vN(
                q_rsrc, b.mul(addr, b.const_i32(2)), b.const_i32(0), dtype, 8
            )
        else:
            raw = b.global_load_vN(q, addr, dtype, 8, align=16)
        elems = [
            b.cast_f32_to(b.fmul(b.cast_to_f32(b.vec_extract(raw, j)), qk_scale), dtype)
            for j in range(8)
        ]
        q_packs.append(b.vec_pack(elems, dtype))

    # ragged: ceil so the partial last KV tile is visited (its OOB keys load 0
    # into LDS and are masked out); aligned: exact.
    n_ktiles = ((Skv + BN - 1) // BN) if RAGGED else (Skv // BN)
    n_per = BLOCK_M // BN

    K_BYTES_PER_BUF = K_ROWS_LDS * LDROW * 2
    # Bytes per DMA destination group. ROWS_PER_INSTR==1 => one row, so this is
    # the row pitch and the D=128 emission is unchanged.
    K_GROUP_BYTES = LDROW * 2
    V_BYTES_PER_BUF = BN * VROW * 2
    # NOT a padded pitch: the DMA writes ROWS_PER_INSTR*D CONTIGUOUS elements, so
    # a padded V would need its second row at +D, not +D+pad. Padding V requires
    # a pad-aware transposed read, not a wider stride here.
    V_GROUP_BYTES = ROWS_PER_INSTR * VROW * 2
    ROWS_PER_WAVE = BN // WAVES
    # Group indexing uses floor division; keep it exact so a future head size or
    # tile width fails loudly instead of silently mis-addressing LDS.
    assert BN % K_GROUP == 0 and ROWS_PER_WAVE % ROWS_PER_INSTR == 0, (
        f"K row-group split must divide evenly: BN={BN} K_GROUP={K_GROUP} "
        f"ROWS_PER_WAVE={ROWS_PER_WAVE} ROWS_PER_INSTR={ROWS_PER_INSTR}"
    )
    WAVE_BYTES = 64 * 16
    V_DMA_PASSES = (BN * D) // (WAVES * 64 * 8)
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    # Emit a SEPARATE b.const_i32 per buffer_rsrc (NOT a shared IR node): develop
    # emits two consts here, so sharing one silently breaks paged=False byte-identity
    # (the attention_dense representative-IR golden). ``_kv_cache_elems`` is a plain
    # Python int; paged widens the bound to the whole cache, contiguous keeps B*Skv.
    _kv_cache_elems = (
        (spec.num_kv_blocks * spec.block_size if spec.paged else B * Skv) * Hkv * D * 2
    )
    k_rsrc = b.buffer_rsrc(k, b.const_i32(_kv_cache_elems))
    v_rsrc = b.buffer_rsrc(v, b.const_i32(_kv_cache_elems))
    v_wave_off_i64 = b.zext(b.to_sgpr_u32(b.mul(wave, b.const_i32(WAVE_BYTES))), I64)
    if spec.paged:
        # ROWS_PER_WAVE <= block_size and block_size % ROWS_PER_WAVE == 0 is
        # enforced at spec construction (__post_init__ paged validation), so every
        # wave's K/V rows fall within one page -- the per-wave block_tables hoist
        # below relies on that.
        # Single-seq: seq_base folds to 0; the bt*bt_stride form keeps bt_stride
        # live and generalizes to multi-seq. kv_lens[bt] bounds the page index.
        _pg_seq_base = b.mul(bt, bt_stride)
        _pg_kv_len = b.global_load_i32(kv_lens, bt)
        _pg_n_pages = b.div(
            b.add(_pg_kv_len, b.const_i32(spec.block_size - 1)),
            b.const_i32(spec.block_size),
        )

    def _async_load(rsrc, lds_base, buf_val, tile_key0, bytes_per_buf, group_bytes):
        """Async DMA one K/V tile into its LDS layout.

        ROWS_PER_INSTR==1 (D==128): one instr per padded row -- 64 lanes x 2 bf16
        fill exactly one D-wide row (the original byte-identical fast path), and
        a group IS a row.
        ROWS_PER_INSTR>1 (D<128): pack ROWS_PER_INSTR rows per instr into one
        contiguous group (lane l -> row l//(D/2), col 2*(l%(D/2))). The group is
        placed at group_bytes stride, which carries K's inter-group pad; V's
        group_bytes is the plain unpadded ROWS_PER_INSTR*D pitch, so its layout
        is unchanged."""
        buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(bytes_per_buf))
        if ROWS_PER_INSTR == 1:
            if spec.paged:
                # All ROWS_PER_WAVE rows of this wave fall in ONE page (asserted at
                # setup), so the block_tables lookup is wave-uniform -- hoist it out
                # of the row loop (was per-row: ~ROWS_PER_WAVE x fewer indirection
                # loads + div/mask). Per-row cost is then just a mod + add.
                _wg0 = b.add(tile_key0, b.mul(wave, b.const_i32(ROWS_PER_WAVE)))
                _wpage = b.div(_wg0, b.const_i32(spec.block_size))
                _wphys = b.masked_global_load(
                    block_tables,
                    b.add(_pg_seq_base, _wpage),
                    b.cmp_lt(_wpage, _pg_n_pages),
                    b.const_i32(0),
                    dtype=I32,
                    align=4,
                )
                _wphys_base = b.mul(_wphys, b.const_i32(spec.block_size))
                # NOTE: _wphys (the physical block id) is used raw -- intentionally
                # NOT range-checked here. The K/V read goes through a bounds-checked
                # CDNA buffer SRD (buffer_rsrc word3 0x00027000, num_records = whole
                # cache = _kv_cache_elems), so an out-of-range id yields a voff beyond
                # num_records and raw.ptr.buffer.load.lds drops it / fills 0 rather
                # than reading OOB (contained, but SILENT wrong output on a malformed
                # table). This backstop holds ONLY while the whole offset stays in the
                # i32 voffset; the deferred i64 path folds physical_block into a
                # 64-bit base (bypassing num_records) and MUST add an explicit id
                # guard there.
            for r in range(ROWS_PER_WAVE):
                row = b.add(b.mul(wave, b.const_i32(ROWS_PER_WAVE)), b.const_i32(r))
                row_lds_off = b.add(
                    buf_off, b.zext(b.mul(row, b.const_i32(group_bytes)), I64)
                )
                row_base = b.smem_ptr_add(lds_base, row_lds_off)
                gkey = b.add(tile_key0, row)
                gcol = b.mul(lane, b.const_i32(2))
                if spec.paged:
                    kv_row = b.add(
                        _wphys_base, b.mod(gkey, b.const_i32(spec.block_size))
                    )
                else:
                    kv_row = gkey
                voff = b.add(
                    b.add(k_base, b.mul(kv_row, b.const_i32(stride_k_tok))), gcol
                )
                b.async_buffer_load_lds_addr(
                    rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                )
        else:
            lanes_per_row = D // 2
            sub_row = b.div(lane, b.const_i32(lanes_per_row))
            col = b.mul(b.mod(lane, b.const_i32(lanes_per_row)), b.const_i32(2))
            groups_per_wave = ROWS_PER_WAVE // ROWS_PER_INSTR
            for it in range(groups_per_wave):
                row0 = b.add(
                    b.mul(wave, b.const_i32(ROWS_PER_WAVE)),
                    b.const_i32(it * ROWS_PER_INSTR),
                )
                grp = b.add(b.mul(wave, b.const_i32(groups_per_wave)), b.const_i32(it))
                row_lds_off = b.add(
                    buf_off, b.zext(b.mul(grp, b.const_i32(group_bytes)), I64)
                )
                row_base = b.smem_ptr_add(lds_base, row_lds_off)
                gkey = b.add(b.add(tile_key0, row0), sub_row)
                voff = b.add(b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), col)
                b.async_buffer_load_lds_addr(
                    rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                )

    def async_load_k(lds_base, buf_val, tile_key0):
        _async_load(
            k_rsrc, lds_base, buf_val, tile_key0, K_BYTES_PER_BUF, K_GROUP_BYTES
        )

    def async_load_v(lds_base, buf_val, tile_key0):
        _async_load(
            v_rsrc, lds_base, buf_val, tile_key0, V_BYTES_PER_BUF, V_GROUP_BYTES
        )

    def load_tile(buf_val, tile_idx):
        tk0 = b.mul(tile_idx, b.const_i32(BN))
        async_load_k(K_lds_addr, buf_val, tk0)
        async_load_v(V_lds_addr, buf_val, tk0)

    # ---- per-tile compute closures ----

    def do_qk(kbuf):
        """QK MFMA: S^T = K@Q^T. mfma(a=K, bv=Q) => key on the 16 per-lane accumulator
        regs (+lane^32), query on lane%32 -- the layout that keeps softmax a cheap
        in-lane reduce + one lane^32 exchange, and lets CK-1's transposed PV consume P
        with no relayout shuffle."""
        s_reg = []
        for nsub in range(N_SUB):
            acc = b.zero_vec_f32(16)
            if K_GROUP == 1:
                krow = b.add(b.const_i32(nsub * 32), lane_m)
            else:
                krow = b.add(b.const_i32(nsub * (32 // K_GROUP)), k_lane_grp)
            for ks in range(K_STEPS):
                col = b.add(b.const_i32(ks * 16), d_base)
                if K_GROUP > 1:
                    col = b.add(k_sub_col, col)
                k_pack = b.smem_load_vN(K_lds, kbuf, krow, col, dtype=dtype, n=8)
                acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
            s_reg.append([b.vec_extract(acc, i) for i in range(16)])
        return s_reg

    def do_mask(s_reg, tile_idx, lower=False, upper=True):
        """Apply causal (upper: ktok<=q) and/or sliding-window (lower:
        ktok>q-W) masks in-place on the QK-output layout. W is compile-time so
        the lower threshold folds to an immediate. No relayout (reuses the same
        lane->ktok/query_tok maps as causal)."""
        if not causal:
            return
        tile_key0 = b.mul(tile_idx, b.const_i32(BN))
        query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        # lower bound key: q - W + 1  (keep iff ktok > q - W)
        win_lo = b.sub(query_tok, b.const_i32(W)) if lower else None
        for nsub in range(N_SUB):
            sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
            for i in range(16):
                ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                if upper:
                    s_reg[nsub][i] = b.select(
                        b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                    )
                if lower:
                    s_reg[nsub][i] = b.select(
                        b.cmp_gt(ktok, win_lo), s_reg[nsub][i], neg_inf
                    )

    def do_kbound_mask(s_reg, tile_idx):
        """ragged non-causal: force scores of padded keys (ktok >= seqlen_kv, the
        OOB rows of the partial last KV tile) to -inf. Causal doesn't need this
        (padded ktok >= seqlen_kv > every real query, so causal already drops
        them). seqlen_kv is compile-time -> the bound folds to an immediate."""
        tile_key0 = b.mul(tile_idx, b.const_i32(BN))
        for nsub in range(N_SUB):
            sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
            for i in range(16):
                ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                s_reg[nsub][i] = b.select(
                    b.cmp_lt(ktok, b.const_i32(Skv)), s_reg[nsub][i], neg_inf
                )

    def softmax_max(s_reg, m_i):
        local_max = neg_inf
        for nsub in range(N_SUB):
            for i in range(16):
                local_max = b.fmax(local_max, s_reg[nsub][i])
        tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
        if LAZY_RESCALE:
            m_diff = b.fsub(tile_max, m_i)
            below_i32 = b.select(
                b.fcmp("ole", m_diff, b.const_f32(_LAZY_RESCALE_THRESHOLD)),
                b.const_i32(1),
                b.const_i32(0),
            )
            skip = b.cmp_ne(b.wave_all(below_i32), b.const_i32(0))
            m_new = b.select(skip, m_i, b.fmax(m_i, tile_max))
        else:
            skip = None
            m_new = b.fmax(m_i, tile_max)
        alpha = _exp2(b.fsub(m_i, m_new))
        return m_new, alpha, skip

    def relayout_p(p):
        """CK-1 half-local P feed: assemble the PV B-operand from lane-local P regs
        only (a bf16 cast + pack, NO cross-half warp_shuffle_xor/select). Pairs with
        the half-local V load in ``read_v`` so the K-axis stays aligned."""
        packs = []
        for kk_step in range(KK_STEPS):
            elems = []
            for kk in range(8):
                local_in_group = kk % 4
                band = kk // 4
                key_idx = kk_step * 16 + band * 8 + local_in_group
                p_tile = key_idx // 32
                row_static = key_idx % 32
                preg = (row_static // 8) * 4 + (row_static % 4)
                elems.append(b.cast_f32_to(p[p_tile][preg], dtype))
            packs.append(b.vec_pack(elems, dtype))
        return packs

    def read_v(dt, kk_step, vbuf):
        """CK-1 half-local transposed V A-operand (matches ``relayout_p``)."""
        return pv32_v_load_paired(
            b,
            V_lds=V_lds,
            v_buf=vbuf,
            n=dt,
            k=kk_step,
            lane_half32=lane_h,
            lane_col32=lane_m,
            dtype=dtype,
        )

    def do_pv(o_acc_in, p_packs, vbuf):
        out = []
        for dt in range(D_TILES):
            acc_o = o_acc_in[dt]
            for kk_step in range(KK_STEPS):
                acc_o = mfma_32x32x16_for_dtype(
                    b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                )
            out.append(acc_o)
        return out

    def rescale_o(o_acc, alpha):
        return [
            b.vec_pack(
                [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)], F32
            )
            for dt in range(D_TILES)
        ]

    def pv_fused_exp(o_acc_in, p_packs, vbuf, s_reg, m_new):
        """Depth-1 cluster: interleave exp2(s - m_new) into the PV MFMA loop so the
        softmax VALU/TRANS co-executes in the MFMA shadow. The full per-step
        instruction population (DS_READ/MFMA/VALU/TRANS) is named to sched_group_barrier
        so the IGLP grouping matches the real stream."""
        exp_per = -(-(N_SUB * 16) // (D_TILES * KK_STEPS))
        slots = [(nsub, i) for nsub in range(N_SUB) for i in range(16)]
        p_vals = [[None] * 16 for _ in range(N_SUB)]
        it = iter(slots)
        out = []
        for dt in range(D_TILES):
            acc_o = o_acc_in[dt]
            for kk_step in range(KK_STEPS):
                acc_o = mfma_32x32x16_for_dtype(
                    b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                )
                n_emit = 0
                for _ in range(exp_per):
                    slot = next(it, None)
                    if slot is None:
                        break
                    nsub, i = slot
                    p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
                    n_emit += 1
                b.sched_group_barrier(DS_READ, 2, 0)
                b.sched_group_barrier(MFMA, 1, 0)
                b.sched_group_barrier(VALU, max(1, n_emit), 0)
                b.sched_group_barrier(TRANS, max(1, n_emit), 0)
            out.append(acc_o)
        for slot in it:
            nsub, i = slot
            p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
        l_local = b.const_f32(0.0)
        for nsub in range(N_SUB):
            for i in range(16):
                l_local = b.fadd(l_local, p_vals[nsub][i])
        l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
        return out, p_vals, l_tile

    n_ktiles_val = (
        b.div(seqlen_kv_b, b.const_i32(BN)) if varlen else b.const_i32(n_ktiles)
    )
    if causal:
        n_upper = b.add(b.mul(qb, b.const_i32(n_per)), b.const_i32(n_per))
        n_upper = b.select(b.cmp_lt(n_upper, n_ktiles_val), n_upper, n_ktiles_val)
    else:
        n_upper = n_ktiles_val

    # Sliding-window: first KV tile any row in this block attends to. Valid band
    # is [start_tile, n_upper); tiles < start_tile are fully outside the window
    # (all -inf) so they are never visited (the KV-loop prune). W==0 keeps
    # start_tile=0 -> full causal, byte-identical to the always-on path.
    if causal and W > 0:
        _diag0 = b.mul(qb, b.const_i32(n_per))
        _lo_raw = b.sub(_diag0, b.const_i32(Wt))
        start_tile = b.select(
            b.cmp_gt(_lo_raw, b.const_i32(0)), _lo_raw, b.const_i32(0)
        )
    else:
        start_tile = b.const_i32(0)
    start_buf = b.mod(start_tile, b.const_i32(NBUF))
    start_buf1 = b.mod(b.add(start_tile, b.const_i32(1)), b.const_i32(NBUF))

    # Prologue: prime the K/V double buffer and compute the first (start) tile.
    load_tile(start_buf, start_tile)
    load_tile(start_buf1, b.add(start_tile, b.const_i32(1)))
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare()
    # ragged non-causal needs the key-pad mask (ktok<seqlen_kv) on any tile that
    # can hold padded keys; causal drops them for free (see do_kbound_mask).
    RAG_KBOUND = RAGGED and (not causal) and (Skv % BN != 0)
    s0 = do_qk(start_buf)
    if causal and W > 0:
        do_mask(s0, start_tile, lower=True, upper=True)
    else:
        do_mask(s0, start_tile)
    if RAG_KBOUND:
        do_kbound_mask(s0, start_tile)

    if use_sinks:
        # Load sink value for this query head and convert to log2 domain
        sink_h = b.global_load(sinks, hq, dtype, align=2)
        sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
        m_init = sink_f
        l_init = one_f
    else:
        m_init = neg_inf

    m0, alpha0, _skip0 = softmax_max(s0, m_init)
    # tile-0 softmax exp + relayout only; PV lags by one tile (fused into the loop).
    p0_vals = [
        [_exp2(b.fsub(s0[nsub][i], m0)) for i in range(16)] for nsub in range(N_SUB)
    ]
    l0_local = b.const_f32(0.0)
    for nsub in range(N_SUB):
        for i in range(16):
            l0_local = b.fadd(l0_local, p0_vals[nsub][i])
    l0 = b.fadd(l0_local, b.warp_shuffle_xor(l0_local, 32))
    if use_sinks:
        # Rescale l_init by alpha0: when m0 > m_init (sink), multiply by alpha0 to change
        # the sink's contribution from exp(sink - m_init) = 1.0 to exp(sink - m0).
        l0 = b.fadd(l0, b.fmul(l_init, alpha0))
    o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
    pk0 = relayout_p(p0_vals)

    iter_args = (
        [("m", m0), ("l", l0)]
        + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]
        + [(f"pk{kk}", pk0[kk]) for kk in range(KK_STEPS)]
    )

    _rs_ctr = [0]

    def emit_loop_body(j, carry, mask_lower=False, mask_upper=False, mask_kbound=False):
        m_i = carry[0]
        l_i = carry[1]
        o_acc = list(carry[2 : 2 + D_TILES])
        p_prev = list(carry[2 + D_TILES : 2 + D_TILES + KK_STEPS])
        kbuf = b.mod(j, b.const_i32(NBUF))
        vbuf_prev = b.mod(b.add(j, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
        pbuf = b.mod(b.add(j, b.const_i32(1)), b.const_i32(NBUF))

        # PF (partial-vmcnt prefetch): keep the freshest V(j) DMA in flight so it
        # overlaps compute instead of a full vmcnt(0) serialize (bit-identical).
        b.s_waitcnt(vmcnt=V_DMA_PASSES)
        b.s_barrier_bare()
        s = do_qk(kbuf)
        if mask_lower or mask_upper:
            do_mask(s, j, lower=mask_lower, upper=mask_upper)
        if mask_kbound:
            do_kbound_mask(s, j)
        m_new, alpha, skip = softmax_max(s, m_i)
        b.sched_barrier(0)  # depth-1 fence: m_new region-live-in
        b.s_setprio(1)  # PV-only s_setprio (paired with PF ~+3.5%)
        o_acc, p_vals, l_tile = pv_fused_exp(o_acc, p_prev, vbuf_prev, s, m_new)
        b.s_setprio(0)
        if LAZY_RESCALE:
            _rs_ctr[0] += 1
            tg = _rs_ctr[0]
            trips = b.select(skip, b.const_i32(0), b.const_i32(1))
            rs_args = [(f"ro{dt}_{tg}", o_acc[dt]) for dt in range(D_TILES)]
            rs_args.append((f"rl_{tg}", l_i))
            rs = b.scf_for_iter(
                b.const_i32(0), trips, b.const_i32(1), rs_args, iv_name=f"rs{tg}"
            )
            with rs as (_iv, rc):
                o_sc = rescale_o(list(rc[:D_TILES]), alpha)
                b.scf_yield(*o_sc, b.fmul(rc[D_TILES], alpha))
            o_acc = list(rs.results[:D_TILES])
            l_new = b.fadd(rs.results[D_TILES], l_tile)
        else:
            l_new = b.fadd(b.fmul(l_i, alpha), l_tile)
            o_acc = rescale_o(o_acc, alpha)
        p_packs = relayout_p(p_vals)
        b.s_barrier_bare()
        load_tile(pbuf, b.add(j, b.const_i32(1)))
        b.scf_yield(m_new, l_new, *o_acc, *p_packs)

    if causal and W > 0:
        # Sliding-window three-phase band loop (prologue already did start_tile):
        #   L: [start+1, mid_lo)  window-edge tiles (masked)
        #   M: [mid_lo, mid_hi)   interior (mask-free: both bounds hold for all rows)
        #   R: [mid_hi, n_upper)  causal-edge tiles (masked)
        # Boundary phases apply BOTH bounds (robust for W<block_m overlap); the
        # redundant bound is a no-op compare. M is provably mask-free by geometry.
        diag_start = b.mul(qb, b.const_i32(n_per))
        a = b.add(start_tile, b.const_i32(1))
        left_end = b.add(diag_start, b.const_i32(n_per - Wt))  # start of mask-free M

        def _clamp(x, lo, hi):
            x = b.select(b.cmp_lt(x, lo), lo, x)  # max(x, lo)
            x = b.select(b.cmp_lt(x, hi), x, hi)  # min(x, hi)
            return x

        mid_lo = _clamp(left_end, a, n_upper)
        mid_hi = _clamp(diag_start, mid_lo, n_upper)

        phL = b.scf_for_iter(a, mid_lo, b.const_i32(1), iter_args, iv_name="swl")
        with phL as (j, carry):
            emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
        mid_args = [
            (name + "_m", val) for (name, _), val in zip(iter_args, phL.results)
        ]
        phM = b.scf_for_iter(mid_lo, mid_hi, b.const_i32(1), mid_args, iv_name="swm")
        with phM as (j, carry):
            emit_loop_body(j, carry)
        rgt_args = [
            (name + "_r", val) for (name, _), val in zip(iter_args, phM.results)
        ]
        loop = b.scf_for_iter(mid_hi, n_upper, b.const_i32(1), rgt_args, iv_name="swr")
        with loop as (j, carry):
            emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
    elif causal:
        # Diagonal-only masking: below-diagonal tiles need no mask (~94% at Sq=8192).
        diag_start = b.mul(qb, b.const_i32(n_per))
        body_upper = b.select(b.cmp_lt(diag_start, n_upper), diag_start, n_upper)
        body = b.scf_for_iter(
            b.const_i32(1), body_upper, b.const_i32(1), iter_args, iv_name="nb"
        )
        with body as (j, carry):
            emit_loop_body(j, carry)
        tail_args = [
            (name + "_t", val) for (name, _), val in zip(iter_args, body.results)
        ]
        tail_lo = b.select(
            b.cmp_lt(diag_start, b.const_i32(1)), b.const_i32(1), diag_start
        )
        loop = b.scf_for_iter(tail_lo, n_upper, b.const_i32(1), tail_args, iv_name="nt")
        with loop as (j, carry):
            emit_loop_body(j, carry, mask_upper=True)
    else:
        loop = b.scf_for_iter(
            b.const_i32(1), n_upper, b.const_i32(1), iter_args, iv_name="nkt"
        )
        with loop as (j, carry):
            emit_loop_body(j, carry, mask_kbound=RAG_KBOUND)

    res = loop.results
    l_i = res[1]
    o_acc = list(res[2 : 2 + D_TILES])
    p_prev = list(res[2 + D_TILES : 2 + D_TILES + KK_STEPS])

    # PF: drain the last iter's in-flight V prefetch before the epilogue do_pv.
    b.s_waitcnt(vmcnt=0)
    b.s_barrier_bare()
    last_vbuf = b.mod(b.add(n_upper, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
    o_acc = do_pv(o_acc, p_prev, last_vbuf)

    # Epilogue: O = (P@V) / l, vectorized bf16 store.
    rcp_l = b.rcp(l_i)
    if varlen:
        o_base = b.add(
            b.mul(q_seq0, b.const_i32(stride_q_tok)), b.mul(hq, b.const_i32(D))
        )
    else:
        o_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
    qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
    q_row_byte = b.add(o_base, b.mul(qtok, b.const_i32(stride_q_tok)))
    d_half = b.mul(lane_h, b.const_i32(4))
    # ragged: drop padded query rows (qtok >= seqlen_q) via a per-lane guard so
    # they never write (and never clobber a neighbouring batch's real rows). A
    # buffer store's OOB-drop only protects the last batch's overflow, so use an
    # explicit predicate that is correct for any batch.
    o_store_ctx = (
        b.scf_if(b.cmp_lt(qtok, b.const_i32(Sq))) if RAGGED else _nullcontext()
    )
    with o_store_ctx:
        for dt in range(D_TILES):
            for g in range(4):
                d0 = b.add(b.const_i32(dt * 32 + g * 8), d_half)
                addr = b.add(q_row_byte, d0)
                vals = [
                    b.cast_f32_to(
                        b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                    )
                    for kk in range(4)
                ]
                b.global_store_vN(o, addr, b.vec_pack(vals, dtype), 4, align=8)
    b.ret()
    return b.kernel


def _build_attention_dense_persistent(spec: AttentionDenseSpec) -> KernelDef:
    """Persistent (grid-stride) variant of the dense flash-attention kernel.

    Launches a 1-D grid of ``spec.num_persistent`` long-lived CTAs; each CTA
    grid-strides over the flattened work-item space ``W = (Sq//BLOCK_M)*Hq*B`` and
    runs the byte-identical inner step-1 CK-1 pipeline per work item, so the per-CTA
    launch/dispatch + scalar setup + K/V-prime cold-start is amortized once per CU
    instead of once per query-block (see the ``persistent`` spec field). Every
    algorithmic lever is the same always-on set as the default build; the only
    differences are the outer work loop, the qb-major work decode (load-balances the
    causal triangle), the per-work-item state reset, and ``exp_per=1`` (keeps the
    extra loop-carried index math within 256 VGPR at 0 spill; numerically identical
    to the default's ``exp_per=2`` — pure emission ordering)."""
    B = spec.batch
    Sq = spec.seqlen_q
    Skv = spec.seqlen_kv
    Hq = spec.num_query_heads
    Hkv = spec.num_kv_heads
    D = spec.head_size
    causal = spec.causal
    dtype = spec.dtype_ir

    BLOCK_M = _BLOCK_M
    WAVES = spec.num_waves
    BN = spec.block_n
    NBUF = _NBUF
    PAD = _LDS_PAD
    NP = spec.num_persistent
    INTERLEAVE = spec.interleave

    K_STEPS = D // 16
    D_TILES = D // 32
    N_SUB = BN // 32
    KK_STEPS = BN // 16
    gqa = Hq // Hkv
    stride_q_tok = Hq * D
    stride_k_tok = Hkv * D
    # DMA row packing (see default builder): 1 row/instr for D==128 (padded fast
    # path), else pack 128//D rows/instr into one contiguous LDS group, with K's
    # groups at a padded pitch and V's at the plain packed pitch.
    ROWS_PER_INSTR = 128 // D
    RAGGED = spec.ragged
    # ragged: ceil both the KV tiles and the query-block count so the partial
    # last block/tile is covered (padded rows/keys are handled on-chip).
    n_ktiles = ((Skv + BN - 1) // BN) if RAGGED else (Skv // BN)
    n_per = BLOCK_M // BN
    NQB = ((Sq + BLOCK_M - 1) // BLOCK_M) if RAGGED else (Sq // BLOCK_M)
    W = NQB * Hq * B  # total work items
    SW = spec.sliding_window  # sliding-window length (0 = disabled)
    SWt = SW // BN  # window length in KV tiles
    LAZY_RESCALE = spec.lazy_rescale
    use_sinks = spec.use_sinks

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = WAVES * 64
    b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)

    q = b.param(
        "q_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    k = b.param(
        "k_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    v = b.param(
        "v_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    o = b.param(
        "o_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    scale = b.param("scale", F32)
    if use_sinks:
        sinks = b.param(
            "sink_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
        )
    qk_scale = b.fmul(scale, b.const_f32(LOG2E))
    _exp2 = b.exp2_fast

    # ----- CTA-invariant scalar setup (paid ONCE per persistent CTA) -----
    tid = b.thread_id_x()
    wave = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    lane_m = b.mod(lane, b.const_i32(32))
    lane_h = b.div(lane, b.const_i32(32))
    d_base = b.mul(lane_h, b.const_i32(8))
    neg_inf = b.const_f32(-1e30)
    if use_sinks:
        rcp_ln2 = b.const_f32(LOG2E)
        one_f = b.const_f32(1.0)

    # 1 row/instr => per-row padded pitch (bank-conflict fix); packed D<128 =>
    # pad between DMA row-GROUPS on K, unpadded on V (see the default builder).
    if ROWS_PER_INSTR == 1:
        K_GROUP = 1
        K_ROWS_LDS = BN
        LDROW = D + PAD
    else:
        K_GROUP = ROWS_PER_INSTR
        K_ROWS_LDS = BN // K_GROUP
        LDROW = K_GROUP * D + spec.lds_k_group_pad
    VROW = (D + _LDS_PAD_V) if ROWS_PER_INSTR == 1 else D
    K_lds = b.smem_alloc(dtype, [NBUF, K_ROWS_LDS, LDROW], name_hint="Klds")
    V_lds = b.smem_alloc(dtype, [NBUF, BN, VROW], name_hint="Vlds")
    if K_GROUP > 1:
        k_lane_grp = b.div(lane_m, b.const_i32(K_GROUP))
        k_sub_col = b.mul(b.mod(lane_m, b.const_i32(K_GROUP)), b.const_i32(D))
    else:
        k_lane_grp = None
        k_sub_col = None

    K_BYTES_PER_BUF = K_ROWS_LDS * LDROW * 2
    K_GROUP_BYTES = LDROW * 2
    V_BYTES_PER_BUF = BN * VROW * 2
    # Not a padded pitch -- see the default builder: the DMA writes contiguous
    # rows, so padding V needs a pad-aware transposed read, not a wider stride.
    V_GROUP_BYTES = ROWS_PER_INSTR * VROW * 2
    ROWS_PER_WAVE = BN // WAVES
    assert BN % K_GROUP == 0 and ROWS_PER_WAVE % ROWS_PER_INSTR == 0, (
        f"K row-group split must divide evenly: BN={BN} K_GROUP={K_GROUP} "
        f"ROWS_PER_WAVE={ROWS_PER_WAVE} ROWS_PER_INSTR={ROWS_PER_INSTR}"
    )
    WAVE_BYTES = 64 * 16
    V_DMA_PASSES = (BN * D) // (WAVES * 64 * 8)
    zero_soff = b.const_i32(0)
    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    k_rsrc = b.buffer_rsrc(k, b.const_i32(B * Skv * Hkv * D * 2))
    v_rsrc = b.buffer_rsrc(v, b.const_i32(B * Skv * Hkv * D * 2))
    # ragged: bounds-checked Q load (OOB partial-block rows -> 0 register pad).
    q_rsrc = b.buffer_rsrc(q, b.const_i32(B * Sq * Hq * D * 2)) if RAGGED else None
    v_wave_off_i64 = b.zext(b.to_sgpr_u32(b.mul(wave, b.const_i32(WAVE_BYTES))), I64)

    # ----- persistent grid-stride loop over the flattened work-item space -----
    cta_id = b.block_id_x()
    outer = b.scf_for(cta_id, b.const_i32(W), b.const_i32(NP), iv_name="wi")
    with outer as wi:
        # Cross-work-item LDS reuse safety: drain the previous item's trailing DMA
        # and barrier so all waves finished the previous epilogue reads before we
        # reissue into the shared K/V buffers.
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()

        if spec.resolved_persist_decode == "hkv_major":
            # hkv-MAJOR + causal-balanced decode:
            #   wi = hkv*(NQB*gqa*B) + blk*(gqa*B) + hql*B + bt
            # * hkv in the MSB -> each grid-stride phase (NP consecutive wi) stays
            #   within ~1 kv-head, so the shared GQA K/V is L2-resident across its
            #   gqa query heads (recovers the non-persistent grid's locality:
            #   measured L2 hit 57% -> ~90%+ vs qb_major).
            # * `blk` (0..NQB-1) is folded to a query-block index that PAIRS a low
            #   and a high qb per CTA: blk<half -> qb=blk (cheap), blk>=half ->
            #   qb=NQB-1-(blk-half) (expensive), so a CTA that grid-strides over
            #   both halves of a kv-head does qb=X and qb=NQB-1-X (constant causal
            #   cost) -> keeps qb_major's load balance while gaining L2 locality.
            half = NQB // 2
            bt = b.mod(wi, b.const_i32(B))
            rem = b.div(wi, b.const_i32(B))  # hkv*(NQB*gqa) + blk*gqa + hql
            hql = b.mod(rem, b.const_i32(gqa))
            r2 = b.div(rem, b.const_i32(gqa))  # hkv*NQB + blk
            blk = b.mod(r2, b.const_i32(NQB))
            hkv = b.div(r2, b.const_i32(NQB))
            hq = b.add(b.mul(hkv, b.const_i32(gqa)), hql)
            # qb = blk<half ? blk : (NQB-1 - (blk-half))
            qb_hi = b.sub(b.const_i32(NQB - 1 + half), blk)  # NQB-1-(blk-half)
            qb = b.select(b.cmp_lt(blk, b.const_i32(half)), blk, qb_hi)
        else:
            # qb-MAJOR decode: wi = qb*(Hq*B) + hq*B + bt. Putting qb (the
            # triangular causal cost index) in the MSB spreads cheap+expensive
            # query blocks across each CTA under grid-stride; a qb-fast decode
            # would alias qb to a constant per CTA when NP is a multiple of NQB
            # (32x imbalance).
            bt = b.mod(wi, b.const_i32(B))
            rem = b.div(wi, b.const_i32(B))
            hq = b.mod(rem, b.const_i32(Hq))
            qb0 = b.div(rem, b.const_i32(Hq))
            if INTERLEAVE and causal and NQB > 1:
                odd = b.cmp_eq(b.mod(rem, b.const_i32(2)), b.const_i32(1))
                qb = b.select(odd, b.sub(b.const_i32(NQB - 1), qb0), qb0)
            else:
                qb = qb0
            hkv = b.div(hq, b.const_i32(gqa))

        q_tok0 = b.add(b.mul(qb, b.const_i32(BLOCK_M)), b.mul(wave, b.const_i32(32)))
        q_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq, b.const_i32(D)),
        )
        k_base = b.add(
            b.mul(b.mul(bt, b.const_i32(Skv)), b.const_i32(stride_k_tok)),
            b.mul(hkv, b.const_i32(D)),
        )

        q_tok = b.add(q_tok0, lane_m)
        q_packs = []
        for ks in range(K_STEPS):
            col = b.add(b.const_i32(ks * 16), d_base)
            addr = b.add(b.add(q_base, b.mul(q_tok, b.const_i32(stride_q_tok))), col)
            if RAGGED:
                raw = b.buffer_load_vN(
                    q_rsrc, b.mul(addr, b.const_i32(2)), b.const_i32(0), dtype, 8
                )
            else:
                raw = b.global_load_vN(q, addr, dtype, 8, align=16)
            elems = [
                b.cast_f32_to(
                    b.fmul(b.cast_to_f32(b.vec_extract(raw, j)), qk_scale), dtype
                )
                for j in range(8)
            ]
            q_packs.append(b.vec_pack(elems, dtype))

        def _async_load(rsrc, lds_base, buf_val, tile_key0, bytes_per_buf, group_bytes):
            """Async DMA one K/V tile (see default builder ``_async_load``).
            ROWS_PER_INSTR==1 (D==128): incremental one-instr-per-padded-row fast
            path (byte-identical), where a group IS a row.
            ROWS_PER_INSTR>1 (D<128): pack rows per instr into one contiguous
            group placed at ``group_bytes`` stride (K carries the inter-group pad,
            V's stride is the plain unpadded ROWS_PER_INSTR*D pitch)."""
            buf_off = b.mul(b.zext(buf_val, I64), b.const_i64(bytes_per_buf))
            if ROWS_PER_INSTR == 1:
                row0 = b.mul(wave, b.const_i32(ROWS_PER_WAVE))
                row_lds_off = b.add(
                    buf_off, b.zext(b.mul(row0, b.const_i32(group_bytes)), I64)
                )
                gcol = b.mul(lane, b.const_i32(2))
                voff = b.add(
                    b.add(
                        k_base,
                        b.mul(b.add(tile_key0, row0), b.const_i32(stride_k_tok)),
                    ),
                    gcol,
                )
                for r in range(ROWS_PER_WAVE):
                    row_base = b.smem_ptr_add(lds_base, row_lds_off)
                    b.async_buffer_load_lds_addr(
                        rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                    )
                    if r + 1 < ROWS_PER_WAVE:
                        row_lds_off = b.add(row_lds_off, b.const_i64(group_bytes))
                        voff = b.add(voff, b.const_i32(stride_k_tok))
            else:
                lanes_per_row = D // 2
                sub_row = b.div(lane, b.const_i32(lanes_per_row))
                col = b.mul(b.mod(lane, b.const_i32(lanes_per_row)), b.const_i32(2))
                groups_per_wave = ROWS_PER_WAVE // ROWS_PER_INSTR
                for it in range(groups_per_wave):
                    row0 = b.add(
                        b.mul(wave, b.const_i32(ROWS_PER_WAVE)),
                        b.const_i32(it * ROWS_PER_INSTR),
                    )
                    grp = b.add(
                        b.mul(wave, b.const_i32(groups_per_wave)), b.const_i32(it)
                    )
                    row_lds_off = b.add(
                        buf_off, b.zext(b.mul(grp, b.const_i32(group_bytes)), I64)
                    )
                    row_base = b.smem_ptr_add(lds_base, row_lds_off)
                    gkey = b.add(b.add(tile_key0, row0), sub_row)
                    voff = b.add(
                        b.add(k_base, b.mul(gkey, b.const_i32(stride_k_tok))), col
                    )
                    b.async_buffer_load_lds_addr(
                        rsrc, row_base, b.mul(voff, b.const_i32(2)), zero_soff, 1
                    )

        def async_load_k(lds_base, buf_val, tile_key0):
            _async_load(
                k_rsrc, lds_base, buf_val, tile_key0, K_BYTES_PER_BUF, K_GROUP_BYTES
            )

        def async_load_v(lds_base, buf_val, tile_key0):
            _async_load(
                v_rsrc, lds_base, buf_val, tile_key0, V_BYTES_PER_BUF, V_GROUP_BYTES
            )

        def load_tile(buf_val, tile_idx):
            tk0 = b.mul(tile_idx, b.const_i32(BN))
            async_load_k(K_lds_addr, buf_val, tk0)
            async_load_v(V_lds_addr, buf_val, tk0)

        def do_qk(kbuf):
            s_reg = []
            for nsub in range(N_SUB):
                acc = b.zero_vec_f32(16)
                if K_GROUP == 1:
                    krow = b.add(b.const_i32(nsub * 32), lane_m)
                else:
                    krow = b.add(b.const_i32(nsub * (32 // K_GROUP)), k_lane_grp)
                for ks in range(K_STEPS):
                    col = b.add(b.const_i32(ks * 16), d_base)
                    if K_GROUP > 1:
                        col = b.add(k_sub_col, col)
                    k_pack = b.smem_load_vN(K_lds, kbuf, krow, col, dtype=dtype, n=8)
                    acc = mfma_32x32x16_for_dtype(b, dtype, k_pack, q_packs[ks], acc)
                s_reg.append([b.vec_extract(acc, i) for i in range(16)])
            return s_reg

        def do_mask(s_reg, tile_idx, lower=False, upper=True):
            if not causal:
                return
            tile_key0 = b.mul(tile_idx, b.const_i32(BN))
            query_tok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
            win_lo = b.sub(query_tok, b.const_i32(SW)) if lower else None
            for nsub in range(N_SUB):
                sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
                for i in range(16):
                    ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                    if upper:
                        s_reg[nsub][i] = b.select(
                            b.cmp_le(ktok, query_tok), s_reg[nsub][i], neg_inf
                        )
                    if lower:
                        s_reg[nsub][i] = b.select(
                            b.cmp_gt(ktok, win_lo), s_reg[nsub][i], neg_inf
                        )

        def do_kbound_mask(s_reg, tile_idx):
            """ragged non-causal: -inf the padded keys (ktok >= seqlen_kv) of the
            partial last KV tile. Causal drops them for free."""
            tile_key0 = b.mul(tile_idx, b.const_i32(BN))
            for nsub in range(N_SUB):
                sub_base = b.add(tile_key0, b.const_i32(nsub * 32))
                for i in range(16):
                    ktok = b.add(sub_base, _mfma_32x32_c_row(b, lane, i))
                    s_reg[nsub][i] = b.select(
                        b.cmp_lt(ktok, b.const_i32(Skv)), s_reg[nsub][i], neg_inf
                    )

        RAG_KBOUND = RAGGED and (not causal) and (Skv % BN != 0)

        def softmax_max(s_reg, m_i):
            local_max = neg_inf
            for nsub in range(N_SUB):
                for i in range(16):
                    local_max = b.fmax(local_max, s_reg[nsub][i])
            tile_max = b.fmax(local_max, b.warp_shuffle_xor(local_max, 32))
            if LAZY_RESCALE:
                # Lazy max: only re-anchor when some lane's tile_max exceeds the
                # running max by > threshold; else keep m_i (skip the rescale).
                m_diff = b.fsub(tile_max, m_i)
                below_i32 = b.select(
                    b.fcmp("ole", m_diff, b.const_f32(_LAZY_RESCALE_THRESHOLD)),
                    b.const_i32(1),
                    b.const_i32(0),
                )
                skip = b.cmp_ne(b.wave_all(below_i32), b.const_i32(0))
                m_new = b.select(skip, m_i, b.fmax(m_i, tile_max))
            else:
                skip = None
                m_new = b.fmax(m_i, tile_max)
            alpha = _exp2(b.fsub(m_i, m_new))
            return m_new, alpha, skip

        def softmax_stats(s_reg, m_i, l_i=None):
            m_new, alpha, _skip = softmax_max(s_reg, m_i)
            p = [
                [_exp2(b.fsub(s_reg[nsub][i], m_new)) for i in range(16)]
                for nsub in range(N_SUB)
            ]
            l_local = b.const_f32(0.0)
            for nsub in range(N_SUB):
                for i in range(16):
                    l_local = b.fadd(l_local, p[nsub][i])
            l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            if use_sinks:
                # Rescale l_i by alpha: when m_new > m_i, multiply by alpha to change
                # the sink's contribution from exp(sink - m_i) to exp(sink - m_new).
                l_tile = b.fadd(l_tile, b.fmul(l_i, alpha))
            return m_new, alpha, p, l_tile

        def relayout_p(p):
            packs = []
            for kk_step in range(KK_STEPS):
                elems = []
                for kk in range(8):
                    local_in_group = kk % 4
                    band = kk // 4
                    key_idx = kk_step * 16 + band * 8 + local_in_group
                    p_tile = key_idx // 32
                    row_static = key_idx % 32
                    preg = (row_static // 8) * 4 + (row_static % 4)
                    elems.append(b.cast_f32_to(p[p_tile][preg], dtype))
                packs.append(b.vec_pack(elems, dtype))
            return packs

        def read_v(dt, kk_step, vbuf):
            return pv32_v_load_paired(
                b,
                V_lds=V_lds,
                v_buf=vbuf,
                n=dt,
                k=kk_step,
                lane_half32=lane_h,
                lane_col32=lane_m,
                dtype=dtype,
            )

        def do_pv(o_acc_in, p_packs, vbuf):
            out = []
            for dt in range(D_TILES):
                acc_o = o_acc_in[dt]
                for kk_step in range(KK_STEPS):
                    acc_o = mfma_32x32x16_for_dtype(
                        b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                    )
                out.append(acc_o)
            return out

        def rescale_o(o_acc, alpha):
            return [
                b.vec_pack(
                    [b.fmul(b.vec_extract(o_acc[dt], i), alpha) for i in range(16)],
                    F32,
                )
                for dt in range(D_TILES)
            ]

        def pv_fused_exp(o_acc_in, p_packs, vbuf, s_reg, m_new):
            exp_per = (
                1  # one exp2 per PV-MFMA step -> 256 VGPR / 0 spill (see docstring)
            )
            slots = [(nsub, i) for nsub in range(N_SUB) for i in range(16)]
            p_vals = [[None] * 16 for _ in range(N_SUB)]
            it = iter(slots)
            out = []
            for dt in range(D_TILES):
                acc_o = o_acc_in[dt]
                for kk_step in range(KK_STEPS):
                    acc_o = mfma_32x32x16_for_dtype(
                        b, dtype, read_v(dt, kk_step, vbuf), p_packs[kk_step], acc_o
                    )
                    n_emit = 0
                    for _ in range(exp_per):
                        slot = next(it, None)
                        if slot is None:
                            break
                        nsub, i = slot
                        p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
                        n_emit += 1
                    b.sched_group_barrier(DS_READ, 2, 0)
                    b.sched_group_barrier(MFMA, 1, 0)
                    b.sched_group_barrier(VALU, max(1, n_emit), 0)
                    b.sched_group_barrier(TRANS, max(1, n_emit), 0)
                out.append(acc_o)
            for slot in it:
                nsub, i = slot
                p_vals[nsub][i] = _exp2(b.fsub(s_reg[nsub][i], m_new))
            l_local = b.const_f32(0.0)
            for nsub in range(N_SUB):
                for i in range(16):
                    l_local = b.fadd(l_local, p_vals[nsub][i])
            l_tile = b.fadd(l_local, b.warp_shuffle_xor(l_local, 32))
            return out, p_vals, l_tile

        _rs_ctr = [0]

        def emit_loop_body(
            j, carry, mask_lower=False, mask_upper=False, mask_kbound=False
        ):
            m_i = carry[0]
            l_i = carry[1]
            o_acc = list(carry[2 : 2 + D_TILES])
            p_prev = list(carry[2 + D_TILES : 2 + D_TILES + KK_STEPS])
            pbuf = b.mod(b.add(j, b.const_i32(1)), b.const_i32(NBUF))
            kbuf = b.mod(j, b.const_i32(NBUF))
            vbuf_prev = b.mod(b.add(j, b.const_i32(NBUF - 1)), b.const_i32(NBUF))

            # PF (partial-vmcnt prefetch): keep the freshest V(j) DMA in flight
            # (drain only V(j-1)+K(j), both older) so DMA overlaps compute instead
            # of a full vmcnt(0) serialize. Bit-identical, raises MfmaUtil.
            b.s_waitcnt(vmcnt=V_DMA_PASSES)
            b.s_barrier_bare()
            s = do_qk(kbuf)
            if mask_lower or mask_upper:
                do_mask(s, j, lower=mask_lower, upper=mask_upper)
            if mask_kbound:
                do_kbound_mask(s, j)
            m_new, alpha, skip = softmax_max(s, m_i)
            b.sched_barrier(0)
            # PV-only s_setprio: the PV MFMA cluster wins issue slots; paired with
            # PF this converts to ~+3.5% (Sq=8192 causal, ~852 -> ~877 TFLOPS).
            b.s_setprio(1)
            o_acc, p_vals, l_tile = pv_fused_exp(o_acc, p_prev, vbuf_prev, s, m_new)
            b.s_setprio(0)
            if LAZY_RESCALE:
                # Skip the O/l rescale via a wave-uniform 0/1-trip loop when the
                # max didn't move (>threshold): 0 trips -> pass o_acc/l_i through
                # unscaled; 1 trip -> scale by alpha. Compiles to a scalar branch.
                # Unique names per emit (called once per KV-loop phase).
                _rs_ctr[0] += 1
                tg = _rs_ctr[0]
                trips = b.select(skip, b.const_i32(0), b.const_i32(1))
                rs_args = [(f"ro{dt}_{tg}", o_acc[dt]) for dt in range(D_TILES)]
                rs_args.append((f"rl_{tg}", l_i))
                rs = b.scf_for_iter(
                    b.const_i32(0), trips, b.const_i32(1), rs_args, iv_name=f"rs{tg}"
                )
                with rs as (_iv, rc):
                    o_sc = rescale_o(list(rc[:D_TILES]), alpha)
                    b.scf_yield(*o_sc, b.fmul(rc[D_TILES], alpha))
                o_acc = list(rs.results[:D_TILES])
                l_new = b.fadd(rs.results[D_TILES], l_tile)
            else:
                l_new = b.fadd(b.fmul(l_i, alpha), l_tile)
                o_acc = rescale_o(o_acc, alpha)
            p_packs = relayout_p(p_vals)
            b.s_barrier_bare()
            load_tile(pbuf, b.add(j, b.const_i32(1)))
            b.scf_yield(m_new, l_new, *o_acc, *p_packs)

        if causal:
            n_upper = b.add(b.mul(qb, b.const_i32(n_per)), b.const_i32(n_per))
            n_upper = b.select(
                b.cmp_lt(n_upper, b.const_i32(n_ktiles)),
                n_upper,
                b.const_i32(n_ktiles),
            )
        else:
            n_upper = b.const_i32(n_ktiles)

        # Sliding-window start tile (see default builder). SW==0 -> start_tile=0.
        if causal and SW > 0:
            _diag0 = b.mul(qb, b.const_i32(n_per))
            _lo_raw = b.sub(_diag0, b.const_i32(SWt))
            start_tile = b.select(
                b.cmp_gt(_lo_raw, b.const_i32(0)), _lo_raw, b.const_i32(0)
            )
        else:
            start_tile = b.const_i32(0)
        start_buf = b.mod(start_tile, b.const_i32(NBUF))
        start_buf1 = b.mod(b.add(start_tile, b.const_i32(1)), b.const_i32(NBUF))

        load_tile(start_buf, start_tile)
        load_tile(start_buf1, b.add(start_tile, b.const_i32(1)))
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        s0 = do_qk(start_buf)
        if causal and SW > 0:
            do_mask(s0, start_tile, lower=True, upper=True)
        else:
            do_mask(s0, start_tile)
        if RAG_KBOUND:
            do_kbound_mask(s0, start_tile)

        if use_sinks:
            # Load sink value for this query head and convert to log2 domain
            sink_h = b.global_load(sinks, hq, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            m_init = sink_f
            l_init = one_f
            m0, _alpha0, p0, l0 = softmax_stats(s0, m_init, l_init)
        else:
            m0, _alpha0, p0, l0 = softmax_stats(s0, neg_inf)

        o0 = [b.zero_vec_f32(16) for _ in range(D_TILES)]
        pk0 = relayout_p(p0)

        iter_args = (
            [("m", m0), ("l", l0)]
            + [(f"o{dt}", o0[dt]) for dt in range(D_TILES)]
            + [(f"pk{kk}", pk0[kk]) for kk in range(KK_STEPS)]
        )

        if causal and SW > 0:
            # Sliding-window three-phase band loop (prologue did start_tile).
            diag_start = b.mul(qb, b.const_i32(n_per))
            a = b.add(start_tile, b.const_i32(1))
            left_end = b.add(diag_start, b.const_i32(n_per - SWt))

            def _clamp(x, lo, hi):
                x = b.select(b.cmp_lt(x, lo), lo, x)
                x = b.select(b.cmp_lt(x, hi), x, hi)
                return x

            mid_lo = _clamp(left_end, a, n_upper)
            mid_hi = _clamp(diag_start, mid_lo, n_upper)

            phL = b.scf_for_iter(a, mid_lo, b.const_i32(1), iter_args, iv_name="swl")
            with phL as (j, carry):
                emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
            mid_args = [
                (name + "_m", val) for (name, _), val in zip(iter_args, phL.results)
            ]
            phM = b.scf_for_iter(
                mid_lo, mid_hi, b.const_i32(1), mid_args, iv_name="swm"
            )
            with phM as (j, carry):
                emit_loop_body(j, carry)
            rgt_args = [
                (name + "_r", val) for (name, _), val in zip(iter_args, phM.results)
            ]
            loop = b.scf_for_iter(
                mid_hi, n_upper, b.const_i32(1), rgt_args, iv_name="swr"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, mask_lower=True, mask_upper=True)
        elif causal:
            diag_start = b.mul(qb, b.const_i32(n_per))
            body_upper = b.select(b.cmp_lt(diag_start, n_upper), diag_start, n_upper)
            body = b.scf_for_iter(
                b.const_i32(1), body_upper, b.const_i32(1), iter_args, iv_name="nb"
            )
            with body as (j, carry):
                emit_loop_body(j, carry)
            tail_args = [
                (name + "_t", val) for (name, _), val in zip(iter_args, body.results)
            ]
            tail_lo = b.select(
                b.cmp_lt(diag_start, b.const_i32(1)), b.const_i32(1), diag_start
            )
            loop = b.scf_for_iter(
                tail_lo, n_upper, b.const_i32(1), tail_args, iv_name="nt"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, mask_upper=True)
        else:
            loop = b.scf_for_iter(
                b.const_i32(1), n_upper, b.const_i32(1), iter_args, iv_name="nkt"
            )
            with loop as (j, carry):
                emit_loop_body(j, carry, mask_kbound=RAG_KBOUND)

        res = loop.results
        l_i = res[1]
        o_acc = list(res[2 : 2 + D_TILES])
        p_prev = list(res[2 + D_TILES : 2 + D_TILES + KK_STEPS])

        # PF: drain the last iter's in-flight V prefetch before the epilogue do_pv.
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        last_vbuf = b.mod(b.add(n_upper, b.const_i32(NBUF - 1)), b.const_i32(NBUF))
        o_acc = do_pv(o_acc, p_prev, last_vbuf)

        # Epilogue: recompute (bt, hq) from the live loop IV so they need not cross
        # the KV loop (keeps the loop-carried live set minimal -> 0 spill). Must
        # mirror the work-item decode used at the top of the loop.
        rcp_l = b.rcp(l_i)
        bt_e = b.mod(wi, b.const_i32(B))
        if spec.resolved_persist_decode == "hkv_major":
            rem_e = b.div(wi, b.const_i32(B))
            hql_e = b.mod(rem_e, b.const_i32(gqa))
            hkv_e = b.div(b.div(rem_e, b.const_i32(gqa)), b.const_i32(NQB))
            hq_e = b.add(b.mul(hkv_e, b.const_i32(gqa)), hql_e)
        else:
            hq_e = b.mod(b.div(wi, b.const_i32(B)), b.const_i32(Hq))
        o_base = b.add(
            b.mul(b.mul(bt_e, b.const_i32(Sq)), b.const_i32(stride_q_tok)),
            b.mul(hq_e, b.const_i32(D)),
        )
        qtok = b.add(q_tok0, _mfma_32x32_c_col(b, lane, 0))
        q_row_byte = b.add(o_base, b.mul(qtok, b.const_i32(stride_q_tok)))
        d_half = b.mul(lane_h, b.const_i32(4))
        # ragged: guard padded query rows (qtok >= seqlen_q) so they never write.
        o_store_ctx = (
            b.scf_if(b.cmp_lt(qtok, b.const_i32(Sq))) if RAGGED else _nullcontext()
        )
        with o_store_ctx:
            for dt in range(D_TILES):
                for g in range(4):
                    d0 = b.add(b.const_i32(dt * 32 + g * 8), d_half)
                    addr = b.add(q_row_byte, d0)
                    vals = [
                        b.cast_f32_to(
                            b.fmul(b.vec_extract(o_acc[dt], g * 4 + kk), rcp_l), dtype
                        )
                        for kk in range(4)
                    ]
                    b.global_store_vN(o, addr, b.vec_pack(vals, dtype), 4, align=8)

    b.ret()
    return b.kernel


# --------------------------------------------------------------------------- #
# Public launch geometry + ABI (promoted from the prefill builder so the kernel
# is dispatchable / framework-callable without the host script).
# --------------------------------------------------------------------------- #


def attention_dense_grid(spec: AttentionDenseSpec) -> Tuple[int, int, int]:
    """Launch grid for ``spec``. Persistent = 1-D grid of ``num_persistent`` CTAs;
    default = one CTA per (query-block, query-head, batch)."""
    if spec.persistent:
        return (spec.num_persistent, 1, 1)
    nqb = (spec.seqlen_q + _BLOCK_M - 1) // _BLOCK_M  # ceil: ragged partial block
    return (nqb, spec.num_query_heads, spec.batch)


def attention_dense_block(spec: AttentionDenseSpec) -> Tuple[int, int, int]:
    """CTA block dims: ``num_waves`` wave64s (= 512 threads)."""
    return (spec.num_waves * 64, 1, 1)


def attention_dense_signature(spec: AttentionDenseSpec):
    """ABI signature for :class:`KernelLauncher`. q/k/v/o pointers + f32 scale,
    plus optional sink_ptr when ``spec.use_sinks``, and the two ``cu_seqlens``
    i32 pointers when ``spec.varlen`` (see :func:`build_attention_dense`)."""
    from rocke.helpers.spec import SignatureBuilder

    sig = (
        SignatureBuilder()
        .ptr("q_ptr", spec.dtype)
        .ptr("k_ptr", spec.dtype)
        .ptr("v_ptr", spec.dtype)
        .ptr("o_ptr", spec.dtype)
        .scalar("scale", "f32")
    )
    if spec.use_sinks:
        sig = sig.ptr("sink_ptr", spec.dtype)
    if spec.varlen:
        sig = sig.ptr("cu_seqlens_q", "i32").ptr("cu_seqlens_kv", "i32")
    if spec.paged:
        sig = (
            sig.ptr("block_tables", "i32")
            .ptr("kv_lens", "i32")
            .scalar("block_table_stride", "i32")
        )
    return sig.build()


_DENSE_LAUNCHER_CACHE: dict = {}


def align_up(n: int, mult: int) -> int:
    """Round ``n`` up to the next multiple of ``mult`` (kernel tile alignment)."""
    return ((int(n) + mult - 1) // mult) * mult


def run_attention_dense_torch(
    *,
    spec: AttentionDenseSpec,
    q,
    k,
    v,
    out,
    scale: float,
    stream: int = 0,
    arch: str = "gfx950",
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    block_tables=None,
    kv_lens=None,
    sinks=None,
    validate_paged: bool = True,
):
    """High-level framework entry: compile (cached) + launch the dense prefill
    kernel on torch tensors. ``q``/``k``/``v``/``out`` are dense contiguous
    tensors ([B, S, H, D] for q/out, [B, Skv, Hkv, D] for k/v); ``scale`` is the
    softmax scale (1/sqrt(D)). Returns ``out``. torch is imported lazily by the
    launcher — this module stays torch-free at import time.

    Arbitrary (non-256-multiple) sequence lengths are served WITHOUT host
    padding by the in-kernel ragged path: build ``spec`` with ``ragged=True``
    and the TRUE (un-rounded) ``seqlen_q``/``seqlen_kv`` and pass the true-length
    q/k/v/out tensors. The kernel pads the boundary tiles on-chip (register-zero
    OOB query rows, LDS-zero OOB keys) and drops the partial O rows; the grid is
    ceil-sized automatically. See the ``ragged`` spec field.

    Varlen (``spec.varlen``): the kernel emits a 7-arg ABI (packed
    ``[total_tok, H, D]`` q/k/v/o + two int32 ``cu_seqlens`` [batch+1]); pass both
    ``cu_seqlens_q`` and ``cu_seqlens_kv`` or a ``ValueError`` is raised (they are
    required — never silently launch the 5-arg ABI against a 7-arg kernel).

    Paged (``spec.paged``): K/V are a PAGED CACHE, not dense tensors -- ``k``/``v``
    are ``[num_kv_blocks, block_size, Hkv, D]`` and are addressed through
    ``block_tables`` indirection. Pass ``block_tables`` (int32
    ``[num_seqs, max_blocks_per_seq]``) and ``kv_lens`` (int32 ``[num_seqs]``); a
    ``ValueError`` is raised if either is missing (or is supplied when
    ``spec.paged`` is False). ``q``/``out`` stay dense/contiguous. Single-sequence
    only in this revision (``batch == 1``); ``spec.block_size`` is the cache page
    size and ``spec.num_kv_blocks`` MUST equal ``k.shape[0]``. ``validate_paged``
    (default True) host-checks the paged CONTENTS (a device->host sync): the used
    ``block_tables`` entries lie in ``[0, num_kv_blocks)``, and each
    ``kv_lens[i] == seqlen_kv`` (the kernel visits all compile-time ``seqlen_kv``
    tiles, so a shorter ``kv_len`` reads page 0 for the uncovered tiles ->
    wrong output). Pass False on the hot / graph-captured path to skip the sync
    (block ids then rely on the bounds-checked cache SRD reading 0, and the
    ``kv_lens == seqlen_kv`` contract becomes the caller's responsibility).

    Sinks (``spec.use_sinks``): Attention sinks -- learned scalar
    logits that participate in the softmax denominator but have no value vector.
    Pass ``sinks`` (``spec.dtype`` ``[num_query_heads]``); a ``ValueError`` is
    raised if ``sinks`` is ``None`` when ``spec.use_sinks`` is True, or if
    ``sinks`` is provided when ``spec.use_sinks`` is False."""
    ok, why = supports_attention_dense(spec, arch=arch)
    if not ok:
        raise NotImplementedError(f"attention_dense unsupported for spec: {why}")
    if spec.varlen and (cu_seqlens_q is None or cu_seqlens_kv is None):
        raise ValueError(
            "varlen=True requires cu_seqlens_q and cu_seqlens_kv (int32 [batch+1]); "
            "the varlen kernel has a 7-arg ABI and cannot be launched with q/k/v/o/scale"
        )
    if not spec.varlen and (cu_seqlens_q is not None or cu_seqlens_kv is not None):
        raise ValueError("cu_seqlens_* provided but spec.varlen is False")
    if spec.paged and (block_tables is None or kv_lens is None):
        raise ValueError("paged=True requires block_tables and kv_lens")
    if not spec.paged and (block_tables is not None or kv_lens is not None):
        raise ValueError("block_tables/kv_lens provided but spec.paged is False")
    if spec.paged:
        # Paged K/V shape guard: the paged buffer-resource bound is sized from the
        # SPEC (num_kv_blocks*block_size*num_kv_heads*head_size -- see
        # ``_kv_cache_elems`` in build_attention_dense), NOT from the tensor.
        # If the passed cache is smaller than the spec claims, that bound
        # over-reaches the real allocation, so the hardware bounds-check no longer
        # guards it and a block-table entry can drive an out-of-bounds paged-cache
        # read. Validate the cache shape against the spec that sizes the bound,
        # before any compile/launch, so a mismatch fails loudly instead of reading
        # OOB. (block_tables/kv_lens presence is already checked above.)
        want = (
            spec.num_kv_blocks,
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
        )
        for name, t in (("k", k), ("v", v)):
            got = tuple(t.shape)
            if got != want:
                raise ValueError(
                    f"paged {name} cache shape {got} != spec-derived "
                    f"[num_kv_blocks, block_size, num_kv_heads, head_size]={want}; "
                    "a mismatch mis-sizes the buffer-resource bound and can read OOB"
                )
        if validate_paged:
            # Physical block-id bounds (a CONTENTS check, unlike the metadata checks
            # above). An entry outside [0, num_kv_blocks) addresses a page outside the
            # cache; the bounds-checked SRD (see _async_load) drops it to 0 rather
            # than reading OOB, but that is silently WRONG output on a malformed
            # table -- so reject it loudly. This reads the tensors (a device->host
            # sync): pass validate_paged=False to skip on the hot/graph-captured path.
            # Only the entries the kernel dereferences are checked -- pages
            # [0, ceil(kv_len/block_size)) per seq; the rest are masked on device.
            _kvl = kv_lens.tolist() if hasattr(kv_lens, "tolist") else list(kv_lens)
            for _i in range(spec.batch):
                _kl = int(_kvl[_i])
                # Single-seq contract: the kernel visits ALL compile-time seqlen_kv
                # tiles, but the page-bounds mask uses the runtime kv_len -- so a
                # kv_len shorter than seqlen_kv leaves the uncovered tiles reading
                # page 0 (the masked block-table default) and folds them into the
                # softmax -> silently wrong output. Enforce the contract here.
                if _kl != spec.seqlen_kv:
                    raise ValueError(
                        f"paged kv_lens[{_i}]={_kl} != seqlen_kv={spec.seqlen_kv}; the "
                        "kernel reads all seqlen_kv tiles, so a shorter kv_len leaves "
                        "uncovered tiles reading page 0 -> silently wrong output"
                    )
                _npages = (_kl + spec.block_size - 1) // spec.block_size
                if _npages <= 0:
                    continue
                _used = block_tables[_i][:_npages]
                _used = _used.tolist() if hasattr(_used, "tolist") else list(_used)
                for _phys in _used:
                    _p = int(_phys)
                    if _p < 0 or _p >= spec.num_kv_blocks:
                        raise ValueError(
                            f"paged block_tables[{_i}] physical block id {_p} "
                            f"outside [0, num_kv_blocks={spec.num_kv_blocks}); a "
                            "malformed entry reads 0 via the bounds-checked cache "
                            "SRD -> silently wrong output"
                        )
    if not spec.use_sinks and sinks is not None:
        raise ValueError("sinks provided but spec.use_sinks is False")
    if spec.use_sinks:
        if sinks is None:
            raise ValueError("spec.use_sinks=True requires sinks that are not None")
        if sinks.shape != (spec.num_query_heads,):
            raise ValueError(
                f"sinks must have shape ({spec.num_query_heads},), got {tuple(sinks.shape)}"
            )
        if sinks.dtype != q.dtype:
            raise ValueError(f"sinks dtype {sinks.dtype} must match q dtype {q.dtype}")
        if not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous")
        if not sinks.is_cuda:
            raise ValueError("sinks must be a CUDA tensor")

    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    # `batch` and `waves_per_eu` are baked into the kernel (K/V buffer extents /
    # persistent work-item count W = NQB*Hq*B, and the amdgpu-waves-per-eu
    # register-file hint respectively) but are NOT part of kernel_name(), so they
    # must be part of the cache key: otherwise two specs differing only in either
    # field collide and the second silently reuses the first binary. Keying here
    # rather than widening kernel_name() keeps the emitted IR (and its hash)
    # untouched.
    key = (spec.kernel_name(), spec.batch, spec.waves_per_eu)
    launcher = _DENSE_LAUNCHER_CACHE.get(key)
    if launcher is None:
        art = compile_kernel(
            build_attention_dense(spec, arch=arch),
            arch=arch,
            backend="python",
            capture_ir_text=False,
        )
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=attention_dense_signature(spec),
        )
        _DENSE_LAUNCHER_CACHE[key] = launcher
    vals = {"q_ptr": q, "k_ptr": k, "v_ptr": v, "o_ptr": out, "scale": float(scale)}
    if spec.varlen:
        vals["cu_seqlens_q"] = cu_seqlens_q
        vals["cu_seqlens_kv"] = cu_seqlens_kv
    if spec.paged:
        vals["block_tables"] = block_tables
        vals["kv_lens"] = kv_lens
        vals["block_table_stride"] = int(block_tables.stride(0))
    if spec.use_sinks:
        vals["sink_ptr"] = sinks
    launcher(
        vals,
        config=LaunchConfig(
            grid=attention_dense_grid(spec),
            block=attention_dense_block(spec),
            stream=int(stream),
        ),
    )
    return out
