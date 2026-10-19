# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RocKE gfx1151 swapqk FMHA attention backend for vLLM v1.

Routes full-sequence prefill requests through the rocKE swapqk kernel
(faster than Triton on gfx1151 for Sq >= 512).  All other cases —
chunked prefill, mixed prefill+decode, decode-only, non-gfx1151 devices
— fall through to the existing RocmAttentionImpl logic unchanged.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
)

logger = init_logger(__name__)

# Minimum prefill seqlen to use swapqk (below this Triton is faster).
# Raise this above the prompt length to force every request down the
# context_attention_fwd path: that yields a matched control arm in which the
# backend, plumbing and KV handling are identical and only the attention
# kernel differs.
_SWAPQK_MIN_SEQLEN = int(os.environ.get("ROCKE_MIN_SEQLEN", "512"))

# Which Triton path serves work that does not go to swapqk.
#   "slice" (default) - one dense triton_prefill_attention call per request,
#                       the historical behaviour of this file
#   "super"           - delegate the whole forward to RocmAttentionImpl, i.e.
#                       chunked_prefill_paged_decode -> paged prefix_prefill.
#                       This is what vLLM's own ROCM_ATTN backend does, so it
#                       isolates the kernel swap from backend selection.
#   "batch"           - one dense triton_prefill_attention call for the entire
#                       batch. The dense kernel already offsets Q, K, V and Out
#                       by B_Start_Loc, so batching is valid whenever every
#                       request is a full prefill (which is exactly the
#                       precondition _full_prefill_layout checks).
_TRITON_MODE = os.environ.get("ROCKE_TRITON_MODE", "slice")

# GQA head fusion: how many query heads sharing a KV head ride in one CTA.
#   "auto" (default) - largest power-of-two factor of the GQA ratio, capped at
#                      _GQA_FUSE_CAP; 1 under MHA, where there is nothing to fuse
#   "1"              - off, the pre-fusion launch geometry (control arm)
#   "2"/"4"/...      - pinned, rejected by is_valid_spec if it does not divide
# Measured at Qwen3-8B prefill (Hq32/Hk8/D128/causal/S2048): F=4 halves both
# SQ_BUSY_CYCLES and TA_TA_BUSY at an identical SQ_WAVES and identical VGPR.
_GQA_FUSE = os.environ.get("ROCKE_GQA_FUSE", "auto")
# F=8 is legal (512 threads) but unmeasured, and a too-wide CTA trades the
# fetch win for launch quantization at short seqlens. Stay on measured ground.
_GQA_FUSE_CAP = int(os.environ.get("ROCKE_GQA_FUSE_CAP", "4"))

# KV tile strategy. (min_seqlen_k, block_n, k_lds), ASCENDING in min_seqlen_k;
# the last row that both matches and exactly divides seqlen_k wins.
#
# One row, because bn64+k_lds measured the fastest arm at EVERY sequence length
# tried -- there is nothing to switch between. Interleaved, 3 reps, min per rep,
# Hq32/Hk8/D128/causal/gqa_fuse=4, dispatch us (rocprofv3):
#
#   S     bn64    bn64+k_lds   bn128   Triton
#   1024   279.7     269.4     396.0    423.3
#   2048  1231.2    1112.8    1652.4   1795.4
#   4096  7484.4    4730.7    6579.7   6341.1
#   8192 37708.0   20936.7   28320.5  23945.4
#
# This retires the old two-row table. bn128 used to win past S=4096 by buying L0
# hits with a bigger tile; staging K in LDS buys strictly more of them (L2
# requests 59.0M -> 17.5M at S=8192) without bn128's 112 B of scratch, so bn128
# is now dominated at every S. Dropping it also widens eligibility, since bn64
# divides every seqlen bn128 does and more -- fewer requests routed to Triton.
# k_lds is a bn64-only lever by LDS budget; see k_lds_staging_08_18_2026.md.
_BLOCK_N_TABLE = ((0, 64, True),)

# "auto" (default) walks the table; "64"/"128" pin it for A/B control arms.
_BLOCK_N = os.environ.get("ROCKE_BLOCK_N", "auto")
# "auto" (default) takes k_lds from the table; "0"/"1" pin it for A/B control.
_K_LDS = os.environ.get("ROCKE_K_LDS", "auto")

# V layout. "1" (default) builds cfg.v_transposed, which obliges the caller to
# hand V over as [Hk,D,Sk] -- a strided permute that is a measurable share of
# per-layer prefill time at S=8192, and grows with S.
# "0" builds the row-major V gather (16 x buffer_load_f16_d16 per fragment
# instead of 2 x dwordx4) and skips the permute. Which wins is length-dependent:
# the gather penalty grows faster than the permute it saves.
_V_TRANSPOSED = os.environ.get("ROCKE_V_TRANSPOSED", "1")

# Source V from the PAGED kv cache instead of the `value` argument. vLLM's V
# cache is already stored [num_blocks, Hk, D, block_size] -- token fastest,
# which is exactly the order the PV A-fragment wants -- so reading it through
# the block table deletes the permute above outright, keeping the fast
# 2 x dwordx4 gather that ROCKE_V_TRANSPOSED=0 gives up.
#
# ON by default: measured token-identical to the permute path and never slower
# on Qwen3-8B prefill -- neutral at S=1024, ~3.5% faster than Triton at S=8192,
# in both graph and eager mode. Unlike ROCKE_V_TRANSPOSED=0, which removes the
# same permute but falls onto the 16 x d16 row-major gather, there is no
# crossover. Set ROCKE_V_PAGED=0 to restore the permute path.
#
# Every prerequisite below falls back SILENTLY to that permute path -- NOT to
# Triton, so the swapqk counter cannot see it -- hence the paged_v /
# paged_v_declined counters in ROCKE_STATS.
_V_PAGED = os.environ.get("ROCKE_V_PAGED", "1")

# Route decode-only batches to the rocKE paged split-K decode kernel instead of
# AMD's hand-tuned ops.paged_attention_rocm. OFF by default: the win is
# conditional, and the condition is not batch size.
#
# Kernel-only against the incumbent (Qwen3-8B shape D128/Hq32/Hk8, page 16,
# arms interleaved in one process, min-of-reps) the result splits exactly on the
# per-layer KV working set against the 32 MiB MALL:
#
#     KV bytes  > 32 MiB (DRAM-bound):     rocKE wins  1.04-1.10x, 12/12 points
#     KV bytes <= 32 MiB (cache-resident): rocKE loses 0.66-0.80x,  8/8 points
#
# with no exceptions -- B=1/Sk=8192, B=4/Sk=2048 and B=8/Sk=1024 are all exactly
# 33.5 MB and all three lose; one step up in either axis and all three win. In
# cache the op is latency/ALU-bound, where this kernel's 20 ds_swizzle per key
# dominate; out of cache it is purely DRAM-bound and the GQA fusion (K/V read
# once instead of 4x) plus the transposed-V vector gather win. Hence a gate on
# BYTES rather than on batch.
_PAGED_DECODE = os.environ.get("ROCKE_PAGED_DECODE", "0")

# Threshold for that gate, in MiB of KV read per layer per step. 0 routes
# unconditionally.
#
# The microbenchmark and the server are not obviously the same experiment: the
# benchmark re-reads one layer's KV in a tight loop, so a small working set
# really is MALL-resident, whereas in the server ~16 GB of weights and 35 other
# layers' KV stream past between one layer's read and its next. If that flushes
# the MALL then the cache-resident regime does not exist in production and the
# right threshold is 0. That is an e2e question, which is why it is a knob.
_PAGED_DECODE_MIN_KV_MIB = int(os.environ.get("ROCKE_PAGED_DECODE_MIN_KV_MIB", "32"))

_ARCH = "gfx1151"
_KERNEL_CACHE: dict[tuple, object] = {}

_DECODE_OP: bool | None = None


def _decode_op_ready() -> bool:
    """Register ``rocke_gfx1151::paged_decode_splitk`` once; False if it fails."""
    global _DECODE_OP
    if _DECODE_OP is None:
        try:
            from rocke.instances.gfx1151.paged_decode_splitk import (
                register_torch_custom_ops,
            )

            _DECODE_OP = bool(register_torch_custom_ops())
        except Exception as exc:
            logger.warning("[RocKE] paged decode op unavailable: %s", exc)
            _DECODE_OP = False
    return _DECODE_OP


def _pick_strategy(seqlen_k: int) -> tuple[int, bool]:
    """Measured-winning ``(block_n, k_lds)`` for this key length.

    ``block_n`` is 0 when nothing divides, which the caller must read as "route
    this request to Triton". The kernel's kv loop bound is ``seqlen_k //
    block_n``: the tail is TRUNCATED, not masked, so a non-divisible launch is a
    wrong answer rather than a slow one. This is also why the dispatch gate keys
    off this function instead of a fixed alignment -- the old ``seqlen % 32``
    gate admitted 32-aligned lengths into a bn64 kernel, silently dropping 32
    keys.

    """
    best = (0, False)
    for lo, bn, klds in _BLOCK_N_TABLE:
        if seqlen_k >= lo and seqlen_k % bn == 0:
            best = (bn, klds)
    if _BLOCK_N != "auto":
        bn = int(_BLOCK_N)
        best = (bn if seqlen_k % bn == 0 else 0, best[1])
    if _K_LDS != "auto":
        best = (best[0], _K_LDS == "1")
    return best


def _pick_gqa_fuse(num_query_heads: int, num_kv_heads: int) -> int:
    if _GQA_FUSE != "auto":
        return int(_GQA_FUSE)
    if not num_kv_heads or num_query_heads % num_kv_heads:
        return 1
    ratio = num_query_heads // num_kv_heads
    f = 1
    while f * 2 <= min(ratio, _GQA_FUSE_CAP) and ratio % (f * 2) == 0:
        f *= 2
    return f


ROCKE_STATS = {
    "swapqk": 0,
    "triton_slice": 0,
    "fallback": 0,
    "paged_v": 0,
    "paged_v_declined": 0,
    "paged_decode": 0,
    "paged_decode_declined": 0,
}

# Which decode gate turned a batch away, keyed by reason. paged_decode_declined
# says only that something refused; every gate below returns silently into the
# incumbent, so without this a misrouted config is indistinguishable from a
# correctly-declined one.
ROCKE_DECODE_DECLINE: dict[str, int] = {}


def _decline(reason: str) -> None:
    ROCKE_DECODE_DECLINE[reason] = ROCKE_DECODE_DECLINE.get(reason, 0) + 1


import atexit as _atexit


@_atexit.register
def _dump_rocke_stats():
    print(f"[RocKE] dispatch stats: {ROCKE_STATS}", flush=True)


def _get_compiled_kernel(
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    causal: bool,
    block_n: int = 64,
    k_lds: bool = True,
    kv_block_size: int = 0,
):
    """Compile+cache the swapqk HSACO for the given config. Returns None on failure."""
    # kv_block_size is in the key, not just _V_PAGED: the page size is baked
    # into the V addressing, so a cache hit across two block sizes would gather
    # from the wrong offsets while still producing plausible numbers.
    key = (
        num_query_heads,
        num_kv_heads,
        head_size,
        causal,
        _pick_gqa_fuse(num_query_heads, num_kv_heads),
        block_n,
        k_lds,
        _V_TRANSPOSED,
        kv_block_size,
    )
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    try:
        from kernels.gfx1151.wmma_fmha_swapqk import (
            SwapQKCfg,
            build_wmma_fmha_swapqk,
            is_valid_spec,
            swapqk_grid,
        )
        from rocke.helpers import compile_kernel
        from rocke.runtime.hip_module import Runtime

        cfg = SwapQKCfg(
            head_size=head_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            mask_mode="causal" if causal else "none",
            v_transposed=_V_TRANSPOSED == "1",
            # Kept as an env knob only. The kernel default is now False: the
            # recorded +3.3% did not reproduce and its sign was wrong (-4.7%
            # H24 MHA dense, -35.7% on this Qwen3-8B causal GQA shape).
            qk_douter=os.environ.get("ROCKE_QK_DOUTER", "0") == "1",
            gqa_fuse=_pick_gqa_fuse(num_query_heads, num_kv_heads),
            block_n=block_n,
            k_lds=k_lds,
            v_paged=bool(kv_block_size),
            kv_block_size=kv_block_size,
        )
        ok, why = is_valid_spec(cfg, _ARCH)
        if not ok:
            logger.warning("[RocKE] invalid swapqk cfg %s: %s", key, why)
            _KERNEL_CACHE[key] = None
            return None

        art = compile_kernel(build_wmma_fmha_swapqk(cfg, arch=_ARCH), arch=_ARCH)
        rt = Runtime()
        module = rt.load_module(art.hsaco)
        fn = module.get_function(art.kernel_name)

        entry = {
            "fn": fn,
            "module": module,
            "rt": rt,
            "cfg": cfg,
            "swapqk_grid": swapqk_grid,
        }
        _KERNEL_CACHE[key] = entry
        logger.info(
            "[RocKE] compiled swapqk for %s gqa_fuse=%d block_n=%d k_lds=%s "
            "block=%d as %s",
            key,
            cfg.gqa_fuse,
            cfg.block_n,
            cfg.k_lds,
            cfg.block_size,
            art.kernel_name,
        )
        return entry
    except Exception as exc:
        logger.warning("[RocKE] compilation failed for %s: %s", key, exc)
        _KERNEL_CACHE[key] = None
        return None


def _launch_swapqk_for_request(
    q: torch.Tensor,  # [Sq, Hq, D]
    k: torch.Tensor,  # [Sk, Hk, D]
    v: torch.Tensor,  # [Sk, Hk, D]
    out: torch.Tensor,  # [Sq, Hq, D]
    scale: float,
    causal: bool,
    entry: dict,
    paged_v: tuple | None = None,
) -> None:
    """Launch swapqk for a single request (no batch dim).

    ``paged_v`` is ``(v_cache, block_table_row, num_blocks)`` when the kernel
    reads V out of the paged cache. That path never touches ``v`` at all -- the
    cache write already laid the tokens out transposed, so the permute below --
    which runs at a small fraction of achievable bandwidth -- is not performed.
    """
    Sq, Hq, D = q.shape
    Sk, Hk, _D = k.shape

    cfg = entry["cfg"]
    swapqk_grid = entry["swapqk_grid"]
    fn = entry["fn"]
    rt = entry["rt"]

    if paged_v is not None:
        v_cache, bt_row, bt_num_entries = paged_v
        v_ptr = v_cache.data_ptr()
        v_t = None
    else:
        # V transpose [Sk,Hk,D] -> [Hk,D,Sk] for cfg.v_transposed, on device.
        v_t = v.permute(1, 2, 0).contiguous() if cfg.v_transposed else v
        v_ptr = v_t.data_ptr()

    scale_log2 = float(scale * math.log2(math.e))
    grid = swapqk_grid(cfg, seqlen_q=Sq, batch=1)
    block = (cfg.block_size, 1, 1)

    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
        q.data_ptr(),
        k.data_ptr(),
        v_ptr,
        out.data_ptr(),
        scale_log2,
        Sq,
        Sk,
        Hq * D,
        D,
        Hk * D,
        D,
        Hk * D,
        D,
        Hq * D,
        D,
    )
    if paged_v is not None:
        # Appended last, so the non-paged pack stays byte-identical. The "4x" is
        # load-bearing: kernargs align to their own size and this pointer follows
        # an odd number of i32s, so without the pad it lands 4 bytes early.
        # bt_stride is unused here (one request per launch means the kernel's
        # batch index is always 0) but must still be passed positionally.
        packed += struct.pack(
            "<4xQii", bt_row.data_ptr(), bt_num_entries, bt_num_entries
        )
    packed_buf = (ctypes.c_uint8 * len(packed)).from_buffer(bytearray(packed))

    from rocke.runtime.torch_interop import resolve_stream

    stream = resolve_stream(0, device=q.device)
    rt.launch(fn, grid, block, packed_buf, stream=stream)
    del v_t  # rt retains packed_buf until stream drain


class RockeAttentionImpl(RocmAttentionImpl):
    """Extends RocmAttentionImpl: uses swapqk for full-sequence prefill on gfx1151."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Eagerly compile causal kernel so first prefill has no compile stall.
        self._rocke_entry_causal = _get_compiled_kernel(
            self.num_heads, self.num_kv_heads, self.head_size, causal=True
        )
        self._rocke_entry_noncausal = None  # compile on first use if needed

    def _full_prefill_layout(
        self, attn_metadata: RocmAttentionMetadata
    ) -> tuple[list[int], list[int]] | None:
        """Host-side (query_start_loc, seq_lens) iff every request in the batch
        is a full-sequence prefill, else None.

        query_start_loc/seq_lens live on the device, so indexing them with
        ``int()`` costs one blocking D2H copy *each*. Both tensors are pulled
        across once here and the result is threaded through to the dispatch
        loop, which then needs no further host/device round-trips.
        """
        if attn_metadata.max_query_len <= 1:
            return None
        q_starts = attn_metadata.query_start_loc.tolist()
        s_lens = attn_metadata.seq_lens.tolist()
        for i, seq_len in enumerate(s_lens):
            if q_starts[i + 1] - q_starts[i] != seq_len:
                return None
        return q_starts, s_lens

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale=None,
        output_block_scale=None,
    ) -> torch.Tensor:
        # Checked ahead of the _TRITON_MODE gate on purpose, so that
        # ROCKE_TRITON_MODE=super + ROCKE_PAGED_DECODE=1 is a clean
        # incumbent-prefill / rocKE-decode arm that isolates this kernel.
        if (
            _PAGED_DECODE == "1"
            and attn_metadata is not None
            and attn_metadata.max_query_len == 1
            and self.attn_type == AttentionType.DECODER
            and output_scale is None
            and output_block_scale is None
            and self.alibi_slopes is None
            and getattr(self, "sinks", None) is None
            and self.sliding_window == (-1, -1)
            and self.kv_cache_dtype in ("auto", "float16")
            and query.dtype == torch.float16
            and self.num_kv_heads
            and self.num_heads % self.num_kv_heads == 0
            and _decode_op_ready()
        ):
            if self._forward_rocke_decode(query, kv_cache, attn_metadata, output):
                return output
            # Every rejection inside _forward_rocke_decode is silent -- the
            # request just goes to the incumbent -- so "the flag was on and
            # nothing happened" has to be distinguishable from "it worked".
            ROCKE_STATS["paged_decode_declined"] += 1

        if _TRITON_MODE == "super":
            ROCKE_STATS["fallback"] += 1
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # Route to swapqk only for simple full-prefill decoder attention.
        if (
            attn_metadata is not None
            and self.attn_type == AttentionType.DECODER
            and output_scale is None
            and output_block_scale is None
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and self.kv_cache_dtype in ("auto", "float16")
            and query.dtype == torch.float16
            and (layout := self._full_prefill_layout(attn_metadata)) is not None
        ):
            causal = attn_metadata.causal
            # Eligibility probe only -- "does swapqk compile for this head config
            # at all". The kernel actually launched is re-resolved per request,
            # because block_n keys off the request's length.
            probe = (
                self._rocke_entry_causal
                if causal
                else (
                    self._rocke_entry_noncausal
                    or _get_compiled_kernel(
                        self.num_heads, self.num_kv_heads, self.head_size, False
                    )
                )
            )
            if not causal:
                self._rocke_entry_noncausal = probe

            if probe is not None:
                return self._forward_rocke(
                    query, key, value, kv_cache, attn_metadata, output, layout
                )

        ROCKE_STATS["fallback"] += 1
        return super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    def _paged_decode_caches(self, kv_cache: torch.Tensor):
        """``(key_cache, value_cache, block_size)`` iff the decode kernel may run.

        Both halves are re-derived from the live tensor rather than trusted from
        the ``split_kv_cache`` view: the kernel's address math is compiled
        against K ``[blocks, Hk, D/8, slot, 8]`` and V ``[blocks, Hk, D, slot]``,
        and a vLLM change to either shape would otherwise mis-address silently
        instead of failing.
        """

        def no(reason):
            _decline(reason)
            return None

        if kv_cache is None or kv_cache.numel() == 0:
            return no("no_kv_cache")
        try:
            from vllm.v1.attention.ops.paged_attn import PagedAttention

            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )
        except Exception as exc:
            logger.warning("[RocKE] paged decode cache unavailable: %s", exc)
            return no("split_kv_cache_raised")

        if key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
            return no(f"cache_dtype={key_cache.dtype}")
        if key_cache.dim() != 5 or value_cache.dim() != 4:
            return no(f"cache_ndim={key_cache.dim()}/{value_cache.dim()}")
        if not key_cache.is_contiguous() or not value_cache.is_contiguous():
            return no("cache_not_contiguous")
        _kb, k_hk, d_over_x, k_slots, x = key_cache.shape
        _vb, v_hk, v_d, block_size = value_cache.shape
        if x != 8 or d_over_x * x != self.head_size or k_slots != block_size:
            return no(f"k_shape={tuple(key_cache.shape)}")
        if k_hk != self.num_kv_heads or v_hk != self.num_kv_heads:
            return no(f"kv_heads={k_hk}/{v_hk}!={self.num_kv_heads}")
        if v_d != self.head_size:
            return no(f"v_head_size={v_d}!={self.head_size}")
        # A lane owns head_size/32 contiguous d-elements and must not straddle
        # two x-groups, which is what is_valid_spec enforces as ept | x.
        ept = self.head_size // 32
        if self.head_size % 32 or ept not in (1, 2, 4, 8):
            return no(f"head_size={self.head_size}")
        # A 16-key tile must not straddle two physical pages.
        if block_size % 16 or block_size & (block_size - 1):
            return no(f"block_size={block_size}")
        # Both caches are indexed with i32 element offsets.
        for t in (key_cache, value_cache):
            if t.numel() * t.element_size() >= 2**31:
                return no("cache_over_2gib")
        return key_cache, value_cache, block_size

    def _forward_rocke_decode(
        self,
        query: torch.Tensor,  # [num_tokens, Hq, D], one token per request
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, Hq, D]
    ) -> bool:
        """rocKE paged split-K decode over the whole batch. False = not taken."""

        def no(reason):
            _decline(reason)
            return False

        caches = self._paged_decode_caches(kv_cache)
        if caches is None:
            return False
        key_cache, value_cache, _block_size = caches

        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        if block_table is None or block_table.dtype != torch.int32:
            return no(
                f"block_table={None if block_table is None else block_table.dtype}"
            )
        if seq_lens is None or seq_lens.dtype != torch.int32:
            return no(f"seq_lens={None if seq_lens is None else seq_lens.dtype}")

        b = attn_metadata.num_actual_tokens
        if b <= 0 or b > block_table.shape[0] or b > seq_lens.shape[0]:
            return no(f"b={b} vs bt={block_table.shape[0]} sl={seq_lens.shape[0]}")

        # KV read per layer per step -- the quantity the MALL boundary is in.
        # max_seq_len is used rather than the true per-request sum because that
        # sum lives on the device, and pulling it across would be a D2H stall
        # per layer per step and is illegal under graph capture anyway. It
        # over-estimates, so the gate errs toward routing.
        kv_mib = (
            b * attn_metadata.max_seq_len * self.num_kv_heads * self.head_size * 4
        ) / (1024 * 1024)
        if kv_mib < _PAGED_DECODE_MIN_KV_MIB:
            return no("below_kv_threshold")

        from kernels.gfx1151.paged_decode_splitk import choose_decode_cfg

        # Same host-side max_seq_len the gate above uses, for the same reason.
        num_splits, heads_per_cta = choose_decode_cfg(
            b,
            attn_metadata.max_seq_len,
            self.num_kv_heads,
            self.num_heads // self.num_kv_heads,
        )
        torch.ops.rocke_gfx1151.paged_decode_splitk(
            query[:b],
            key_cache,
            value_cache,
            block_table[:b],
            seq_lens[:b],
            output[:b],
            self.scale,
            num_splits,
            0,  # d_lanes -> ROCKE_PAGED_DECODE_DLANES
            0,  # block_n -> derived from d_lanes
            1,  # unroll
            heads_per_cta,
        )
        ROCKE_STATS["paged_decode"] += 1
        n = ROCKE_STATS["paged_decode"]
        if n == 1 or n % 3600 == 0:
            print(
                f"[RocKE] stats {ROCKE_STATS} decode b={b} kv={kv_mib:.0f}MiB",
                flush=True,
            )
        return True

    def _paged_v_cache(self, kv_cache: torch.Tensor):
        """The V half of the paged cache, iff the kernel may gather from it.

        Returns ``(value_cache, block_size)`` or None. Every rejection here is
        silent by design -- the caller just keeps the permute -- so each one is
        counted, because "the flag was on and nothing happened" is otherwise
        indistinguishable from "the flag was on and it worked".

        The layout this depends on is ``[num_blocks, Hk, D, block_size]``: the
        cache write already transposed V, which is the whole premise. It is
        re-derived from the tensor here rather than assumed, since a vLLM change
        to the cache shape would otherwise mis-address silently.
        """
        if _V_PAGED != "1" or kv_cache is None or kv_cache.numel() == 0:
            return None
        try:
            from vllm.v1.attention.ops.paged_attn import PagedAttention

            _, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )
        except Exception as exc:
            logger.warning("[RocKE] paged V unavailable: %s", exc)
            return None

        if value_cache.dtype != torch.float16 or not value_cache.is_contiguous():
            return None
        if value_cache.dim() != 4:
            return None
        _nb, hk, d, block_size = value_cache.shape
        if hk != self.num_kv_heads or d != self.head_size:
            return None
        # A 16-key WMMA A-fragment must not straddle two physical pages.
        if block_size % 16:
            return None
        # soffset is a 32-bit BYTE offset into the buffer descriptor, so the
        # whole V cache has to be addressable in 2 GiB. ~1 GiB/layer today.
        if value_cache.numel() * value_cache.element_size() >= 2**31:
            return None
        return value_cache, block_size

    def _forward_rocke(
        self,
        query: torch.Tensor,  # [num_tokens, Hq, D] flat
        key: torch.Tensor,  # [num_tokens, Hk, D] flat
        value: torch.Tensor,  # [num_tokens, Hk, D] flat
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, Hq, D] flat
        layout: tuple[list[int], list[int]],
    ) -> torch.Tensor:
        """Per-request swapqk dispatch over the flat token batch.

        The kernel is resolved inside the loop, not passed in: block_n depends
        on each request's own length.
        """
        q_starts, s_lens_host = layout
        s_lens = attn_metadata.seq_lens
        causal = attn_metadata.causal
        num_reqs = len(s_lens_host)
        num_actual = attn_metadata.num_actual_tokens
        paged = self._paged_v_cache(kv_cache)
        block_table = attn_metadata.block_table if paged is not None else None
        if block_table is not None and block_table.dtype != torch.int32:
            paged, block_table = None, None

        if _TRITON_MODE == "batch":
            from vllm.v1.attention.ops.triton_prefill_attention import (
                context_attention_fwd,
            )

            context_attention_fwd(
                q=query[:num_actual],
                k=key[:num_actual],
                v=value[:num_actual],
                o=output[:num_actual],
                b_start_loc=attn_metadata.query_start_loc[: num_reqs + 1],
                b_seq_len=s_lens[:num_reqs],
                max_input_len=attn_metadata.max_query_len,
                is_causal=causal,
                softmax_scale=self.scale,
            )
            ROCKE_STATS["triton_slice"] += num_reqs
            if ROCKE_STATS["triton_slice"] % 112 == 0:
                print(f"[RocKE] stats {ROCKE_STATS} batched", flush=True)
            return output[:num_actual]

        for i in range(num_reqs):
            start = q_starts[i]
            end = q_starts[i + 1]
            seqlen = end - start

            # block_n is per-request (it keys off this request's length), so the
            # kernel is resolved here rather than once in forward(). A miss here
            # means either "too short to beat Triton" or "no kv tile divides this
            # length", and the second is a correctness constraint, not a
            # preference: the kv loop truncates its tail instead of masking it.
            block_n, k_lds = _pick_strategy(seqlen)
            # The paged gather needs block_n and the page size to divide one
            # another, so each sub-tile's page is a compile-time offset.
            kv_bs = 0
            if paged is not None:
                bs = paged[1]
                if block_n and (block_n % bs == 0 or bs % block_n == 0):
                    kv_bs = bs
            req_entry = (
                _get_compiled_kernel(
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_size,
                    causal,
                    block_n,
                    k_lds,
                    kv_bs,
                )
                if block_n
                else None
            )
            if kv_bs and req_entry is None:
                # The paged build failed to compile but the contiguous one may
                # still serve this request. Do not lose the request to Triton.
                kv_bs = 0
                req_entry = (
                    _get_compiled_kernel(
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_size,
                        causal,
                        block_n,
                        k_lds,
                    )
                    if block_n
                    else None
                )

            if seqlen < _SWAPQK_MIN_SEQLEN or req_entry is None:
                # Short sequence: use Triton for this slice.
                from vllm.v1.attention.ops.triton_prefill_attention import (
                    context_attention_fwd,
                )

                local_starts = torch.tensor(
                    [0, seqlen], dtype=torch.int32, device=query.device
                )
                context_attention_fwd(
                    q=query[start:end],
                    k=key[start:end],
                    v=value[start:end],
                    o=output[start:end],
                    b_start_loc=local_starts,
                    b_seq_len=s_lens[i : i + 1],
                    max_input_len=seqlen,
                    is_causal=causal,
                    softmax_scale=self.scale,
                )
                ROCKE_STATS["triton_slice"] += 1
                if (
                    ROCKE_STATS["triton_slice"] == 1
                    or ROCKE_STATS["triton_slice"] % 112 == 0
                ):
                    print(f"[RocKE] stats {ROCKE_STATS} seqlen={seqlen}", flush=True)
                continue

            if kv_bs:
                # Row i of the block table maps this request's logical pages to
                # physical ones. The kernel's batch index is 0 (one request per
                # launch), so the ROW is handed over, not the table base.
                n_pages = (seqlen + kv_bs - 1) // kv_bs
                paged_v = (paged[0], block_table[i], n_pages)
                ROCKE_STATS["paged_v"] += 1
            else:
                paged_v = None
                if paged is not None:
                    ROCKE_STATS["paged_v_declined"] += 1

            _launch_swapqk_for_request(
                q=query[start:end].contiguous(),
                k=key[start:end].contiguous(),
                # The paged path never reads this; slicing it is free, but the
                # .contiguous() would be a copy of the whole V slice.
                v=value[start:end] if kv_bs else value[start:end].contiguous(),
                out=output[start:end],
                scale=self.scale,
                causal=causal,
                entry=req_entry,
                paged_v=paged_v,
            )
            ROCKE_STATS["swapqk"] += 1
            if ROCKE_STATS["swapqk"] == 1 or ROCKE_STATS["swapqk"] % 112 == 0:
                print(f"[RocKE] stats {ROCKE_STATS} seqlen={seqlen}", flush=True)

        # Drain this stream's swapqk launches before handing output back to vLLM.
        # Per-stream event wait, not a device-wide hipDeviceSynchronize: the
        # launches go to torch's current stream (resolve_stream(0)), so waiting
        # on that stream alone is sufficient and does not stall unrelated work.
        from rocke.runtime.launcher import wait_stream_and_release

        wait_stream_and_release(0)

        return output[:num_actual]


class RockeAttentionBackend(RocmAttentionBackend):
    """RocKE-accelerated attention backend for gfx1151 (Strix Halo iGPU)."""

    @staticmethod
    def get_name() -> str:
        return "ROCKE_GFX1151"

    @staticmethod
    def get_impl_cls() -> type[RockeAttentionImpl]:
        return RockeAttentionImpl
