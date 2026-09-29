# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fixed inference-only SDPA case corpus + independent CPU oracle (gfx1151 harness).

This module is pure NumPy (``ml_dtypes`` imported lazily, only where an
actual bf16/fp8 array is materialized) so it can be imported and exercised
without a HIP device, a build toolchain, or torch. It owns exactly four
public surfaces the benchmark runner (owned by the parent) composes with the
native rocKE / AOTriton launchers:

- ``SdpaCase``   -- one frozen, hashable row of the fixed synthetic suite.
- ``CASES``      -- the immutable tuple of every row (the suite).
- ``make_inputs``-- deterministic NumPy/ml_dtypes input generation for a case.
- ``reference``  -- an independent FP64-internal / FP32-output CPU oracle.

plus ``suite_hash``, a stable identity digest over every case field, so the
runner/report can detect a suite drift between a correctness run and a later
performance run.

--------------------------------------------------------------------------
Reference semantics (cite your sources before you change this section)
--------------------------------------------------------------------------
The math below, and its *order of operations*, is a line-for-line NumPy port
of ``ref_paged_attn`` in
``library/builders/gfx950/attention/prefill/parity_unified_attention.py``
(docstring there: "Reference paged attention. Matches AITER's reference plus
ALiBi/QQ-bias.") -- the exact oracle the gfx950 CK DSL ``unified_attention``
kernels are validated against in this repo -- generalized here to also cover
a plain ``mask="none"`` (no causal mask at all) case. The published pipeline
order (``library/builders/gfx950/attention/ALGORITHM.md`` Sec. 1, and mirrored
by ``library/kernels/common/attention_unified.py``'s ``UnifiedAttentionProblem``
flags) is, entirely in the natural (non-log2) score domain:

  1. scale:    ``s = tau * Q @ K^T``                       (``tau = 1/sqrt(D)``)
  2. softcap:  ``s = softcap * tanh(s / softcap)``          (optional)
  3. ALiBi:    ``s[h, q, k] += alibi_slope[h] * (k - ctx)``  (optional; NOT a
     function of ``q`` -- same additive column for every query row)
  4. QQ-bias:  ``s[q, k] += qq_bias[q, k - ctx]`` when
     ``0 <= k - ctx < qq_bias.shape[1]`` and ``q < qq_bias.shape[0]``, else 0
     (optional; broadcast across heads, per ``ref_paged_attn``)
  5. causal:   ``s[q, k] = -inf`` for ``k > q + ctx``
  6. window:   also ``s[q, k] = -inf`` for ``k <= q + ctx - window``
     (optional; requires a causal mask, matching
     ``AttentionDenseSpec.__post_init__``: ``sliding_window>0 requires causal``)
  7. sinks:    append one extra per-head logit column (no value row) before
     the softmax row-reduction, then drop that column from the weights
     (optional; mirrors the ``use_sinks`` "softmax(concat([QK, sink]))[...,:-1]"
     formula in ``library/tests/test_attention_dense_gfx950_numeric.py``)
  8. softmax, then ``O = P @ V``.

``ctx`` ("context length") is the bottom-right causal offset:
``ctx = 0`` for ``mask="causal_topleft"`` (plain prefill self-attention,
``seqlen_q == seqlen_k``), ``ctx = seqlen_k - seqlen_q`` for
``mask="causal_bottomright"`` (decode / chunked-prefill against a longer KV
context -- the offset formula matches
``library/tests/test_attention_dense_gfx950_numeric.py::_bottom_right_reference``:
``diagonal = 0 if top_left else skv - sq``), and masking steps 5-6 are
skipped entirely for ``mask="none"``. Per-sequence lengths (ragged/paged) use
each sequence's own ``(seqlen_q, seqlen_k)`` for ``ctx``.

A fully-masked row (an empty KV sequence, i.e. ``seqlen_k == 0`` for that row)
returns exactly zero, never NaN -- the same guard the production kernels use
("softmax denominator 0 => output 0", see
``library/builders/gfx1151/attention/ALGORITHM.md`` Sec. 6: "guarding the
empty-row case l=0"). Sinks make a row's softmax denominator strictly
positive (the sink logit is always a valid, unmasked column), so a
sink-enabled row is never all-zero even if every real key is masked; this
oracle mirrors that by construction (sinks are concatenated before the
"denominator is zero" guard is evaluated).

FP8 KV (``kv_dtype="fp8e4m3"``): the byte format is the OCP finite/NaN-only
e4m3 layout (no infinities), matching this repo's own terminology in
``library/kernels/common/fmha_fwd_fp8.py`` ("interpreted as OCP fp8 (e4m3fn /
e5m2), which matches gfx950 / gfx11 hardware decode"); RDNA3.5 (gfx1151) is
in the same non-fnuz decode family as gfx950, so the default (``fp8_fnuz`` not
part of this harness's scope) OCP decode is the correct one here. Dequant is
``value = fp8_byte.astype(f64) * scale`` for both K and V (a pure per-tensor
affine dequant -- mathematically identical whether the K-side scale is folded
into the softmax ``scale`` before the QK matmul, as
``library/kernels/common/fmha_fwd_fp8.py`` does on-device, or applied to K
directly, as this oracle does; commutativity of scalar multiplication makes
the two forms bit-for-bit equivalent in FP64).

--------------------------------------------------------------------------
Feature-row -> gfx950 evidence map (every ``group`` used in ``CASES`` below)
--------------------------------------------------------------------------
See ``GROUP_EVIDENCE`` for the exact citation per group. In summary: dense
D64/D128/D256, MHA/GQA/MQA, causal top-left & bottom-right, sliding window,
softcap, attention sinks, FP8 KV, and paged batching with a shuffled
(non-identity) block table are all fields on gfx950's
``UnifiedAttentionProblem`` / ``AttentionDenseSpec`` and are exercised by
gfx950's own benchmark/test suite (see per-group citations). ALiBi and
QQ-bias are declared fields on the same problem type and have a validated
CPU reference (``ref_paged_attn`` above) and a working gfx950 kernel path
(the transposed-32x32 "combo" 2D kernel -- see
``library/kernels/common/attention_unified.py`` comment: "ALiBi / QQ bias ARE
admitted: the transposed softmax body applies them per-score"), even though
the *scalar* CK DSL 2D backend currently rejects them
(``supports_native_unified_attention``: "ALiBi slopes are not enabled in CK
DSL attention yet" / "QQ bias is not enabled in CK DSL attention yet") --
included per the contract's "do not drop features just because current
gfx1151 lacks them"; whether gfx1151 or the comparator can run a given row is
a runner-time support/backend decision, not a reason to omit the row here.

--------------------------------------------------------------------------
CaseInputs layout
--------------------------------------------------------------------------
- dense:  q [B,Sq,Hq,D], k/v [B,Sk,Hkv,D].
- ragged: q packed [total_q,Hq,D], k/v packed [total_k,Hkv,D],
  ``cu_seqlens_q``/``cu_seqlens_k`` are int32 prefix sums (length
  num_seqs+1). Self-attention or cross-length (bottom-right) chunked-prefill,
  per-sequence lengths taken from ``case.q_lengths``/``case.k_lengths``.
- paged: q packed [total_q,Hq,D] (``cu_seqlens_q`` prefix sums), KV cache
  [num_pages,block_size,Hkv,D], ``seqused_k`` int32 per-sequence KV length,
  ``block_table`` int32 [num_seqs,max_blocks_per_seq] -- a shuffled
  (non-identity) permutation of physical page ids, built from a padded pool
  so no sequence's physical pages are contiguous either (see
  ``_build_paged_kv``).

All arrays are C-contiguous. Dense duplication is avoided: dense cases never
also carry a ragged/paged encoding of the same data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "SdpaCase",
    "CaseInputs",
    "CASES",
    "make_inputs",
    "reference",
    "suite_hash",
    "GROUP_EVIDENCE",
]

_DTYPES = ("fp16", "bf16")
_KV_DTYPES = ("", "fp8e4m3")
_LAYOUTS = ("dense", "paged", "ragged")
_MASKS = ("none", "causal_topleft", "causal_bottomright")

# Per-tensor FP8 dequant scales used by every fp8 KV case. Deliberately not
# 1.0: exercises the affine dequant path (see module docstring), not just an
# identity cast. Values keep quantized magnitudes well inside e4m3's +-448
# finite range for unit-variance source data.
_FP8_K_SCALE = 0.5
_FP8_V_SCALE = 0.25
_FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Feature-row -> gfx950 evidence (contract: "explicit mapping ... to existing
# gfx950 support"). Keyed by SdpaCase.group.
# ---------------------------------------------------------------------------
GROUP_EVIDENCE = {
    "dense": (
        "AttentionDenseSpec (library/kernels/common/attention_dense_spec.py): "
        "head_size in {64,128}; UnifiedAttentionProblem head_size in "
        "{64,128,256} (attention_unified.py UNIFIED_HEAD_SIZES); dtype in "
        "{fp16,bf16} (UNIFIED_DTYPES)."
    ),
    "gqa_mqa": (
        "AttentionDenseSpec.num_query_heads/num_kv_heads with the "
        "'num_query_heads % num_kv_heads == 0' GQA/MQA constraint "
        "(attention_dense_spec.py __post_init__)."
    ),
    "decode": (
        "UnifiedAttentionProblem decode regime (seqlen_q=1, long seqlen_k); "
        "library/builders/gfx950/attention/prefill/parity_unified_attention.py "
        "decode scenarios. These are ordinary equal-head-dimension SDPA cases."
    ),
    "boundary": (
        "AttentionDenseSpec.causal_bottom_right + ragged relaxation "
        "(attention_dense_spec.py __post_init__: 'ragged is self-attention "
        "only ... unless causal_bottom_right'); "
        "test_attention_dense_bottom_right.py covers unaligned/edge lengths."
    ),
    "bottom_right": (
        "AttentionDenseSpec.causal_bottom_right + "
        "test_attention_dense_gfx950_numeric.py::_bottom_right_reference "
        "(diagonal = 0 if top_left else skv - sq)."
    ),
    "window": (
        "AttentionDenseSpec.sliding_window / UnifiedAttentionProblem."
        "sliding_window; gpt-oss SWA-128 shapes in "
        "library/benchmarks/gfx950/attention/{decode,prefill}/gpt_oss_sink_*"
        ".json (window_size=[127,0])."
    ),
    "softcap": (
        "UnifiedAttentionProblem.softcap + apply_softcap_log2 "
        "(library/kernels/common/attention_unified.py, "
        "platform/python/rocke/helpers/attention.py); "
        "supports_native_unified_attention: 'softcap: yes'."
    ),
    "sinks": (
        "UnifiedAttentionProblem.use_sinks; gfx950 sink numeric oracle in "
        "test_attention_dense_gfx950_numeric.py::_sink_reference and "
        "gpt_oss_sink_{shapes,prefill_shapes}.json."
    ),
    "alibi": (
        "UnifiedAttentionProblem.use_alibi; formula 'S += alibi_slope[h] * "
        "(key_pos - context_len)' in "
        "library/builders/gfx950/attention/prefill/parity_unified_attention.py"
        "::ref_paged_attn; combo-path support noted in attention_unified.py "
        "('ALiBi / QQ bias ARE admitted' on the transposed 32x32 path)."
    ),
    "qq_bias": (
        "UnifiedAttentionProblem.use_qq_bias; formula 'S += qq_bias[q_local, "
        "k_local-context_len]' in parity_unified_attention.py::ref_paged_attn; "
        "qq_bias_prefill_d128_b16 scenario in the same file."
    ),
    "fp8_kv": (
        "UnifiedAttentionProblem.use_fp8 and kv_storage_dtype='fp8e4m3' in "
        "library/kernels/gfx950/attention_tiled_2d.py and attention_tiled_3d.py. "
        "FP8 KV storage is distinct from full FP8 arithmetic or absorbed attention."
    ),
    "paged": (
        "UnifiedAttentionProblem block_size in {16,32,64} "
        "(UNIFIED_BLOCK_SIZES) and block_tables_ptr paged-KV addressing in "
        "library/kernels/common/attention_unified.py."
    ),
    "ragged": (
        "AttentionDenseSpec.ragged + cu_seqlens_q/cu_seqlens_k packed varlen "
        "batching (attention_dense_spec.py __post_init__ ragged branch); "
        "alibi_mixed_d128_b16 scenario (heterogeneous per-sequence lengths) "
        "in parity_unified_attention.py."
    ),
    "combo": (
        "Feature-composition coverage explicitly called out by "
        "library/builders/gfx950/attention/README.md's combo scenarios "
        "(e.g. combo_bf16_d64_b32_gqa8_*) and attention_unified.py's "
        "_bias_active / mask_opts gating, which special-cases exactly these "
        "layered combinations."
    ),
}


def _seed_for(name: str) -> int:
    """Stable per-case seed derived from the case name (order-independent,
    collision-free in practice), so adding/reordering cases never perturbs an
    existing case's random stream."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass(frozen=True)
class SdpaCase:
    """One frozen, hashable row of the fixed synthetic SDPA suite.

    ``batch`` is the literal dense batch dimension for ``layout="dense"``, and
    the sequence count (``num_seqs``) for ``layout in {"paged","ragged"}``
    (required to equal ``len(q_lengths)``/``len(k_lengths)`` whenever those
    tuples are non-empty). ``seqlen_q``/``seqlen_k`` are the literal per-batch
    lengths for dense rows, and the *uniform* per-sequence length used only
    when ``q_lengths``/``k_lengths`` are empty for paged/ragged rows.
    """

    name: str
    group: str
    dtype: str  # 'fp16' | 'bf16'
    batch: int
    seqlen_q: int
    seqlen_k: int
    heads_q: int
    heads_kv: int
    head_dim: int
    layout: str = "dense"  # 'dense' | 'paged' | 'ragged'
    mask: str = "causal_topleft"  # 'none' | 'causal_topleft' | 'causal_bottomright'
    block_size: int = 0  # paged page size; 0 elsewhere
    window: int = 0  # sliding window width; 0 = disabled
    softcap: float = 0.0  # 0 = disabled
    sinks: bool = False
    alibi: bool = False
    qq_bias: bool = False
    kv_dtype: str = ""  # '' = same as Q; 'fp8e4m3'
    q_lengths: Tuple[int, ...] = ()  # empty = uniform seqlen_q per sequence
    k_lengths: Tuple[int, ...] = ()  # empty = uniform seqlen_k per sequence
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"{self.name}: dtype must be one of {_DTYPES}")
        if self.kv_dtype not in _KV_DTYPES:
            raise ValueError(f"{self.name}: kv_dtype must be one of {_KV_DTYPES}")
        if self.layout not in _LAYOUTS:
            raise ValueError(f"{self.name}: layout must be one of {_LAYOUTS}")
        if self.mask not in _MASKS:
            raise ValueError(f"{self.name}: mask must be one of {_MASKS}")
        if self.heads_kv <= 0 or self.heads_q % self.heads_kv != 0:
            raise ValueError(
                f"{self.name}: heads_q ({self.heads_q}) must be a positive "
                f"multiple of heads_kv ({self.heads_kv})"
            )
        if self.head_dim <= 0:
            raise ValueError(f"{self.name}: head_dim must be positive")
        if self.batch <= 0:
            raise ValueError(f"{self.name}: batch must be positive")
        if self.window < 0:
            raise ValueError(f"{self.name}: window must be >= 0")
        if self.window > 0 and self.mask == "none":
            raise ValueError(f"{self.name}: window>0 requires a causal mask")
        if self.softcap < 0:
            raise ValueError(f"{self.name}: softcap must be >= 0")
        if (
            self.q_lengths
            and self.k_lengths
            and len(self.q_lengths) != len(self.k_lengths)
        ):
            raise ValueError(f"{self.name}: q_lengths/k_lengths length mismatch")
        if self.q_lengths and len(self.q_lengths) != self.batch:
            raise ValueError(f"{self.name}: len(q_lengths) must equal batch")
        if self.k_lengths and len(self.k_lengths) != self.batch:
            raise ValueError(f"{self.name}: len(k_lengths) must equal batch")
        if self.layout == "paged":
            if self.block_size <= 0:
                raise ValueError(f"{self.name}: paged layout requires block_size>0")
            if self.block_size & (self.block_size - 1):
                raise ValueError(
                    f"{self.name}: paged block_size must be a power of two"
                )

    @property
    def scale(self) -> float:
        """Softmax scale ``tau = 1/sqrt(head_dim)``."""
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def atol(self) -> float:
        """Absolute tolerance vs. this FP32 CPU oracle, matching the gfx950
        numeric harness's per-dtype bound (rtol=0.0): 2e-2 for fp16, 4e-2 for
        bf16 (``library/tests/test_attention_dense_gfx950_numeric.py::
        _tolerance``)."""
        return 2e-2 if self.dtype == "fp16" else 4e-2


@dataclass
class CaseInputs:
    """Generated inputs for one :class:`SdpaCase`. See the module docstring
    ("CaseInputs layout") for the exact per-layout array shapes."""

    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    cu_seqlens_q: Optional[np.ndarray] = None
    cu_seqlens_k: Optional[np.ndarray] = None
    seqused_k: Optional[np.ndarray] = None
    block_table: Optional[np.ndarray] = None
    sinks: Optional[np.ndarray] = None
    alibi_slopes: Optional[np.ndarray] = None
    qq_bias: Optional[np.ndarray] = None
    k_scale: Optional[np.ndarray] = None
    v_scale: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# dtype helpers (ml_dtypes imported lazily -- module docstring / contract:
# "optional-at-function-use ml_dtypes")
# ---------------------------------------------------------------------------
def _activation_np_dtype(name: str):
    if name == "fp16":
        return np.float16
    if name == "bf16":
        import ml_dtypes

        return ml_dtypes.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def _fp8_np_dtype():
    import ml_dtypes

    return ml_dtypes.float8_e4m3fn


def _quantize_fp8(x_f64: np.ndarray, scale: np.ndarray) -> np.ndarray:
    clipped = np.clip(x_f64 / scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return clipped.astype(_fp8_np_dtype())


def _dequant_kv(
    arr: np.ndarray, kv_dtype: str, scale: Optional[np.ndarray]
) -> np.ndarray:
    x = arr.astype(np.float64)
    if kv_dtype == "fp8e4m3":
        return x * scale
    return x


def _alibi_slopes(n_heads: int) -> np.ndarray:
    """Standard ALiBi geometric slope schedule (Press et al. 2021), extended
    to non-power-of-two head counts via the closest-power-of-two
    interpolation used by common open-source ALiBi implementations. Purely
    synthetic test input -- only the downstream ``S += slope*(k-ctx)`` math
    is oracled against production semantics, not this exact slope table."""

    def _pow2_slopes(n: int) -> np.ndarray:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        return start ** np.arange(1, n + 1, dtype=np.float64)

    if float(math.log2(n_heads)).is_integer():
        return _pow2_slopes(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    base = _pow2_slopes(closest)
    extra = _pow2_slopes(2 * closest)[0::2][: n_heads - closest]
    return np.concatenate([base, extra])


def _prefix_sums(lengths) -> np.ndarray:
    cu = np.zeros(len(lengths) + 1, dtype=np.int32)
    cu[1:] = np.cumsum(np.asarray(lengths, dtype=np.int64))
    return cu


def _resolved_lengths(case: SdpaCase):
    q_lens = list(case.q_lengths) if case.q_lengths else [case.seqlen_q] * case.batch
    k_lens = list(case.k_lengths) if case.k_lengths else [case.seqlen_k] * case.batch
    if len(q_lens) != len(k_lens):
        raise ValueError(f"{case.name}: resolved q/k length count mismatch")
    return q_lens, k_lens


def _build_paged_kv(rng, k_lens, block_size: int, hkv: int, d: int):
    """Build a shuffled (non-identity) block table plus its backing K/V page
    caches. Physical pages are drawn from a permutation of a pool padded past
    the number of pages actually needed, so a sequence's own pages are neither
    contiguous nor in logical order, and cross-sequence page ids interleave."""
    blocks_needed = [0 if kl == 0 else -(-kl // block_size) for kl in k_lens]
    max_blocks = max(max(blocks_needed), 1)
    total_blocks = sum(blocks_needed)
    pool_size = max(total_blocks + max(4, total_blocks // 4), 1)
    perm = rng.permutation(pool_size)
    if pool_size > 1 and np.array_equal(perm, np.arange(pool_size)):
        perm = np.roll(perm, 1)
    block_table = np.zeros((len(k_lens), max_blocks), dtype=np.int32)
    cursor = 0
    for i, nb in enumerate(blocks_needed):
        if nb == 0:
            continue
        block_table[i, :nb] = perm[cursor : cursor + nb]
        cursor += nb
    k_cache64 = rng.standard_normal((pool_size, block_size, hkv, d))
    v_cache64 = rng.standard_normal((pool_size, block_size, hkv, d))
    return block_table, k_cache64, v_cache64


def _gather_paged(
    cache64: np.ndarray, block_table_row: np.ndarray, kv_len: int, block_size: int
):
    if kv_len == 0:
        hkv, d = cache64.shape[2], cache64.shape[3]
        return np.zeros((0, hkv, d), dtype=np.float64)
    nb = -(-kv_len // block_size)
    phys = block_table_row[:nb]
    chunk = cache64[phys]  # [nb, block_size, Hkv, D]
    flat = chunk.reshape(-1, chunk.shape[2], chunk.shape[3])
    return flat[:kv_len]


def _gen_aux(case: SdpaCase, rng, q_lens, k_lens, act_dtype):
    """Draw sinks / alibi_slopes / qq_bias in a fixed order (after Q/K/V),
    conditioned only on which flags are set, so enabling one feature never
    perturbs another feature's random stream."""
    sinks = alibi_slopes = qq_bias = None
    if case.sinks:
        sinks = np.ascontiguousarray(
            (rng.standard_normal(case.heads_q) * 2.0).astype(act_dtype)
        )
    if case.alibi:
        alibi_slopes = np.ascontiguousarray(
            _alibi_slopes(case.heads_q).astype(np.float32)
        )
    if case.qq_bias:
        # Shared across all sequences in the batch, sized off the *shortest*
        # sequence: whenever lengths are heterogeneous (e.g. the ragged
        # ALiBi/QQ-bias rows below) this deliberately makes the bias tensor
        # smaller than at least one sequence's (q_len, k_len), exercising the
        # "outside qq_bias bounds contributes 0" clamp in-suite rather than
        # needing a dedicated case.
        n0, n1 = max(min(q_lens), 1), max(min(k_lens), 1)
        qq_bias = np.ascontiguousarray(
            (rng.standard_normal((n0, n1)) * 0.5).astype(np.float32)
        )
    return sinks, alibi_slopes, qq_bias


def make_inputs(case: SdpaCase) -> CaseInputs:
    """Deterministic NumPy/ml_dtypes input generation for ``case``.

    Draw order (fixed, so every array is reproducible independent of which
    optional features are enabled): Q, then K, then V, then paged block-table
    permutation (paged only), then sinks/alibi/qq_bias. Q/K/V are always
    drawn from independent slices of the RNG stream -- never the same
    values -- because each is a separate ``rng.standard_normal`` call.
    """
    rng = np.random.default_rng(case.seed)
    act = _activation_np_dtype(case.dtype)

    if case.layout == "dense":
        b, sq, sk = case.batch, case.seqlen_q, case.seqlen_k
        hq, hkv, d = case.heads_q, case.heads_kv, case.head_dim
        q64 = rng.standard_normal((b, sq, hq, d))
        k64 = rng.standard_normal((b, sk, hkv, d))
        v64 = rng.standard_normal((b, sk, hkv, d))
        q = np.ascontiguousarray(q64.astype(act))
        if case.kv_dtype == "fp8e4m3":
            k_scale = np.asarray(_FP8_K_SCALE, dtype=np.float32)
            v_scale = np.asarray(_FP8_V_SCALE, dtype=np.float32)
            k = np.ascontiguousarray(_quantize_fp8(k64, k_scale))
            v = np.ascontiguousarray(_quantize_fp8(v64, v_scale))
        else:
            k_scale = v_scale = None
            k = np.ascontiguousarray(k64.astype(act))
            v = np.ascontiguousarray(v64.astype(act))
        sinks, alibi_slopes, qq_bias = _gen_aux(case, rng, [sq] * b, [sk] * b, act)
        return CaseInputs(
            q=q,
            k=k,
            v=v,
            sinks=sinks,
            alibi_slopes=alibi_slopes,
            qq_bias=qq_bias,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    if case.layout == "ragged":
        q_lens, k_lens = _resolved_lengths(case)
        hq, hkv, d = case.heads_q, case.heads_kv, case.head_dim
        total_q, total_k = sum(q_lens), sum(k_lens)
        q64 = rng.standard_normal((total_q, hq, d))
        k64 = rng.standard_normal((total_k, hkv, d))
        v64 = rng.standard_normal((total_k, hkv, d))
        q = np.ascontiguousarray(q64.astype(act))
        if case.kv_dtype == "fp8e4m3":
            k_scale = np.asarray(_FP8_K_SCALE, dtype=np.float32)
            v_scale = np.asarray(_FP8_V_SCALE, dtype=np.float32)
            k = np.ascontiguousarray(_quantize_fp8(k64, k_scale))
            v = np.ascontiguousarray(_quantize_fp8(v64, v_scale))
        else:
            k_scale = v_scale = None
            k = np.ascontiguousarray(k64.astype(act))
            v = np.ascontiguousarray(v64.astype(act))
        sinks, alibi_slopes, qq_bias = _gen_aux(case, rng, q_lens, k_lens, act)
        return CaseInputs(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=_prefix_sums(q_lens),
            cu_seqlens_k=_prefix_sums(k_lens),
            sinks=sinks,
            alibi_slopes=alibi_slopes,
            qq_bias=qq_bias,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    if case.layout == "paged":
        q_lens, k_lens = _resolved_lengths(case)
        hq, hkv, d, bs = case.heads_q, case.heads_kv, case.head_dim, case.block_size
        total_q = sum(q_lens)
        q64 = rng.standard_normal((total_q, hq, d))
        q = np.ascontiguousarray(q64.astype(act))
        block_table, k_cache64, v_cache64 = _build_paged_kv(rng, k_lens, bs, hkv, d)
        if case.kv_dtype == "fp8e4m3":
            k_scale = np.asarray(_FP8_K_SCALE, dtype=np.float32)
            v_scale = np.asarray(_FP8_V_SCALE, dtype=np.float32)
            k_cache = np.ascontiguousarray(_quantize_fp8(k_cache64, k_scale))
            v_cache = np.ascontiguousarray(_quantize_fp8(v_cache64, v_scale))
        else:
            k_scale = v_scale = None
            k_cache = np.ascontiguousarray(k_cache64.astype(act))
            v_cache = np.ascontiguousarray(v_cache64.astype(act))
        sinks, alibi_slopes, qq_bias = _gen_aux(case, rng, q_lens, k_lens, act)
        return CaseInputs(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=_prefix_sums(q_lens),
            seqused_k=np.asarray(k_lens, dtype=np.int32),
            block_table=block_table,
            sinks=sinks,
            alibi_slopes=alibi_slopes,
            qq_bias=qq_bias,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    raise ValueError(f"{case.name}: unknown layout {case.layout!r}")


# ---------------------------------------------------------------------------
# Reference (CPU oracle)
# ---------------------------------------------------------------------------
def _mask_params(mask: str, sq: int, sk: int):
    if mask == "none":
        return 0, False
    if mask == "causal_topleft":
        return 0, True
    if mask == "causal_bottomright":
        return sk - sq, True
    raise ValueError(f"unknown mask {mask!r}")


def _seq_reference(
    q64: np.ndarray,
    k64: np.ndarray,
    v64: np.ndarray,
    *,
    scale: float,
    ctx: int,
    window: int,
    softcap: float,
    sinks_h: Optional[np.ndarray],
    alibi_slopes: Optional[np.ndarray],
    qq_bias: Optional[np.ndarray],
    apply_causal: bool,
) -> np.ndarray:
    """FP64 attention for one (sequence, all-heads) slice.
    ``q64``: [Sq,Hq,D]; ``k64``/``v64``: [Sk,Hkv,D]. Returns [Sq,Hq,D] FP64.
    Implements steps 1-8 of the module-docstring pipeline, in order."""
    sq, hq, d = q64.shape
    sk, hkv, _ = k64.shape
    if sk == 0:
        # Empty KV: nothing to attend to at all -- safe zero (no matmul to
        # even form; avoids a 0-sized einsum edge case).
        return np.zeros((sq, hq, d), dtype=np.float64)
    if hq != hkv:
        rep = hq // hkv
        k64 = np.repeat(k64, rep, axis=1)
        v64 = np.repeat(v64, rep, axis=1)

    attn = np.einsum("qhd,khd->hqk", q64 * scale, k64)  # [Hq,Sq,Sk], step 1

    if softcap > 0:
        attn = softcap * np.tanh(attn / softcap)  # step 2

    if alibi_slopes is not None:
        pos = np.arange(sk, dtype=np.float64) - ctx
        attn = attn + alibi_slopes.reshape(hq, 1, 1) * pos.reshape(1, 1, sk)  # step 3

    if qq_bias is not None:
        n0, n1 = qq_bias.shape
        qb = np.zeros((sq, sk), dtype=np.float64)
        krp = np.arange(sk) - ctx
        valid_k = (krp >= 0) & (krp < n1)
        valid_q = min(sq, n0)
        if valid_q > 0 and valid_k.any():
            qb[np.ix_(np.arange(valid_q), valid_k)] = qq_bias[
                np.ix_(np.arange(valid_q), krp[valid_k])
            ]
        attn = attn + qb.reshape(1, sq, sk)  # step 4

    if apply_causal:
        qi = np.arange(sq).reshape(sq, 1)
        ki = np.arange(sk).reshape(1, sk)
        masked = ki > (qi + ctx)
        if window > 0:
            masked = masked | (ki <= (qi + ctx - window))
        attn = np.where(masked.reshape(1, sq, sk), -np.inf, attn)  # steps 5-6

    if sinks_h is not None:
        sink_col = np.broadcast_to(sinks_h.reshape(hq, 1, 1), (hq, sq, 1))
        attn = np.concatenate([attn, sink_col.astype(np.float64)], axis=-1)  # step 7

    row_max = np.max(attn, axis=-1, keepdims=True)
    row_max = np.where(np.isneginf(row_max), 0.0, row_max)  # fully-masked guard
    p = np.exp(attn - row_max)
    denom = np.sum(p, axis=-1, keepdims=True)
    safe_denom = np.where(denom == 0, 1.0, denom)
    p = p / safe_denom  # step 8 (softmax); denom==0 => p==0 => output 0

    if sinks_h is not None:
        p = p[..., :-1]

    return np.einsum("hqk,khd->qhd", p, v64)


def _reference_dense(case: SdpaCase, inputs: CaseInputs) -> np.ndarray:
    b, sq, hq, d = inputs.q.shape
    sk = inputs.k.shape[1]
    ctx, apply_causal = _mask_params(case.mask, sq, sk)
    q64 = inputs.q.astype(np.float64)
    k64 = _dequant_kv(inputs.k, case.kv_dtype, inputs.k_scale)
    v64 = _dequant_kv(inputs.v, case.kv_dtype, inputs.v_scale)
    sinks_h = inputs.sinks.astype(np.float64) if inputs.sinks is not None else None
    alibi = (
        inputs.alibi_slopes.astype(np.float64)
        if inputs.alibi_slopes is not None
        else None
    )
    qqb = inputs.qq_bias.astype(np.float64) if inputs.qq_bias is not None else None
    out = np.empty((b, sq, hq, d), dtype=np.float64)
    for i in range(b):
        out[i] = _seq_reference(
            q64[i],
            k64[i],
            v64[i],
            scale=case.scale,
            ctx=ctx,
            window=case.window,
            softcap=case.softcap,
            sinks_h=sinks_h,
            alibi_slopes=alibi,
            qq_bias=qqb,
            apply_causal=apply_causal,
        )
    return out.astype(np.float32)


def _reference_ragged(case: SdpaCase, inputs: CaseInputs) -> np.ndarray:
    cu_q, cu_k = inputs.cu_seqlens_q, inputs.cu_seqlens_k
    num_seqs = len(cu_q) - 1
    q64 = inputs.q.astype(np.float64)
    k64 = _dequant_kv(inputs.k, case.kv_dtype, inputs.k_scale)
    v64 = _dequant_kv(inputs.v, case.kv_dtype, inputs.v_scale)
    sinks_h = inputs.sinks.astype(np.float64) if inputs.sinks is not None else None
    alibi = (
        inputs.alibi_slopes.astype(np.float64)
        if inputs.alibi_slopes is not None
        else None
    )
    qqb = inputs.qq_bias.astype(np.float64) if inputs.qq_bias is not None else None
    out = np.zeros(inputs.q.shape, dtype=np.float64)
    for i in range(num_seqs):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        sq, sk = qe - qs, ke - ks
        ctx, apply_causal = _mask_params(case.mask, sq, sk)
        out[qs:qe] = _seq_reference(
            q64[qs:qe],
            k64[ks:ke],
            v64[ks:ke],
            scale=case.scale,
            ctx=ctx,
            window=case.window,
            softcap=case.softcap,
            sinks_h=sinks_h,
            alibi_slopes=alibi,
            qq_bias=qqb,
            apply_causal=apply_causal,
        )
    return out.astype(np.float32)


def _reference_paged(case: SdpaCase, inputs: CaseInputs) -> np.ndarray:
    cu_q = inputs.cu_seqlens_q
    seqused_k = inputs.seqused_k
    block_table = inputs.block_table
    bs = case.block_size
    num_seqs = len(cu_q) - 1
    q64 = inputs.q.astype(np.float64)
    k_cache64 = _dequant_kv(inputs.k, case.kv_dtype, inputs.k_scale)
    v_cache64 = _dequant_kv(inputs.v, case.kv_dtype, inputs.v_scale)
    sinks_h = inputs.sinks.astype(np.float64) if inputs.sinks is not None else None
    alibi = (
        inputs.alibi_slopes.astype(np.float64)
        if inputs.alibi_slopes is not None
        else None
    )
    qqb = inputs.qq_bias.astype(np.float64) if inputs.qq_bias is not None else None
    out = np.zeros(inputs.q.shape, dtype=np.float64)
    for i in range(num_seqs):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        sq = qe - qs
        sk = int(seqused_k[i])
        ctx, apply_causal = _mask_params(case.mask, sq, sk)
        k_seq = _gather_paged(k_cache64, block_table[i], sk, bs)
        v_seq = _gather_paged(v_cache64, block_table[i], sk, bs)
        out[qs:qe] = _seq_reference(
            q64[qs:qe],
            k_seq,
            v_seq,
            scale=case.scale,
            ctx=ctx,
            window=case.window,
            softcap=case.softcap,
            sinks_h=sinks_h,
            alibi_slopes=alibi,
            qq_bias=qqb,
            apply_causal=apply_causal,
        )
    return out.astype(np.float32)


def reference(case: SdpaCase, inputs: CaseInputs) -> np.ndarray:
    """Independent CPU oracle: FP64-internal, FP32-output, same shape as
    ``inputs.q``. See the module docstring for the exact operation order and
    per-mask/layout ``ctx`` derivation."""
    if case.layout == "dense":
        return _reference_dense(case, inputs)
    if case.layout == "ragged":
        return _reference_ragged(case, inputs)
    if case.layout == "paged":
        return _reference_paged(case, inputs)
    raise ValueError(f"{case.name}: unknown layout {case.layout!r}")


# ---------------------------------------------------------------------------
# The fixed case corpus
# ---------------------------------------------------------------------------
def _case(name: str, group: str, **kw) -> SdpaCase:
    return SdpaCase(name=name, group=group, seed=_seed_for(name), **kw)


CASES: Tuple[SdpaCase, ...] = (
    # -- dense: core dtype x head_dim coverage (mask=causal_topleft, Sq==Sk) --
    _case(
        "dense_f16_d64_mha_s256",
        "dense",
        dtype="fp16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
    ),
    _case(
        "dense_bf16_d64_mha_s256",
        "dense",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
    ),
    _case(
        "dense_f16_d128_mha_s256",
        "dense",
        dtype="fp16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=128,
    ),
    _case(
        "dense_bf16_d128_mha_s256",
        "dense",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=128,
    ),
    _case(
        "dense_f16_d128_mha_s512",
        "dense",
        dtype="fp16",
        batch=1,
        seqlen_q=512,
        seqlen_k=512,
        heads_q=8,
        heads_kv=8,
        head_dim=128,
    ),
    _case(
        "dense_bf16_d256_mha_s256",
        "dense",
        dtype="bf16",
        batch=1,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=4,
        heads_kv=4,
        head_dim=256,
    ),
    _case(
        "dense_f16_d64_noncausal_rect",
        "dense",
        dtype="fp16",
        batch=2,
        seqlen_q=128,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        mask="none",
    ),
    _case(
        "dense_f16_d256_mha_s256",
        "dense",
        dtype="fp16",
        batch=1,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=4,
        heads_kv=4,
        head_dim=256,
    ),
    _case(
        "dense_bf16_d64_noncausal_s256",
        "dense",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        mask="none",
    ),
    _case(
        "dense_f16_d128_noncausal_s256",
        "dense",
        dtype="fp16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=128,
        mask="none",
    ),
    _case(
        "dense_bf16_d128_noncausal_s256",
        "dense",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=128,
        mask="none",
    ),
    _case(
        "dense_f16_d256_noncausal_s256",
        "dense",
        dtype="fp16",
        batch=1,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=4,
        heads_kv=4,
        head_dim=256,
        mask="none",
    ),
    _case(
        "dense_bf16_d256_noncausal_s256",
        "dense",
        dtype="bf16",
        batch=1,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=4,
        heads_kv=4,
        head_dim=256,
        mask="none",
    ),
    _case(
        "dense_f16_d64_topleft_rect",
        "dense",
        dtype="fp16",
        batch=2,
        seqlen_q=128,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
    ),
    # -- gqa_mqa: query/kv head ratio coverage --
    _case(
        "gqa_f16_d64_hq64_hkv8_s256",
        "gqa_mqa",
        dtype="fp16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
    ),
    _case(
        "gqa_bf16_d128_hq32_hkv8_s256",
        "gqa_mqa",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=32,
        heads_kv=8,
        head_dim=128,
    ),
    _case(
        "mqa_f16_d128_hq32_hkv1_s256",
        "gqa_mqa",
        dtype="fp16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=32,
        heads_kv=1,
        head_dim=128,
    ),
    # -- decode: dense decode / chunked-prefill (bottom-right, Sq<=Sk) --
    _case(
        "decode_f16_d128_hq16_hkv2_sk2048",
        "decode",
        dtype="fp16",
        batch=4,
        seqlen_q=1,
        seqlen_k=2048,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        mask="causal_bottomright",
    ),
    _case(
        "decode_bf16_d256_hq16_hkv2_sk2048",
        "decode",
        dtype="bf16",
        batch=2,
        seqlen_q=1,
        seqlen_k=2048,
        heads_q=16,
        heads_kv=2,
        head_dim=256,
        mask="causal_bottomright",
    ),
    _case(
        "chunked_prefill_f16_d64_sq128_sk512",
        "bottom_right",
        dtype="fp16",
        batch=2,
        seqlen_q=128,
        seqlen_k=512,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        mask="causal_bottomright",
    ),
    _case(
        "bottomright_f16_d64_masked_prefix",
        "bottom_right",
        dtype="fp16",
        batch=1,
        seqlen_q=64,
        seqlen_k=32,
        heads_q=8,
        heads_kv=4,
        head_dim=64,
        mask="causal_bottomright",
    ),
    _case(
        "bottomright_bf16_d64_masked_prefix",
        "bottom_right",
        dtype="bf16",
        batch=1,
        seqlen_q=64,
        seqlen_k=32,
        heads_q=8,
        heads_kv=4,
        head_dim=64,
        mask="causal_bottomright",
    ),
    # -- window: sliding-window prefill + decode (gpt-oss SWA-128 geometry) --
    _case(
        "window_f16_d64_gptoss_prefill_s512",
        "window",
        dtype="fp16",
        batch=1,
        seqlen_q=512,
        seqlen_k=512,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        mask="causal_topleft",
        window=128,
    ),
    _case(
        "window_bf16_d64_gptoss_decode_kv2048",
        "window",
        dtype="bf16",
        batch=4,
        seqlen_q=1,
        seqlen_k=2048,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        mask="causal_bottomright",
        window=128,
    ),
    # -- softcap --
    _case(
        "softcap_f16_d128_mha_s256",
        "softcap",
        dtype="fp16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=128,
        softcap=30.0,
    ),
    _case(
        "softcap_bf16_d64_gqa_s256",
        "softcap",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=32,
        heads_kv=8,
        head_dim=64,
        softcap=50.0,
    ),
    # -- sinks (gpt-oss gpt_oss_sink_shapes.json geometry: Hq64/Hkv8/D64) --
    _case(
        "sinks_bf16_d64_decode_full_kv2048",
        "sinks",
        dtype="bf16",
        batch=4,
        seqlen_q=1,
        seqlen_k=2048,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        mask="causal_bottomright",
        sinks=True,
    ),
    _case(
        "sinks_bf16_d64_decode_swa_kv2048",
        "sinks",
        dtype="bf16",
        batch=4,
        seqlen_q=1,
        seqlen_k=2048,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        mask="causal_bottomright",
        window=128,
        sinks=True,
    ),
    _case(
        "sinks_f16_d64_prefill_full_s512",
        "sinks",
        dtype="fp16",
        batch=1,
        seqlen_q=512,
        seqlen_k=512,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        mask="causal_topleft",
        sinks=True,
    ),
    _case(
        "sinks_f16_d64_prefill_swa_s512",
        "sinks",
        dtype="fp16",
        batch=1,
        seqlen_q=512,
        seqlen_k=512,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        mask="causal_topleft",
        window=128,
        sinks=True,
    ),
    # -- alibi --
    _case(
        "alibi_f16_d128_hq16_hkv2_decode",
        "alibi",
        dtype="fp16",
        batch=4,
        seqlen_q=1,
        seqlen_k=2048,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        mask="causal_bottomright",
        alibi=True,
    ),
    _case(
        "alibi_f16_d128_mixed_lengths_ragged",
        "alibi",
        dtype="fp16",
        batch=3,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        layout="ragged",
        mask="causal_bottomright",
        alibi=True,
        q_lengths=(1, 5, 129),
        k_lengths=(1328, 18, 463),
    ),
    # -- qq_bias (parity_unified_attention.py qq_bias_prefill_d128_b16) --
    _case(
        "qqbias_f16_d128_prefill_ragged",
        "qq_bias",
        dtype="fp16",
        batch=3,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        layout="ragged",
        mask="causal_bottomright",
        qq_bias=True,
        q_lengths=(64, 128, 32),
        k_lengths=(64, 256, 256),
    ),
    _case(
        "qqbias_f16_d128_dense",
        "qq_bias",
        dtype="fp16",
        batch=1,
        seqlen_q=128,
        seqlen_k=128,
        heads_q=8,
        heads_kv=2,
        head_dim=128,
        qq_bias=True,
    ),
    # -- fp8 KV --
    _case(
        "fp8kv_bf16_d128_mha_prefill_s256",
        "fp8_kv",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        kv_dtype="fp8e4m3",
    ),
    _case(
        "fp8kv_bf16_d128_gqa_decode_sk4096",
        "fp8_kv",
        dtype="bf16",
        batch=4,
        seqlen_q=1,
        seqlen_k=4096,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        mask="causal_bottomright",
        kv_dtype="fp8e4m3",
    ),
    # -- paged: page-size sweep (16/32/64) + multi-seq + decode + fp8 --
    _case(
        "paged_prefill_bf16_d64_bs16_singleseq",
        "paged",
        dtype="bf16",
        batch=1,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        layout="paged",
        mask="causal_bottomright",
        block_size=16,
        q_lengths=(512,),
        k_lengths=(512,),
    ),
    _case(
        "paged_prefill_f16_d128_bs32_multiseq",
        "paged",
        dtype="fp16",
        batch=3,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        layout="paged",
        mask="causal_bottomright",
        block_size=32,
        q_lengths=(64, 128, 32),
        k_lengths=(64, 256, 256),
    ),
    _case(
        "paged_decode_bf16_d64_bs64_mixed",
        "paged",
        dtype="bf16",
        batch=4,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        layout="paged",
        mask="causal_bottomright",
        block_size=64,
        q_lengths=(1, 1, 1, 1),
        k_lengths=(2048, 2048, 8192, 2048),
    ),
    _case(
        "paged_decode_sinks_swa_bf16_d64_bs16",
        "paged",
        dtype="bf16",
        batch=3,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        layout="paged",
        mask="causal_bottomright",
        block_size=16,
        window=128,
        sinks=True,
        q_lengths=(1, 1, 1),
        k_lengths=(2048, 2048, 8192),
    ),
    _case(
        "paged_decode_fully_masked_row",
        "paged",
        dtype="bf16",
        batch=2,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        layout="paged",
        mask="causal_bottomright",
        block_size=16,
        q_lengths=(1, 1),
        k_lengths=(0, 64),
    ),
    _case(
        "paged_fp8_decode_d128_mixed",
        "fp8_kv",
        dtype="fp16",
        batch=2,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        layout="paged",
        mask="causal_bottomright",
        block_size=16,
        kv_dtype="fp8e4m3",
        q_lengths=(1, 1),
        k_lengths=(2048, 4096),
    ),
    # -- ragged: self-attention packing + chunked-prefill + fully-masked row --
    _case(
        "ragged_selfattn_f16_d64_uniform",
        "ragged",
        dtype="fp16",
        batch=3,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        layout="ragged",
        mask="causal_topleft",
        q_lengths=(128, 128, 128),
        k_lengths=(128, 128, 128),
    ),
    _case(
        "ragged_selfattn_bf16_d128_variable_len",
        "ragged",
        dtype="bf16",
        batch=3,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=16,
        heads_kv=2,
        head_dim=128,
        layout="ragged",
        mask="causal_topleft",
        q_lengths=(37, 200, 89),
        k_lengths=(37, 200, 89),
    ),
    _case(
        "ragged_chunked_prefill_bottomright",
        "ragged",
        dtype="fp16",
        batch=2,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        layout="ragged",
        mask="causal_bottomright",
        q_lengths=(64, 32),
        k_lengths=(320, 160),
    ),
    _case(
        "ragged_fully_masked_zero_kv",
        "ragged",
        dtype="fp16",
        batch=2,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        layout="ragged",
        mask="causal_bottomright",
        q_lengths=(5, 3),
        k_lengths=(0, 10),
    ),
    # -- boundary: non-aligned / degenerate shapes --
    _case(
        "boundary_prime_len_dense_f16_d64",
        "boundary",
        dtype="fp16",
        batch=1,
        seqlen_q=137,
        seqlen_k=137,
        heads_q=4,
        heads_kv=4,
        head_dim=64,
        mask="causal_topleft",
    ),
    _case(
        "boundary_rect_tail_nomask_f16_d64",
        "boundary",
        dtype="fp16",
        batch=2,
        seqlen_q=100,
        seqlen_k=260,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        mask="none",
    ),
    _case(
        "boundary_single_token_decode_bf16_d64",
        "boundary",
        dtype="bf16",
        batch=8,
        seqlen_q=1,
        seqlen_k=1,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        mask="causal_bottomright",
    ),
    _case(
        "boundary_wide_batch_mqa_f16_d64",
        "boundary",
        dtype="fp16",
        batch=32,
        seqlen_q=32,
        seqlen_k=32,
        heads_q=8,
        heads_kv=1,
        head_dim=64,
        mask="causal_topleft",
    ),
    # -- combo: layered feature compositions --
    _case(
        "combo_softcap_alibi_bf16_d64_s256",
        "combo",
        dtype="bf16",
        batch=2,
        seqlen_q=256,
        seqlen_k=256,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        softcap=30.0,
        alibi=True,
    ),
    _case(
        "combo_window_qqbias_ragged_f16_d64",
        "combo",
        dtype="fp16",
        batch=2,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=8,
        heads_kv=8,
        head_dim=64,
        layout="ragged",
        mask="causal_bottomright",
        window=64,
        qq_bias=True,
        q_lengths=(96, 64),
        k_lengths=(256, 192),
    ),
    _case(
        "combo_fp8_gqa_window_bf16_d128_s512",
        "combo",
        dtype="bf16",
        batch=2,
        seqlen_q=512,
        seqlen_k=512,
        heads_q=32,
        heads_kv=8,
        head_dim=128,
        mask="causal_topleft",
        window=256,
        kv_dtype="fp8e4m3",
    ),
    _case(
        "combo_paged_sinks_alibi_bf16_d64_decode",
        "combo",
        dtype="bf16",
        batch=2,
        seqlen_q=0,
        seqlen_k=0,
        heads_q=64,
        heads_kv=8,
        head_dim=64,
        layout="paged",
        mask="causal_bottomright",
        block_size=32,
        sinks=True,
        alibi=True,
        q_lengths=(1, 1),
        k_lengths=(2048, 2048),
    ),
)


def suite_hash() -> str:
    """Hash every field of every frozen case, including ordering and seeds."""
    payload = json.dumps(
        [asdict(case) for case in CASES], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
