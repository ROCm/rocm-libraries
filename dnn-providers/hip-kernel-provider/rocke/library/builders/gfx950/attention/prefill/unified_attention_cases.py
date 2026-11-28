# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paged unified-attention cases shared by the gfx950 parity harness and the
torch-free numeric tests.

:class:`Scenario` describes one case; the mask cohorts (top-left causal, no mask,
and the cuDNN-style bands) are defined here once. ``dtype`` is a rocKE name, so
this module needs no torch; the parity harness maps it to a torch dtype where it
builds tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Scenario", "band_scenarios", "nomask_scenarios", "topleft_scenarios"]


@dataclass(frozen=True)
class Scenario:
    """One paged unified-attention case: the batch's ``(query, kv)`` lengths, the
    head geometry, ``dtype`` (``"fp16"`` or ``"bf16"``), the score modifiers and
    the mask band (see the parity harness's ``ref_paged_attn``)."""

    name: str
    seq_lens: list[tuple[int, int]]
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    dtype: str
    sliding_window: int | None = None
    softcap: float = 0.0
    use_sinks: bool = False
    num_blocks: int = 32768
    use_alibi: bool = False
    use_qq_bias: bool = False
    qq_bias_stride_0: int = 0
    # Band (see ``ref_paged_attn``): diagonal alignment and right bound
    # (0 = causal, -1 = unbounded, R > 0 = lookahead); ``sliding_window`` is the left
    # bound. Defaults are the paged bottom-right causal mask; the Triton lane only
    # implements that one, so it is skipped for any other.
    causal_top_left: bool = False
    right_bound: int = 0

    @property
    def default_mask(self) -> bool:
        return not self.causal_top_left and self.right_bound == 0


def topleft_scenarios() -> list[Scenario]:
    """Unshifted (top-left) causal cohort: query p attends keys <= p per sequence.

    Mixes ``q < kv``, ``q > kv`` and ``q == kv`` sequences in one batch. The
    ``q > kv`` rows past ``kv_len`` see every key, so the in-prefix bound (not
    the causal bound) ends their KV loop.
    """
    fp16, bf16 = "fp16", "bf16"

    def tl(name, seq_lens, **kw):
        base = {
            "num_query_heads": 16,
            "num_kv_heads": 2,
            "head_size": 128,
            "block_size": 16,
            "dtype": fp16,
        }
        base.update(kw)
        return Scenario(
            name=f"tl_{name}", seq_lens=seq_lens, causal_top_left=True, **base
        )

    lt = [(256, 1024), (100, 300), (1, 700)]
    gt = [(1024, 256), (300, 100), (700, 1)]
    mixed = [(512, 512), (700, 200), (200, 700), (64, 64)]
    return [
        tl("q_lt_kv", lt),
        tl("q_gt_kv", gt),
        tl("mixed", mixed),
        tl("mixed_bf16_b64", mixed, dtype=bf16, block_size=64),
        tl(
            "gqa64x8_prefill",
            [(2048, 4096), (1500, 700)],
            num_query_heads=64,
            num_kv_heads=8,
            dtype=bf16,
        ),
        tl("d64", mixed, head_size=64, num_query_heads=8, num_kv_heads=1),
        tl(
            "d256_bf16",
            [(1024, 2048), (900, 300)],
            head_size=256,
            dtype=bf16,
            num_query_heads=64,
            num_kv_heads=8,
        ),
        tl("sinks", mixed, use_sinks=True),
        tl("softcap", mixed, softcap=30.0),
        tl("sliding_window", [(300, 1024), (500, 200), (256, 256)], sliding_window=64),
        tl("alibi", mixed, use_alibi=True),
        tl("qq_bias", mixed, use_qq_bias=True, qq_bias_stride_0=1024),
        # ALiBi + sliding window together is NOT covered: that combination already
        # yields NaN/wrong rows at bottom-right on the unmodified 2D kernel.
        tl("decode_q1", [(1, 1024), (1, 4096), (1, 33)]),
        # D256 bf16 decode cohort (``attention_d256_decode`` candidate).
        tl(
            "d256_decode_q1",
            [(1, 2048), (1, 4096), (1, 512)],
            head_size=256,
            dtype=bf16,
        ),
    ]


def nomask_scenarios() -> list[Scenario]:
    """No-mask cohort: every query attends every key (standard SDPA default).

    The unified kernels were unconditionally causal, so this also covers equal
    ``q == kv`` sequences, where the old (silently causal) kernel was wrong too.
    """
    fp16, bf16 = "fp16", "bf16"

    def nm(name, seq_lens, **kw):
        base = {
            "num_query_heads": 16,
            "num_kv_heads": 2,
            "head_size": 128,
            "block_size": 16,
            "dtype": fp16,
        }
        base.update(kw)
        return Scenario(name=f"nm_{name}", seq_lens=seq_lens, right_bound=-1, **base)

    mixed = [(512, 512), (700, 200), (200, 700), (64, 64)]
    return [
        nm("equal", [(512, 512), (257, 257), (64, 64)]),
        nm("q_lt_kv", [(256, 1024), (100, 300), (1, 700)]),
        nm("q_gt_kv", [(1024, 256), (300, 100), (700, 1)]),
        nm("mixed", mixed),
        nm("mixed_bf16_b64", mixed, dtype=bf16, block_size=64),
        nm(
            "gqa64x8_prefill",
            [(2048, 4096), (1500, 700)],
            num_query_heads=64,
            num_kv_heads=8,
            dtype=bf16,
        ),
        nm("d64", mixed, head_size=64, num_query_heads=8, num_kv_heads=1),
        nm(
            "d256_bf16",
            [(1024, 2048), (900, 300)],
            head_size=256,
            dtype=bf16,
            num_query_heads=64,
            num_kv_heads=8,
        ),
        nm("sinks", mixed, use_sinks=True),
        nm("softcap", mixed, softcap=30.0),
        nm("decode_q1", [(1, 1024), (1, 4096), (1, 33)]),
        nm(
            "d256_decode_q1",
            [(1, 2048), (1, 4096), (1, 512)],
            head_size=256,
            dtype=bf16,
        ),
    ]


def band_scenarios() -> list[Scenario]:
    """Non-causal sliding-window / lookahead bands: ``(sliding_window, right_bound)``
    around the top-left or bottom-right diagonal, cuDNN-style.

    ``right_bound=-1`` is a left-only window (the future is visible); ``R > 0`` a
    two-sided window or a lookahead (``sliding_window == 0``). Sequences mix
    ``q < kv``, ``q > kv`` and ``q == kv`` so both alignments differ, and ``q > kv``
    top-left rows past ``kv_len`` exercise empty-band rows.
    """
    fp16, bf16 = "fp16", "bf16"

    def bd(name, seq_lens, *, tl, window=0, right=-1, **kw):
        base = {
            "num_query_heads": 16,
            "num_kv_heads": 2,
            "head_size": 128,
            "block_size": 16,
            "dtype": fp16,
        }
        base.update(kw)
        align = "tl" if tl else "br"
        return Scenario(
            name=f"bd_{align}_{name}",
            seq_lens=seq_lens,
            causal_top_left=tl,
            right_bound=right,
            sliding_window=window or None,
            **base,
        )

    mixed = [(512, 512), (700, 200), (200, 700), (64, 64)]
    out: list[Scenario] = []
    for tl in (False, True):
        out += [
            bd("left_only_w64", mixed, tl=tl, window=64),
            bd("two_sided_w64_r32", mixed, tl=tl, window=64, right=32),
            bd("lookahead_r16", mixed, tl=tl, right=16),
            # right=0 is a plain causal window; the mixed lengths include q > kv, whose
            # rows past the band are empty (output 0).
            bd("causal_w100", mixed, tl=tl, window=100, right=0),
            bd("q_lt_kv_w128", [(256, 1024), (100, 300), (1, 700)], tl=tl, window=128),
            bd(
                "q_gt_kv_w128_r8",
                [(1024, 256), (300, 100), (700, 1)],
                tl=tl,
                window=128,
                right=8,
            ),
            bd(
                "bf16_b64_w96_r48",
                mixed,
                tl=tl,
                window=96,
                right=48,
                dtype=bf16,
                block_size=64,
            ),
            bd(
                "gqa64x8_w256",
                [(2048, 4096), (1500, 700)],
                tl=tl,
                window=256,
                num_query_heads=64,
                num_kv_heads=8,
                dtype=bf16,
            ),
            bd(
                "gqa64x8_r64",
                [(2048, 4096), (1500, 700)],
                tl=tl,
                right=64,
                num_query_heads=64,
                num_kv_heads=8,
                dtype=bf16,
            ),
            bd(
                "d64_w64_r32",
                mixed,
                tl=tl,
                window=64,
                right=32,
                head_size=64,
                num_query_heads=8,
                num_kv_heads=1,
            ),
            bd(
                "d256_bf16_w128_r16",
                [(1024, 2048), (900, 300)],
                tl=tl,
                window=128,
                right=16,
                head_size=256,
                dtype=bf16,
                num_query_heads=64,
                num_kv_heads=8,
            ),
            bd("sinks_w64_r32", mixed, tl=tl, window=64, right=32, use_sinks=True),
            bd("softcap_w64", mixed, tl=tl, window=64, softcap=30.0),
            bd("decode_q1_r8", [(1, 1024), (1, 4096), (1, 33)], tl=tl, right=8),
        ]
    return out
