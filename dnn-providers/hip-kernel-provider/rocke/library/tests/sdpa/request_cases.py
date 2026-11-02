# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The SDPA request case table: one named ``SdpaRequest`` per case.

This is the dispatch-contract table, keyed by request; the kernel coverage
table keyed by ``SdpaCase`` is ``cases.py``. Small shapes on purpose: the table
is for dispatch verdicts and for checking kernels against
``sdpa_reference_from_request``, and every case has to run on the CPU
reference in milliseconds. Together the cases report every feature in
``SDPA_FEATURES`` at least once, including the ones every candidate declines,
so a candidate's verdict on each one is recorded rather than assumed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from dispatch.sdpa import (
    SdpaBias,
    SdpaDropout,
    SdpaFp8,
    SdpaPaged,
    SdpaRequest,
    SdpaTensor,
)

__all__ = [
    "CASES",
    "bhsd",
    "bshd",
    "make_request",
    "make_tensors",
    "packed",
    "ragged_offset",
]

ARCH = "gfx950"


def packed(dims: tuple[int, ...], dtype: str = "fp32") -> SdpaTensor:
    """A row-major (fully packed) tensor."""
    strides, acc = [], 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d
    return SdpaTensor(dtype, tuple(dims), tuple(reversed(strides)))


def bhsd(b: int, h: int, s: int, d: int, dtype: str = "fp16") -> SdpaTensor:
    return packed((b, h, s, d), dtype)


def bshd(b: int, h: int, s: int, d: int, dtype: str = "fp16") -> SdpaTensor:
    return SdpaTensor(dtype, (b, h, s, d), (s * h * d, d, h * d, 1))


def make_request(
    b: int = 2,
    h_q: int = 4,
    h_k: int = 4,
    s_q: int = 8,
    s_kv: int = 8,
    d: int = 16,
    d_v: int | None = None,
    h_v: int | None = None,
    dtype: str = "fp16",
    layout=bshd,
    **overrides,
) -> SdpaRequest:
    """A contiguous BSHD request with a baked scale; ``overrides`` replace fields."""
    d_v = d if d_v is None else d_v
    h_v = h_k if h_v is None else h_v
    base = SdpaRequest(
        arch=ARCH,
        q=layout(b, h_q, s_q, d, dtype),
        k=layout(b, h_k, s_kv, d, dtype),
        v=layout(b, h_v, s_kv, d_v, dtype),
        o=layout(b, h_q, s_q, d_v, dtype),
        scale_source="value",
        scale=d**-0.5,
    )
    return replace(base, **overrides) if overrides else base


def _scalar(dtype: str = "fp32") -> SdpaTensor:
    return packed((1,), dtype)


def _lse(b: int = 2, h: int = 4, s: int = 8) -> SdpaTensor:
    return packed((b, h, s, 1), "fp32")


def ragged_offset(batch: int, dtype: str = "int32") -> SdpaTensor:
    """The ``(B + 1, 1, 1, 1)`` offset tensor hipDNN pairs with a ragged operand."""
    return packed((batch + 1, 1, 1, 1), dtype)


def _ragged(t: SdpaTensor) -> SdpaTensor:
    return replace(
        t, ragged_offset=ragged_offset(t.dims[0]), ragged_offset_multiplier=t.strides[2]
    )


def _batch_padded(t: SdpaTensor, extra: int) -> SdpaTensor:
    """Same layout, with ``extra`` elements of padding between batches."""
    return replace(t, strides=(t.strides[0] + extra,) + t.strides[1:])


def _cases() -> dict[str, SdpaRequest]:
    base = make_request()
    seq_lens = packed((2, 1, 1, 1), "int32")
    cases = {
        "dense_bshd": base,
        "dense_bhsd": make_request(layout=bhsd),
        "bf16": make_request(dtype="bf16"),
        "gqa": make_request(h_q=8, h_k=2),
        "mqa": make_request(h_q=8, h_k=1),
        "kv_heads_differ": make_request(h_q=8, h_k=2, h_v=4),
        "d_v_differs": make_request(d=16, d_v=8),
        "dtype_mixed": replace(base, o=bshd(2, 4, 8, 16, "fp32")),
        "compute_fp16": replace(base, compute_dtype="fp16"),
        "strided": replace(
            base, q=SdpaTensor("fp16", (2, 4, 8, 16), (1024, 16, 128, 1))
        ),
        "kv_layout_differs": replace(base, k=bhsd(2, 4, 8, 16)),
        "o_layout_differs": replace(base, o=bhsd(2, 4, 8, 16)),
        "batch_stride": replace(base, q=_batch_padded(base.q, 64)),
        "d_stride": replace(
            base, v=SdpaTensor("fp16", (2, 4, 8, 16), (1024, 32, 128, 2))
        ),
        "unaligned": replace(base, q=replace(base.q, alignment=4)),
        "scale_default": replace(base, scale_source="default"),
        "scale_tensor": replace(base, scale_source="tensor", scale_tensor=_scalar()),
        "causal_tl": replace(base, right_bound=0),
        "causal_br_square": replace(base, right_bound=0, diagonal="bottom_right"),
        "causal_br": make_request(
            s_q=4, s_kv=8, right_bound=0, diagonal="bottom_right"
        ),
        "window_causal": replace(base, left_bound=3, right_bound=0),
        "window_br": make_request(
            s_q=4, s_kv=8, left_bound=3, right_bound=0, diagonal="bottom_right"
        ),
        "window_right": replace(base, left_bound=2, right_bound=1),
        "window_left_only": replace(base, left_bound=2),
        "bias_full": replace(base, bias=SdpaBias(packed((2, 4, 8, 8)))),
        "bias_broadcast": replace(base, bias=SdpaBias(packed((8, 8)))),
        "alibi": replace(base, alibi=True),
        "padded": replace(base, seq_len_q=seq_lens, seq_len_kv=seq_lens),
        "padding_mask_only": replace(base, padding_mask=True),
        "ragged": replace(
            base,
            q=_ragged(base.q),
            k=_ragged(base.k),
            v=_ragged(base.v),
            o=_ragged(base.o),
        ),
        "dropout": replace(
            base,
            dropout=SdpaDropout(0.1, seed=_scalar("int64"), offset=_scalar("int64")),
        ),
        "paged": replace(
            base,
            paged=SdpaPaged(
                packed((2, 1, 4, 1), "int32"), packed((2, 1, 4, 1), "int32"), 8
            ),
        ),
        "block_mask": replace(base, block_mask=packed((2, 4, 1, 1), "uint8")),
        "sinks": replace(base, sink_token=packed((1, 4, 1, 1), "fp32")),
        "fp8": make_request(
            dtype="fp8_e4m3",
            o=bshd(2, 4, 8, 16, "bf16"),
            fp8=SdpaFp8(descale_q=_scalar(), descale_k=_scalar(), descale_v=_scalar()),
        ),
        "fp8_s_o": make_request(
            dtype="fp8_e4m3",
            fp8=SdpaFp8(
                descale_q=_scalar(),
                descale_k=_scalar(),
                descale_v=_scalar(),
                descale_s=_scalar(),
                scale_s=_scalar(),
                scale_o=_scalar(),
                amax_s=_scalar(),
                amax_o=_scalar(),
            ),
        ),
        "lse": replace(base, stats=_lse()),
        "causal_br_lse": make_request(
            s_q=4,
            s_kv=8,
            right_bound=0,
            diagonal="bottom_right",
            stats=_lse(s=4),
        ),
        "max_sum_exp": replace(base, max=_lse(), sum_exp=_lse()),
        "mma_core_mode": replace(base, mma_core_mode="fp32"),
        "implementation_unified": replace(base, implementation="unified"),
        "implementation_composite": replace(base, implementation="composite"),
    }
    return cases


CASES: dict[str, SdpaRequest] = _cases()


def _pack_ragged(x: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Logical ``(B, H, S_max, D)`` -> flat packed BSHD holding each batch's first rows."""
    return np.concatenate(
        [x[b, :, :n].transpose(1, 0, 2).reshape(-1) for b, n in enumerate(lengths)]
    )


def make_tensors(req: SdpaRequest, seed: int = 0) -> dict[str, np.ndarray]:
    """Random inputs for ``sdpa_reference(req, ...)``, keyed as it expects.

    Dense operands are logical float32 arrays. Ragged Q/K/V are packed buffers
    with token-unit offsets (``ragged_offset_multiplier`` is the sequence
    stride, as in hipDNN's golden bundles); K and V share K's offsets.
    """
    rng = np.random.default_rng(seed)
    b = req.batch
    tensors: dict[str, np.ndarray] = {}
    lengths = {
        "q": rng.integers(1, req.s_q + 1, b),
        "k": rng.integers(1, req.s_kv + 1, b),
    }
    for name in ("q", "k", "v"):
        t = getattr(req, name)
        x = rng.standard_normal(t.dims).astype(np.float32)
        if not t.ragged:
            tensors[name] = x
            continue
        n = lengths["q" if name == "q" else "k"]
        tensors[name] = _pack_ragged(x, n)
        tensors[f"ragged_offset_{name}"] = np.concatenate([[0], np.cumsum(n)])
    if req.seq_len_q is not None:
        tensors["seq_len_q"] = lengths["q"]
    if req.seq_len_kv is not None:
        tensors["seq_len_kv"] = lengths["k"]
    if req.scale_tensor is not None:
        tensors["scale"] = np.full(req.scale_tensor.dims, 0.25, np.float32)
    if req.bias is not None:
        tensors["bias"] = rng.standard_normal(req.bias.tensor.dims).astype(np.float32)
    if req.fp8 is not None:
        for name in ("descale_q", "descale_k", "descale_v"):
            t = getattr(req.fp8, name)
            if t is not None:
                tensors[name] = rng.uniform(0.5, 1.5, t.dims).astype(np.float32)
    return tensors
