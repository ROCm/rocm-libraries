# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The arithmetic that turns a declaration and a time into RFC 0019.13 §8.3's two metrics.

Worth testing on its own because nothing downstream can see it go wrong. A mistaken FLOP formula
or byte width does not raise -- it produces a plausible throughput, and every model trained on the
corpus ranks by a number wrong by a constant factor. So these check against hand-computed values
rather than against the implementation's own output.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from results_import.derive import (
    DTYPE_BYTES,
    UnsupportedExpression,
    derive_metrics,
    dtype_width,
    evaluate,
)

OPERATIONS = pathlib.Path(__file__).resolve().parents[2] / "corpus_gen" / "operations"


def opmeta(name: str) -> dict:
    with (OPERATIONS / f"{name}.opmeta.json").open() as handle:
        return json.load(handle)


def test_matmul_flops_is_two_mnk():
    got = evaluate(opmeta("matmul")["flops"], {"M": 1024, "N": 1024, "K": 1024})
    assert got == 2 * 1024**3


def test_conv_output_extent_follows_stride_and_padding():
    """The half of the conv formula that is easy to get wrong and impossible to notice.

    ceil_div(H + 2*pad - dil*(R-1), stride) is floor((H + 2*pad - dil*(R-1) - 1)/stride) + 1 --
    the standard extent, expressed without a floor operator. A wrong extent still yields a
    positive count of the right magnitude.
    """
    base = dict(N=1, C=64, K=64, groups=1, R=3, S=3, pad_h=1, pad_w=1,
                dilation_h=1, dilation_w=1)
    conv = opmeta("conv_fwd")["flops"]

    unit_stride = evaluate(conv, dict(base, H=56, W=56, stride_h=1, stride_w=1))
    assert unit_stride == 2 * 64 * 56 * 56 * 64 * 3 * 3

    # 224 at stride 2 with pad 1 is a 112 extent, not 111 or 113.
    strided = evaluate(conv, dict(base, H=224, W=224, stride_h=2, stride_w=2))
    assert strided == 2 * 64 * 112 * 112 * 64 * 3 * 3


def test_conv_flops_divides_channels_by_groups():
    base = dict(N=1, C=64, K=64, H=56, W=56, R=3, S=3, pad_h=1, pad_w=1,
                stride_h=1, stride_w=1, dilation_h=1, dilation_w=1)
    conv = opmeta("conv_fwd")["flops"]
    assert evaluate(conv, dict(base, groups=2)) == evaluate(conv, dict(base, groups=1)) / 2


def test_every_operation_declares_elements():
    """gbs is the metric for the memory-bound operations, so none of them may lack the count."""
    for path in OPERATIONS.glob("*.opmeta.json"):
        with path.open() as handle:
            declaration = json.load(handle)
        assert "elements" in declaration, f"{declaration['operation']} declares no elements"


def test_every_declared_dtype_has_a_width():
    """A dtype an operation admits but the table does not know makes gbs silently null."""
    for path in OPERATIONS.glob("*.opmeta.json"):
        with path.open() as handle:
            declaration = json.load(handle)
        for dtype in declaration["parameters"].get("dtype", {}).get("values", []):
            assert dtype in DTYPE_BYTES, f"no width for {dtype}"


def test_metrics_are_derived_from_the_declaration_and_the_time():
    query = dict(M=1024, N=1024, K=1024, dtype="float32")
    got = derive_metrics(query, time_ms=1.0, opmeta=opmeta("matmul"))

    assert got["tflops"] == pytest.approx(2 * 1024**3 / 1e-3 / 1e12)
    elements = 1024 * 1024 * 3
    assert got["gbs"] == pytest.approx(elements * 4 / 1e-3 / 1e9)


def test_dtype_changes_bandwidth_but_not_throughput():
    """The reason elements is declared rather than bytes: only one of the two metrics moves."""
    query = dict(M=512, N=512, K=512, dtype="float32")
    wide = derive_metrics(query, 1.0, opmeta("matmul"))
    narrow = derive_metrics(dict(query, dtype="float16"), 1.0, opmeta("matmul"))

    assert narrow["tflops"] == wide["tflops"]
    assert narrow["gbs"] == pytest.approx(wide["gbs"] / 2)


def test_a_memory_bound_operation_reports_bandwidth_and_no_throughput():
    """layernorm declares no flops on purpose; a number there would be a convention we invented."""
    got = derive_metrics(
        dict(batch=4, seq_len=128, hidden_dim=768, dtype="float16"), 1.0, opmeta("layernorm_fwd")
    )
    assert got["tflops"] is None
    assert got["gbs"] is not None


@pytest.mark.parametrize("time_ms", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_no_measurement_yields_null_never_a_winner(time_ms):
    """The direction of failure that decided null over zero.

    A zero time is impossible rather than fast, and dividing by it gives an infinity that
    outranks every real measurement. Nothing here may manufacture a value from an absent one.
    """
    got = derive_metrics(dict(M=8, N=8, K=8, dtype="float32"), time_ms, opmeta("matmul"))
    assert got == {"tflops": None, "gbs": None}


def test_an_unreadable_declaration_raises_rather_than_nulling():
    """Null means "not measured". A declaration this cannot read is a different thing, and
    reporting it the same way would hide it behind a case consumers already tolerate."""
    with pytest.raises(UnsupportedExpression):
        evaluate({"log2": ["$q.M"]}, {"M": 1024})
    with pytest.raises(UnsupportedExpression):
        evaluate("$kernel.tile_m", {"M": 1024})
    with pytest.raises(UnsupportedExpression):
        evaluate({"*": ["$q.absent", 2]}, {"M": 1024})
    with pytest.raises(UnsupportedExpression):
        dtype_width("float64")
