# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import re
from pathlib import Path

import pytest

from hipdnn_vllm.declines import Decline, DeclineError, Stage, parse_op_error, parse_reason

CSRC = Path(__file__).resolve().parents[2] / "csrc"


def test_reason_strings():
    assert Decline(Stage.NO_ENGINE).reason == "NO_ENGINE"
    assert Decline.feature("softcap").reason == "UNSUPPORTED_FEATURE:softcap"
    assert Decline(Stage.DISABLED).reason == "DISABLED"


@pytest.mark.parametrize("stage", list(Stage))
def test_round_trip(stage):
    d = Decline(stage, "fp8_kv") if stage is Stage.UNSUPPORTED_FEATURE else Decline(stage)
    assert parse_reason(d.reason) == d


def test_feature_needs_detail():
    with pytest.raises(ValueError):
        Decline(Stage.UNSUPPORTED_FEATURE)


@pytest.mark.parametrize("detail", ["Sinks", "a b", "x:y", "é"])
def test_detail_charset(detail):
    with pytest.raises(ValueError):
        Decline(Stage.UNSUPPORTED_FEATURE, detail)


@pytest.mark.parametrize("bad", ["", "NOPE", "no_engine", "FALLBACK:x"])
def test_parse_rejects_unknown(bad):
    with pytest.raises(ValueError):
        parse_reason(bad)


def test_cpp_taxonomy_matches_python():
    """csrc/decline.h must name exactly the Python stages."""
    text = (CSRC / "decline.h").read_text()
    enum_body = re.search(r"enum class Stage\s*\{(.*?)\};", text, re.S).group(1)
    cpp = {tok.strip().rstrip(",") for tok in enum_body.split() if tok.strip().rstrip(",")}
    assert cpp == {s.value for s in Stage}
    for s in Stage:
        assert f'return "{s.value}";' in text


def test_parse_op_error():
    exc = RuntimeError(
        "hipdnn_vllm decline [rms_norm]: NO_ENGINE :: no engine applicable\n"
        "Exception raised from ... (most recent call first)"
    )
    op, d = parse_op_error(exc)
    assert op == "rms_norm"
    assert d.stage is Stage.NO_ENGINE
    assert d.message.startswith("no engine applicable")
    op, d = parse_op_error(RuntimeError("hipdnn_vllm decline [rms_norm]: UNSUPPORTED_FEATURE:strided_rows :: "))
    assert d == Decline.feature("strided_rows")
    assert parse_op_error(RuntimeError("CUDA error: out of memory")) is None


def test_decline_error_message():
    e = DeclineError("attention.mixed", Decline.feature("paged", "no paged engine"))
    assert "attention.mixed" in str(e) and "UNSUPPORTED_FEATURE:paged" in str(e)
