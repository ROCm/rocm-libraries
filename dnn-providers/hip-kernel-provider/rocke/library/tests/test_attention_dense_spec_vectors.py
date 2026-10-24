# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Drift guard for the C++ mirror of the shared dense attention spec.

``kernels/common/attention_dense_spec.py`` has a C++ port
(``platform/cpp/instances/common/attention_dense_spec.cpp``). Its verdicts,
reject reasons and kernel names do not all reach the emitted IR, so the
byte-identity gate cannot keep the two in step. Instead, both engines are pinned
to one file, ``golden/attention_dense_spec_vectors.json``:

* this test recomputes every case with the current Python and requires the
  recorded answers;
* the C++ test ``rocke_attention_dense_spec``
  (``platform/tests/instances/test_attention_dense_spec.cpp``) embeds the same
  file at build time and requires the same answers from the C++ port.

So a change to the Python spec fails here first. After an intended change:
regenerate the file (``python test_attention_dense_spec_vectors.py
--regenerate``), mirror the change in the C++ port, and the C++ test confirms
the two agree again.

Each case is built without running ``__post_init__`` so invalid specs can be
recorded too; the validator is then called explicitly, as
``check_dense_spec_preflight`` does. Pure host code -- no GPU, no compiler.

Not pinned: the paged geometries where ``__post_init__`` divides by zero
(``block_m < 32``, or more waves than ``block_n`` rows). That raises
``ZeroDivisionError``, whose text depends on the CPython version, so it cannot be
a cross-engine contract.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

from kernels.common.attention_dense_spec import (
    AttentionDenseSpec,
    check_dense_spec_preflight,
)

VECTORS = (
    Path(__file__).resolve().parent / "golden" / "attention_dense_spec_vectors.json"
)
CPP_MIRROR = "platform/cpp/instances/common/attention_dense_spec.cpp"

_BASE = dict(
    batch=1,
    seqlen_q=512,
    seqlen_kv=512,
    num_query_heads=8,
    num_kv_heads=2,
    head_size=128,
)

# One case per reject reason (each reached by the smallest change from _BASE), the
# 32-bit and paged extent edges, and valid specs that exercise every kernel-name
# token and both persist decodes.
_CASES: tuple[dict, ...] = (
    # ---- __post_init__ rejects, in validator order ----
    dict(dtype="fp32"),
    dict(dtype=None),
    dict(block_m=0),
    dict(block_m=-32),
    dict(block_n=0),
    dict(block_n=48),
    dict(head_size=96),
    dict(head_size=0),
    dict(lds_k_group_pad=4),
    dict(lds_k_group_pad=-8),
    dict(causal=False, causal_bottom_right=True),
    dict(seqlen_q=1024, causal_bottom_right=True),
    dict(ragged=True, seqlen_q=0),
    dict(ragged=True, seqlen_q=300, causal=False),
    dict(ragged=True, seqlen_q=500, varlen=True),
    dict(ragged=True, seqlen_q=500, sliding_window=64),
    dict(seqlen_q=500),
    dict(seqlen_kv=500),
    dict(seqlen_q=-256),
    dict(num_kv_heads=3),
    dict(num_kv_heads=0),
    dict(num_kv_heads=-2),
    dict(persistent=True, num_persistent=0),
    dict(persistent=True, num_persistent=-1),
    dict(persist_decode="gqa_pair"),
    dict(persist_decode=None),
    dict(persist_decode="it's"),
    dict(persist_decode='x"y'),
    dict(sliding_window=-64),
    dict(causal=False, sliding_window=64),
    dict(sliding_window=100),
    dict(varlen=True, persistent=True),
    dict(varlen=True, causal=False),
    dict(waves_per_eu=0),
    dict(waves_per_eu=9),
    dict(paged=True, block_size=0),
    dict(paged=True, block_size=48),
    dict(paged=True, block_size=128),
    dict(paged=True, block_size=4),
    dict(paged=True, block_size=16, num_kv_blocks=0),
    dict(paged=True, block_size=16, num_kv_blocks=1 << 20),
    dict(paged=True, block_size=16, num_kv_blocks=64, batch=2),
    dict(paged=True, block_size=16, num_kv_blocks=64, varlen=True),
    dict(paged=True, block_size=16, num_kv_blocks=64, persistent=True),
    dict(paged=True, block_size=16, num_kv_blocks=64, head_size=64),
    dict(paged=True, block_size=16, num_kv_blocks=64),
    dict(
        use_sinks=True, paged=True, block_size=16, num_kv_blocks=64, sliding_window=256
    ),
    dict(use_sinks=True, varlen=True),
    # ---- preflight rejects (the spec itself is valid) ----
    dict(batch=0),
    dict(batch=-1),
    dict(seqlen_kv=-512),
    dict(num_query_heads=0, num_kv_heads=-2),
    dict(block_m=128, block_n=96, seqlen_q=384, seqlen_kv=384),
    dict(seqlen_q=65536, seqlen_kv=65536, num_query_heads=128, num_kv_heads=128),
    dict(
        batch=2147483647,
        seqlen_q=2147483520,
        seqlen_kv=2147483520,
        num_query_heads=2147483647,
        num_kv_heads=2147483647,
        block_m=128,
    ),
    dict(batch=2048, seqlen_q=65536, seqlen_kv=64, num_query_heads=8, num_kv_heads=8),
    # ---- valid specs: every kernel-name token ----
    {},
    dict(seqlen_q=65536, seqlen_kv=65536, num_query_heads=127, num_kv_heads=127),
    dict(causal=False, dtype="fp16", lazy_rescale=False),
    dict(head_size=64, lds_k_group_pad=0, block_m=128),
    dict(head_size=64, block_n=32, lds_k_group_pad=16),
    dict(seqlen_q=256, seqlen_kv=1024, causal_bottom_right=True, sliding_window=256),
    dict(ragged=True, seqlen_q=500, seqlen_kv=500, use_sinks=True),
    dict(varlen=True, batch=4, sliding_window=128),
    dict(paged=True, block_size=16, num_kv_blocks=64, sliding_window=256),
    dict(persistent=True, num_persistent=304, interleave=True),
    dict(persistent=True, num_persistent=4, batch=16, seqlen_q=2048, seqlen_kv=2048),
    dict(persistent=True, num_persistent=4, persist_decode="qb_major", batch=16),
    dict(persistent=True, num_persistent=256, persist_decode="hkv_major"),
    dict(persistent=True, num_persistent=8, num_kv_heads=8, batch=64),
)


def _field_names() -> list[str]:
    return [f.name for f in fields(AttentionDenseSpec)]


def _spec_fields(overrides: dict) -> dict:
    spec = AttentionDenseSpec(**_BASE)
    values = {name: getattr(spec, name) for name in _field_names()}
    values.update(overrides)
    return values


def _unchecked(values: dict) -> AttentionDenseSpec:
    """The spec with ``values``, without running ``__post_init__``."""
    spec = object.__new__(AttentionDenseSpec)
    for name, value in values.items():
        object.__setattr__(spec, name, value)
    return spec


def _evaluate(values: dict) -> dict:
    """Every answer the C++ port must reproduce for ``values``."""
    spec = _unchecked(values)
    try:
        AttentionDenseSpec.__post_init__(spec)
        valid, reason = True, ""
    except ValueError as e:
        valid, reason = False, str(e)
    preflight_ok, preflight_reason = check_dense_spec_preflight(spec)
    out = {
        "valid": valid,
        "reason": reason,
        "preflight_ok": bool(preflight_ok),
        "preflight_reason": preflight_reason,
    }
    if valid:
        out["kernel_name"] = spec.kernel_name()
        out["resolved_persist_decode"] = spec.resolved_persist_decode
        out["num_waves"] = spec.num_waves
        out["num_queries_per_kv"] = spec.num_queries_per_kv
    return out


def _build() -> dict:
    return {
        "_comment": (
            "Generated by library/tests/test_attention_dense_spec_vectors.py "
            "--regenerate. Read by that test and by the C++ test "
            "rocke_attention_dense_spec. Do not edit by hand."
        ),
        "fields": _field_names(),
        "cases": [
            {"spec": _spec_fields(c), "expect": _evaluate(_spec_fields(c))}
            for c in _CASES
        ],
    }


def _load() -> dict:
    return json.loads(VECTORS.read_text())


_STALE = (
    "AttentionDenseSpec no longer matches {path}. If the Python change is intended: "
    "run `python {test} --regenerate`, mirror the change in the C++ port ({cpp}), "
    "and run the rocke_attention_dense_spec ctest."
)


def _stale(detail: str) -> str:
    return (
        _STALE.format(path=VECTORS.name, test=Path(__file__).name, cpp=CPP_MIRROR)
        + "\n"
        + detail
    )


def test_fields_match_the_dataclass():
    """A new or renamed spec field needs a new C++ struct field too."""
    recorded = _load()["fields"]
    assert recorded == _field_names(), _stale(
        f"recorded fields: {recorded}\ncurrent fields:  {_field_names()}"
    )


def test_every_case_matches_the_current_python():
    recorded = _load()["cases"]
    assert len(recorded) == len(_CASES), _stale(
        f"{len(recorded)} recorded cases, {len(_CASES)} defined"
    )
    for i, case in enumerate(recorded):
        got = _evaluate(case["spec"])
        assert got == case["expect"], _stale(
            f"case {i} {case['spec']}\nrecorded: {case['expect']}\ncurrent:  {got}"
        )


def test_cases_cover_every_reject_reason_and_valid_specs():
    """The C++ port is only checked where a case reaches; keep the table wide."""
    cases = _load()["cases"]
    rejects = {c["expect"]["reason"] for c in cases if not c["expect"]["valid"]}
    preflight = {
        c["expect"]["preflight_reason"]
        for c in cases
        if c["expect"]["valid"] and not c["expect"]["preflight_ok"]
    }
    valid = [c for c in cases if c["expect"]["preflight_ok"]]
    assert len(rejects) >= 40
    assert len(preflight) >= 4
    assert len(valid) >= 12
    decodes = {c["expect"]["resolved_persist_decode"] for c in valid}
    assert decodes == {"qb_major", "hkv_major"}


if __name__ == "__main__":
    if sys.argv[1:] == ["--regenerate"]:
        VECTORS.write_text(json.dumps(_build(), indent=1) + "\n")
        print(f"wrote {VECTORS}")
    else:
        sys.exit("usage: test_attention_dense_spec_vectors.py --regenerate")
