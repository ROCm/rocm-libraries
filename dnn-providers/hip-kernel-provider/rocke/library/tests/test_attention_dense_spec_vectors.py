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

Two tables:

* ``cases`` run the base class, so the C++ port is checked with its base hooks.
* ``hook_cases`` run ``Gfx950AttentionDenseSpec``, which overrides every hook the
  shared spec takes from its subclass (the ``supports_*`` flags, the persist-decode
  set, and the head / shape / layout / band / algorithm / persist-decode name
  parts). The file records that class's hook answers, and the C++ test feeds them
  through the traits and name-parts structs and a validator context distinct from
  the spec. Each hook case must be judged the same by the base checks with those
  hooks and by the full gfx950 ``__post_init__`` (the generator refuses it
  otherwise), so the C++ port, which has only the base checks, owes the same
  answer.

The paged geometries where ``__post_init__`` divides by zero (``block_m < 32``, or
more waves than ``block_n`` rows) raise ``ZeroDivisionError``, whose text depends
on the CPython version. Those cases record ``null`` for both reasons, and only
the verdicts are compared.
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
from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec

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
# 32-bit and paged extent edges, the boundary on each side of every comparison,
# and valid specs that exercise every kernel-name token and both persist decodes.
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
    dict(paged=True, block_size=8, num_kv_blocks=64, sliding_window=256),
    dict(
        paged=True,
        block_size=8,
        block_m=1024,
        block_n=224,
        seqlen_q=1024,
        seqlen_kv=448,
        num_kv_blocks=64,
        sliding_window=224,
    ),
    # Division by zero: block_m < 32, and more waves than block_n rows.
    dict(paged=True, block_m=16, block_size=16),
    dict(paged=True, block_m=2048, seqlen_q=2048, block_n=32, block_size=16),
    dict(paged=True, block_size=16, num_kv_blocks=0),
    dict(paged=True, block_size=16, num_kv_blocks=1 << 20),
    dict(
        paged=True,
        block_size=16,
        num_kv_blocks=65536,
        num_kv_heads=8,
        sliding_window=256,
    ),
    dict(
        paged=True,
        block_size=16,
        num_kv_blocks=65536,
        num_kv_heads=-8,
        sliding_window=256,
    ),
    dict(paged=True, block_size=16, num_kv_blocks=64, batch=2),
    dict(paged=True, block_size=16, num_kv_blocks=64, varlen=True),
    dict(paged=True, block_size=16, num_kv_blocks=64, persistent=True),
    dict(paged=True, block_size=16, num_kv_blocks=64, head_size=64),
    dict(paged=True, block_size=16, num_kv_blocks=64),
    dict(
        use_sinks=True, paged=True, block_size=16, num_kv_blocks=64, sliding_window=256
    ),
    dict(use_sinks=True, varlen=True),
    # ---- one step either side of a validator comparison ----
    dict(num_kv_heads=7),
    dict(seqlen_q=320),
    dict(ragged=True, seqlen_kv=0),
    dict(causal_bottom_right=True),
    dict(waves_per_eu=8),
    dict(num_persistent=0),
    dict(persist_decode="qb_majorx"),
    dict(persist_decode="qb_maj"),  # string compares are exact, not prefix
    dict(dtype="bf16x"),
    dict(persist_decode="a\\b"),
    dict(persist_decode="a\nb"),
    dict(persist_decode="a\x01b"),
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
    dict(seqlen_q=131072, seqlen_kv=64, num_query_heads=128, num_kv_heads=1),
    dict(batch=1953125),
    dict(batch=2147483647, seqlen_kv=1048576, num_kv_heads=8),
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
    # The auto persist decode one CTA either side of its threshold.
    dict(persistent=True, num_persistent=32, batch=8),
    dict(persistent=True, num_persistent=33, batch=8),
    dict(ragged=True, seqlen_q=500, seqlen_kv=500, persistent=True, num_persistent=4),
)


class _RuntimeHeadsSpec(Gfx950AttentionDenseSpec):
    """gfx950 with head counts as kernel arguments, so the symbol drops ``hq``/``kv``.

    The override open PR #14059 adds to Gfx950AttentionDenseSpec. Until a shipped
    class drops the head parts, this is the only way to record a case with them
    off; once #14059 lands it can be replaced by the real class."""

    def _head_name_parts(self) -> tuple[str, ...]:
        return () if self.runtime_shape else super()._head_name_parts()


# Gfx950AttentionDenseSpec cases: each one depends on a gfx950 hook answer (a
# verdict the base hooks would give differently, or a non-default name part).
_G = Gfx950AttentionDenseSpec
_HOOK_CASES: tuple[tuple[type, dict], ...] = (
    (_G, dict(causal=False, sliding_window=128)),  # supports_noncausal_window
    (_G, dict(ragged=True, seqlen_q=300, seqlen_kv=500, causal=False)),  # cross-length
    (_G, dict(persistent=True, num_persistent=4, persist_decode="gqa_pair")),
    (_G, dict(persistent=True, num_persistent=8, persist_decode="gqa_pair_2phase")),
    (_G, dict(persist_decode="bogus")),  # the gfx950 decode set, in the reason
    (_G, dict(lds_k_row_pad=16, lds_v_row_pad=16)),  # layout parts; shape parts off
    (_G, dict(right_bound=64)),  # band part
    (_G, dict(sliding_window=256, right_bound=64)),  # band part with shape parts on
    (_G, dict(persistent=True, num_persistent=256, wide_lds_dma=True, waves_per_eu=1)),
    (_G, dict(emit_lse=True, pv_loop_order="k_major", lazy_rescale_threshold=4.0)),
    (_G, dict(batch=0)),  # preflight reason carries the gfx950 type name
    (_RuntimeHeadsSpec, {}),  # head parts off
)


def _field_names() -> list[str]:
    return [f.name for f in fields(AttentionDenseSpec)]


def _spec_fields(overrides: dict) -> dict:
    spec = AttentionDenseSpec(**_BASE)
    values = {name: getattr(spec, name) for name in _field_names()}
    values.update(overrides)
    return values


def _unchecked(cls, values: dict):
    """A ``cls`` with ``values``, without running ``__post_init__``."""
    spec = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(spec, name, value)
    return spec


def _base_verdict(spec) -> tuple[bool, str | None]:
    """``AttentionDenseSpec.__post_init__`` on ``spec``, with ``spec``'s hooks.

    ``None`` for a ZeroDivisionError: its text is interpreter-specific."""
    try:
        AttentionDenseSpec.__post_init__(spec)
        return True, ""
    except ValueError as e:
        return False, str(e)
    except ZeroDivisionError:
        return False, None


def _expect(spec) -> dict:
    """Every answer the C++ port must reproduce for ``spec``."""
    valid, reason = _base_verdict(spec)
    preflight_ok, preflight_reason = check_dense_spec_preflight(spec)
    out = {
        "valid": valid,
        "reason": reason,
        "preflight_ok": bool(preflight_ok),
        "preflight_reason": None if reason is None else preflight_reason,
    }
    if valid:
        out["kernel_name"] = spec.kernel_name()
        out["num_waves"] = spec.num_waves
        out["num_queries_per_kv"] = spec.num_queries_per_kv
    return out


def _evaluate(values: dict) -> dict:
    spec = _unchecked(AttentionDenseSpec, values)
    out = _expect(spec)
    if out["valid"]:
        out["resolved_persist_decode"] = spec.resolved_persist_decode
    return out


_NAME_PART_SLOTS = 8  # ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS
_NAME_PART_CHARS = 31  # ROCKE_ATTENTION_DENSE_NAME_PART_CAP - 1


def _hook_spec(cls: type, overrides: dict) -> Gfx950AttentionDenseSpec:
    default = cls(**_BASE)
    values = {f.name: getattr(default, f.name) for f in fields(default)}
    values.update(overrides)
    return _unchecked(cls, values)


def _hooks() -> dict:
    """The subclass answers the shared spec asks for, from the gfx950 class."""
    spec = Gfx950AttentionDenseSpec(**_BASE)
    return {
        "type_name": type(spec).__name__,
        "supports_noncausal_window": spec._supports_noncausal_window(),
        "supports_noncausal_cross_length_ragged": (
            spec._supports_noncausal_cross_length_ragged()
        ),
        "persist_decodes": sorted(spec.supported_persist_decodes()),
    }


def _name_parts(spec) -> dict:
    """The spec's name-part hooks, in rocke_attention_dense_name_parts_t's shape."""
    head = spec._head_name_parts()
    shape = spec._shape_name_parts()
    # The C struct carries these two as on/off flags over the base tokens.
    assert head in ((), AttentionDenseSpec._head_name_parts(spec)), head
    assert shape in ((), AttentionDenseSpec._shape_name_parts(spec)), shape
    parts = {
        "head_parts": bool(head),
        "shape_parts": bool(shape),
        "layout": list(spec._layout_name_parts()),
        "band": list(spec._band_name_parts()),
        "algorithm": list(spec._algorithm_name_parts()),
        "persist_decode": spec._persist_decode_name_part() if spec.persistent else "",
    }
    for key in ("layout", "band", "algorithm"):
        assert len(parts[key]) <= _NAME_PART_SLOTS, (key, parts[key])
        assert all(len(p) <= _NAME_PART_CHARS for p in parts[key]), parts[key]
    assert len(parts["persist_decode"]) <= _NAME_PART_CHARS
    return parts


def _evaluate_hook_case(cls: type, overrides: dict) -> dict:
    spec = _hook_spec(cls, overrides)
    expect = _expect(spec)
    # The C++ port has only the base checks, so it owes gfx950's verdict only where
    # the full gfx950 validator gives the same one.
    try:
        Gfx950AttentionDenseSpec.__post_init__(spec)
        full = (True, "")
    except ValueError as e:
        full = (False, str(e))
    assert full == (expect["valid"], expect["reason"]), (overrides, full, expect)
    # The C++ test names every hook case's type with the recorded gfx950 hooks.
    if cls is not Gfx950AttentionDenseSpec:
        assert expect["preflight_ok"], (cls.__name__, expect)
    case = {"spec": {name: getattr(spec, name) for name in _field_names()}}
    if expect["valid"]:
        case["name_parts"] = _name_parts(spec)
    case["expect"] = expect
    return case


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
        "hooks": _hooks(),
        "hook_cases": [_evaluate_hook_case(cls, c) for cls, c in _HOOK_CASES],
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


def test_hook_cases_match_the_current_python():
    """The gfx950 hook answers and every hook case, as the generator records them."""
    recorded = _load()
    assert recorded["hooks"] == _hooks(), _stale(
        f"recorded hooks: {recorded['hooks']}\ncurrent hooks:  {_hooks()}"
    )
    assert len(recorded["hook_cases"]) == len(_HOOK_CASES), _stale(
        f"{len(recorded['hook_cases'])} recorded hook cases, {len(_HOOK_CASES)} defined"
    )
    for i, (case, (cls, overrides)) in enumerate(
        zip(recorded["hook_cases"], _HOOK_CASES)
    ):
        got = _evaluate_hook_case(cls, overrides)
        assert got == case, _stale(f"hook case {i}\nrecorded: {case}\ncurrent:  {got}")


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
    assert len(rejects - {None}) >= 40
    assert None in rejects  # the division-by-zero paths, verdict only
    assert len(preflight) >= 4
    assert len(valid) >= 12
    decodes = {c["expect"]["resolved_persist_decode"] for c in valid}
    assert decodes == {"qb_major", "hkv_major"}


def test_hook_cases_exercise_every_hook():
    """Each subclass answer must change some recorded result, or the C++ port could
    ignore it and still pass."""
    hooks = _load()["hooks"]
    assert hooks["supports_noncausal_window"]
    assert hooks["supports_noncausal_cross_length_ragged"]
    assert set(hooks["persist_decodes"]) - {"auto", "hkv_major", "qb_major"}
    named = [c for c in _load()["hook_cases"] if "name_parts" in c]
    parts = [c["name_parts"] for c in named]
    assert any(p["layout"] for p in parts)
    assert any(p["band"] for p in parts)
    assert any(not p["shape_parts"] for p in parts)
    assert any(not p["head_parts"] for p in parts)
    assert any(len(p["algorithm"]) > 1 for p in parts)
    assert any(p["persist_decode"] not in ("", "hkvmaj") for p in parts)
    specs = [c["spec"] for c in named]
    assert any(not s["causal"] and s["sliding_window"] > 0 for s in specs)
    assert any(
        s["ragged"] and not s["causal"] and s["seqlen_q"] != s["seqlen_kv"]
        for s in specs
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--regenerate"]:
        VECTORS.write_text(json.dumps(_build(), indent=1) + "\n")
        print(f"wrote {VECTORS}")
    else:
        sys.exit("usage: test_attention_dense_spec_vectors.py --regenerate")
