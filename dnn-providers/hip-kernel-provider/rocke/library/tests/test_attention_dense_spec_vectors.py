# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Drift guard for the C++ mirror of the shared dense attention spec.

``kernels/common/attention_dense_spec.py`` has a C++ port
(``platform/cpp/instances/common/attention_dense_spec.cpp``). Its verdicts,
reject reasons, block orders and kernel names do not all reach the emitted IR,
so the byte-identity gate cannot keep the two in step. Instead, both engines are
pinned to one file, ``platform/tests/golden/attention_dense_spec_vectors.json``
(platform owns it: it is the C ABI's contract):

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

Three kinds of table:

* ``cases`` run the base class: one case per reject reason, the extent edges,
  hand-picked boundaries, then ``_GENERATED_COUNT`` specs drawn with a fixed
  seed around the auto block-order thresholds. A hand-picked table keeps
  missing the next boundary; the seeded slice is what covers the rest.
* ``hook_tables`` hold one table per shipped subclass the C++ port serves
  (``Gfx950AttentionDenseSpec``, ``Gfx942AttentionDenseSpec``). Each records the
  class's per-class answers (``supports_*`` flags, both decode sets) and, per
  case, its name parts and ``lookahead``: whether the spec sees keys ahead of its
  diagonal (gfx950 ``right_bound != 0``), which turns ``_aligned_causal`` off.
  The C++ test feeds them through the traits and name-parts structs, the
  ``lookahead`` argument, and a validator context distinct from the spec.
  Each hook case must be judged the same by the base checks with those hooks and
  by the class's full ``__post_init__`` (the generator refuses it otherwise), so
  the C++ port, which has only the base checks, owes the same answer.
  ``kernel_name`` is the shared ``AttentionDenseSpec.kernel_name`` (gfx942
  appends its own suffix in its own port).
* ``test_hook_tables_exercise_every_hook`` reverts each hook a class overrides,
  one at a time, and requires a recorded answer to change: a port that ignores
  that hook then fails the C++ test.

The paged geometries where ``__post_init__`` divides by zero (``block_m < 32``, or
more waves than ``block_n`` rows) raise ``ZeroDivisionError``, whose text depends
on the CPython version. Those cases record ``null`` for both reasons, and only
the verdicts are compared.
"""

from __future__ import annotations

import inspect
import json
import random
import sys
from dataclasses import fields
from pathlib import Path

from kernels.common.attention_dense_decode import NONPERSIST_DECODES, PERSIST_DECODES
from kernels.common.attention_dense_spec import (
    AttentionDenseSpec,
    check_dense_spec_preflight,
)
from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec

# Source tree: rocke/platform/tests/golden. Installed tree (library tests under
# tests/library/tests, platform tests under tests/): tests/golden.
_ROOT = Path(__file__).resolve().parents[2]
VECTORS = next(
    (
        p
        for p in (
            _ROOT
            / "platform"
            / "tests"
            / "golden"
            / "attention_dense_spec_vectors.json",
            _ROOT / "golden" / "attention_dense_spec_vectors.json",
        )
        if p.is_file()
    ),
    _ROOT / "platform" / "tests" / "golden" / "attention_dense_spec_vectors.json",
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
# and valid specs that exercise every persistent and non-persistent decode/name token.
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
    dict(persist_decode=None),
    dict(persist_decode="it's"),
    dict(persist_decode='x"y'),
    dict(nonpersist_decode=None),
    dict(nonpersist_decode="bogus"),
    dict(chiplet_num_xcds=0),
    dict(chiplet_num_xcds=-1),
    dict(nonpersist_decode="hq_minor_swz", num_query_heads=7, num_kv_heads=1),
    dict(persistent=True, persist_decode="hq_minor_swz", num_persistent=257),
    dict(
        persistent=True, persist_decode="hq_minor_swz", num_persistent=256, seqlen_q=768
    ),
    dict(persistent=True, persist_decode="gqa_pair", num_persistent=4, causal=False),
    dict(
        persistent=True,
        persist_decode="gqa_pair",
        num_persistent=4,
        ragged=True,
    ),
    dict(persistent=True, persist_decode="gqa_pair", num_persistent=4, seqlen_q=768),
    dict(
        persistent=True,
        persist_decode="gqa_pair",
        num_persistent=4,
        num_query_heads=6,
    ),
    dict(persistent=True, persist_decode="gqa_pair", num_persistent=8),
    dict(
        persistent=True,
        persist_decode="gqa_pair_2phase",
        num_persistent=1,
        num_query_heads=1,
        num_kv_heads=1,
    ),
    dict(persistent=True, persist_decode="gqa_pair_2phase", num_persistent=4),
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
    dict(persist_decode="gqa_pair"),  # known but inactive on the non-persistent grid
    dict(nonpersist_decode="qb_minor"),
    dict(nonpersist_decode="bt_hkv_minor"),
    dict(
        nonpersist_decode="hq_minor_swz",
        num_query_heads=8,
        num_kv_heads=8,
        chiplet_num_xcds=4,
    ),
    dict(chiplet_num_xcds=4),
    dict(
        persistent=True,
        num_persistent=256,
        persist_decode="bt_hkv_minor",
        chiplet_num_xcds=4,
    ),
    dict(
        persistent=True,
        num_persistent=256,
        persist_decode="hq_minor_swz",
        num_query_heads=8,
        num_kv_heads=8,
        seqlen_q=768,
        sliding_window=64,
    ),
    dict(persistent=True, num_persistent=4, persist_decode="gqa_pair"),
    dict(persistent=True, num_persistent=8, persist_decode="gqa_pair_2phase"),
    dict(batch=64, seqlen_q=4096, num_query_heads=8, num_kv_heads=8),
    # The auto persist decode one CTA either side of its threshold.
    dict(persistent=True, num_persistent=32, batch=8),
    dict(persistent=True, num_persistent=33, batch=8),
    dict(ragged=True, seqlen_q=500, seqlen_kv=500, persistent=True, num_persistent=4),
    # Auto-decode boundaries a one-token C change crossed unseen (review, 10-29).
    # work == 8*NP exactly: hq_minor_swz needs work > 8*NP.
    dict(
        batch=8,
        seqlen_q=4096,
        seqlen_kv=4096,
        num_query_heads=8,
        num_kv_heads=8,
        persistent=True,
        num_persistent=128,
    ),
    # A window equal to the keys still balances causal work.
    dict(sliding_window=512, persistent=True),
    # num_persistent not a multiple of chiplet_num_xcds blocks hq_minor_swz.
    dict(
        batch=4,
        seqlen_q=2048,
        seqlen_kv=64,
        num_query_heads=4,
        num_kv_heads=4,
        head_size=64,
        chiplet_num_xcds=4,
        persistent=True,
        num_persistent=1,
    ),
    # A moving bottom-right diagonal is not aligned causal.
    dict(causal_bottom_right=True, seqlen_q=256, seqlen_kv=1024),
)

_GENERATED_SEED = 2711
_GENERATED_COUNT = 400


def _near(rng: random.Random, value: int) -> int:
    """``value`` or one step either side of it: comparisons break at the edge."""
    return max(1, value + rng.choice((-1, 0, 0, 1)))


def _generated_spec(rng: random.Random) -> dict:
    """One spec drawn around the auto block-order thresholds (work vs 8*NP, per-kv
    work vs 2*NP, batch vs chiplet_num_xcds, the hq_minor_swz divisibility and
    query-block counts, window vs keys, the bottom-right diagonal)."""
    block_m = rng.choice((256, 256, 128))
    seqlen_q = block_m * rng.choice((1, 2, 3, 4, 8, 15, 16, 17, 32))
    seqlen_kv = 64 * rng.choice((1, 4, 8, 16, 32, 64))
    hkv = rng.choice((1, 2, 4, 8, 16))
    hq = hkv * rng.choice((1, 1, 2, 4))
    batch = rng.choice((1, 2, 3, 4, 7, 8, 9, 16, 32, 64))
    xcds = rng.choice((8, 8, 1, 2, 4))
    values = dict(
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=rng.choice((128, 64)),
        block_m=block_m,
        chiplet_num_xcds=xcds,
        causal=rng.random() < 0.8,
    )
    nqb = -(-seqlen_q // block_m)
    if rng.random() < 0.3:
        values["sliding_window"] = rng.choice(
            (64, max(64, seqlen_kv - 64), seqlen_kv, seqlen_kv + 64)
        )
    if values["causal"] and seqlen_q <= seqlen_kv and rng.random() < 0.2:
        values["causal_bottom_right"] = True
    if rng.random() < 0.5:
        values["persistent"] = True
        work = nqb * hq * batch
        per_hkv = (hq // hkv) * nqb * batch
        values["num_persistent"] = rng.choice(
            (
                _near(rng, work // 8),
                _near(rng, per_hkv // 2),
                xcds * rng.choice((1, 2, 16, 32)),
                _near(rng, 256),
            )
        )
        values["interleave"] = rng.random() < 0.1
        if rng.random() < 0.15:
            values["persist_decode"] = rng.choice(sorted(PERSIST_DECODES))
    elif rng.random() < 0.15:
        values["nonpersist_decode"] = rng.choice(sorted(NONPERSIST_DECODES))
    return values


def _generated_cases() -> list[dict]:
    rng = random.Random(_GENERATED_SEED)
    return [_generated_spec(rng) for _ in range(_GENERATED_COUNT)]


# Hook tables: one per shipped subclass the C++ port serves. Each case depends on
# a hook answer of its class (a verdict, block order or name part the base hooks
# would give differently), or pins a boundary of one.
_G950 = Gfx950AttentionDenseSpec
_G942 = Gfx942AttentionDenseSpec
_HOOK_CASES: dict[type, tuple[dict, ...]] = {
    _G950: (
        dict(causal=False, sliding_window=128),  # supports_noncausal_window
        dict(ragged=True, seqlen_q=300, seqlen_kv=500, causal=False),  # cross-length
        dict(persistent=True, num_persistent=4, persist_decode="gqa_pair"),
        dict(persistent=True, num_persistent=8, persist_decode="gqa_pair_2phase"),
        # A lookahead rejects the pair decodes.
        dict(
            persistent=True, num_persistent=4, persist_decode="gqa_pair", right_bound=64
        ),
        dict(persist_decode="bogus"),  # the gfx950 decode set, in the reason
        dict(lds_k_row_pad=16, lds_v_row_pad=16),  # layout parts; shape parts off
        dict(right_bound=64),  # band part
        dict(sliding_window=256, right_bound=64),  # band part with shape parts on
        # right_bound turns gfx950's _aligned_causal off, so the auto decodes fall
        # back to qb_major / qb_minor where the base hooks pick bt_hkv_minor.
        dict(
            seqlen_q=4096,
            seqlen_kv=4096,
            num_query_heads=16,
            num_kv_heads=16,
            right_bound=2,
        ),
        dict(
            persistent=True,
            seqlen_q=4096,
            seqlen_kv=4096,
            num_query_heads=16,
            num_kv_heads=16,
            right_bound=2,
        ),
        dict(persistent=True, num_persistent=256, wide_lds_dma=True, waves_per_eu=1),
        dict(emit_lse=True, pv_loop_order="k_major", lazy_rescale_threshold=4.0),
        dict(batch=0),  # preflight reason carries the gfx950 type name
        # Auto-decode boundaries (review, 10-29).
        dict(seqlen_q=512, seqlen_kv=512, sliding_window=512, persistent=True),
        dict(causal_bottom_right=True, seqlen_q=256, seqlen_kv=1024),
    ),
    _G942: (
        {},  # runtime shape: head and shape parts off
        dict(ragged=True, seqlen_q=500, seqlen_kv=500),  # baked shape: parts on
        # gfx942's persist set has no gqa_pair*: rejected where the base accepts.
        dict(persistent=True, num_persistent=4, persist_decode="gqa_pair"),
        dict(persistent=True, num_persistent=8, persist_decode="gqa_pair_2phase"),
        dict(persistent=True, num_persistent=256, persist_decode="hq_minor_swz"),
        dict(persistent=True, num_persistent=304),
        dict(nonpersist_decode="bt_hkv_minor"),
        dict(causal=False, sliding_window=128),  # no noncausal window on gfx942
        dict(batch=0),  # preflight-only reject, with gfx942's validator
    ),
}

# Every per-class or per-spec answer the shared spec takes from its subclass.
# test_hook_tables_exercise_every_hook checks each one a class overrides.
_HOOKS = (
    "_supports_noncausal_window",
    "_supports_noncausal_cross_length_ragged",
    "supported_persist_decodes",
    "supported_nonpersist_decodes",
    "_aligned_causal",
    "_head_name_parts",
    "_shape_name_parts",
    "_layout_name_parts",
    "_band_name_parts",
    "_algorithm_name_parts",
    "_persist_decode_name_part",
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
    """Every answer the C++ port must reproduce for ``spec``. ``kernel_name`` is the
    shared one: a subclass's own suffix (gfx942's) belongs to its own port."""
    valid, reason = _base_verdict(spec)
    preflight_ok, preflight_reason = check_dense_spec_preflight(spec)
    out = {
        "valid": valid,
        "reason": reason,
        "preflight_ok": bool(preflight_ok),
        "preflight_reason": None if reason is None else preflight_reason,
    }
    if valid:
        out["kernel_name"] = AttentionDenseSpec.kernel_name(spec)
        out["num_waves"] = spec.num_waves
        out["num_queries_per_kv"] = spec.num_queries_per_kv
        out["resolved_persist_decode"] = spec.resolved_persist_decode
        out["resolved_nonpersist_decode"] = spec.resolved_nonpersist_decode
    return out


def _evaluate(values: dict) -> dict:
    return _expect(_unchecked(AttentionDenseSpec, values))


_NAME_PART_SLOTS = 8  # ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS
_NAME_PART_CHARS = 31  # ROCKE_ATTENTION_DENSE_NAME_PART_CAP - 1


def _hook_spec(cls: type, overrides: dict):
    default = cls(**_BASE)
    values = {f.name: getattr(default, f.name) for f in fields(default)}
    values.update(overrides)
    return _unchecked(cls, values)


def _hooks(cls: type) -> dict:
    """The per-class answers the shared spec asks ``cls`` for."""
    spec = cls(**_BASE)
    return {
        "type_name": cls.__name__,
        "supports_noncausal_window": spec._supports_noncausal_window(),
        "supports_noncausal_cross_length_ragged": (
            spec._supports_noncausal_cross_length_ragged()
        ),
        "persist_decodes": sorted(spec.supported_persist_decodes()),
        "nonpersist_decodes": sorted(spec.supported_nonpersist_decodes()),
    }


def _lookahead(spec) -> bool:
    """The C ``lookahead`` argument: keys visible ahead of the diagonal (gfx950
    ``right_bound``). It must be exactly what turns ``_aligned_causal`` off."""
    lookahead = bool(getattr(spec, "right_bound", 0))
    base = AttentionDenseSpec._aligned_causal.fget(spec)
    assert spec._aligned_causal == (base and not lookahead), type(spec).__name__
    return lookahead


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
        "nonpersist_decode": (
            NONPERSIST_DECODES[spec.resolved_nonpersist_decode].tag
            if not spec.persistent
            else ""
        ),
    }
    for key in ("layout", "band", "algorithm"):
        assert len(parts[key]) <= _NAME_PART_SLOTS, (key, parts[key])
        assert all(len(p) <= _NAME_PART_CHARS for p in parts[key]), parts[key]
    assert len(parts["persist_decode"]) <= _NAME_PART_CHARS
    assert len(parts["nonpersist_decode"]) <= _NAME_PART_CHARS
    return parts


def _record_hook_case(spec) -> dict:
    """What the C++ test reads for one hook case."""
    expect = _expect(spec)
    case = {
        "spec": {name: getattr(spec, name) for name in _field_names()},
        "lookahead": _lookahead(spec),
    }
    if expect["valid"]:
        case["name_parts"] = _name_parts(spec)
    case["expect"] = expect
    return case


def _full_verdict(cls: type, spec) -> tuple[bool, str]:
    try:
        cls.__post_init__(spec)
        return True, ""
    except ValueError as e:
        return False, str(e)


def _evaluate_hook_case(cls: type, overrides: dict) -> dict:
    spec = _hook_spec(cls, overrides)
    case = _record_hook_case(spec)
    # The C++ port has only the base checks, so it owes the class's verdict only
    # where the class's full validator gives the same one.
    expect = case["expect"]
    full = _full_verdict(cls, spec)
    assert full == (expect["valid"], expect["reason"]), (overrides, full, expect)
    return case


def _hook_table(cls: type) -> dict:
    return {
        "hooks": _hooks(cls),
        "cases": [_evaluate_hook_case(cls, c) for c in _HOOK_CASES[cls]],
    }


def _build() -> dict:
    base_cases = list(_CASES) + _generated_cases()
    return {
        "_comment": (
            "Generated by library/tests/test_attention_dense_spec_vectors.py "
            "--regenerate. Read by that test and by the C++ test "
            "rocke_attention_dense_spec. Do not edit by hand."
        ),
        "fields": _field_names(),
        "generated": {"seed": _GENERATED_SEED, "count": _GENERATED_COUNT},
        "cases": [
            {"spec": _spec_fields(c), "expect": _evaluate(_spec_fields(c))}
            for c in base_cases
        ],
        "hook_tables": [_hook_table(cls) for cls in _HOOK_CASES],
    }


def _dump(data: dict) -> str:
    """One case per line: diffs stay readable and the file stays small."""
    lines = ["{"]
    items = list(data.items())
    for i, (key, value) in enumerate(items):
        tail = "," if i + 1 < len(items) else ""
        if key == "cases":
            rows = [json.dumps(c, separators=(",", ":")) for c in value]
            lines.append(f' "{key}": [\n  ' + ",\n  ".join(rows) + f"\n ]{tail}")
        elif key == "hook_tables":
            tables = []
            for t in value:
                rows = [json.dumps(c, separators=(",", ":")) for c in t["cases"]]
                tables.append(
                    '  {"hooks": '
                    + json.dumps(t["hooks"], separators=(",", ":"))
                    + ',\n   "cases": [\n    '
                    + ",\n    ".join(rows)
                    + "\n   ]}"
                )
            lines.append(f' "{key}": [\n' + ",\n".join(tables) + f"\n ]{tail}")
        else:
            lines.append(f' "{key}": {json.dumps(value)}{tail}')
    lines.append("}")
    return "\n".join(lines) + "\n"


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
    recorded = _load()
    assert recorded["generated"] == {
        "seed": _GENERATED_SEED,
        "count": _GENERATED_COUNT,
    }, _stale(f"recorded generator: {recorded['generated']}")
    defined = list(_CASES) + _generated_cases()
    cases = recorded["cases"]
    assert len(cases) == len(defined), _stale(
        f"{len(cases)} recorded cases, {len(defined)} defined"
    )
    for i, (case, overrides) in enumerate(zip(cases, defined)):
        assert case["spec"] == _spec_fields(overrides), _stale(f"case {i} spec moved")
        got = _evaluate(case["spec"])
        assert got == case["expect"], _stale(
            f"case {i} {case['spec']}\nrecorded: {case['expect']}\ncurrent:  {got}"
        )


def test_hook_tables_match_the_current_python():
    """Each class's hook answers and every hook case, as the generator records them."""
    tables = _load()["hook_tables"]
    assert [t["hooks"]["type_name"] for t in tables] == [
        c.__name__ for c in _HOOK_CASES
    ], _stale("hook table classes moved")
    for table, cls in zip(tables, _HOOK_CASES):
        assert table["hooks"] == _hooks(cls), _stale(
            f"recorded {cls.__name__} hooks: {table['hooks']}\n"
            f"current: {_hooks(cls)}"
        )
        assert len(table["cases"]) == len(_HOOK_CASES[cls]), _stale(
            f"{cls.__name__}: {len(table['cases'])} recorded hook cases, "
            f"{len(_HOOK_CASES[cls])} defined"
        )
        for i, (case, overrides) in enumerate(zip(table["cases"], _HOOK_CASES[cls])):
            got = _evaluate_hook_case(cls, overrides)
            assert got == case, _stale(
                f"{cls.__name__} hook case {i}\nrecorded: {case}\ncurrent:  {got}"
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
    assert len(rejects - {None}) >= 40
    assert None in rejects  # the division-by-zero paths, verdict only
    assert len(preflight) >= 4
    assert len(valid) >= 12
    persist_decodes = {c["expect"]["resolved_persist_decode"] for c in valid}
    assert persist_decodes == set(PERSIST_DECODES)
    nonpersist_decodes = {c["expect"]["resolved_nonpersist_decode"] for c in valid}
    assert nonpersist_decodes == set(NONPERSIST_DECODES)
    # The seeded slice alone must reach both answers of every auto policy, or it
    # is not exercising the thresholds it was drawn around.
    generated = [
        c
        for c in cases[len(_CASES) :]
        if c["expect"]["preflight_ok"]
        and c["spec"]["persist_decode"] == "auto"
        and c["spec"]["nonpersist_decode"] == "auto"
    ]
    for persistent, key, want in (
        (
            True,
            "resolved_persist_decode",
            {"qb_major", "hkv_major", "bt_hkv_minor", "hq_minor_swz"},
        ),
        (
            False,
            "resolved_nonpersist_decode",
            {"qb_minor", "bt_hkv_minor", "hq_minor_swz"},
        ),
    ):
        got = {
            c["expect"][key] for c in generated if c["spec"]["persistent"] == persistent
        }
        assert got == want, (key, got)


def _hook_answer(owner, spec, hook: str):
    """``hook``'s answer for ``spec``, using ``owner``'s implementation of it."""
    attr = inspect.getattr_static(owner, hook)
    if isinstance(attr, property):
        return attr.fget(spec)
    if isinstance(attr, classmethod):
        return attr.__func__(type(spec))
    return attr(spec)


def _answers(cls: type, overrides: dict) -> tuple:
    """Everything recorded for a hook case except ``lookahead``: what the C++ test
    compares once a port has taken a hook from the base class instead."""
    spec = _hook_spec(cls, overrides)
    expect = _expect(spec)
    return expect, _name_parts(spec) if expect["valid"] else None


def test_hook_tables_exercise_every_hook():
    """Every hook a class overrides must answer differently from the base on some
    recorded spec, and reverting it to the base must change a recorded answer: a
    port that ignores the hook (uses the base answer) then fails the C++ test.
    The one exemption is a per-class override that returns the base answer
    outright (gfx950's persist-decode set): there is nothing to tell apart."""
    for cls, table in _HOOK_CASES.items():
        specs = [_hook_spec(cls, c) for c in table]
        specs = [s for s in specs if _base_verdict(s)[0]]
        for hook in _HOOKS:
            mine = inspect.getattr_static(cls, hook)
            base = inspect.getattr_static(AttentionDenseSpec, hook)
            if mine is base:
                continue
            live = any(
                _hook_answer(cls, s, hook) != _hook_answer(AttentionDenseSpec, s, hook)
                for s in specs
            )
            if not live and isinstance(mine, classmethod):
                continue
            assert live, f"{cls.__name__}.{hook}: no recorded spec where it differs"
            # Same class name, so preflight reasons only move if the hook does.
            reverted = type(cls.__name__, (cls,), {hook: base})
            assert any(
                _answers(cls, c) != _answers(reverted, c) for c in table
            ), f"{cls.__name__}.{hook} differs from the base but no case pins it"


if __name__ == "__main__":
    if sys.argv[1:] == ["--regenerate"]:
        VECTORS.write_text(_dump(_build()))
        print(f"wrote {VECTORS}")
    else:
        sys.exit("usage: test_attention_dense_spec_vectors.py --regenerate")
