# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ unified-attention tiled selectors agree with Python.

Two levels:

* Every public C++ tiled-2D selector, one by one, against the Python selector of
  the same name, on gfx942, gfx950 and gfx1250 and under each env knob C++ reads.
* The assembled spec (``<arch>_attn_tiled_spec_from_problem``) against
  ``_tiled_spec_from_problem``, on every field both specs carry, and the tiled-3D
  spec (``<arch>_attn_tiled_3d_spec_from_problem``) against
  ``_tiled_3d_spec_from_problem``.

Drift that a C++ port still has to close is listed in ``_KNOWN_SPEC_GAPS``: those
fields are left out of the main comparison, only inside the cohort that drifts,
and a strict xfail per gap checks them on their own. When a port lands, its
xfail passes, strict turns that into a failure, and the gap row and marker go.
"""

from __future__ import annotations

import inspect
import itertools
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import kernels.common.attention_unified as au
import pytest
from kernels.common.attention_unified import UnifiedAttentionProblem

_ENV_PREFIXES = ("HIPDNN_GFX942_", "HIPDNN_GFX950_")


@pytest.fixture(scope="module")
def engine():
    return pytest.importorskip("rocke_engine")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(k)


def _problem(**kw) -> UnifiedAttentionProblem:
    q = kw.get("max_seqlen_q", 1)
    base = {
        "total_q": q * kw.get("num_seqs", 1),
        "num_seqs": 1,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": max(q, 2048),
        "dtype": "bf16",
    }
    base.update(kw)
    return UnifiedAttentionProblem(**base)


_BIAS = {
    "none": {},
    "alibi": {"use_alibi": True},
    "qq": {"use_qq_bias": True},
    "softcap": {"softcap": 50.0},
    "additive": {"use_additive_bias": True},
}


_HEADS = ((64, 8), (32, 8), (16, 16), (32, 4), (8, 8))


def _grid(
    heads=_HEADS,
    seqlens=(1, 128, 512, 2048),
    biases=tuple(_BIAS),
    extras=({},),
):
    for (
        dtype,
        hs,
        bs,
        (nq, nkv),
        ns,
        q,
        sw,
        sinks,
        bias,
        fp8,
        extra,
    ) in itertools.product(
        ("bf16", "fp16"),
        (64, 128, 256),
        (16, 32, 64),
        heads,
        (1, 4),
        seqlens,
        (0, 256),
        (False, True),
        biases,
        (False, True),
        extras,
    ):
        yield _problem(
            dtype=dtype,
            head_size=hs,
            block_size=bs,
            num_query_heads=nq,
            num_kv_heads=nkv,
            num_seqs=ns,
            total_q=q * ns,
            max_seqlen_q=q,
            max_seqlen_k=max(q, 2048),
            sliding_window=sw,
            use_sinks=sinks,
            use_fp8=fp8,
            **_BIAS[bias],
            **extra,
        )


# ---------------------------------------------------------------------------
# Per-selector parity
# ---------------------------------------------------------------------------

_SELECTOR_ARCHES = ("gfx942", "gfx950", "gfx1250")

# The env knobs the C++ selectors read, one setting each (default env first).
_SELECTOR_ENVS = (
    {},
    {"HIPDNN_GFX942_FLASH_MLIM": "d128"},
    {"HIPDNN_GFX942_FLASH_MLIM": "0"},
    {"HIPDNN_GFX942_K_SLICED_RING": "off"},
    {"HIPDNN_GFX942_FLASH_WIDE": "2"},
    {"HIPDNN_GFX942_K_SLICE_HD": "64"},
    {"HIPDNN_GFX942_BF16_CFVST": "0"},
    {"HIPDNN_GFX942_D128_SMALLTILE_DK": "0"},
    {"HIPDNN_GFX950_D128_SMALL_TILE": "0"},
    {"HIPDNN_GFX950_D128_SOFTMAX_INTERLEAVE": "0"},
)


def _python_selectors(keys):
    """Map each binding key to ``au._<key>`` and whether it takes ``arch``."""
    out = {}
    for key in keys:
        fn = getattr(au, "_" + key)
        positional = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        out[key] = (fn, len(positional) >= 2)
    return out


def _check_selectors(engine, problems):
    mapping = None
    bad = []
    for p in problems:
        d = asdict(p)
        for arch in _SELECTOR_ARCHES:
            got = engine.unified_attention_tiled_2d_selectors(d, arch)
            if mapping is None:
                mapping = _python_selectors(got)
            for key, (fn, takes_arch) in mapping.items():
                want = fn(p, arch) if takes_arch else fn(p)
                if isinstance(want, tuple):
                    want = tuple(want)
                if got[key] != want:
                    bad.append(f"{arch} {key}: cpp={got[key]!r} py={want!r} {p}")
    assert not bad, f"{len(bad)} selector mismatches, first ones:\n" + "\n".join(
        bad[:10]
    )


def _selector_problems():
    yield from _grid(extras=({}, {"num_kv_blocks": 1 << 20}))
    # Problem fields the selectors read beyond shape and features: a
    # waves_per_eu pin, an explicit CTA target and CU count.
    yield from _grid(
        heads=((32, 8), (16, 16)),
        seqlens=(1, 512),
        biases=("none", "softcap"),
        extras=({"waves_per_eu": 2}, {"target_ctas": 16, "num_cus": 304}),
    )


def test_selector_keys_cover_every_public_selector(engine):
    got = engine.unified_attention_tiled_2d_selectors(asdict(_problem()), "gfx950")
    # Every key resolves to a Python selector, so nothing the hook adds is
    # silently uncompared.
    assert set(_python_selectors(got)) == set(got)


def test_selectors_match_python(engine):
    _check_selectors(engine, _selector_problems())


@pytest.mark.parametrize(
    "env", _SELECTOR_ENVS[1:], ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items())
)
def test_selectors_follow_env(engine, monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    _check_selectors(
        engine,
        _grid(heads=((32, 8), (16, 16), (32, 4)), biases=("none", "alibi")),
    )


# ---------------------------------------------------------------------------
# Spec parity
# ---------------------------------------------------------------------------

# Spec fields Python has and the C++ spec does not, each with the missing
# behaviour. Once a port adds the field to the binding's spec dict, the
# field-set test fails until the entry is removed here and the field compared.
_PY_ONLY_SPEC_FIELDS = {
    "use_additive_bias": "additive bias is not in the C++ tiled-2D spec",
}

# gfx942 knobs carried by the C spec struct both arches share; the gfx950
# Python spec has no such fields.
_CPP_GFX942_ONLY_SPEC_FIELDS = frozenset(
    {
        "k_slice_hd",
        "kv_cache_policy",
        "ring_depth",
        "use_conflict_free_v",
        "use_conflict_free_v_ck_vlds",
        "use_conflict_free_v_store",
        "use_conflict_free_v_store_split",
        "use_global_load_lds_k",
        "use_causal_mask_phase_split",
        "use_iglp_opt",
        "use_k_hbm_direct",
        "use_k_sliced_ldsseq",
        "use_k_sliced_ring",
        "use_mfma_32x32x8",
        "use_q_direct_global",
        "use_q_major_grid",
        "use_qk_pv_sched_group_barrier",
        "use_v_hbm_direct",
    }
)


@dataclass(frozen=True)
class _Gap:
    behaviour: str
    fields: frozenset
    cohort: Callable[[UnifiedAttentionProblem, str], bool]


# Spec drift the C++ side still has to port. Remove a row (and its xfail
# below) when the port lands.
_KNOWN_SPEC_GAPS: dict[str, _Gap] = {}


def _norm(v):
    return list(v) if isinstance(v, tuple) else v


def _spec_pair(engine, p, arch):
    py = asdict(au._tiled_spec_from_problem(p, arch))
    cpp = engine.gfx950_attn_tiled_spec_from_problem(asdict(p), arch)
    return py, cpp


def _diff(py, cpp, fields):
    return sorted(k for k in fields if _norm(py[k]) != _norm(cpp[k]))


def _matrix_problems():
    """Hand-picked problems: the combo / single-batch combo / transposed cohorts
    and their boundaries, bias flags on each, fp8 and sinks."""

    def p(tq, ns, nq, nkv, hs, bs, q, k, **kw):
        return _problem(
            total_q=tq,
            num_seqs=ns,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_size=hs,
            block_size=bs,
            max_seqlen_q=q,
            max_seqlen_k=k,
            **kw,
        )

    return [
        # multi-batch decode / short-seqlen paths
        p(64, 64, 32, 32, 64, 32, 1, 4096),
        p(64, 64, 32, 32, 64, 32, 64, 4096),
        p(64, 64, 32, 32, 64, 32, 256, 4096),
        # combo_2d cohort (bf16 d64 b32 GQA-8 seqlen_q > 256), plain and biased
        p(512, 2, 64, 8, 64, 32, 512, 4096),
        p(1024, 4, 64, 8, 64, 32, 512, 4096),
        p(512, 2, 64, 8, 64, 32, 512, 4096, use_qq_bias=True),
        p(512, 2, 64, 8, 64, 32, 512, 4096, use_alibi=True),
        p(512, 2, 64, 8, 64, 32, 512, 4096, softcap=50.0),
        # combo_2d boundaries: seqlen_q, sliding window, dtype, head size, GQA
        p(512, 2, 64, 8, 64, 32, 256, 4096, use_qq_bias=True),
        p(512, 2, 64, 8, 64, 32, 200, 4096),
        p(512, 2, 64, 8, 64, 32, 512, 4096, sliding_window=512),
        p(512, 2, 64, 8, 64, 32, 512, 4096, dtype="fp16"),
        p(512, 2, 64, 8, 64, 32, 512, 4096, dtype="fp16", use_qq_bias=True),
        p(512, 2, 64, 8, 128, 32, 512, 4096),
        p(512, 2, 64, 8, 256, 32, 512, 4096),
        p(512, 2, 32, 4, 64, 32, 512, 4096),
        p(512, 2, 16, 4, 64, 32, 512, 4096),
        p(512, 2, 32, 32, 64, 32, 512, 4096),
        p(512, 2, 40, 8, 64, 32, 512, 4096),
        # single-batch combo cohort
        p(512, 1, 32, 32, 64, 32, 512, 4096),
        p(512, 1, 32, 32, 128, 32, 512, 4096),
        p(512, 1, 32, 32, 64, 32, 2048, 4096),
        p(512, 1, 32, 32, 128, 32, 2048, 4096),
        p(512, 1, 32, 32, 128, 64, 512, 4096),
        p(512, 1, 32, 32, 64, 32, 512, 4096, use_qq_bias=True),
        p(512, 1, 32, 32, 64, 32, 512, 4096, sliding_window=256),
        p(512, 1, 32, 32, 128, 16, 512, 4096, dtype="fp16", sliding_window=256),
        # transposed 32x32 multi-batch path
        p(512, 4, 32, 32, 64, 32, 512, 4096),
        p(512, 4, 32, 32, 128, 32, 512, 4096),
        p(512, 4, 32, 32, 64, 16, 800, 4096),
        # fp8, sinks, the i64 KV exclusion
        p(512, 2, 32, 32, 64, 32, 512, 4096, use_fp8=True),
        p(512, 2, 32, 32, 64, 32, 512, 4096, use_fp8=True, sliding_window=256),
        p(64, 64, 32, 32, 64, 32, 1, 4096, use_sinks=True),
        p(512, 2, 64, 8, 64, 32, 512, 4096, use_sinks=True),
        p(512, 2, 64, 8, 64, 32, 512, 4096, num_kv_blocks=1 << 20),
        # D256 gfx950 fast path
        p(2048, 1, 32, 8, 256, 64, 2048, 2048),
        # mask band (gfx950 only), incl. the D256 fast path
        p(512, 2, 64, 8, 64, 32, 512, 4096, causal_top_left=True),
        p(512, 4, 32, 32, 128, 32, 512, 4096, right_bound=5),
        p(512, 1, 32, 32, 64, 32, 512, 4096, causal_top_left=True, right_bound=-1),
        p(2048, 1, 32, 8, 256, 64, 2048, 2048, right_bound=-1),
    ]


def _spec_problems():
    yield from _matrix_problems()
    # The binding rejects additive bias (not in the C++ spec), so no "additive".
    yield from _grid(biases=("none", "alibi", "qq", "softcap"))


def test_spec_field_sets(engine):
    py, cpp = _spec_pair(engine, _problem(), "gfx950")
    assert set(py) - set(cpp) == set(_PY_ONLY_SPEC_FIELDS)
    assert set(cpp) - set(py) == _CPP_GFX942_ONLY_SPEC_FIELDS


def test_gfx950_spec_matches_python(engine):
    arch = "gfx950"
    bad = []
    hits = dict.fromkeys(_KNOWN_SPEC_GAPS, 0)
    for p in _spec_problems():
        py, cpp = _spec_pair(engine, p, arch)
        skip = set()
        for name, gap in _KNOWN_SPEC_GAPS.items():
            if gap.cohort(p, arch):
                hits[name] += 1
                skip |= gap.fields
        diff = _diff(py, cpp, (set(py) & set(cpp)) - skip)
        if diff:
            bad.append(f"{diff}: {p}")
    assert not bad, f"{len(bad)} spec mismatches, first ones:\n" + "\n".join(bad[:10])
    # Each gap's cohort must be exercised, or its xfail below passes vacuously.
    assert all(hits.values()), hits


@pytest.mark.parametrize(
    "gap",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=gap.behaviour
            ),
        )
        for name, gap in _KNOWN_SPEC_GAPS.items()
    ],
)
def test_known_cpp_spec_gap(engine, gap):
    arch = "gfx950"
    g = _KNOWN_SPEC_GAPS[gap]
    bad = []
    for p in _spec_problems():
        if not g.cohort(p, arch):
            continue
        py, cpp = _spec_pair(engine, p, arch)
        diff = _diff(py, cpp, g.fields & set(py) & set(cpp))
        if diff:
            bad.append(f"{diff}: {p}")
    assert not bad, f"{len(bad)} problems still drift, first ones:\n" + "\n".join(
        bad[:5]
    )


def _outcome(fn, *args):
    try:
        return fn(*args), None
    except (ValueError, RuntimeError, NotImplementedError) as e:
        return None, type(e)


def test_gfx942_spec_matches_python(engine):
    arch = "gfx942"
    bad = []
    for p in _matrix_problems():
        if au._gfx942_4warp_fast(p, arch):
            continue  # routed to the 4-warp GQA builder, which has its own spec
        py, py_err = _outcome(au._tiled_spec_from_problem, p, arch)
        cpp, cpp_err = _outcome(
            engine.gfx942_attn_tiled_spec_from_problem, asdict(p), arch
        )
        if py_err or cpp_err:
            if py_err != cpp_err:
                bad.append(f"raises {py_err} in Python, {cpp_err} in C++: {p}")
            continue
        py = asdict(py)
        diff = _diff(py, cpp, set(py) & set(cpp))
        if diff:
            bad.append(f"{diff}: {p}")
    assert not bad, f"{len(bad)} spec mismatches, first ones:\n" + "\n".join(bad[:10])


# Python tiled-3D spec fields the C++ 3D spec has no slot for, per arch.
_PY_ONLY_3D_SPEC_FIELDS = {
    "gfx942": {},
    "gfx950": {
        "use_additive_bias": "additive bias is not in the C++ tiled-3D spec",
    },
}


# The C 3D spec struct both arches share carries the gfx950 mask band.
_CPP_ONLY_3D_SPEC_FIELDS = {
    "gfx942": {"causal_top_left", "right_bound"},
    "gfx950": set(),
}


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_tiled_3d_spec_matches_python(engine, arch):
    bad = []
    for p in _matrix_problems():
        p = replace(p, max_seqlen_q=1, total_q=p.num_seqs)
        hook = getattr(engine, f"{arch}_attn_tiled_3d_spec_from_problem")
        if arch == "gfx942" and not p.default_mask:
            # Python's gfx942 spec class has no band fields (TypeError); the
            # C++ gfx942 validator rejects the band instead.
            with pytest.raises(TypeError):
                au._tiled_3d_spec_from_problem(p, arch)
            with pytest.raises(ValueError, match="gfx942 tiled attention does not"):
                hook(asdict(p), arch)
            continue
        py, py_err = _outcome(au._tiled_3d_spec_from_problem, p, arch)
        cpp, cpp_err = _outcome(hook, asdict(p), arch)
        if py_err or cpp_err:
            if py_err != cpp_err:
                bad.append(f"raises {py_err} in Python, {cpp_err} in C++: {p}")
            continue
        py = asdict(py)
        assert set(py) - set(cpp) == set(_PY_ONLY_3D_SPEC_FIELDS[arch])
        assert set(cpp) - set(py) == _CPP_ONLY_3D_SPEC_FIELDS[arch]
        diff = _diff(py, cpp, set(py) & set(cpp))
        if diff:
            bad.append(f"{diff}: {p}")
    assert not bad, "\n".join(bad[:10])
