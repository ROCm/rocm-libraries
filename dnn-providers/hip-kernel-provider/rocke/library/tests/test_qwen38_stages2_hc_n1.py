# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the N1 HC stage functions (``kernels.gfx950.qwen38_mono.stages2.hc_n1``, P3x / WP15).

The GPU gates (T0 stage drift vs the fp32 golden, determinism, must-fail mutants, T1 Tier A, in-situ timing on the
served stack) run in the Qwen3.8 harness (``hackathon-prep/qwen38/truemono/W11/``)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace

import pytest

pytest.importorskip("rocke")

from kernels.gfx950.qwen38_mono.stages2 import hc as H  # noqa: E402
from kernels.gfx950.qwen38_mono.stages2 import hc_n1 as N  # noqa: E402

FLAVOR = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22")
KINDS = ("delta", "delta_ple", "qsa")


def _spec(stage, kind="delta", v="r1", spread=10, pre="args", **kw):
    rows, seq, _ = H.parse_variant(v)
    if pre == "args" and (stage != "a1" or kind == "delta_ple"):
        pre = "none"
    k = H.knobs_from_spec(f"numa=1,spread={spread},pre={pre}")
    return N.N1Spec(stage, kind, rows, seq, k, **kw)


def _ir(spec):
    return N.lower_n1_function(spec, flavor=FLAVOR).ir


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("form", ("ks", "na", "null"))
@pytest.mark.parametrize("v,spread,g", [("r1", 10, 1), ("r1", 1, 1), ("r1", 2, 4), ("r3s", 1, 1), ("r4s", 2, 4)])
def test_every_layout_lowers(kind, form, v, spread, g):
    for st in ("a1", "m1"):
        for kh in (1, 2) if g == 4 else (1, 2, 4, 5):
            s = _spec(st, kind, v, spread, g=g, kh=kh, form=form)
            assert N.n1_supported(s) is None, N.n1_supported(s)
            ir = _ir(s)
            assert f"@{s.symbol}(" in ir
    for st in ("a2", "m2"):
        s = _spec(st, kind, v, spread, g=g, form="na" if form == "na" else "ks")
        assert "mfma" in _ir(s).lower()


def test_refusals():
    assert "spread" in N.n1_supported(_spec("a1", "delta", "r1", 2, g=1))  # g=1 needs a units layout
    assert "spread" in N.n1_supported(_spec("a1", "delta", "r4s", 1, g=4))  # g=4 needs spread 2
    assert "tasks" in N.n1_supported(_spec("a1", "delta", "r4s", 1, g=1))  # 4 x 80 + 16 > 256
    assert "spread 2" in N.n1_supported(_spec("a1", "delta", "r4s", 3, g=4))  # 3 deduplicates past one row
    assert N.n1_supported(_spec("a1", "delta", "r1", 3, g=4)) is None
    assert "kh" in N.n1_supported(_spec("a1", "delta", "r4s", 2, g=4, kh=4))  # 16 waves
    assert "nab" in N.n1_supported(_spec("a1", "delta", "r1", 10, g=1, nab=0))
    assert "form" in N.n1_supported(_spec("a1", "delta", "r1", 10, g=1, form="xx"))
    assert "mutate" in N.n1_supported(_spec("a1", "delta", "r1", 10, g=1, mutate="xx"))
    s = _spec("a1", "delta", "r1", 10, g=1)
    assert "acc" in N.n1_supported(replace(s, knobs=replace(s.knobs, acc=2)))
    assert "WP2a mutants" in N.n1_supported(replace(s, knobs=replace(s.knobs, mutate="eps")))
    with pytest.raises(ValueError):
        N.build_n1_kernel(_spec("a1", "delta", "r1", 2, g=1))


def test_spec_string_round_trip():
    s = _spec("a1", "qsa", "r4s", 2, g=4, kh=2, form="na", nab=0, mutate="drop")
    t = N.n1_spec_from_str(s.stage, s.kind, s.rows, s.seq, s.knobs, N.n1_spec_str(s))
    assert t == s
    assert N.n1_spec_str(_spec("a1", g=1)) == "form=ks,g=1,kh=1"


@pytest.mark.parametrize("stage,mut", [("a2", "drop"), ("a2", "dup"), ("m2", "drop"), ("m2", "dup"), ("a1", "rrms")])
def test_mutants_change_the_ir(stage, mut):
    s = _spec(stage, "delta", "r1", 10, g=1)
    assert _ir(s) != _ir(replace(s, mutate=mut))


def test_na_rrms_mutant_and_forms_differ():
    a = _spec("a2", "delta", "r4s", 2, g=4, form="na")
    assert _ir(a) != _ir(replace(a, mutate="rrms"))
    assert _ir(a) != _ir(replace(a, form="ks"))
    u = _spec("a1", "delta", "r4s", 2, g=4, form="ks")
    assert _ir(u) != _ir(replace(u, form="na")) != _ir(replace(u, form="null"))
    assert _ir(replace(u, form="na")) != _ir(replace(u, form="na", nab=0))


def test_ks_unit_has_no_full_row_norm():
    """The point of N1: a ks / na down unit reads one stream (320 vectors), not the row's 1280; the na unit has no
    inverse RMS (no f64 sqrt) at all."""
    ks = _ir(_spec("a1", "delta", "r1", 10, g=1, form="ks", pre="none"))
    na = _ir(_spec("a1", "delta", "r1", 10, g=1, form="na", pre="none"))
    full = H.lower_hc_function(H.HcSpec("a1", "delta", 1, False, H.knobs_from_spec("numa=1,spread=10")), flavor=FLAVOR).ir
    # the injection tasks (unchanged) still normalize the whole row: count sqrt calls, not presence
    n_full, n_ks, n_na = (len(re.findall(r"llvm\.sqrt\.f64", x.split("define")[-1])) for x in (full, ks, na))
    assert n_ks <= n_full and n_na < n_ks


def test_sidecar_and_writer(tmp_path):
    specs = [_spec(st, "delta", "r4s", 2, g=4, kh=1, form="ks") for st in ("a1", "a2", "m1", "m2")]
    paths = N.write_n1_functions(str(tmp_path), specs, flavor=FLAVOR)
    assert len(paths) == 4
    for st in ("a1", "a2", "m1", "m2"):
        meta = json.load(open(tmp_path / "delta" / f"{st}_r4s.json"))
        assert meta["hc"]["spread"] == 2 and meta["hc"]["n1"] == "form=ks,g=4,kh=1" and meta["hc"]["tolerance"]
        assert meta["mailboxes"] == "numa1"
        assert (meta.get("pre_args") == "a1_preload") == (st == "a1")
    nul = N.sidecar(_spec("a1", "delta", "r1", 10, g=1, form="null"))
    assert nul["hc"]["timing_only"] and not nul["hc"]["tolerance"]


def test_wp2a_module_untouched_by_import():
    """Importing hc_n1 registers nothing and changes no WP2a default (every existing build byte-identical)."""
    spec = H.HcSpec("a1", "delta", 1, False, H.knobs_from_spec("numa=1,spread=10,pre=args"))
    before = H.lower_hc_function(spec, flavor=FLAVOR).ir
    import importlib

    importlib.reload(N)
    assert H.lower_hc_function(spec, flavor=FLAVOR).ir == before
    assert H.SPREAD_VALUES == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v,spread", [("r1", 0), ("r1", 3), ("r4s", 0), ("r4s", 2), ("r2s", 2)])
def test_nb_served_layouts_lower(kind, v, spread):
    """N1b proper (``nb``) and its bound (``nbz``) build in every served down-task layout, down stages only."""
    for st in ("a1", "m1"):
        for form in ("nb", "nbz"):
            s = _spec(st, kind, v, spread, g=0, form=form)
            assert N.n1_supported(s) is None, N.n1_supported(s)
            assert f"@{s.symbol}(" in _ir(s)
    assert "down stage only" in N.n1_supported(_spec("a2", kind, v, spread, g=0, form="nb"))


def test_nb_refusals_and_mutants():
    assert "layout" in N.n1_supported(_spec("a1", "delta", "r1", 10, g=0, form="nb"))
    assert "g=0" in N.n1_supported(_spec("a1", "delta", "r1", 0, g=1, form="nb"))
    s = _spec("a1", "delta", "r4s", 2, g=0, form="nb")
    base = _ir(s)
    for m in N.N1_MUTANTS:
        assert _ir(replace(s, mutate=m)) != base
    assert _ir(replace(s, nab=0)) != base
    assert _ir(replace(s, form="nbz")) != base
    assert N.sidecar(replace(s, form="nbz"))["hc"]["timing_only"]


def test_nb_statistics_follow_the_chains():
    """N1b's point: in the down task the row statistics (``llvm.sqrt.f64`` calls) come AFTER its first MFMA in program
    order; in the WP2a body every statistic precedes the chain."""
    nb = _ir(_spec("a1", "qsa", "r1", 0, g=0, form="nb", pre="none"))
    wp = H.lower_hc_function(H.HcSpec("a1", "qsa", 1, False, H.knobs_from_spec("numa=1")), flavor=FLAVOR).ir

    def sqrt_after_first_mfma(ir):
        body = ir[ir.index("define "):]
        i_m = body.index("mfma.scale")
        return body[i_m:].count("call double @llvm.sqrt.f64(")

    assert sqrt_after_first_mfma(nb) >= 4
    assert sqrt_after_first_mfma(wp) == 0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v", ["r2s", "r3s", "r4s"])
def test_dedupe_units_lower(kind, v):
    """The deduplicated stream-split unit (ATOM spread 4): one unit per (n-block, stream) for every row."""
    for st in ("a1", "m1"):
        for form in ("ks", "na", "null"):
            for kh in (1, 2):
                s = _spec(st, kind, v, 4, g=1, kh=kh, form=form)
                assert N.n1_supported(s) is None, N.n1_supported(s)
                assert f"@{s.symbol}(" in _ir(s)
    assert "rows >= 2" in N.n1_supported(_spec("a1", kind, "r1", 4, g=1))
