# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernel catalog of the native hipDNN backward engine (CPU only).

* the distinct binaries per arch and stage;
* one row per (class, tuned-table key), each kernel set built by its
  builder's support predicate;
* every row's representative request, through its hipDNN graph, plans to
  the row's kernels;
* no XCD remap knob reaches a catalog spec; every pack stays within the
  per-arch kernel budget.
"""

from __future__ import annotations

import collections

import pytest

from dispatch.attention_bwd.hipdnn import (
    ENGINE_ARCHS,
    engine_verdict,
    from_hipdnn_graph,
    main_fields,
    prep_fields,
    to_hipdnn_graph,
    tuned_keys,
)
from dispatch.attention_bwd.hipdnn_catalog import (
    CLASS_FIELDS,
    catalog_rows,
    default_block_n,
    prep_rows,
    row_key,
)
from kernels.common.attention_bwd import supports_attn_bwd_main
from kernels.common.attention_bwd_aux import (
    supports_attn_bwd_convert,
    supports_attn_bwd_prep,
)
from kernels.common.attention_bwd_dq import supports_attn_bwd_dq

# Distinct code objects per stage (the class defaults are exhaustive; the
# tuned rows cover every table key over its representative requests).
_BINARIES = {
    "gfx942": {"prep": 24, "main": 200, "convert": 24, "dq": 0},
    "gfx950": {"prep": 25, "main": 213, "convert": 26, "dq": 2},
}
_AOT_BUDGET_PER_PACK = 1000


def _binaries(arch):
    out = collections.defaultdict(set)
    for r in catalog_rows(arch):
        out["prep"].add(r.kernels["prep"])
        out["main"].add(r.kernels["main"])
        if r.dq_spec is not None:
            out["dq"].add(r.kernels["dq"])
        else:
            out["convert"].add(r.kernels["dq"])
        if r.dkv_mode == "atomic":
            out["convert"].add(r.kernels["convert_dk"])
    return out


@pytest.mark.parametrize("arch", ENGINE_ARCHS)
def test_distinct_binaries(arch):
    got = {k: len(v) for k, v in _binaries(arch).items()}
    assert {k: got.get(k, 0) for k in _BINARIES[arch]} == _BINARIES[arch]
    assert {p.kernel for p in prep_rows(arch)} == _binaries(arch)["prep"]


@pytest.mark.parametrize("arch", ENGINE_ARCHS)
def test_rows_are_unique_and_ordered(arch):
    rows = catalog_rows(arch)
    keys = [row_key(r) for r in rows]
    assert len(keys) == len(set(keys))
    assert catalog_rows(arch) == rows
    ids = {i for i, _k, _s in tuned_keys(arch)} | {0}
    for r in rows:
        assert r.arch == arch and r.tuning_key_id in ids
        want = "main_splitdq_" if r.dq_mode == "split" else "main_"
        assert r.pack == want + arch
        assert r.kernels["convert_dk"] == r.kernels["convert_dv"]
        assert (r.dq_spec is None) == (r.dq_mode == "atomic")
    preps = prep_rows(arch)
    assert len({p.kernel for p in preps}) == len(preps)
    assert all(p.pack == f"prep_{arch}" for p in preps)


@pytest.mark.parametrize("arch", ENGINE_ARCHS)
def test_every_kernel_passes_its_support_predicate(arch):
    for r in catalog_rows(arch):
        for name, ok in (
            ("main", supports_attn_bwd_main(r.main_spec, arch=arch)),
            ("prep", supports_attn_bwd_prep(r.prep_spec, arch=arch)),
            ("convert", supports_attn_bwd_convert(r.convert_spec, arch=arch)),
        ):
            assert ok == (True, ""), (name, r.kernels, ok)
        if r.dq_spec is not None:
            assert supports_attn_bwd_dq(r.dq_spec, arch=arch) == (True, "")


def test_support_predicates_refuse_what_the_builders_refuse():
    from kernels.common.attention_bwd import AttnBwdSpec
    from kernels.common.attention_bwd_aux import AttnBwdAuxSpec

    bad = AttnBwdSpec(head_size=64, block_m=64, block_n=64, waves=1)
    ok, why = supports_attn_bwd_main(bad, arch="gfx942")
    assert not ok and "waves=1" in why
    assert (
        supports_attn_bwd_prep(AttnBwdAuxSpec(head_size=64), arch="gfx90x")[0] is False
    )
    vec4 = AttnBwdAuxSpec(head_size=64, convert_vec=4)
    assert supports_attn_bwd_prep(vec4, arch="gfx942")[0] is False
    assert supports_attn_bwd_convert(vec4, arch="gfx942") == (True, "")
    wl = AttnBwdAuxSpec(head_size=64, seq_mode="thd", worklist=True)
    assert supports_attn_bwd_prep(wl, arch="gfx950") == (True, "")
    assert supports_attn_bwd_convert(wl, arch="gfx950")[0] is False
    odd = AttnBwdAuxSpec(head_size=64, rows_per_cta=3)
    assert supports_attn_bwd_prep(odd, arch="gfx942")[0] is False


@pytest.mark.parametrize("arch", ENGINE_ARCHS)
def test_every_representative_plans_its_row_through_its_graph(arch):
    for r in catalog_rows(arch):
        rep = r.representative
        for req in (
            rep,
            from_hipdnn_graph(to_hipdnn_graph(rep), arch, num_cus=rep.num_cus),
        ):
            code, detail, plan = engine_verdict(req, arch)
            assert code == "OK", (r.kernels, detail)
            steps = {s.stage: s.kernel for s in plan.launches}
            assert steps["main"] == r.kernels["main"]
            assert steps["prep"] == r.kernels["prep"]
            mf = main_fields(plan)
            assert tuple(mf[f] for f in CLASS_FIELDS) == r.class_key
            assert mf["tuning_key"] == r.tuning_key_str
            assert mf == r.main_kmd
    for p in prep_rows(arch):
        _c, _d, plan = engine_verdict(p.representative, arch)
        assert prep_fields(plan) == dict(p.fields)


@pytest.mark.parametrize("arch", ENGINE_ARCHS)
def test_no_xcd_knob_and_constant_layouts(arch):
    rows = catalog_rows(arch)
    for r in rows:
        assert not r.main_spec.xcd_remap
        assert "xcd_chunk" not in r.tuned_knobs and "dq_xcd_chunk" not in r.tuned_knobs
        if r.dq_spec is not None:
            assert not r.dq_spec.xcd_remap
    assert len({r.prep_spec.ws_layout for r in rows}) == 1
    assert len({r.prep_spec.kv_target for r in rows}) == 1
    assert len({r.main_spec.ws_layout for r in rows}) == 1


@pytest.mark.parametrize("arch", ENGINE_ARCHS)
def test_packs_within_the_kernel_budget(arch):
    per_pack = collections.Counter(r.pack for r in catalog_rows(arch))
    per_pack[f"prep_{arch}"] = len(prep_rows(arch))
    assert all(n <= _AOT_BUDGET_PER_PACK for n in per_pack.values()), per_pack
    assert (f"main_splitdq_{arch}" in per_pack) is (arch == "gfx950")


def test_default_block_n_is_the_default_geometry():
    for r in catalog_rows("gfx942"):
        if r.tuning_key_id == 0:
            assert r.main_block_n == default_block_n(
                "gfx942", r.head_size, r.dtype, r.stage_vec, r.seq_mode
            )
