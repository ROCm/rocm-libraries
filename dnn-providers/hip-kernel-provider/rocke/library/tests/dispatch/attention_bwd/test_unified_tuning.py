# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuning registry and legality of the unified (paged KV) backward, CPU only.

* every tuning field of ``AttnBwdUnifiedSpec`` and every runtime value of the
  unified plan has exactly one axis; the shared knobs are the general axes,
  narrowed to what the unified shell builds;
* :func:`unified_spec_legality` agrees with the unified validator and marks
  the general-shell knobs ``not_built``;
* every tuned page-tile entry passes its check (also run at import), and a
  bad entry is refused;
* the shipped tuning hook reads the registered table.
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention_bwd import tuning_common as tc
from dispatch.attention_bwd import unified_tuning as ut
from kernels.common import attention_bwd_unified_plan as up
from kernels.common.attention_bwd_unified import (
    UNIFIED_KNOBS,
    AttnBwdUnifiedSpec,
    _SHELL_KNOBS,
    validate_attn_bwd_unified_spec,
)

UNIFIED_ARCHS = sorted(up.VALIDATED_ARCHS)


def test_one_axis_per_field_and_runtime_value():
    names = [ax.name for ax in ut.UNIFIED_BWD_KNOB_AXES]
    assert len(names) == len(set(names))
    fields = {f.name for f in dataclasses.fields(AttnBwdUnifiedSpec)}
    assert fields - ut.UNIFIED_PROBLEM_FIELDS <= set(names)
    runtime = {ax.name for ax in ut.UNIFIED_BWD_KNOB_AXES if ax.kind == tc.KIND_RUNTIME}
    assert runtime == set(up.RUNTIME_KNOBS)
    for name in UNIFIED_KNOBS:
        general = tc.backward_axis(name)
        ax = ut.unified_axis(name)
        assert (ax.choices, ax.default, ax.family, ax.stage) == (
            general.choices,
            general.default,
            general.family,
            general.stage,
        )
    with pytest.raises(ValueError):
        ut.unified_axis("dq_mode")  # the unified dQ is atomic: not a knob
    ut._check_registry()


def test_shell_knobs_are_registered_not_built():
    for name, keep in _SHELL_KNOBS.items():
        ax = ut.unified_axis(name)
        assert ax.default == keep
        assert set(ax.not_built) == set(ax.choices) - {keep}
    assert ut.unified_axis("global_path").not_built == ("dma",)
    ring = ut.unified_axis("ring_depth")
    assert not ring.applies(2, {"barrier_merge": "off"})


@pytest.mark.parametrize("arch", UNIFIED_ARCHS)
@pytest.mark.parametrize("d", (32, 64, 128))
def test_legality_agrees_with_the_validator(arch, d):
    cases = [{}, {"sub_page": 64}, {"sub_page": 48}, {"edge_tiles": True}]
    cases += [{name: v} for name, keep in _SHELL_KNOBS.items()
              for v in tc.backward_axis(name).choices if v != keep]  # fmt: skip
    cases += [{"global_path": "dma"}, {"ring_depth": 2}]
    for kw in cases:
        try:
            spec = AttnBwdUnifiedSpec(head_size=d, **kw)
        except ValueError:
            continue  # refused at construction (arch-free rule)
        status = ut.unified_spec_legality(spec, arch)
        try:
            validate_attn_bwd_unified_spec(spec, arch)
            valid = True
        except ValueError:
            valid = False
        if status == "legal":
            assert valid, (kw, d)
        if not valid:
            assert status != "legal", (kw, d)
        if any(k in _SHELL_KNOBS for k in kw) or kw.get("global_path") == "dma":
            assert status == "not_built", (kw, d)
    assert ut.unified_spec_legality(AttnBwdUnifiedSpec(head_size=d), arch) == "legal"


def test_tuned_page_tiles_pass_and_are_the_plans_table():
    assert ut.UNIFIED_TUNED_PAGE_TILES is up.TUNED_PAGE_TILES
    assert ut.UNIFIED_TUNED_PAGE_TILES
    for key, name in ut.UNIFIED_TUNED_PAGE_TILES.items():
        ut.check_page_tile_entry(key, name)
        arch, d, page = key
        req = up.AttnBwdUnifiedRequest(
            num_seqs=2, h_q=4, h_kv=4, d_qk=d, d_v=d, total_q=64,
            max_seqlen_q=32, max_seqlen_kv=4 * page, page=page, num_blocks=16,
            max_blocks_per_seq=8,
        )  # fmt: skip
        knobs = up.shipped_tuning(req, arch, "multi")
        cfg = up.attn_bwd_unified_tile_configs(arch, d, 8)[name][1]
        assert knobs == dict(cfg)


@pytest.mark.parametrize(
    "key,name,match",
    [
        (("gfx1151", 64, 64), "sub64", "not a unified target"),
        (("gfx950", 64, 40), "sub64", "multiple of 16"),
        (("gfx950", 64, 64), "nope", "not a configuration"),
        (("gfx950", 64, 32), "sub64", "does not serve page 32"),
        (("gfx950", 64), "sub64", "not .arch, head size, page."),
    ],
)
def test_bad_page_tile_entries_are_refused(key, name, match):
    with pytest.raises(ValueError, match=match):
        ut.check_page_tile_entry(key, name)


def test_a_bad_shipped_entry_fails_the_registry_check(monkeypatch):
    bad = dict(ut.UNIFIED_TUNED_PAGE_TILES)
    bad[("gfx950", 64, 32)] = "sub64"
    monkeypatch.setattr(ut, "UNIFIED_TUNED_PAGE_TILES", bad)
    with pytest.raises(AssertionError, match="tuned page-tile table"):
        ut._check_registry()


@pytest.mark.parametrize("arch", UNIFIED_ARCHS)
def test_tuning_axes_per_arch(arch):
    names = {ax.name for ax in ut.unified_tuning_axes(arch)}
    assert {"sub_page", "page_tile", "page_mode", "g_split"} <= names
    assert set(ut.unified_axis("page_tile").values(arch)) == set(ut.PAGE_TILE_NAMES)
    assert ut.unified_axis("page_tile").values("gfx1151") == ("default",)
