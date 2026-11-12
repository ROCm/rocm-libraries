# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx942-only cases of the shipped hipkernel:Gfx942ConvDgradAot bundle and its
catalog, made by tools/make_conv_dgrad_aot_config.py --arch gfx942. What every dgrad
engine shares is test_make_conv_dgrad_aot_config.py's, run for gfx942 too; these
reuse its loaders."""

import collections
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_conv_dgrad_aot_config as tool  # noqa: E402
import test_make_conv_dgrad_aot_config as shared  # noqa: E402

_ARCH = "gfx942"


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    before = list(sys.path)
    tool._bind_rocke(shared._ROCKE)
    yield
    sys.path[:] = before


@pytest.fixture(scope="module")
def identities():
    return list(shared.load_identities(_ARCH))


@pytest.fixture(scope="module")
def specs():
    return list(shared.load_specs(_ARCH))


def test_the_catalog_is_173_selectable_kernels_and_8_helpers_per_dtype(specs):
    """The gfx942 dgrad instance list, one count per (kernel_family, direct_variant)
    pair: 128 implicit-GEMM and 45 direct selectable kernels, and the 8 weight-transform
    helpers its MFMA rows launch."""
    by_dtype = collections.defaultdict(collections.Counter)
    for spec in specs:
        assert spec.ARCH == _ARCH and spec.wave_size == 64
        by_dtype[spec.dtype][(spec.kernel_family, spec.direct_variant)] += 1
    for dtype in ("bf16", "fp16"):
        assert by_dtype[dtype] == {
            (0, "none"): 128,
            (1, "depthwise_dgrad"): 9,
            (1, "depthwise_dgrad_win"): 14,
            (1, "grouped_dgrad_fused"): 5,
            (1, "grouped_dgrad_mfma"): 6,
            (1, "grouped_dgrad_4c"): 1,
            (1, "dgrad_mfma_rtc"): 10,
            (2, "transpose"): 4,
            (2, "reorganize"): 2,
            (2, "transpose_rtc"): 1,
            (2, "reorganize_rtc"): 1,
        }
    assert sum(1 for s in specs if not s.is_helper) == 346
    assert len(specs) == 362


def test_the_catalog_leaves_every_gfx950_only_knob_off(specs):
    """gfx942 has no 16x16x32 MFMA atom and no ds_read_b64_tr_b16: no instance folds
    K by 32, stages weights or rows in LDS, pairs taps with dot2, runs the windowed
    depthwise on MFMA or uses the gfx950 implicit-GEMM LDS layouts."""
    for spec in specs:
        assert spec.warp_tile_k in (0, 8, 16), spec.instance_id
        for knob in (
            "fold_k32",
            "lds_k_outer",
            "dy_halo_kouter_pad",
            "dgrad_weights_lds",
            "stage_rows",
            "xcd_tiles",
            "dot2",
            "mfma",
        ):
            assert getattr(spec, knob) == 0, (spec.instance_id, knob)


def test_the_streaming_depthwise_channels_per_lane_is_an_int_knob(specs):
    """gfx942's knob table is gfx950's plus the streaming depthwise ``cpl``, an int KMD
    field defaulting to 0 that only depthwise instances set."""
    kmd = shared.document(_ARCH, "kmd")
    cpl = next(f for f in kmd["fields"] if f["name"] == "cpl")
    assert cpl["type"] == "int" and cpl["default_value"] == 0
    assert "cpl" in shared.document(_ARCH, "ued")["knobs"]
    assert "cpl" not in shared.document("gfx950", "ued")["knobs"]
    assert {s.direct_variant for s in specs if s.cpl} == {"depthwise_dgrad"}


def test_every_mfma_instance_finds_its_helpers_in_the_bundle(specs):
    """As gfx950's: each MFMA instance's helper lookup (by KMD field values) finds
    exactly one helper kernel of its own shard, and every helper is used."""
    from builders.common.convolution_backward_data import (
        gfx950_conv_dgrad_aot_helper_fields,
    )

    shipped = shared.load_shipped(_ARCH)
    defaults = {
        f["name"]: f.get("default_value")
        for f in shared.document(_ARCH, "kmd")["fields"]
    }
    helpers = [
        (shard, dict(defaults, **k["metadata"]))
        for shard, k in shipped.values()
        if k["metadata"].get("kernel_family") == 2
    ]
    assert helpers
    by_id = {s.instance_id: s for s in specs}
    used = set()
    for shard, kernel in shipped.values():
        spec = by_id[kernel["metadata"]["instance_id"]]
        if not spec.is_mfma:
            continue
        for wanted in gfx950_conv_dgrad_aot_helper_fields(spec):
            found = [
                (s, h)
                for s, h in helpers
                if all(h[name] == value for name, value in wanted.items())
            ]
            assert len(found) == 1, (spec.instance_id, wanted)
            assert found[0][0] == shard
            used.add(found[0][1]["instance_id"])
    assert used == {h["instance_id"] for _, h in helpers}


def test_a_gfx950_instance_is_refused_from_the_gfx942_catalog(tmp_path, identities):
    gfx950 = shared.load_identities("gfx950")[0]
    instances = tmp_path / "mixed.json"
    instances.write_text(json.dumps([identities[0], gfx950]))
    argv = [
        "--arch",
        _ARCH,
        "--rocke-root",
        str(shared._ROCKE),
        "--instances",
        str(instances),
        "--output",
        str(tmp_path / "out.yaml"),
    ]
    assert tool.main(argv) == 1
    assert not (tmp_path / "out.yaml").exists()
