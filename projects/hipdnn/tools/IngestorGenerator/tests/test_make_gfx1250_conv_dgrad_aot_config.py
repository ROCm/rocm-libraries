# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx1250-only cases of the shipped hipkernel:Gfx1250ConvDgradAot bundle and its
catalog, made by tools/make_conv_dgrad_aot_config.py --arch gfx1250. What every dgrad
engine shares is test_make_conv_dgrad_aot_config.py's, run for gfx1250 too; these
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

_ARCH = "gfx1250"


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


def test_the_catalog_is_109_kernels_per_dtype(specs):
    """The gfx1250 dgrad instance list, one count per (kernel_family, direct_variant)
    pair: no helpers (gfx1250 has no MFMA pipeline)."""
    by_dtype = collections.defaultdict(collections.Counter)
    for spec in specs:
        assert spec.ARCH == _ARCH and spec.wave_size == 32
        by_dtype[spec.dtype][(spec.kernel_family, spec.direct_variant)] += 1
    for dtype in ("bf16", "fp16"):
        assert by_dtype[dtype] == {
            (0, "none"): 25,
            (1, "depthwise_dgrad"): 7,
            (1, "depthwise_dgrad_win"): 17,
            (1, "dgrad_wmma"): 53,
            (1, "dgrad_smallc_wmma"): 4,
            (1, "dgrad_patch_wmma"): 3,
        }
    assert len(specs) == 218


def test_the_schedule_knob_is_a_string_field():
    """gfx1250's WMMA schedule knob is a string KMD field, empty by default."""
    kmd = shared.document(_ARCH, "kmd")
    sched = next(f for f in kmd["fields"] if f["name"] == "sched")
    assert sched["type"] == "string" and sched["default_value"] == ""
    assert "sched" in shared.document(_ARCH, "ued")["knobs"]


def test_a_gfx950_instance_is_refused_from_the_gfx1250_catalog(tmp_path, identities):
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
