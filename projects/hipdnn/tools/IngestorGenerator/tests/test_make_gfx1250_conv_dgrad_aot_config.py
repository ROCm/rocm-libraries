# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the shipped hipkernel:Gfx1250ConvDgradAot bundle and its catalog, made by
tools/make_conv_dgrad_aot_config.py --arch gfx1250 (the gfx950 bundle's tests are
test_make_conv_dgrad_aot_config.py, whose helpers these reuse)."""

import collections
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import enumerate_conv_dgrad_aot_instances as enumerate_tool  # noqa: E402
import launch_surface  # noqa: E402
import make_conv_dgrad_aot_config as tool  # noqa: E402
import test_make_conv_dgrad_aot_config as gfx950_tests  # noqa: E402

_ARCH = "gfx1250"
_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_PROVIDER = (
    _REPO / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
)
_SLUG = "gfx1250_conv_dgrad_aot"
_SHIPPED_DIR = _PROVIDER / "descriptors/rocKE" / _SLUG
_GFX950_DIR = _PROVIDER / "descriptors/rocKE/gfx950_conv_dgrad_aot"
_CENSUS = (
    _REPO / "dnn-providers/hip-kernel-provider/src/tests/engines"
    "/kernel_ingestor_engine/packs/TestGfx1250ConvDgradAotPacks.cpp"
)
_INSTANCES = [
    _CONFIGS / enumerate_tool.INSTANCE_LIST.format(arch=_ARCH, dtype=dtype)
    for dtype in ("bf16", "fp16")
]
_FILE_SIZE_LIMIT = 50 * 1000 * 1000
_JOBS = min(32, os.cpu_count() or 1)
_ENGINE = tool.ENGINES[_ARCH][0]


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    before = list(sys.path)
    tool._bind_rocke(_ROCKE)
    yield
    sys.path[:] = before


@pytest.fixture(scope="module")
def identities():
    return [i for path in _INSTANCES for i in json.loads(path.read_text())]


@pytest.fixture(scope="module")
def specs(identities):
    out = []
    for spec, reason in tool.instance_specs(identities, _JOBS):
        assert spec is not None, reason
        out.append(spec)
    return sorted(out, key=lambda s: (s.dtype, s.instance_id))


@pytest.fixture(scope="module")
def shipped():
    """instance_id -> (shard, kernel descriptor) of the shipped bundle."""
    out = {}
    for kdp in sorted((_SHIPPED_DIR / "artifacts").glob("*.kdp.json")):
        document = json.loads(kdp.read_text())
        shard = kdp.name[: -len(".kdp.json")][len(_SLUG) + 1 :]
        assert document["name"] == f"hipkernel:{_SLUG}_{shard}"
        for kernel in document["kernelDescriptors"]:
            out[kernel["metadata"]["instance_id"]] = (shard, kernel)
    return out


def _document(kind):
    return json.loads((_SHIPPED_DIR / f"{_SLUG}.{kind}.json").read_text())


def test_the_engines_agree_with_the_adapter():
    from builders.common.convolution_backward_data import TARGETS

    assert set(tool.ENGINES) == set(TARGETS)
    for arch, (engine, builder) in tool.ENGINES.items():
        assert TARGETS[arch].engine == engine
        assert builder == f"build_{arch}_conv_dgrad_aot"


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


def test_the_shipped_bundle_is_its_catalog_with_instance_derived_ids(specs, shipped):
    assert sorted(shipped) == sorted(s.instance_id for s in specs)
    for spec in specs:
        shard, kernel = shipped[spec.instance_id]
        assert shard == tool.shard_name(spec)
        expected = {name: getattr(spec, name) for name in tool.spec_order(spec)}
        assert kernel["kernel_source"]["spec"] == expected
        assert kernel["metadata"] == expected
        assert kernel["kernel_source"]["source"] == tool.ADAPTER_SOURCE
        assert kernel["kernel_source"]["builder"] == tool.ENGINES[_ARCH][1]
        assert kernel["id"] == str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id)
        )
        assert kernel["id"] == tool.kernel_uuid(spec.instance_id, arch=_ARCH)
        template = (
            tool.IMPLICIT_VARIANT_NAME
            if not spec.is_direct
            else tool._DIRECT_VARIANT_NAMES[spec.direct_variant]
        )
        assert kernel["name"] == template.format(
            arch=_ARCH, dtype=spec.dtype, instance_id=spec.instance_id
        )


def test_the_shipped_bundle_is_sharded_below_the_hosting_size_limit(specs):
    names = sorted(
        path.name
        for path in _SHIPPED_DIR.iterdir()
        if path.name not in ("artifacts", "heuristics")
    )
    shards = {tool.shard_name(spec) for spec in specs}
    assert shards == {
        f"{dtype}_{p}"
        for dtype in ("bf16", "fp16")
        for p in ("mem", "wavelet", "direct")
    }
    expected = [f"{_SLUG}.{kind}.json" for kind in ("kmd", "udd", "ued", "uhd")] + [
        "kernel_dtype_matches_graph.umd.json",
        "artifacts.dvc",
        ".gitignore",
    ]
    if (_SHIPPED_DIR / "heuristics.dvc").exists():
        expected.append("heuristics.dvc")
    assert names == sorted(expected)
    matcher = json.loads(
        (_SHIPPED_DIR / "kernel_dtype_matches_graph.umd.json").read_text()
    )
    top_level = [path for path in _SHIPPED_DIR.iterdir() if path.is_file()]
    for path in top_level + _INSTANCES + [_CENSUS]:
        assert path.stat().st_size < _FILE_SIZE_LIMIT, path.name
    for kdp in (_SHIPPED_DIR / "artifacts").glob("*.kdp.json"):
        document = json.loads(kdp.read_text())
        assert document["matchers"] == [matcher["id"]]
        assert document["engine"] == _document("ued")["id"]
        assert document["dispatch"] == _document("udd")["id"]


def test_the_bundle_names_its_own_native_symbols():
    ued, udd, uhd, kmd = (_document(k) for k in ("ued", "udd", "uhd", "kmd"))
    umd = json.loads((_SHIPPED_DIR / "kernel_dtype_matches_graph.umd.json").read_text())
    assert ued["name"] == _ENGINE
    assert ued["graph_match"] == {"native": f"hipkernel.{_SLUG}.graph_match"}
    assert udd["dispatch_symbol"] == f"hipkernel.{_SLUG}.dispatch"
    assert uhd["native"] == {"symbol": f"hipkernel.{_SLUG}.score"}
    assert umd["match_symbol"] == f"hipkernel.{_SLUG}.kernel_match"
    assert ued["metadata"] == kmd["id"]
    # The native fallback scorer comes first; a promoted model packet appends its own id
    # after it, so "default" is a list once one ships.
    assert list(ued["sort_kernel_catalog"]) == ["default"]
    default = ued["sort_kernel_catalog"]["default"]
    default = default if isinstance(default, list) else [default]
    assert default[0] == uhd["id"]
    assert ued["knobs"] == list(tool.engine_knobs(_ARCH))
    kmd_names = [f["name"] for f in kmd["fields"]]
    assert kmd_names == [f["name"] for f in tool.kmd_fields(_ARCH)]
    assert set(ued["knobs"]) <= set(kmd_names)
    # The schedule knob is a string field.
    sched = next(f for f in kmd["fields"] if f["name"] == "sched")
    assert sched["type"] == "string" and sched["default_value"] == ""


def test_bundle_ids_are_unique_across_the_dgrad_bundles():
    """Both dgrad bundles load into one process: no document or kernel id may repeat."""
    ids = {}
    for folder in (_GFX950_DIR, _SHIPPED_DIR, _SHIPPED_DIR / "artifacts"):
        for path in folder.glob("*.json"):
            document = json.loads(path.read_text())
            found = [document["id"]] + [
                kernel["id"] for kernel in document.get("kernelDescriptors", [])
            ]
            for value in found:
                assert ids.setdefault(value, path) == path, value


def test_knob_tuples_are_unique_within_every_dtype_family_and_capability(shipped):
    """GenericPlanBuilder refuses a graph whose matched catalog repeats a UED knob
    tuple; instances that can meet in one graph share a (dtype, family, capability)
    partition."""
    defaults = {f["name"]: f.get("default_value") for f in _document("kmd")["fields"]}
    knobs = tool.engine_knobs(_ARCH)
    partitions = collections.defaultdict(list)
    for _, kernel in shipped.values():
        values = dict(defaults, **kernel["metadata"])
        partitions[gfx950_tests._capability(kernel, defaults)].append(
            tuple(values[k] for k in knobs)
        )
    for key, tuples in partitions.items():
        assert len(tuples) == len(set(tuples)), key


def test_the_census_names_every_pack_and_kernel(shipped):
    text = _CENSUS.read_text()
    for _, kernel in shipped.values():
        assert f'"{kernel["name"]}"' in text
    for kdp in (_SHIPPED_DIR / "artifacts").glob("*.kdp.json"):
        assert f'"{json.loads(kdp.read_text())["name"]}"' in text
    assert f'"{_ENGINE}"' in text
    assert f"hipkernel.{_SLUG}.score" in text
    assert f"knobs.size(), {len(tool.engine_knobs(_ARCH))}U" in text


def test_stabilize_ids_derives_product_ids_and_matches_the_shipped_kdp(tmp_path):
    shipped = sorted((_SHIPPED_DIR / "artifacts").glob("*_direct.kdp.json"))[0]
    document = json.loads(shipped.read_text())
    for kernel in document["kernelDescriptors"]:
        kernel["id"] = str(uuid.uuid4())
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(json.dumps(document, indent=2) + "\n")
    assert tool.stabilize_ids(kdp, arch=_ARCH) == len(document["kernelDescriptors"])
    assert json.loads(kdp.read_text()) == json.loads(shipped.read_text())


def test_a_gfx950_instance_is_refused_from_the_gfx1250_catalog(tmp_path, identities):
    gfx950 = json.loads(
        (_CONFIGS / "gfx950_conv_dgrad_aot_fp16.instances.json").read_text()
    )[0]
    instances = tmp_path / "mixed.json"
    instances.write_text(json.dumps([identities[0], gfx950]))
    argv = [
        "--arch",
        _ARCH,
        "--rocke-root",
        str(_ROCKE),
        "--instances",
        str(instances),
        "--output",
        str(tmp_path / "out.yaml"),
    ]
    assert tool.main(argv) == 1
    assert not (tmp_path / "out.yaml").exists()
