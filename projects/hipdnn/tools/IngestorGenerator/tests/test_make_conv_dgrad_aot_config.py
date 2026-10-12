# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_conv_dgrad_aot_config.py and the shipped hipkernel:Gfx950ConvDgradAot bundle."""

import collections
import copy
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import launch_surface  # noqa: E402
import make_conv_aot_config as fwd_tool  # noqa: E402
import make_conv_dgrad_aot_config as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_PROVIDER = (
    _REPO / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
)
_SHIPPED_DIR = _PROVIDER / "descriptors/rocKE/gfx950_conv_dgrad_aot"
_FWD_DIRS = (
    _PROVIDER / "descriptors/rocKE/gfx950_conv_fwd_aot",
    _PROVIDER / "test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed",
)
_CENSUS = (
    _REPO / "dnn-providers/hip-kernel-provider/src/tests/engines"
    "/kernel_ingestor_engine/packs/TestGfx950ConvDgradAotPacks.cpp"
)
_INSTANCES = [
    _CONFIGS / f"gfx950_conv_dgrad_aot_{dtype}.instances.json"
    for dtype in ("bf16", "fp16")
]
_FILE_SIZE_LIMIT = 50 * 1000 * 1000
_JOBS = min(32, os.cpu_count() or 1)
_SLUG = "gfx950_conv_dgrad_aot"


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
    for kdp in sorted(_SHIPPED_DIR.glob("*.kdp.json")):
        document = json.loads(kdp.read_text())
        shard = kdp.name[: -len(".kdp.json")][len(_SLUG) + 1 :]
        assert document["name"] == f"hipkernel:{_SLUG}_{shard}"
        for kernel in document["kernelDescriptors"]:
            out[kernel["metadata"]["instance_id"]] = (shard, kernel)
    return out


def _document(kind):
    return json.loads((_SHIPPED_DIR / f"{_SLUG}.{kind}.json").read_text())


def test_the_catalog_is_291_selectable_kernels_and_9_helpers_per_dtype(specs):
    by_dtype = collections.defaultdict(collections.Counter)
    for spec in specs:
        by_dtype[spec.dtype][(spec.kernel_family, spec.direct_variant)] += 1
    for dtype in ("bf16", "fp16"):
        assert by_dtype[dtype] == {
            (0, "none"): 235,
            (1, "depthwise_dgrad"): 30,
            (1, "grouped_dgrad_mfma"): 26,
            (2, "transpose"): 6,
            (2, "reorganize"): 3,
        }
    assert sum(1 for s in specs if not s.is_helper) == 582
    assert len(specs) == 600


def test_instance_id_is_the_identity_stable_hash(identities):
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in identities:
        spec, _ = tool.instance_spec(identity)
        assert spec.instance_id == KernelIdentity.from_dict(identity).stable_hash()


def test_the_shipped_bundle_is_its_catalog_with_instance_derived_ids(specs, shipped):
    """Every shard holds exactly the lists' instances of that shard; spec and metadata
    are the instance's fields in its variant's order; ids are uuid5(NS, instance_id)
    because this engine is its own product engine."""
    assert sorted(shipped) == sorted(s.instance_id for s in specs)
    for spec in specs:
        shard, kernel = shipped[spec.instance_id]
        assert shard == tool.shard_name(spec)
        expected = {name: getattr(spec, name) for name in tool.spec_order(spec)}
        assert kernel["kernel_source"]["spec"] == expected
        assert kernel["metadata"] == expected
        assert kernel["kernel_source"]["source"] == tool.ADAPTER_SOURCE
        assert kernel["kernel_source"]["builder"] == tool.ADAPTER_BUILDER
        assert kernel["id"] == str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id)
        )
        assert kernel["id"] == tool.kernel_uuid(spec.instance_id)
        prefix = {
            (0, "none"): "gfx950_conv_dgrad_aot_",
            (1, "depthwise_dgrad"): "gfx950_conv_dgrad_aot_dw_",
            (1, "grouped_dgrad_mfma"): "gfx950_conv_dgrad_aot_mfma_",
            (2, "transpose"): "gfx950_conv_dgrad_aot_transpose_",
            (2, "reorganize"): "gfx950_conv_dgrad_aot_reorganize_",
        }[(spec.kernel_family, spec.direct_variant)]
        assert kernel["name"] == f"{prefix}{spec.dtype}_{spec.instance_id}"


def test_the_shipped_bundle_is_sharded_below_the_hosting_size_limit(specs):
    # A promoted model packet adds a heuristics/ folder next to the bundle; the bundle's
    # own files are the top-level ones.
    names = sorted(
        path.name for path in _SHIPPED_DIR.iterdir() if path.name != "heuristics"
    )
    shards = {tool.shard_name(spec) for spec in specs}
    assert shards == {
        f"{dtype}_{p}"
        for dtype in ("bf16", "fp16")
        for p in ("mem", "compv3", "compv4", "direct")
    }
    assert names == sorted(
        [f"{_SLUG}.{kind}.json" for kind in ("kmd", "udd", "ued", "uhd")]
        + [f"{_SLUG}_{shard}.kdp.json" for shard in shards]
        + ["kernel_dtype_matches_graph.umd.json"]
    )
    matcher = json.loads(
        (_SHIPPED_DIR / "kernel_dtype_matches_graph.umd.json").read_text()
    )
    top_level = [path for path in _SHIPPED_DIR.iterdir() if path.is_file()]
    for path in top_level + _INSTANCES + [_CENSUS]:
        assert path.stat().st_size < _FILE_SIZE_LIMIT, path.name
    for kdp in _SHIPPED_DIR.glob("*.kdp.json"):
        document = json.loads(kdp.read_text())
        assert document["matchers"] == [matcher["id"]]
        assert document["engine"] == _document("ued")["id"]
        assert document["dispatch"] == _document("udd")["id"]


def test_the_bundle_names_its_own_native_symbols():
    ued, udd, uhd, kmd = (_document(k) for k in ("ued", "udd", "uhd", "kmd"))
    umd = json.loads((_SHIPPED_DIR / "kernel_dtype_matches_graph.umd.json").read_text())
    assert ued["name"] == tool.PRODUCT_ENGINE
    assert ued["graph_match"] == {
        "native": "hipkernel.gfx950_conv_dgrad_aot.graph_match"
    }
    assert udd["dispatch_symbol"] == "hipkernel.gfx950_conv_dgrad_aot.dispatch"
    assert uhd["native"] == {"symbol": "hipkernel.gfx950_conv_dgrad_aot.score"}
    assert umd["match_symbol"] == "hipkernel.gfx950_conv_dgrad_aot.kernel_match"
    assert ued["metadata"] == kmd["id"]
    # The native fallback scorer comes first; a promoted model packet appends its own.
    assert list(ued["sort_kernel_catalog"]) == ["default"]
    assert ued["sort_kernel_catalog"]["default"][0] == uhd["id"]
    assert ued["knobs"] == list(tool.ENGINE_KNOBS)
    kmd_names = [f["name"] for f in kmd["fields"]]
    assert kmd_names == [f["name"] for f in tool.kmd_fields()]
    assert set(ued["knobs"]) <= set(kmd_names)
    assert "instance_id" not in ued["knobs"] and "max_sub_gemms" not in ued["knobs"]


def test_the_config_borrows_no_native_symbols(specs):
    config = tool.build_config(specs)
    assert "native_symbols_from" not in config["engine"]
    assert config["workspace_policy"] == "derived"
    other = tool.build_config(specs[:3], "hipkernel:Gfx950ConvDgradAotSeed")
    assert other["engine"]["native_symbols_from"] == tool.PRODUCT_ENGINE
    assert tool.kernel_uuid(specs[0].instance_id, "hipkernel:X") != tool.kernel_uuid(
        specs[0].instance_id
    )


def test_generated_config_expands_to_one_kernel_per_instance(tmp_path, specs):
    path = tmp_path / "dgrad.yaml"
    path.write_text(tool.dump_config(tool.build_config(specs)))
    config = load_config(path)
    assert config.engine.name == tool.PRODUCT_ENGINE
    assert config.is_multi_pack and not config.has_operation_matchers
    assert sum(len(pack.kernels) for pack in config.packs) == len(specs)


def test_bundle_ids_are_unique_across_the_conv_aot_bundles():
    """The fwd, seed and dgrad bundles load into one process: no document or kernel id
    may repeat across them."""
    ids = {}
    for folder in (*_FWD_DIRS, _SHIPPED_DIR):
        for path in folder.glob("*.json"):
            document = json.loads(path.read_text())
            found = [document["id"]] + [
                kernel["id"] for kernel in document.get("kernelDescriptors", [])
            ]
            for value in found:
                assert ids.setdefault(value, path) == path, value


def _knob_tuple(kernel, kmd_defaults):
    values = dict(kmd_defaults, **kernel["metadata"])
    return tuple(values[k] for k in tool.ENGINE_KNOBS)


def _capability(kernel, kmd_defaults):
    values = dict(kmd_defaults, **kernel["metadata"])
    return tuple(
        values[k]
        for k in (
            "dtype",
            "kernel_family",
            "filter_h",
            "filter_w",
            "stride_h",
            "stride_w",
            "pad_h",
            "pad_w",
            "dilation_h",
            "dilation_w",
            "cpg",
            "kpg",
        )
    )


def test_knob_tuples_are_unique_within_every_dtype_family_and_capability(shipped):
    """GenericPlanBuilder refuses a graph whose matched catalog repeats a UED knob
    tuple. The matcher pins dtype and every direct capability exactly, so instances
    that can meet in one graph share a (dtype, family, capability) partition."""
    defaults = {f["name"]: f.get("default_value") for f in _document("kmd")["fields"]}
    partitions = collections.defaultdict(list)
    for _, kernel in shipped.values():
        if kernel["metadata"].get("kernel_family") == 2:
            continue  # helpers are never matched
        partitions[_capability(kernel, defaults)].append(_knob_tuple(kernel, defaults))
    assert len(partitions) == 2 * (1 + 3 + 6)
    for key, tuples in partitions.items():
        assert len(tuples) == len(set(tuples)), key
    # Implicit GEMM and the direct kernels can meet in one graph too.
    for dtype in ("bf16", "fp16"):
        tuples = [
            _knob_tuple(kernel, defaults)
            for _, kernel in shipped.values()
            if kernel["metadata"]["dtype"] == dtype
            and kernel["metadata"].get("kernel_family", 0) != 2
        ]
        implicit = [t for t in tuples if t[0] == 0]
        assert len(set(implicit)) == len(implicit) == 235


def test_the_mfma_row_tile_is_the_knob_token_not_the_identity_field(identities, specs):
    by_id = {s.instance_id: s for s in specs}
    from benchmarks.common.kernel_cache import KernelIdentity

    seen = set()
    for identity in identities:
        ident = KernelIdentity.from_dict(identity)
        if ident.algorithm != "direct_grouped_dgrad_mfma":
            continue
        assert ident.block_h == 0
        spec = by_id[ident.stable_hash()]
        assert spec.block_h == json.loads(ident.knobs)["block_h"] > 0
        seen.add(spec.block_h)
    assert seen == {8, 16}


def test_every_mfma_instance_finds_its_helpers_in_the_bundle(shipped):
    """The MFMA dispatch looks its helpers up by KMD field values; each lookup finds
    exactly one helper kernel of the same shard."""
    from builders.common.convolution_backward_data import (
        gfx950_conv_dgrad_aot_helper_fields,
    )

    defaults = {f["name"]: f.get("default_value") for f in _document("kmd")["fields"]}
    # Every helper states each lookup field itself (fold_k32 included, 0 for the
    # transpose), so the lookup does not depend on KMD defaults being applied.
    for _, kernel in shipped.values():
        if kernel["metadata"].get("kernel_family") == 2:
            assert "fold_k32" in kernel["metadata"], kernel["metadata"]["instance_id"]
    helpers = [
        (shard, dict(defaults, **k["metadata"]))
        for shard, k in shipped.values()
        if k["metadata"].get("kernel_family") == 2
    ]
    used = set()
    specs = {s.instance_id: s for s in _specs_of(shipped)}
    for shard, kernel in shipped.values():
        meta = kernel["metadata"]
        if meta.get("direct_variant") != "grouped_dgrad_mfma":
            continue
        for wanted in gfx950_conv_dgrad_aot_helper_fields(specs[meta["instance_id"]]):
            found = [
                (s, h)
                for s, h in helpers
                if all(h[name] == value for name, value in wanted.items())
            ]
            assert len(found) == 1, (meta["instance_id"], wanted)
            assert found[0][0] == shard
            used.add(found[0][1]["instance_id"])
    assert used == {h["instance_id"] for _, h in helpers}


def _specs_of(shipped):
    from builders.common.convolution_backward_data import Gfx950ConvDgradAotSpec

    return [
        Gfx950ConvDgradAotSpec(**kernel["kernel_source"]["spec"])
        for _, kernel in shipped.values()
    ]


def test_the_census_names_every_pack_and_kernel(shipped):
    text = _CENSUS.read_text()
    for _, kernel in shipped.values():
        assert f'"{kernel["name"]}"' in text
    for kdp in _SHIPPED_DIR.glob("*.kdp.json"):
        assert f'"{json.loads(kdp.read_text())["name"]}"' in text
    assert '"hipkernel:Gfx950ConvDgradAot"' in text
    assert "hipkernel.gfx950_conv_dgrad_aot.score" in text
    assert f"knobs.size(), {len(tool.ENGINE_KNOBS)}U" in text


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"direction": "fwd"}, "dgrad implicit-GEMM"),
        ({"direction": "wgrad"}, "dgrad implicit-GEMM"),
        ({"arch": "gfx942"}, "dgrad implicit-GEMM"),
        ({"dtype_d": "fp32"}, "mixed operand dtypes"),
        ({"split_k": 2}, "split_k"),
        ({"stride_h": 2}, "stride_h"),
        ({"unroll_k": True}, "unroll_k"),
        ({"pipeline": "basic"}, "not produced"),
    ],
)
def test_unsupported_implicit_identities_are_named_with_their_reason(
    identities, change, reason
):
    identity = copy.deepcopy(identities[0])
    assert identity["algorithm"] == "implicit_gemm"
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


@pytest.mark.parametrize(
    ("algorithm", "change", "reason"),
    [
        (
            "direct_depthwise_dgrad",
            {"algorithm": "direct_grouped_dgrad"},
            "no packaged",
        ),
        ("direct_depthwise_dgrad", {"direction": "direct_fwd"}, "direct_dgrad"),
        ("direct_depthwise_dgrad", {"knobs": '{"block_w": 4}'}, "are not"),
        ("direct_grouped_dgrad_mfma", {"cpg": 32, "kpg": 32}, ""),
        ("direct_dgrad_transpose", {"stride_h": 1, "stride_w": 1}, "cannot build"),
    ],
)
def test_unsupported_direct_identities_are_named_with_their_reason(
    identities, algorithm, change, reason
):
    identity = copy.deepcopy(next(i for i in identities if i["algorithm"] == algorithm))
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    if algorithm == "direct_grouped_dgrad_mfma":
        # c32k32 is a sweep row the curated catalog leaves out; it converts.
        assert spec is not None, why
        return
    assert spec is None and reason in why, why


def test_main_dedups_and_refuses_unsupported_unless_asked(tmp_path, identities, capsys):
    unsupported = dict(identities[0], split_k=2)
    entries = [identities[0], {"identity": identities[0]}, unsupported]
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps(entries))
    out = tmp_path / "out.yaml"
    argv = [
        "--rocke-root",
        str(_ROCKE),
        "--instances",
        str(instances),
        "--output",
        str(out),
    ]
    assert tool.main(argv) == 1
    assert not out.exists()
    assert "unsupported instance [2]" in capsys.readouterr().err
    assert tool.main(argv + ["--skip-unsupported"]) == 0
    assert "1 instance(s)" in capsys.readouterr().out


def test_stabilize_ids_derives_product_ids_and_matches_the_shipped_kdp(tmp_path):
    shipped = sorted(_SHIPPED_DIR.glob("*_direct.kdp.json"))[0]
    document = json.loads(shipped.read_text())
    for kernel in document["kernelDescriptors"]:
        kernel["id"] = str(uuid.uuid4())
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(json.dumps(document, indent=2) + "\n")
    assert tool.stabilize_ids(kdp) == len(document["kernelDescriptors"])
    assert json.loads(kdp.read_text()) == json.loads(shipped.read_text())
    # The forward tool's default product engine would derive other ids.
    for kernel in document["kernelDescriptors"]:
        assert fwd_tool.kernel_uuid(
            kernel["metadata"]["instance_id"], tool.PRODUCT_ENGINE
        ) != tool.kernel_uuid(kernel["metadata"]["instance_id"])


def test_dumped_config_loads_back(specs):
    text = tool.dump_config(tool.build_config(specs))
    assert yaml.safe_load(text) == tool.build_config(specs)
