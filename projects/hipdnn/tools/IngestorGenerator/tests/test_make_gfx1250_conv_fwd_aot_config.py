# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_gfx1250_conv_fwd_aot_config.py and the shipped
hipkernel:Gfx1250ConvFwdAot bundle."""

import collections
import copy
import json
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
import make_gfx1250_conv_fwd_aot_config as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_PROVIDER = (
    _REPO / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
)
_SLUG = "gfx1250_conv_fwd_aot"
_SHIPPED_DIR = _PROVIDER / f"descriptors/rocKE/{_SLUG}"
_OTHER_CONV_AOT_DIRS = (_PROVIDER / "descriptors/rocKE/gfx950_conv_dgrad_aot",)
_CENSUS = (
    _REPO / "dnn-providers/hip-kernel-provider/src/tests/engines"
    "/kernel_ingestor_engine/packs/TestGfx1250ConvFwdAotPacks.cpp"
)
_INSTANCES = [
    _CONFIGS / f"{_SLUG}_{dtype}.instances.json" for dtype in ("fp16", "bf16")
]
_FILE_SIZE_LIMIT = 50 * 1000 * 1000


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
def specs():
    out, rejected = tool.read_catalog(_INSTANCES)
    assert not rejected, rejected
    return out


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


def test_the_catalog_is_263_kernels_per_dtype(specs):
    """The deliverable set of the gfx1250 forward AOT catalog, one count per family."""
    by_dtype = collections.defaultdict(collections.Counter)
    for spec in specs:
        by_dtype[spec.dtype][spec.algorithm] += 1
    for dtype in ("fp16", "bf16"):
        assert by_dtype[dtype] == {
            "implicit_gemm": 12,
            "direct_fwd_wmma": 112,
            "direct_fwd_gstream_wmma": 26,
            "direct_fwd_tapouter_wmma": 14,
            "direct_fwd_smallc_wmma": 30,
            "direct_fwd_smallk_wmma": 4,
            "direct_dw_fwd_wmma": 37,
            "direct_depthwise": 4,
            "direct_depthwise_tiled": 22,
            "direct_depthwise_col": 1,
            "direct_depthwise_spatial": 1,
        }
    assert len(specs) == 526


def test_instance_id_is_the_identity_stable_hash(identities):
    from kernels.conv.kernel_identity import KernelIdentity

    for identity in identities:
        spec, _ = tool.instance_spec(identity)
        assert spec.instance_id == KernelIdentity.from_dict(identity).stable_hash()
        # The spec restores the identity it was built from.
        assert spec.to_identity() == KernelIdentity.from_dict(identity)


def test_the_shipped_bundle_is_its_catalog_with_instance_derived_ids(specs, shipped):
    """Every shard holds exactly the lists' instances of that shard; spec and metadata
    are the instance's fields in its family's order; ids are uuid5(NS, instance_id);
    the priority falls with the instance's list position within its dtype."""
    assert sorted(shipped) == sorted(s.instance_id for s in specs)
    priority = tool.priorities(specs)
    for spec in specs:
        shard, kernel = shipped[spec.instance_id]
        assert shard == tool.shard_name(spec)
        expected = {
            name: getattr(spec, name) for name in tool.spec_order(spec.algorithm)
        }
        assert kernel["kernel_source"]["spec"] == expected
        assert kernel["metadata"] == expected
        assert kernel["kernel_source"]["source"] == tool.ADAPTER_SOURCE
        assert kernel["kernel_source"]["builder"] == tool.ADAPTER_BUILDER
        assert kernel["id"] == str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id)
        )
        assert kernel["id"] == tool.kernel_uuid(spec.instance_id)
        assert kernel["name"] == tool._VARIANT_NAMES[spec.algorithm].format(
            dtype=spec.dtype, instance_id=spec.instance_id
        )
        assert kernel["priority"] == priority[spec.instance_id]


def test_the_list_order_is_the_priority_order(specs):
    """The engine tries the kernels a graph admits by priority where no ranker decides:
    the first entry of each dtype's list highest, each later one lower."""
    priority = tool.priorities(specs)
    for dtype in ("fp16", "bf16"):
        ordered = [priority[s.instance_id] for s in specs if s.dtype == dtype]
        assert ordered == list(range(len(ordered) - 1, -1, -1))


def test_the_shipped_bundle_is_sharded_below_the_hosting_size_limit(specs):
    # The generated KDPs and a promoted model's heuristics/ both live under DVC, each
    # pinned by its own pointer; the committed files are the small ones beside them.
    names = sorted(
        path.name
        for path in _SHIPPED_DIR.iterdir()
        if path.name not in ("artifacts", "heuristics")
    )
    assert {tool.shard_name(spec) for spec in specs} == {
        f"{dtype}_{group}"
        for dtype in ("fp16", "bf16")
        for group in ("implicit", "direct")
    }
    assert names == sorted(
        [f"{_SLUG}.{kind}.json" for kind in ("kmd", "udd", "ued", "uhd")]
        + [
            "kernel_dtype_matches_graph.umd.json",
            "artifacts.dvc",
            "heuristics.dvc",
            ".gitignore",
        ]
    )
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
    assert ued["name"] == tool.PRODUCT_ENGINE
    assert ued["graph_match"] == {"native": f"hipkernel.{_SLUG}.graph_match"}
    assert udd["dispatch_symbol"] == f"hipkernel.{_SLUG}.dispatch"
    assert uhd["native"] == {"symbol": f"hipkernel.{_SLUG}.score"}
    assert umd["match_symbol"] == f"hipkernel.{_SLUG}.kernel_match"
    assert ued["metadata"] == kmd["id"]
    # The native fallback scorer comes first; a promoted model appends its own id after
    # it instead of replacing it.
    assert list(ued["sort_kernel_catalog"]) == ["default"]
    default = ued["sort_kernel_catalog"]["default"]
    assert isinstance(default, list) and default[0] == uhd["id"]
    assert ued["knobs"] == list(tool.engine_knobs())
    kmd_names = [f["name"] for f in kmd["fields"]]
    assert kmd_names == [f["name"] for f in tool.kmd_fields()]
    assert set(ued["knobs"]) <= set(kmd_names)
    for restated in ("instance_id", "dtype", "filter_h", "stride_h", "pad_h", "cpg"):
        assert restated not in ued["knobs"]


def test_the_config_borrows_no_native_symbols(specs):
    config = tool.build_config(specs)
    assert "native_symbols_from" not in config["engine"]
    assert config["workspace_policy"] == "none"
    other = tool.build_config(specs[:3], "hipkernel:Gfx1250ConvFwdAotSeed")
    assert other["engine"]["native_symbols_from"] == tool.PRODUCT_ENGINE
    assert "native_symbols_from_slug" not in other["engine"]
    slugged = tool.build_config(
        specs[:3], "hipkernel:Gfx1250ConvFwdAotSeed", native_symbols_from_slug="fwd"
    )
    assert slugged["engine"]["native_symbols_from_slug"] == "fwd"
    # The envelope: one native-heuristic engine, no workspace, one shared-shape graph
    # match, every KMD field a metadata field bound to the spec field of its name.
    metadata = [f["name"] for f in tool.kmd_fields()]
    assert {k: v for k, v in config.items() if k != "packs"} == {
        "dialect": "packaged",
        "authored_subpath": f"rocKE/{_SLUG}",
        "engine": {
            "name": tool.PRODUCT_ENGINE,
            "sdk_version": "1.0.0",
            "knobs": list(tool.engine_knobs()),
            "heuristic": "native",
        },
        "kmd_fields": tool.kmd_fields(),
        "kernel_source_kind": "rocke",
        "workspace_policy": "none",
        "graph_match": {"shape": "shared_shape", "discriminator": "none"},
        "specialization": {
            "metadata_fields": metadata,
            "matcher_only_fields": [],
            "bindings": {name: {"field": name} for name in metadata},
        },
    }
    assert tool.kernel_uuid(specs[0].instance_id, "hipkernel:X") != tool.kernel_uuid(
        specs[0].instance_id
    )


def test_generated_config_expands_to_one_kernel_per_instance(tmp_path, specs):
    path = tmp_path / "fwd.yaml"
    path.write_text(tool.dump_config(tool.build_config(specs)))
    config = load_config(path)
    assert config.engine.name == tool.PRODUCT_ENGINE
    assert config.is_multi_pack and not config.has_operation_matchers
    kernels = [k for pack in config.packs for k in pack.kernels]
    assert len(kernels) == len(specs)
    priority = tool.priorities(specs)
    assert sorted(k.priority for k in kernels) == sorted(priority.values())


def test_bundle_ids_are_unique_across_the_conv_aot_bundles():
    """The conv AOT bundles load into one process: no document or kernel id may repeat
    across them."""
    ids = {}
    for folder in (*_OTHER_CONV_AOT_DIRS, _SHIPPED_DIR):
        for path in [*folder.glob("*.json"), *folder.glob("artifacts/*.kdp.json")]:
            document = json.loads(path.read_text())
            found = [document["id"]] + [
                kernel["id"] for kernel in document.get("kernelDescriptors", [])
            ]
            for value in found:
                assert ids.setdefault(value, path) == path, value


_CAPABILITY = (
    "dtype",
    "algorithm",
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


_EXACT_WHEN_DIRECT = ("filter_h", "filter_w", "stride_h", "stride_w", "pad_h", "pad_w",
                     "dilation_h", "dilation_w")  # fmt: skip


def _baked(values: dict, field: str):
    """The value ``values`` admits for ``field``, or None for any (read at run time). A
    direct kernel bakes its filter, stride, padding and dilation exactly (0 padding is a
    value); every other capability, and all of an implicit GEMM's, is runtime when 0."""
    if values["algorithm"] != "implicit_gemm" and field in _EXACT_WHEN_DIRECT:
        return values[field]
    return values[field] or None


def _disjoint(left: dict, right: dict) -> bool:
    """No graph is served by both: their dtypes differ, or some capability field admits
    one value in each, and the two values differ."""
    if left["dtype"] != right["dtype"]:
        return True
    for field in _CAPABILITY[2:]:
        a, b = _baked(left, field), _baked(right, field)
        if a is not None and b is not None and a != b:
            return True
    return False


def test_two_kernels_one_graph_can_meet_differ_in_their_knobs(shipped):
    """GenericPlanBuilder refuses a graph whose matched catalog repeats a UED knob tuple.
    Kernels that share a tuple must therefore never serve the same graph: within a
    dtype, some capability baked in both tells them apart."""
    defaults = {f["name"]: f.get("default_value") for f in _document("kmd")["fields"]}
    by_tuple = collections.defaultdict(list)
    for _, kernel in shipped.values():
        values = dict(defaults, **kernel["metadata"])
        by_tuple[tuple(values[k] for k in tool.engine_knobs())].append(values)
    shared = 0
    for kernels in by_tuple.values():
        for i, left in enumerate(kernels):
            for right in kernels[i + 1 :]:
                shared += 1
                assert _disjoint(left, right), (
                    left["instance_id"],
                    right["instance_id"],
                )
    # Not vacuous: the direct families' baked filters do share tuples.
    assert shared > 0


def _bound_models() -> list:
    """The heuristic.uhd.json of every trained model the UED's sort_kernel_catalog
    binds, as (path, document); skips when the DVC-tracked heuristics/ are not pulled.
    """
    bound = {
        model_id
        for ids in _document("ued")["sort_kernel_catalog"].values()
        for model_id in ids
    }
    found = []
    for path in sorted((_SHIPPED_DIR / "heuristics").glob("**/heuristic.uhd.json")):
        document = json.loads(path.read_text())
        if document["id"] in bound:
            found.append((path, document))
    if not found:
        pytest.skip(f"heuristics/ not pulled, run `dvc pull` in {_SHIPPED_DIR}")
    return found


def _unencoded_kernel_values(model: dict, shipped: dict) -> dict:
    """``(feature, value) -> [instance_id, ...]`` for every shipped kernel whose KMD
    field read as a categorical ``$kernel.<field>`` feature holds a value with no code
    in the model's categorical_encoding. A field a kernel's family does not state is
    read at its KMD default."""
    defaults = {f["name"]: f.get("default_value") for f in _document("kmd")["fields"]}
    missing = collections.defaultdict(list)
    for feature, codes in model["categorical_encoding"].items():
        if not feature.startswith("$kernel."):
            continue
        field = feature[len("$kernel.") :]
        assert field in defaults, feature
        for instance_id, (_, kernel) in shipped.items():
            value = kernel["metadata"].get(field, defaults[field])
            if str(value) not in codes:
                missing[(feature, value)].append(instance_id)
    return dict(missing)


def test_the_bound_ranker_encodes_every_shipped_kernels_categorical_values(shipped):
    """The trained ranker the UED binds encodes every value a shipped kernel carries in
    a categorical field it reads. Otherwise the feature extractor throws on that kernel
    and the engine silently ranks every graph admitting it by declared priority."""
    if not shipped:
        pytest.skip(f"artifacts/ not pulled, run `dvc pull` in {_SHIPPED_DIR}")
    for path, model in _bound_models():
        assert any(k.startswith("$kernel.") for k in model["categorical_encoding"])
        assert _unencoded_kernel_values(model, shipped) == {}, path


def test_the_census_names_every_pack_and_kernel(shipped):
    text = _CENSUS.read_text()
    for _, kernel in shipped.values():
        assert f'"{kernel["name"]}"' in text
    for kdp in (_SHIPPED_DIR / "artifacts").glob("*.kdp.json"):
        assert f'"{json.loads(kdp.read_text())["name"]}"' in text
    assert f'"{tool.PRODUCT_ENGINE}"' in text
    assert f"hipkernel.{_SLUG}.score" in text
    assert f"knobs.size(), {len(tool.engine_knobs())}U" in text


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"arch": "gfx950"}, "arch="),
        ({"direction": "dgrad"}, "is not a dgrad algorithm"),
        ({"dtype_d": "fp32"}, "mixed operand dtypes"),
        ({"split_k": 2}, "split_k"),
        ({"llvm_flavor": "llvm22"}, "llvm_flavor"),
        ({"algorithm": "winograd"}, "no packaged launch"),
        ({"knobs": '{"unknown": 1}'}, "knob unknown"),
    ],
)
def test_unsupported_identities_are_named_with_their_reason(identities, change, reason):
    identity = copy.deepcopy(identities[0])
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


def test_a_bool_knob_stated_false_keeps_its_identity(identities):
    """An instance may state a bool knob false; the spec keeps that apart from unset, so
    the restored identity (and its stable_hash) is the listed one."""
    from builders.gfx1250.conv_fwd_aot import BOOL_FALSE

    stated_false = [
        i
        for i in identities
        if json.loads(i.get("knobs") or "{}").get("unroll_rows") is False
    ]
    assert stated_false
    for identity in stated_false:
        spec, why = tool.instance_spec(identity)
        assert spec is not None, why
        assert spec.unroll_rows == BOOL_FALSE


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
    shipped = sorted((_SHIPPED_DIR / "artifacts").glob("*_direct.kdp.json"))[0]
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
