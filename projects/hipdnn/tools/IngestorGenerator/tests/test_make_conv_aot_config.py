# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_conv_aot_config.py and the shipped hipkernel:Gfx950ConvFwdAot bundle."""

import copy
import dataclasses
import json
import os
import re
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
import make_conv_aot_config as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_SEED_ENGINE = "hipkernel:Gfx950ConvFwdAotSeed"
_INSTANCES = _CONFIGS / "gfx950_conv_fwd_aot_seed.instances.json"
_CONFIG = _CONFIGS / "gfx950_conv_fwd_aot_seed.yaml"
_PROVIDER = (
    _REPO / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
)
_SHIPPED_DIR = _PROVIDER / "descriptors/rocKE/gfx950_conv_fwd_aot"
_SEED_DIR = _PROVIDER / "test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed"
#: One small shipped shard, for the tests that rewrite a KDP.
_SEED_KDP = _SEED_DIR / "gfx950_conv_fwd_aot_seed_fp16.kdp.json"
#: GitHub warns about any file above 50 MB; every emitted file stays below it.
_FILE_SIZE_LIMIT = 50 * 1000 * 1000

#: Each bundle the tool emits: (engine, its descriptor folder, its instance lists,
#: its sharding).
_BUNDLES = {
    "product": (
        tool.PRODUCT_ENGINE,
        _SHIPPED_DIR,
        [
            _CONFIGS / f"gfx950_conv_fwd_aot_{dtype}.instances.json"
            for dtype in ("bf16", "fp16")
        ],
        "k-loop",
    ),
    "seed": (_SEED_ENGINE, _SEED_DIR, [_INSTANCES], "dtype"),
}
#: Worker processes that convert a catalog.
_JOBS = min(32, os.cpu_count() or 1)


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    """Bind rocKE for this module only: test_dispatch_parity asserts what a provider
    root ADDS to sys.path, so a binding left behind at collection would hide it."""
    before = list(sys.path)
    tool._bind_rocke(_ROCKE)
    yield
    sys.path[:] = before


def _specs(identities):
    specs = []
    for spec, reason in tool.instance_specs(identities, _JOBS):
        assert spec is not None, reason
        specs.append(spec)
    return sorted(specs, key=lambda s: (s.dtype, s.instance_id))


def _listed(paths):
    return [identity for path in paths for identity in json.loads(path.read_text())]


@pytest.fixture(scope="module")
def seed_identities():
    return json.loads(_INSTANCES.read_text())


@pytest.fixture(scope="module")
def seed_specs(seed_identities):
    return _specs(seed_identities)


@pytest.fixture(scope="module")
def catalogs():
    """The specs of the seed's and the product's committed instance lists."""
    return {name: _specs(_listed(_BUNDLES[name][2])) for name in sorted(_BUNDLES)}


@pytest.fixture(scope="module", params=sorted(_BUNDLES))
def bundle(request, catalogs):
    """``(engine, descriptor folder, specs, sharding)`` of each emitted bundle."""
    engine, folder, _, sharding = _BUNDLES[request.param]
    return engine, folder, catalogs[request.param], sharding


def test_seed_config_is_what_the_tool_generates(seed_specs):
    assert yaml.safe_load(_CONFIG.read_text()) == tool.build_config(
        seed_specs, _SEED_ENGINE
    )


def test_shipped_bundle_is_its_catalog_with_instance_derived_ids(bundle):
    engine, folder, specs, sharding = bundle
    slug = tool.engine_slug(engine)
    shipped = {}
    for kdp in sorted(folder.glob("*.kdp.json")):
        document = json.loads(kdp.read_text())
        shard = kdp.name[: -len(".kdp.json")][len(slug) + 1 :]
        assert document["name"] == f"hipkernel:{slug}_{shard}"
        for kernel in document["kernelDescriptors"]:
            shipped[kernel["metadata"]["instance_id"]] = (shard, kernel)
    assert sorted(shipped) == sorted(s.instance_id for s in specs)
    for spec in specs:
        shard, kernel = shipped[spec.instance_id]
        assert shard == tool.shard_name(spec, sharding)
        expected = {name: getattr(spec, name) for name in tool.spec_order(spec)}
        assert kernel["kernel_source"]["spec"] == expected
        assert kernel["metadata"] == expected
        assert kernel["kernel_source"]["builder"] == "build_gfx950_conv_fwd_aot"
        assert kernel["id"] == tool.kernel_uuid(spec.instance_id, engine)


def test_shipped_bundle_is_sharded_below_the_hosting_size_limit(bundle):
    engine, folder, specs, sharding = bundle
    slug = tool.engine_slug(engine)
    names = sorted(path.name for path in folder.iterdir())
    shards = {tool.shard_name(spec, sharding) for spec in specs}
    assert names == sorted(
        [f"{slug}.{kind}.json" for kind in ("kmd", "udd", "ued", "uhd")]
        + [f"{slug}_{shard}.kdp.json" for shard in shards]
        + ["kernel_dtype_matches_graph.umd.json"]
    )
    matcher = json.loads((folder / "kernel_dtype_matches_graph.umd.json").read_text())
    for path in folder.iterdir():
        assert path.stat().st_size < _FILE_SIZE_LIMIT, path.name
    # No operation-scoped matcher: every shard names the shared kernel matcher only.
    for kdp in folder.glob("*.kdp.json"):
        assert json.loads(kdp.read_text())["matchers"] == [matcher["id"]]


def test_committed_instance_lists_and_census_are_below_the_hosting_size_limit():
    census = (
        _REPO / "dnn-providers/hip-kernel-provider/src/tests/engines"
        "/kernel_ingestor_engine/packs/TestGfx950ConvFwdAotPacks.cpp"
    )
    for path in _BUNDLES["product"][2] + [census]:
        assert path.stat().st_size < _FILE_SIZE_LIMIT, path.name


def test_every_bundle_runs_the_shipped_engines_native_code():
    for engine, folder, _, _ in _BUNDLES.values():
        ued = json.loads((folder / f"{tool.engine_slug(engine)}.ued.json").read_text())
        assert ued["name"] == engine
        assert ued["graph_match"] == {
            "native": "hipkernel.gfx950_conv_fwd_aot.graph_match"
        }
        udd = json.loads((folder / f"{tool.engine_slug(engine)}.udd.json").read_text())
        assert udd["dispatch_symbol"] == "hipkernel.gfx950_conv_fwd_aot.dispatch"


def test_the_seed_is_a_second_engine_with_its_own_ids(seed_specs):
    config = tool.build_config(seed_specs, _SEED_ENGINE)
    assert config["engine"]["name"] == _SEED_ENGINE
    assert config["engine"]["native_symbols_from"] == tool.PRODUCT_ENGINE
    assert config["authored_subpath"] == "rocKE/gfx950_conv_fwd_aot_seed"
    assert "native_symbols_from" not in tool.build_config(seed_specs)["engine"]
    # Both engines load into one process, so neither a document id nor a kernel id
    # may repeat across their bundles.
    ids = {}
    for _, folder, _, _ in _BUNDLES.values():
        for path in folder.glob("*.json"):
            document = json.loads(path.read_text())
            found = [document["id"]] + [
                kernel["id"] for kernel in document.get("kernelDescriptors", [])
            ]
            for value in found:
                assert ids.setdefault(value, path) == path, value
    for spec in seed_specs:
        assert tool.kernel_uuid(spec.instance_id, _SEED_ENGINE) != tool.kernel_uuid(
            spec.instance_id
        )


def test_config_shards_the_catalog_into_one_pack_per_dtype(tmp_path, seed_specs):
    config = tool.build_config(seed_specs)
    assert [pack["name"] for pack in config["packs"]] == ["bf16", "fp16"]
    for pack in config["packs"]:
        assert "discriminator" not in pack
        dtypes = {
            shape["dtype"]
            for variant in pack["variants"]
            for shape in variant["shapes"]
        }
        assert dtypes == {pack["name"]}
    path = tmp_path / "aot.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    loaded = load_config(path)
    assert loaded.is_multi_pack and not loaded.has_operation_matchers
    assert [loaded.kdp_stem(pack) for pack in loaded.packs] == [
        "gfx950_conv_fwd_aot_bf16",
        "gfx950_conv_fwd_aot_fp16",
    ]


def test_k_loop_sharding_splits_by_dtype_pipeline_and_k_loop(seed_specs):
    config = tool.build_config(seed_specs, sharding="k-loop")
    expected = set()
    for spec in seed_specs:
        if spec.is_direct:
            expected.add(f"{spec.dtype}_direct")
        else:
            loop = "async" if spec.async_dma else "unroll" if spec.unroll_k else "plain"
            expected.add(f"{spec.dtype}_{spec.pipeline}_{loop}")
    assert [pack["name"] for pack in config["packs"]] == sorted(expected)
    for pack in config["packs"]:
        assert "discriminator" not in pack
        for variant in pack["variants"]:
            for shape in variant["shapes"]:
                spec = next(
                    s for s in seed_specs if s.instance_id == shape["instance_id"]
                )
                assert tool.shard_name(spec, "k-loop") == pack["name"]
    assert sum(
        len(variant["shapes"])
        for pack in config["packs"]
        for variant in pack["variants"]
    ) == len(seed_specs)
    with pytest.raises(ValueError, match="unknown sharding"):
        tool.shard_name(seed_specs[0], "tile")


def test_instance_specs_in_parallel_are_the_serial_ones(seed_identities):
    identities = seed_identities * 40
    assert tool.instance_specs(identities, 4) == tool.instance_specs(identities, 1)


def test_instance_id_is_the_identity_stable_hash(seed_identities):
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in seed_identities:
        spec, _ = tool.instance_spec(identity)
        assert spec.instance_id == KernelIdentity.from_dict(identity).stable_hash()


def test_generated_config_expands_to_one_kernel_per_instance(tmp_path, seed_specs):
    path = tmp_path / "aot.yaml"
    path.write_text(yaml.safe_dump(tool.build_config(seed_specs), sort_keys=False))
    config = load_config(path)
    assert config.engine.name == "hipkernel:Gfx950ConvFwdAot"
    assert sum(len(pack.kernels) for pack in config.packs) == len(seed_specs)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"direction": "wgrad"}, "forward implicit-GEMM"),
        ({"arch": "gfx942"}, "forward implicit-GEMM"),
        ({"algorithm": "direct_grouped"}, "forward implicit-GEMM"),
        ({"dtype_d": "fp32"}, "mixed operand dtypes"),
        ({"split_k": 2}, "split_k"),
        ({"chiplet_swizzle": True}, "chiplet_swizzle"),
        ({"is_3d": True}, "is_3d"),
        ({"pipeline": "wavelet"}, "pipelines"),
    ],
)
def test_unsupported_identities_are_named_with_their_reason(
    seed_identities, change, reason
):
    identity = copy.deepcopy(seed_identities[0])
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


def test_main_dedups_and_refuses_unsupported_unless_asked(
    tmp_path, seed_identities, capsys
):
    unsupported = dict(seed_identities[0], split_k=2)
    entries = [seed_identities[0], {"identity": seed_identities[0]}, unsupported]
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
    assert (
        len(
            yaml.safe_load(out.read_text().split("\n", 1)[1])["packs"][0]["variants"][
                0
            ]["shapes"]
        )
        == 1
    )


def test_stabilize_ids_rewrites_kernel_ids_and_refuses_referenced_ones(tmp_path):
    kdp = tmp_path / "x.kdp.json"
    document = json.loads(_SEED_KDP.read_text())
    for kernel in document["kernelDescriptors"]:
        kernel["id"] = str(uuid.uuid4())
    kdp.write_text(json.dumps(document))
    assert tool.stabilize_ids(kdp, _SEED_ENGINE) == len(document["kernelDescriptors"])
    assert json.loads(kdp.read_text()) == json.loads(_SEED_KDP.read_text())

    first = document["kernelDescriptors"][0]
    document["provenance"] = {"note": first["id"]}
    kdp.write_text(json.dumps(document))
    with pytest.raises(SystemExit, match="refusing to rewrite a referenced id"):
        tool.stabilize_ids(kdp)


# ---- Direct identities ----


@pytest.fixture(scope="module")
def forward_direct_identities():
    """Every forward identity of rocKE's direct sweep, both dtypes."""
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx("gfx950")
    return [
        job.identity.to_dict()
        for dtype in ("fp16", "bf16")
        for job in enumerate_direct_jobs(
            arch="gfx950", target=target, directions=["fwd"], dtype=dtype
        )
    ]


#: Forward identities per dtype and algorithm in rocKE's direct sweep grid.
_DIRECT_GRID = {
    "direct_depthwise": 54,
    "direct_depthwise_spatial": 30,
    "direct_depthwise_tiled": 1536,
    "direct_depthwise_col": 540,
    "direct_grouped": 80,
    "direct_nongrouped": 3058,
}


def test_direct_identities_are_packaged_exactly_when_the_launch_runs_them(
    forward_direct_identities,
):
    from builders.common.convolution_forward import AOT_DIRECT_ALGORITHMS

    assert tool.PACKAGED_DIRECT_ALGORITHMS == AOT_DIRECT_ALGORITHMS
    assert tool.DIRECT_ALGORITHMS.items() <= tool.PACKAGED_DIRECT_ALGORITHMS.items()
    accepted = {}
    for identity in forward_direct_identities:
        spec, why = tool.instance_spec(identity)
        if identity["algorithm"] not in tool.PACKAGED_DIRECT_ALGORITHMS:
            assert spec is None and "has no packaged launch" in why, why
            continue
        assert spec is not None, why
        assert spec.is_direct
        assert (
            spec.direct_variant
            == tool.PACKAGED_DIRECT_ALGORITHMS[identity["algorithm"]]
        )
        key = (spec.dtype, identity["algorithm"])
        accepted[key] = accepted.get(key, 0) + 1
    # Every forward identity of the packaged algorithms, in both dtypes.
    assert accepted == {
        (dtype, algorithm): count
        for dtype in ("fp16", "bf16")
        for algorithm, count in _DIRECT_GRID.items()
        if algorithm in tool.PACKAGED_DIRECT_ALGORITHMS
    }


def test_a_direct_instance_carries_the_identitys_capabilities_and_knobs(
    forward_direct_identities,
):
    from benchmarks.common.direct_kernel_sweep import direct_job_for_identity
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in forward_direct_identities[::7]:
        if identity["algorithm"] not in tool.PACKAGED_DIRECT_ALGORITHMS:
            continue
        spec, _ = tool.instance_spec(identity)
        assert spec.instance_id == KernelIdentity.from_dict(identity).stable_hash()
        assert (
            spec.filter_h,
            spec.stride_h,
            spec.pad_h,
            spec.dilation_h,
            spec.dilation_w,
        ) == (
            identity["filter_h"],
            identity["stride_h"],
            identity["pad_h"],
            identity["dilation_h"],
            identity["dilation_w"],
        )
        # The knobs round-trip, and so does the job the direct sweep builds from them.
        assert spec.direct_knobs() == json.loads(identity["knobs"])
        job = direct_job_for_identity(KernelIdentity.from_dict(identity))
        assert spec.probe_problem().cpg == (job.caps["cpg"] or 256)
        assert spec.probe_problem().kpg == (job.caps["kpg"] or 256)
        # The implicit-GEMM fields are the identity's placeholders.
        assert (spec.tile_m, spec.pipeline, spec.vector_size_c, spec.grouped) == (
            0,
            "direct",
            0,
            0,
        )
        # Only the variant's own knobs leave their KMD default.
        own = set(tool.DIRECT_VARIANT_FIELDS[spec.direct_variant])
        for name, default in tool.DIRECT_KNOB_FIELDS.items():
            if name not in own:
                assert getattr(spec, name) == default, (name, identity["algorithm"])


@pytest.mark.parametrize(
    ("algorithm", "change", "reason"),
    [
        ("direct_depthwise", {"arch": "gfx942"}, "gfx950"),
        (
            "direct_depthwise",
            {"algorithm": "direct_depthwise_dgrad"},
            "no packaged launch",
        ),
        ("direct_depthwise", {"dtype_d": "fp32"}, "mixed operand dtypes"),
        ("direct_depthwise", {"cpg": 2, "kpg": 2}, "cannot build"),
        ("direct_depthwise", {"filter_w": 5, "pad_w": 2}, "cannot build"),
        ("direct_depthwise", {"pad_h": 0, "pad_w": 0}, "cannot build"),
        ("direct_depthwise", {"split_k": 2}, "cannot build"),
        ("direct_depthwise", {"knobs": '{"block_waves": 1}'}, "knobs"),
        (
            "direct_depthwise",
            {"knobs": '{"block_w": 4, "block_waves": 1, "x": 1}'},
            "x",
        ),
        (
            "direct_depthwise",
            {"knobs": '{"block_w": 4, "block_waves": 99}'},
            "block_waves",
        ),
        ("direct_depthwise_tiled", {"filter_h": 33, "filter_w": 33}, "cannot build"),
        ("direct_depthwise_col", {"pad_h": 2, "pad_w": 2}, "cannot build"),
        ("direct_grouped", {"cpg": 6, "kpg": 6}, "cannot build"),
        ("direct_grouped", {"stride_h": 2, "stride_w": 2}, "cannot build"),
        ("direct_nongrouped", {"stride_h": 3, "stride_w": 3}, "cannot build"),
        ("direct_nongrouped", {"cpg": 64, "kpg": 64}, "cannot build"),
    ],
)
def test_unsupported_direct_identities_are_named_with_their_reason(
    forward_direct_identities, algorithm, change, reason
):
    identity = copy.deepcopy(
        next(i for i in forward_direct_identities if i["algorithm"] == algorithm)
    )
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


def test_direct_instances_are_variants_stating_their_own_fields(
    seed_specs, forward_direct_identities
):
    # The seed's implicit-GEMM instances and the first identity of each direct
    # algorithm in each dtype.
    firsts = {}
    for identity in forward_direct_identities:
        if identity["algorithm"] in tool.PACKAGED_DIRECT_ALGORITHMS:
            firsts.setdefault((identity["dtype_a"], identity["algorithm"]), identity)
    config = tool.build_config(
        [s for s in seed_specs if not s.is_direct] + _specs(firsts.values())
    )
    for pack in config["packs"]:
        implicit, *direct = pack["variants"]
        assert implicit["spec_order"] == list(tool.SPEC_ORDER)
        assert not any("kernel_family" in shape for shape in implicit["shapes"])
        variants = {}
        for variant in direct:
            names = {shape["direct_variant"] for shape in variant["shapes"]}
            assert all(shape["kernel_family"] == 1 for shape in variant["shapes"])
            (own,) = {tool.DIRECT_VARIANT_FIELDS[name] for name in names}
            assert variant["spec_order"] == list(tool.DIRECT_SPEC_ORDER + own)
            variants[tuple(sorted(names))] = variant["name"]
        # std and spatial keep the variant they had before the other direct kernels.
        assert variants == {
            ("spatial", "std"): "gfx950_conv_fwd_aot_dw_{dtype}_{instance_id}",
            ("tiled",): "gfx950_conv_fwd_aot_dwtiled_{dtype}_{instance_id}",
            ("col",): "gfx950_conv_fwd_aot_dwcol_{dtype}_{instance_id}",
            ("grouped",): "gfx950_conv_fwd_aot_grouped_{dtype}_{instance_id}",
            ("nongrouped",): "gfx950_conv_fwd_aot_nongrouped_{dtype}_{instance_id}",
        }
    # Every direct field is builder_default, so implicit-GEMM descriptors (and std and
    # spatial ones, for the block knobs) leave them to the KMD.
    defaults = {
        field["name"]: field["default_value"]
        for field in config["kmd_fields"]
        if field.get("builder_default")
    }
    assert defaults == {**tool.DIRECT_FIELDS, **tool.DIRECT_KNOB_FIELDS}
    from builders.common.convolution_forward import Gfx950ConvFwdAotSpec

    assert {
        f.name: f.default
        for f in dataclasses.fields(Gfx950ConvFwdAotSpec)
        if f.name in defaults
    } == defaults
    assert [f.name for f in dataclasses.fields(Gfx950ConvFwdAotSpec)] == list(
        tool.SPEC_ORDER + tuple(defaults)
    )
    assert config["engine"]["knobs"] == ["tile_k", "kernel_family"]


def test_the_seed_runs_every_direct_variant_knob(
    seed_identities, forward_direct_identities
):
    """The seed pins each of its instances on the GPU, so per dtype it holds at least 20
    instances of each tiled, col, grouped and non-grouped kernel, together every value
    of every knob the direct sweep builds for it, and every grouped channel count and
    non-grouped dilation."""
    for dtype in ("fp16", "bf16"):
        for algorithm in (
            "direct_depthwise_tiled",
            "direct_depthwise_col",
            "direct_grouped",
            "direct_nongrouped",
        ):

            def values(identities):
                found = set()
                for identity in identities:
                    if (identity["dtype_a"], identity["algorithm"]) == (
                        dtype,
                        algorithm,
                    ):
                        found |= set(json.loads(identity["knobs"]).items())
                        found.add(
                            (
                                "caps",
                                identity["cpg"],
                                identity["dilation_h"],
                                identity["dilation_w"],
                            )
                        )
                return found

            seeded = [
                i
                for i in seed_identities
                if (i["dtype_a"], i["algorithm"]) == (dtype, algorithm)
            ]
            assert len(seeded) >= 20, (dtype, algorithm)
            assert values(forward_direct_identities) == values(seed_identities), (
                dtype,
                algorithm,
            )


def _stabilize_ids_one_kernel_at_a_time(kdp_path):
    """The original quadratic algorithm, kept as the reference for the linear one."""
    document = json.loads(kdp_path.read_text())
    text = kdp_path.read_text()
    for kernel in tool._kernels(document):
        new_id = str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, kernel["metadata"]["instance_id"])
        )
        assert text.count(kernel["id"]) == 1
        text = text.replace(kernel["id"], new_id)
    kdp_path.write_text(text)


def test_stabilize_ids_matches_the_one_kernel_at_a_time_rewrite(tmp_path):
    document = json.loads(_SEED_KDP.read_text())
    for kernel in document["kernelDescriptors"]:
        kernel["id"] = str(uuid.uuid4())
    # An unrelated UUID elsewhere in the file is left alone.
    document["provenance"] = {"note": str(uuid.uuid4())}
    text = json.dumps(document, indent=2)
    linear, reference = tmp_path / "linear.kdp.json", tmp_path / "reference.kdp.json"
    linear.write_text(text)
    reference.write_text(text)

    assert tool.stabilize_ids(linear) == len(document["kernelDescriptors"])
    _stabilize_ids_one_kernel_at_a_time(reference)
    assert linear.read_bytes() == reference.read_bytes()
    assert linear.read_text() != text

    # Idempotent: stabilized ids map to themselves.
    tool.stabilize_ids(linear)
    assert linear.read_bytes() == reference.read_bytes()


def test_stabilize_ids_refuses_a_kernel_id_shared_by_two_kernels(tmp_path):
    document = json.loads(_SEED_KDP.read_text())
    first, second = document["kernelDescriptors"][:2]
    second["id"] = first["id"]
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(json.dumps(document))
    with pytest.raises(SystemExit, match="appears 2 times"):
        tool.stabilize_ids(kdp)


@pytest.mark.skipif(
    not getattr(yaml, "__with_libyaml__", False), reason="PyYAML without libyaml"
)
def test_libyaml_and_python_dumpers_write_the_same_config(seed_specs):
    config = tool.build_config(seed_specs)
    text = tool.dump_config(config, yaml.CSafeDumper)
    assert text == tool.dump_config(config, yaml.SafeDumper)
    assert text == tool.dump_config(config)


def test_dumped_config_has_one_line_per_shape_and_loads_back(seed_specs):
    config = tool.build_config(seed_specs)
    text = tool.dump_config(config, yaml.SafeDumper)
    assert yaml.safe_load(text) == config
    rows = [line for line in text.splitlines() if line.startswith("    - {dtype: ")]
    assert len(rows) == len(seed_specs)


# ---- The GPU suites' selected counts ----

_GPU_SUITE = (
    _REPO
    / "dnn-providers/hip-kernel-provider/src/integration_tests"
    / "kernel_ingestor_engine/IntegrationGpuGfx950ConvFwd.cpp"
)
#: One aotGraphs() row: name, problem (empty = ConvProblem's defaults), then the
#: seed's direct and implicit counts, the full catalog's direct count and its
#: implicit counts in fp16 and in bf16.
_GRAPH_ROW = re.compile(
    r'^\s*\{"(\w+)", \{([\d, ]*)\}, (\d+), (\d+), (\d+), (\d+), (\d+)\},$',
    re.MULTILINE,
)
#: One aotSeedDirectGraphs() row: name, problem, the seed's direct and implicit counts.
_SEED_GRAPH_ROW = re.compile(
    r'^\s*\{"(\w+)", \{([\d, ]*)\}, (\d+), (\d+)\},$', re.MULTILINE
)
#: ConvProblem's defaults, in its field order: n, c, k, hi, wi, y, x, strides,
#: paddings, dilations, groups.
_DEFAULT_PROBLEM = (2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1)


def _runtime_problem(problem, dtype):
    from builders.common.convolution_forward import Gfx950ConvFwdSpec

    n, c, k, hi, wi, y, x, sh, sw, ph, pw, dh, dw, groups = problem
    return Gfx950ConvFwdSpec(
        N=n, C=c, K=k, Hi=hi, Wi=wi, Y=y, X=x, sH=sh, sW=sw, pH=ph, pW=pw,
        dH=dh, dW=dw, groups=groups, dtype=dtype,
    )  # fmt: skip


def _serving_counts(specs, problem, dtype):
    """How many instances of each family (direct, implicit GEMM) serve the problem:
    the capability, gfx950_conv_fwd_aot_serves."""
    from builders.common.convolution_forward import gfx950_conv_fwd_aot_serves

    runtime = _runtime_problem(problem, dtype)
    counts = [0, 0]
    for spec in specs:
        if spec.dtype == dtype and gfx950_conv_fwd_aot_serves(spec, runtime)[0]:
            counts[0 if spec.is_direct else 1] += 1
    return tuple(counts)


def _selected(specs, problem, dtype):
    """The instances the engine selects for the problem: rocKE's runtime selection among
    those that serve it, gfx950_conv_fwd_aot_runtime_candidates."""
    from builders.common.convolution_forward import (
        gfx950_conv_fwd_aot_runtime_candidates,
    )

    return gfx950_conv_fwd_aot_runtime_candidates(
        _runtime_problem(problem, dtype), [s for s in specs if s.dtype == dtype]
    )


def _selected_counts(specs, problem, dtype):
    selected = _selected(specs, problem, dtype)
    direct = sum(spec.is_direct for spec in selected)
    return direct, len(selected) - direct


def _gpu_suite_graphs():
    rows = _GRAPH_ROW.findall(_GPU_SUITE.read_text())
    assert len(rows) >= 19, "aotGraphs() rows not found; keep one row per line"
    for name, problem, *row in rows:
        values = tuple(int(v) for v in problem.split(",")) if problem else ()
        if len(values) == 13:
            values += (1,)
        yield name, values or _DEFAULT_PROBLEM, tuple(int(v) for v in row)


def _seed_direct_graphs():
    """aotSeedDirectGraphs(): name, problem, the seed's direct and implicit counts."""
    text = _GPU_SUITE.read_text()
    start = text.index("aotSeedDirectGraphs()\n{")
    rows = _SEED_GRAPH_ROW.findall(text[start : text.index("return s_graphs", start)])
    assert len(rows) >= 50, "aotSeedDirectGraphs() rows not found; one row per line"
    for name, problem, *row in rows:
        values = tuple(int(v) for v in problem.split(","))
        yield name, values, tuple(int(v) for v in row)


def test_gpu_suite_selected_counts_are_the_python_runtime_candidates(catalogs):
    """IntegrationGpuGfx950ConvFwd.cpp states, per graph, how many instances of each
    family the seed and the full catalog select; the engine's kernel matcher and
    candidate filter are checked against those numbers on the GPU, so they must be what
    the Python runtime selection counts over the committed lists, in both dtypes. Every
    graph has a selected instance in both, and selection only narrows what serves."""
    for name, problem, row in _gpu_suite_graphs():
        seed_direct, seed_implicit, direct, fp16, bf16 = row
        for dtype, implicit in (("fp16", fp16), ("bf16", bf16)):
            stated = (seed_direct, seed_implicit, direct, implicit)
            counted = _selected_counts(catalogs["seed"], problem, dtype)
            counted += _selected_counts(catalogs["product"], problem, dtype)
            assert counted == stated, (name, dtype)
            assert sum(counted[:2]) > 0 and sum(counted[2:]) > 0, (name, dtype)
            serving = _serving_counts(catalogs["seed"], problem, dtype)
            serving += _serving_counts(catalogs["product"], problem, dtype)
            assert all(c <= s for c, s in zip(counted, serving)), (name, dtype)
            # rocKE's direct benchmark runs every direct kernel that serves.
            assert (counted[0], counted[2]) == (serving[0], serving[2]), (name, dtype)

    # The seed-only graphs of its tiled, col, grouped and non-grouped instances: each has
    # both families selected, so the kernel_family knob test runs both.
    for name, problem, row in _seed_direct_graphs():
        for dtype in ("fp16", "bf16"):
            counted = _selected_counts(catalogs["seed"], problem, dtype)
            assert counted == row, (name, dtype)
            assert counted[0] > 0 and counted[1] > 0, (name, dtype)
            serving = _serving_counts(catalogs["seed"], problem, dtype)
            assert counted[0] == serving[0] and counted[1] <= serving[1], (name, dtype)


def test_gpu_suite_selects_every_seed_instance_for_some_graph(catalogs):
    """The seed suite pins every selected instance of each graph, so every seed
    instance has to be selected for at least one of them to be checked on the GPU."""
    graphs = [problem for _, problem, _ in _gpu_suite_graphs()]
    graphs += [problem for _, problem, _ in _seed_direct_graphs()]
    for dtype in ("fp16", "bf16"):
        selected = {
            spec.instance_id
            for problem in graphs
            for spec in _selected(catalogs["seed"], problem, dtype)
        }
        missing = [
            spec
            for spec in catalogs["seed"]
            if spec.dtype == dtype and spec.instance_id not in selected
        ]
        assert not missing, (dtype, missing)


# ---- The unit suite's selection sample ----

_SELECTION_SUITE = (
    _PROVIDER.parent.parent
    / "tests/engines/kernel_ingestor_engine/packs/TestGfx950ConvFwdAotSelection.cpp"
)
_SAMPLE_ROW = re.compile(
    r'^\s*\{"(\w+)", \{([\d, ]+)\}, "(\w+)", \{([\d, ]+)\}\},$', re.MULTILINE
)
_CASE_ROW = re.compile(r"^\s*\{\{([\d, ]+)\}, \{([\d, ]*)\}\},$", re.MULTILINE)


def test_unit_suite_selection_cases_are_the_python_runtime_candidates():
    """TestGfx950ConvFwdAotSelection.cpp checks the native selection over a sample of
    sweep identities against stated row indices; they must be what the Python runtime
    selection picks from the same sample (rocKE's own selection, per the rocKE test)."""
    from builders.common.convolution_forward import (
        Gfx950ConvFwdAotSpec,
        gfx950_conv_fwd_aot_runtime_candidates,
    )

    text = _SELECTION_SUITE.read_text()
    specs = []
    for row, (epilogue, values, pipeline, loop) in enumerate(_SAMPLE_ROW.findall(text)):
        a, b, c, tm, tn, tk, wm, wn, wtm, wtn, wtk = (int(v) for v in values.split(","))
        unroll, async_dma, chunk_a, chunk_b = (int(v) for v in loop.split(","))
        specs.append(
            Gfx950ConvFwdAotSpec(
                dtype="fp16", tile_m=tm, tile_n=tn, tile_k=tk, warp_m=wm, warp_n=wn,
                warp_tile_m=wtm, warp_tile_n=wtn, warp_tile_k=wtk, wave_size=64,
                pipeline=pipeline, epilogue=epilogue, vector_size_a=a,
                vector_size_b=b, vector_size_c=c, unroll_k=unroll, async_dma=async_dma,
                async_chunk_a=chunk_a, async_chunk_b=chunk_b, grouped=1,
                instance_id=f"{row:040x}",
            )  # fmt: skip
        )
    cases = _CASE_ROW.findall(text)
    assert (
        len(specs) >= 40 and len(cases) >= 10
    ), "keep one sample row and case per line"
    for problem, stated in cases:
        values = tuple(int(v) for v in problem.split(","))
        selected = _selected(specs, values, "fp16")
        assert [int(s.instance_id, 16) for s in selected] == [
            int(v) for v in stated.split(",")
        ], values


#: One graph per problem class the product catalog must serve, in the field order of
#: _DEFAULT_PROBLEM. Channel counts are per group.
_PROBLEM_CLASSES = {
    "odd C": (2, 3, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    "odd K": (2, 16, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    "odd C and K": (2, 3, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    "C % 4 == 2": (2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    "K % 4 == 2": (2, 16, 6, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    "C and K % 4 == 2": (2, 6, 10, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    "grouped": (2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4),
    "grouped, odd C/groups and K/groups": (
        2,
        12,
        20,
        14,
        14,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        4,
    ),
    "pointwise": (2, 32, 64, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    "grouped pointwise, odd C/groups": (2, 12, 12, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 4),
    "depthwise": (2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32),
    "depthwise outside the direct guard": (
        2,
        32,
        32,
        14,
        14,
        3,
        3,
        1,
        1,
        0,
        0,
        1,
        1,
        32,
    ),
}


@pytest.mark.parametrize("problem_class", sorted(_PROBLEM_CLASSES))
def test_the_catalog_serves_every_problem_class(catalogs, problem_class):
    """Every problem class is served by at least one implicit-GEMM instance, in both
    dtypes."""
    for dtype in ("fp16", "bf16"):
        counted = _serving_counts(
            catalogs["product"], _PROBLEM_CLASSES[problem_class], dtype
        )
        assert counted[1] > 0, (problem_class, dtype, counted)


def test_the_seed_runs_every_knob_value_the_catalog_ships(seed_identities):
    """The seed pins each of its instances on the GPU, so it holds, per dtype, at least
    one implicit-GEMM instance of every pipeline, epilogue, K loop, load width and
    store width the full catalog ships, and the default epilogue's store width."""
    product = _listed(_BUNDLES["product"][2])

    def knobs(identity):
        k_loop = (
            "async"
            if identity.get("async_dma")
            else "unroll" if identity.get("unroll_k") else "plain"
        )
        epilogue = identity.get("epilogue", "default")
        return (
            identity["pipeline"],
            epilogue,
            k_loop,
            identity["vector_size_a"],
            identity["vector_size_c"],
            (epilogue, identity["vector_size_c"]) if epilogue == "default" else None,
        )

    for dtype in ("fp16", "bf16"):
        implicit = [
            knobs(i)
            for i in seed_identities
            if i["dtype_a"] == dtype and i["algorithm"] == "implicit_gemm"
        ]
        shipped = {
            knobs(i)
            for i in product
            if i["dtype_a"] == dtype and i["algorithm"] == "implicit_gemm"
        }
        for position in range(6):
            assert {k[position] for k in shipped} <= {k[position] for k in implicit}, (
                dtype,
                position,
            )


# ---- The native direct launch's golden table ----

_GEOMETRY_TEST = (
    _REPO
    / "dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine"
    / "packs/TestGfx950ConvFwdGeometry.cpp"
)
#: One directLaunchCases() row: the instance (compiled values, variant, knobs), the
#: problem, then the verdict, grid and block.
_LAUNCH_ROW = re.compile(
    r"^\s*\{\{\{([\d, ]+)\}, DirectVariant::(\w+), ([\d, ]+)\}, \{([\d, ]+)\}, "
    r"(true|false), (\d+), (\d+), (\d+), (\d+)\},$",
    re.MULTILINE,
)


def test_native_direct_launch_cases_are_the_adapters():
    """TestGfx950ConvFwdGeometry.cpp holds conv::directInstanceLaunch to a table of
    (instance, problem) cases; every row must be what the adapter decides, so the native
    launch is checked against gfx950_conv_fwd_aot_serves, grid() and block()."""
    from builders.common.convolution_forward import (
        Gfx950ConvFwdAotSpec,
        Gfx950ConvFwdSpec,
        gfx950_conv_fwd_aot_serves,
    )

    rows = _LAUNCH_ROW.findall(_GEOMETRY_TEST.read_text())
    assert len(rows) >= 30, "directLaunchCases() rows not found; keep one row per line"
    variants = set()
    for compiled, variant, knobs, problem, served, *launch in rows:
        fy, fx, sh, sw, ph, pw, dh, dw = (int(v) for v in compiled.split(","))
        # A row states the DirectInstance fields up to its variant's last; the rest are 0.
        values = [int(v) for v in knobs.split(",")]
        values += [0] * (14 - len(values))
        bw, waves, bh, bq, bg, cpg, kpg, th, tw, tk, ck, wm, wn, atom_tile = values
        # The table records the atom's tile only; the gfx950 sweep's atom of that tile.
        atom = {32: "32x32x16", 16: "16x16x32"}.get(atom_tile, "none")
        spec = Gfx950ConvFwdAotSpec(
            dtype="fp16", tile_m=0, tile_n=0, tile_k=0, warp_m=0, warp_n=0,
            warp_tile_m=0, warp_tile_n=0, warp_tile_k=0, wave_size=64,
            pipeline="direct", epilogue="direct", vector_size_a=0, vector_size_b=0,
            vector_size_c=0, unroll_k=0, async_dma=0, async_chunk_a=0,
            async_chunk_b=0, grouped=0, instance_id="0" * 40, kernel_family=1,
            direct_variant=variant.lower(), filter_h=fy, filter_w=fx, stride_h=sh,
            stride_w=sw, pad_h=ph, pad_w=pw, dilation_h=dh, dilation_w=dw,
            block_w=bw, block_waves=waves, block_h=bh, block_q=bq, block_groups=bg,
            cpg=cpg, kpg=kpg, tile_h=th, tile_w=tw, tile_k_out=tk, ck=ck, waves_m=wm,
            waves_n=wn, atom=atom, swizzle_wgm=8 if atom_tile else 0,
            iglp=0 if atom_tile else -1,
        )  # fmt: skip
        n, c, k, hi, wi, y, x, psh, psw, pph, ppw, pdh, pdw, groups = (
            int(v) for v in problem.split(",")
        )
        runtime = Gfx950ConvFwdSpec(
            N=n, C=c, K=k, Hi=hi, Wi=wi, Y=y, X=x, sH=psh, sW=psw, pH=pph, pW=ppw,
            dH=pdh, dW=pdw, groups=groups,
        )  # fmt: skip
        ok, why = gfx950_conv_fwd_aot_serves(spec, runtime)
        assert ok == (served == "true"), (variant, problem, why)
        if ok:
            gx, gy, gz, block = (int(v) for v in launch)
            assert spec.grid(runtime) == (gx, gy, gz), (variant, problem)
            assert spec.block()[0] == block, (variant, problem)
        variants.add((variant, ok))
    assert variants == {
        (v, ok)
        for v in ("TILED", "COL", "GROUPED", "NONGROUPED")
        for ok in (True, False)
    }
