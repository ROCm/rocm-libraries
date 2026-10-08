# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_conv_aot_config.py and the shipped hipkernel:Gfx950ConvFwdAot bundle."""

import copy
import dataclasses
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
import make_conv_aot_config as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_INSTANCES = _CONFIGS / "gfx950_conv_fwd_aot.instances.json"
_CONFIG = _CONFIGS / "gfx950_conv_fwd_aot.yaml"
_SHIPPED_DIR = (
    _REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
    / "rocKE/gfx950_conv_fwd_aot"
)
#: One shipped shard, for the tests that rewrite a KDP.
_SHIPPED_KDP = _SHIPPED_DIR / "gfx950_conv_fwd_aot_fp16.kdp.json"
#: GitHub warns about any file above 50 MB; every emitted file stays below it.
_FILE_SIZE_LIMIT = 50 * 1000 * 1000


def _shipped_kdps():
    return sorted(_SHIPPED_DIR.glob("*.kdp.json"))


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    """Bind rocKE for this module only: test_dispatch_parity asserts what a provider
    root ADDS to sys.path, so a binding left behind at collection would hide it."""
    before = list(sys.path)
    tool._bind_rocke(_ROCKE)
    yield
    sys.path[:] = before


@pytest.fixture(scope="module")
def seed_identities():
    return json.loads(_INSTANCES.read_text())


@pytest.fixture(scope="module")
def seed_specs(seed_identities):
    specs = []
    for identity in seed_identities:
        spec, reason = tool.instance_spec(identity)
        assert spec is not None, reason
        specs.append(spec)
    return sorted(specs, key=lambda s: (s.dtype, s.instance_id))


def test_seed_config_is_what_the_tool_generates(seed_specs):
    assert yaml.safe_load(_CONFIG.read_text()) == tool.build_config(seed_specs)


def test_shipped_bundle_is_the_seed_catalog_with_instance_derived_ids(seed_specs):
    shipped = {}
    for kdp in _shipped_kdps():
        document = json.loads(kdp.read_text())
        dtype = kdp.name[: -len(".kdp.json")].rsplit("_", 1)[1]
        assert document["name"] == f"hipkernel:gfx950_conv_fwd_aot_{dtype}"
        for kernel in document["kernelDescriptors"]:
            assert kernel["metadata"]["dtype"] == dtype
            shipped[kernel["metadata"]["instance_id"]] = kernel
    assert sorted(shipped) == sorted(s.instance_id for s in seed_specs)
    for spec in seed_specs:
        kernel = shipped[spec.instance_id]
        expected = {name: getattr(spec, name) for name in tool.spec_order(spec)}
        assert kernel["kernel_source"]["spec"] == expected
        assert kernel["metadata"] == expected
        assert kernel["kernel_source"]["builder"] == "build_gfx950_conv_fwd_aot"
        assert kernel["id"] == str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id)
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


def test_shipped_bundle_is_sharded_below_the_hosting_size_limit():
    names = sorted(path.name for path in _SHIPPED_DIR.iterdir())
    assert names == [
        "gfx950_conv_fwd_aot.kmd.json",
        "gfx950_conv_fwd_aot.udd.json",
        "gfx950_conv_fwd_aot.ued.json",
        "gfx950_conv_fwd_aot.uhd.json",
        "gfx950_conv_fwd_aot_bf16.kdp.json",
        "gfx950_conv_fwd_aot_fp16.kdp.json",
        "kernel_dtype_matches_graph.umd.json",
    ]
    matcher = json.loads(
        (_SHIPPED_DIR / "kernel_dtype_matches_graph.umd.json").read_text()
    )
    for path in _SHIPPED_DIR.iterdir():
        assert path.stat().st_size < _FILE_SIZE_LIMIT, path.name
    # No operation-scoped matcher: every shard names the shared kernel matcher only.
    for kdp in _shipped_kdps():
        assert json.loads(kdp.read_text())["matchers"] == [matcher["id"]]


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
    document = json.loads(_SHIPPED_KDP.read_text())
    for kernel in document["kernelDescriptors"]:
        kernel["id"] = str(uuid.uuid4())
    kdp.write_text(json.dumps(document))
    assert tool.stabilize_ids(kdp) == len(document["kernelDescriptors"])
    assert json.loads(kdp.read_text()) == json.loads(_SHIPPED_KDP.read_text())

    first = document["kernelDescriptors"][0]
    document["provenance"] = {"note": first["id"]}
    kdp.write_text(json.dumps(document))
    with pytest.raises(SystemExit, match="refusing to rewrite a referenced id"):
        tool.stabilize_ids(kdp)


# ---- Direct depthwise identities ----


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


def test_direct_identities_are_packaged_exactly_when_the_native_launch_runs_them(
    forward_direct_identities,
):
    accepted = {}
    for identity in forward_direct_identities:
        spec, why = tool.instance_spec(identity)
        if identity["algorithm"] in tool.DIRECT_ALGORITHMS:
            assert spec is not None, why
            assert spec.is_direct
            assert spec.direct_variant == tool.DIRECT_ALGORITHMS[identity["algorithm"]]
            key = (spec.dtype, identity["algorithm"])
            accepted[key] = accepted.get(key, 0) + 1
        else:
            assert spec is None and "has no packaged launch" in why, why
    # The study's runnable subset: 54 std and 30 spatial identities per dtype.
    assert accepted == {
        (dtype, algorithm): count
        for dtype in ("fp16", "bf16")
        for algorithm, count in (
            ("direct_depthwise", 54),
            ("direct_depthwise_spatial", 30),
        )
    }


def test_a_direct_instance_carries_the_identitys_capabilities_and_knobs(
    forward_direct_identities,
):
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in forward_direct_identities:
        if identity["algorithm"] not in tool.DIRECT_ALGORITHMS:
            continue
        spec, _ = tool.instance_spec(identity)
        knobs = json.loads(identity["knobs"])
        assert spec.instance_id == KernelIdentity.from_dict(identity).stable_hash()
        assert (spec.filter_h, spec.stride_h, spec.pad_h, spec.dilation_h) == (
            identity["filter_h"],
            identity["stride_h"],
            identity["pad_h"],
            identity["dilation_h"],
        )
        assert (spec.block_w, spec.block_waves) == (
            knobs.get("block_w", 0),
            knobs["block_waves"],
        )
        # The implicit-GEMM fields are the identity's placeholders.
        assert (spec.tile_m, spec.pipeline, spec.vector_size_c, spec.grouped) == (
            0,
            "direct",
            0,
            0,
        )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"arch": "gfx942"}, "gfx950"),
        ({"algorithm": "direct_depthwise_tiled"}, "has no packaged launch"),
        ({"dtype_d": "fp32"}, "mixed operand dtypes"),
        ({"cpg": 2, "kpg": 2}, "not depthwise"),
        ({"filter_w": 5, "pad_w": 2}, "cannot build"),
        ({"pad_h": 0, "pad_w": 0}, "cannot build"),
        ({"split_k": 2}, "cannot build"),
        ({"knobs": '{"block_waves": 1}'}, "knobs"),
        ({"knobs": '{"block_w": 4, "block_waves": 1, "x": 1}'}, "x"),
        ({"knobs": '{"block_w": 4, "block_waves": 99}'}, "block_waves"),
    ],
)
def test_unsupported_direct_identities_are_named_with_their_reason(
    forward_direct_identities, change, reason
):
    identity = copy.deepcopy(
        next(
            i
            for i in forward_direct_identities
            if i["algorithm"] == "direct_depthwise"
            and json.loads(i["knobs"]) == {"block_w": 4, "block_waves": 1}
        )
    )
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


def test_direct_instances_are_a_variant_stating_the_family_fields(seed_specs):
    config = tool.build_config(seed_specs)
    for pack in config["packs"]:
        implicit, direct = pack["variants"]
        assert implicit["spec_order"] == list(tool.SPEC_ORDER)
        assert direct["spec_order"] == list(tool.DIRECT_SPEC_ORDER)
        assert all(shape["kernel_family"] == 1 for shape in direct["shapes"])
        assert not any("kernel_family" in shape for shape in implicit["shapes"])
    # Implicit-GEMM descriptors leave the direct fields to their builder default.
    defaults = {
        field["name"]: field["default_value"]
        for field in config["kmd_fields"]
        if field.get("builder_default")
    }
    assert defaults == tool.DIRECT_FIELDS
    from builders.common.convolution_forward import Gfx950ConvFwdAotSpec

    assert {
        f.name: f.default
        for f in dataclasses.fields(Gfx950ConvFwdAotSpec)
        if f.name in tool.DIRECT_FIELDS
    } == tool.DIRECT_FIELDS
    assert config["engine"]["knobs"] == ["tile_k", "kernel_family"]
    # Both families are in the shipped seed.
    assert {s.is_direct for s in seed_specs} == {False, True}


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
    document = json.loads(_SHIPPED_KDP.read_text())
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
    document = json.loads(_SHIPPED_KDP.read_text())
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
