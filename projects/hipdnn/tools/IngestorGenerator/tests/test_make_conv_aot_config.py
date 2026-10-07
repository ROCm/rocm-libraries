# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_conv_aot_config.py and the shipped hipkernel:Gfx950ConvFwdAot bundle."""

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
import make_conv_aot_config as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_INSTANCES = _CONFIGS / "gfx950_conv_fwd_aot.instances.json"
_CONFIG = _CONFIGS / "gfx950_conv_fwd_aot.yaml"
_SHIPPED_KDP = (
    _REPO
    / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors"
    / "rocKE/gfx950_conv_fwd_aot/gfx950_conv_fwd_aot.kdp.json"
)


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
    kernels = json.loads(_SHIPPED_KDP.read_text())["kernelDescriptors"]
    shipped = {k["metadata"]["instance_id"]: k for k in kernels}
    assert sorted(shipped) == sorted(s.instance_id for s in seed_specs)
    for spec in seed_specs:
        kernel = shipped[spec.instance_id]
        expected = {name: getattr(spec, name) for name in tool.SPEC_ORDER}
        assert kernel["kernel_source"]["spec"] == expected
        assert kernel["metadata"] == expected
        assert kernel["kernel_source"]["builder"] == "build_gfx950_conv_fwd_aot"
        assert kernel["id"] == str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id)
        )


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
