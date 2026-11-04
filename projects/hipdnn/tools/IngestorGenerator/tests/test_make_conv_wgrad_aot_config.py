# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_conv_wgrad_aot_config.py.

Phase 1 only: there is no shipped hipkernel:Gfx950ConvWgradAot bundle yet (the native
pack and its generated KDPs are a later phase), so this covers what Phase 1 owns --
instance_spec conversion correctness, sharding, UUID stability, and the generator
config this tool emits loading back -- not the descriptors/artifacts a build would
produce. Compare tests/test_make_conv_dgrad_aot_config.py, the sibling whose bundle
already ships.
"""

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
import make_conv_wgrad_aot_config as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_INSTANCES = [
    _CONFIGS / f"gfx950_conv_wgrad_aot_{dtype}.instances.json"
    for dtype in ("bf16", "fp16")
]
_FILE_SIZE_LIMIT = 50 * 1000 * 1000
_JOBS = min(32, os.cpu_count() or 1)


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


def test_the_committed_lists_hold_122_instances_per_dtype_across_all_8_families(specs):
    """46 implicit-GEMM + 67 direct + 9 fold-helper per dtype: the deliverable set for
    the gfx950 wgrad AOT catalog (two-stage implicit-GEMM instances excluded), one
    count per (kernel_family, direct_variant) pair."""
    by_dtype = collections.defaultdict(collections.Counter)
    for spec in specs:
        by_dtype[spec.dtype][(spec.kernel_family, spec.direct_variant)] += 1
    for dtype in ("bf16", "fp16"):
        assert by_dtype[dtype] == {
            (0, "none"): 46,
            (1, "direct_wgrad"): 5,
            (1, "direct_wgrad_tiled"): 20,
            (1, "direct_wgrad_dw"): 5,
            (1, "direct_wgrad_dw_cols"): 1,
            (1, "direct_wgrad_dw_mfma"): 14,
            (1, "direct_wgrad_gather"): 16,
            (1, "direct_wgrad_grouped_mfma"): 6,
            (2, "direct_wgrad_fold"): 9,
        }
    assert sum(1 for s in specs if not s.is_helper) == 226
    assert len(specs) == 244


def test_no_shipped_instance_is_two_stage(specs):
    """enumerate_conv_wgrad_aot_instances.py drops every two-stage identity before
    this tool ever sees one (its Stage 2 reduce is not wired in the adapter)."""
    assert all(not spec.two_stage for spec in specs)


def test_instance_id_is_the_identity_stable_hash(identities):
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in identities:
        spec, reason = tool.instance_spec(identity)
        assert spec is not None, reason
        assert spec.instance_id == KernelIdentity.from_dict(identity).stable_hash()


def test_every_committed_instance_validates(identities):
    from builders.common.convolution_backward_filter import (
        supports_gfx950_conv_wgrad_aot,
    )

    for identity in identities:
        spec, reason = tool.instance_spec(identity)
        assert spec is not None, reason
        ok, why = supports_gfx950_conv_wgrad_aot(spec)
        assert ok, (identity, why)


def test_a_fold_helper_instance_bakes_no_geometry(specs):
    """Unlike dgrad's transpose/reorganize helpers, a wgrad fold-helper instance bakes
    no filter, stride, padding, dilation or channel count at all -- only taps."""
    helpers = [s for s in specs if s.is_helper]
    assert helpers, "the catalog must carry the fold helper"
    for spec in helpers:
        assert spec.direct_variant == "direct_wgrad_fold"
        assert spec.taps > 0
        for name in (
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
        ):
            assert getattr(spec, name) == 0, name


def test_kernel_uuids_are_derived_and_stable_across_regenerations(specs):
    for spec in specs[:20]:
        assert tool.kernel_uuid(spec.instance_id) == str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id)
        )
        # Twice: regenerating the bundle must not mint a new id.
        assert tool.kernel_uuid(spec.instance_id) == tool.kernel_uuid(spec.instance_id)


def test_kernel_uuids_differ_from_another_engines_namespace(specs):
    assert tool.kernel_uuid(
        specs[0].instance_id, "hipkernel:X"
    ) != tool.kernel_uuid(specs[0].instance_id)


def test_every_shard_holds_only_its_own_family(specs):
    by_shard = collections.defaultdict(list)
    for spec in specs:
        by_shard[tool.shard_name(spec)].append(spec)
    for shard, members in by_shard.items():
        dtype, _, rest = shard.partition("_")
        assert dtype in ("bf16", "fp16")
        if rest == "implicit":
            assert all(not s.is_direct for s in members)
        else:
            assert all(s.is_direct and s.direct_variant == rest for s in members)


def test_dtype_sharding_is_one_pack_per_dtype(specs):
    shards = {tool.shard_name(spec, "dtype") for spec in specs}
    assert shards == {"bf16", "fp16"}


def test_generated_config_expands_to_one_kernel_per_instance(tmp_path, specs):
    path = tmp_path / "wgrad.yaml"
    path.write_text(tool.dump_config(tool.build_config(specs)))
    config = load_config(path)
    assert config.engine.name == tool.PRODUCT_ENGINE
    assert config.is_multi_pack and not config.has_operation_matchers
    assert sum(len(pack.kernels) for pack in config.packs) == len(specs)


def test_the_config_borrows_no_native_symbols(specs):
    config = tool.build_config(specs)
    assert "native_symbols_from" not in config["engine"]
    assert config["workspace_policy"] == "none"
    other = tool.build_config(specs[:3], "hipkernel:Gfx950ConvWgradAotSeed")
    assert other["engine"]["native_symbols_from"] == tool.PRODUCT_ENGINE
    assert tool.kernel_uuid(specs[0].instance_id, "hipkernel:X") != tool.kernel_uuid(
        specs[0].instance_id
    )


def test_dumped_config_loads_back(specs):
    text = tool.dump_config(tool.build_config(specs))
    assert yaml.safe_load(text) == tool.build_config(specs)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"direction": "fwd"}, "wgrad implicit-GEMM"),
        ({"direction": "dgrad"}, "wgrad implicit-GEMM"),
        ({"arch": "gfx942"}, "wgrad implicit-GEMM"),
        ({"dtype_d": "fp32"}, "mixed operand dtypes"),
        ({"filter_h": 3}, "filter_h"),
        ({"pad_h": 1}, "pad_h"),
        ({"pipeline": "basic"}, "not produced"),
    ],
)
def test_unsupported_implicit_identities_are_named_with_their_reason(
    identities, change, reason
):
    identity = copy.deepcopy(next(i for i in identities if i["algorithm"] == "implicit_gemm"))
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


@pytest.mark.parametrize(
    ("algorithm", "change", "reason"),
    [
        ("direct_wgrad", {"algorithm": "direct_grouped_dgrad"}, "no packaged"),
        ("direct_wgrad", {"direction": "direct_fwd"}, "direct_wgrad"),
        (
            "direct_wgrad",
            {"knobs": '{"ho_per_block": 30, "mfma_k": 32, "waves_c": 1, '
            '"waves_k": 1, "waves_q": 1, "nonsense": 1}'},
            "nonsense",
        ),
    ],
)
def test_unsupported_direct_identities_are_named_with_their_reason(
    identities, algorithm, change, reason
):
    identity = copy.deepcopy(next(i for i in identities if i["algorithm"] == algorithm))
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


def test_main_dedups_and_refuses_unsupported_unless_asked(tmp_path, identities, capsys):
    unsupported = dict(identities[0], filter_h=1)
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


def test_stabilize_ids_rewrites_kernel_uuids_and_refuses_referenced_ids(tmp_path, specs):
    spec = next(s for s in specs if not s.is_direct)
    document = {
        "kernelDescriptors": [
            {
                "id": str(uuid.uuid4()),
                "kernel_source": {"kind": "rocke"},
                "metadata": {"instance_id": spec.instance_id},
            }
        ]
    }
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(json.dumps(document))
    assert tool.stabilize_ids(kdp) == 1
    rewritten = json.loads(kdp.read_text())
    assert rewritten["kernelDescriptors"][0]["id"] == tool.kernel_uuid(spec.instance_id)
    # The forward tool's default product engine would derive a different id.
    assert fwd_tool.kernel_uuid(
        spec.instance_id, tool.PRODUCT_ENGINE
    ) != tool.kernel_uuid(spec.instance_id)
