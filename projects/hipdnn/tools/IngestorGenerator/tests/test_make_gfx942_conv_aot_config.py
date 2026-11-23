# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx942 cases of tools/make_conv_aot_config.py: the hipkernel:Gfx942ConvFwdAot
config made with --arch gfx942 from its committed instance lists. What both forward
engines share is test_make_conv_aot_config.py's; these reuse its rocKE binding."""

import collections
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_conv_aot_config as tool  # noqa: E402
import test_make_conv_aot_config as shared  # noqa: E402

# The shared module's rocKE binding.
from test_make_conv_aot_config import _rocke_on_path  # noqa: E402, F401

_ARCH = "gfx942"
_ENGINE = "hipkernel:Gfx942ConvFwdAot"
_LISTS = shared._GFX942_LISTS
#: Kernels per pack and dtype of the gfx942 catalog under the default k-loop
#: sharding, and the group-merged implicit-GEMM kernels among them.
_SHARDS = {
    "bf16": {
        "mem_plain": (36, 0),
        "mem_unroll": (12, 0),
        "compv3_plain": (30, 1),
        "compv3_unroll": (7, 0),
        "compv4_plain": (29, 2),
        "compv4_unroll": (2, 0),
        "direct": (74, 0),
        "direct_split": (30, 0),
    },
    "fp16": {
        "mem_plain": (34, 0),
        "mem_unroll": (10, 0),
        "compv3_plain": (28, 1),
        "compv3_unroll": (7, 0),
        "compv4_plain": (27, 2),
        "compv4_unroll": (2, 0),
        "direct": (101, 9),
        "direct_split": (30, 0),
    },
}


@pytest.fixture(scope="module")
def identities():
    return shared._listed(_LISTS)


@pytest.fixture(scope="module")
def specs(identities):
    return shared._specs(identities)


@pytest.fixture(scope="module")
def config(specs):
    return tool.build_config(specs, sharding="k-loop")


def test_the_gfx942_engine_is_the_adapters():
    assert tool.arches() == ("gfx950", "gfx942")
    assert tool.product_engine(_ARCH) == _ENGINE
    assert tool.product_native_namespace(_ARCH) == "hipkernel.gfx942_conv_fwd_aot"
    assert tool.adapter_builder(_ARCH) == "build_gfx942_conv_fwd_aot"
    assert (
        tool.adapter_builder(_ARCH, "split_reduce")
        == "build_gfx942_conv_fwd_aot_split_reduce"
    )
    # The default architecture's engine is the one the constants name.
    assert tool.product_engine() == tool.PRODUCT_ENGINE
    assert tool.product_native_namespace() == tool.PRODUCT_NATIVE_NAMESPACE
    with pytest.raises(ValueError, match="no packaged forward engine"):
        tool.product_engine("gfx90a")


def test_gfx942_packages_every_direct_algorithm_but_the_windowed_depthwise():
    assert tool.packaged_direct_algorithms("gfx950") == tool.PACKAGED_DIRECT_ALGORITHMS
    assert tool.packaged_direct_algorithms(_ARCH) == {
        algorithm: variant
        for algorithm, variant in tool.PACKAGED_DIRECT_ALGORITHMS.items()
        if algorithm != "direct_depthwise_win"
    }


def test_the_config_is_the_gfx942_engine_over_its_own_native_code(config, specs):
    engine = config["engine"]
    assert engine["name"] == _ENGINE
    assert "native_symbols_from" not in engine
    assert engine["sdk_version"] == "1.0.0"
    assert config["authored_subpath"] == "rocKE/gfx942_conv_fwd_aot"
    # The split-C kernels need a workspace, and their packs name the split-C dispatch.
    assert config["workspace_policy"] == "derived"
    for pack in config["packs"]:
        assert pack["arch"] == [_ARCH]
        defaults = {
            "kind": "rocke",
            "source": "builders/common/convolution_forward.py",
            "builder": "build_gfx942_conv_fwd_aot",
        }
        if pack["name"].endswith("_direct_split"):
            assert pack["dispatch"] == tool.SPLIT_C_DISPATCH
            defaults["launches"] = {
                "conv": {},
                "reduce": {"builder": "build_gfx942_conv_fwd_aot_split_reduce"},
            }
        else:
            assert "dispatch" not in pack
        assert pack["kernel_defaults"] == defaults
        for variant in pack["variants"]:
            assert variant["name"].startswith("gfx942_conv_fwd_aot_")
    assert all(spec.ARCH == _ARCH for spec in specs)


def test_the_gfx942_kmd_and_knobs_are_the_gfx950_engines(config):
    """The two engines tune the same knobs over the same KMD, so one census, ranker
    feature set and native decode serve both."""
    seed = tool.build_config(shared._specs(json.loads(shared._INSTANCES.read_text())))
    assert (
        config["engine"]["knobs"] == list(tool.ENGINE_KNOBS) == seed["engine"]["knobs"]
    )
    assert len(config["engine"]["knobs"]) == 40
    assert config["kmd_fields"] == seed["kmd_fields"]
    assert config["specialization"] == seed["specialization"]


def test_the_catalog_is_sharded_by_pipeline_and_k_loop(config):
    names = {}
    for pack in config["packs"]:
        dtype, shard = pack["name"].split("_", 1)
        shapes = [shape for variant in pack["variants"] for shape in variant["shapes"]]
        merged = sum(1 for shape in shapes if shape.get("group_merge", 1) > 1)
        names[(dtype, shard)] = (len(shapes), merged)
    assert names == {
        (dtype, shard): counts
        for dtype, shards in _SHARDS.items()
        for shard, counts in shards.items()
    }


def test_the_catalog_leaves_every_gfx950_only_feature_off(specs):
    """gfx942 has no 16x16x32 or 32x32x16 MFMA atom and no windowed depthwise; its
    catalog also ships no async-DMA kernel, and split-C only for non-grouped ones."""
    variants = collections.Counter(spec.direct_variant for spec in specs)
    assert variants == {
        "none": 224,
        "nongrouped": 164,
        "tiled": 35,
        "grouped_halo": 17,
        "grouped": 4,
        "std": 8,
        "col": 4,
        "spatial": 3,
    }
    for spec in specs:
        assert not spec.async_dma, spec.instance_id
        assert (
            not spec.is_split_c or spec.direct_variant == "nongrouped"
        ), spec.instance_id
        if spec.direct_variant == "nongrouped":
            assert spec.atom in ("16x16x16", "32x32x8"), spec.instance_id
        if not spec.is_direct:
            assert (spec.warp_tile_m, spec.warp_tile_k) in (
                (16, 16),
                (32, 8),
            ), spec.instance_id


def test_the_config_expands_to_one_kernel_per_instance(tmp_path, config, specs):
    path = tmp_path / "aot.yaml"
    path.write_text(tool.dump_config(config))
    loaded = load_config(path)
    assert loaded.engine.name == _ENGINE
    assert sum(len(pack.kernels) for pack in loaded.packs) == len(specs) == 459
    assert sorted(loaded.kdp_stem(pack) for pack in loaded.packs) == sorted(
        f"gfx942_conv_fwd_aot_{dtype}_{shard}"
        for dtype, shards in _SHARDS.items()
        for shard in shards
    )


def test_one_catalog_is_one_architectures(specs):
    seed = shared._specs(json.loads(shared._INSTANCES.read_text()))
    with pytest.raises(ValueError, match="one architecture's"):
        tool.build_config(specs[:3] + seed[:3])


def test_main_writes_the_gfx942_config(tmp_path, capsys, config):
    out = tmp_path / "aot.yaml"
    argv = ["--arch", _ARCH, "--instances", *map(str, _LISTS), "--output", str(out)]
    assert tool.main(argv) == 0
    assert "459 instance(s)" in capsys.readouterr().out
    header, body = out.read_text().split("\n", 1)
    assert header.startswith("# Generated by tools/make_conv_aot_config.py")
    assert yaml.safe_load(body) == config


@pytest.mark.parametrize(("arch", "listed"), [(_ARCH, "seed"), ("gfx950", "gfx942")])
def test_main_refuses_an_instance_of_another_architecture(
    tmp_path, capsys, identities, arch, listed
):
    entries = (
        json.loads(shared._INSTANCES.read_text())[:1]
        if listed == "seed"
        else identities[:1]
    )
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps(entries))
    out = tmp_path / "aot.yaml"
    argv = ["--arch", arch, "--instances", str(instances), "--output", str(out)]
    assert tool.main(argv) == 1
    other = "gfx950" if arch == _ARCH else _ARCH
    assert f"a {other} instance in the {arch} engine's catalog" in (
        capsys.readouterr().err
    )
    assert not out.exists()


def test_main_refuses_an_architecture_without_a_forward_engine(tmp_path, capsys):
    argv = ["--arch", "gfx90a", "--instances", str(_LISTS[0])]
    with pytest.raises(SystemExit):
        tool.main(argv + ["--output", str(tmp_path / "aot.yaml")])
    assert "invalid choice: 'gfx90a'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arch, engine_name",
    [("gfx950", _ENGINE), (_ARCH, "hipkernel:Gfx950ConvFwdAot")],
)
@pytest.mark.parametrize("stabilize", [False, True])
def test_main_refuses_another_architectures_engine_name(
    tmp_path, capsys, arch, engine_name, stabilize
):
    """An --engine-name that is another architecture's product engine is refused
    before anything is written: its catalog would borrow the wrong arch's symbols,
    and --stabilize-ids would derive namespaced ids instead of its own."""
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(json.dumps({"id": str(uuid.uuid4()), "kernelDescriptors": []}))
    argv = ["--arch", arch, "--engine-name", engine_name]
    if stabilize:
        argv += ["--stabilize-ids", str(kdp)]
    else:
        argv += ["--instances", str(_LISTS[0]), "--output", str(tmp_path / "aot.yaml")]
    with pytest.raises(SystemExit):
        tool.main(argv)
    assert f"{engine_name} is the " in capsys.readouterr().err
    assert not (tmp_path / "aot.yaml").exists()


def test_gfx942_kernel_ids_are_its_own_products(tmp_path, specs):
    """The gfx942 engine is its own product: a kernel's id is uuid5 of its instance_id,
    as gfx950's are, and --stabilize-ids --arch gfx942 writes exactly that.
    instance_id hashes the architecture, so no id repeats across the two engines."""
    kernels = [
        {
            "id": str(uuid.uuid4()),
            "kernel_source": {"builder": "build_gfx942_conv_fwd_aot"},
            "metadata": {"instance_id": spec.instance_id},
        }
        for spec in specs[:5]
    ]
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(json.dumps({"id": str(uuid.uuid4()), "kernelDescriptors": kernels}))
    assert tool.main(["--arch", _ARCH, "--stabilize-ids", str(kdp)]) == 0
    for spec, kernel in zip(specs, json.loads(kdp.read_text())["kernelDescriptors"]):
        expected = str(uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id))
        assert kernel["id"] == expected
        assert tool.kernel_uuid(spec.instance_id, arch=_ARCH) == expected
        assert tool.kernel_uuid(spec.instance_id, _ENGINE, _ARCH) == expected
    seed = shared._specs(json.loads(shared._INSTANCES.read_text()))
    assert not {s.instance_id for s in specs} & {s.instance_id for s in seed}
