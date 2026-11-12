# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_conv_dgrad_aot_config.py and the shipped dgrad AOT bundles.

The bundle and catalog tests run for every engine (the ``arch`` fixture: the
adapter's ``TARGETS``); the rest pin hipkernel:Gfx950ConvDgradAot. Cases only
hipkernel:Gfx1250ConvDgradAot or hipkernel:Gfx942ConvDgradAot has are
test_make_gfx1250_conv_dgrad_aot_config.py's and
test_make_gfx942_conv_dgrad_aot_config.py's, which reuse the loaders here."""

import collections
import copy
import functools
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config
from codegen.models import EngineSpec

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS))

import enumerate_conv_dgrad_aot_instances as enumerate_tool  # noqa: E402
import launch_surface  # noqa: E402
import conv_aot_common  # noqa: E402
import make_conv_dgrad_aot_config as tool  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_PROVIDER = (
    _REPO / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine"
)
_FWD_DIRS = (
    _PROVIDER / "descriptors/rocKE/gfx950_conv_fwd_aot",
    _PROVIDER / "test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed",
)
_FILE_SIZE_LIMIT = 50 * 1000 * 1000
#: An architecture with no dgrad engine.
_NO_ENGINE_ARCH = "gfx90a"
_JOBS = min(32, os.cpu_count() or 1)


def _adapter_arches():
    """The adapter's ``TARGETS``, read at collection (pytest parametrizes before any
    fixture can bind rocKE), leaving ``sys.path`` as it was."""
    before = list(sys.path)
    try:
        tool._bind_rocke(_ROCKE)
        return tool.arches()
    finally:
        sys.path[:] = before


#: The engines these tests cover: every one the adapter packages.
ARCHES = _adapter_arches()
#: Per engine, the implicit-GEMM pipelines its shards are named for.
_PIPELINES = {
    "gfx942": ("mem", "compv3", "compv4"),
    "gfx950": ("mem", "compv3", "compv4"),
    "gfx1250": ("mem", "wavelet"),
}
#: Per engine, the (dtype, family, capability) partitions its matched kernels span,
#: and its implicit-GEMM kernels per dtype.
_PARTITIONS = {"gfx942": 32, "gfx950": 34, "gfx1250": 46}
_IMPLICIT_PER_DTYPE = {"gfx942": 128, "gfx950": 141, "gfx1250": 25}


def slug(arch):
    return f"{arch}_conv_dgrad_aot"


def shipped_dir(arch):
    return _PROVIDER / "descriptors/rocKE" / slug(arch)


def census(arch):
    return (
        _REPO / "dnn-providers/hip-kernel-provider/src/tests/engines"
        f"/kernel_ingestor_engine/packs/Test{arch.capitalize()}ConvDgradAotPacks.cpp"
    )


def instance_lists(arch):
    return [
        _CONFIGS / enumerate_tool.INSTANCE_LIST.format(arch=arch, dtype=dtype)
        for dtype in ("bf16", "fp16")
    ]


@functools.cache
def load_identities(arch):
    return tuple(
        i for path in instance_lists(arch) for i in json.loads(path.read_text())
    )


@functools.cache
def load_specs(arch):
    """The converted catalog of ``arch``; rocKE must be bound."""
    out = []
    for spec, reason in tool.instance_specs(list(load_identities(arch)), _JOBS):
        assert spec is not None, reason
        out.append(spec)
    return tuple(sorted(out, key=lambda s: (s.dtype, s.instance_id)))


@functools.cache
def load_shipped(arch):
    """instance_id -> (shard, kernel descriptor) of ``arch``'s shipped bundle."""
    out = {}
    for kdp in sorted((shipped_dir(arch) / "artifacts").glob("*.kdp.json")):
        document = json.loads(kdp.read_text())
        shard = kdp.name[: -len(".kdp.json")][len(slug(arch)) + 1 :]
        assert document["name"] == f"hipkernel:{slug(arch)}_{shard}"
        for kernel in document["kernelDescriptors"]:
            out[kernel["metadata"]["instance_id"]] = (shard, kernel)
    return out


def document(arch, kind):
    return json.loads((shipped_dir(arch) / f"{slug(arch)}.{kind}.json").read_text())


def ships_a_ranker(arch):
    """Whether ``arch``'s engine ships a trained ranker: a promoted model packet
    turns the UED's default sort list into a list (the native scorer, then the
    model) and adds the DVC-tracked ``heuristics/`` beside the KDPs."""
    return isinstance(document(arch, "ued")["sort_kernel_catalog"]["default"], list)


@pytest.fixture(scope="module", autouse=True)
def _rocke_on_path():
    before = list(sys.path)
    tool._bind_rocke(_ROCKE)
    yield
    sys.path[:] = before


@pytest.fixture(scope="module", params=ARCHES)
def arch(request):
    return request.param


@pytest.fixture(scope="module")
def identities(arch):
    return list(load_identities(arch))


@pytest.fixture(scope="module")
def specs(arch):
    return list(load_specs(arch))


@pytest.fixture(scope="module")
def shipped(arch):
    return load_shipped(arch)


@pytest.fixture(scope="module")
def gfx950_identities():
    return list(load_identities("gfx950"))


@pytest.fixture(scope="module")
def gfx950_specs():
    return list(load_specs("gfx950"))


@pytest.fixture(scope="module")
def gfx950_shipped():
    return load_shipped("gfx950")


def test_every_engine_states_its_expectations():
    """A dgrad engine the adapter gains fails here until its expectations are stated."""
    for table in (_PIPELINES, _PARTITIONS, _IMPLICIT_PER_DTYPE):
        assert set(table) == set(ARCHES)


def test_the_catalog_is_203_selectable_kernels_and_4_helpers_per_dtype(gfx950_specs):
    """203 (141 implicit-GEMM + 62 direct) selectable kernels and 4 helpers per dtype:
    the deliverable set for the gfx950 dgrad AOT catalog, one count per (kernel_family,
    direct_variant) pair."""
    by_dtype = collections.defaultdict(collections.Counter)
    for spec in gfx950_specs:
        by_dtype[spec.dtype][(spec.kernel_family, spec.direct_variant)] += 1
    for dtype in ("bf16", "fp16"):
        assert by_dtype[dtype] == {
            (0, "none"): 141,
            (1, "depthwise_dgrad"): 9,
            (1, "depthwise_dgrad_win"): 31,
            (1, "grouped_dgrad_fused"): 16,
            (1, "grouped_dgrad_mfma"): 3,
            (1, "grouped_dgrad_4c"): 2,
            (1, "dgrad_mfma_rtc"): 1,
            (2, "transpose"): 1,
            (2, "reorganize"): 1,
            (2, "transpose_rtc"): 1,
            (2, "reorganize_rtc"): 1,
        }
    assert sum(1 for s in gfx950_specs if not s.is_helper) == 406
    assert len(gfx950_specs) == 414


def test_instance_id_is_the_identity_stable_hash(identities):
    from benchmarks.common.kernel_cache import KernelIdentity

    for identity in identities:
        spec, _ = tool.instance_spec(identity)
        assert spec.instance_id == KernelIdentity.from_dict(identity).stable_hash()


def test_the_shipped_bundle_is_its_catalog_with_instance_derived_ids(
    arch, specs, shipped
):
    """Every shard holds exactly the lists' instances of that shard; spec and metadata
    are the instance's fields in its variant's order; ids are uuid5(NS, instance_id)
    because this engine is its own product engine."""
    assert sorted(shipped) == sorted(s.instance_id for s in specs)
    for spec in specs:
        assert spec.ARCH == arch
        shard, kernel = shipped[spec.instance_id]
        assert shard == tool.shard_name(spec)
        expected = {name: getattr(spec, name) for name in tool.spec_order(spec)}
        assert kernel["kernel_source"]["spec"] == expected
        assert kernel["metadata"] == expected
        assert kernel["kernel_source"]["source"] == tool.ADAPTER_SOURCE
        assert kernel["kernel_source"]["builder"] == tool.adapter_builder(arch)
        assert kernel["id"] == str(
            uuid.uuid5(tool.INSTANCE_UUID_NAMESPACE, spec.instance_id)
        )
        assert kernel["id"] == tool.kernel_uuid(spec.instance_id, arch=arch)
        # The tool's own templates, not a second copy of them: a variant this engine
        # gains is named once.
        template = (
            tool.IMPLICIT_VARIANT_NAME
            if not spec.is_direct
            else tool._DIRECT_VARIANT_NAMES[spec.direct_variant]
        )
        assert kernel["name"] == template.format(
            arch=arch, dtype=spec.dtype, instance_id=spec.instance_id
        )


def test_the_shipped_bundle_is_sharded_below_the_hosting_size_limit(arch, specs):
    # The generated KDPs (one per shard) and a promoted model packet's heuristics/
    # both live under DVC, each pinned by its own top-level pointer; the bundle's
    # directly-committed files are the small, hand-authored ones beside them.
    folder = shipped_dir(arch)
    names = sorted(
        path.name
        for path in folder.iterdir()
        if path.name not in ("artifacts", "heuristics")
    )
    shards = {tool.shard_name(spec) for spec in specs}
    assert shards == {
        f"{dtype}_{p}"
        for dtype in ("bf16", "fp16")
        for p in _PIPELINES[arch] + ("direct",)
    }
    assert names == sorted(
        [f"{slug(arch)}.{kind}.json" for kind in ("kmd", "udd", "ued", "uhd")]
        + ["kernel_dtype_matches_graph.umd.json", "artifacts.dvc", ".gitignore"]
        + (["heuristics.dvc"] if ships_a_ranker(arch) else [])
    )
    matcher = json.loads((folder / "kernel_dtype_matches_graph.umd.json").read_text())
    top_level = [path for path in folder.iterdir() if path.is_file()]
    for path in top_level + instance_lists(arch) + [census(arch)]:
        assert path.stat().st_size < _FILE_SIZE_LIMIT, path.name
    for kdp in (folder / "artifacts").glob("*.kdp.json"):
        kdp_document = json.loads(kdp.read_text())
        assert kdp_document["matchers"] == [matcher["id"]]
        assert kdp_document["engine"] == document(arch, "ued")["id"]
        assert kdp_document["dispatch"] == document(arch, "udd")["id"]


def test_the_bundle_names_its_own_native_symbols(arch):
    ued, udd, uhd, kmd = (document(arch, k) for k in ("ued", "udd", "uhd", "kmd"))
    umd = json.loads(
        (shipped_dir(arch) / "kernel_dtype_matches_graph.umd.json").read_text()
    )
    symbol = f"hipkernel.{slug(arch)}"
    assert ued["name"] == tool.product_engine(arch)
    assert ued["graph_match"] == {"native": f"{symbol}.graph_match"}
    assert udd["dispatch_symbol"] == f"{symbol}.dispatch"
    assert uhd["native"] == {"symbol": f"{symbol}.score"}
    assert umd["match_symbol"] == f"{symbol}.kernel_match"
    assert ued["metadata"] == kmd["id"]
    # The native fallback scorer comes first; a promoted model packet appends its own
    # id after it instead of replacing it, so "default" is a list once one ships.
    assert list(ued["sort_kernel_catalog"]) == ["default"]
    default = ued["sort_kernel_catalog"]["default"]
    if ships_a_ranker(arch):
        assert len(default) > 1 and default[0] == uhd["id"]
        assert all(isinstance(i, str) for i in default)
    else:
        assert default == uhd["id"]
    assert ued["knobs"] == list(tool.engine_knobs(arch))
    kmd_names = [f["name"] for f in kmd["fields"]]
    assert kmd_names == [f["name"] for f in tool.kmd_fields(arch)]
    assert set(ued["knobs"]) <= set(kmd_names)
    assert "instance_id" not in ued["knobs"] and "max_sub_gemms" not in ued["knobs"]


def test_the_config_borrows_no_native_symbols(arch, tmp_path, specs):
    product = tool.product_engine(arch)
    config = tool.build_config(specs)
    assert config["engine"]["name"] == product
    assert "native_symbols_from" not in config["engine"]
    assert config["workspace_policy"] == "derived"
    other = tool.build_config(specs[:3], f"{product}Seed")
    namespace = tool.product_native_namespace(arch)
    assert other["engine"]["native_symbols_from"] == namespace
    # The stated namespace is the one the product engine registers.
    path = tmp_path / "dgrad.yaml"
    path.write_text(tool.dump_config(tool.build_config(specs[:3])))
    assert load_config(path).native_symbol_namespace == namespace
    assert namespace == EngineSpec(name=product).own_native_symbol_namespace
    assert tool.kernel_uuid(
        specs[0].instance_id, "hipkernel:X", arch=arch
    ) != tool.kernel_uuid(specs[0].instance_id, arch=arch)


def test_generated_config_expands_to_one_kernel_per_instance(tmp_path, arch, specs):
    path = tmp_path / "dgrad.yaml"
    path.write_text(tool.dump_config(tool.build_config(specs)))
    config = load_config(path)
    assert config.engine.name == tool.product_engine(arch)
    assert config.is_multi_pack and not config.has_operation_matchers
    assert sum(len(pack.kernels) for pack in config.packs) == len(specs)


def test_bundle_ids_are_unique_across_the_conv_aot_bundles():
    """The fwd, seed and every dgrad bundle load into one process: no document or
    kernel id may repeat across them, KDPs included."""
    dgrad = [shipped_dir(arch) for arch in ARCHES]
    for folder in dgrad:
        assert any((folder / "artifacts").glob("*.kdp.json")), folder
    ids = {}
    for folder in (*_FWD_DIRS, *dgrad):
        for path in (*folder.glob("*.json"), *folder.glob("artifacts/*.json")):
            found = json.loads(path.read_text())
            values = [found["id"]] + [
                kernel["id"] for kernel in found.get("kernelDescriptors", [])
            ]
            for value in values:
                assert ids.setdefault(value, path) == path, value


def _knob_tuple(kernel, kmd_defaults, arch="gfx950"):
    values = dict(kmd_defaults, **kernel["metadata"])
    return tuple(values[k] for k in tool.engine_knobs(arch))


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


def test_knob_tuples_are_unique_within_every_dtype_family_and_capability(arch, shipped):
    """GenericPlanBuilder refuses a graph whose matched catalog repeats a UED knob
    tuple. The matcher pins dtype and every direct capability exactly, so instances
    that can meet in one graph share a (dtype, family, capability) partition."""
    defaults = {
        f["name"]: f.get("default_value") for f in document(arch, "kmd")["fields"]
    }
    partitions = collections.defaultdict(list)
    for _, kernel in shipped.values():
        if kernel["metadata"].get("kernel_family") == 2:
            continue  # helpers are never matched
        partitions[_capability(kernel, defaults)].append(
            _knob_tuple(kernel, defaults, arch)
        )
    # Two dtypes x the capability sets the implicit, direct and helper families
    # span, which grew with the pipelines the sweep now enumerates.
    assert len(partitions) == _PARTITIONS[arch]
    for key, tuples in partitions.items():
        assert len(tuples) == len(set(tuples)), key
    # Implicit GEMM and the direct kernels can meet in one graph too. Uniqueness is
    # per capability, checked above: two tiles baked for different geometries never
    # serve the same graph, so they may share a knob tuple.
    for dtype in ("bf16", "fp16"):
        implicit = [
            kernel
            for _, kernel in shipped.values()
            if kernel["metadata"]["dtype"] == dtype
            and kernel["metadata"].get("kernel_family", 0) == 0
        ]
        assert len(implicit) == _IMPLICIT_PER_DTYPE[arch]


def test_the_mfma_row_tile_is_the_knob_token_not_the_identity_field(
    gfx950_identities, gfx950_specs
):
    by_id = {s.instance_id: s for s in gfx950_specs}
    from benchmarks.common.kernel_cache import KernelIdentity

    seen = set()
    for identity in gfx950_identities:
        ident = KernelIdentity.from_dict(identity)
        if ident.algorithm != "direct_grouped_dgrad_mfma":
            continue
        assert ident.block_h == 0
        spec = by_id[ident.stable_hash()]
        assert spec.block_h == json.loads(ident.knobs)["block_h"] > 0
        seen.add(spec.block_h)
    assert seen == {8, 16}


def test_every_mfma_instance_finds_its_helpers_in_the_bundle(gfx950_shipped):
    """The MFMA dispatch looks its helpers up by KMD field values; each lookup finds
    exactly one helper kernel of the same shard."""
    from builders.common.convolution_backward_data import (
        gfx950_conv_dgrad_aot_helper_fields,
    )

    shipped = gfx950_shipped
    defaults = {
        f["name"]: f.get("default_value") for f in document("gfx950", "kmd")["fields"]
    }
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
        # Both MFMA pipelines: the baked-channel one and the runtime-channel one,
        # whose transforms are the _rtc pair.
        if meta.get("direct_variant") not in ("grouped_dgrad_mfma", "dgrad_mfma_rtc"):
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


def test_the_census_names_every_pack_and_kernel(arch, shipped):
    text = census(arch).read_text()
    for _, kernel in shipped.values():
        assert f'"{kernel["name"]}"' in text
    for kdp in (shipped_dir(arch) / "artifacts").glob("*.kdp.json"):
        assert f'"{json.loads(kdp.read_text())["name"]}"' in text
    assert f'"{tool.product_engine(arch)}"' in text
    assert f"hipkernel.{slug(arch)}.score" in text
    assert f"knobs.size(), {len(tool.engine_knobs(arch))}U" in text


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"direction": "fwd"}, "dgrad implicit-GEMM"),
        ({"direction": "wgrad"}, "dgrad implicit-GEMM"),
        ({"arch": _NO_ENGINE_ARCH}, "no packaged dgrad engine"),
        ({"dtype_d": "fp32"}, "mixed operand dtypes"),
        ({"split_k": 2}, "split_k"),
        # Padding stays runtime-only: the tilde records carry it, and nothing bakes it.
        ({"pad_h": 1}, "pad_h"),
        ({"unroll_k": True}, "unroll_k"),
        ({"pipeline": "basic"}, "not produced"),
    ],
)
def test_unsupported_implicit_identities_are_named_with_their_reason(
    gfx950_identities, change, reason
):
    identity = copy.deepcopy(gfx950_identities[0])
    assert identity["algorithm"] == "implicit_gemm"
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


@pytest.fixture(scope="module")
def swept_implicit_identities():
    """Implicit-GEMM dgrad identities straight from rocKE's sweep, so these tests
    state the packaging contract rather than whatever the committed catalog holds."""
    from benchmarks.common import kernel_sweep
    from rocke.core.arch import ArchTarget

    jobs = kernel_sweep.enumerate_jobs(
        arch="gfx950",
        dtype="bf16",
        target=ArchTarget.from_gfx("gfx950"),
        directions=("dgrad",),
        jobs=1,
    )
    return [
        job.identity.to_dict()
        for job in jobs
        if job.identity.algorithm == "implicit_gemm"
    ]


@pytest.mark.parametrize("field", ["stride_h", "filter_h", "dilation_h"])
def test_an_implicit_identity_may_bake_its_geometry(swept_implicit_identities, field):
    """A tilde kernel compiled for one filter, stride or dilation states it, instead
    of leaving it 0 and reading it from the sub-GEMM records at run time. The matcher
    then gates on it (convolution_backward_data._implicit_serves)."""
    baked = [i for i in swept_implicit_identities if i[field]]
    if not baked:
        pytest.skip(f"the sweep enumerates no implicit dgrad kernel baking {field}")

    spec, why = tool.instance_spec(copy.deepcopy(baked[0]))

    assert spec is not None, why
    assert getattr(spec, field) == baked[0][field]


def test_an_implicit_identity_without_baked_geometry_stays_runtime(
    swept_implicit_identities,
):
    """The default is unchanged: 0, which is what serves every geometry."""
    runtime = [i for i in swept_implicit_identities if not i["stride_h"]]
    assert runtime, "the sweep must still enumerate runtime-geometry dgrad kernels"

    spec, why = tool.instance_spec(copy.deepcopy(runtime[0]))

    assert spec is not None, why
    for name in ("filter_h", "filter_w", "stride_h", "stride_w", "dilation_h"):
        assert getattr(spec, name) == 0


@pytest.mark.parametrize(
    ("algorithm", "change", "reason"),
    [
        (
            "direct_depthwise_dgrad",
            {"algorithm": "direct_grouped_dgrad"},
            "no packaged",
        ),
        ("direct_depthwise_dgrad", {"direction": "direct_fwd"}, "direct_dgrad"),
        # A knob the variant does not take is named and refused -- by rocKE's own
        # spec first, by aot_dgrad_direct_fields_from_knobs otherwise. Dropping it
        # silently would package the instance for a kernel nobody asked for.
        (
            "direct_depthwise_dgrad",
            {"knobs": '{"block_w": 4, "block_waves": 2, "nonsense": 1}'},
            "nonsense",
        ),
        # c32k32 leaves one K-atom, which the row's waves_k=2 cannot split.
        ("direct_grouped_dgrad_mfma", {"cpg": 32, "kpg": 32}, "K-atoms"),
        ("direct_dgrad_transpose", {"stride_h": 1, "stride_w": 1}, "cannot build"),
    ],
)
def test_unsupported_direct_identities_are_named_with_their_reason(
    gfx950_identities, algorithm, change, reason
):
    identity = copy.deepcopy(
        next(i for i in gfx950_identities if i["algorithm"] == algorithm)
    )
    identity.update(change)
    spec, why = tool.instance_spec(identity)
    assert spec is None and reason in why, why


def test_a_variant_may_state_only_the_knobs_its_mode_uses(gfx950_identities):
    """A variant's knobs are what it may set, not what it must: the windowed
    depthwise has a dot2 and an MFMA mode taking different ones, and the sweep
    states only the mode's. An omitted knob keeps rocKE's own default."""
    windowed = [
        i for i in gfx950_identities if i["algorithm"] == "direct_depthwise_dgrad_win"
    ]
    assert windowed, "the catalog must carry the windowed depthwise pipeline"

    modes = {("mfma" in i["knobs"], "dot2" in i["knobs"]) for i in windowed}
    assert len(modes) > 1, "both windowed modes must be exercised"
    for identity in windowed:
        spec, why = tool.instance_spec(copy.deepcopy(identity))
        assert spec is not None, why


def test_main_dedups_and_refuses_unsupported_unless_asked(
    tmp_path, arch, identities, capsys
):
    unsupported = dict(identities[0], split_k=2)
    entries = [identities[0], {"identity": identities[0]}, unsupported]
    instances = tmp_path / "instances.json"
    instances.write_text(json.dumps(entries))
    out = tmp_path / "out.yaml"
    argv = [
        "--arch",
        arch,
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


def test_main_refuses_an_arch_the_adapter_does_not_package(capsys):
    with pytest.raises(SystemExit):
        tool.main(["--arch", _NO_ENGINE_ARCH, "--stabilize-ids", "x.kdp.json"])
    assert f"invalid choice: '{_NO_ENGINE_ARCH}'" in capsys.readouterr().err


def test_stabilize_ids_derives_product_ids_and_matches_the_shipped_kdp(
    tmp_path, arch, capsys
):
    shipped = sorted((shipped_dir(arch) / "artifacts").glob("*_direct.kdp.json"))[0]
    scrambled = json.loads(shipped.read_text())
    for kernel in scrambled["kernelDescriptors"]:
        kernel["id"] = str(uuid.uuid4())
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(json.dumps(scrambled, indent=2) + "\n")
    assert tool.stabilize_ids(kdp, arch=arch) == len(scrambled["kernelDescriptors"])
    assert json.loads(kdp.read_text()) == json.loads(shipped.read_text())
    # The command line does the same, reading the engine from the in-repo rocKE.
    kdp.write_text(json.dumps(scrambled, indent=2) + "\n")
    assert tool.main(["--arch", arch, "--stabilize-ids", str(kdp)]) == 0
    assert json.loads(kdp.read_text()) == json.loads(shipped.read_text())
    capsys.readouterr()
    # Another product engine (the forward one) would derive other ids.
    for kernel in scrambled["kernelDescriptors"]:
        assert conv_aot_common.kernel_uuid(
            kernel["metadata"]["instance_id"],
            tool.product_engine(arch),
            "hipkernel:Gfx950ConvFwdAot",
        ) != tool.kernel_uuid(kernel["metadata"]["instance_id"], arch=arch)


def test_dumped_config_loads_back(specs):
    text = tool.dump_config(tool.build_config(specs))
    assert yaml.safe_load(text) == tool.build_config(specs)


class _Bound(Exception):
    """Raised by a stand-in _bind_rocke to stop a tool where it binds rocKE."""


def _bound_root(monkeypatch, module, attribute, argv, binder=None):
    """The rocKE root ``module.main(argv)`` binds, by way of ``<attribute>`` of
    ``binder`` (default: ``module``)."""

    def bind(root):
        raise _Bound(root)

    monkeypatch.setattr(binder or module, attribute, bind)
    with pytest.raises(_Bound) as bound:
        module.main(argv)
    return Path(bound.value.args[0])


def test_without_rocke_root_the_tool_binds_the_in_repo_tree(tmp_path, monkeypatch):
    assert tool.DEFAULT_ROCKE_ROOT.resolve() == _ROCKE.resolve()
    argv = ["--instances", str(tmp_path / "x.json"), "--output", str(tmp_path / "o")]
    root = _bound_root(monkeypatch, tool, "_bind_rocke", argv)
    assert root.resolve() == _ROCKE.resolve()
    # An explicit root wins.
    other = _bound_root(monkeypatch, tool, "_bind_rocke", argv + ["--rocke-root", "/x"])
    assert other == Path("/x")
