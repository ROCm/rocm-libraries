# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Each engine's authored selection (configs/<engine>.yaml), gated against the rocKE
registry space named by its manifest (configs/<engine>.engine.yaml) and against the
shipped catalog.

Spec fields the selection omits are compared against their dataclass defaults, so a
changed default (or builder) recompiles shipped kernels with every gate here green.
"""

from __future__ import annotations

import collections
import dataclasses
import importlib
import inspect
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config
from codegen.generator import IngestorGenerator

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import catalog_diff  # noqa: E402
import registry_enumerator as enumerator  # noqa: E402

# hkp_pack's own spec hydration and support predicate: g6 runs the pack-time checks that
# need no compiler, so a spec hkp_pack would refuse fails here and not in a build.
sys.path.insert(
    0,
    str(
        enumerator._REPO
        / "dnn-providers/hip-kernel-provider/descriptor-packaging/python"
    ),
)
from hkp_pack import rocke_compile  # noqa: E402

_PROVIDER = enumerator._REPO / "dnn-providers" / "hip-kernel-provider"
_ROCKE_LIBRARY = _PROVIDER / "rocke" / "library"
_DESCRIPTORS = _PROVIDER / "src/engines/kernel_ingestor_engine/descriptors"

# Every engine manifest is gated: a new configs/<engine>.engine.yaml gets drift
# gating by existing, with its selection at configs/<engine>.yaml.
_MANIFESTS = sorted((_ROOT / "configs").glob("*.engine.yaml"))


@pytest.fixture(scope="module", params=_MANIFESTS, ids=lambda p: p.name)
def manifest_path(request) -> Path:
    return request.param


@pytest.fixture(scope="module")
def selection_path(manifest_path) -> Path:
    path = manifest_path.with_name(manifest_path.name.replace(".engine.yaml", ".yaml"))
    # Not a skip: a manifest without its selection gates nothing, and a strict run
    # (hipdnn_strict_skips) would fail the skip anyway.
    assert path.is_file(), f"{manifest_path.name} has no authored selection {path.name}"
    return path


@pytest.fixture(scope="module")
def shipped(manifest, manifest_path) -> tuple:
    """(KDP, KMD) the provider ships for this manifest's pack."""
    pack = manifest["pack"]
    directory = _DESCRIPTORS / manifest["authored_subpath"]
    kdp = directory / "artifacts" / f"{pack}.kdp.json"
    if not kdp.is_file():
        pytest.skip(f"shipped KDP not in this checkout: {kdp}")
    return kdp, directory / f"{pack}.kmd.json"


@pytest.fixture(scope="module")
def selection(selection_path):
    return load_config(selection_path)


@pytest.fixture(scope="module")
def manifest(manifest_path) -> dict:
    return enumerator.load_manifest(str(manifest_path))


@pytest.fixture(scope="module")
def registry(manifest):
    if not _ROCKE_LIBRARY.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE_LIBRARY}")
    enumerator.bind_provider(manifest.get("provider_root"))
    return enumerator.Registry(manifest)


@pytest.fixture(scope="module")
def registry_config(manifest, registry) -> dict:
    return enumerator.build_config(manifest, registry)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _selection_kernels(selection) -> list:
    return [k for pack in selection.packs for k in pack.kernels]


def _shipped_spec_fields(selection) -> frozenset:
    return frozenset().union(
        *(k.kernel_source.spec.keys() for k in _selection_kernels(selection))
    )


def _key(
    name: str,
    source: tuple,
    spec: dict,
    metadata: dict,
    default: bool,
    fields: frozenset,
    launches: dict,
):
    """`source` is the kernel_source (kind, source, builder); `default` the kernel's
    `default` marker, which reduction keeps as the cell's floor; `launches` maps each
    launch name to its (source, builder, role), so a dropped or re-roled launch is a
    different kernel."""
    return (
        name,
        source,
        default,
        _canonical({k: v for k, v in spec.items() if k in fields}),
        _canonical(metadata),
        _canonical(launches),
    )


def _selection_launches(kernel) -> dict:
    return {
        name: [launch.source, launch.builder, launch.role]
        for name, launch in kernel.kernel_source.launches.items()
    }


def _registry_launches(source: dict) -> dict:
    return {
        name: [
            override.get("source", source.get("source", "")),
            override.get("builder", source.get("builder", "")),
            override.get("role", ""),
        ]
        for name, override in (source.get("launches") or {}).items()
    }


def _selection_key(kernel, fields: frozenset) -> tuple:
    return _key(
        kernel.name,
        (
            kernel.kernel_source.kind,
            kernel.kernel_source.source,
            kernel.kernel_source.builder,
        ),
        kernel.kernel_source.spec,
        kernel.metadata,
        kernel.default,
        fields,
        _selection_launches(kernel),
    )


def _selection_keys(selection) -> collections.Counter:
    fields = _shipped_spec_fields(selection)
    return collections.Counter(
        _selection_key(k, fields) for k in _selection_kernels(selection)
    )


def _registry_key(kernel: dict, fields: frozenset) -> tuple:
    source = kernel["kernel_source"]
    return _key(
        kernel["name"],
        (source["kind"], source.get("source", ""), source.get("builder", "")),
        source["spec"],
        kernel["metadata"],
        kernel.get("default", False),
        fields,
        _registry_launches(source),
    )


def _registry_keys(registry_config: dict, fields: frozenset) -> collections.Counter:
    return collections.Counter(
        _registry_key(k, fields)
        for pack in registry_config["packs"]
        for k in pack["kernels"]
    )


def _field_default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def test_g1_every_selection_kernel_is_a_registry_kernel_at_defaults(
    selection, manifest, registry, registry_config
):
    """Every selection kernel is in the enumeration, with the same `default` marker,
    and every spec field it omits sits at its dataclass default (what hkp_pack compiles for an omitted field).
    """
    fields = _shipped_spec_fields(selection)
    missing = _selection_keys(selection) - _registry_keys(registry_config, fields)
    assert not missing, sorted(key[0] for key in missing)[:10]
    off_default = _off_default(selection, manifest, registry, registry_config)
    assert not off_default, sorted({repr(o) for o in off_default})[:10]


def _off_default(selection, manifest, registry, registry_config) -> list:
    """``(field, registry value, default)`` for every spec field a shipped selection
    kernel omits while the registry spec it keys onto holds it off its dataclass
    default: hkp_pack compiles the default for an omitted field.

    The omitted fields are read from the SELECTION kernel: one the selection drops
    from every kernel also leaves the shared key, so the registry kernel's own spec
    keys would never name it."""
    fields = _shipped_spec_fields(selection)
    by_key = {_selection_key(k, fields): k for k in _selection_kernels(selection)}
    enumeration = enumerator.enumerate_space(manifest, registry)
    kernels = [k for pack in registry_config["packs"] for k in pack["kernels"]]
    assert len(kernels) == len(enumeration.rows)
    off_default = []
    for (_, ks, _), kernel in zip(enumeration.rows, kernels):
        shipped = by_key.get(_registry_key(kernel, fields))
        if shipped is None:
            continue
        carried = shipped.kernel_source.spec
        for f in dataclasses.fields(ks):
            if f.name in carried:
                continue
            default = _field_default(f)
            if default is dataclasses.MISSING or getattr(ks, f.name) != default:
                off_default.append((f.name, getattr(ks, f.name), default))
    return off_default


def test_g1_catches_a_field_the_selection_omits_from_every_kernel(
    selection, manifest, registry, registry_config
):
    """The negative control for g1's default check: drop, from EVERY selection kernel,
    a field whose value differs from its dataclass default. The selection still keys
    onto the registry (the field left the shared key with it), and hkp_pack would
    compile the default, so the gate must fire on that field."""
    import copy

    fields = _shipped_spec_fields(selection)
    registry_keys = _registry_keys(registry_config, fields)
    spec_cls = type(enumerator.enumerate_space(manifest, registry).rows[0][1])
    defaults = {f.name: _field_default(f) for f in dataclasses.fields(spec_cls)}
    target = next(
        (
            name
            for kernel in _selection_kernels(selection)
            if _selection_key(kernel, fields) in registry_keys
            for name, value in kernel.kernel_source.spec.items()
            if defaults.get(name, dataclasses.MISSING) is not dataclasses.MISSING
            and value != defaults[name]
            and all(name in k.kernel_source.spec for k in _selection_kernels(selection))
        ),
        None,
    )
    assert target, "no selection field sits off its default in every kernel"
    stripped = copy.deepcopy(selection)
    for kernel in _selection_kernels(stripped):
        kernel.kernel_source.spec.pop(target, None)
    assert not (
        _selection_keys(stripped)
        - _registry_keys(registry_config, _shipped_spec_fields(stripped))
    ), "the stripped selection should still key onto the registry"
    off_default = _off_default(stripped, manifest, registry, registry_config)
    assert target in {name for name, _, _ in off_default}, (target, off_default[:5])


def test_g2_the_shipped_catalog_is_generated_from_the_selection(
    selection, shipped, tmp_path_factory, capsys
):
    shipped_kdp, shipped_kmd = shipped
    out = tmp_path_factory.mktemp("g2")
    IngestorGenerator(_ROOT / "templates").render(selection, out)
    descriptors = out / selection.descriptor_dir
    (kdp,) = descriptors.glob("*.kdp.json")
    (kmd,) = descriptors.glob("*.kmd.json")
    # --strict-spec: compare the FULL spec dict of every entry, not just the
    # intersection of spec keys present in the shipped catalog.  Without it,
    # a newly added spec field that hkp_pack would hydrate (and that changes
    # the compiled binary) passes silently if it is absent from the shipped
    # catalog.  The selection carries exactly the shipped spec fields, so
    # --strict-spec holds here; the registry enumeration (g3) carries one more.
    rc = catalog_diff.main(
        [
            "--shipped",
            str(shipped_kdp),
            "--regenerated",
            str(kdp),
            "--ignore-ids",
            "--strict-spec",
            "--shipped-kmd",
            str(shipped_kmd),
            "--regenerated-kmd",
            str(kmd),
            "--shipped-descriptors",
            str(shipped_kmd.parent),
            "--regenerated-descriptors",
            str(descriptors),
        ]
    )
    report = capsys.readouterr()
    assert rc == 0, report.out + report.err


def _full_space(manifest: dict) -> bool:
    decl = manifest["enumerate"]
    return decl.get("space") == "full" or any(
        section.get("space", decl.get("space")) == "full"
        for section in decl.get("also") or []
    )


def test_g3_the_enumerated_registry_reproduces_the_shipped_catalog(
    manifest, registry_config, shipped, tmp_path_factory, capsys
):
    """The manifest's own enumeration, rendered and diffed against the shipped
    catalog: the gate that the registry, not the authored selection, still yields
    the shipped kernels. Keyed on the shipped spec fields, because the enumeration
    also carries `lds_k_row_pad`, which the shipped specs omit and g1 asserts sits at
    its default. A ``space: full`` catalog is reduced (variant_reduce), so the shipped
    kernels must be a SUBSET of the enumeration rather than equal to it."""
    shipped_kdp, shipped_kmd = shipped
    out = tmp_path_factory.mktemp("g3")
    path = out / "registry.yaml"
    path.write_text(yaml.safe_dump(registry_config, sort_keys=False))
    config = load_config(path)
    IngestorGenerator(_ROOT / "templates").render(config, out / "generated")
    descriptors = out / "generated" / config.descriptor_dir
    (kdp,) = descriptors.glob("*.kdp.json")
    (kmd,) = descriptors.glob("*.kmd.json")
    rc = catalog_diff.main(
        [
            "--shipped",
            str(shipped_kdp),
            "--regenerated",
            str(kdp),
            "--ignore-ids",
            "--shipped-kmd",
            str(shipped_kmd),
            "--regenerated-kmd",
            str(kmd),
            "--shipped-descriptors",
            str(shipped_kmd.parent),
            "--regenerated-descriptors",
            str(descriptors),
        ]
        + (["--subset"] if _full_space(manifest) else [])
    )
    report = capsys.readouterr()
    assert rc == 0, report.out + report.err


def test_g4_the_selection_header_equals_the_manifest(
    selection, registry_config, tmp_path_factory
):
    path = tmp_path_factory.mktemp("g4") / "registry.yaml"
    path.write_text(yaml.safe_dump(registry_config, sort_keys=False))
    built = load_config(path)
    assert selection.engine == built.engine
    assert selection.kmd_fields == built.kmd_fields
    assert selection.specialization == built.specialization
    assert selection.reduction == built.reduction
    assert selection.dialect == built.dialect
    assert selection.kernel_source_kind == built.kernel_source_kind
    assert selection.workspace_policy == built.workspace_policy
    assert selection.authored_subpath == built.authored_subpath
    # One handler, one launch set (launchNames()): every selection kernel launches what
    # the manifest declares, by name, builder and role.
    assert {
        _canonical(_selection_launches(k)) for k in _selection_kernels(selection)
    } == {_canonical(_selection_launches(k)) for k in _selection_kernels(built)}


#: The graph dtypes the C++ matcher translates, as ``return std::string("<spelling>")``.
_MATCHER_DTYPE_SPELLING = re.compile(
    r'DataType::\w+\)\s*\{\s*return std::string\("(\w+)"\)'
)


def _matcher_source(manifest: dict) -> Path:
    """The engine's own ``*Native.cpp``, as its ``launch_surface`` names it."""
    mirrors = [
        surface["cpp_mirror"].split(":")[0]
        for surface in manifest.get("launch_surface") or []
        if surface.get("cpp_mirror", "").split(":")[0].endswith("Native.cpp")
    ]
    assert mirrors, "manifest names no *Native.cpp in launch_surface"
    return enumerator._REPO / mirrors[0]


def _vocabulary_gaps(vocabulary: set, matcher_text: str) -> tuple:
    """``(spellings the matcher never produces, spellings it produces that no
    vocabulary entry carries)``; both empty when they agree. The second direction
    catches a dtype the engine claims and ships nothing for."""
    spellings = set(_MATCHER_DTYPE_SPELLING.findall(matcher_text))
    assert spellings, "no dtype spellings read from the matcher source"
    return sorted(vocabulary - spellings), sorted(spellings - vocabulary)


def test_g5_the_dtype_vocabulary_is_what_the_cpp_matcher_compares(manifest):
    """`specialization.vocabulary.dtype` is the spelling the generated KDP's metadata
    carries; the hand-written matcher compares kernel dtype metadata against its own
    literals. A spelling the matcher never produces matches nothing, silently."""
    source = _matcher_source(manifest)
    assert source.is_file(), f"matcher source not readable: {source}"
    vocabulary = set(manifest["specialization"]["vocabulary"]["dtype"].values())
    assert _vocabulary_gaps(vocabulary, source.read_text()) == ([], [])


def test_g5_catches_a_misspelled_and_an_unclaimed_dtype(manifest):
    """The positive control: each direction fails on its own, and a matcher whose
    literals cannot be read fails rather than passing on an empty set."""
    text = _matcher_source(manifest).read_text()
    vocabulary = set(manifest["specialization"]["vocabulary"]["dtype"].values())
    missing, unclaimed = _vocabulary_gaps(vocabulary | {"BFLOAT16"}, text)
    assert (missing, unclaimed) == (["BFLOAT16"], [])
    missing, unclaimed = _vocabulary_gaps({"BF16"}, text)
    assert (missing, unclaimed) == ([], ["FP16"])
    with pytest.raises(AssertionError, match="no dtype spellings"):
        _vocabulary_gaps(vocabulary, "// no literals")


def _hydration_failures(manifest: dict, specs: list) -> list:
    """What `hkp_pack` does to a rocke spec before compiling it: hydrate the builder's
    spec dataclass with ``build_spec`` and ask the builder's support predicate. Neither
    needs comgr."""
    arch = manifest["arch"]
    module = importlib.import_module(
        rocke_compile._module_from_source(manifest["source"])
    )
    builder = manifest["builder"]
    spec_cls = rocke_compile._resolve_spec_class(module, getattr(module, builder))
    failures = []
    for name, source in _launch_sources(specs):
        compile_arch = rocke_compile.rocke_compile_arch(
            arch, source.get("target", rocke_compile.ROCKE_TARGET_NATIVE)
        )
        try:
            spec = rocke_compile.build_spec(spec_cls, source["spec"])
            rocke_compile._check_support_predicate(module, builder, spec, compile_arch)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    return failures


def _launch_sources(specs: list) -> list:
    """``(name, kernel_source)`` per launch: a multi-launch kernel's launches each
    compile on their own (a packed KDP carries them as ``sources``), a single-launch
    kernel is itself. A role-only launch shares the kernel's builder and spec."""
    out = []
    for name, source in specs:
        launches = source.get("launches") or source.get("sources")
        if not launches:
            out.append((name, source))
            continue
        for launch, override in launches.items():
            out.append((f"{name}[{launch}]", {**source, **override, "launches": None}))
    return out


def test_g6_every_enumerated_spec_hydrates_and_passes_the_builders_predicate(
    manifest, registry_config
):
    """A spec the registry yields but `hkp_pack` cannot build, or the builder's
    predicate declines, ships a catalog that fails at pack time on a machine with
    comgr -- and nowhere before it."""
    specs = [
        (k["name"], k["kernel_source"])
        for pack in registry_config["packs"]
        for k in pack["kernels"]
    ]
    assert specs
    failures = _hydration_failures(manifest, specs)
    assert not failures, failures[:5]


def test_g6_every_shipped_spec_hydrates_and_passes_the_builders_predicate(
    manifest, registry, shipped
):
    """The same check on the bytes that ship, which can name a spec field the
    registry has since renamed."""
    kdp = json.loads(shipped[0].read_text())
    specs = [
        (
            k["name"],
            k["kernel_source"] if "kernel_source" in k else {"sources": k["sources"]},
        )
        for k in kdp["kernelDescriptors"]
    ]
    assert specs
    failures = _hydration_failures(manifest, specs)
    assert not failures, failures[:5]


#: Per builder, a real spec field and a value its spec class accepts and its support
#: predicate declines, so g6's positive control reaches the predicate. A new builder
#: needs an entry here.
_PREDICATE_DECLINES = {
    # gfx950 dense ships block_m 128 and 256 only.
    "build_attention_dense": {"block_m": 64},
    # The tiled 2D/3D kernels are built for head_size 64, 128 and 256.
    "build_unified_attention_2d_tiled": {"head_size": 96},
    "build_unified_attention_3d_tiled": {"head_size": 96},
    "build_unified_attention_3d_tiled_module": {"head_size": 96},
}


def test_g6_the_module_yields_exactly_the_launched_roles(manifest, registry_config):
    """hkp_pack requires the launches to claim exactly the kernels a module builder
    compiles: its primary (no role) and every companion role. Build the first kernel's
    module (IR only, no comgr) and compare."""
    kernel = registry_config["packs"][0]["kernels"][0]
    launches = kernel["kernel_source"].get("launches")
    if not launches:
        assert not manifest.get("launches")
        return
    module = importlib.import_module(
        rocke_compile._module_from_source(manifest["source"])
    )
    builder = getattr(module, manifest["builder"])
    spec_cls = rocke_compile._resolve_spec_class(module, builder)
    built = builder(
        rocke_compile.build_spec(spec_cls, kernel["kernel_source"]["spec"]),
        arch=manifest["arch"],
    )
    roles = {override.get("role", "") for override in launches.values()}
    assert roles == {""} | set(getattr(built, "companions", {})), roles


def test_g6_catches_a_renamed_field_and_a_declined_spec(manifest, registry_config):
    """The positive control for g6: it fails on each way a spec can be refused, so a
    green g6 is not a gate that checks nothing. The declined spec names only real
    fields, so it fails in the builder's support predicate, not in hydration."""
    name, source = next(
        (k["name"], k["kernel_source"])
        for pack in registry_config["packs"]
        for k in pack["kernels"]
    )
    assert manifest["builder"] in _PREDICATE_DECLINES, (
        f"add a predicate-declined spec for {manifest['builder']} to "
        f"_PREDICATE_DECLINES"
    )
    renamed = {**source, "spec": {**source["spec"], "no_such_field": 1}}
    declined = {
        **source,
        "spec": {**source["spec"], **_PREDICATE_DECLINES[manifest["builder"]]},
    }
    failures = _hydration_failures(
        manifest, [("renamed", renamed), ("declined", declined)]
    )
    # One failure per launch: a multi-launch kernel's launches each hydrate on their own.
    by_kernel = collections.defaultdict(list)
    for failure in failures:
        by_kernel[failure.split(":")[0].split("[")[0]].append(failure)
    assert list(by_kernel) == ["renamed", "declined"], failures
    assert all("unexpected spec field" in f for f in by_kernel["renamed"]), failures
    assert all("spec rejected by" in f for f in by_kernel["declined"]), failures


def _observation_failures(manifest: dict, kernels: list) -> list:
    """What hkp_pack's specialization agreement does after a compile, minus the compile:
    observe each metadata field through its binding (a direct field, or a bound method) on
    the hydrated spec, translate strings through the vocabulary, and compare with the
    kernel's metadata."""
    arch = manifest["arch"]
    module = importlib.import_module(
        rocke_compile._module_from_source(manifest["source"])
    )
    builder = manifest["builder"]
    spec_cls = rocke_compile._resolve_spec_class(module, getattr(module, builder))
    specialization = manifest["specialization"]
    vocabulary = specialization.get("vocabulary") or {}
    types = {f["name"]: f["type"] for f in manifest["kmd_fields"]}
    failures = []
    for name, source, metadata in kernels:
        spec = rocke_compile.build_spec(spec_cls, source["spec"])
        for field, binding in specialization["bindings"].items():
            mode, attribute = next(iter(binding.items()))
            try:
                if mode == "method":
                    observed = getattr(spec, attribute)()
                else:
                    assert not isinstance(
                        inspect.getattr_static(type(spec), attribute, None), property
                    ), f"{attribute} is a property, not a direct field"
                    observed = getattr(spec, attribute)
            except Exception as exc:
                failures.append(f"{name}: {field} via {binding}: {exc}")
                continue
            if isinstance(observed, str):
                observed = vocabulary.get(field, {}).get(observed, observed)
            if types[field] == "int" and isinstance(observed, bool):
                observed = int(observed)
            if observed != metadata[field]:
                failures.append(
                    f"{name}: {field} observed {observed!r}, metadata {metadata[field]!r}"
                )
    return failures


def test_g7_every_enumerated_kernels_metadata_is_what_its_spec_observes(
    manifest, registry_config
):
    """hkp_pack refuses a kernel whose metadata disagrees with what its compiled spec
    reports through the manifest's bindings -- or whose binding names an attribute the
    spec lacks -- and only after a comgr build. This runs that comparison without one.
    """
    kernels = [
        (k["name"], k["kernel_source"], k["metadata"])
        for pack in registry_config["packs"]
        for k in pack["kernels"]
    ]
    assert kernels
    failures = _observation_failures(manifest, kernels)
    assert not failures, failures[:5]


def test_g7_catches_a_missing_accessor_and_a_disagreeing_value(
    manifest, registry_config
):
    """The positive control for g7."""
    k = registry_config["packs"][0]["kernels"][0]
    name, source, metadata = k["name"], k["kernel_source"], k["metadata"]
    field = next(iter(manifest["specialization"]["bindings"]))
    wrong = {**metadata, field: "not-the-observed-value"}
    assert _observation_failures(manifest, [("wrong", source, wrong)])

    broken = {
        **manifest,
        "specialization": {
            **manifest["specialization"],
            "bindings": {
                **manifest["specialization"]["bindings"],
                field: {"method": "no_such_accessor"},
            },
        },
    }
    failures = _observation_failures(broken, [(name, source, metadata)])
    assert any("no_such_accessor" in f for f in failures), failures
