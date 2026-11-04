# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Each shipped rocKE bundle's authored selection (configs/<engine>.yaml), gated against
the rocKE registry space named by its manifest and against the shipped catalog.

Spec fields the selection omits are compared against their dataclass defaults, so a
changed default (or builder) recompiles shipped kernels with every gate here green.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import sys
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config
from codegen.generator import IngestorGenerator, resolve_ids

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import catalog_diff  # noqa: E402
import registry_enumerator as enumerator  # noqa: E402
from tests.helpers import shipped_bundles  # noqa: E402

_PROVIDER = enumerator._REPO / "dnn-providers" / "hip-kernel-provider"
_ROCKE_LIBRARY = _PROVIDER / "rocke" / "library"
_DESCRIPTORS = _PROVIDER / "src/engines/kernel_ingestor_engine/descriptors/rocKE"


@dataclasses.dataclass(frozen=True)
class _Engine:
    """One shipped bundle. `reduced` marks a selection that variant_reduce cut from the
    enumeration, so g3 holds the shipped catalog to a subset of it rather than to all
    of it."""

    name: str
    reduced: bool

    @property
    def shipped_dir(self) -> Path:
        return _DESCRIPTORS / self.name

    @property
    def shipped_kdp(self) -> Path:
        return self.shipped_dir / "artifacts" / f"{self.name}.kdp.json"

    @property
    def shipped_kmd(self) -> Path:
        return self.shipped_dir / f"{self.name}.kmd.json"

    @property
    def selection(self) -> Path:
        return _ROOT / "configs" / f"{self.name}.yaml"

    @property
    def manifest(self) -> Path:
        return _ROOT / "configs" / f"{self.name}.engine.yaml"


# Bundles whose selection is a variant_reduce subset of the enumeration. Opt-in, so a
# new bundle gets the exact g3 by default and a registry that grows fails it.
_REDUCED = frozenset({"gfx942_attention_dense"})
_ENGINES = tuple(_Engine(name, reduced=name in _REDUCED) for name in shipped_bundles())


def test_every_reduced_bundle_is_shipped():
    assert _REDUCED <= set(shipped_bundles())


@pytest.fixture(scope="module", params=_ENGINES, ids=lambda e: e.name)
def engine(request) -> _Engine:
    return request.param


@pytest.fixture(scope="module")
def selection(engine):
    return load_config(engine.selection)


@pytest.fixture(scope="module")
def manifest(engine) -> dict:
    return enumerator.load_manifest(str(engine.manifest))


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
):
    """`source` is the kernel_source (kind, source, builder); `default` the kernel's
    `default` marker, which reduction keeps as the cell's floor."""
    return (
        name,
        source,
        default,
        _canonical({k: v for k, v in spec.items() if k in fields}),
        _canonical(metadata),
    )


def _selection_keys(selection) -> collections.Counter:
    fields = _shipped_spec_fields(selection)
    return collections.Counter(
        _key(
            k.name,
            (k.kernel_source.kind, k.kernel_source.source, k.kernel_source.builder),
            k.kernel_source.spec,
            k.metadata,
            k.default,
            fields,
        )
        for k in _selection_kernels(selection)
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
    shipped = set(_selection_keys(selection))
    enumeration = enumerator.enumerate_space(manifest, registry)
    kernels = [k for pack in registry_config["packs"] for k in pack["kernels"]]
    assert len(kernels) == len(enumeration.rows)
    off_default = []
    for (_, ks, _), kernel in zip(enumeration.rows, kernels):
        if _registry_key(kernel, fields) not in shipped:
            continue
        for f in dataclasses.fields(ks):
            if f.name in fields:
                continue
            default = _field_default(f)
            if default is dataclasses.MISSING or getattr(ks, f.name) != default:
                off_default.append((f.name, getattr(ks, f.name), default))
    assert not off_default, sorted({repr(o) for o in off_default})[:10]


def test_g2_the_shipped_catalog_is_generated_from_the_selection(
    engine, selection, tmp_path_factory, capsys
):
    """Generated with the shipped ids reused, the six descriptors are the shipped ones
    byte for byte."""
    if not engine.shipped_kdp.is_file():
        pytest.skip(f"shipped KDP not in this checkout: {engine.shipped_kdp}")
    out = tmp_path_factory.mktemp("g2")
    generator = IngestorGenerator(_ROOT / "templates")
    generator.render(
        selection,
        out,
        resolution=resolve_ids(selection, out, ids_from=engine.shipped_dir),
    )
    descriptors = out / selection.descriptor_dir
    (kdp,) = descriptors.glob("*.kdp.json")
    (kmd,) = descriptors.glob("*.kmd.json")
    # --strict-spec: compare the FULL spec dict of every entry, not just the
    # intersection of spec keys present in the shipped catalog.  Without it,
    # a newly added spec field that hkp_pack would hydrate (and that changes
    # the compiled binary) passes silently if it is absent from the shipped
    # catalog.  The selection carries exactly the shipped spec fields, so
    # --strict-spec holds here; a registry enumeration (g3) can carry one more.
    rc = catalog_diff.main(
        [
            "--shipped",
            str(engine.shipped_kdp),
            "--regenerated",
            str(kdp),
            "--expect-ids",
            "--strict-spec",
            "--shipped-kmd",
            str(engine.shipped_kmd),
            "--regenerated-kmd",
            str(kmd),
        ]
    )
    report = capsys.readouterr()
    assert rc == 0, report.out + report.err

    # The catalog diff covers the KDP's catalog; this covers every byte of all six
    # files. The KDP's DVC object is LF; the five git-tracked files are compared as
    # text because an autocrlf checkout holds them as CRLF.
    assert kdp.read_bytes() == engine.shipped_kdp.read_bytes()
    others = sorted(p for p in descriptors.glob("*.json") if p != kdp)
    assert len(others) == 5, [p.name for p in others]
    for generated in others:
        shipped = engine.shipped_dir / generated.name
        assert shipped.is_file(), f"{generated.name} is not a shipped descriptor"
        assert generated.read_text(encoding="utf-8") == shipped.read_text(
            encoding="utf-8"
        ), generated.name


def test_g3_the_enumerated_registry_reproduces_the_shipped_catalog(
    engine, registry_config, tmp_path_factory, capsys
):
    """The manifest's own enumeration, rendered and diffed against the shipped
    catalog: the gate that the registry, not the authored selection, still yields
    the shipped kernels. Keyed on the shipped spec fields, because an enumeration can
    carry a field the shipped specs omit (gfx950's `lds_k_row_pad`), which g1 asserts
    sits at its default. A reduced catalog is held to shipped ⊆ enumeration
    (`--subset`): every shipped kernel, id and KMD must still come out of the
    registry, and the kernels variant_reduce dropped are listed as extras."""
    if not engine.shipped_kdp.is_file():
        pytest.skip(f"shipped KDP not in this checkout: {engine.shipped_kdp}")
    if engine.reduced:
        shipped = len(json.loads(engine.shipped_kdp.read_text())["kernelDescriptors"])
        enumerated = sum(len(pack["kernels"]) for pack in registry_config["packs"])
        assert shipped < enumerated, (
            f"{engine.name} ships its whole enumeration ({shipped} kernels); drop it "
            f"from _REDUCED so g3 holds it to equality"
        )
    out = tmp_path_factory.mktemp("g3")
    path = out / "registry.yaml"
    path.write_text(yaml.safe_dump(registry_config, sort_keys=False))
    config = load_config(path)
    generated_root = out / "generated"
    IngestorGenerator(_ROOT / "templates").render(
        config,
        generated_root,
        resolution=resolve_ids(config, generated_root, ids_from=engine.shipped_dir),
    )
    descriptors = generated_root / config.descriptor_dir
    (kdp,) = descriptors.glob("*.kdp.json")
    (kmd,) = descriptors.glob("*.kmd.json")
    rc = catalog_diff.main(
        [
            "--shipped",
            str(engine.shipped_kdp),
            "--regenerated",
            str(kdp),
            "--expect-ids",
            "--shipped-kmd",
            str(engine.shipped_kmd),
            "--regenerated-kmd",
            str(kmd),
            *(["--subset"] if engine.reduced else []),
        ]
    )
    report = capsys.readouterr()
    assert rc == 0, report.out + report.err


def _knob_clashes(features, knobs, kernels) -> list:
    """Names of the kernels sharing one engine-knob tuple inside one cell, per clash.
    `kernels` is `(metadata, name)` pairs."""
    cells = collections.defaultdict(list)
    for metadata, name in kernels:
        cell = tuple(metadata.get(f) for f in features)
        cells[cell].append((tuple(metadata.get(k) for k in knobs), name))
    return [
        names
        for entries in cells.values()
        for names in _group_names(entries).values()
        if len(names) > 1
    ]


def test_g5_every_cell_is_addressable_by_the_engine_knobs(selection):
    """hipDNN pages a graph's candidates by their engine-knob tuple and refuses a page in
    which two candidates share one ("Ambiguous enrolled knob tuple"). A graph's candidates
    all sit in one cell (the reduction's feature fields), so inside every cell each
    kernel's tuple of engine knobs must be unique."""
    features = list(selection.reduction["fields"]["feature"])
    knobs = list(selection.engine.knobs)
    clashes = _knob_clashes(
        features,
        knobs,
        [(k.metadata, k.name) for k in _selection_kernels(selection)],
    )
    assert not clashes, (
        f"engine knobs {knobs} do not tell these kernels apart within a cell; "
        f"add the KMD field that differs to engine.knobs: {clashes[:3]}"
    )


# Bundles whose whole enumeration is a catalog a sweep installs: every kernel a reduction
# may keep must already page and load without a collision, before any is chosen.
_SWEPT_WHOLE = frozenset({"gfx942_attention_dense"})


@pytest.mark.parametrize("name", sorted(_SWEPT_WHOLE))
def test_g5_the_whole_enumeration_pages_and_loads_without_a_collision(name):
    """g5 over the manifest's full enumeration, not only the shipped selection: the swept
    catalog is the enumeration, so every field that varies inside a cell must be an engine
    knob (else enumerateCandidates refuses the page), and every kernel's metadata completed
    with the KMD defaults -- the loader's catalog key -- must be unique (else the loader
    keeps one and drops the other)."""
    if not _ROCKE_LIBRARY.is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE_LIBRARY}")
    manifest = enumerator.load_manifest(str(_Engine(name, reduced=True).manifest))
    enumerator.bind_provider(manifest.get("provider_root"))
    registry_config = enumerator.build_config(
        manifest, enumerator.Registry(manifest), warn=lambda _: None, strict=True
    )
    kernels = [k for pack in registry_config["packs"] for k in pack["kernels"]]
    features = list(manifest["reduction"]["fields"]["feature"])
    knobs = list(registry_config["engine"]["knobs"])
    clashes = _knob_clashes(
        features, knobs, [(k["metadata"], k["name"]) for k in kernels]
    )
    assert not clashes, (
        f"engine knobs {knobs} do not tell these enumerated kernels apart within a cell: "
        f"{clashes[:3]}"
    )
    varying = {
        f
        for f in {name for k in kernels for name in k["metadata"]}
        if len({tuple(k["metadata"].get(x) for x in features + [f]) for k in kernels})
        > len({tuple(k["metadata"].get(x) for x in features) for k in kernels})
    }
    assert varying <= set(knobs), (
        f"KMD fields {sorted(varying - set(knobs))} vary inside a cell but are not "
        f"engine knobs"
    )
    fields = registry_config["kmd_fields"]
    keys = collections.Counter(
        tuple(
            json.dumps(k["metadata"].get(f["name"], f.get("default_value")))
            for f in fields
        )
        for k in kernels
    )
    repeated = [key for key, count in keys.items() if count > 1]
    assert not repeated, f"{len(repeated)} catalog keys repeat, e.g. {repeated[:2]}"
    assert len(keys) == len(kernels)


def _group_names(entries):
    grouped = collections.defaultdict(list)
    for knob_values, name in entries:
        grouped[knob_values].append(name)
    return grouped


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
