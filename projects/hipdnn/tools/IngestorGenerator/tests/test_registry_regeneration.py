# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The authored selection (configs/gfx950_attention_dense.yaml), gated against the
rocKE registry space named by the manifest and against the shipped catalog.

Spec fields the selection omits are compared against their dataclass defaults, so a
changed default (or builder) recompiles shipped kernels with every gate here green.
"""

from __future__ import annotations

import collections
import dataclasses
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

_PROVIDER = enumerator._REPO / "dnn-providers" / "hip-kernel-provider"
_ROCKE_LIBRARY = _PROVIDER / "rocke" / "library"
_SHIPPED_DIR = (
    _PROVIDER
    / "src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_attention_dense"
)
_SHIPPED_KDP = _SHIPPED_DIR / "gfx950_attention_dense.kdp.json"
_SHIPPED_KMD = _SHIPPED_DIR / "gfx950_attention_dense.kmd.json"
_SELECTION = _ROOT / "configs" / "gfx950_attention_dense.yaml"
_MANIFEST = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"


@pytest.fixture(scope="module")
def selection():
    return load_config(_SELECTION)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return enumerator.load_manifest(str(_MANIFEST))


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


def _key(name: str, source: tuple, spec: dict, metadata: dict, fields: frozenset):
    """`source` is the kernel_source (kind, source, builder)."""
    return (
        name,
        source,
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
    """Every selection kernel is in the enumeration, and every spec field it omits
    sits at its dataclass default (what hkp_pack compiles for an omitted field)."""
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
    selection, tmp_path_factory, capsys
):
    if not _SHIPPED_KDP.is_file():
        pytest.skip(f"shipped KDP not in this checkout: {_SHIPPED_KDP}")
    out = tmp_path_factory.mktemp("g2")
    IngestorGenerator(_ROOT / "templates").render(selection, out)
    descriptors = out / selection.descriptor_dir
    (kdp,) = descriptors.glob("*.kdp.json")
    (kmd,) = descriptors.glob("*.kmd.json")
    rc = catalog_diff.main(
        [
            "--shipped",
            str(_SHIPPED_KDP),
            "--regenerated",
            str(kdp),
            "--ignore-ids",
            "--shipped-kmd",
            str(_SHIPPED_KMD),
            "--regenerated-kmd",
            str(kmd),
        ]
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


def test_g5_the_dtype_vocabulary_is_spelled_as_the_cpp_matcher_compares_it(manifest):
    """The manifest's vocabulary is what the generator writes into every kernel's
    metadata, so a misspelling there agrees with itself in every Python gate while the
    matcher's exact string comparison never selects the kernel."""
    matcher = next(s for s in manifest["launch_surface"] if s["name"] == "matcher")
    source = (enumerator._REPO / matcher["cpp_mirror"]).read_text(encoding="utf-8")
    function = re.search(r"supportedDataTypeName\([^)]*\)\s*\{(.*?)\n\}", source, re.S)
    assert function, "supportedDataTypeName is not in the matcher source"
    compared = set(re.findall(r'std::string\("([^"]+)"\)', function[1]))
    declared = set(manifest["specialization"]["vocabulary"]["dtype"].values())
    assert compared and declared <= compared, (sorted(declared), sorted(compared))
