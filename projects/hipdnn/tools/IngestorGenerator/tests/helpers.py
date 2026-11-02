# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Factory helpers for building minimal valid model instances in unit tests, and
readers for the descriptor trees the generator writes."""

import json
from pathlib import Path
from types import SimpleNamespace

from codegen.models import (
    DIALECT_DIRECT_LOAD,
    DIALECT_PACKAGED,
    EngineSpec,
    GraphMatchSpec,
    IngestorConfig,
    KernelSource,
    KernelSpec,
    KmdField,
    PackSpec,
)
from codegen.generator import resolve_ids


def make_engine(**overrides) -> EngineSpec:
    defaults = dict(name="hipkernel:Test", knobs=["block_size"])
    defaults.update(overrides)
    return EngineSpec(**defaults)


def make_kmd_field(**overrides) -> KmdField:
    defaults = dict(name="block_size", type="int", default_value=64)
    defaults.update(overrides)
    return KmdField(**defaults)


def make_kernel(**overrides) -> KernelSpec:
    defaults = dict(
        name="test.f32_block64",
        kernel_source=KernelSource(
            kind="embedded_source", source_file="Test.cpp", entry_point="Test"
        ),
        metadata={"block_size": 64, "dtype": "FLOAT"},
    )
    defaults.update(overrides)
    return KernelSpec(**defaults)


def make_pack(**overrides) -> PackSpec:
    defaults = dict(name="test", kernels=[make_kernel()])
    defaults.update(overrides)
    return PackSpec(**defaults)


def make_minimal_config(**overrides) -> IngestorConfig:
    """A minimal valid single-pack IngestorConfig. ``specialization`` is derived from
    the caller's final ``kmd_fields`` as the all-matcher-only declaration the
    direct-load dialect requires; pass your own to test the declaration itself."""
    defaults = dict(
        engine=make_engine(),
        kmd_fields=[make_kmd_field(), KmdField(name="dtype", type="string")],
        packs=[make_pack()],
        graph_match=GraphMatchSpec(),
    )
    defaults.update(overrides)
    if defaults.get("dialect", DIALECT_DIRECT_LOAD) == DIALECT_DIRECT_LOAD:
        # Required of a direct-load bundle; the loader refuses one without it. The
        # packaged dialect legitimately falls back to <kind>/<slug>.
        defaults.setdefault("authored_subpath", "unit")
    defaults.setdefault(
        "specialization",
        {
            "metadata_fields": [],
            "matcher_only_fields": [f.name for f in defaults["kmd_fields"]],
            "bindings": {},
            "vocabulary": {},
        },
    )
    return IngestorConfig(**defaults)


def make_rocke_kernel(name: str, block: int) -> KernelSpec:
    """A packaged kernel whose candidate identity follows ``block``: two kernels built
    from one ``block`` are the same candidate, whatever their names."""
    return make_kernel(
        name=name,
        kernel_source=KernelSource(
            kind="rocke", source="m.py", builder="build_x", spec={"block": block}
        ),
        metadata={"block_size": block, "dtype": "FLOAT"},
    )


def make_packaged_config(subpath: str, blocks: list, **overrides) -> IngestorConfig:
    """A packaged single-pack bundle of the engine ``hipkernel:Test`` at
    ``descriptors/<subpath>``. Bundles of one engine differ in ``subpath``."""
    tag = subpath.rsplit("/", 1)[-1]
    return make_minimal_config(
        dialect=DIALECT_PACKAGED,
        kernel_source_kind="rocke",
        authored_subpath=subpath,
        packs=[
            make_pack(
                kernels=[make_rocke_kernel(f"{tag}.b{b}", b) for b in blocks],
                arch=["gfx950"],
            )
        ],
        **overrides,
    )


def read_docs(directory: Path) -> dict:
    """Every ``*.json`` directly under ``directory``, file name -> parsed document."""
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


def json_bytes(directory: Path) -> dict:
    """Every ``*.json`` directly under ``directory``, file name -> raw bytes."""
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.json"))}


def kdps(docs: dict) -> list:
    """The KDP documents of a :func:`read_docs` mapping."""
    return [doc for name, doc in docs.items() if name.endswith(".kdp.json")]


def all_ids(docs: dict) -> set:
    """Every descriptor id of a tree: documents and the kernels inside the KDPs."""
    ids = {doc["id"] for doc in docs.values()}
    for kdp in kdps(docs):
        ids.update(entry["id"] for entry in kdp["kernelDescriptors"])
    return ids


def render_resolved(generator, config, root, **resolve_kwargs):
    """Resolve ids, then render with exactly that resolution, as the CLI does.
    Returns ``(resolution, written)``."""
    resolution = resolve_ids(config, root, **resolve_kwargs)
    written = generator.render(config, root, resolution=resolution)
    return resolution, written


def home_and_satellite(generator, tmp_path, same_tree=False):
    """Bundle ``a`` (the home: every engine-level file) and bundle ``b`` (a satellite
    holding only its KDP) of one engine. The kernel at block 128 is in both.
    ``same_tree`` renders both into one output root, as an installed provider tree
    holds them: the bundles then sit side by side under ``descriptors/rocKE/``."""
    home = make_packaged_config("rocKE/a", [64, 128])
    home_root = tmp_path / "a"
    render_resolved(generator, home, home_root)
    satellite = make_packaged_config("rocKE/b", [128, 256])
    satellite_root = home_root if same_tree else tmp_path / "b"
    home_dir = home_root / home.descriptor_dir
    resolution, written = render_resolved(
        generator, satellite, satellite_root, engine_ids_from=[home_dir]
    )
    return SimpleNamespace(
        home=home,
        home_root=home_root,
        home_dir=home_dir,
        satellite=satellite,
        satellite_root=satellite_root,
        satellite_dir=satellite_root / satellite.descriptor_dir,
        satellite_resolution=resolution,
        satellite_written=written,
    )
