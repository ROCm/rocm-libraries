# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A small registry the registry_enumerator tests drive in place of rocKE.

`registry_enumerator` reaches a registry through the manifest's ``enumerate`` block:
a candidate-list callable, a request dataclass, and ``rocke.dispatch.tuning.walk``'s
``sweep_level``. This module supplies those three for two invented families, so a
test states exactly the registry behaviour it depends on and does not move when
rocKE's catalog, tiles, names or admission rules do:

- the *probe* family: an enum-typed request field (``mask``), a tri-state spec knob
  (``order``, ``None`` when unset), a free tile knob (``tile_n``) and a candidate that
  declines a head size it does not support;
- the *conv probe* family: a second, unrelated request/spec pair whose request-only
  fields (``C``, ``K``, ``G``) the spec does not carry.

``provider(tmp_path, monkeypatch)`` binds both to a throw-away provider root.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import sys
import types
import typing
from pathlib import Path

import yaml

ARCH = "gfx_probe"
_MODULE = "tests.synthetic_registry"


class Mask(enum.Enum):
    """Member values differ from member names, so a by-value spelling is detectable."""

    NO_MASK = 0
    CAUSAL = 1


@dataclasses.dataclass(frozen=True)
class ProbeRequest:
    """``algorithm`` and ``spec_id`` are request fields so a manifest can pin on them,
    as the real requests allow."""

    algorithm: str = "probe"
    spec_id: str = "grid"
    arch: str = ARCH
    hdim: int = 64
    heads: int = 8
    batch: int = 1
    dtype: str = "bf16"
    mask: Mask = Mask.NO_MASK

    def __post_init__(self):
        if not isinstance(self.mask, Mask):
            raise ValueError(f"mask must be a Mask, got {self.mask!r}")


@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    dtype: str
    tile_m: int
    tile_n: int
    order: typing.Optional[str] = None
    causal: bool = False

    def resolved_order(self):
        return "row" if self.order is None else self.order


class ProbeCandidate:
    """Declines any head size but 64 and 128 and any other arch; a ``None`` head size
    raises ``TypeError`` as a registry comparing it would."""

    def __init__(self, name: str, algorithm: str, spec_id: str):
        self.name, self.algorithm, self.spec_id = name, algorithm, spec_id

    def admits(self, request: ProbeRequest):
        if request.hdim <= 0:
            return False, "hdim must be positive"
        if request.arch != ARCH:
            return False, f"arch {request.arch!r} not in ({ARCH!r},)"
        if request.hdim not in (64, 128):
            return False, "hdim must be 64 or 128"
        return True, ""

    def select_spec(self, request: ProbeRequest):
        return ProbeSpec(
            dtype=request.dtype,
            tile_m=128,
            tile_n=64,
            causal=request.mask is Mask.CAUSAL,
        )

    def sweep_space(self, request: ProbeRequest):
        base = self.select_spec(request)
        return [
            dataclasses.replace(base, tile_n=tile_n, order=order)
            for tile_n in (64, 32, 128)
            for order in (None, "row", "col")
        ]


def probe_candidates():
    """``persist`` is shared by two candidates, so pinning it alone is ambiguous."""
    return [
        ProbeCandidate("probe_grid", "probe", "grid"),
        ProbeCandidate("probe_persist_a", "persist", "a"),
        ProbeCandidate("probe_persist_b", "persist", "b"),
    ]


@dataclasses.dataclass(frozen=True)
class ConvProbeRequest:
    algorithm: str = "conv_fwd"
    spec_id: str = "tile64"
    arch: str = ARCH
    N: int = 1
    C: int = 1
    K: int = 1
    G: int = 1
    dtype: str = "fp16"


@dataclasses.dataclass(frozen=True)
class ConvProbeSpec:
    direction: str
    dtype: str
    arch: str
    tile_m: int
    tile_n: int
    tile_k: int


class ConvProbeCandidate:
    name, algorithm, spec_id = "conv_probe", "conv_fwd", "tile64"

    def admits(self, request: ConvProbeRequest):
        return True, ""

    def select_spec(self, request: ConvProbeRequest):
        return ConvProbeSpec("fwd", request.dtype, request.arch, 64, 64, 64)

    def sweep_space(self, request: ConvProbeRequest):
        base = self.select_spec(request)
        return [base, dataclasses.replace(base, tile_m=128)]


def conv_probe_candidates():
    return [ConvProbeCandidate()]


def probe_manifest(provider_root, **enumerate_overrides) -> dict:
    """A slice-space manifest over the probe family: two cells, ``tile_n`` free."""
    kmd = [("dtype", "string")] + [
        (n, "int") for n in ("hdim", "heads", "tile_m", "tile_n")
    ]
    names = [n for n, _ in kmd]
    enumerate_block = {
        "candidates": f"{_MODULE}:probe_candidates",
        "request": f"{_MODULE}:ProbeRequest",
        "pin": {"algorithm": "probe", "spec_id": "grid"},
        "fixed": {"arch": ARCH, "dtype": "bf16"},
        "cells": [
            {"fields": ["hdim", "heads"], "values": [[64, 8], [128, 8]]},
            {"fields": ["mask"], "values": ["NO_MASK", "CAUSAL"]},
        ],
        "free_fields": ["tile_n"],
        "sweep_level": "production",
    }
    enumerate_block.update(enumerate_overrides)
    return {
        "dialect": "packaged",
        "authored_subpath": "synthetic/probe",
        "engine": {
            "name": "hipkernel:Probe",
            "sdk_version": "1.0.0",
            "knobs": ["tile_n"],
            "heuristic": "native",
        },
        "kmd_fields": [{"name": n, "type": t} for n, t in kmd],
        "specialization": {
            "metadata_fields": names,
            "matcher_only_fields": [],
            "vocabulary": {"dtype": {"bf16": "BF16", "fp16": "FP16"}},
            "bindings": {n: {"field": n} for n in names},
        },
        "kernel_source_kind": "rocke",
        "workspace_policy": "none",
        "pack": "probe",
        "arch": ARCH,
        "provider_root": str(provider_root),
        "source": "kernels/probe.py",
        "builder": "build_probe",
        "kernel_name_template": "probe.{dtype}_d{hdim}_h{heads}_c{causal}_tn{tile_n}",
        "enumerate": enumerate_block,
    }


def conv_probe_manifest(provider_root) -> dict:
    """A full-space manifest over the conv probe family; ``C``, ``K`` and ``G`` are
    request fields only."""
    names = ["dtype", "C", "K", "G", "tile_m", "tile_n", "tile_k"]
    return {
        "dialect": "packaged",
        "authored_subpath": "synthetic/conv_probe",
        "engine": {
            "name": "hipkernel:ConvProbe",
            "sdk_version": "1.0.0",
            "knobs": [],
            "heuristic": "native",
        },
        "kmd_fields": [
            {"name": n, "type": "string" if n == "dtype" else "int"} for n in names
        ],
        "specialization": {
            "metadata_fields": names,
            "matcher_only_fields": [],
            "vocabulary": {"dtype": {"fp16": "FP16", "bf16": "BF16"}},
            "bindings": {n: {"field": n} for n in names},
        },
        "kernel_source_kind": "rocke",
        "workspace_policy": "none",
        "pack": "conv_probe",
        "arch": ARCH,
        "provider_root": str(provider_root),
        "source": "kernels/conv_probe.py",
        "builder": "build_conv_probe",
        "kernel_name_template": (
            "conv_probe.{dtype}_c{C}_k{K}_g{G}_t{tile_m}x{tile_n}x{tile_k}"
        ),
        "enumerate": {
            "candidates": f"{_MODULE}:conv_probe_candidates",
            "request": f"{_MODULE}:ConvProbeRequest",
            "pin": {"algorithm": "conv_fwd", "spec_id": "tile64"},
            "fixed": {"arch": ARCH, "N": 1},
            "cells": [
                {"fields": ["C", "K", "G"], "values": [[64, 64, 1], [128, 128, 1]]},
                {"fields": ["dtype"], "values": ["fp16", "bf16"]},
            ],
            "sweep_level": "production",
            "space": "full",
        },
        "reduction": {
            "fields": {
                "feature": ["direction", "dtype", "arch"],
                "performance": [],
                "ignored": ["tile_m", "tile_n", "tile_k"],
            }
        },
    }


def write_manifest(path: Path, manifest: dict) -> Path:
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


@contextlib.contextmanager
def _sweep_level(level):
    yield


def provider(tmp_path: Path, monkeypatch) -> Path:
    """A throw-away provider root (``rocke/library`` and ``rocke/platform/python``
    present and empty) and a stand-in for ``rocke.dispatch.tuning.walk``, so the
    tool's registry boundary is reachable with no rocKE on disk. Restored when the
    test ends."""
    root = tmp_path / "provider"
    for sub in ("rocke/library", "rocke/platform/python"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    walk = types.ModuleType("rocke.dispatch.tuning.walk")
    walk.sweep_level = _sweep_level
    tuning = types.ModuleType("rocke.dispatch.tuning")
    tuning.walk = walk
    dispatch = types.ModuleType("rocke.dispatch")
    dispatch.tuning = tuning
    package = types.ModuleType("rocke")
    package.dispatch = dispatch
    for module in (package, dispatch, tuning, walk):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return root
