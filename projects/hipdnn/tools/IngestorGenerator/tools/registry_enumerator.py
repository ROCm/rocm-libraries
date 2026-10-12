# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Enumerate an engine's kernel catalog from the rocKE dispatcher registry.

The registry is the single source of truth for which kernels exist. An engine
manifest (``configs/<slug>.engine.yaml``) states only what rocKE cannot know --
the engine's identity, its KMD schema, how a record is read as metadata, and
which request slice of the registry the engine ships. This tool walks that
slice and writes a generator config (the schema ``generate.py`` already
consumes, one kernel per registry spec), so adding, dropping or re-defaulting a
registry candidate changes the catalog on the next regeneration.

    registry_enumerator.py --manifest configs/gfx950_attention_dense.engine.yaml \\
                           --out <scratch>/config.yaml

The slice rule, from the manifest's ``enumerate`` block:

  * the candidate whose ``algorithm`` matches, and the request's ``spec_id``;
  * one request per ``(dtype, head_size, heads, causal)`` at the manifest's
    fixed shape;
  * every spec the candidate's ``sweep_space`` yields at ``sweep_level`` that
    differs from the candidate's own ``select_spec`` result ONLY in
    ``free_fields`` -- the tile is a knob axis, every other knob axis is
    crossed by the sweep but is not part of this catalog;
  * a cell lists the default spec first and the rest sorted by ``free_fields``;
  * more than ``max_kernels`` is an error, so a registry that grows fails here
    rather than at pack time.

A head configuration the registry serves with no spec is an error naming it: a
silently empty cell would ship a catalog that cannot serve a shape the manifest
promised.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]


class EnumerationError(RuntimeError):
    """The registry could not be reached, or disagrees with the manifest."""


def load_manifest(path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise EnumerationError("PyYAML is required to read the manifest") from exc
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise EnumerationError(f"manifest {path} must be a mapping")
    for key in (
        "engine",
        "kmd_fields",
        "specialization",
        "arch",
        "source",
        "builder",
        "enumerate",
    ):
        if key not in loaded:
            raise EnumerationError(f"manifest {path} is missing '{key}'")
    return loaded


def bind_provider(provider_root: str | None) -> None:
    """Put the rocKE library on ``sys.path``. ``ROCKE_PROVIDER_ROOT`` is the
    fallback; a relative root is repository-relative."""
    root = provider_root or os.environ.get("ROCKE_PROVIDER_ROOT")
    if not root:
        raise EnumerationError(
            "no provider_root in the manifest and no ROCKE_PROVIDER_ROOT set; the "
            "dispatcher registry cannot be imported without the rocKE library."
        )
    expanded = Path(os.path.expanduser(root))
    base = expanded if expanded.is_absolute() else _REPO / expanded
    for sub in ("rocke/library", "rocke/platform/python"):
        candidate = base / sub
        if not candidate.is_dir():
            raise EnumerationError(f"{candidate} does not exist; is {base} a provider root?")
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _plain(value):
    """A spec field as it is written into a descriptor."""
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _attention_slice(manifest: dict):
    """A function yielding the registry's specs for one attention head
    configuration."""
    try:
        from dispatch.attention import attention_candidates
        from dispatch.attention.common import AttentionMaskType, AttentionRequest
        from rocke.dispatch.tuning import walk
    except ImportError as exc:
        raise EnumerationError(f"cannot import the rocKE dispatcher registry: {exc}") from exc

    decl = manifest["enumerate"]
    algorithm = decl["algorithm"]
    candidate = next((c for c in attention_candidates() if c.algorithm == algorithm), None)
    if candidate is None:
        raise EnumerationError(
            f"no registry candidate has algorithm '{algorithm}'. Available: "
            f"{sorted(c.algorithm for c in attention_candidates())}."
        )
    shape = decl["request"]
    free = set(decl["free_fields"])

    def specs(dtype: str, head_size: int, hq: int, hk: int, causal: bool) -> list:
        request = AttentionRequest(
            arch=manifest["arch"],
            algorithm=algorithm,
            spec_id=decl["spec_id"],
            dtype=dtype,
            hdim_q=head_size,
            hdim_v=head_size,
            nhead_q=hq,
            nhead_k=hk,
            mask_type=(
                AttentionMaskType.TOP_LEFT_CAUSAL if causal else AttentionMaskType.NO_MASK
            ),
            batch=shape["batch"],
            seqlen_q=shape["seqlen_q"],
            seqlen_k=shape["seqlen_kv"],
        )
        base = candidate.select_spec(request)
        base = getattr(base, "kernel_spec", base)
        out = []
        with walk.sweep_level(decl["sweep_level"]):
            for sp in candidate.sweep_space(request):
                ks = getattr(sp, "kernel_spec", sp)
                if all(
                    getattr(ks, f.name) == getattr(base, f.name)
                    for f in dataclasses.fields(ks)
                    if f.name not in free
                ):
                    out.append(ks)
        # The default spec first, then the rest by the free fields: the catalog
        # order must not depend on how the sweep happens to generate its space.
        free_order = sorted(free)
        key = lambda ks: tuple(getattr(ks, n) for n in free_order)
        base_key = key(base)
        return sorted(out, key=lambda ks: (key(ks) != base_key, key(ks)))

    return specs


def enumerate_specs(manifest: dict) -> list[tuple[tuple, object]]:
    """``[(head_cell, spec), ...]`` for every cell the manifest names."""
    decl = manifest["enumerate"]
    specs = _attention_slice(manifest)
    out = []
    for head_size, heads in decl["head_configs"].items():
        for hq, hk in heads:
            for dtype in decl["dtypes"]:
                for causal in decl["causal"]:
                    cell = (dtype, int(head_size), hq, hk, bool(causal))
                    try:
                        found = specs(*cell)
                    except ValueError as exc:
                        raise EnumerationError(
                            f"the registry rejects {cell} (dtype, head_size, "
                            f"num_query_heads, num_kv_heads, causal), which the "
                            f"manifest's head_configs promises: {exc}"
                        ) from exc
                    if not found:
                        raise EnumerationError(
                            f"the registry yields no spec for {cell} (dtype, head_size, "
                            f"num_query_heads, num_kv_heads, causal), which the "
                            f"manifest's head_configs promises."
                        )
                    out.extend((cell, ks) for ks in found)
    limit = decl.get("max_kernels")
    if limit is not None and len(out) > limit:
        raise EnumerationError(
            f"the registry yields {len(out)} kernels, over the manifest's "
            f"max_kernels {limit}. Raise it deliberately: every kernel is a comgr "
            f"build in the package."
        )
    return out


def _metadata(spec: dict, manifest: dict) -> dict:
    spec_block = manifest["specialization"]
    vocabulary = spec_block.get("vocabulary") or {}
    metadata = {}
    for name in spec_block["metadata_fields"]:
        if name not in spec:
            raise EnumerationError(
                f"metadata field '{name}' is not a field of the projected spec "
                f"({sorted(spec)}); add it to enumerate.spec_fields."
            )
        value = spec[name]
        if isinstance(value, bool):
            value = int(value)
        mapping = vocabulary.get(name)
        if isinstance(mapping, dict) and isinstance(value, str):
            value = mapping.get(value, value)
        metadata[name] = value
    return metadata


def build_config(manifest: dict) -> dict:
    spec_fields = manifest["enumerate"].get("spec_fields")
    template = manifest.get("kernel_name_template")
    if not template:
        raise EnumerationError("manifest declares no kernel_name_template")
    kernels = []
    for _, ks in enumerate_specs(manifest):
        names = spec_fields or [f.name for f in dataclasses.fields(ks)]
        missing = [n for n in names if not hasattr(ks, n)]
        if missing:
            raise EnumerationError(
                f"enumerate.spec_fields names {missing}, which the registry spec "
                f"{type(ks).__name__} does not have."
            )
        spec = {n: _plain(getattr(ks, n)) for n in names}
        metadata = _metadata(spec, manifest)
        # Spec fields win over metadata, and a bool renders 1/0 so `c{causal}`
        # matches the shipped spelling rather than `cTrue`.
        fields = {
            **metadata,
            **{k: (int(v) if isinstance(v, bool) else v) for k, v in spec.items()},
        }
        try:
            name = template.format(**fields)
        except (KeyError, IndexError, ValueError) as exc:
            raise EnumerationError(
                f"kernel_name_template {template!r} cannot be rendered: {exc!r}. "
                f"Available: {sorted(fields)}."
            ) from exc
        kernels.append(
            {
                "name": name,
                "kernel_source": {
                    "kind": manifest.get("kernel_source_kind", "rocke"),
                    "source": manifest["source"],
                    "builder": manifest["builder"],
                    "spec": spec,
                },
                "metadata": metadata,
            }
        )
    slug = manifest.get("pack") or Path(manifest["authored_subpath"]).name
    return {
        "dialect": manifest.get("dialect", "packaged"),
        "authored_subpath": manifest["authored_subpath"],
        "engine": manifest["engine"],
        "kmd_fields": manifest["kmd_fields"],
        "specialization": manifest["specialization"],
        "kernel_source_kind": manifest.get("kernel_source_kind", "rocke"),
        "workspace_policy": manifest.get("workspace_policy", "none"),
        "packs": [{"name": slug, "arch": [manifest["arch"]], "kernels": kernels}],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate an engine's kernels from the rocKE dispatcher registry."
    )
    parser.add_argument("--manifest", required=True, help="Engine manifest (YAML).")
    parser.add_argument("--out", required=True, help="Write the generator config here.")
    args = parser.parse_args(argv)
    try:
        import yaml

        manifest = load_manifest(args.manifest)
        bind_provider(manifest.get("provider_root"))
        config = build_config(manifest)
    except EnumerationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(
        yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8"
    )
    print(f"{len(config['packs'][0]['kernels'])} kernels -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
