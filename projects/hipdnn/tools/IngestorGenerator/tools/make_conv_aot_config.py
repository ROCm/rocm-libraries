#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT forward-conv instances into the hipkernel:Gfx950ConvFwdAot config.

An instance is one rocKE ``KernelIdentity`` (``benchmarks/common/kernel_cache.py``):
a compiled configuration with no problem extents, which serves every problem its
capabilities admit. The input is a JSON list whose entries are either an identity
dict (``KernelIdentity.to_dict()``) or an object carrying one under ``"identity"``;
other keys of such an object are ignored here.

Every entry must be a gfx950 forward implicit-GEMM identity in fp16 or bf16 that the
packaged launch supports. Anything else is listed with its reason and the tool
exits non-zero, unless ``--skip-unsupported`` is given. Entries are de-duplicated by
``instance_id`` (``KernelIdentity.stable_hash()``), which is the key that ties a
packaged kernel back to the identity it was built from.

Then run ``generate.py --config <output>`` and ``--stabilize-ids`` on the emitted
bundle: the generator mints random kernel UUIDs, and a recorded autotune winner is
keyed by kernel UUID, so this tool re-derives each kernel's UUID from its
``instance_id`` to keep it stable across regenerations::

    python tools/make_conv_aot_config.py --rocke-root "$ROCKE" \\
        --instances instances.json --output configs/gfx950_conv_fwd_aot.yaml
    python generate.py --config configs/gfx950_conv_fwd_aot.yaml \\
        --output-dir "$GENERATED" --force
    python tools/make_conv_aot_config.py --stabilize-ids "$GENERATED"/<...>/gfx950_conv_fwd_aot.kdp.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

#: Namespace for kernel UUIDs derived from an instance_id (uuid5).
INSTANCE_UUID_NAMESPACE = uuid.UUID("6b0d3b8e-6a58-5d0f-9f43-3c6f5f2a9c11")

#: Identity fields the packaged forward launch takes from the identity.
_CARRIED = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "wave_size",
    "pipeline",
    "epilogue",
    "vector_size_a",
    "vector_size_b",
    "vector_size_c",
    "unroll_k",
    "async_dma",
    "async_chunk_a",
    "async_chunk_b",
    "grouped",
)
#: Identity fields that are not carried; each must hold its KernelIdentity default
#: (they select emission the packaged launch or matcher does not model).
_CONSTANT = (
    "chiplet_swizzle",
    "lds_layout",
    "lds_k_pad",
    "lds_k_outer",
    "waves_per_eu",
    "acc_epilogue",
    "split_k",
    "two_stage",
    "ws_replicas",
    "group_merge",
    "num_load_waves",
    "cshuffle_no_alias",
    "is_3d",
    "is_pointwise",
    "filter_h",
    "filter_w",
    "filter_d",
    "stride_h",
    "stride_w",
    "dilation_h",
    "dilation_w",
    "pad_h",
    "pad_w",
    "cpg",
    "kpg",
    "block_h",
    "max_sub_gemms",
    "knobs",
)
#: Spec field order of Gfx950ConvFwdAotSpec; also the descriptor spec order.
SPEC_ORDER = ("dtype",) + _CARRIED + ("instance_id", "layout")
_STRING_FIELDS = ("dtype", "pipeline", "epilogue", "instance_id", "layout")
_INT_FIELDS = tuple(f for f in SPEC_ORDER if f not in _STRING_FIELDS)


def _bind_rocke(rocke_root: Path) -> None:
    for sub in ("platform/python", "library"):
        path = str((rocke_root / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def instance_spec(identity_dict: dict):
    """``(Gfx950ConvFwdAotSpec, "")`` for a supported identity, else ``(None, reason)``."""
    from benchmarks.common.kernel_cache import KernelIdentity
    from builders.common.convolution_forward import (
        Gfx950ConvFwdAotSpec,
        supports_gfx950_conv_fwd_aot,
    )

    identity = KernelIdentity.from_dict(identity_dict)
    defaults = KernelIdentity(
        arch=identity.arch,
        direction=identity.direction,
        algorithm=identity.algorithm,
        dtype_a=identity.dtype_a,
        dtype_b=identity.dtype_b,
        dtype_d=identity.dtype_d,
        tile_m=0,
        tile_n=0,
        tile_k=0,
        warp_m=0,
        warp_n=0,
        warp_tile_m=0,
        warp_tile_n=0,
        warp_tile_k=0,
        pipeline="",
        epilogue="",
        wave_size=0,
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        llvm_flavor=identity.llvm_flavor,
    )
    if (identity.arch, identity.direction, identity.algorithm) != (
        "gfx950",
        "fwd",
        "implicit_gemm",
    ):
        return None, (
            f"not a gfx950 forward implicit-GEMM identity "
            f"({identity.arch}, {identity.direction}, {identity.algorithm})"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
    for name in _CONSTANT:
        if getattr(identity, name) != getattr(defaults, name):
            return None, (
                f"{name}={getattr(identity, name)!r} is not packaged "
                f"(only {getattr(defaults, name)!r})"
            )
    values = {name: getattr(identity, name) for name in _CARRIED}
    for name in ("unroll_k", "async_dma", "grouped"):
        values[name] = int(values[name])
    spec = Gfx950ConvFwdAotSpec(
        dtype=identity.dtype_a, instance_id=identity.stable_hash(), **values
    )
    ok, reason = supports_gfx950_conv_fwd_aot(spec)
    if not ok:
        return None, reason
    return spec, ""


def build_config(specs: list) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated)."""
    kmd = [
        {"name": name, "type": "int" if name in _INT_FIELDS else "string"}
        for name in SPEC_ORDER
    ]
    metadata = list(SPEC_ORDER)
    shapes = []
    for spec in specs:
        shape = {name: getattr(spec, name) for name in SPEC_ORDER}
        shape["knobs"] = "instance"
        shapes.append(shape)
    return {
        "dialect": "packaged",
        "authored_subpath": "rocKE/gfx950_conv_fwd_aot",
        "engine": {
            "name": "hipkernel:Gfx950ConvFwdAot",
            "sdk_version": "1.0.0",
            "knobs": ["tile_k"],
            "heuristic": "native",
        },
        "kmd_fields": kmd,
        "kernel_source_kind": "rocke",
        "workspace_policy": "none",
        "graph_match": {"shape": "shared_shape", "discriminator": "none"},
        "specialization": {
            "metadata_fields": metadata,
            "matcher_only_fields": [],
            "bindings": {name: {"field": name} for name in metadata},
        },
        "packs": [
            {
                "name": "gfx950_conv_fwd_aot",
                "arch": ["gfx950"],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": "builders/common/convolution_forward.py",
                    "builder": "build_gfx950_conv_fwd_aot",
                },
                "variants": [
                    {
                        "name": "gfx950_conv_fwd_aot_{dtype}_{instance_id}",
                        "metadata": metadata,
                        "spec_order": list(SPEC_ORDER),
                        "spec_defaults": {},
                        "knob_sets": {"instance": [{}]},
                        "shapes": shapes,
                    }
                ],
            }
        ],
    }


def stabilize_ids(kdp_path: Path) -> int:
    """Re-derive every kernel UUID from its instance_id; returns the count."""
    document = json.loads(kdp_path.read_text())
    text = kdp_path.read_text()
    replaced = 0
    for kernel in _kernels(document):
        instance_id = kernel["metadata"]["instance_id"]
        new_id = str(uuid.uuid5(INSTANCE_UUID_NAMESPACE, instance_id))
        old_id = kernel["id"]
        if text.count(old_id) != 1:
            raise SystemExit(
                f"kernel id {old_id} appears {text.count(old_id)} times in "
                f"{kdp_path}; refusing to rewrite a referenced id"
            )
        text = text.replace(old_id, new_id)
        replaced += 1
    kdp_path.write_text(text)
    return replaced


def _kernels(node):
    if isinstance(node, dict):
        if isinstance(node.get("kernel_source"), dict) and "metadata" in node:
            yield node
            return
        for value in node.values():
            yield from _kernels(value)
    elif isinstance(node, list):
        for value in node:
            yield from _kernels(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root", type=Path, help="rocKE tree (platform/, library/)."
    )
    parser.add_argument(
        "--instances", type=Path, help="JSON list of KernelIdentity dicts."
    )
    parser.add_argument("--output", type=Path, help="Generator config to write.")
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Drop unsupported identities (each is still reported) instead of failing.",
    )
    parser.add_argument(
        "--stabilize-ids",
        type=Path,
        metavar="KDP",
        help="Rewrite a generated .kdp.json's kernel UUIDs from instance_id, then exit.",
    )
    args = parser.parse_args(argv)

    if args.stabilize_ids:
        print(f"stabilized {stabilize_ids(args.stabilize_ids)} kernel id(s)")
        return 0
    if not (args.rocke_root and args.instances and args.output):
        parser.error("--rocke-root, --instances and --output are required")
    _bind_rocke(args.rocke_root)
    import yaml

    entries = json.loads(args.instances.read_text())
    if not isinstance(entries, list):
        parser.error("--instances must hold a JSON list")
    specs, seen, rejected = [], set(), []
    for index, entry in enumerate(entries):
        identity = entry.get("identity", entry) if isinstance(entry, dict) else None
        if not isinstance(identity, dict):
            rejected.append((index, "entry is not an identity dict"))
            continue
        spec, reason = instance_spec(identity)
        if spec is None:
            rejected.append((index, reason))
        elif spec.instance_id not in seen:
            seen.add(spec.instance_id)
            specs.append(spec)
    specs.sort(key=lambda s: (s.dtype, s.instance_id))
    for index, reason in rejected:
        print(f"unsupported instance [{index}]: {reason}", file=sys.stderr)
    if rejected and not args.skip_unsupported:
        print(
            f"{len(rejected)} unsupported instance(s); nothing written", file=sys.stderr
        )
        return 1
    if not specs:
        print("no supported instances; nothing written", file=sys.stderr)
        return 1
    header = (
        "# Generated by tools/make_conv_aot_config.py from "
        f"{args.instances.name}; do not edit by hand.\n"
    )
    args.output.write_text(
        header + yaml.safe_dump(build_config(specs), sort_keys=False, width=1000)
    )
    print(
        f"{len(specs)} instance(s) written to {args.output} "
        f"({len(entries) - len(rejected) - len(specs)} duplicate(s), {len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
