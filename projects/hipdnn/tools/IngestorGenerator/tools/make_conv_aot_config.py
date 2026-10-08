#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT forward-conv instances into the hipkernel:Gfx950ConvFwdAot config.

An instance is one rocKE ``KernelIdentity`` (``benchmarks/common/kernel_cache.py``):
a compiled configuration with no problem extents, which serves every problem its
capabilities admit. The input is a JSON list whose entries are either an identity
dict (``KernelIdentity.to_dict()``) or an object carrying one under ``"identity"``;
other keys of such an object are ignored here.

Every entry must be a gfx950 forward identity in fp16 or bf16 that the packaged
launch supports: an implicit-GEMM identity of the AOT sweep, or a ``direct_depthwise``
(std) or ``direct_depthwise_spatial`` identity of the direct sweep
(``benchmarks/common/direct_kernel_sweep.py``). Anything else is listed with its
reason and the tool exits non-zero, unless ``--skip-unsupported`` is given. Entries are de-duplicated by
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
    python tools/make_conv_aot_config.py --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The config holds one pack, so one KDP, per dtype: a shard of the one catalog, which
keeps every emitted file well below the size a source host accepts.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import uuid
from pathlib import Path

#: Namespace for kernel UUIDs derived from an instance_id (uuid5).
INSTANCE_UUID_NAMESPACE = uuid.UUID("6b0d3b8e-6a58-5d0f-9f43-3c6f5f2a9c11")

#: A UUID as ``str(uuid.UUID)`` spells it: the form of every generated kernel id.
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

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
#: Spec field order of an implicit-GEMM Gfx950ConvFwdAotSpec; also its descriptor
#: spec order.
SPEC_ORDER = ("dtype",) + _CARRIED + ("instance_id", "layout")
#: The direct fields Gfx950ConvFwdAotSpec appends. Their KMD default_value is the
#: dataclass default (builder_default), so implicit-GEMM descriptors omit them.
DIRECT_FIELDS = {
    "kernel_family": 0,
    "direct_variant": "none",
    "filter_h": 0,
    "filter_w": 0,
    "stride_h": 0,
    "stride_w": 0,
    "pad_h": 0,
    "pad_w": 0,
    "dilation_h": 0,
    "dilation_w": 0,
    "block_w": 0,
    "block_waves": 0,
}
#: Spec field order of a direct instance.
DIRECT_SPEC_ORDER = SPEC_ORDER + tuple(DIRECT_FIELDS)
_STRING_FIELDS = (
    "dtype",
    "pipeline",
    "epilogue",
    "instance_id",
    "layout",
    "direct_variant",
)
_INT_FIELDS = tuple(f for f in DIRECT_SPEC_ORDER if f not in _STRING_FIELDS)
#: Direct-sweep algorithm -> the instance's direct_variant, for the algorithms the
#: exact-shape engine's direct launch runs.
DIRECT_ALGORITHMS = {"direct_depthwise": "std", "direct_depthwise_spatial": "spatial"}
#: The knob keys each direct algorithm's identity carries.
_DIRECT_KNOBS = {
    "direct_depthwise": ("block_w", "block_waves"),
    "direct_depthwise_spatial": ("block_waves",),
}


def _bind_rocke(rocke_root: Path) -> None:
    for sub in ("platform/python", "library"):
        path = str((rocke_root / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def spec_order(spec) -> tuple:
    """The descriptor spec order of ``spec``'s family."""
    return DIRECT_SPEC_ORDER if spec.is_direct else SPEC_ORDER


def instance_spec(identity_dict: dict):
    """``(Gfx950ConvFwdAotSpec, "")`` for a supported identity, else ``(None, reason)``."""
    from benchmarks.common.kernel_cache import KernelIdentity

    identity = KernelIdentity.from_dict(identity_dict)
    if identity.direction == "direct_fwd":
        return _direct_instance_spec(identity)
    return _implicit_instance_spec(identity)


def _carried_values(identity) -> dict:
    values = {name: getattr(identity, name) for name in _CARRIED}
    for name in ("unroll_k", "async_dma", "grouped"):
        values[name] = int(values[name])
    return values


def _checked(spec):
    from builders.common.convolution_forward import supports_gfx950_conv_fwd_aot

    ok, reason = supports_gfx950_conv_fwd_aot(spec)
    return (spec, "") if ok else (None, reason)


def _implicit_instance_spec(identity):
    from benchmarks.common.kernel_cache import KernelIdentity
    from builders.common.convolution_forward import Gfx950ConvFwdAotSpec

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
    spec = Gfx950ConvFwdAotSpec(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        **_carried_values(identity),
    )
    return _checked(spec)


def _direct_instance_spec(identity):
    """A forward direct identity: the direct sweep must be able to build it, and
    the exact-shape engine's direct launch must run its kernel unchanged."""
    from benchmarks.common.direct_kernel_sweep import direct_job_for_identity
    from builders.common.convolution_forward import Gfx950ConvFwdAotSpec

    if identity.arch != "gfx950":
        return None, f"not a gfx950 forward direct identity ({identity.arch})"
    variant = DIRECT_ALGORITHMS.get(identity.algorithm)
    if variant is None:
        return None, (
            f"direct algorithm {identity.algorithm!r} has no packaged launch "
            f"(supported: {sorted(DIRECT_ALGORITHMS)})"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
    if (identity.cpg, identity.kpg) != (1, 1):
        return None, f"cpg={identity.cpg}, kpg={identity.kpg} is not depthwise"
    # Every other field must be what the direct grid writes for these caps and
    # knobs; the sweep's own round trip refuses anything else, after asking the
    # kernel's validator.
    try:
        direct_job_for_identity(identity)
    except (LookupError, TypeError, ValueError) as exc:
        return None, str(exc)
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if tuple(sorted(knobs)) != _DIRECT_KNOBS[identity.algorithm]:
        return None, (
            f"knobs {sorted(knobs)} are not {list(_DIRECT_KNOBS[identity.algorithm])}"
        )
    spec = Gfx950ConvFwdAotSpec(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        **_carried_values(identity),
        kernel_family=1,
        direct_variant=variant,
        filter_h=identity.filter_h,
        filter_w=identity.filter_w,
        stride_h=identity.stride_h,
        stride_w=identity.stride_w,
        pad_h=identity.pad_h,
        pad_w=identity.pad_w,
        dilation_h=identity.dilation_h,
        dilation_w=identity.dilation_w,
        block_w=int(knobs.get("block_w", 0)),
        block_waves=int(knobs["block_waves"]),
    )
    return _checked(spec)


def _variant(name: str, order: tuple, specs: list) -> dict:
    shapes = []
    for spec in specs:
        shape = {field: getattr(spec, field) for field in order}
        shape["knobs"] = "instance"
        shapes.append(shape)
    return {
        "name": name,
        "metadata": list(order),
        "spec_order": list(order),
        "spec_defaults": {},
        "knob_sets": {"instance": [{}]},
        "shapes": shapes,
    }


def build_config(specs: list) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated).

    One pack per dtype, each its own KDP, so no file has to hold the whole catalog.
    The packs declare no discriminator: they are shards of one catalog, and the
    engine's graph_match and the shared kernel matcher admit every one of them.
    Within a pack, implicit-GEMM instances keep their own variant and leave the
    direct fields to their builder_default; direct instances are a second variant
    stating them.
    """
    kmd = [
        {"name": name, "type": "int" if name in _INT_FIELDS else "string"}
        for name in SPEC_ORDER
    ] + [
        {
            "name": name,
            "type": "int" if name in _INT_FIELDS else "string",
            "default_value": default,
            "builder_default": True,
        }
        for name, default in DIRECT_FIELDS.items()
    ]
    metadata = list(DIRECT_SPEC_ORDER)
    packs = []
    for dtype in sorted({s.dtype for s in specs}):
        variants = []
        for name, order, family in (
            ("gfx950_conv_fwd_aot_{dtype}_{instance_id}", SPEC_ORDER, False),
            ("gfx950_conv_fwd_aot_dw_{dtype}_{instance_id}", DIRECT_SPEC_ORDER, True),
        ):
            members = [s for s in specs if s.dtype == dtype and s.is_direct == family]
            if members:
                variants.append(_variant(name, order, members))
        packs.append(
            {
                "name": dtype,
                "arch": ["gfx950"],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": "builders/common/convolution_forward.py",
                    "builder": "build_gfx950_conv_fwd_aot",
                },
                "variants": variants,
            }
        )
    return {
        "dialect": "packaged",
        "authored_subpath": "rocKE/gfx950_conv_fwd_aot",
        "engine": {
            "name": "hipkernel:Gfx950ConvFwdAot",
            "sdk_version": "1.0.0",
            "knobs": ["tile_k", "kernel_family"],
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
        "packs": packs,
    }


class _FlowRow(dict):
    """A mapping :func:`dump_config` writes on one line (a YAML flow mapping)."""


def dump_config(config: dict, dumper=None) -> str:
    """``config`` as YAML, each ``shapes`` row a one-line flow mapping.

    ``dumper`` defaults to libyaml's ``CSafeDumper`` when PyYAML was built with
    it, else to the pure-Python ``SafeDumper``; both emit the same text.
    """
    import yaml

    if dumper is None:
        dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

    class _Dumper(dumper):
        pass

    _Dumper.add_representer(
        _FlowRow,
        lambda representer, row: representer.represent_mapping(
            "tag:yaml.org,2002:map", row, flow_style=True
        ),
    )
    packs = [
        dict(
            pack,
            variants=[
                dict(variant, shapes=[_FlowRow(row) for row in variant["shapes"]])
                for variant in pack["variants"]
            ],
        )
        for pack in config["packs"]
    ]
    return yaml.dump(
        dict(config, packs=packs), Dumper=_Dumper, sort_keys=False, width=1000
    )


def stabilize_ids(kdp_path: Path) -> int:
    """Re-derive every kernel UUID of one KDP from its instance_id; returns the count.

    A kernel id found anywhere else in the file is referenced, and is refused.
    Every UUID in the text is counted in one pass and all kernel ids are then
    substituted in another, so the cost stays linear in the size of the file.
    """
    text = kdp_path.read_text()
    kernels = list(_kernels(json.loads(text)))
    counts = collections.Counter(_UUID.findall(text))
    new_ids = {}
    for kernel in kernels:
        old_id = kernel["id"]
        if not _UUID.fullmatch(old_id):
            raise SystemExit(f"kernel id {old_id!r} in {kdp_path} is not a UUID")
        if counts[old_id] != 1:
            raise SystemExit(
                f"kernel id {old_id} appears {counts[old_id]} times in "
                f"{kdp_path}; refusing to rewrite a referenced id"
            )
        new_ids[old_id] = kernel_uuid(kernel["metadata"]["instance_id"])
    kdp_path.write_text(
        _UUID.sub(lambda match: new_ids.get(match.group(0), match.group(0)), text)
    )
    return len(kernels)


def kernel_uuid(instance_id: str) -> str:
    """The kernel UUID an instance is shipped under."""
    return str(uuid.uuid5(INSTANCE_UUID_NAMESPACE, instance_id))


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
        nargs="+",
        metavar="KDP",
        help="Rewrite each generated .kdp.json's kernel UUIDs from instance_id, then exit.",
    )
    args = parser.parse_args(argv)

    if args.stabilize_ids:
        for kdp in args.stabilize_ids:
            print(f"{kdp}: stabilized {stabilize_ids(kdp)} kernel id(s)")
        return 0
    if not (args.rocke_root and args.instances and args.output):
        parser.error("--rocke-root, --instances and --output are required")
    _bind_rocke(args.rocke_root)

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
    args.output.write_text(header + dump_config(build_config(specs)))
    print(
        f"{len(specs)} instance(s) written to {args.output} "
        f"({len(entries) - len(rejected) - len(specs)} duplicate(s), {len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
