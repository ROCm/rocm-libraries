#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT forward-conv instances into a hipkernel:Gfx950ConvFwdAot config.

An instance is one rocKE ``KernelIdentity`` (``benchmarks/common/kernel_cache.py``):
a compiled configuration with no problem extents, which serves every problem its
capabilities admit. The input is a JSON list whose entries are either an identity
dict (``KernelIdentity.to_dict()``) or an object carrying one under ``"identity"``;
other keys of such an object are ignored here.

Every entry must be a gfx950 forward identity in fp16 or bf16 that the packaged
launch supports: an implicit-GEMM identity of the AOT sweep, or a forward identity of
the direct sweep (``benchmarks/common/direct_kernel_sweep.py``), any of its algorithms
(``PACKAGED_DIRECT_ALGORITHMS``). Anything else is listed with its reason and the tool
exits non-zero, unless ``--skip-unsupported`` is given. Entries
are de-duplicated by ``instance_id`` (``KernelIdentity.stable_hash()``), which is the
key that ties a packaged kernel back to the identity it was built from.

The shipped catalog is rocKE's full forward sweep, one list per dtype
(``configs/gfx950_conv_fwd_aot_<dtype>.instances.json``), written by
``tools/enumerate_conv_aot_instances.py``; ``--instances`` takes them all as one
catalog. ``--engine-name`` emits the same instances as another engine over the
shipped engine's native code: the test-only seed is
``configs/gfx950_conv_fwd_aot_seed.instances.json`` as
``hipkernel:Gfx950ConvFwdAotSeed``. The config is an intermediate, regenerated from
the lists on every run; only the seed's small one is kept, as a worked example.

Then run ``generate.py --config <output>`` and ``--stabilize-ids`` on the emitted
bundle: the generator mints random kernel UUIDs, and a recorded autotune winner is
keyed by kernel UUID, so this tool re-derives each kernel's UUID from its
``instance_id`` (and engine) to keep it stable across regenerations::

    python tools/make_conv_aot_config.py --rocke-root "$ROCKE" \\
        --instances configs/gfx950_conv_fwd_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/aot.yaml
    python generate.py --config "$WORK"/aot.yaml --output-dir "$GENERATED" --force
    python tools/make_conv_aot_config.py --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The config splits the catalog into packs, so KDPs, by ``--sharding``
(:data:`SHARDINGS`): by default one per dtype, pipeline and K loop, plus one per
dtype for the direct instances. Each is a shard of the one catalog, which keeps
every emitted file well below the size a source host accepts.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import re
import sys
import uuid
from pathlib import Path

#: Namespace for kernel UUIDs derived from an instance_id (uuid5).
INSTANCE_UUID_NAMESPACE = uuid.UUID("6b0d3b8e-6a58-5d0f-9f43-3c6f5f2a9c11")

#: The shipped engine. Its native symbols are the ones every bundle this tool emits
#: names; another engine name (a test-only subset of the catalog) borrows them.
PRODUCT_ENGINE = "hipkernel:Gfx950ConvFwdAot"

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
#: The direct fields Gfx950ConvFwdAotSpec appends, which every direct instance
#: states. Their KMD default_value is the dataclass default (builder_default), so
#: implicit-GEMM descriptors omit them.
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
#: The block knobs of the tiled, col, grouped and non-grouped direct variants, after
#: DIRECT_FIELDS, with their builder_default. A direct instance states those of its
#: variant only (DIRECT_VARIANT_FIELDS), so std and spatial descriptors are as they
#: were before these existed.
DIRECT_KNOB_FIELDS = {
    "block_h": 0,
    "unroll_rows": 0,
    "block_q": 0,
    "block_groups": 0,
    "double_buffer": 0,
    "cpg": 0,
    "kpg": 0,
    "tile_h": 0,
    "tile_w": 0,
    "tile_k_out": 0,
    "ck": 0,
    "waves_m": 0,
    "waves_n": 0,
    "atom": "none",
    "swizzle_wgm": 0,
    "iglp": -1,
    "waves_per_eu": 0,
}
#: Spec field order of a std or spatial direct instance.
DIRECT_SPEC_ORDER = SPEC_ORDER + tuple(DIRECT_FIELDS)
#: The DIRECT_KNOB_FIELDS each direct variant states, in their KMD order.
DIRECT_VARIANT_FIELDS = {
    "std": (),
    "spatial": (),
    "tiled": ("block_h", "unroll_rows"),
    "col": ("block_h",),
    "grouped": ("block_h", "block_q", "block_groups", "double_buffer", "cpg", "kpg"),
    "nongrouped": (
        "tile_h",
        "tile_w",
        "tile_k_out",
        "ck",
        "waves_m",
        "waves_n",
        "atom",
        "swizzle_wgm",
        "iglp",
        "waves_per_eu",
    ),
}
_STRING_FIELDS = (
    "dtype",
    "pipeline",
    "epilogue",
    "instance_id",
    "layout",
    "direct_variant",
    "atom",
)
_INT_FIELDS = tuple(
    f for f in DIRECT_SPEC_ORDER + tuple(DIRECT_KNOB_FIELDS) if f not in _STRING_FIELDS
)
#: Direct-sweep algorithm -> the instance's direct_variant, for the algorithms the
#: exact-shape engine's direct launch runs.
DIRECT_ALGORITHMS = {"direct_depthwise": "std", "direct_depthwise_spatial": "spatial"}
#: Every forward direct-sweep algorithm the packaged AOT launch runs, and its
#: direct_variant: AOT_DIRECT_ALGORITHMS of the rocKE adapter.
#: tools/enumerate_conv_aot_instances.py ships every identity of these.
PACKAGED_DIRECT_ALGORITHMS = {
    **DIRECT_ALGORITHMS,
    "direct_depthwise_tiled": "tiled",
    "direct_depthwise_col": "col",
    "direct_grouped": "grouped",
    "direct_nongrouped": "nongrouped",
}
#: The generator variant (kernel name) of each direct variant. std and spatial keep
#: the name they had when they were the only direct instances.
_DIRECT_VARIANT_NAMES = {
    "std": "gfx950_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "spatial": "gfx950_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "tiled": "gfx950_conv_fwd_aot_dwtiled_{dtype}_{instance_id}",
    "col": "gfx950_conv_fwd_aot_dwcol_{dtype}_{instance_id}",
    "grouped": "gfx950_conv_fwd_aot_grouped_{dtype}_{instance_id}",
    "nongrouped": "gfx950_conv_fwd_aot_nongrouped_{dtype}_{instance_id}",
}


def _bind_rocke(rocke_root: Path) -> None:
    for sub in ("platform/python", "library"):
        path = str((rocke_root / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def spec_order(spec) -> tuple:
    """The descriptor spec order of ``spec``'s family and direct variant."""
    if not spec.is_direct:
        return SPEC_ORDER
    return DIRECT_SPEC_ORDER + DIRECT_VARIANT_FIELDS[spec.direct_variant]


def instance_spec(identity_dict: dict):
    """``(Gfx950ConvFwdAotSpec, "")`` for a supported identity, else ``(None, reason)``."""
    from benchmarks.common.kernel_cache import KernelIdentity

    identity = KernelIdentity.from_dict(identity_dict)
    if identity.direction == "direct_fwd":
        return _direct_instance_spec(identity)
    return _implicit_instance_spec(identity)


def instance_specs(identities: list, jobs: int = 1) -> list:
    """:func:`instance_spec` of each identity, in order, over ``jobs`` forked worker
    processes (serially where ``fork`` is not available)."""
    if (
        jobs <= 1
        or len(identities) < 1000
        or "fork" not in multiprocessing.get_all_start_methods()
    ):
        return [instance_spec(identity) for identity in identities]
    chunk = max(1, len(identities) // (jobs * 8))
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(instance_spec, identities, chunksize=chunk)


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
    """A forward direct identity: the direct sweep must be able to build it, and the
    packaged AOT launch must run its algorithm."""
    from benchmarks.common.direct_kernel_sweep import direct_job_for_identity
    from builders.common.convolution_forward import (
        Gfx950ConvFwdAotSpec,
        aot_direct_instance_fields,
    )

    if identity.arch != "gfx950":
        return None, f"not a gfx950 forward direct identity ({identity.arch})"
    if identity.algorithm not in PACKAGED_DIRECT_ALGORITHMS:
        return None, (
            f"direct algorithm {identity.algorithm!r} has no packaged launch "
            f"(supported: {sorted(PACKAGED_DIRECT_ALGORITHMS)})"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
    # Every other field must be what the direct grid writes for these caps and
    # knobs; the sweep's own round trip refuses anything else, after asking the
    # kernel's validator.
    try:
        direct_job_for_identity(identity)
        fields = aot_direct_instance_fields(identity)
    except (LookupError, TypeError, ValueError) as exc:
        return None, str(exc)
    spec = Gfx950ConvFwdAotSpec(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        **_carried_values(identity),
        **fields,
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


def engine_slug(engine_name: str) -> str:
    """The generator's slug for ``engine_name`` (``hipkernel:Gfx950ConvFwdAot`` ->
    ``gfx950_conv_fwd_aot``)."""
    local_name = engine_name.split(":", 1)[1]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", local_name).lower()


#: How :func:`build_config` splits a catalog into packs, one KDP each.
#: ``dtype``: one pack per dtype. ``k-loop``: one pack per dtype, implicit-GEMM
#: pipeline and K loop (``plain``, ``unroll`` or ``async``), and one per dtype for
#: the direct instances, which keeps every KDP of rocKE's full sweep well below
#: the size a source host accepts.
SHARDINGS = ("dtype", "k-loop")


def shard_name(spec, sharding: str = "dtype") -> str:
    """The pack ``spec`` belongs to under ``sharding`` (see :data:`SHARDINGS`)."""
    if sharding == "dtype":
        return spec.dtype
    if sharding != "k-loop":
        raise ValueError(f"unknown sharding {sharding!r}; expected one of {SHARDINGS}")
    if spec.is_direct:
        return f"{spec.dtype}_direct"
    k_loop = "async" if spec.async_dma else ("unroll" if spec.unroll_k else "plain")
    return f"{spec.dtype}_{spec.pipeline}_{k_loop}"


def build_config(
    specs: list,
    engine_name: str = PRODUCT_ENGINE,
    authored_subpath: str = "",
    sharding: str = "dtype",
) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated).

    ``engine_name`` other than :data:`PRODUCT_ENGINE` is a second engine over the
    same native code: its descriptors name the shipped engine's symbols
    (``native_symbols_from``). ``authored_subpath`` defaults to
    ``rocKE/<engine slug>``.

    The catalog is split into packs by ``sharding`` (:func:`shard_name`), each
    its own KDP, so no file has to hold the whole catalog. The packs declare no
    discriminator: they are shards of one catalog, and the engine's graph_match
    and the shared kernel matcher admit every one of them. Within a pack,
    implicit-GEMM instances keep their own variant and leave the direct fields to
    their builder_default. Direct instances are one variant per direct kernel
    (std and spatial share one), each stating DIRECT_FIELDS and its own block
    knobs (DIRECT_VARIANT_FIELDS) and leaving the others to theirs.
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
        for name, default in {**DIRECT_FIELDS, **DIRECT_KNOB_FIELDS}.items()
    ]
    metadata = list(DIRECT_SPEC_ORDER + tuple(DIRECT_KNOB_FIELDS))
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        implicit = [s for s in shards[shard] if not s.is_direct]
        if implicit:
            variants.append(
                _variant(
                    "gfx950_conv_fwd_aot_{dtype}_{instance_id}", SPEC_ORDER, implicit
                )
            )
        # std and spatial share one variant, as before the other direct variants.
        for group in (
            ("std", "spatial"),
            ("tiled",),
            ("col",),
            ("grouped",),
            ("nongrouped",),
        ):
            members = [
                s for s in shards[shard] if s.is_direct and s.direct_variant in group
            ]
            if members:
                variants.append(
                    _variant(
                        _DIRECT_VARIANT_NAMES[group[0]],
                        spec_order(members[0]),
                        members,
                    )
                )
        packs.append(
            {
                "name": shard,
                "arch": ["gfx950"],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": "builders/common/convolution_forward.py",
                    "builder": "build_gfx950_conv_fwd_aot",
                },
                "variants": variants,
            }
        )
    engine = {
        "name": engine_name,
        "sdk_version": "1.0.0",
        "knobs": ["tile_k", "kernel_family"],
        "heuristic": "native",
    }
    if engine_name != PRODUCT_ENGINE:
        engine["native_symbols_from"] = PRODUCT_ENGINE
    return {
        "dialect": "packaged",
        "authored_subpath": authored_subpath or f"rocKE/{engine_slug(engine_name)}",
        "engine": engine,
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


def stabilize_ids(kdp_path: Path, engine_name: str = PRODUCT_ENGINE) -> int:
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
        new_ids[old_id] = kernel_uuid(kernel["metadata"]["instance_id"], engine_name)
    kdp_path.write_text(
        _UUID.sub(lambda match: new_ids.get(match.group(0), match.group(0)), text)
    )
    return len(kernels)


def kernel_uuid(instance_id: str, engine_name: str = PRODUCT_ENGINE) -> str:
    """The kernel UUID an instance is shipped under by ``engine_name``.

    The shipped engine's ids are uuid5 of the instance_id. Any other engine derives
    them under a namespace of its own name, so an instance it shares with the shipped
    catalog never carries the same kernel id into one process twice.
    """
    namespace = INSTANCE_UUID_NAMESPACE
    if engine_name != PRODUCT_ENGINE:
        namespace = uuid.uuid5(INSTANCE_UUID_NAMESPACE, engine_name)
    return str(uuid.uuid5(namespace, instance_id))


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
        "--instances",
        type=Path,
        nargs="+",
        help="JSON lists of KernelIdentity dicts, read as one catalog.",
    )
    parser.add_argument("--output", type=Path, help="Generator config to write.")
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Drop unsupported identities (each is still reported) instead of failing.",
    )
    parser.add_argument(
        "--engine-name",
        default=PRODUCT_ENGINE,
        help="Engine the bundle declares, and whose kernel ids --stabilize-ids "
        "derives (default: %(default)s). Another name borrows that engine's native "
        "symbols.",
    )
    parser.add_argument(
        "--authored-subpath",
        default="",
        help="The bundle's authored subpath (default: rocKE/<engine slug>).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Worker processes that convert the identities (default: %(default)s).",
    )
    parser.add_argument(
        "--sharding",
        choices=SHARDINGS,
        default="k-loop",
        help="How the catalog is split into packs, one KDP each (default: "
        "%(default)s; the test-only seed uses dtype).",
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
            count = stabilize_ids(kdp, args.engine_name)
            print(f"{kdp}: stabilized {count} kernel id(s)")
        return 0
    if not (args.rocke_root and args.instances and args.output):
        parser.error("--rocke-root, --instances and --output are required")
    _bind_rocke(args.rocke_root)

    entries = []
    for path in args.instances:
        listed = json.loads(path.read_text())
        if not isinstance(listed, list):
            parser.error(f"--instances {path} must hold a JSON list")
        entries += listed
    specs, seen, rejected = [], set(), []
    identities = {}
    for index, entry in enumerate(entries):
        identity = entry.get("identity", entry) if isinstance(entry, dict) else None
        if isinstance(identity, dict):
            identities[index] = identity
        else:
            rejected.append((index, "entry is not an identity dict"))
    converted = instance_specs(list(identities.values()), args.jobs)
    for index, (spec, reason) in zip(identities, converted):
        if spec is None:
            rejected.append((index, reason))
        elif spec.instance_id not in seen:
            seen.add(spec.instance_id)
            specs.append(spec)
    specs.sort(key=lambda s: (s.dtype, s.instance_id))
    rejected.sort()
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
    sources = ", ".join(path.name for path in args.instances)
    header = (
        f"# Generated by tools/make_conv_aot_config.py from {sources}; "
        "do not edit by hand.\n"
    )
    config = build_config(specs, args.engine_name, args.authored_subpath, args.sharding)
    args.output.write_text(header + dump_config(config))
    print(
        f"{len(specs)} instance(s) written to {args.output} "
        f"({len(entries) - len(rejected) - len(specs)} duplicate(s), {len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
