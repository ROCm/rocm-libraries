#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT backward-data conv instances into a hipkernel:Gfx950ConvDgradAot config.

The dgrad sibling of ``make_conv_aot_config.py``: same input (JSON lists of rocKE
``KernelIdentity`` dicts, or objects carrying one under ``"identity"``), same output
(a packaged-dialect generator config), same ``--stabilize-ids`` step. It reuses that
tool's YAML dumper, variant layout and id derivation, with this engine as its own
product engine: kernel ids are ``uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`` and the
bundle names its own native symbols (no ``native_symbols_from``).

Every entry must be a gfx950 dgrad identity in fp16 or bf16 that the packaged launch
runs: an implicit-GEMM identity of the dgrad AOT sweep
(``benchmarks/common/kernel_sweep.py``; it carries ``lds_k_outer`` and
``max_sub_gemms``), or a dgrad identity of the direct sweep
(``benchmarks/common/direct_kernel_sweep.py``) of a packaged algorithm
(:data:`PACKAGED_DIRECT_ALGORITHMS`), including the MFMA pipeline's weight-transform
helpers (``direct_dgrad_helper``). The helpers are packaged as kernels of their own
family (``kernel_family`` 2) that the kernel matcher never admits; the MFMA dispatch
loads them from the same pack. Anything else is listed with its reason and the tool
exits non-zero, unless ``--skip-unsupported`` is given.

The shipped catalog is ``configs/gfx950_conv_dgrad_aot_<dtype>.instances.json``,
written by ``tools/enumerate_conv_dgrad_aot_instances.py``::

    python tools/make_conv_dgrad_aot_config.py --rocke-root "$ROCKE" \\
        --instances configs/gfx950_conv_dgrad_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/dgrad_aot.yaml
    python generate.py --config "$WORK"/dgrad_aot.yaml --output-dir "$GENERATED" --force
    python tools/make_conv_dgrad_aot_config.py --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The catalog is split into packs (:data:`SHARDINGS`): by default one per dtype and
implicit-GEMM pipeline, plus one per dtype for the direct and helper kernels.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import make_conv_aot_config as _fwd  # noqa: E402

#: The shipped dgrad engine: its own product engine (ids are uuid5 of instance_id).
PRODUCT_ENGINE = "hipkernel:Gfx950ConvDgradAot"
#: The rocKE adapter every kernel of the bundle is built by.
ADAPTER_SOURCE = "builders/common/convolution_backward_data.py"
ADAPTER_BUILDER = "build_gfx950_conv_dgrad_aot"

#: Identity fields the packaged dgrad launch takes from an implicit-GEMM identity.
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
    "lds_k_outer",
    "max_sub_gemms",
    "grouped",
)
_BOOL_CARRIED = ("lds_k_outer", "grouped")
#: Identity fields an implicit-GEMM identity must hold at their KernelIdentity
#: default (they select emission the packaged launch or matcher does not model).
#: Stride, dilation and padding are 0 (runtime, read from the tilde records).
_CONSTANT = (
    "chiplet_swizzle",
    "unroll_k",
    "async_dma",
    "async_chunk_a",
    "async_chunk_b",
    "lds_layout",
    "lds_k_pad",
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
    "knobs",
)
#: Spec field order of an implicit-GEMM Gfx950ConvDgradAotSpec; also its descriptor
#: spec order (the KMD's required fields).
SPEC_ORDER = ("dtype",) + _CARRIED + ("instance_id", "layout")
#: The fields every direct and helper instance states after SPEC_ORDER, with their
#: KMD builder_default (an implicit-GEMM descriptor omits them).
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
    "cpg": 0,
    "kpg": 0,
}
#: The variant knobs after DIRECT_FIELDS, with their builder_default. A direct or
#: helper instance states those of its variant only (DIRECT_VARIANT_FIELDS).
DIRECT_KNOB_FIELDS = {
    "block_w": 0,
    "block_waves": 0,
    "block_h": 0,
    "block_q": 0,
    "block_groups": 0,
    "fold_k32": 0,
    "waves_k": 0,
    "waves_q": 0,
    "persistent_grid": 0,
    "runtime_k_loop": 0,
}
DIRECT_SPEC_ORDER = SPEC_ORDER + tuple(DIRECT_FIELDS)
#: The DIRECT_KNOB_FIELDS each direct and helper variant states, in KMD order.
DIRECT_VARIANT_FIELDS = {
    "depthwise_dgrad": ("block_w", "block_waves"),
    "grouped_dgrad_mfma": (
        "block_h",
        "block_q",
        "block_groups",
        "fold_k32",
        "waves_k",
        "waves_q",
        "persistent_grid",
        "runtime_k_loop",
    ),
    # The transpose states fold_k32 (always 0) too, so the MFMA dispatch's helper
    # lookup (dtype, family, variant, capabilities, cpg/kpg, fold_k32) reads stated
    # values for both helpers rather than relying on the KMD default.
    "transpose": ("fold_k32",),
    "reorganize": ("fold_k32",),
}
_STRING_FIELDS = (
    "dtype",
    "pipeline",
    "epilogue",
    "instance_id",
    "layout",
    "direct_variant",
)
_INT_FIELDS = tuple(
    f for f in DIRECT_SPEC_ORDER + tuple(DIRECT_KNOB_FIELDS) if f not in _STRING_FIELDS
)
#: The UED knobs: every KMD field that tunes a kernel, so the ranking model can tell
#: apart the instances one problem admits; together they make every per-graph
#: candidate tuple unique. Left out are the fields that only restate the problem the
#: instance was matched to (dtype, layout, the direct filter, stride, padding,
#: dilation, cpg/kpg), the constants (wave_size, grouped, max_sub_gemms) and
#: instance_id. The MFMA block_h is its row tile (the knob token), not a constant.
ENGINE_KNOBS = (
    "kernel_family",
    "direct_variant",
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "pipeline",
    "epilogue",
    "vector_size_a",
    "vector_size_b",
    "vector_size_c",
    "lds_k_outer",
    "block_w",
    "block_waves",
    "block_h",
    "block_q",
    "block_groups",
    "fold_k32",
    "waves_k",
    "waves_q",
    "persistent_grid",
    "runtime_k_loop",
)
#: Every dgrad direct-sweep algorithm the packaged launch runs (or loads, for the
#: MFMA helpers), and its direct_variant: AOT_DGRAD_DIRECT_ALGORITHMS of the adapter.
PACKAGED_DIRECT_ALGORITHMS = {
    "direct_depthwise_dgrad": "depthwise_dgrad",
    "direct_grouped_dgrad_mfma": "grouped_dgrad_mfma",
    "direct_dgrad_transpose": "transpose",
    "direct_dgrad_reorganize": "reorganize",
}
#: Direction of each packaged direct algorithm's identities.
_DIRECT_DIRECTIONS = {
    "direct_depthwise_dgrad": "direct_dgrad",
    "direct_grouped_dgrad_mfma": "direct_dgrad",
    "direct_dgrad_transpose": "direct_dgrad_helper",
    "direct_dgrad_reorganize": "direct_dgrad_helper",
}
#: The generator variant (kernel name) of the implicit-GEMM instances and of each
#: direct and helper variant.
IMPLICIT_VARIANT_NAME = "gfx950_conv_dgrad_aot_{dtype}_{instance_id}"
_DIRECT_VARIANT_NAMES = {
    "depthwise_dgrad": "gfx950_conv_dgrad_aot_dw_{dtype}_{instance_id}",
    "grouped_dgrad_mfma": "gfx950_conv_dgrad_aot_mfma_{dtype}_{instance_id}",
    "transpose": "gfx950_conv_dgrad_aot_transpose_{dtype}_{instance_id}",
    "reorganize": "gfx950_conv_dgrad_aot_reorganize_{dtype}_{instance_id}",
}

_bind_rocke = _fwd._bind_rocke
engine_slug = _fwd.engine_slug
dump_config = _fwd.dump_config
INSTANCE_UUID_NAMESPACE = _fwd.INSTANCE_UUID_NAMESPACE


def kernel_uuid(instance_id: str, engine_name: str = PRODUCT_ENGINE) -> str:
    """:func:`make_conv_aot_config.kernel_uuid` with this engine as the product."""
    return _fwd.kernel_uuid(instance_id, engine_name, PRODUCT_ENGINE)


def stabilize_ids(kdp_path: Path, engine_name: str = PRODUCT_ENGINE) -> int:
    """:func:`make_conv_aot_config.stabilize_ids` with this engine as the product."""
    return _fwd.stabilize_ids(kdp_path, engine_name, PRODUCT_ENGINE)


def spec_order(spec) -> tuple:
    """The descriptor spec order of ``spec``'s family and variant."""
    if not spec.is_direct:
        return SPEC_ORDER
    return DIRECT_SPEC_ORDER + DIRECT_VARIANT_FIELDS[spec.direct_variant]


def instance_spec(identity_dict: dict):
    """``(Gfx950ConvDgradAotSpec, "")`` for a supported identity, else ``(None, reason)``."""
    from benchmarks.common.kernel_cache import KernelIdentity

    identity = KernelIdentity.from_dict(identity_dict)
    if identity.is_direct:
        return _direct_instance_spec(identity)
    return _implicit_instance_spec(identity)


def instance_specs(identities: list, jobs: int = 1) -> list:
    """:func:`instance_spec` of each identity, in order, over ``jobs`` forked workers."""
    if (
        jobs <= 1
        or len(identities) < 200
        or "fork" not in multiprocessing.get_all_start_methods()
    ):
        return [instance_spec(identity) for identity in identities]
    chunk = max(1, len(identities) // (jobs * 8))
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(instance_spec, identities, chunksize=chunk)


def _checked(spec):
    from builders.common.convolution_backward_data import (
        supports_gfx950_conv_dgrad_aot,
    )

    ok, reason = supports_gfx950_conv_dgrad_aot(spec)
    return (spec, "") if ok else (None, reason)


def _implicit_instance_spec(identity):
    from benchmarks.common.kernel_cache import KernelIdentity
    from builders.common.convolution_backward_data import Gfx950ConvDgradAotSpec

    if (identity.arch, identity.direction, identity.algorithm) != (
        "gfx950",
        "dgrad",
        "implicit_gemm",
    ):
        return None, (
            f"not a gfx950 dgrad implicit-GEMM identity "
            f"({identity.arch}, {identity.direction}, {identity.algorithm})"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
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
    for name in _CONSTANT:
        if getattr(identity, name) != getattr(defaults, name):
            return None, (
                f"{name}={getattr(identity, name)!r} is not packaged "
                f"(only {getattr(defaults, name)!r})"
            )
    # Every packaged identity must be one rocKE's dgrad generator produces.
    from benchmarks.common.kernel_sweep import job_for_identity

    try:
        job_for_identity(identity)
    except LookupError as exc:
        return None, str(exc)
    values = {name: getattr(identity, name) for name in _CARRIED}
    for name in _BOOL_CARRIED:
        values[name] = int(values[name])
    spec = Gfx950ConvDgradAotSpec(
        dtype=identity.dtype_a, instance_id=identity.stable_hash(), **values
    )
    return _checked(spec)


def _direct_instance_spec(identity):
    """A dgrad direct or helper identity: the direct sweep must be able to build it,
    and the packaged launch must run (or load) its algorithm."""
    from benchmarks.common.direct_kernel_sweep import direct_job_for_identity
    from builders.common.convolution_backward_data import (
        Gfx950ConvDgradAotSpec,
        aot_dgrad_direct_instance_fields,
    )

    if identity.arch != "gfx950":
        return None, f"not a gfx950 dgrad direct identity ({identity.arch})"
    if identity.algorithm not in PACKAGED_DIRECT_ALGORITHMS:
        return None, (
            f"direct algorithm {identity.algorithm!r} has no packaged dgrad launch "
            f"(supported: {sorted(PACKAGED_DIRECT_ALGORITHMS)})"
        )
    if identity.direction != _DIRECT_DIRECTIONS[identity.algorithm]:
        return None, (
            f"{identity.algorithm} is a {_DIRECT_DIRECTIONS[identity.algorithm]} "
            f"algorithm, the identity says {identity.direction}"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        return None, "mixed operand dtypes are not packaged"
    try:
        direct_job_for_identity(identity)
        fields = aot_dgrad_direct_instance_fields(identity)
    except (LookupError, TypeError, ValueError) as exc:
        return None, str(exc)
    spec = Gfx950ConvDgradAotSpec(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        wave_size=identity.wave_size,
        tile_m=0,
        tile_n=0,
        tile_k=0,
        warp_m=0,
        warp_n=0,
        warp_tile_m=0,
        warp_tile_n=0,
        warp_tile_k=0,
        pipeline=identity.pipeline,
        epilogue=identity.epilogue,
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        lds_k_outer=0,
        max_sub_gemms=0,
        grouped=0,
        **fields,
    )
    return _checked(spec)


#: How :func:`build_config` splits a catalog into packs, one KDP each.
#: ``dtype``: one pack per dtype. ``pipeline``: one per dtype and implicit-GEMM
#: pipeline, plus one per dtype for the direct and helper kernels.
SHARDINGS = ("dtype", "pipeline")


def shard_name(spec, sharding: str = "pipeline") -> str:
    """The pack ``spec`` belongs to under ``sharding`` (see :data:`SHARDINGS`)."""
    if sharding == "dtype":
        return spec.dtype
    if sharding != "pipeline":
        raise ValueError(f"unknown sharding {sharding!r}; expected one of {SHARDINGS}")
    if spec.is_direct:
        return f"{spec.dtype}_direct"
    return f"{spec.dtype}_{spec.pipeline}"


def kmd_fields() -> list:
    """The KMD: SPEC_ORDER required, then the direct fields and knobs with defaults."""
    return [
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


def build_config(
    specs: list,
    engine_name: str = PRODUCT_ENGINE,
    authored_subpath: str = "",
    sharding: str = "pipeline",
) -> dict:
    """The generator config for ``specs`` (already validated, de-duplicated).

    The packs are shards of one catalog with no discriminator. Within a pack the
    implicit-GEMM instances are one variant, and each direct or helper variant its
    own, stating DIRECT_FIELDS and its knobs and leaving the others to their
    builder_default. ``engine_name`` other than :data:`PRODUCT_ENGINE` borrows this
    engine's native symbols.
    """
    metadata = list(DIRECT_SPEC_ORDER + tuple(DIRECT_KNOB_FIELDS))
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        implicit = [s for s in shards[shard] if not s.is_direct]
        if implicit:
            variants.append(_fwd._variant(IMPLICIT_VARIANT_NAME, SPEC_ORDER, implicit))
        for variant_name in _DIRECT_VARIANT_NAMES:
            members = [
                s
                for s in shards[shard]
                if s.is_direct and s.direct_variant == variant_name
            ]
            if members:
                variants.append(
                    _fwd._variant(
                        _DIRECT_VARIANT_NAMES[variant_name],
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
                    "source": ADAPTER_SOURCE,
                    "builder": ADAPTER_BUILDER,
                },
                "variants": variants,
            }
        )
    engine = {
        "name": engine_name,
        "sdk_version": "1.0.0",
        "knobs": list(ENGINE_KNOBS),
        "heuristic": "native",
    }
    if engine_name != PRODUCT_ENGINE:
        engine["native_symbols_from"] = PRODUCT_ENGINE
    return {
        "dialect": "packaged",
        "authored_subpath": authored_subpath or f"rocKE/{engine_slug(engine_name)}",
        "engine": engine,
        "kmd_fields": kmd_fields(),
        "kernel_source_kind": "rocke",
        # The implicit-GEMM sub-GEMM record table and the MFMA weight scratch are
        # sized per problem by the native handler's workspaceBytes.
        "workspace_policy": "derived",
        "graph_match": {"shape": "shared_shape", "discriminator": "none"},
        "specialization": {
            "metadata_fields": metadata,
            "matcher_only_fields": [],
            "bindings": {name: {"field": name} for name in metadata},
        },
        "packs": packs,
    }


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
        "derives (default: %(default)s).",
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
        default="pipeline",
        help="How the catalog is split into packs, one KDP each (default: %(default)s).",
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
        f"# Generated by tools/make_conv_dgrad_aot_config.py from {sources}; "
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
