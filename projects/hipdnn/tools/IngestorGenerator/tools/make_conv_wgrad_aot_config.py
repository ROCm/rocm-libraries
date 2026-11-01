#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE AOT backward-filter conv instances into a hipkernel:Gfx950ConvWgradAot config.

The wgrad sibling of ``make_conv_dgrad_aot_config.py``: same input (JSON lists of rocKE
``KernelIdentity`` dicts, or objects carrying one under ``"identity"``), same output (a
packaged-dialect generator config), same ``--stabilize-ids`` step. It reuses that tool's
YAML dumper, variant layout and id derivation, with this engine as its own product engine:
kernel ids are ``uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`` and the bundle names its own
native symbols (no ``native_symbols_from``).

Every entry must be a gfx950 wgrad identity in fp16 or bf16 that the packaged launch runs:
an implicit-GEMM identity of the wgrad AOT sweep (``benchmarks/common/kernel_sweep.py``),
or a wgrad identity of the direct sweep (``benchmarks/common/direct_kernel_sweep.py``) of a
packaged algorithm (:data:`builders.common.convolution_backward_filter.AOT_WGRAD_DIRECT_PREFIXES`),
including the partial-slot fold helper (``direct_wgrad_helper``). The helper is packaged as
a kernel of its own family (``kernel_family`` 2) that the kernel matcher never admits.
Anything else is listed with its reason and the tool exits non-zero, unless
``--skip-unsupported`` is given.

The identity -> spec bridge goes through the catalog label: the adapter's
``parse_wgrad_instance_name`` already turns one ``KernelIdentity.label()`` string into a
``Gfx950ConvWgradAotSpec`` (the catalog file *is* a list of such labels), so this tool
re-derives the label from each identity rather than duplicating that parse.

The shipped catalog is ``configs/gfx950_conv_wgrad_aot_<dtype>.instances.json``, written by
``tools/enumerate_conv_wgrad_aot_instances.py``::

    python tools/make_conv_wgrad_aot_config.py --rocke-root "$ROCKE" \\
        --instances configs/gfx950_conv_wgrad_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/wgrad_aot.yaml
    python generate.py --config "$WORK"/wgrad_aot.yaml --output-dir "$GENERATED" --force
    python tools/make_conv_wgrad_aot_config.py --stabilize-ids "$GENERATED"/<...>/*.kdp.json

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

#: The shipped wgrad engine: its own product engine (ids are uuid5 of instance_id).
PRODUCT_ENGINE = "hipkernel:Gfx950ConvWgradAot"
#: The rocKE adapter every kernel of the bundle is built by.
ADAPTER_SOURCE = "builders/common/convolution_backward_filter.py"
ADAPTER_BUILDER = "build_gfx950_conv_wgrad_aot"

#: Identity fields the packaged wgrad launch takes from an implicit-GEMM identity
#: (Gfx950ConvWgradAotSpec's leading, shape-generic fields).
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
    "group_merge",
    "prefetch_addr",
    "split_k",
    "two_stage_flag",
    "ws_replicas",
)
_BOOL_CARRIED = ("lds_k_outer", "group_merge", "prefetch_addr", "split_k", "two_stage_flag")
#: Spec field order of an implicit-GEMM Gfx950ConvWgradAotSpec; also its descriptor
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
    #: direct_wgrad_fold-only (kernels/common/conv_direct_dw_wgrad.py's DwWgradFoldSpec's
    #: only real build-time parameter); 0 for every other instance. Not an
    #: AOT_WGRAD_DIRECT_KNOBS entry (the fold states no knobs), so _direct_knob_fields()
    #: does not carry it -- it needs its own DIRECT_FIELDS entry like filter_h/cpg/kpg.
    "taps": 0,
}
DIRECT_SPEC_ORDER = SPEC_ORDER + tuple(DIRECT_FIELDS)
_STRING_FIELDS = ("dtype", "pipeline", "epilogue", "instance_id", "layout", "direct_variant")


def _direct_knob_fields() -> dict:
    """Every direct/helper knob with its KMD default: the adapter's
    ``_KNOB_DEFAULTS``, not a second list that could drift from it."""
    from builders.common.convolution_backward_filter import _KNOB_DEFAULTS

    return dict(_KNOB_DEFAULTS)


def _direct_variant_fields() -> dict:
    """Per variant, the knobs it states, in KMD order (the adapter's
    ``AOT_WGRAD_DIRECT_KNOBS``; the fold helper states none, the workspace-reduce
    helper states none either -- both are shape-generic passes)."""
    from builders.common.convolution_backward_filter import (
        AOT_WGRAD_DIRECT_KNOBS,
        HELPER_VARIANT_FOLD,
        HELPER_VARIANT_WORKSPACE_REDUCE,
    )

    order = list(_direct_knob_fields())
    stated = {variant: set(fields) for variant, fields in AOT_WGRAD_DIRECT_KNOBS.items()}
    stated[HELPER_VARIANT_FOLD] = set()
    stated[HELPER_VARIANT_WORKSPACE_REDUCE] = set()
    return {variant: tuple(f for f in order if f in fields) for variant, fields in stated.items()}


def _int_fields() -> tuple:
    """Every KMD field that is an int. Resolved on first use: the knob set comes
    from the adapter, which is bound at run time."""
    return tuple(f for f in DIRECT_SPEC_ORDER + tuple(_direct_knob_fields()) if f not in _STRING_FIELDS)


#: The UED knobs: every KMD field that tunes a kernel, so the ranking model can tell
#: apart the instances one problem admits. Left out: fields that only restate the
#: problem the instance was matched to (dtype, layout, the direct filter, stride,
#: padding, dilation, cpg/kpg), the constants (wave_size) and instance_id.
def engine_knobs() -> tuple:
    """The UED knobs: every KMD field that tunes a kernel, so the ranking model can
    tell apart the instances one problem admits.

    Int-typed only: the config loader's ``_KNOWN_KMD_FIELD``/knob-type check accepts
    a non-int knob and then silently produces no knob at all
    (``GenericPlanBuilder::getCustomKnobs``), so ``direct_variant``, ``pipeline`` and
    ``epilogue`` stay KMD/specialization fields but are not listed here even though
    the dgrad adapter's otherwise-identical ``engine_knobs()`` lists them -- that
    config has not been regenerated since the loader started enforcing this (see
    ``WGRAD_AOT_PROGRESS.md``).
    """
    fixed = (
        "kernel_family",
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
        "vector_size_a",
        "vector_size_b",
        "vector_size_c",
        "lds_k_outer",
        "group_merge",
        "prefetch_addr",
        "split_k",
    )
    return fixed + tuple(_direct_knob_fields())


_PACKAGED_DIRECT_CACHE = None


def packaged_direct_algorithms() -> dict:
    """Each wgrad direct-sweep algorithm with a packaged launch, and its
    ``direct_variant`` (``AOT_WGRAD_DIRECT_PREFIXES`` of the adapter; the fold
    helper is derived, not listed by its own algorithm name)."""
    return _packaged_direct()[0]


def direct_directions() -> dict:
    """The ``KernelIdentity.direction`` each packaged direct algorithm carries."""
    return _packaged_direct()[1]


def _packaged_direct():
    global _PACKAGED_DIRECT_CACHE
    if _PACKAGED_DIRECT_CACHE is None:
        from builders.common.convolution_backward_filter import AOT_WGRAD_DIRECT_PREFIXES

        algorithms, directions = {}, {}
        for algorithm, (_family, variant) in AOT_WGRAD_DIRECT_PREFIXES.items():
            algorithms[algorithm] = variant
            directions[algorithm] = "direct_wgrad"
        _PACKAGED_DIRECT_CACHE = (algorithms, directions)
    return _PACKAGED_DIRECT_CACHE


#: The generator variant (kernel name) of the implicit-GEMM instances and of each
#: direct and helper variant.
IMPLICIT_VARIANT_NAME = "gfx950_conv_wgrad_aot_{dtype}_{instance_id}"
_DIRECT_VARIANT_NAMES = {
    "direct_wgrad": "gfx950_conv_wgrad_aot_plain_{dtype}_{instance_id}",
    "direct_wgrad_tiled": "gfx950_conv_wgrad_aot_tiled_{dtype}_{instance_id}",
    "direct_wgrad_dw": "gfx950_conv_wgrad_aot_dw_{dtype}_{instance_id}",
    "direct_wgrad_dw_cols": "gfx950_conv_wgrad_aot_dw_cols_{dtype}_{instance_id}",
    "direct_wgrad_dw_mfma": "gfx950_conv_wgrad_aot_dw_mfma_{dtype}_{instance_id}",
    "direct_wgrad_grouped_mfma": "gfx950_conv_wgrad_aot_grouped_mfma_{dtype}_{instance_id}",
    "direct_wgrad_gather": "gfx950_conv_wgrad_aot_gather_{dtype}_{instance_id}",
    "direct_wgrad_fold": "gfx950_conv_wgrad_aot_fold_{dtype}_{instance_id}",
    "direct_wgrad_cast": "gfx950_conv_wgrad_aot_cast_{dtype}_{instance_id}",
}

_bind_rocke = _fwd._bind_rocke
engine_slug = _fwd.engine_slug
dump_config = _fwd.dump_config
INSTANCE_UUID_NAMESPACE = _fwd.INSTANCE_UUID_NAMESPACE


def kernel_uuid(instance_id: str, engine_name: str = PRODUCT_ENGINE) -> str:
    return _fwd.kernel_uuid(instance_id, engine_name, PRODUCT_ENGINE)


def stabilize_ids(kdp_path: Path, engine_name: str = PRODUCT_ENGINE) -> int:
    return _fwd.stabilize_ids(kdp_path, engine_name, PRODUCT_ENGINE)


#: Every KMD field, in declaration order -- what a kernel's metadata/spec_order
#: must cover in full: the config loader now requires every ``engine.knobs`` entry
#: to be either in a kernel's metadata or pinned in its ``kernel_source.spec``, so a
#: variant can no longer list only the fields its own family uses and leave the
#: rest to the KMD ``default_value`` alone (unlike the otherwise-identical dgrad
#: adapter's ``spec_order()``, not regenerated since the loader started enforcing
#: this; see ``WGRAD_AOT_PROGRESS.md``). Every ``Gfx950ConvWgradAotSpec`` already
#: carries a concrete value (0/none) for a field its own family does not use, so
#: this is just "read every field", not a semantic change per variant.
_FULL_SPEC_ORDER = DIRECT_SPEC_ORDER + tuple(_direct_knob_fields())


def spec_order(spec) -> tuple:
    del spec
    return _FULL_SPEC_ORDER


def _label_of(identity) -> str:
    return identity.label()


def instance_spec(identity_dict: dict):
    """``(Gfx950ConvWgradAotSpec, "")`` for a supported identity, else ``(None, reason)``.

    Bridges through the catalog label: the adapter's own
    ``parse_wgrad_instance_name`` is the single source of truth for turning a label
    into a spec (the same parse ``instances_wgrad.txt`` is read with), so this does
    not duplicate its field mapping. The label alone carries no identity hash, so
    ``instance_id`` is set here from the actual ``KernelIdentity`` (``parse_...``
    leaves it ``""``) -- this is what ``specs()`` below de-duplicates on, mirroring
    ``make_conv_dgrad_aot_config.py``'s ``instance_id=identity.stable_hash()``.
    """
    from dataclasses import replace

    from benchmarks.common.kernel_cache import KernelIdentity
    from builders.common.convolution_backward_filter import (
        parse_wgrad_instance_name,
        supports_gfx950_conv_wgrad_aot,
    )

    identity = KernelIdentity.from_dict(identity_dict)
    try:
        spec = parse_wgrad_instance_name(_label_of(identity))
    except (ValueError, KeyError) as exc:
        return None, f"cannot parse as a wgrad AOT instance name: {exc}"
    spec = replace(spec, instance_id=identity.stable_hash())
    ok, reason = supports_gfx950_conv_wgrad_aot(spec)
    return (spec, "") if ok else (None, reason)


def instance_specs(identities: list, jobs: int = 1) -> list:
    if (
        jobs <= 1
        or len(identities) < 200
        or "fork" not in multiprocessing.get_all_start_methods()
    ):
        return [instance_spec(identity) for identity in identities]
    chunk = max(1, len(identities) // (jobs * 8))
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        return pool.map(instance_spec, identities, chunksize=chunk)


SHARDINGS = ("dtype", "pipeline")


def shard_name(spec, sharding: str = "pipeline") -> str:
    if sharding == "dtype":
        return spec.dtype
    if sharding != "pipeline":
        raise ValueError(f"unknown sharding {sharding!r}; expected one of {SHARDINGS}")
    if spec.is_direct:
        return f"{spec.dtype}_direct"
    return f"{spec.dtype}_{spec.pipeline}"


def kmd_fields() -> list:
    """The KMD field list: ``SPEC_ORDER`` plus every direct/helper field, each with
    its KMD ``default_value`` (an implicit-GEMM descriptor states only ``SPEC_ORDER``
    and omits the rest, so they need one). The config loader's KMD field schema is
    ``{name, type, default_value}`` (``codegen/config_loader.py``'s
    ``_KNOWN_KMD_FIELD``) -- no fourth key, unlike ``make_conv_aot_config.py``'s
    ``builder_default``, which the loader has never read (see
    ``WGRAD_AOT_PROGRESS.md``).
    """
    return [
        {"name": name, "type": "int" if name in _int_fields() else "string"}
        for name in SPEC_ORDER
    ] + [
        {
            "name": name,
            "type": "int" if name in _int_fields() else "string",
            "default_value": default,
        }
        for name, default in {**DIRECT_FIELDS, **_direct_knob_fields()}.items()
    ]


#: A two-stage instance's packaged KDP names two launches sharing one spec:
#: Stage 1 (the f32-scratch-atomic partial-sums GEMM, the pack's own
#: ``kernel_defaults`` builder) and Stage 2 (the workspace-reduce fold, a
#: different builder in the same adapter module). An engine's kernels must
#: all name the same launch set (see the README's "Multi-launch kernels"), so
#: two-stage instances ship as their own engine (``--multi-launch``), never
#: mixed with single-launch ones.
MULTI_LAUNCH_STAGES = {
    "stage1": {},
    "stage2": {"builder": "build_gfx950_conv_wgrad_aot_stage2"},
}


def build_config(
    specs: list,
    engine_name: str = PRODUCT_ENGINE,
    authored_subpath: str = "",
    sharding: str = "pipeline",
    multi_launch: bool = False,
) -> dict:
    metadata = list(DIRECT_SPEC_ORDER + tuple(_direct_knob_fields()))
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        implicit = [s for s in shards[shard] if not s.is_direct]
        if implicit:
            variants.append(
                _fwd._variant(IMPLICIT_VARIANT_NAME, spec_order(implicit[0]), implicit)
            )
        for variant_name in _DIRECT_VARIANT_NAMES:
            members = [
                s for s in shards[shard] if s.is_direct and s.direct_variant == variant_name
            ]
            if members:
                variants.append(
                    _fwd._variant(
                        _DIRECT_VARIANT_NAMES[variant_name], spec_order(members[0]), members
                    )
                )
        kernel_defaults = {
            "kind": "rocke",
            "source": ADAPTER_SOURCE,
            "builder": ADAPTER_BUILDER,
        }
        if multi_launch:
            kernel_defaults["launches"] = MULTI_LAUNCH_STAGES
        packs.append(
            {
                "name": shard,
                "arch": ["gfx950"],
                # A multi-pack engine needs one discriminator per pack (the config
                # loader's `_check_pack_discriminators`); the shard name is already a
                # unique, valid C++ identifier, so it doubles as the pack's
                # operation-scoped matcher symbol name.
                "discriminator": shard,
                "kernel_defaults": kernel_defaults,
                "variants": variants,
            }
        )
    engine = {
        "name": engine_name,
        "sdk_version": "1.0.0",
        "knobs": list(engine_knobs()),
        "heuristic": "native",
    }
    return {
        "dialect": "packaged",
        "authored_subpath": authored_subpath or f"rocKE/{engine_slug(engine_name)}",
        "engine": engine,
        "kmd_fields": kmd_fields(),
        "kernel_source_kind": "rocke",
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
    parser.add_argument("--rocke-root", type=Path, help="rocKE tree (platform/, library/).")
    parser.add_argument(
        "--instances", type=Path, nargs="+", help="JSON lists of KernelIdentity dicts, read as one catalog."
    )
    parser.add_argument("--output", type=Path, help="Generator config to write.")
    parser.add_argument(
        "--skip-unsupported",
        action="store_true",
        help="Drop unsupported identities (each is still reported) instead of failing.",
    )
    parser.add_argument("--engine-name", default=PRODUCT_ENGINE)
    parser.add_argument(
        "--multi-launch",
        action="store_true",
        help=(
            "Every instance is two-stage (Stage 1 + Stage 2 workspace-reduce); "
            "ship as a sibling engine naming both launches. Mixing multi- and "
            "single-launch instances in one engine is a config-loader error, so "
            "this flag is all-or-nothing for the whole --instances set."
        ),
    )
    parser.add_argument("--authored-subpath", default="")
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--sharding", choices=SHARDINGS, default="pipeline")
    parser.add_argument("--stabilize-ids", type=Path, nargs="+", metavar="KDP")
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
        print(f"{len(rejected)} unsupported instance(s); nothing written", file=sys.stderr)
        return 1
    if not specs:
        print("no supported instances; nothing written", file=sys.stderr)
        return 1
    if args.multi_launch:
        non_two_stage = [s.instance_id for s in specs if not s.is_two_stage]
        if non_two_stage:
            print(
                f"--multi-launch given but {len(non_two_stage)} instance(s) are not "
                f"two-stage (an engine cannot mix launch sets): {non_two_stage[:5]}",
                file=sys.stderr,
            )
            return 1
    else:
        two_stage = [s.instance_id for s in specs if s.is_two_stage]
        if two_stage:
            print(
                f"{len(two_stage)} two-stage instance(s) in a non-multi-launch run "
                f"(an engine cannot mix launch sets); pass --multi-launch for a "
                f"two-stage-only --instances set: {two_stage[:5]}",
                file=sys.stderr,
            )
            return 1
    sources = ", ".join(path.name for path in args.instances)
    header = f"# Generated by tools/make_conv_wgrad_aot_config.py from {sources}; do not edit by hand.\n"
    config = build_config(
        specs, args.engine_name, args.authored_subpath, args.sharding, multi_launch=args.multi_launch
    )
    args.output.write_text(header + dump_config(config))
    print(
        f"{len(specs)} instance(s) written to {args.output} "
        f"({len(entries) - len(rejected) - len(specs)} duplicate(s), {len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
