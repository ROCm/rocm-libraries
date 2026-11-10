#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn rocKE gfx1250 forward-conv AOT instances into a hipkernel:Gfx1250ConvFwdAot config.

The gfx1250 sibling of ``make_conv_aot_config.py`` and ``make_conv_dgrad_aot_config.py``:
same input (JSON lists of rocKE ``KernelIdentity`` dicts, or objects carrying one under
``"identity"``), same output (a packaged-dialect generator config), same
``--stabilize-ids`` step, reusing that tool's config envelope, list reader, YAML
dumper, variant layout and id derivation. Kernel ids are ``uuid5(INSTANCE_UUID_NAMESPACE, instance_id)``.

Every entry must be a gfx1250 forward identity in fp16 or bf16 that the packaged launch
runs: the rocKE adapter (``builders/gfx1250/conv_fwd_aot.py``) converts it into a
``Gfx1250ConvFwdAotSpec`` or says why not. Anything else is listed with its reason and
the tool exits non-zero, unless ``--skip-unsupported`` is given.

The catalog is ordered: within a dtype, a kernel's ``priority`` falls with its position
in the input lists (the order rocKE registers it in), which is the order the engine tries
the kernels a graph admits when no ranker decides. The shipped lists are
``configs/gfx1250_conv_fwd_aot_<dtype>.instances.json``, written by
``tools/enumerate_gfx1250_conv_fwd_aot_instances.py``::

    python tools/make_gfx1250_conv_fwd_aot_config.py --rocke-root "$ROCKE" \\
        --instances configs/gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json \\
        --output "$WORK"/gfx1250_fwd_aot.yaml
    python generate.py --config "$WORK"/gfx1250_fwd_aot.yaml --output-dir "$GENERATED" --force
    python tools/make_gfx1250_conv_fwd_aot_config.py --stabilize-ids "$GENERATED"/<...>/*.kdp.json

The catalog is split into packs (:data:`SHARDINGS`): by default one per dtype for the
implicit-GEMM instances and one per dtype for the direct ones.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import make_conv_aot_config as _fwd  # noqa: E402

PRODUCT_ENGINE = "hipkernel:Gfx1250ConvFwdAot"
ARCH = "gfx1250"
#: The rocKE adapter every kernel of the bundle is built by.
ADAPTER_SOURCE = "builders/gfx1250/conv_fwd_aot.py"
ADAPTER_BUILDER = "build_gfx1250_conv_fwd_aot"

_bind_rocke = _fwd._bind_rocke
engine_slug = _fwd.engine_slug
dump_config = _fwd.dump_config
INSTANCE_UUID_NAMESPACE = _fwd.INSTANCE_UUID_NAMESPACE


def _adapter():
    from builders.gfx1250 import conv_fwd_aot

    return conv_fwd_aot


def kernel_uuid(instance_id: str, engine_name: str = PRODUCT_ENGINE) -> str:
    """:func:`make_conv_aot_config.kernel_uuid` with this engine as the product."""
    return _fwd.kernel_uuid(instance_id, engine_name, PRODUCT_ENGINE)


def stabilize_ids(kdp_path: Path, engine_name: str = PRODUCT_ENGINE) -> int:
    """:func:`make_conv_aot_config.stabilize_ids` with this engine as the product."""
    return _fwd.stabilize_ids(kdp_path, engine_name, PRODUCT_ENGINE)


def common_fields() -> tuple:
    """The fields every instance states: dtype, instance_id and the identity fields
    the adapter carries."""
    A = _adapter()
    return ("dtype", "instance_id") + A.CARRIED_STRINGS + A.CARRIED_INTS


def spec_order(algorithm: str) -> tuple:
    """The descriptor spec order of a family: the common fields, then its knobs (a
    knob carried in an identity field of the same name is already among them)."""
    A = _adapter()
    common = common_fields()
    return common + tuple(k for k in A.FAMILY_KNOBS[algorithm] if k not in common)


def _is_string(name: str) -> bool:
    return name in _adapter().STRING_FIELDS


def kmd_fields() -> list:
    """The KMD: the common fields required, then every family knob with its builder
    default (0, or "" for a string), which a family that does not take it keeps."""
    A = _adapter()
    common = common_fields()
    required = [
        {"name": name, "type": "string" if _is_string(name) else "int"}
        for name in common
    ]
    knobs = [
        {
            "name": name,
            "type": "string" if _is_string(name) else "int",
            "default_value": "" if _is_string(name) else 0,
            "builder_default": True,
        }
        for name in A.KNOB_FIELDS
        if name not in common
    ]
    return required + knobs


def engine_knobs() -> tuple:
    """The UED knobs: every KMD field that tunes a kernel, so a ranking model can tell
    apart the instances one problem admits. Left out are the fields that only restate
    the problem the instance was matched to (dtype, the baked filter, stride, padding,
    dilation and channel counts), the constant wave_size and instance_id."""
    A = _adapter()
    tuning = (
        "algorithm",
        "pipeline",
        "epilogue",
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
        "group_merge",
        "grouped",
    )
    return tuning + tuple(k for k in A.KNOB_FIELDS if k not in tuning)


def instance_spec(identity_dict: dict):
    """``(Gfx1250ConvFwdAotSpec, "")`` for a supported identity, else ``(None, reason)``."""
    from kernels.conv.kernel_identity import KernelIdentity

    A = _adapter()
    try:
        identity = KernelIdentity.from_dict(identity_dict)
        spec = A.spec_from_identity(identity)
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc)
    ok, reason = A.supports_gfx1250_conv_fwd_aot(spec)
    return (spec, "") if ok else (None, reason)


def instance_specs(identities: list, jobs: int = 1) -> list:
    """:func:`instance_spec` of each identity, in order (``jobs`` is accepted for the
    sibling tools' interface; the conversion is cheap)."""
    return [instance_spec(identity) for identity in identities]


#: How :func:`build_config` splits a catalog into packs, one KDP each.
#: ``dtype``: one pack per dtype. ``family``: one per dtype for the implicit-GEMM
#: instances and one per dtype for the direct ones.
SHARDINGS = ("dtype", "family")

#: The generator variant (kernel name) of each family.
_VARIANT_NAMES = {
    "implicit_gemm": "gfx1250_conv_fwd_aot_igemm_{dtype}_{instance_id}",
    "direct_fwd_wmma": "gfx1250_conv_fwd_aot_wmma_{dtype}_{instance_id}",
    "direct_fwd_gstream_wmma": "gfx1250_conv_fwd_aot_gstream_{dtype}_{instance_id}",
    "direct_fwd_tapouter_wmma": "gfx1250_conv_fwd_aot_tapouter_{dtype}_{instance_id}",
    "direct_fwd_smallc_wmma": "gfx1250_conv_fwd_aot_smallc_{dtype}_{instance_id}",
    "direct_fwd_smallk_wmma": "gfx1250_conv_fwd_aot_smallk_{dtype}_{instance_id}",
    "direct_dw_fwd_wmma": "gfx1250_conv_fwd_aot_dw_wmma_{dtype}_{instance_id}",
    "direct_depthwise": "gfx1250_conv_fwd_aot_dw_{dtype}_{instance_id}",
    "direct_depthwise_tiled": "gfx1250_conv_fwd_aot_dw_tiled_{dtype}_{instance_id}",
    "direct_depthwise_col": "gfx1250_conv_fwd_aot_dw_col_{dtype}_{instance_id}",
    "direct_depthwise_spatial": "gfx1250_conv_fwd_aot_dw_spatial_{dtype}_{instance_id}",
}


def shard_name(spec, sharding: str = "family") -> str:
    """The pack ``spec`` belongs to under ``sharding`` (see :data:`SHARDINGS`)."""
    if sharding == "dtype":
        return spec.dtype
    if sharding != "family":
        raise ValueError(f"unknown sharding {sharding!r}; expected one of {SHARDINGS}")
    return f"{spec.dtype}_{'direct' if spec.is_direct else 'implicit'}"


def priorities(specs: list) -> dict:
    """``instance_id -> priority``: within a dtype, falling with list position."""
    by_dtype = collections.defaultdict(list)
    for spec in specs:
        by_dtype[spec.dtype].append(spec.instance_id)
    return {
        instance_id: len(ids) - 1 - index
        for ids in by_dtype.values()
        for index, instance_id in enumerate(ids)
    }


def _variant(name: str, order: tuple, specs: list, priority: dict) -> dict:
    variant = _fwd._variant(name, order, specs)
    for shape, spec in zip(variant["shapes"], specs):
        shape["priority"] = priority[spec.instance_id]
    return variant


def build_config(
    specs: list,
    engine_name: str = PRODUCT_ENGINE,
    authored_subpath: str = "",
    sharding: str = "family",
    native_symbols_from_slug: str | None = None,
) -> dict:
    """The generator config for ``specs`` (validated, de-duplicated, in list order),
    in :func:`make_conv_aot_config.packaged_config`'s envelope (``engine_name``,
    ``authored_subpath`` and ``native_symbols_from_slug`` as there)."""
    priority = priorities(specs)
    metadata = [field["name"] for field in kmd_fields()]
    shards = collections.defaultdict(list)
    for spec in specs:
        shards[shard_name(spec, sharding)].append(spec)
    packs = []
    for shard in sorted(shards):
        variants = []
        for algorithm, name in _VARIANT_NAMES.items():
            members = [s for s in shards[shard] if s.algorithm == algorithm]
            if members:
                variants.append(
                    _variant(name, spec_order(algorithm), members, priority)
                )
        packs.append(
            {
                "name": shard,
                "arch": [ARCH],
                "kernel_defaults": {
                    "kind": "rocke",
                    "source": ADAPTER_SOURCE,
                    "builder": ADAPTER_BUILDER,
                },
                "variants": variants,
            }
        )
    return _fwd.packaged_config(
        engine_name=engine_name,
        product_engine=PRODUCT_ENGINE,
        knobs=engine_knobs(),
        kmd_fields=kmd_fields(),
        metadata=metadata,
        packs=packs,
        authored_subpath=authored_subpath,
        native_symbols_from_slug=native_symbols_from_slug,
    )


def read_catalog(paths: list) -> tuple:
    """``(specs, rejected)`` of the identity lists ``paths``: specs in list order,
    de-duplicated by instance_id; rejected as ``(index, reason)``."""
    _, identities, rejected = _fwd.identity_entries(paths)
    specs, seen = [], set()
    for index, identity in identities.items():
        spec, reason = instance_spec(identity)
        if spec is None:
            rejected.append((index, reason))
        elif spec.instance_id not in seen:
            seen.add(spec.instance_id)
            specs.append(spec)
    return specs, sorted(rejected)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root", type=Path, help="rocKE tree (platform/, library/)."
    )
    parser.add_argument(
        "--instances",
        type=Path,
        nargs="+",
        help="JSON lists of KernelIdentity dicts, read as one catalog, in order.",
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
        "--sharding",
        choices=SHARDINGS,
        default="family",
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

    try:
        specs, rejected = read_catalog(args.instances)
    except ValueError as exc:
        parser.error(str(exc))
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
        f"# Generated by tools/make_gfx1250_conv_fwd_aot_config.py from {sources}; "
        "do not edit by hand.\n"
    )
    config = build_config(specs, args.engine_name, args.authored_subpath, args.sharding)
    args.output.write_text(header + dump_config(config), newline="")
    print(
        f"{len(specs)} instance(s) written to {args.output} ({len(rejected)} unsupported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
