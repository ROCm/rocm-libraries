# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Helpers the rocKE gfx950 conv AOT tools share, whatever their direction.

The tools that turn rocKE's registered conv instances into a hipkernel engine --
``enumerate_conv_dgrad_aot_instances.py`` writes the instance lists,
``make_conv_dgrad_aot_config.py`` turns them into a generator config and stabilizes
the kernel ids of what ``generate.py`` emits -- read the catalog, write the lists,
dump the config and derive kernel UUIDs the same way for every direction. Those
steps live here, so each direction's tools hold only what is particular to it.
rocKE is bound on ``sys.path`` by :func:`bind_rocke` before any helper that reads it.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import re
import sys
import uuid
from pathlib import Path

_GENERATOR_ROOT = Path(__file__).resolve().parent.parent
if str(_GENERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_ROOT))

from codegen.models import EngineSpec  # noqa: E402

ARCH = "gfx950"
DTYPES = ("fp16", "bf16")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "configs"

#: Namespace for kernel UUIDs derived from an instance_id (uuid5).
INSTANCE_UUID_NAMESPACE = uuid.UUID("6b0d3b8e-6a58-5d0f-9f43-3c6f5f2a9c11")

#: A UUID as ``str(uuid.UUID)`` spells it: the form of every generated kernel id.
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def bind_rocke(rocke_root: Path) -> None:
    """Put rocKE's ``platform/python`` and ``library`` on ``sys.path``."""
    for sub in ("platform/python", "library"):
        path = str((rocke_root / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Instance lists
# ---------------------------------------------------------------------------


def read_rocke_catalog(direction: str, arch: str = ARCH) -> list:
    """rocKE's own registered instances of ``(arch, direction)``, as dicts, in
    rocKE's on-disk order (a set sorted by label for dgrad, the priority order for
    fwd: ``dispatch.grouped_convolution``'s module docstring). rocKE must already be
    bound (:func:`bind_rocke`)."""
    from dispatch.grouped_convolution import (
        _INSTANCE_DIR,
        _INSTANCE_LISTS,
        load_instance_list,
    )

    path = _INSTANCE_DIR / _INSTANCE_LISTS[(arch, direction)]
    return [i.to_dict() for i in load_instance_list(path).identities]


def read_excludes(path: Path) -> dict:
    """``label -> reason`` of a known-bug exclusion list (a JSON list of
    ``{"label": ..., "reason": ...}``); ``{}`` when ``path`` does not exist."""
    if not path.exists():
        return {}
    entries = json.loads(path.read_text())
    out = {}
    for entry in entries:
        out[entry["label"]] = entry["reason"]
    return out


def apply_excludes(
    identities: list, excludes: dict, excludes_path: Path, catalog_name: str
) -> list | None:
    """``identities`` without every label ``excludes`` names, or None (after
    printing the offending label(s) to stderr) when ``excludes`` names a label
    ``identities`` does not carry: a stale exclude must mean the instance it
    names already dropped out some other way, and a plain dict lookup would
    silently stop applying the exclusion instead of surfacing that."""
    labels = {identity_label(i) for i in identities}
    stale = set(excludes) - labels
    if stale:
        print(
            f"{excludes_path}: exclude(s) not in {catalog_name}: {sorted(stale)}",
            file=sys.stderr,
        )
        return None
    kept = []
    for identity in identities:
        reason = excludes.get(identity_label(identity))
        if reason is None:
            kept.append(identity)
        else:
            print(f"excluded {identity_label(identity)}: {reason}", file=sys.stderr)
    return kept


def stable_hash(identity: dict) -> str:
    from benchmarks.common.kernel_cache import KernelIdentity

    return KernelIdentity.from_dict(identity).stable_hash()


def identity_label(identity: dict) -> str:
    from benchmarks.common.kernel_cache import KernelIdentity

    return KernelIdentity.from_dict(identity).label()


def compact_identity(identity: dict) -> dict:
    """``identity`` without the fields that hold their ``KernelIdentity`` default.

    A field with no plain default is always kept: the required ones, and
    ``llvm_flavor``, whose default is this process's flavor and which
    ``KernelIdentity.from_dict`` fills with ``""`` when it is missing. A value
    equal to its default but of another type (``0`` for ``False``) is kept, so
    ``from_dict`` rebuilds exactly the identity, and its ``stable_hash()``.
    """
    from benchmarks.common.kernel_cache import KernelIdentity

    compact = {}
    for field in dataclasses.fields(KernelIdentity):
        value = identity[field.name]
        default = field.default
        if (
            default is dataclasses.MISSING
            or type(value) is not type(default)
            or value != default
        ):
            compact[field.name] = value
    return compact


def write_instances(path: Path, identities: list) -> None:
    """A JSON list, one :func:`compact_identity` per line.

    ``newline=""`` so the file is LF-terminated on every platform: the committed
    lists are LF, and ``write_text``'s default universal-newlines translation would
    otherwise turn each ``\\n`` into ``\\r\\n`` on Windows, byte-diffing from them."""
    body = ",\n".join(
        json.dumps(compact_identity(i), separators=(",", ":")) for i in identities
    )
    path.write_text(f"[\n{body}\n]\n", newline="")


# ---------------------------------------------------------------------------
# Generator config
# ---------------------------------------------------------------------------


def variant(name: str, order: tuple, specs: list) -> dict:
    """One config variant: a shape per spec, each field of ``order`` its metadata."""
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
    """The generator's slug for ``engine_name`` (``hipkernel:Gfx950ConvDgradAot`` ->
    ``gfx950_conv_dgrad_aot``): :attr:`EngineSpec.slug`."""
    return EngineSpec(name=engine_name).slug


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


# ---------------------------------------------------------------------------
# Kernel ids
# ---------------------------------------------------------------------------


def kernel_uuid(instance_id: str, engine_name: str, product_engine: str) -> str:
    """The kernel UUID an instance is shipped under by ``engine_name``.

    The shipped engine's ids (``product_engine``'s) are uuid5 of the instance_id.
    Any other engine derives them under a namespace of its own name, so an instance
    it shares with the shipped catalog never carries the same kernel id into one
    process twice.
    """
    namespace = INSTANCE_UUID_NAMESPACE
    if engine_name != product_engine:
        namespace = uuid.uuid5(INSTANCE_UUID_NAMESPACE, engine_name)
    return str(uuid.uuid5(namespace, instance_id))


def stabilize_ids(kdp_path: Path, engine_name: str, product_engine: str) -> int:
    """Re-derive every kernel UUID of one KDP from its instance_id; returns the count.

    A kernel id found anywhere else in the file is referenced, and is refused.
    Every UUID in the text is counted in one pass and all kernel ids are then
    substituted in another, so the cost stays linear in the size of the file.
    ``product_engine`` is the shipped engine whose ids are plain uuid5 of the
    instance_id (:func:`kernel_uuid`).
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
        new_ids[old_id] = kernel_uuid(
            kernel["metadata"]["instance_id"], engine_name, product_engine
        )
    kdp_path.write_text(
        _UUID.sub(lambda match: new_ids.get(match.group(0), match.group(0)), text)
    )
    return len(kernels)


def _kernels(node):
    if isinstance(node, dict):
        # A multi-launch (split-C) kernel carries ``sources`` instead.
        source = node.get("kernel_source", node.get("sources"))
        if isinstance(source, dict) and "metadata" in node:
            yield node
            return
        for value in node.values():
            yield from _kernels(value)
    elif isinstance(node, list):
        for value in node:
            yield from _kernels(value)
