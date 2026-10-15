# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Provenance records for the checked-in code objects of one op and one arch.

Two artifacts, written by every generator run:

* ``<content>/<op>/<arch>/manifest.json`` -- the machine-readable record, beside
  the objects: the toolchain the objects were built with, a digest of the
  generator inputs, and one row per instance (knobs, priority, symbol, kernarg
  layout, SHA256). The descriptor generator reads it; nothing else is a source
  of truth for what an object is.
* ``<content>/<op>/<arch>.SOURCE.md`` -- the human-readable view of the same
  record, beside ``<arch>/``, following the convention
  ``src/engines/asm_sdpa_engine/asm/asm_kernels/`` established for checked-in
  ``.co``: upstream source, toolchain, and a full SHA256 manifest. It is how a
  change to the objects shows up readably in a review.

A binary is only reviewable through these files, so a *stale* one is worse than
a missing one. ``refresh_source_md`` therefore rebuilds the document from the
manifest on every run rather than patching it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import _arch_families as families

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

FLYDSL_DIR = Path(__file__).resolve().parent.parent

# What one op's objects are generated from. Each op names its kernel directories
# under kernels_src/kernels/; a new op must add itself, and an op missing here is
# an error rather than an empty digest. Ops that share a directory, as rmsnorm and
# rmsnorm_bwd share norm/, and every op through the shared generator modules
# below, go stale together: an edit to any of their files, comments included,
# needs a regeneration of each.
OP_KERNEL_DIRS = {
    "rmsnorm": ("norm",),
    "rmsnorm_bwd": ("norm",),
    "sdpa_rdna": ("attention_rdna",),
    "sdpa_rdna_decode": ("attention_rdna",),
    "sdpa_aiw": ("attention_rdna4",),
}
# Shared by every op: the package root, the kernel helpers every op imports, and
# the generator machinery -- every module under generators/ but the per-op
# generators (gen_<op>.py, counted for their own op only), the instance table,
# which holds every op's rows and is counted as the op's own rows instead, and
# the family table, counted as the target's own row, so adding a family leaves
# every other target's objects current.
# gen_descriptors.py is not an input to the objects, and its own --check
# re-derives every descriptor.
_SHARED_KERNEL_PATHS = ("kernels_src/kernels/__init__.py", "kernels_src/kernels/common")
_INSTANCE_TABLE = "generators/_instances.py"
_FAMILY_TABLE = "generators/arch_families.json"
_INPUT_SUFFIXES = (".py", ".json")


def _input_files(flydsl_dir: Path, op: str) -> list[Path]:
    roots = [flydsl_dir / path for path in _SHARED_KERNEL_PATHS]
    roots += [flydsl_dir / "kernels_src" / "kernels" / d for d in OP_KERNEL_DIRS[op]]
    generators = flydsl_dir / "generators"
    files = [
        path
        for path in generators.iterdir()
        if path.is_file()
        and path.suffix in _INPUT_SUFFIXES
        and not path.name.startswith("gen_")
        and path.relative_to(flydsl_dir).as_posix()
        not in (_INSTANCE_TABLE, _FAMILY_TABLE)
    ]
    files.append(generators / f"gen_{op}.py")
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files += [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in _INPUT_SUFFIXES
            and "__pycache__" not in path.parts
        ]
    return sorted(set(files))


def inputs_digest(op: str, arch: str, flydsl_dir: Path = FLYDSL_DIR) -> str:
    """SHA256 over one op's generator inputs for one target.

    The op's files, then its instance rows, then the target's family row (a
    concrete arch has none). Recorded in the target's manifest; ``gen_descriptors.py`` compares it with the
    tree, so a source edit without a regeneration fails the build instead of
    shipping objects built from other sources. Line endings are normalized, so a
    checkout with CRLF text does not read as a change.
    """
    if op not in OP_KERNEL_DIRS:
        raise KeyError(
            f"op {op!r} names no kernel directories in _manifest.OP_KERNEL_DIRS; "
            "add its row so its objects are tied to their sources"
        )
    from ._instances import instances_for  # noqa: PLC0415

    digest = hashlib.sha256()
    for path in _input_files(flydsl_dir, op):
        relative = path.relative_to(flydsl_dir).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    rows = [instance.to_record() for instance in instances_for(op, arch)]
    digest.update(json.dumps(rows, sort_keys=True).encode("utf-8") + b"\0")
    family = families.table().get(arch) if families.is_generic(arch) else None
    digest.update(json.dumps(family, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_op_manifest(
    op_dir: Path, arch: str, op: str, provenance: dict, records: list[dict]
) -> Path:
    """Write ``<op_dir>/manifest.json`` for one generator run."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "arch": arch,
        "op": op,
        "toolchain": {**provenance, "inputs_sha256": inputs_digest(op, arch)},
        "instances": records,
    }
    path = op_dir / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(manifest) + "\n", encoding="utf-8")
    return path


def _dumps(manifest: dict) -> str:
    """Indented JSON, but each kernel-argument record on one line.

    The argument lists are most of a manifest, and one record per line keeps a
    hundred-instance manifest reviewable and under the repository's size check.
    """
    tokens: dict[str, str] = {}
    shaped = json.loads(json.dumps(manifest))
    for record in shaped.get("instances", []):
        for key in ("args", "merge_args"):
            if not isinstance(record.get(key), list):
                continue
            compact = []
            for argument in record[key]:
                token = f"@@arg{len(tokens)}@@"
                tokens[token] = json.dumps(argument, separators=(", ", ": "))
                compact.append(token)
            record[key] = compact
    text = json.dumps(shaped, indent=2, sort_keys=False)
    for token, value in tokens.items():
        text = text.replace(f'"{token}"', value, 1)
    return text


def source_md_path(op_dir: Path) -> Path:
    """``<content>/<op>/<arch>.SOURCE.md``: beside ``<arch>/``, not in it."""
    return op_dir.parent / f"{op_dir.name}.SOURCE.md"


def refresh_source_md(op_dir: Path) -> Path:
    """Rebuild ``<op_dir>.SOURCE.md`` from the manifest in ``op_dir``."""
    manifest_path = op_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {op_dir}; nothing to describe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    op = manifest["op"]
    arch = manifest["arch"]

    lines = [
        f"# FlyDSL AOT kernel objects — `{op}`, `{arch}`",
        "",
        "<!-- Generated by flydsl/generators/. Do not edit by hand: every generator",
        f"     run rebuilds this file from {arch}/manifest.json. -->",
        "",
        f"{len(manifest['instances'])} code object(s), committed under `{arch}/`, "
        "compiled from the vendored sources under",
        "`flydsl/kernels_src/` in the hip-kernel-provider tree. Regenerate with:",
        "",
        "```",
        f"python -m generators.gen_{op} --arch {arch}",
        "```",
        "",
        "run from `flydsl/`; see `flydsl/REGEN.md` for the environment.",
        "",
    ]
    if families.is_generic(arch):
        lines += [
            f"`{arch}` is an LLVM generic target: these objects run unchanged on "
            + ", ".join(f"`{member}`" for member in families.members(arch))
            + ". The packer ships them into each of those arches' shards.",
            "",
        ]
    lines += [
        "## Toolchain",
        "",
        "The toolchain is part of the artifact. Regenerating against a different FlyDSL",
        "or ROCm version is a *change* to these binaries, not a refresh, even when the",
        "instance list is untouched.",
        "",
        "| Component | Version |",
        "| --- | --- |",
    ]
    lines += [
        f"| {key} | `{value}` |" for key, value in manifest.get("toolchain", {}).items()
    ]
    lines += [
        "",
        "## Instances",
        "",
        "One row per object: the name spells out its knobs, and the SHA256 prefix",
        f"changes when the object does. The full record is `{arch}/manifest.json`.",
        "",
        "| Instance | Priority | VGPRs | LDS | SHA256 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for record in manifest["instances"]:
        lines.append(
            f"| `{record['name']}` | {record['priority']} | {record.get('vgprs', '')} "
            f"| {record.get('lds_bytes', '')} | `{record['sha256'][:16]}` |"
        )
    lines.append("")

    path = source_md_path(op_dir)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
