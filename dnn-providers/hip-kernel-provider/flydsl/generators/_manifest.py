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

import ast
import hashlib
import json
from pathlib import Path

from . import _arch_families as families
from ._ops import OP_KERNEL_DIRS

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

FLYDSL_DIR = Path(__file__).resolve().parent.parent

# Shared by every op: the package root and the kernel helpers every op imports.
# On the generator side an op digests only the modules its gen_<op>.py reaches
# through imports (generator_modules), so a module one op's build uses -- the
# CDNA build module, say -- leaves the other ops current. Three tables are
# digested as the op's or target's own row instead of as files, so a row added
# for one op or family leaves every other op's objects current: the op registry,
# the instance table, and the family table.
# gen_descriptors.py is not an input to the objects, and its own --check
# re-derives every descriptor.
_SHARED_KERNEL_PATHS = ("kernels_src/kernels/__init__.py", "kernels_src/kernels/common")
_ROW_TABLES = ("_ops.py", "_instances.py")
_INPUT_SUFFIXES = (".py", ".json")


def _imports(path: Path) -> list[tuple[int, str, list[str]]]:
    """Every import in @p path, at any depth: (level, module, names)."""
    found = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            found += [(0, alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            found.append((node.level, node.module or "", names))
    return found


def generator_modules(flydsl_dir: Path, op: str) -> list[Path]:
    """The generators/ modules ``gen_<op>.py`` reaches through its imports.

    Function-local imports count; ``generators/__init__.py`` counts, since every
    import of the package runs it. A module only reachable by a dynamic import
    would be missed, which tests/test_inputs_digest.py refuses.
    """
    generators = flydsl_dir / "generators"
    seen: set[Path] = set()
    pending = [generators / f"gen_{op}.py", generators / "__init__.py"]
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for level, module, names in _imports(path):
            if level == 0 and module.startswith("generators."):
                module = module.removeprefix("generators.")
            elif level != 1:
                continue
            if module:
                pending.append(generators / f"{module.split('.')[0]}.py")
            else:
                pending += [generators / f"{name}.py" for name in names]
    return sorted(seen)


def _input_files(flydsl_dir: Path, op: str) -> list[Path]:
    roots = [flydsl_dir / path for path in _SHARED_KERNEL_PATHS]
    roots += [flydsl_dir / "kernels_src" / "kernels" / d for d in OP_KERNEL_DIRS[op]]
    files = [
        path
        for path in generator_modules(flydsl_dir, op)
        if path.name not in _ROW_TABLES
    ]
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

    The op's files, then its registry row, then its instance rows, then the
    target's family row (a concrete arch has none). Recorded in the target's manifest; ``gen_descriptors.py`` compares it with the
    tree, so a source edit without a regeneration fails the build instead of
    shipping objects built from other sources. Line endings are normalized, so a
    checkout with CRLF text does not read as a change.
    """
    if op not in OP_KERNEL_DIRS:
        raise KeyError(
            f"op {op!r} names no kernel directories in _ops.OP_KERNEL_DIRS; "
            "add its row so its objects are tied to their sources"
        )
    from ._instances import instances_for  # noqa: PLC0415

    digest = hashlib.sha256()
    for path in _input_files(flydsl_dir, op):
        relative = path.relative_to(flydsl_dir).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    digest.update(json.dumps(list(OP_KERNEL_DIRS[op])).encode("utf-8") + b"\0")
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
        f"{len(manifest['instances'])} code object(s) under `{arch}/` (stored with DVC, "
        f"pointer `{arch}.dvc`), "
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
