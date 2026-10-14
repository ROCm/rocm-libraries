# Adapted from dnn-providers/hip-kernel-provider/flydsl/generators/_manifest.py at commit 94ce74f41ed (FlyDSL provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Write ``<arch>/manifest.json``, render ``<arch>.SOURCE.md`` from it, and digest the
generator inputs it records.

Manifests are indented JSON in insertion order, one kernel-argument record per line,
with no paths, hostnames, timestamps or host environment values.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from . import _family as F


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- Generator inputs --------------------------------------------------------------


def input_files(op, triton_dir: Path = F.TRITON_DIR) -> list[Path]:
    """What ``op``'s objects are generated from: every family module of ``tritongen``
    (all of it but ``ops/``), the op contract, the kernel source and the pinned
    requirements. Source files only; caches are never inputs."""
    package = triton_dir / F.PACKAGE_DIR.name
    files = [path for path in package.glob("*.py") if path.is_file()]
    files += [
        package / "ops" / f"{op.OP}.py",
        triton_dir / op.KERNEL_SOURCE_FILE,
        triton_dir / F.REQUIREMENTS.name,
    ]
    return sorted(set(files))


def inputs_digest(op, triton_dir: Path = F.TRITON_DIR) -> str:
    """SHA256 over ``op``'s generator inputs, then its instance rows.

    Each file contributes its path relative to ``triton/``, NUL, its bytes with CRLF
    normalized to LF, NUL; then the rows as sorted-key JSON. A checkout with CRLF text
    does not read as a change.
    """
    digest = hashlib.sha256()
    for path in input_files(op, triton_dir):
        relative = path.relative_to(triton_dir).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    rows = [instance.row() for instance in op.instances()]
    digest.update(json.dumps(rows, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


# --- Manifest ----------------------------------------------------------------------


def dumps(manifest: dict) -> str:
    """Indented JSON in insertion order, each kernel-argument record on one line."""
    tokens: dict[str, str] = {}
    shaped = json.loads(json.dumps(manifest))
    for record in shaped.get("instances", []):
        if not isinstance(record.get("args"), list):
            continue
        compact = []
        for argument in record["args"]:
            token = f"@@arg{len(tokens)}@@"
            tokens[token] = json.dumps(argument, separators=(", ", ": "))
            compact.append(token)
        record["args"] = compact
    text = json.dumps(shaped, indent=2, sort_keys=False)
    for token, value in tokens.items():
        text = text.replace(f'"{token}"', value, 1)
    return text + "\n"


def write_manifest(
    arch_dir: Path, op: str, arch: str, toolchain: dict, records: list
) -> Path:
    """Write ``<arch_dir>/manifest.json`` for one generator run."""
    manifest = {
        "manifest_version": F.MANIFEST_VERSION,
        "arch": arch,
        "op": op,
        "toolchain": toolchain,
        "instances": records,
    }
    path = arch_dir / F.MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(manifest), encoding="utf-8")
    return path


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --- SOURCE.md ---------------------------------------------------------------------


def source_md_path(op_dir: Path, arch: str) -> Path:
    """``<op_dir>/<arch>.SOURCE.md``: beside ``<arch>/``, not in it."""
    return op_dir / f"{arch}{F.SOURCE_MD_SUFFIX}"


# SOURCE.md labels for toolchain keys whose value is not what the key suggests.
_TOOLCHAIN_LABELS = {
    "triton_wheel_sha256": "triton_wheel_sha256 (pinned in requirements.txt; not measured)",
    "libtriton_sha256": "libtriton_sha256 (measured: triton/_C/libtriton.so)",
}


def render_source_md(manifest: dict) -> str:
    """The provenance note for one arch, built from its manifest only."""
    op, arch = manifest["op"], manifest["arch"]
    lines = [
        f"# Triton AOT kernel objects — `{op}`, `{arch}`",
        "",
        f"<!-- Generated from {arch}/manifest.json by triton/tritongen/gen.py. "
        "Do not edit. -->",
        "",
        f"{len(manifest['instances'])} code object(s) in `{arch}/`, compiled from "
        "`triton/kernels_src/` in the hip-kernel-provider tree. Regenerate them, and "
        "this arch's descriptors, from `triton/` (environment: `triton/README.md`):",
        "",
        "```",
        f"<venv>/bin/python -m tritongen.gen --op {op} --arch {arch}",
        "```",
        "",
        "## Toolchain",
        "",
        "| Component | Value |",
        "| --- | --- |",
    ]
    for key, value in manifest.get("toolchain", {}).items():
        label = _TOOLCHAIN_LABELS.get(key, key)
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        lines.append(f"| {label} | `{text}` |")
    lines += [
        "",
        "## Instances",
        "",
        f"Full records: `{arch}/manifest.json`.",
        "",
        "| Instance | Priority | Dynamic LDS (bytes, passed at launch) | SHA256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for record in manifest["instances"]:
        lines.append(
            f"| `{record['name']}` | {record['priority']} | "
            f"{record['dynamic_lds_bytes']} | `{record['sha256'][:16]}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_source_md(op_dir: Path, manifest: dict) -> Path:
    path = source_md_path(op_dir, manifest["arch"])
    path.write_text(render_source_md(manifest), encoding="utf-8")
    return path


# --- Staging -------------------------------------------------------------------------


def new_stage(parent: Path) -> Path:
    """A fresh empty staging directory beside the tree it will replace."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=parent, prefix=".tritongen-stage-"))


def replace_dir(staged: Path, dest: Path) -> None:
    """Swap the staged directory in for ``dest``; ``dest`` is unchanged on failure."""
    if not dest.exists():
        os.replace(staged, dest)
        return
    old = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".tritongen-old-"))
    os.replace(dest, old)
    try:
        os.replace(staged, dest)
    except BaseException:
        os.replace(old, dest)
        raise
    shutil.rmtree(old, ignore_errors=True)
