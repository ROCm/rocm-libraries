# Adapted from dnn-providers/hip-kernel-provider/flydsl/generators/_manifest.py at commit 03911535bd0 (FlyDSL provider)
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Write ``<arch>/manifest.json`` and render ``<arch>.SOURCE.md`` from it.

Manifests are canonical JSON (sorted keys, 2-space indent, trailing newline) with no
paths, hostnames or timestamps.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from . import _contract as C


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dumps(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, indent=2) + "\n"


def write_manifest(arch_dir: Path, arch: str, toolchain: dict, records: list) -> Path:
    """Write ``<arch_dir>/manifest.json`` for one generator run."""
    manifest = {
        "manifest_version": C.MANIFEST_VERSION,
        "arch": arch,
        "op": C.MANIFEST_OP,
        "toolchain": toolchain,
        "instances": records,
    }
    path = arch_dir / C.MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(manifest), encoding="utf-8")
    return path


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_md_path(op_dir: Path, arch: str) -> Path:
    """``<op_dir>/<arch>.SOURCE.md``: beside ``<arch>/``, not in it."""
    return op_dir / f"{arch}.SOURCE.md"


# SOURCE.md labels for toolchain keys whose value is not what the key suggests.
_TOOLCHAIN_LABELS = {
    "triton_wheel_sha256": "triton_wheel_sha256 (pinned in requirements.txt; not measured)",
}


def render_source_md(manifest: dict) -> str:
    """The provenance note for one arch, built from its manifest only."""
    op, arch = manifest["op"], manifest["arch"]
    lines = [
        f"# Triton AOT kernel objects — `{op}`, `{arch}`",
        "",
        f"<!-- Generated from {arch}/manifest.json by triton/generators/gen_{op}.py. "
        "Do not edit. -->",
        "",
        f"{len(manifest['instances'])} code object(s) in `{arch}/`, compiled from "
        "`triton/kernels_src/` in the hip-kernel-provider tree. Regenerate them, and "
        "this arch's descriptors, from `triton/` (environment: `triton/README.md`):",
        "",
        "```",
        f"<venv>/bin/python -m generators.gen_{op} --arch {arch}",
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
            f"| `{record['name']}` | {record['priority']} | {record['shared']} "
            f"| `{record['sha256'][:16]}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_source_md(op_dir: Path, manifest: dict) -> Path:
    path = source_md_path(op_dir, manifest["arch"])
    path.write_text(render_source_md(manifest), encoding="utf-8")
    return path


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
