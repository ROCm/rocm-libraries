# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Where a trained bundle came from, as MLREC_v3 files record it
(`mlrec.PROVENANCE_KEYS`).

The values name the training config, the data, the pipeline's sources, the
time and the launch model, and nothing about where training ran: no host or
user names and no filesystem paths."""
from __future__ import annotations

import datetime
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from . import features as fs

PIPELINE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
_READ_BYTES = 1 << 20


def training_data_hash(files: Iterable[os.PathLike]) -> str:
    """ "sha256:<hex>" over the contents of `files` in the given order, each
    preceded by its size in decimal and a newline, so that the hash depends
    on the bytes and their split into files but not on any path. Reads in
    blocks."""
    h = hashlib.sha256()
    for path in files:
        h.update(f"{os.path.getsize(path)}\n".encode("ascii"))
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(_READ_BYTES), b""):
                h.update(block)
    return "sha256:" + h.hexdigest()


def _git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def source_revision(checkout: Path = PIPELINE_DIR) -> str:
    """The commit of the checkout holding `checkout` (`git rev-parse HEAD`),
    with "-dirty" when tracked files differ from it, or "unknown" outside a
    git checkout."""
    head = _git(["rev-parse", "--verify", "HEAD"], checkout)
    if not head or not re.fullmatch(r"[0-9a-f]{40,64}", head):
        return "unknown"
    status = _git(["status", "--porcelain", "--untracked-files=no"], checkout)
    return head + ("-dirty" if status is None or status else "")


def utc_date(now: Optional[datetime.datetime] = None) -> str:
    """UTC ISO 8601 date and time to the second, e.g. 2026-10-06T21:04:05Z."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def training_provenance(
    cfg: Mapping[str, Any],
    catalog: fs.Catalog,
    chunk_files: Sequence[os.PathLike],
    *,
    checkout: Path = PIPELINE_DIR,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, str]:
    """The provenance of a bundle trained under config `cfg` on the enriched
    CSV `chunk_files` (in the order trained on) with `catalog`: `config_id`
    (when the config has one that is a name of letters, digits, '_', '.' and
    '-', as run_orchestrator requires), `training_data_hash`,
    `source_revision`, `date` and, for a v3 catalog whose launch source has an
    `id`, `launch_model`."""
    out: Dict[str, str] = {}
    config_id = str(cfg.get("config_id") or "") if isinstance(cfg, Mapping) else ""
    if _CONFIG_ID_RE.fullmatch(config_id):
        out["config_id"] = config_id
    out["training_data_hash"] = training_data_hash(chunk_files)
    out["source_revision"] = source_revision(checkout)
    out["date"] = utc_date(now)
    launch = getattr(catalog.launch, "id", "") if catalog.version >= 3 else ""
    if launch:
        out["launch_model"] = str(launch)
    return out
