# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit a CI-provenance sidecar JSON for a benchmark run.

``results.json`` self-describes the host/GPU/software environment but lacks the
*CI* identity: which commit, which PR, which workflow run, and which TheRock
artifact pin produced it. This script writes that context to ``provenance.json``
(uploaded alongside results) so an archived result is fully self-contained.

All inputs come from environment variables (populated by the workflow). Missing
values are recorded as ``null`` rather than failing — provenance is best-effort
metadata and must never block the job.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _env(name: str) -> Optional[str]:
    """Return the env var, treating empty string as absent."""
    value = os.environ.get(name)
    return value if value else None


def _pr_number() -> Optional[str]:
    """Resolve the PR number from explicit env or the GITHUB_REF refs/pull/N/merge form."""
    explicit = _env("GH_PR_NUMBER")
    if explicit:
        return explicit
    ref = _env("GITHUB_REF") or ""
    # refs/pull/<n>/merge
    parts = ref.split("/")
    if len(parts) >= 3 and parts[1] == "pull":
        return parts[2]
    return None


def _run_url() -> Optional[str]:
    """Compose the workflow run URL from the standard GitHub Actions env vars."""
    server = _env("GITHUB_SERVER_URL")
    repo = _env("GITHUB_REPOSITORY")
    run_id = _env("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def build_provenance() -> Dict[str, Any]:
    """Assemble the provenance record from the environment."""
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit_sha": _env("GITHUB_SHA"),
        "ref": _env("GITHUB_REF"),
        "pr_number": _pr_number(),
        "repository": _env("GITHUB_REPOSITORY"),
        "run_id": _env("GITHUB_RUN_ID"),
        "run_url": _run_url(),
        "therock_ref": _env("THEROCK_REF"),
        "artifact_run_id": _env("ARTIFACT_RUN_ID"),
        "run_github_repo": _env("RUN_GITHUB_REPO"),
        "amdgpu_family": _env("AMDGPU_FAMILY"),
        "amdgpu_targets": _env("AMDGPU_TARGETS"),
        "graph_glob": _env("GRAPH_GLOB"),
        "warmup": _env("WARMUP"),
        "iters": _env("ITERS"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="provenance.json",
        type=Path,
        help="path to write the provenance JSON (default: provenance.json)",
    )
    args = parser.parse_args(argv)

    provenance = build_provenance()
    args.output.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote provenance to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
