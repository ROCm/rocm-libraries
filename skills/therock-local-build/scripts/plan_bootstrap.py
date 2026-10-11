#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Plan a partial TheRock rebuild on top of a baseline CI run's artifacts.

Given the artifacts to rebuild, print two shell assignments on stdout (use
``eval "$(plan_bootstrap.py ...)"``):

  EXCLUDE_ARTIFACTS  value for ``artifact_manager.py fetch --stage all
                     --exclude-artifacts``: every artifact that must NOT be
                     bootstrapped as prebuilt.
  REBUILD_ARTIFACTS  value for ``configure_stage.py --artifacts``: the requested
                     artifacts plus same-stage dependencies that cannot be used
                     prebuilt.

``artifact_manager.py fetch --stage S`` only bootstraps artifacts produced by
other stages, so a same-stage dependency (for example hipdnn when rebuilding
hipkernelprovider, both in math-libs) would otherwise be compiled from source.
A same-stage dependency whose baseline run has no ``dev`` archive ships no CMake
package config, so consumers cannot use it prebuilt; it is added to the rebuild
set. Commentary and warnings go to stderr.

Needs a TheRock checkout (BUILD_TOPOLOGY.toml and ``build_tools``) and, unless
``--offline``, network access to the public CI bucket listing. Python standard
library only. Written against TheRock's ``_therock_utils.build_topology``.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BUCKET = "https://therock-ci-artifacts.s3.amazonaws.com"


def err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _has_key(bucket: str, prefix: str) -> bool:
    """True if the bucket listing has at least one key under ``prefix``."""
    if not bucket.startswith("https://"):
        sys.exit(f"--bucket must be an https:// URL, got {bucket!r}")
    query = urllib.parse.urlencode({"prefix": prefix, "max-keys": 1})
    try:
        with urllib.request.urlopen(  # nosec B310
            f"{bucket}/?{query}", timeout=30
        ) as resp:
            body = resp.read()
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"cannot query {bucket}: {e}. Use --offline to skip bucket checks.")
    if b"<Error>" in body:
        sys.exit(f"{bucket} returned an error listing for prefix {prefix!r}")
    return b"<Key>" in body


def transitive_deps(topo, names) -> set[str]:
    """All artifact_deps of ``names``, transitively, excluding ``names``."""
    seen: set[str] = set()
    todo = list(names)
    while todo:
        for dep in topo.artifacts[todo.pop()].artifact_deps:
            if dep not in seen:
                seen.add(dep)
                todo.append(dep)
    return seen - set(names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--therock", type=Path, required=True, help="TheRock checkout")
    ap.add_argument(
        "--stage", required=True, help="build stage that produces the artifacts"
    )
    ap.add_argument(
        "--rebuild",
        required=True,
        help="comma-separated artifact names to rebuild, e.g. hipkernelprovider",
    )
    ap.add_argument("--run-id", required=True, help="baseline ROCm/TheRock run id")
    ap.add_argument(
        "--platform", default="linux", help="bucket platform prefix (only linux tested)"
    )
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip bucket queries (no baseline or dev-component checks)",
    )
    args = ap.parse_args()

    build_tools = args.therock / "build_tools"
    if not (build_tools / "_therock_utils").is_dir():
        sys.exit(f"{args.therock} does not look like a TheRock checkout")
    sys.path.insert(0, str(build_tools))
    from _therock_utils.build_topology import get_topology  # noqa: E402

    topo = get_topology(args.therock / "BUILD_TOPOLOGY.toml")
    if args.stage not in topo.build_stages:
        sys.exit(f"unknown stage {args.stage!r}; stages: {sorted(topo.build_stages)}")
    rebuild = {a.strip() for a in args.rebuild.split(",") if a.strip()}
    unknown = rebuild - set(topo.artifacts)
    if unknown:
        sys.exit(f"unknown artifact(s): {sorted(unknown)}")

    produced = set(topo.get_produced_artifacts(args.stage))
    foreign = rebuild - produced
    if foreign:
        sys.exit(
            f"{sorted(foreign)} are not produced by stage {args.stage!r}; "
            f"stage {args.stage!r} produces: {sorted(produced)}"
        )
    inbound = set(topo.get_inbound_artifacts(args.stage))
    same_stage_deps = sorted(transitive_deps(topo, rebuild) & produced)

    forced: list[str] = []
    missing_inbound: list[str] = []
    if not args.offline:
        prefix = f"{args.run_id}-{args.platform}/"
        if not _has_key(args.bucket, prefix + "core-hip_"):
            sys.exit(
                f"run {args.run_id} has no artifacts in {args.bucket} for "
                f"{args.platform}. Pick a ROCm/TheRock main run that built "
                "artifacts; runs whose setup skipped the build publish nothing."
            )
        forced = [
            d for d in same_stage_deps if not _has_key(args.bucket, f"{prefix}{d}_dev_")
        ]
        missing_inbound = sorted(
            a for a in inbound if not _has_key(args.bucket, f"{prefix}{a}_")
        )

    final_rebuild = sorted(rebuild | set(forced))
    prebuilt_same_stage = [d for d in same_stage_deps if d not in forced]
    keep = (inbound | set(prebuilt_same_stage)) - set(final_rebuild)
    exclude = sorted(set(topo.artifacts) - keep)

    err(f"# stage {args.stage}; baseline run {args.run_id} ({args.platform})")
    err(f"# rebuild requested: {', '.join(sorted(rebuild))}")
    if forced:
        err(
            "# also rebuilt (no dev archive in the baseline, so no CMake package "
            f"config): {', '.join(forced)}"
        )
    err(f"# bootstrapped, same-stage: {', '.join(prebuilt_same_stage) or '-'}")
    err(f"# bootstrapped, inbound: {len(inbound) - len(missing_inbound)} artifacts")
    if missing_inbound:
        err(
            "# NOTE: inbound artifacts with no archives in the baseline (built from "
            f"source only if an enabled feature needs them): {', '.join(missing_inbound)}"
        )
    print("export EXCLUDE_ARTIFACTS=" + ",".join(exclude))
    print("export REBUILD_ARTIFACTS=" + ",".join(final_rebuild))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
