#!/usr/bin/env python3
"""Read and write support claim sidecars as sets of claims.

A sidecar is flattened into one Claim per (case, engine, arch, platform) and
regrouped on render, as SupportClaimWriter.cpp does:

- X.json, a single-graph bundle, has X.support.json:
  {"claims": {ENGINE: {ARCH: [PLATFORM, ...]}}, "version": 1}
  Its claims have case "".
- D/sweep.json, a sweep, has D/support.json:
  {"claims": {ENGINE: [{"cases": [...], "support": {ARCH: [...]}}]},
   "version": 1}
  An engine's cases with equal support share one group; groups are ordered
  by their first case.

Rendered text is in the canonical form verify_support_claims.py enforces.
A sidecar that does not render back to its own text is refused on load, so
a rewrite can never drop anything the flat claims do not capture.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Dict, List, NamedTuple, Set, Tuple

BUNDLE_DIR = "integration-test-bundles"
SCHEMA_VERSION = 1


class Claim(NamedTuple):
    case: str  # "" for a single-graph bundle
    engine: str
    arch: str
    platform: str


class SidecarError(ValueError):
    pass


def sidecar_path(bundle: str) -> PurePosixPath:
    """The sidecar of a bundle path relative to the integration-tests dir."""
    path = PurePosixPath(bundle)
    parts = bundle.split("/")  # PurePosixPath would drop "." and "" segments
    if (
        len(parts) < 3
        or parts[0] != BUNDLE_DIR
        or any(p in ("", ".", "..") for p in parts)
        or path.suffix != ".json"
    ):
        raise SidecarError(f"{bundle}: not a bundle under {BUNDLE_DIR}/")
    if path.name == "sweep.json":
        return path.with_name("support.json")
    return path.with_name(f"{path.stem}.support.json")


def _is_sweep(path: Path) -> bool:
    return path.name == "support.json"


def load(path: Path) -> Set[Claim]:
    """The claims of a sidecar; none if it does not exist."""
    if not path.exists():
        return set()
    return parse(path, path.read_bytes().decode("utf-8"))


def parse(path: Path, text: str) -> Set[Claim]:
    """The claims of TEXT, the content of the sidecar at PATH."""
    try:
        claims = json.loads(text)["claims"]
        if _is_sweep(path):
            cells = [
                (case, engine, group["support"])
                for engine, groups in claims.items()
                for group in groups
                for case in group["cases"]
            ]
        else:
            cells = [("", engine, support) for engine, support in claims.items()]
        result = {
            Claim(case, engine, arch, platform)
            for case, engine, support in cells
            for arch, platforms in support.items()
            for platform in platforms
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SidecarError(f"{path}: unreadable: {exc!r}") from exc
    if render(path, result) != text:
        raise SidecarError(f"{path}: not in the form this writer renders")
    return result


def _support(platforms: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    return {arch: sorted(p) for arch, p in platforms.items()}


def render(path: Path, claims: Set[Claim]) -> str:
    """The canonical text of a sidecar holding exactly these claims."""
    sweep = _is_sweep(path)
    by_engine: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}
    for claim in claims:
        if bool(claim.case) != sweep:
            raise SidecarError(f"{path}: claim {claim} does not fit this sidecar")
        by_case = by_engine.setdefault(claim.engine, {})
        by_case.setdefault(claim.case, {}).setdefault(claim.arch, set())
        by_case[claim.case][claim.arch].add(claim.platform)

    out: Dict[str, object] = {}
    for engine, by_case in by_engine.items():
        if not sweep:
            out[engine] = _support(by_case[""])
            continue
        # Visiting cases in order creates each group at its first case.
        groups: Dict[Tuple, List[str]] = {}
        for case in sorted(by_case):
            key = tuple(sorted((a, tuple(sorted(p))) for a, p in by_case[case].items()))
            groups.setdefault(key, []).append(case)
        out[engine] = [
            {"cases": cases, "support": _support(by_case[cases[0]])}
            for cases in groups.values()
        ]
    data = {"claims": out, "version": SCHEMA_VERSION}
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
