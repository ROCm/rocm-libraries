#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Resolve the newest Quartz-selected ROCm nightly for a platform and GPU target.

Reads status documents published by https://github.com/ROCm/quartz under
``nightly/`` (walking back from ``nightly/latest.json`` up to ``--max-age-days``)
and selects the newest build where:

1. ``schema_version`` has major version 2.
2. ``summary.<platform>.rocm.build.status`` is ``"success"``.
3. For concrete targets / families (anything other than ``multiarch``), the
   Quartz architecture for ``--gpu-target`` appears in
   ``summary.<platform>.architectures``.
4. The requested artifact URL (``wheels`` or ``tarballs``) is an HTTPS URL on
   an allowlisted ROCm artifact host.

Selection preference across the lookback window:

1. Newest nightly where ``pipelines.rocm.test.<platform>.<arch>.status == "success"``
   (or immediately for ``multiarch``).
2. Otherwise, newest nightly where that architecture has no test entry
   (``"untested"``, for example ``gfx950-dcgpu`` on Linux).
3. Otherwise, when ``--allow-build-only-fallback`` is passed, newest nightly
   satisfying (1)-(4) regardless of test outcome (with a warning).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import http.client
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

QUARTZ_NIGHTLY_BASE_URL = "https://raw.githubusercontent.com/ROCm/quartz/main/nightly/"
SUPPORTED_SCHEMA_MAJOR = 2
DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_TIMEOUT_SECS = 20
DEFAULT_FETCH_RETRIES = 2

_ALLOWED_ARTIFACT_HOSTS = frozenset(
    {
        "nightly.repo.amd.com",
        "rocm.nightlies.amd.com",
    }
)

_DATE_PREFIX_RE = re.compile(r"^(\d{8})(?:/|$)")
_SAFE_VERSION_RE = re.compile(r"^[0-9A-Za-z._+-]+$")
_SAFE_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")

# Concrete gfx targets that map to a Quartz family name in status.json.
_TARGET_TO_QUARTZ_ARCH = {
    "gfx90a-dcgpu": "gfx90a",
    "gfx940": "gfx94X-dcgpu",
    "gfx941": "gfx94X-dcgpu",
    "gfx942": "gfx94X-dcgpu",
    "gfx94x": "gfx94X-dcgpu",
    "gfx950": "gfx950-dcgpu",
    "gfx1010": "gfx101X-dgpu",
    "gfx1011": "gfx101X-dgpu",
    "gfx1012": "gfx101X-dgpu",
    "gfx1030": "gfx103X-all",
    "gfx1031": "gfx103X-all",
    "gfx1032": "gfx103X-all",
    "gfx1033": "gfx103X-all",
    "gfx1034": "gfx103X-all",
    "gfx1035": "gfx103X-all",
    "gfx1036": "gfx103X-all",
    "gfx1100": "gfx110X-all",
    "gfx1101": "gfx110X-all",
    "gfx1102": "gfx110X-all",
    "gfx1103": "gfx110X-all",
    "gfx1170": "gfx117X-all",
    "gfx1171": "gfx117X-all",
    "gfx1201": "gfx120X-all",
    "gfx1250": "gfx125X-dcgpu",
    "gfx1251": "gfx125X-dcgpu",
}


class QuartzResolutionError(RuntimeError):
    """Raised when no usable ROCm nightly can be resolved from Quartz."""


@dataclass(frozen=True)
class QuartzSelection:
    rocm_version: str
    wheels_url: str
    tarballs_url: str
    build_date: str
    therock_commit: str
    quartz_arch: str
    test_status: str
    fallback_used: bool


def _get_dict(obj: object, *keys: str) -> dict:
    """Walk nested dicts safely, returning ``{}`` on any missing or null node."""
    cur: object = obj
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key)
    return cur if isinstance(cur, dict) else {}


def gpu_target_to_quartz_arch(gpu_target: str) -> str:
    """Map a GPU target (e.g. gfx942, gfx1100-all) or family name to Quartz's arch key."""
    cleaned = gpu_target.strip()
    if not cleaned:
        raise QuartzResolutionError("gpu_target must not be empty")
    lowered = cleaned.lower()
    if lowered in _TARGET_TO_QUARTZ_ARCH:
        return _TARGET_TO_QUARTZ_ARCH[lowered]
    if lowered.endswith("-all"):
        bare = lowered[:-4]
        if bare in _TARGET_TO_QUARTZ_ARCH:
            return _TARGET_TO_QUARTZ_ARCH[bare]
        if re.fullmatch(r"gfx[0-9a-f]+", bare):
            return bare
    return cleaned


def _validate_artifact_url(url: object) -> Optional[str]:
    if not isinstance(url, str):
        return None
    stripped = url.strip()
    if not stripped or any(ch in stripped for ch in ("\r", "\n", " ", "\t")):
        return None
    parsed = urllib.parse.urlparse(stripped)
    if parsed.scheme != "https":
        return None
    if (parsed.hostname or "").lower() not in _ALLOWED_ARTIFACT_HOSTS:
        return None
    return stripped


def _fetch_text(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECS,
    retries: int = DEFAULT_FETCH_RETRIES,
) -> Optional[str]:
    """Fetch UTF-8 text from an HTTPS URL; return None on HTTP 404."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, headers={"User-Agent": "hipdnn-quartz-resolver/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_exc = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise QuartzResolutionError(f"HTTP {exc.code} fetching {url}") from exc
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            UnicodeDecodeError,
        ) as exc:
            last_exc = exc
            if attempt == retries:
                raise QuartzResolutionError(f"Failed to fetch {url}: {exc}") from exc
        time.sleep(0.5 * (attempt + 1))
    raise QuartzResolutionError(f"Failed to fetch {url}: {last_exc}")


def _parse_latest_date(pointer_text: str) -> datetime.date:
    """Extract the YYYYMMDD build date from Quartz's nightly/latest.json pointer."""
    stripped = pointer_text.strip()
    match = _DATE_PREFIX_RE.match(stripped)
    if match:
        date_str = match.group(1)
    else:
        try:
            doc = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise QuartzResolutionError(
                f"Unrecognized nightly/latest.json payload: {stripped[:120]!r}"
            ) from exc
        if not isinstance(doc, dict) or not isinstance(doc.get("build_date"), str):
            raise QuartzResolutionError(
                "nightly/latest.json JSON payload is missing string 'build_date'"
            )
        date_str = doc["build_date"].strip()
    try:
        return datetime.datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError as exc:
        raise QuartzResolutionError(
            f"Invalid build date {date_str!r} in nightly/latest.json"
        ) from exc


def _evaluate_status(
    doc: object,
    *,
    platform_name: str,
    requested_arch: str,
    artifact: str,
) -> tuple[Optional[QuartzSelection], str]:
    """Evaluate one status.json document."""
    if not isinstance(doc, dict):
        return None, "status document is not a JSON object"

    schema_version = str(doc.get("schema_version", ""))
    major = schema_version.split(".", 1)[0]
    if major != str(SUPPORTED_SCHEMA_MAJOR):
        return None, f"unsupported schema_version={schema_version!r}"

    rocm_version = doc.get("rocm_version")
    if not isinstance(rocm_version, str) or not _SAFE_VERSION_RE.match(rocm_version):
        return None, f"invalid rocm_version={rocm_version!r}"

    build_date = str(doc.get("build_date", "")).strip()
    if not re.match(r"^\d{8}$", build_date):
        return None, f"invalid build_date={build_date!r}"

    therock_commit = str(doc.get("therock_commit", "")).strip()
    if therock_commit and not _SAFE_COMMIT_RE.match(therock_commit):
        return None, f"invalid therock_commit={therock_commit!r}"

    plat_summary = _get_dict(doc, "summary", platform_name)
    if not plat_summary:
        return None, f"missing summary.{platform_name}"

    build_status = _get_dict(plat_summary, "rocm", "build").get("status")
    if build_status != "success":
        return None, f"rocm.build.status={build_status!r}"

    urls = _get_dict(plat_summary, "urls")
    wheels_url = _validate_artifact_url(urls.get("wheels")) or ""
    tarballs_url = _validate_artifact_url(urls.get("tarballs")) or ""
    if artifact == "wheels" and not wheels_url:
        return None, f"missing or disallowed summary.{platform_name}.urls.wheels"
    if artifact == "tarballs" and not tarballs_url:
        return None, f"missing or disallowed summary.{platform_name}.urls.tarballs"

    if requested_arch.lower() == "multiarch":
        return (
            QuartzSelection(
                rocm_version=rocm_version,
                wheels_url=wheels_url,
                tarballs_url=tarballs_url,
                build_date=build_date,
                therock_commit=therock_commit,
                quartz_arch="multiarch",
                test_status="multiarch",
                fallback_used=False,
            ),
            "multiarch",
        )

    archs = plat_summary.get("architectures")
    if not isinstance(archs, list):
        return None, f"missing summary.{platform_name}.architectures"
    arch_lookup = {a.lower(): a for a in archs if isinstance(a, str)}
    canonical_arch = arch_lookup.get(requested_arch.lower())
    if canonical_arch is None:
        return (
            None,
            f"arch {requested_arch!r} not in summary.{platform_name}.architectures",
        )

    test_entries = _get_dict(doc, "pipelines", "rocm", "test", platform_name)
    test_entry = test_entries.get(canonical_arch)
    if test_entry is None:
        test_status = "untested"
    elif isinstance(test_entry, dict):
        test_status = str(test_entry.get("status") or "unknown")
    else:
        test_status = "unknown"

    selection = QuartzSelection(
        rocm_version=rocm_version,
        wheels_url=wheels_url,
        tarballs_url=tarballs_url,
        build_date=build_date,
        therock_commit=therock_commit,
        quartz_arch=canonical_arch,
        test_status=test_status,
        fallback_used=False,
    )
    return selection, f"rocm.test={test_status}"


def resolve_quartz_rocm(
    *,
    platform_name: str,
    gpu_target: str,
    artifact: str = "wheels",
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    allow_build_only_fallback: bool = False,
    base_url: str = QUARTZ_NIGHTLY_BASE_URL,
    fetcher: Optional[Callable[[str], Optional[str]]] = None,
) -> QuartzSelection:
    """Resolve the newest usable ROCm nightly from Quartz.

    Ranks candidates across ``max_age_days`` as:
    1. Newest ``test_status == "success"`` (or ``"multiarch"``), returned immediately.
    2. Newest ``test_status == "untested"`` (if no day in the window passed tests).
    3. Newest build-passing candidate (only when ``allow_build_only_fallback`` is True).
    """
    if platform_name not in ("linux", "windows"):
        raise QuartzResolutionError(
            f"Unsupported platform {platform_name!r}; expected 'linux' or 'windows'"
        )
    if artifact not in ("wheels", "tarballs"):
        raise QuartzResolutionError(
            f"Unsupported artifact {artifact!r}; expected 'wheels' or 'tarballs'"
        )
    if max_age_days < 1:
        raise QuartzResolutionError(f"max_age_days must be >= 1, got {max_age_days}")

    fetch = fetcher or _fetch_text
    base = base_url.rstrip("/") + "/"
    pointer_url = urllib.parse.urljoin(base, "latest.json")
    pointer_text = fetch(pointer_url)
    if not pointer_text:
        raise QuartzResolutionError(f"Quartz pointer not found at {pointer_url}")
    latest_date = _parse_latest_date(pointer_text)
    requested_arch = gpu_target_to_quartz_arch(gpu_target)

    untested_candidate: Optional[QuartzSelection] = None
    build_only_candidate: Optional[QuartzSelection] = None
    diagnostics: list[str] = []

    for offset in range(max_age_days):
        day = (latest_date - datetime.timedelta(days=offset)).strftime("%Y%m%d")
        status_url = urllib.parse.urljoin(base, f"{day}/status.json")
        try:
            body = fetch(status_url)
        except QuartzResolutionError as exc:
            diagnostics.append(f"{day}: fetch error ({exc})")
            continue
        if body is None:
            diagnostics.append(f"{day}: missing status.json")
            continue
        try:
            doc = json.loads(body)
        except json.JSONDecodeError as exc:
            diagnostics.append(f"{day}: invalid JSON ({exc})")
            continue

        selection, reason = _evaluate_status(
            doc,
            platform_name=platform_name,
            requested_arch=requested_arch,
            artifact=artifact,
        )
        if selection is None:
            diagnostics.append(f"{day}: {reason}")
            continue
        if selection.test_status in ("success", "multiarch"):
            return selection
        diagnostics.append(f"{day} ({selection.rocm_version}): {reason}")
        if selection.test_status == "untested":
            if untested_candidate is None:
                untested_candidate = selection
        elif build_only_candidate is None:
            build_only_candidate = QuartzSelection(
                rocm_version=selection.rocm_version,
                wheels_url=selection.wheels_url,
                tarballs_url=selection.tarballs_url,
                build_date=selection.build_date,
                therock_commit=selection.therock_commit,
                quartz_arch=selection.quartz_arch,
                test_status=selection.test_status,
                fallback_used=True,
            )

    if untested_candidate is not None:
        return untested_candidate

    if allow_build_only_fallback and build_only_candidate is not None:
        return build_only_candidate

    summary_lines = "; ".join(diagnostics[:6])
    if len(diagnostics) > 6:
        summary_lines += f"; ... ({len(diagnostics)} days checked)"
    raise QuartzResolutionError(
        f"No passing Quartz ROCm nightly found for {platform_name}/{requested_arch} "
        f"within {max_age_days} day(s) of {latest_date.isoformat()} ({summary_lines})"
    )


def _output_entries(selection: QuartzSelection, artifact: str) -> dict[str, str]:
    entries = {"ROCM_VERSION": selection.rocm_version}
    if artifact == "wheels" and selection.wheels_url:
        entries["ROCM_WHEELS_URL"] = selection.wheels_url
    elif artifact == "tarballs" and selection.tarballs_url:
        entries["ROCM_TARBALLS_URL"] = selection.tarballs_url
    return entries


def _write_github_env(
    env_path: Path, selection: QuartzSelection, artifact: str = "wheels"
) -> None:
    entries = _output_entries(selection, artifact)
    for key, value in entries.items():
        if "\n" in value or "\r" in value:
            raise QuartzResolutionError(f"Refusing to write multiline value for {key}")
    with env_path.open("a", encoding="utf-8") as fh:
        for key, value in entries.items():
            fh.write(f"{key}={value}\n")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        required=True,
        choices=("linux", "windows"),
        help="Target OS platform in Quartz status.json",
    )
    parser.add_argument(
        "--gpu-target",
        required=True,
        help="GPU target, Quartz family, or 'multiarch' (e.g. gfx942, gfx1151, gfx94X-dcgpu)",
    )
    parser.add_argument(
        "--artifact",
        default="wheels",
        choices=("wheels", "tarballs"),
        help="Required published artifact URL in Quartz summary.<platform>.urls (default: wheels)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"Maximum lookback window in days from latest.json (default: {DEFAULT_MAX_AGE_DAYS})",
    )
    parser.add_argument(
        "--allow-build-only-fallback",
        action="store_true",
        help=(
            "If no nightly in the lookback window passed ROCm tests for the "
            "architecture, fall back with a warning to the newest build-passing "
            "nightly."
        ),
    )
    parser.add_argument(
        "--github-env",
        type=Path,
        help="Append ROCM_* variables to this GitHub Actions environment file",
    )
    parser.add_argument(
        "--version-only",
        action="store_true",
        help="Print only the resolved rocm_version to stdout",
    )
    args = parser.parse_args(argv)

    try:
        selection = resolve_quartz_rocm(
            platform_name=args.platform,
            gpu_target=args.gpu_target,
            artifact=args.artifact,
            max_age_days=args.max_age_days,
            allow_build_only_fallback=args.allow_build_only_fallback,
        )
    except QuartzResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if selection.fallback_used:
        msg = (
            f"No Quartz nightly passed ROCm tests for {args.platform}/{selection.quartz_arch} "
            f"in the last {args.max_age_days} day(s); falling back to build-passing nightly "
            f"{selection.rocm_version} (test status: {selection.test_status})."
        )
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::warning::{msg}", file=sys.stderr)
        else:
            print(f"WARNING: {msg}", file=sys.stderr)
    else:
        print(
            f"Quartz selected {selection.rocm_version} ({args.platform}/{selection.quartz_arch}, "
            f"rocm.test={selection.test_status}, therock={selection.therock_commit[:12]})",
            file=sys.stderr,
        )

    if args.github_env:
        _write_github_env(args.github_env, selection, args.artifact)

    if args.version_only:
        print(selection.rocm_version)
    else:
        for key, value in _output_entries(selection, args.artifact).items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
