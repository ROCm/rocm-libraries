# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fail CI when the installed ROCm nightly wheels are too old.

Multi-arch nightly wheels are versioned like ``7.15.0a20260709`` where the
trailing eight digits after the alpha marker encode the build date
(``YYYYMMDD``). A stalled or broken nightly index would silently keep serving an
old build, so we parse that date from the installed package metadata and fail
when it is older than the allowed age. This turns "the nightly stopped
publishing" from a mystery into a loud CI failure.
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

# Matches an 8-digit YYYYMMDD run; the nightly date is the last such group in a
# version like "7.15.0a20260709".
_DATE_RE = re.compile(r"(\d{8})")


def parse_build_date(pkg_version):
    """Return the build ``date`` encoded in a nightly wheel version, or None."""
    for match in reversed(_DATE_RE.findall(pkg_version)):
        try:
            return datetime.strptime(match, "%Y%m%d").date()
        except ValueError:
            continue
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--package",
        default="rocm-sdk-core",
        help="Installed distribution to inspect (default: rocm-sdk-core)",
    )
    p.add_argument(
        "--max-age-days",
        type=int,
        default=14,
        help="Fail if the wheel build date is older than this many days",
    )
    args = p.parse_args()

    try:
        pkg_version = version(args.package)
    except PackageNotFoundError:
        print(
            f"ERROR: package '{args.package}' is not installed; cannot check "
            "wheel freshness.",
            file=sys.stderr,
        )
        return 1

    build_date = parse_build_date(pkg_version)
    if build_date is None:
        print(
            f"ERROR: could not parse a build date from '{args.package}' version "
            f"'{pkg_version}'.",
            file=sys.stderr,
        )
        return 1

    today = datetime.now(timezone.utc).date()
    age_days = (today - build_date).days
    print(
        f"{args.package}=={pkg_version} built {build_date.isoformat()} "
        f"({age_days} day(s) old, threshold {args.max_age_days})"
    )

    if age_days > args.max_age_days:
        print(
            f"ERROR: ROCm nightly wheels are {age_days} days old, which exceeds "
            f"the {args.max_age_days}-day threshold. The nightly index may be "
            "stale or broken.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
