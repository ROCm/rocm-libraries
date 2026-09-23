#!/usr/bin/env python3
"""Write a content digest of the rocke wheels, but only when it changes.

`ROCKE_WHEEL_VERSION` is pinned at "0.1.0" and never bumps, so the wheel
filenames are constant and mtime is the wrong staleness trigger: `pip wheel`
rewrites both files on every build, so an mtime-keyed dependency would recompile
every kernel for every arch on every build even when the wheels are byte-identical.

This writes a digest over the wheels' *contents* and -- crucially -- leaves the
stamp file untouched when the digest has not changed. CMake and Ninja key on
mtime, so an unconditional write would reintroduce exactly the churn this exists
to prevent. Ninja's `restat = 1` (which CMake emits for add_custom_command
OUTPUT edges) then observes the unchanged mtime and prunes the downstream
packaging step.

Exit code is 0 whether or not the stamp was rewritten; "no change" is a normal
outcome, not a failure. A missing wheel IS a failure: it means the caller wired
the dependency wrong, and silently digesting nothing would make the staleness
chain inert in the one case it must not be.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def digest_wheels(wheels: list[Path]) -> str:
    """Order-independent content digest over the wheel set.

    Each wheel contributes its name and its content hash, sorted by name, so the
    digest is stable regardless of the order the caller passes them in and
    changes if any wheel's bytes change.
    """
    h = hashlib.sha256()
    for wheel in sorted(wheels, key=lambda p: p.name):
        h.update(wheel.name.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(wheel.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hkp_wheel_digest",
        description="Conditionally stamp a content digest of the rocke wheels.",
    )
    ap.add_argument("--stamp", required=True, help="Digest stamp file to maintain.")
    ap.add_argument(
        "--wheel",
        action="append",
        default=[],
        required=True,
        help="A wheel to digest; repeatable.",
    )
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    wheels = [Path(w) for w in args.wheel]
    missing = [str(w) for w in wheels if not w.is_file()]
    if missing:
        print(
            "hkp_wheel_digest: wheel(s) not found: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    stamp = Path(args.stamp)
    new = digest_wheels(wheels)
    old = stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else None

    if old == new:
        # Deliberately do NOT touch the file: an unchanged mtime is what lets
        # Ninja's restat prune the downstream pack step.
        print(f"hkp_wheel_digest: unchanged ({new[:12]})")
        return 0

    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(new + "\n", encoding="utf-8")
    print(f"hkp_wheel_digest: {'updated' if old else 'created'} ({new[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
