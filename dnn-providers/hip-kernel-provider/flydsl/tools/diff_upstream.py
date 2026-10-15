# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Diff the vendored FlyDSL kernel sources against an upstream checkout.

    python tools/diff_upstream.py --upstream /path/to/FlyDSL

``kernels_src/`` is a vendored copy of a few FlyDSL kernel modules carrying
deliberate local modifications, each recorded in the copied file's own header.
This tool answers the question those headers cannot: *has anything else moved?*
-- either a local edit nobody wrote down, or an upstream re-vendor that did not
update the pins in ``generators/_flydsl_env.py``.

**Both sides are normalized with black before diffing, and that is a
requirement rather than a nicety.** The repo's pre-commit hook reformats our
copy (``.pre-commit-config.yaml``: ``psf/black`` 25.12.0, and hipDNN is opted
*in*), so an un-normalized diff reports a whole-file delta on every vendored
file and buries the real semantic drift inside it. A tool that produces that
output would be worse than no tool: it reads as "everything changed" and gets
ignored. If black is unavailable we refuse rather than emit it.

Normalization is symmetric, so the exact formatting settings do not matter to
the result -- only that both sides receive the same ones. No black config
applies to this tree, so the defaults are what the hook uses.

**A file showing a diff is the expected output, not a failure.** The vendored
copy carries deliberate modifications -- recorded in its own header -- so it
differs from upstream by construction, and the provenance header itself always
shows as an addition. This tool is a review aid: it shows you what to read, and
you decide whether each hunk is one of the recorded modifications or something
nobody wrote down. Exit status therefore reports whether the *comparison* held
together, not whether files matched: 0 when every vendored file was compared
against a real upstream counterpart, 1 when one had none (wrong checkout, or a
file that moved upstream), and 2 when the comparison could not be run at all.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
PROVIDER_DIR = TOOLS_DIR.parent
KERNELS_SRC_DIR = PROVIDER_DIR / "kernels_src"

# Single source of truth for the pins: the generators assert against these, so
# reading them here keeps the differ and the compiler agreeing about which
# upstream commit the checked-in objects were built from.
sys.path.insert(0, str(PROVIDER_DIR))
from generators._flydsl_env import (  # noqa: E402
    AITER_KERNELS_COMMIT,
    FLYDSL_KERNELS_COMMIT,
    FLYDSL_KERNELS_DESCRIBE,
)

# The upstreams a vendored file can come from, keyed by a substring of the URL its
# provenance header records. A file whose header names no URL is a FlyDSL file at its
# own relative path -- the package `__init__` modules carry no provenance block.
ORIGINS = {"aiter": "aiter", "FlyDSL": "flydsl"}
_HEADER_LINES = 20


class DiffError(RuntimeError):
    """The comparison could not be performed."""


def origin_of(relative: Path) -> tuple[str, Path]:
    """Which upstream @p relative was vendored from, and its path there.

    Read from the file's own provenance header (``# Vendored from: <url>`` and
    ``#   path: <path>``), so the record that justifies a modification is also the one
    that says where to diff it.
    """
    lines = (KERNELS_SRC_DIR / relative).read_text(encoding="utf-8").splitlines()
    url = path = None
    for line in lines[:_HEADER_LINES]:
        text = line.lstrip("#").strip()
        if text.startswith("Vendored from:"):
            url = text.split(":", 1)[1].strip()
        elif text.startswith("path:") and url is not None:
            path = text.split(":", 1)[1].strip()
    if url is None:
        return "flydsl", relative
    for marker, origin in ORIGINS.items():
        if marker in url:
            return origin, Path(path) if path else relative
    raise DiffError(
        f"{relative}: vendored from {url}, which no --upstream option covers"
    )


def black_command(explicit: str | None) -> list[str]:
    """Resolve an invocable black, preferring an explicitly named one.

    Falls back to ``python -m black`` because the regeneration venv described in
    REGEN.md holds flydsl and torch but has no reason to carry a formatter,
    while the developer's environment generally does.
    """
    if explicit:
        return [explicit]

    found = shutil.which("black")
    if found:
        return [found]

    probe = subprocess.run(
        [sys.executable, "-m", "black", "--version"],
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "black"]

    raise DiffError(
        "black is not available, and diffing without it reports a whole-file "
        "delta on every vendored file (the repo's pre-commit hook reformats our "
        "copy). Install it -- `pip install black==25.12.0`, matching "
        "`.pre-commit-config.yaml` -- or pass --black."
    )


def normalize(command: list[str], source: str, label: str, line_length: int) -> str:
    """Return `source` as black would format it, without touching any file."""
    result = subprocess.run(
        [*command, "--quiet", f"--line-length={line_length}", "-"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DiffError(
            f"black failed on {label}: {result.stderr.strip() or 'no diagnostic'}"
        )
    return result.stdout


def _not_vendored(path: Path) -> bool:
    """A module written for this provider says so in its header; it has no upstream."""
    lines = path.read_text(encoding="utf-8").splitlines()[:_HEADER_LINES]
    return any(line.lstrip("#").strip().startswith("Not vendored") for line in lines)


def vendored_files() -> list[Path]:
    """Every vendored Python module, as paths relative to ``kernels_src/``."""
    if not KERNELS_SRC_DIR.is_dir():
        raise DiffError(f"vendored sources missing: {KERNELS_SRC_DIR}")
    return sorted(
        path.relative_to(KERNELS_SRC_DIR)
        for path in KERNELS_SRC_DIR.rglob("*.py")
        if "__pycache__" not in path.parts and not _not_vendored(path)
    )


def upstream_head(upstream: Path) -> str | None:
    """The checkout's HEAD, or None when it is not a git worktree."""
    result = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def check_commit(upstream: Path, pinned: str, label: str, require: bool) -> bool:
    """Report the checkout's commit against the pin. True when it is safe to go on."""
    head = upstream_head(upstream)
    if head is None:
        message = f"{upstream} is not a git checkout, so its commit cannot be confirmed"
    elif head == pinned:
        print(f"{label} at the pinned commit {head[:12]}")
        return True
    else:
        message = (
            f"{label} is at {head[:12]}, but the vendored sources were taken from "
            f"{pinned[:12]}. Differences below mix local modifications with "
            "upstream's own movement."
        )

    if require:
        raise DiffError(message)
    print(f"warning: {message}", file=sys.stderr)
    return True


def compare(
    upstreams: dict[str, Path | None],
    command: list[str],
    line_length: int,
    context: int,
) -> tuple[int, int, int]:
    """Diff every vendored file against its upstream. Returns (differing, missing,
    skipped), where skipped counts files whose upstream checkout was not supplied."""
    differing = missing = skipped = 0

    for relative in vendored_files():
        ours = KERNELS_SRC_DIR / relative
        origin, upstream_path = origin_of(relative)
        upstream = upstreams.get(origin)
        if upstream is None:
            print(f"=== {relative}: from {origin}; pass --{origin} to compare it")
            skipped += 1
            continue
        theirs = upstream / upstream_path

        if not theirs.is_file():
            print(f"=== {relative}: no upstream counterpart at {theirs}")
            missing += 1
            continue

        ours_text = normalize(
            command,
            ours.read_text(encoding="utf-8"),
            f"vendored {relative}",
            line_length,
        )
        theirs_text = normalize(
            command,
            theirs.read_text(encoding="utf-8"),
            f"{origin} {upstream_path}",
            line_length,
        )

        if ours_text == theirs_text:
            continue

        differing += 1
        diff = difflib.unified_diff(
            theirs_text.splitlines(keepends=True),
            ours_text.splitlines(keepends=True),
            fromfile=f"{origin}/{upstream_path}",
            tofile=f"kernels_src/{relative}",
            n=context,
        )
        print(f"=== {relative}")
        sys.stdout.writelines(diff)

    return differing, missing, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--upstream",
        type=Path,
        required=True,
        help="FlyDSL checkout to compare against; its kernels/ must mirror kernels_src/",
    )
    parser.add_argument(
        "--aiter",
        type=Path,
        default=None,
        help="AITER checkout, for the files whose header records AITER as their origin "
        "(default: report them as not compared)",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="fail unless the checkout sits at the pinned commit (default: warn and continue)",
    )
    parser.add_argument(
        "--black",
        default=None,
        help="black executable to normalize with (default: PATH, then python -m black)",
    )
    parser.add_argument(
        "--line-length",
        type=int,
        default=88,
        help="black line length, applied to both sides (default: 88, black's own default)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=3,
        help="unified-diff context lines (default: 3)",
    )
    args = parser.parse_args(argv)

    try:
        if not args.upstream.is_dir():
            raise DiffError(f"upstream checkout not found: {args.upstream}")
        if args.aiter is not None and not args.aiter.is_dir():
            raise DiffError(f"AITER checkout not found: {args.aiter}")
        command = black_command(args.black)
        check_commit(
            args.upstream,
            FLYDSL_KERNELS_COMMIT,
            f"FlyDSL ({FLYDSL_KERNELS_DESCRIBE})",
            args.commit,
        )
        if args.aiter is not None:
            check_commit(args.aiter, AITER_KERNELS_COMMIT, "AITER", args.commit)
        differing, missing, skipped = compare(
            {"flydsl": args.upstream, "aiter": args.aiter},
            command,
            args.line_length,
            args.context,
        )
    except DiffError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    total = len(vendored_files())
    identical = total - differing - missing - skipped
    print(
        f"\n{identical}/{total} vendored files identical to upstream after "
        f"normalization; {differing} differ, {missing} absent upstream, "
        f"{skipped} not compared"
    )
    if differing:
        print(
            "Differences are expected where a file's header records them. Read each "
            "hunk against that header; anything it does not account for is drift."
        )
    # Only a missing counterpart is an error: it means this is not the checkout
    # the vendored tree was taken from, so no hunk below can be trusted.
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
