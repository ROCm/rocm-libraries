#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Vendor AOTriton's flash test suite and verify the copy.

    sync.py --source <aotriton checkout>   copy the files, rewrite vendor.lock.json
    sync.py --check [--no-collect]         verify the copy against vendor.lock.json

``--source`` refuses a checkout with uncommitted changes under the vendored paths, copies into a
temporary directory and replaces ``vendor/`` only once the copy is complete; the lock is written
last. ``--no-collect`` verifies hashes only and skips the collection counts.

``--check`` fails when a vendored file's sha256 differs, when ``aotriton_flash.py`` (replaced by
``shim/aotriton_flash.py``) is present under ``vendor/``, or when ``pytest --collect-only``
reports a different count than the lock records, for the default collection and for
``test_forward.py`` at every ``FOR_RELEASE`` level 0-3. Collection imports torch and the vendored
modules on CPU; it never touches a GPU.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from aotriton_layout import FLASH_TESTS, LOCK, VENDOR, clean_env, SDPA_ROOT
from conformance_contract import FORWARD_TESTS

TESTS_SRC = "modules/flash/tests"
PLUGIN_SRC = "python/pytest-gpu-lease/pytest_gpu_lease"
LICENSE_SRC = "LICENSE"
REPLACED = "modules/flash/tests/aotriton_flash.py"
RELEASES = (0, 1, 2, 3)
TARGETS = {
    "default": FLASH_TESTS,
    "forward": FLASH_TESTS / FORWARD_TESTS,
}
_COLLECTED = re.compile(r"(\d+) tests? collected")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(source):
    """Relative paths (posix) of every file to vendor from an AOTriton checkout."""
    rels = []
    for sub in (TESTS_SRC, PLUGIN_SRC):
        for path in sorted((source / sub).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                rels.append(path.relative_to(source).as_posix())
    rels.append(LICENSE_SRC)
    return [r for r in rels if r != REPLACED]


def vendored_files():
    return sorted(
        p.relative_to(VENDOR).as_posix()
        for p in VENDOR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )


def collect_count(release, target):
    """Number of items ``pytest --collect-only`` reports; raises on any collection error."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--rootdir={SDPA_ROOT}",
            str(TARGETS[target]),
        ],
        env=clean_env(release=release),
        capture_output=True,
        text=True,
    )
    match = _COLLECTED.search(proc.stdout.rsplit("\n\n", 1)[-1])
    if (
        proc.returncode != 0
        or not match
        or " error" in proc.stdout.rsplit("\n\n", 1)[-1]
    ):
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-30:])
        raise RuntimeError(
            f"collection of {target} at FOR_RELEASE={release} failed "
            f"(rc={proc.returncode}):\n{tail}"
        )
    return int(match.group(1))


def collection_counts():
    return {
        str(release): {t: collect_count(release, t) for t in TARGETS}
        for release in RELEASES
    }


def _git(source, *args):
    return subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
    ).stdout


def refuse_dirty(source):
    """Exit unless every vendored path of ``source`` is committed (the pin must describe the bytes)."""
    dirty = _git(
        source,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        TESTS_SRC,
        PLUGIN_SRC,
        LICENSE_SRC,
    )
    if dirty:
        sys.exit(
            f"refusing {source}: uncommitted changes under the vendored paths:\n{dirty}"
        )


def copy(source):
    refuse_dirty(source)
    pin = _git(source, "rev-parse", "HEAD").strip()
    rels = source_files(source)
    staging = Path(tempfile.mkdtemp(prefix=".vendor-new-", dir=VENDOR.parent))
    backup = staging.with_name(staging.name + "-old")
    swapped = False
    try:
        for rel in rels:
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, dest)
        files = {rel: sha256(staging / rel) for rel in rels}
        if VENDOR.exists():
            VENDOR.rename(backup)
        staging.rename(VENDOR)
        swapped = True
        lock = {
            "pin": pin,
            "replaced": REPLACED,
            "files": files,
            "collection": collection_counts(),
        }
        LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    except BaseException:
        if swapped:
            shutil.rmtree(VENDOR, ignore_errors=True)
        if backup.exists():
            backup.rename(VENDOR)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
    print(f"vendored {len(rels)} files at {pin}; lock written to {LOCK}")


def check(collect=True):
    lock = json.loads(LOCK.read_text())
    problems = []
    if (VENDOR / lock["replaced"]).exists():
        problems.append(f"replaced file present: {lock['replaced']}")
    want, have = set(lock["files"]), set(vendored_files())
    problems += [f"missing: {r}" for r in sorted(want - have)]
    problems += [f"unexpected: {r}" for r in sorted(have - want - {lock["replaced"]})]
    for rel in sorted(want & have):
        if sha256(VENDOR / rel) != lock["files"][rel]:
            problems.append(f"sha256 differs: {rel}")
    if collect and not problems:
        for release, targets in lock["collection"].items():
            for target, expected in targets.items():
                got = collect_count(int(release), target)
                if got != expected:
                    problems.append(
                        f"collection count {target} FOR_RELEASE={release}: {got} != {expected}"
                    )
    for p in problems:
        print(f"FAIL {p}")
    if not problems:
        print(
            f"OK: {len(want)} files at {lock['pin']}"
            + ("; counts match" if collect else "")
        )
    return 1 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=lambda p: Path(p).resolve())
    group.add_argument("--check", action="store_true")
    parser.add_argument(
        "--no-collect", action="store_true", help="--check without counting"
    )
    args = parser.parse_args(argv)
    if args.check:
        return check(collect=not args.no_collect)
    copy(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
