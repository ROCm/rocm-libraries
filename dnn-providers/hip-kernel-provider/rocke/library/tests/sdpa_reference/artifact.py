# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Package and stage a qualified SDPA bundle without build-time NumPy or HIP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def validate_bundle(bundle: Path, lock: Path) -> None:
    """Authenticate the manifest and every payload file before installation."""
    expected = json.loads(lock.read_text())
    if _digest(bundle / "manifest.json") != expected["manifest_sha256"]:
        raise ValueError("SDPA manifest does not match the pinned lock")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest["schema"] != expected["schema"]
        or manifest["baseline_revision"] != expected["baseline_revision"]
    ):
        raise ValueError("SDPA baseline identity mismatch")
    if manifest["schema"] == 2 and any(
        p.is_file() and p.suffix in (".npz", ".npy") for p in bundle.rglob("*")
    ):
        raise ValueError("generated-input SDPA bundles must not contain tensor files")
    payload = bundle / "payload"
    files = {
        p.relative_to(payload).as_posix(): _digest(p)
        for p in payload.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    if files != manifest["files"]:
        raise ValueError("SDPA bundle payload has missing, modified, or extra files")


def pack(bundle: Path, archive: Path, lock: Path) -> None:
    """Create a reproducible archive without host paths, users, or timestamps."""
    validate_bundle(bundle, lock)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with tarfile.open(
                fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT
            ) as tar:
                for path in sorted(bundle.rglob("*")):
                    if (
                        not path.is_file()
                        or "__pycache__" in path.parts
                        or path.suffix == ".pyc"
                    ):
                        continue
                    info = tarfile.TarInfo(
                        "sdpa_reference_bundle/" + path.relative_to(bundle).as_posix()
                    )
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)


def unpack(archive: Path, output: Path, lock: Path) -> None:
    """Extract only regular bundle files, verify them, then expose the result."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix="sdpa-stage-"
    ) as temporary:
        staged = Path(temporary)
        seen = set()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                    or not name.parts
                    or name.parts[0] != "sdpa_reference_bundle"
                    or "\\" in member.name
                    or ":" in member.name
                    or member.name in seen
                ):
                    raise ValueError(f"invalid SDPA archive member: {member.name}")
                seen.add(member.name)
                target = staged.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        bundle = staged / "sdpa_reference_bundle"
        validate_bundle(bundle, lock)
        if output.exists():
            validate_bundle(output, lock)
        else:
            bundle.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "unpack", "validate"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        validate_bundle(args.bundle, args.lock)
    else:
        if args.archive is None:
            parser.error("--archive is required for pack/unpack")
        if args.command == "pack":
            pack(args.bundle, args.archive, args.lock)
        else:
            unpack(args.archive, args.bundle, args.lock)


if __name__ == "__main__":
    main()
