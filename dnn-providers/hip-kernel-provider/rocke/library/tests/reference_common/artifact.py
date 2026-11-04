# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Package and stage a qualified GPU reference bundle without build-time NumPy or HIP."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


_TESTS = Path(__file__).resolve().parents[1]
# Each tests/<operation>_reference package with an architecture registry is a
# reference operation. CMake discovers the same packages to stage bundles and
# register CTests, and requires a publication list for each of them.
OPERATIONS = tuple(
    sorted(
        registry.parents[1].name.removesuffix("_reference")
        for registry in _TESTS.glob("*_reference/architectures/registry.json")
    )
)


def _check_operation(operation: str) -> None:
    if operation not in OPERATIONS:
        raise ValueError(f"unsupported reference operation: {operation}")


def manifest_operation(operation: str) -> str:
    """Return the id that the operation's bundle manifests declare.

    The id names the qualified contract (e.g. forward-only convolution) and
    need not equal the package name, so each package records it in its
    operation.json.
    """
    _check_operation(operation)
    metadata = _TESTS / f"{operation}_reference" / "operation.json"
    return json.loads(metadata.read_text())["manifest_operation"]


def display_command(arguments: Iterable[str]) -> str:
    """Show a suggested command as plain text, with its arguments as typed.

    Recovery hints are read, never executed, so they follow no one shell's
    escaping rules: POSIX quoting would wrap Windows paths in single quotes,
    which cmd.exe keeps as part of the argument. Only an argument containing
    whitespace is wrapped, in double quotes, which group it in bash, cmd.exe,
    and PowerShell alike. Every other character, such as a backslash or a
    ``$``, is shown as is, even where a shell would expand it.
    """
    return " ".join(
        f'"{argument}"' if any(char.isspace() for char in argument) else argument
        for argument in arguments
    )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def validate_bundle(bundle: Path, lock: Path, *, operation: str) -> None:
    """Authenticate the manifest and every payload file before installation."""
    _check_operation(operation)
    expected = json.loads(lock.read_text())
    if not (bundle / "manifest.json").is_file():
        raise ValueError(f"GPU reference bundle has no manifest.json: {bundle}")
    digest = _digest(bundle / "manifest.json")
    if digest != expected["manifest_sha256"]:
        # Indented detail lines stay intact when CMake formats the message.
        raise ValueError(
            "GPU reference manifest does not match the pinned lock\n"
            f"  manifest.json sha256 {digest[:12]}, {lock.name} pins "
            f"{expected['manifest_sha256'][:12]}"
        )
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (
        manifest["schema"] != expected["schema"]
        or manifest["baseline_revision"] != expected["baseline_revision"]
    ):
        raise ValueError("GPU reference baseline identity mismatch")
    # The deployed SDPA schema-2 manifest predates the operation field and is
    # identified through the selected lock; other operations declare their id.
    declared = manifest.get("operation", "sdpa")
    if declared != manifest_operation(operation):
        raise ValueError(f"bundle operation {declared!r} does not match {operation!r}")
    if manifest["schema"] != 2:
        raise ValueError("only tensor-free schema-2 reference bundles are supported")
    if any(p.is_file() and p.suffix in (".npz", ".npy") for p in bundle.rglob("*")):
        raise ValueError(
            "generated-input GPU reference bundles must not contain tensor files"
        )
    payload = bundle / "payload"
    files = {
        p.relative_to(payload).as_posix(): _digest(p)
        for p in payload.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    if files != manifest["files"]:
        raise ValueError(
            "GPU reference bundle payload has missing, modified, or extra files"
        )


def pack(bundle: Path, archive: Path, lock: Path, *, operation: str) -> None:
    """Create a reproducible archive without host paths, users, or timestamps."""
    validate_bundle(bundle, lock, operation=operation)
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
                        f"{operation}_reference_bundle/"
                        + path.relative_to(bundle).as_posix()
                    )
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)


def _is_published_pair(archive: Path, lock: Path, *, operation: str) -> bool:
    """Whether ``archive`` is a DVC-tracked archive and ``lock`` is its committed lock.

    Fetching the archive again repairs only that pairing. A candidate archive,
    a candidate lock, or another architecture's lock does not belong with the
    archive, and no fetch makes them agree.
    """
    architecture = archive.name.removesuffix(".tar.gz")
    tracked = _TESTS / "reference_bundles" / operation / f"{architecture}.tar.gz"
    committed = (
        _TESTS
        / f"{operation}_reference"
        / "architectures"
        / architecture
        / "baseline_lock.json"
    )
    return (
        tracked.with_name(f"{tracked.name}.dvc").is_file()
        and archive.resolve() == tracked.resolve()
        and lock.resolve() == committed.resolve()
    )


def unpack(archive: Path, output: Path, lock: Path, *, operation: str) -> None:
    """Extract only regular bundle files, verify them, then expose the result."""
    _check_operation(operation)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix="reference-stage-"
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
                    or name.parts[0] != f"{operation}_reference_bundle"
                    or "\\" in member.name
                    or ":" in member.name
                    or member.name in seen
                ):
                    raise ValueError(
                        f"invalid {operation.upper()} archive member: {member.name}"
                    )
                seen.add(member.name)
                target = staged.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
        bundle = staged / f"{operation}_reference_bundle"
        try:
            validate_bundle(bundle, lock, operation=operation)
        except ValueError as error:
            mismatch = f"{archive} is not the archive that {lock} pins"
            if _is_published_pair(archive, lock, operation=operation):
                fetch = display_command(["dvc", "pull", f"{archive}.dvc"])
                mismatch += f"; fetch that archive with: {fetch}"
            raise ValueError(f"{error}\n  {mismatch}") from None
        if output.exists():
            # Unpack never replaces an existing bundle; it must already match.
            try:
                validate_bundle(output, lock, operation=operation)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"{error}\n  {output} is a stale or modified extracted bundle; "
                    "delete it and re-run this unpack command"
                ) from None
        else:
            bundle.rename(output)


def _parser() -> argparse.ArgumentParser:
    """Build the parser for this script and for the unpack commands that hints show."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "unpack", "validate"))
    parser.add_argument(
        "--operation",
        choices=OPERATIONS,
        required=True,
        help="reference operation (required for the common command)",
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    operation = args.operation
    if args.command == "validate":
        validate_bundle(args.bundle, args.lock, operation=operation)
    else:
        if args.archive is None:
            parser.error("--archive is required for pack/unpack")
        if args.command == "pack":
            pack(args.bundle, args.archive, args.lock, operation=operation)
        else:
            unpack(args.archive, args.bundle, args.lock, operation=operation)


if __name__ == "__main__":
    main()
