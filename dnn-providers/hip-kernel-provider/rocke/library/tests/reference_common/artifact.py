# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Package, stage, and list GPU reference bundles without build-time NumPy or HIP.

Tools list the published variants with the published command instead of
building archive paths from the CMake publication lists.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
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


def cohort_digest(rows: Iterable[Mapping[str, object]]) -> str:
    """Digest a cohort from one ``case``/``input_digests``/``tolerance`` row per case.

    Rows are hashed as sorted compact JSON, so neither case order nor manifest
    keys, which may use an earlier id format, change the digest. Outputs and
    kernels are left out, so processors that enroll the same cases share it. A
    lock records it as ``cohort_sha256``; configuration compares it with the
    pinned manifest, and host tests compare it with the adapter's cases, which
    needs no bundle data.
    """
    lines = sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def manifest_cohort_digest(manifest: Mapping) -> str:
    """Digest the cohort that a schema-2 manifest's cases record."""
    return cohort_digest(
        {
            "case": entry["case"],
            "input_digests": entry["input_digests"],
            "tolerance": entry["budget"]["tolerance"],
        }
        for entry in manifest["cases"].values()
    )


class InconsistentLockError(ValueError):
    """A lock that disagrees with itself or its variant, so no archive satisfies it."""


def validate_bundle(
    bundle: Path, lock: Path, *, operation: str, variant: str | None = None
) -> None:
    """Authenticate the manifest and every payload file before installation.

    ``variant`` names the variant whose lock ``lock`` is; a manifest that
    declares its variant must declare that one.
    """
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
    # Manifests qualified before variants predate the variant field and are
    # likewise identified through the selected lock, which is the variant's.
    # Every archive that the lock accepts declares the same variant, so a
    # mismatch is the lock's and no fetch repairs it.
    if variant is not None and manifest.get("variant", variant) != variant:
        raise InconsistentLockError(
            f"bundle variant {manifest['variant']!r} does not match {variant!r}\n"
            f"  {lock.name} pins a bundle qualified for variant "
            f"{manifest['variant']!r}; a variant's lock and archive must come "
            "from its own qualification"
        )
    if manifest["schema"] != 2:
        raise ValueError("only tensor-free schema-2 reference bundles are supported")
    # A candidate lock from an earlier qualification may lack the digest; host
    # tests require it in every committed lock.
    if "cohort_sha256" in expected:
        cohort = manifest_cohort_digest(manifest)
        if cohort != expected["cohort_sha256"]:
            raise InconsistentLockError(
                "GPU reference lock pins a cohort that its manifest does not "
                "record\n"
                f"  manifest.json cases digest {cohort[:12]}, {lock.name} pins "
                f"cohort_sha256 {expected['cohort_sha256'][:12]}\n"
                "  every lock field must come from the qualification-lock.json "
                "written with this bundle; a cohort change needs requalification"
            )
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


def pack(
    bundle: Path,
    archive: Path,
    lock: Path,
    *,
    operation: str,
    variant: str | None = None,
) -> None:
    """Create a reproducible archive without host paths, users, or timestamps.

    A processor's first variant has no archive directory yet, so a bundle that
    validates creates it; one that does not writes nothing.
    """
    validate_bundle(bundle, lock, operation=operation, variant=variant)
    archive.parent.mkdir(parents=True, exist_ok=True)
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
    a candidate lock, or another variant's lock does not belong with the
    archive, and no fetch makes them agree. A variant's archive is
    ``<arch>/<variant>.tar.gz`` and its lock ``<arch>/<variant>/baseline_lock.json``.
    """
    archive = archive.resolve()
    architecture = archive.parent.name
    variant = archive.name.removesuffix(".tar.gz")
    tracked = _TESTS / "reference_bundles" / operation / architecture / archive.name
    committed = (
        _TESTS
        / f"{operation}_reference"
        / "architectures"
        / architecture
        / variant
        / "baseline_lock.json"
    )
    return (
        tracked.with_name(f"{tracked.name}.dvc").is_file()
        and archive == tracked.resolve()
        and lock.resolve() == committed.resolve()
    )


def unpack(
    archive: Path,
    output: Path,
    lock: Path,
    *,
    operation: str,
    variant: str | None = None,
) -> None:
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
            validate_bundle(bundle, lock, operation=operation, variant=variant)
        except InconsistentLockError:
            # The archive has the manifest that the lock pins; no fetch can help.
            raise
        except ValueError as error:
            mismatch = f"{archive} is not the archive that {lock} pins"
            if _is_published_pair(archive, lock, operation=operation):
                fetch = display_command(["dvc", "pull", f"{archive}.dvc"])
                mismatch += f"; fetch that archive with: {fetch}"
            raise ValueError(f"{error}\n  {mismatch}") from None
        if output.exists():
            # Unpack never replaces an existing bundle; it must already match.
            try:
                validate_bundle(output, lock, operation=operation, variant=variant)
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"{error}\n  {output} is a stale or modified extracted bundle; "
                    "delete it and re-run this unpack command"
                ) from None
        else:
            bundle.rename(output)


def published(operation: str | None = None) -> list[dict[str, str]]:
    """List the variants that committed locks publish, by operation and architecture.

    A variant's committed ``<arch>/<variant>/baseline_lock.json`` publishes it
    once its architecture is in the operation's CMake publication list, and
    configuration rejects a lock under any other, so the locks alone list the
    published variants. Each entry names the variant's lock, DVC pointer, and
    archive, which a source checkout tracks and fetches; an installed tree has
    the locks only. Architectures keep ``registry.json`` order and variants
    name order. A lock under an architecture that the registry does not
    enroll, which configuration rejects, is listed after the enrolled ones
    rather than hidden. ``operation`` restricts the listing to one operation.
    """
    if operation is not None:
        _check_operation(operation)
    entries = []
    for name in OPERATIONS if operation is None else (operation,):
        root = _TESTS / f"{name}_reference" / "architectures"
        enrolled = json.loads((root / "registry.json").read_text())
        locked = {lock.parents[1].name for lock in root.glob("*/*/baseline_lock.json")}
        for architecture in enrolled + sorted(locked - set(enrolled)):
            for lock in sorted((root / architecture).glob("*/baseline_lock.json")):
                variant = lock.parent.name
                archive = (
                    _TESTS
                    / "reference_bundles"
                    / name
                    / architecture
                    / f"{variant}.tar.gz"
                )
                entries.append(
                    {
                        "operation": name,
                        "architecture": architecture,
                        "variant": variant,
                        "lock": str(lock),
                        "pointer": f"{archive}.dvc",
                        "archive": str(archive),
                    }
                )
    return entries


def _parser() -> argparse.ArgumentParser:
    """Build the parser for this script and for the unpack commands that hints show."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "unpack", "validate", "published"))
    parser.add_argument(
        "--operation",
        choices=OPERATIONS,
        help="reference operation (required except by published, which it "
        "restricts to one operation)",
    )
    parser.add_argument(
        "--variant",
        help="variant that the lock belongs to; a manifest that declares its "
        "variant must declare this one",
    )
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print published as JSON, with each variant's lock, DVC pointer, "
        "and archive paths",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "published":
        entries = published(args.operation)
        if args.json:
            print(json.dumps(entries, indent=2))
        else:
            for entry in entries:
                print(entry["operation"], entry["architecture"], entry["variant"])
        return
    missing = [
        option
        for option, value in (
            ("--operation", args.operation),
            ("--bundle", args.bundle),
            ("--lock", args.lock),
        )
        if value is None
    ]
    if missing:
        parser.error(
            f"the following arguments are required for {args.command}: "
            + ", ".join(missing)
        )
    operation, variant = args.operation, args.variant
    if args.command == "validate":
        validate_bundle(args.bundle, args.lock, operation=operation, variant=variant)
    else:
        if args.archive is None:
            parser.error("--archive is required for pack/unpack")
        if args.command == "pack":
            pack(
                args.bundle,
                args.archive,
                args.lock,
                operation=operation,
                variant=variant,
            )
        else:
            unpack(
                args.archive,
                args.bundle,
                args.lock,
                operation=operation,
                variant=variant,
            )


if __name__ == "__main__":
    main()
