# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Export immutable committed rocKE sources for offline qualification."""

from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .numeric import payload_digests, write_json

_SOURCE_PREFIX = PurePosixPath("dnn-providers/hip-kernel-provider/rocke")


def snapshot(repository: Path, revision: str, output: Path) -> None:
    """Export committed baseline sources, recording every file's identity."""
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", f"{revision}^{{commit}}"],
        text=True,
    ).strip()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "archive",
                revision,
                str(_SOURCE_PREFIX / "platform/python"),
                str(_SOURCE_PREFIX / "library"),
            ],
            stdout=archive,
            check=True,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                relative = PurePosixPath(member.name).relative_to(_SOURCE_PREFIX)
                if ".." in relative.parts:
                    raise ValueError("invalid source archive path")
                destination = source / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as stream:
                    destination.write_bytes(stream.read())
    write_json(
        output / "snapshot.json",
        {"revision": revision, "files": payload_digests(source)},
    )
    print(f"Exported baseline {revision}", flush=True)
