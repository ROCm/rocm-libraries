# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read and write skinny GEMM sweep result documents (YAML or msgpack)."""

from __future__ import annotations

import os
from typing import Any

import yaml

try:
    import msgpack
except (
    ImportError
):  # pragma: no cover - optional at import, required for msgpack output
    msgpack = None

FORMAT_YAML = "yaml"
FORMAT_MSGPACK = "msgpack"
MSGPACK_MAGIC = b"SGSW\x01"

YAML_HEADER = """\
# Skinny GEMM sweep results written by sweep_skinny_gemm.py.
# Times are microseconds: timeUs is the refined minimum of gemm (+ reduce for
# split-K), screeningUs the minimum of the screening pass. verified is full (the
# whole result was compared with a dense host reference), sampled(N) (N output
# elements were compared, each over all its splits), failed (one-shot verify
# mismatch), or no (--no-verify). In refine+top mode a failing winner is dropped;
# in one-shot every timed configuration stays listed. phaseSeconds
# splits the wall time of a case on its GPU.
"""


def output_format(path: str, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    lower = path.lower()
    if lower.endswith(".msgpack") or lower.endswith(".mpack"):
        return FORMAT_MSGPACK
    return FORMAT_YAML


def read_results(path: str) -> dict[str, Any]:
    """Loads a sweep results document from YAML or msgpack."""
    with open(path, "rb") as stream:
        data = stream.read()
    if data.startswith(MSGPACK_MAGIC):
        if msgpack is None:
            raise RuntimeError("msgpack is not installed; pip install msgpack")
        return msgpack.unpackb(data[len(MSGPACK_MAGIC) :], raw=False)
    text = data.decode("utf-8")
    if text.startswith("#"):
        text = text[text.find("\n") + 1 :]
    return yaml.safe_load(text)


def write_results(path: str, document: dict[str, Any], fmt: str) -> None:
    """Writes a sweep document atomically."""
    temporary = f"{path}.{os.getpid()}.tmp"
    if fmt == FORMAT_MSGPACK:
        if msgpack is None:
            raise RuntimeError("msgpack is not installed; pip install msgpack")
        meta = dict(document.get("meta") or {})
        meta["outputFormat"] = FORMAT_MSGPACK
        document = {**document, "meta": meta}
        packed = MSGPACK_MAGIC + msgpack.packb(document, use_bin_type=True)
        with open(temporary, "wb") as stream:
            stream.write(packed)
    else:
        meta = dict(document.get("meta") or {})
        meta["outputFormat"] = FORMAT_YAML
        document = {**document, "meta": meta}
        with open(temporary, "w") as stream:
            stream.write(YAML_HEADER)
            yaml.safe_dump(
                document,
                stream,
                sort_keys=False,
                width=1000,
                default_flow_style=None,
            )
    os.replace(temporary, path)
