#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Replay the COMGR capability probe across public historical wheel snapshots.

The ROCm SDK wheels are large because they contain the full compiler toolchain.
This tool uses HTTP range requests and extracts only COMGR plus its shared-library
dependencies. Each unique wheel is downloaded/probed once; duplicate weekly
selections reuse that evidence. Results are checkpointed after every row.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

if __package__:
    from . import _probe_support as support
    from . import probe
else:
    import sys

    _PLATFORM = Path(__file__).resolve().parents[2]
    _PYTHON = _PLATFORM / "python"
    if str(_PYTHON) not in sys.path:
        sys.path.insert(0, str(_PYTHON))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compiler_capabilities import _probe_support as support
    from compiler_capabilities import probe

_FORMAT = "rocke-comgr-historical-observations/v3"
_DEFAULT_PROCESSORS = ("gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250")


class _HttpRangeReader(io.RawIOBase):
    """Seekable, block-cached reader over an HTTP range-capable artifact."""

    def __init__(self, url: str, *, block_bytes: int = 8 << 20) -> None:
        super().__init__()
        if block_bytes < 1:
            raise ValueError(
                f"historical replay failed - block_bytes={block_bytes}, expected > 0"
            )
        self._url = url
        self._block_bytes = block_bytes
        self._position = 0
        self._blocks: dict[int, bytes] = {}
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_size = response.headers.get("Content-Length")
        if raw_size is None:
            raise support.ProbeError(
                f"historical replay failed - artifact has no Content-Length: {url}"
            )
        self._size = int(raw_size)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"historical replay failed - invalid seek whence={whence}")
        if position < 0:
            raise ValueError(
                f"historical replay failed - seek position={position}, expected >= 0"
            )
        self._position = position
        return position

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached
        start = index * self._block_bytes
        end = min(self._size - 1, start + self._block_bytes - 1)
        request = urllib.request.Request(
            self._url, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise support.ProbeError(
                    "historical replay failed - artifact server ignored byte range "
                    f"{start}-{end}: {self._url}"
                )
            data = response.read()
        self._blocks[index] = data
        return data

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._position
        remaining = min(size, self._size - self._position)
        pieces = []
        while remaining > 0:
            index, offset = divmod(self._position, self._block_bytes)
            piece = self._block(index)[offset : offset + remaining]
            if not piece:
                break
            pieces.append(piece)
            self._position += len(piece)
            remaining -= len(piece)
        return b"".join(pieces)


def _compiler_members(names: list[str]) -> list[str]:
    """Return the minimal wheel members needed to load its COMGR library."""
    selected = []
    suffixes = (
        "/lib/libamd_comgr.so.3",
        "/lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1",
        "/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1",
    )
    fragments = (
        "/lib/llvm/lib/libLLVM.so.",
        "/lib/llvm/lib/libclang-cpp.so.",
    )
    for name in names:
        if name.endswith(suffixes) or any(fragment in name for fragment in fragments):
            selected.append(name)
    if not any(name.endswith("/lib/libamd_comgr.so.3") for name in selected):
        raise support.ProbeError("historical wheel contains no libamd_comgr.so.3")
    if not any("/lib/llvm/lib/libLLVM.so." in name for name in selected):
        raise support.ProbeError("historical wheel contains no libLLVM shared library")
    if not any("/lib/llvm/lib/libclang-cpp.so." in name for name in selected):
        raise support.ProbeError(
            "historical wheel contains no libclang-cpp shared library"
        )
    return selected


def _extract_wheel_comgr(url: str, destination: Path, *, block_bytes: int) -> Path:
    reader = _HttpRangeReader(url, block_bytes=block_bytes)
    with zipfile.ZipFile(reader) as wheel:
        members = _compiler_members(wheel.namelist())
        for member in members:
            wheel.extract(member, destination)
    matches = list(destination.glob("*/lib/libamd_comgr.so.3"))
    if len(matches) != 1:
        raise support.ProbeError(
            f"historical replay found {len(matches)} COMGR libraries, expected one"
        )
    return matches[0]


def _ar_members(archive: bytes) -> dict[str, bytes]:
    """Read the simple ar container used by Debian packages."""
    if not archive.startswith(b"!<arch>\n"):
        raise support.ProbeError("historical Debian artifact has no ar signature")
    members = {}
    offset = 8
    while offset < len(archive):
        header = archive[offset : offset + 60]
        if len(header) != 60 or header[58:60] != b"`\n":
            raise support.ProbeError(
                "historical Debian artifact has a malformed ar header"
            )
        name = header[:16].decode("ascii").strip().removesuffix("/")
        try:
            size = int(header[48:58].decode("ascii").strip())
        except ValueError as exc:
            raise support.ProbeError(
                "historical Debian artifact has a malformed member size"
            ) from exc
        start = offset + 60
        members[name] = archive[start : start + size]
        offset = start + size + (size % 2)
    return members


def _extract_deb_comgr(url: str, destination: Path) -> Path:
    with urllib.request.urlopen(url, timeout=300) as response:
        members = _ar_members(response.read())
    data_name = next((name for name in members if name.startswith("data.tar")), None)
    if data_name is None:
        raise support.ProbeError("historical Debian artifact has no data archive")
    payload = members[data_name]
    if data_name.endswith(".gz"):
        payload = gzip.decompress(payload)
    elif data_name.endswith(".xz"):
        payload = lzma.decompress(payload)
    elif data_name.endswith(".zst"):
        try:
            from compression import zstd
        except ImportError as exc:
            raise support.ProbeError(
                "historical Debian zstd extraction requires Python 3.14+"
            ) from exc
        payload = zstd.decompress(payload)
    elif not data_name.endswith(".tar"):
        raise support.ProbeError(
            f"historical Debian artifact uses unsupported {data_name!r}"
        )
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as data:
        names = [name for name in data.getnames() if "/lib/libamd_comgr.so" in name]
        if not names:
            raise support.ProbeError(
                "historical Debian artifact contains no COMGR library"
            )
        for name in names:
            data.extract(name, destination, filter="data")
    matches = [
        path
        for path in destination.glob("opt/rocm*/lib/libamd_comgr.so.3")
        if path.is_file()
    ]
    if len(matches) != 1:
        raise support.ProbeError(
            f"historical Debian extraction found {len(matches)} COMGR libraries"
        )
    return matches[0]


def _compact_report(report: dict[str, object]) -> dict[str, object]:
    """Keep auditable behavior while excluding local paths and diagnostics."""
    compiler = report["compiler"]
    compact_capabilities = {}
    for name, result in report["capabilities"].items():
        observations = []
        for entry in result["value"]["observations"]:
            compact = {
                "family": entry["family"],
                "candidate": entry["candidate"],
                "outcome": entry["outcome"],
            }
            if "failed_stage" in entry:
                compact["failed_stage"] = entry["failed_stage"]
            observations.append(compact)
        compact_capabilities[name] = observations
    return {
        "probe": {
            "implementation_sha256": report["probe"]["implementation_sha256"],
            "schema_sha256": report["probe"]["schema_sha256"],
            "processor": report["probe"]["processor"],
            "effective_processor": report["probe"]["effective_processor"],
        },
        "compiler": {
            "sha256": compiler["sha256"],
            "llvm_version": compiler["llvm_version"],
            "version_source": compiler["version_source"],
        },
        "capabilities": compact_capabilities,
    }


def _intern_reports(
    output: dict[str, object], reports: dict[str, dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    """Deduplicate structural vectors while retaining codegen fingerprints."""
    compiler = None
    references = {}
    for processor, report in reports.items():
        if compiler is None:
            compiler = report["compiler"]
        elif report["compiler"] != compiler:
            raise support.ProbeError(
                "one historical artifact reported different COMGR identities"
            )
        probe_info = report["probe"]
        root_probe = output.get("probe")
        if not root_probe:
            root_probe = {
                "implementation_sha256": probe_info["implementation_sha256"],
                "schema_sha256": probe_info["schema_sha256"],
            }
            output["probe"] = root_probe
        if (
            root_probe["implementation_sha256"] != probe_info["implementation_sha256"]
            or root_probe["schema_sha256"] != probe_info["schema_sha256"]
        ):
            raise support.ProbeError(
                "historical reports were produced by different probe/schema sources"
            )
        capabilities = json.loads(json.dumps(report["capabilities"]))
        encoded = json.dumps(
            capabilities, sort_keys=True, separators=(",", ":")
        ).encode()
        vector_id = hashlib.sha256(encoded).hexdigest()
        output.setdefault("vectors", {}).setdefault(vector_id, capabilities)
        references[processor] = {
            "effective_processor": probe_info["effective_processor"],
            "vector": vector_id,
        }
    if compiler is None:
        raise support.ProbeError("historical artifact produced no processor reports")
    return compiler, references


def _load_inputs(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "rocke-comgr-historical-inputs/v1":
        raise support.ProbeError("historical input manifest has an unknown format")
    weekly = document.get("weekly")
    if not isinstance(weekly, list) or not weekly:
        raise support.ProbeError("historical input manifest has no weekly rows")
    ids = [row.get("id") for row in weekly if isinstance(row, dict)]
    if len(ids) != len(weekly) or len(ids) != len(set(ids)):
        raise support.ProbeError("historical weekly identifiers must be unique")
    for group in (weekly, document.get("releases", [])):
        if not isinstance(group, list):
            raise support.ProbeError("historical input groups must be lists")
        for row in group:
            if not isinstance(row.get("artifact_url"), str) or not row[
                "artifact_url"
            ].startswith("https://"):
                raise support.ProbeError(
                    f"historical row {row.get('id')} has no public HTTPS artifact"
                )
            if row.get("kind") not in {"wheel", "deb"}:
                raise support.ProbeError(
                    f"historical row {row.get('id')} has unknown artifact kind"
                )
    return document


def _checkpoint(path: Path, document: dict[str, object]) -> None:
    support._write_json(path, document)


def replay(
    inputs_path: Path,
    output_path: Path,
    *,
    processors: tuple[str, ...],
    jobs: int,
    timeout: int,
    block_bytes: int,
) -> dict[str, object]:
    """Replay every selected snapshot, checkpointing each completed row."""
    inputs = _load_inputs(inputs_path)
    output: dict[str, object] = {
        "format": _FORMAT,
        "public_sources": inputs.get("public_sources", []),
        "processors": list(processors),
        "probe": {},
        "vectors": {},
        "weekly": [],
        "releases": [],
    }
    if output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if prior.get("format") != _FORMAT or prior.get("processors") != list(
            processors
        ):
            raise support.ProbeError(
                "existing historical output has incompatible format or processors"
            )
        expected_probe = {
            "implementation_sha256": probe._implementation_sha256(),
            "schema_sha256": probe._schema_sha256(),
        }
        if prior.get("probe") == expected_probe:
            output = prior
            output.setdefault("releases", [])
            for row in output["weekly"]:
                row.setdefault("kind", "wheel")
    all_output_rows = [*output["weekly"], *output["releases"]]
    completed = {(row["kind"], row["id"]): row for row in all_output_rows}
    by_url = {
        row["artifact_url"]: (row["compiler"], row["reports"])
        for row in all_output_rows
    }

    for group_name in ("weekly", "releases"):
        for source_row in inputs.get(group_name, []):
            identifier = source_row["id"]
            kind = source_row["kind"]
            if (kind, identifier) in completed:
                continue
            url = source_row["artifact_url"]
            interned = by_url.get(url)
            if interned is None:
                with tempfile.TemporaryDirectory(prefix="rocke-comgr-study-") as tmp:
                    root = Path(tmp)
                    if kind == "wheel":
                        comgr_lib = _extract_wheel_comgr(
                            url, root, block_bytes=block_bytes
                        )
                    else:
                        comgr_lib = _extract_deb_comgr(url, root)
                    raw_reports = {}
                    for processor in processors:
                        report, _ = probe.load_or_probe(
                            comgr_lib=comgr_lib,
                            processor=processor,
                            cache_dir=root / "cache" / processor,
                            refresh=True,
                            jobs=jobs,
                            timeout=timeout,
                        )
                        raw_reports[processor] = _compact_report(report)
                compiler, reports = _intern_reports(output, raw_reports)
                by_url[url] = (compiler, reports)
            else:
                compiler, reports = interned
            row = {
                "id": identifier,
                "kind": kind,
                "artifact": source_row["artifact"],
                "index_url": source_row["index_url"],
                "artifact_url": url,
                "compiler": compiler,
                "reports": reports,
            }
            output[group_name].append(row)
            _checkpoint(output_path, output)
    return output


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="replay COMGR capability evidence across public snapshots"
    )
    parser.add_argument("--inputs", type=Path, default=here / "historical_inputs.json")
    parser.add_argument(
        "--output", type=Path, default=here / "historical_observations.json"
    )
    parser.add_argument(
        "--processors", default=",".join(_DEFAULT_PROCESSORS), help="comma-separated"
    )
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--download-block-mib", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    processors = tuple(
        part.strip() for part in args.processors.split(",") if part.strip()
    )
    if not processors:
        raise SystemExit("--processors must name at least one target")
    replay(
        args.inputs,
        args.output,
        processors=processors,
        jobs=args.jobs,
        timeout=args.timeout,
        block_bytes=args.download_block_mib << 20,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
