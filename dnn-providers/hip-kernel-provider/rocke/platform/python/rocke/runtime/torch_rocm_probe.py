# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare this process's mapped ROCm binaries with a fresh torch import.

This is an opt-in diagnostic, not a loader policy or a guarantee of safe
in-process imports or HIP handle ownership. The worker is this file run as a
standalone script so it does not import rocke before importing torch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from types import ModuleType

__all__ = [
    "RocmLibrary",
    "TorchRocmReport",
    "TorchRocmStatus",
    "TorchWorkerStatus",
    "probe_torch_rocm",
]


class TorchRocmStatus(str, Enum):
    """Comparison outcome for the observed binaries."""

    DIFFERENT_BINARIES_OBSERVED = "different_binaries_observed"
    SAME_BINARIES_OBSERVED = "same_binaries_observed"
    INCONCLUSIVE = "inconclusive"

    __str__ = str.__str__


class TorchWorkerStatus(str, Enum):
    """Outcome of the subprocess that imports torch."""

    IMPORTED = "imported"
    TORCH_NOT_INSTALLED = "torch_not_installed"
    TORCH_IMPORT_ERROR = "torch_import_error"
    WORKER_ERROR = "worker_error"
    CRASHED = "crashed"
    TIMEOUT = "timeout"

    __str__ = str.__str__


@cache
def _inventory_helpers() -> ModuleType:
    if __package__:
        from . import runtime_coexistence

        return runtime_coexistence

    # The standalone worker reaches this after importing torch. Load only the
    # sibling module from this installation, without importing rocke or using
    # ambient sys.path to locate the helper.
    path = Path(__file__).with_name("runtime_coexistence.py")
    spec = importlib.util.spec_from_file_location("_rocke_probe_coexistence", path)
    if spec is None or spec.loader is None:
        raise OSError(f"could not locate inventory helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class RocmLibrary:
    """An observed mapped file; its digest is absent when it cannot be verified."""

    kind: str
    path: str
    sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TorchRocmReport:
    """Observed library evidence, never a verdict on HIP instance ownership."""

    status: TorchRocmStatus
    reason: str
    parent_libraries: tuple[RocmLibrary, ...]
    torch_libraries: tuple[RocmLibrary, ...] = ()
    torch_version: str | None = None
    torch_hip_version: str | None = None
    worker_status: TorchWorkerStatus | None = None
    stderr: str = ""
    torch_file: str | None = None


def _fingerprint(
    path: str, kind: str, mapped_identity: tuple[int, int] | None = None
) -> RocmLibrary:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            stat = os.fstat(stream.fileno())
            if mapped_identity is not None and mapped_identity != (
                stat.st_dev,
                stat.st_ino,
            ):
                return RocmLibrary(kind, path, error="mapped file does not match disk")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        return RocmLibrary(kind, path, error=str(exc))
    return RocmLibrary(kind, path, sha256=digest.hexdigest())


def _linux_libraries(lines: Iterable[str]) -> tuple[RocmLibrary, ...]:
    helpers = _inventory_helpers()
    mapped: dict[tuple[str, str, str], RocmLibrary] = {}
    for permissions, dev, inode, pathname in helpers._iter_proc_maps_rows(lines):
        if "x" not in permissions:
            continue  # Exclude ordinary data mappings of files named like libraries.
        path = pathname.removesuffix(" (deleted)")
        kind = helpers._mapped_library_kind(Path(path).name, platform="linux")
        key = pathname, dev, inode
        if not kind or key in mapped:
            continue
        if path != pathname:
            mapped[key] = RocmLibrary(kind, path, error="mapped file was deleted")
            continue
        major, minor = (int(part, 16) for part in dev.split(":"))
        mapped[key] = _fingerprint(path, kind, (os.makedev(major, minor), int(inode)))
    return tuple(sorted(mapped.values(), key=lambda row: (row.kind, row.path)))


def _windows_module_paths() -> Iterable[str]:
    # Enumerate the current process; do not probe or load any ROCm DLL.
    import ctypes
    from ctypes import wintypes as w

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    current = kernel.GetCurrentProcess
    current.argtypes = []
    current.restype = w.HANDLE
    enumerate_modules = kernel.K32EnumProcessModules
    enumerate_modules.argtypes = [
        w.HANDLE,
        ctypes.POINTER(w.HMODULE),
        w.DWORD,
        ctypes.POINTER(w.DWORD),
    ]
    enumerate_modules.restype = w.BOOL
    module_name = kernel.GetModuleFileNameW
    module_name.argtypes = [w.HMODULE, w.LPWSTR, w.DWORD]
    module_name.restype = w.DWORD

    needed = w.DWORD()
    proc = current()
    modules = (w.HMODULE * 128)()
    while True:
        if not enumerate_modules(proc, modules, ctypes.sizeof(modules), needed):
            raise ctypes.WinError(ctypes.get_last_error())
        if needed.value <= ctypes.sizeof(modules):
            break
        modules = (w.HMODULE * (needed.value // ctypes.sizeof(w.HMODULE) + 32))()
    buffer = ctypes.create_unicode_buffer(32768)
    for index in range(needed.value // ctypes.sizeof(w.HMODULE)):
        length = module_name(modules[index], buffer, len(buffer))
        if not length or length >= len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        yield buffer.value


def _snapshot() -> tuple[tuple[RocmLibrary, ...], str | None]:
    try:
        if sys.platform == "linux":
            with open(
                _inventory_helpers()._PROC_MAPS,
                encoding="utf-8",
                errors="surrogateescape",
            ) as maps:
                return _linux_libraries(maps), None
        if sys.platform == "win32":
            classify = _inventory_helpers()._mapped_library_kind
            return (
                tuple(
                    _fingerprint(path, kind)
                    for path in sorted(set(_windows_module_paths()))
                    if (kind := classify(Path(path).name, platform="win32"))
                ),
                None,
            )
    except (OSError, ValueError) as exc:
        return (), str(exc)
    return (), f"module inventory is not supported on {sys.platform}"


def _compare(
    parent: tuple[RocmLibrary, ...], child: tuple[RocmLibrary, ...]
) -> tuple[TorchRocmStatus, str]:
    different, incomplete = [], []
    comparable = 0
    for kind in _inventory_helpers()._MAPPED_LIBRARY_NAMES:
        left, right = (
            [item for item in rows if item.kind == kind] for rows in (parent, child)
        )
        if not left and not right:
            continue
        if (
            len(left) != 1
            or len(right) != 1
            or not left[0].sha256
            or not right[0].sha256
        ):
            incomplete.append(
                f"{kind}: missing, ambiguous, or unreadable binary evidence"
            )
            continue
        comparable += 1
        if left[0].sha256 != right[0].sha256:
            different.append(
                f"{kind}: different binaries ({left[0].path} vs {right[0].path})"
            )
    if different:
        return TorchRocmStatus.DIFFERENT_BINARIES_OBSERVED, "; ".join(
            different + incomplete
        )
    if incomplete or not comparable:
        return (
            TorchRocmStatus.INCONCLUSIVE,
            "; ".join(incomplete) or "no ROCm binaries observed in either process",
        )
    return (
        TorchRocmStatus.SAME_BINARIES_OBSERVED,
        "fingerprints match for the observed library roles",
    )


def _worker(output: Path) -> None:
    # Import only in the throwaway child. Do not initialize devices or query
    # candidate libraries, which would change the mappings being measured.
    # Optional package initialization can surface imports or native load errors.
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        status = (
            TorchWorkerStatus.TORCH_NOT_INSTALLED
            if isinstance(exc, ModuleNotFoundError) and exc.name == "torch"
            else TorchWorkerStatus.TORCH_IMPORT_ERROR
        )
        report = {"status": status, "error": f"{type(exc).__name__}: {exc}"}
    else:
        libraries, error = _snapshot()
        report = {
            "status": TorchWorkerStatus.IMPORTED,
            "version": torch.__version__,
            "hip_version": getattr(getattr(torch, "version", None), "hip", None),
            "torch_file": getattr(torch, "__file__", None),
            "libraries": [asdict(library) for library in libraries],
            "error": error,
        }
    output.write_text(json.dumps(report), encoding="utf-8")


def _stderr_tail(path: Path) -> str:
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 4096))
        return stream.read().decode("utf-8", errors="replace").strip()


def probe_torch_rocm(*, timeout: float = 30.0) -> TorchRocmReport:
    """Observe the parent and import torch in a fresh, time-bounded interpreter.

    Does not load COMGR, HIP or torch in the parent. The child reports its
    torch module's file and uses its own startup path, not the caller's
    in-memory sys.path. Libraries not mapped by the child import are unknown;
    this does not exercise JIT or GPU launch.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    parent, parent_error = _snapshot()
    with tempfile.TemporaryDirectory(prefix="rocke-torch-probe-") as directory:
        output = Path(directory) / "report.json"
        log = Path(directory) / "stderr.txt"
        worker_status, reason, payload = TorchWorkerStatus.WORKER_ERROR, "", None
        with log.open("wb") as stderr:
            try:
                child = subprocess.run(
                    [
                        sys.executable,
                        "-X",
                        "faulthandler",
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(output),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                worker_status, reason = (
                    TorchWorkerStatus.TIMEOUT,
                    "torch import probe timed out",
                )
            except OSError as exc:
                reason = f"could not start torch import probe: {exc}"
            else:
                if child.returncode:
                    crashed = child.returncode < 0 or (
                        sys.platform == "win32" and child.returncode >= 0x80000000
                    )
                    if crashed:
                        worker_status = TorchWorkerStatus.CRASHED
                    reason = f"torch import probe exited with status {child.returncode}"
                else:
                    try:
                        payload = json.loads(output.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        reason = f"could not read torch import probe report: {exc}"
        stderr_text = _stderr_tail(log)
    if payload is None:
        return TorchRocmReport(
            TorchRocmStatus.INCONCLUSIVE,
            reason,
            parent,
            worker_status=worker_status,
            stderr=stderr_text,
        )
    worker_status = TorchWorkerStatus(payload["status"])
    if worker_status is not TorchWorkerStatus.IMPORTED:
        return TorchRocmReport(
            TorchRocmStatus.INCONCLUSIVE,
            payload["error"],
            parent,
            worker_status=worker_status,
            stderr=stderr_text,
        )
    observed = tuple(RocmLibrary(**item) for item in payload["libraries"])
    if parent_error or payload["error"]:
        status = TorchRocmStatus.INCONCLUSIVE
        reason = "; ".join(
            label + error
            for label, error in (
                ("parent inventory: ", parent_error),
                ("child inventory: ", payload["error"]),
            )
            if error
        )
    else:
        status, reason = _compare(parent, observed)
    return TorchRocmReport(
        status=status,
        reason=reason,
        parent_libraries=parent,
        torch_libraries=observed,
        torch_version=payload["version"],
        torch_hip_version=payload["hip_version"],
        worker_status=worker_status,
        stderr=stderr_text,
        torch_file=payload["torch_file"],
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="child timeout in seconds"
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        _worker(args.worker)
        return
    try:
        report = probe_torch_rocm(timeout=args.timeout)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    _main()
