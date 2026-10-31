"""Which toolchain actually produced a shipped kernel.

The authored fields (source, entry, build / builder, spec) say what was *asked
for*. They do not say what answered. Two builds of byte-identical descriptors
can differ because the HIP SDK changed, comgr changed, or the rocKE wheel
changed -- and without this, a shipped artifact carries no way to tell those
apart after the fact.

This belongs in provenance, not in the variant key. In the key, a wheel bump
would rename every rocKE artifact including kernels it could not have affected;
in provenance it is traceability, while the build-level wheel digest handles
invalidation.

Probes are memoized per process: the answers cannot change mid-run, and
reading a version file is cheap but still should not be repeated per kernel.
"""

import functools
import hashlib
import re
from pathlib import Path


_HIP_VERSION_PARTS = ("MAJOR", "MINOR", "PATCH")


def _read_hip_build_version(path):
    """Parse HIP_VERSION_MAJOR/MINOR/PATCH from a HIP metadata file.

    Handles both ``share/hip/version`` (KEY=VALUE) and
    ``include/hip/hip_version.h`` (#define) formats.
    """
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return None
    values = []
    for part in _HIP_VERSION_PARTS:
        match = re.search(
            rf"^\s*(?:#define\s+)?HIP_VERSION_{part}(?:\s*=\s*|\s+)(\d+)",
            contents,
            re.MULTILINE,
        )
        if not match:
            return None
        values.append(match.group(1))
    githash = re.search(
        r'^\s*(?:#define\s+)?HIP_VERSION_GITHASH(?:\s*=\s*|\s+)"?(\w+)"?',
        contents,
        re.MULTILINE,
    )
    version = ".".join(values)
    if githash:
        version = f"{version}-{githash.group(1)}"
    return version


@functools.lru_cache(maxsize=None)
def hip_version(hip_path):
    """HIP SDK version from the version metadata files under hip_path.

    Best-effort by design: provenance is a record, not a gate. A build that
    cannot determine the HIP version still compiles, and failing the pack over
    a missing version string would trade a real artifact for a cosmetic one.

    Reads ``share/hip/version`` first (KEY=VALUE), falling back to
    ``include/hip/hip_version.h`` (#define).
    """
    if not hip_path:
        return None
    root = Path(hip_path)
    for rel in ("share/hip/version", "include/hip/hip_version.h"):
        version = _read_hip_build_version(root / rel)
        if version:
            return version
    return None


@functools.lru_cache(maxsize=None)
def comgr_info():
    """(path, version) of the comgr rocke resolved, or (None, None).

    Reports the library that was actually LOADED rather than the one that was
    requested. That distinction matters: rocke treats an explicit
    ROCKE_COMGR_LIB as the first candidate and silently falls through to the
    next when it will not load, so the requested path can differ from the real
    one and only this side of it is true.
    """
    try:
        from rocke.runtime import comgr
    except Exception:
        return None, None

    path = None
    version = None
    try:
        path = comgr.resolved_lib_path()
    except Exception:
        pass
    try:
        raw = comgr.resolved_lib_rocm_version()
        if raw is not None:
            version = ".".join(str(part) for part in raw)
    except Exception:
        pass
    return path, version


@functools.lru_cache(maxsize=None)
def wheel_digest(stamp_path):
    """The rocKE wheel content digest the build stamped, or None.

    Read from the build's stamp rather than recomputed: the stamp is what the
    build keyed its staleness decisions on, so recording anything else would let
    provenance and invalidation disagree.
    """
    if not stamp_path:
        return None
    path = Path(stamp_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def rocke_provenance(wheel_stamp=None):
    """Toolchain fields for a rocKE-produced kernel."""
    comgr_path, comgr_version = comgr_info()
    fields = {}
    if comgr_path:
        fields["comgr_path"] = comgr_path
    if comgr_version:
        fields["comgr_rocm_version"] = comgr_version
    digest = wheel_digest(wheel_stamp)
    if digest:
        fields["rocke_wheel_sha256"] = digest
    return fields


def hip_provenance(hip_path):
    """Toolchain fields for a hip-produced kernel."""
    version = hip_version(hip_path)
    return {"hip_version": version} if version else {}
