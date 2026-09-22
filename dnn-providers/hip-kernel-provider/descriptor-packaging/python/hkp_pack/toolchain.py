"""Which toolchain actually produced a shipped kernel.

The authored fields (source, entry, build / builder, spec) say what was *asked
for*. They do not say what answered. Two builds of byte-identical descriptors
can differ because hipcc changed, comgr changed, or the rocKE wheel changed --
and without this, a shipped artifact carries no way to tell those apart after
the fact.

This belongs in provenance, not in the variant key. In the key, a wheel bump
would rename every rocKE artifact including kernels it could not have affected;
in provenance it is traceability, while the build-level wheel digest handles
invalidation.

Probes are memoized per process: the answers cannot change mid-run, and
`hipcc --version` is a subprocess we should not pay per kernel.
"""

import functools
import hashlib
import subprocess
from pathlib import Path


@functools.lru_cache(maxsize=None)
def hipcc_version(hipcc):
    """First line of `hipcc --version`, or None if it cannot be determined.

    Best-effort by design: provenance is a record, not a gate. A hipcc that
    cannot report a version still compiles, and failing the pack over a missing
    version string would trade a real artifact for a cosmetic one.
    """
    if not hipcc:
        return None
    try:
        proc = subprocess.run(
            [hipcc, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    if not text:
        return None
    return text.splitlines()[0].strip()


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


def hip_provenance(hipcc):
    """Toolchain fields for a hip-produced kernel."""
    version = hipcc_version(hipcc)
    return {"hipcc_version": version} if version else {}
