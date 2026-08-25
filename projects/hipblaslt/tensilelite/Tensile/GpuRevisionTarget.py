# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""gfx1250 v0/v1 ASIC-revision -> Tensile --gpu-targets detection.

gfx1250 ships as two silicon revisions (v0 and v1) that share the same ISA and
compiler arch name "gfx1250", so rocm_agent_enumerator/amdgpu-arch cannot tell
them apart. The only in-process signal is hipDeviceProp_t::asicRevision
(empirically v0 -> 0, v1 -> 1). The functions here let tox generate/test v0
kernels on a v0 machine while defaulting to v1 everywhere else.

This module deliberately carries no invoke dependency and lives inside the
packaged Tensile tree so the ROCm test artifacts can exercise the mapping and
probe-fallback logic directly; tensilelite/tasks.py only wraps it in invoke
@task entry points.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Tensile/GpuRevisionTarget.py -> Tensile -> tensilelite root.
_TENSILELITE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REVISION_PROBE_SRC = _TENSILELITE_ROOT / "tools" / "gpu_revision_probe.cpp"


def detect_gpu_arch():
    """First non-gfx000 gfx arch rocm_agent_enumerator reports, or None.

    Kept rocisa-free (no Tensile.Common import) so the invoke build path never
    pulls in rocisa just to probe the arch.
    """
    try:
        result = subprocess.run(["rocm_agent_enumerator", "-v"], capture_output=True, text=True, timeout=5, check=True)
        if result.returncode == 0:
            target = next((line.strip() for line in result.stdout.splitlines() if line.startswith("gfx") and line.strip() != "gfx000"), None)
            if target:
                return target
    except FileNotFoundError:
        print("Error: 'rocm_agent_enumerator' command not found. Please install ROCm.", file=sys.stderr)

    except subprocess.TimeoutExpired:
        print("Error: GPU detection timed out. Hardware might be unresponsive.", file=sys.stderr)

    except Exception as e:
        print(f"An unexpected error occurred during GPU detection: {e}", file=sys.stderr)

    print(f"Failed to detect a valid GPU architecture (gfx target not found).", file=sys.stderr)
    return None


def _normalize_gfx1250_asic_revision(base_arch, asic_revision):
    """Coerce FFM's gfx1250 asicRevision 2 to the shipping v1 value 1.

    FFM functional-model parts report asicRevision 2, which is not a
    shipping revision. Treat it as revision 1 so library / skip-set /
    --gpu-targets selection match revision 1. Warn on stderr, never stdout:
    callers capture stdout as TENSILE_TARGET=$(invoke get-gpu-revision-target).

    Real rev0 (asicRevision 0) and every non-gfx1250 arch are unchanged.
    Feature suffixes on gcnArchName (e.g. gfx1250:sramecc+:xnack-) still match.
    """
    token = str(base_arch).strip().split(":")[0] if base_arch else ""
    if token == "gfx1250" and asic_revision == 2:
        print("warning: gfx1250 asicRevision 2 treated as 1 (FFM workaround)",
              file=sys.stderr)
        return 1
    return asic_revision


def _revision_to_gpu_target(base_arch, asic_revision):
    """Map a detected base arch + ASIC revision to a Tensile --gpu-targets value.

    Only gfx1250 revision 0 is the pre-production v0. Everything else -- the
    v1 (revision 1), FFM's revision 2 after it is coerced to 1, an unknown
    revision (-1 when HIP is too old to expose the field), any future/
    unexpected value, and every non-gfx1250 arch -- is returned unchanged
    so tests default to revision 1.
    """
    asic_revision = _normalize_gfx1250_asic_revision(base_arch, asic_revision)
    if base_arch == "gfx1250" and asic_revision == 0:
        return "gfx1250v0"
    return base_arch


def _probe_asic_revision(build_dir=None, device_id=0):
    """Compile (once, cached) and run the HIP revision probe.

    Returns a (arch, revision) tuple on success, or None on any failure (hipcc
    missing, build error, no device, non-zero exit, or unparsable output) so the
    caller can fall back to the v1 default. Never raises.
    """
    hipcc = shutil.which("hipcc")
    if not hipcc:
        print("warning: hipcc not found; cannot probe gfx1250 ASIC revision.", file=sys.stderr)
        return None

    out_dir = pathlib.Path(build_dir) if build_dir else _TENSILELITE_ROOT / "build_tmp"
    probe_bin = out_dir / "gpu_revision_probe"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stale = (not probe_bin.exists()
                 or probe_bin.stat().st_mtime < _REVISION_PROBE_SRC.stat().st_mtime)
        if stale:
            # Compile to a unique temp file then atomically rename, so concurrent
            # callers (e.g. pytest-xdist workers) can never observe or run a
            # half-written binary (ETXTBSY / truncated exe).
            fd, tmp_bin = tempfile.mkstemp(
                dir=str(out_dir), prefix=".gpu_revision_probe.", suffix=".tmp")
            os.close(fd)
            try:
                subprocess.run(
                    [hipcc, "-O0", str(_REVISION_PROBE_SRC), "-o", tmp_bin],
                    check=True, capture_output=True, text=True, timeout=180,
                )
                os.chmod(tmp_bin, 0o755)
                os.replace(tmp_bin, str(probe_bin))
            finally:
                if os.path.exists(tmp_bin):
                    os.remove(tmp_bin)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"warning: failed to build ASIC revision probe: {e}", file=sys.stderr)
        return None

    try:
        result = subprocess.run(
            [str(probe_bin), str(device_id)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"warning: ASIC revision probe failed to run: {e}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"warning: ASIC revision probe exited {result.returncode}: "
              f"{result.stderr.strip()}", file=sys.stderr)
        return None

    # Forward probe stderr (FFM asicRevision-2 warning, hipcc noise) so it is
    # not swallowed by capture_output. Stdout stays the two-line arch/revision.
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    lines = result.stdout.splitlines()  # arch on line 1, revision on line 2
    if len(lines) < 2:
        print(f"warning: unexpected ASIC revision probe output: {result.stdout!r}",
              file=sys.stderr)
        return None
    try:
        revision = int(lines[1].strip())
    except ValueError:
        print(f"warning: could not parse ASIC revision from: {result.stdout!r}",
              file=sys.stderr)
        return None
    arch = lines[0].strip()
    return (arch, _normalize_gfx1250_asic_revision(arch, revision))


def detect_gpu_revision_target(build_dir=None, device_id=0):
    """Detect the Tensile --gpu-targets value, distinguishing gfx1250 v0 from v1.

    Non-gfx1250 arches are returned unchanged without probing. For gfx1250, the
    ASIC revision is probed via HIP: revision 0 -> gfx1250v0, otherwise (and on
    any probe failure or arch mismatch) -> gfx1250 (the v1 default).
    """
    base_arch = detect_gpu_arch()
    if base_arch != "gfx1250":
        return base_arch

    probed = _probe_asic_revision(build_dir=build_dir, device_id=device_id)
    if probed is None:
        print("warning: could not determine gfx1250 ASIC revision; "
              "defaulting to gfx1250 (v1).", file=sys.stderr)
        return "gfx1250"

    probe_arch, revision = probed
    # gcnArchName carries feature suffixes on real hardware
    # (e.g. "gfx1250:sramecc+:xnack-"); compare only the base arch token.
    if probe_arch.split(":")[0] != "gfx1250":
        print(f"warning: revision probe reported arch '{probe_arch}' != detected "
              "'gfx1250'; defaulting to gfx1250 (v1).", file=sys.stderr)
        return "gfx1250"

    target = _revision_to_gpu_target("gfx1250", revision)
    # Only revision 0 is v0; anything else (incl. unseen values) is v1, so
    # the raw number is the only thing telling a confirmed v1 from an unknown.
    print(f"gfx1250 asicRevision {revision} -> {target}", file=sys.stderr)
    return target


# --------------------------------------------------------------------------- #
# Pytest collection skip identity. Compile targets (--gpu-targets) are not the
# skip set. gfx1250 is the common ISA name for both ASIC revisions; gfx1250v0
# is the only extra skip identity (rev0 hardware). skip-gfx1250 still matches
# the whole family because gfx1250v0 expands *to* {gfx1250, gfx1250v0} rather
# than replacing gfx1250. Callers must only apply the probed-revision helper
# when the enumerator actually listed gfx1250 -- detect_gpu_revision_target()
# returns the string "gfx1250" for probe-fail / non-v0, which is not "this
# machine is gfx1250" on a gfx950.
# --------------------------------------------------------------------------- #
GFX1250_ARCH = "gfx1250"
GFX1250_V0 = "gfx1250v0"


def arch_skip_token(arch):
    """Bare gfx token for pytest skip/xfail matching (no feature suffix / predicate)."""
    if not arch:
        return ""
    token = str(arch).strip().split(":")[0]
    return token.split("[", 1)[0]


def enumerator_reports_gfx1250(enumerated_archs):
    """True when rocm_agent_enumerator listed a gfx1250 device.

    gfx1250v0 is a Tensile compile target, not an enumerator name. Do not treat
    detect_gpu_revision_target()'s "gfx1250" default as proof the device is
    gfx1250.
    """
    return any(arch_skip_token(arch) == GFX1250_ARCH for arch in (enumerated_archs or ()))


def expand_revision_skip_archs(arch):
    """Expand one compile/enumerator token into the pytest skip-identity set.

    gfx1250v0 -> {gfx1250, gfx1250v0}. Bare gfx1250 is left unexpanded: a
    probed revision (or an explicit gfx1250v0 compile target) is what adds
    gfx1250v0. Never replace gfx1250 with an alias.
    """
    token = arch_skip_token(arch)
    if not token:
        return frozenset()
    if token == GFX1250_V0:
        return frozenset({GFX1250_ARCH, GFX1250_V0})
    return frozenset({token})


def skip_archs_for_gfx1250_revision_target(revision_target):
    """Skip set for a detect_gpu_revision_target() result on real gfx1250 HW.

    Rev0 -> {gfx1250, gfx1250v0}. Anything else -- including the v1 default
    string "gfx1250" and probe failure -- is fail-open: {gfx1250} so
    skip-gfx1250v0 does not fire. Only call this when
    enumerator_reports_gfx1250() is true.
    """
    token = arch_skip_token(revision_target)
    if token == GFX1250_V0:
        return frozenset({GFX1250_ARCH, GFX1250_V0})
    return frozenset({GFX1250_ARCH})
