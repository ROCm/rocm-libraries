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

# Skip identity: gfx1250 is the family name for both revisions; gfx1250v0 is
# the only extra token (rev0). Expand gfx1250v0 *to* {gfx1250, gfx1250v0}.
# Apply HIP revision only when the enumerator listed gfx1250 — the string
# "gfx1250" from probe-fail is not "this machine is gfx1250" on a gfx950.
GFX1250_ARCH = "gfx1250"
GFX1250_V0 = "gfx1250v0"


def arch_skip_token(arch):
    """Bare gfx token for pytest skip/xfail matching (no feature suffix / predicate)."""
    if not arch:
        return ""
    token = str(arch).strip().split(":")[0]
    return token.split("[", 1)[0]


def _rocm_agent_enumerator():
    """Prefer $ROCM_PATH/bin, then PATH. Same lookup as pytest get_available_archs()."""
    rocmpath = os.environ.get(
        "TENSILE_ROCM_PATH", os.environ.get("ROCM_PATH", "/opt/rocm")
    )
    candidate = os.path.join(rocmpath, "bin", "rocm_agent_enumerator")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which("rocm_agent_enumerator") or "rocm_agent_enumerator"


def detect_gpu_arch():
    """First non-gfx000 gfx arch rocm_agent_enumerator reports, or None.

    Kept rocisa-free (no Tensile.Common import) so the invoke build path never
    pulls in rocisa just to probe the arch. Uses ``-t GPU`` (bare gfx names)
    so pytest skip identity and invoke get-gpu-arch see the same enumerator.
    """
    enumerator = _rocm_agent_enumerator()
    for args in ([enumerator, "-t", "GPU"], [enumerator, "-v"]):
        try:
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=5, check=True
            )
        except FileNotFoundError:
            print("Error: 'rocm_agent_enumerator' command not found. Please install ROCm.", file=sys.stderr)
            break
        except subprocess.TimeoutExpired:
            print("Error: GPU detection timed out. Hardware might be unresponsive.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"An unexpected error occurred during GPU detection: {e}", file=sys.stderr)
            continue
        target = next(
            (
                line.strip()
                for line in result.stdout.splitlines()
                if line.startswith("gfx") and line.strip() != "gfx000"
            ),
            None,
        )
        if target:
            return target

    print(f"Failed to detect a valid GPU architecture (gfx target not found).", file=sys.stderr)
    return None


def _normalize_gfx1250_asic_revision(base_arch, asic_revision):
    """Coerce gfx1250 asicRevision 2 to 1. Warn on stderr only (stdout is TENSILE_TARGET)."""
    token = str(base_arch).strip().split(":")[0] if base_arch else ""
    if token == "gfx1250" and asic_revision == 2:
        print("warning: gfx1250 asicRevision 2 treated as 1 (FFM workaround)",
              file=sys.stderr)
        return 1
    return asic_revision


def _revision_to_gpu_target(base_arch, asic_revision):
    """Map a detected base arch + ASIC revision to a Tensile --gpu-targets value.

    Only gfx1250 revision 0 is v0. Everything else defaults to revision 1.
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
    if arch_skip_token(base_arch) != GFX1250_ARCH:
        return base_arch

    probed = _probe_asic_revision(build_dir=build_dir, device_id=device_id)
    if probed is None:
        print("warning: could not determine gfx1250 ASIC revision; "
              "defaulting to gfx1250 (v1).", file=sys.stderr)
        return "gfx1250"

    probe_arch, revision = probed
    # gcnArchName carries feature suffixes on real hardware
    # (e.g. "gfx1250:sramecc+:xnack-"); compare only the base arch token.
    if arch_skip_token(probe_arch) != GFX1250_ARCH:
        print(f"warning: revision probe reported arch '{probe_arch}' != detected "
              "'gfx1250'; defaulting to gfx1250 (v1).", file=sys.stderr)
        return "gfx1250"

    target = _revision_to_gpu_target("gfx1250", revision)
    # Only revision 0 is v0; anything else (incl. unseen values) is v1, so
    # the raw number is the only thing telling a confirmed v1 from an unknown.
    print(f"gfx1250 asicRevision {revision} -> {target}", file=sys.stderr)
    return target


def enumerator_reports_gfx1250(enumerated_archs):
    """True when rocm_agent_enumerator listed a gfx1250 device, not gfx1250v0."""
    return any(arch_skip_token(arch) == GFX1250_ARCH for arch in (enumerated_archs or ()))


def expand_revision_skip_archs(arch):
    """gfx1250v0 -> {gfx1250, gfx1250v0}; never drop the family name."""
    token = arch_skip_token(arch)
    if not token:
        return frozenset()
    if token == GFX1250_V0:
        return frozenset({GFX1250_ARCH, GFX1250_V0})
    return frozenset({token})


def gpu_targets_from_argv(argv):
    """``--gpu-targets`` / ``--gpu-target`` values from Tensile or pytest argv."""
    if not argv:
        return []
    args = [str(a).strip() for a in argv if str(a).strip()]
    targets = []
    i = 0
    while i < len(args):
        if args[i] in ("--gpu-targets", "--gpu-target") and i + 1 < len(args):
            targets.extend(
                t.strip()
                for t in args[i + 1].replace(",", ";").split(";")
                if t.strip()
            )
            i += 2
            continue
        i += 1
    return targets


def argv_selects_gfx1250v0(argv):
    """True for pytest ``--gpu-targets gfx1250v0`` or comma-split tensile-options."""
    return any(arch_skip_token(t) == GFX1250_V0 for t in gpu_targets_from_argv(argv))


def gfx1250_revision_skip_target(device_id=0):
    """HIP asicRevision -> skip identity on a machine the enumerator listed as gfx1250."""
    probed = _probe_asic_revision(device_id=device_id)
    if probed is not None:
        probe_arch, revision = probed
        if arch_skip_token(probe_arch) == GFX1250_ARCH:
            return _revision_to_gpu_target("gfx1250", revision)
    return GFX1250_ARCH
