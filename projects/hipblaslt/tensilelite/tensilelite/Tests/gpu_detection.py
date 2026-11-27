################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
GPU architecture detection for TensileLite unit and common tests.

``get_available_archs``: every arch on the machine, for common-config ``skip-gfxNNNN``.
``detect_gpu_target``: the device HIP gives this process, for the unit-test ``gpu`` mark.
"""

import contextlib
import functools
import os
import re
import subprocess
import sys

from tensilelite.GpuArch import (
    _ROCMINFO_RELPATHS,
    _run,
    _tool,
    cmake_gpu_target,
    detect_gpu_archs,
)


@contextlib.contextmanager
def _rocmPathFromTestOverride():
    """``TENSILE_ROCM_PATH`` standing in for ``ROCM_PATH`` inside the block.

    ``tensilelite.GpuArch`` reads ``ROCM_PATH``, the variable the build uses. The
    tests keep their own override so a run can probe a different install without
    moving the one the build points at, so it is applied here and taken back
    out again rather than leaking into anything the tests go on to launch.
    """
    override = os.environ.get("TENSILE_ROCM_PATH")
    if not override:
        yield
        return

    previous = os.environ.get("ROCM_PATH")
    os.environ["ROCM_PATH"] = override
    try:
        yield
    finally:
        if previous is None:
            del os.environ["ROCM_PATH"]
        else:
            os.environ["ROCM_PATH"] = previous


def get_available_archs() -> list[str]:
    """Get list of available GPU architectures, one entry per distinct name.

    Reads through ``tensilelite.GpuArch``, the same detection the build uses, so a
    stepping suffix survives. It has to: ``config_helpers.configMarks`` keys its
    ``skip-<arch>`` marks off this list, and the two spellings of a config carry
    mirrored marks -- a strict config says ``skip-gfx1250``, a base config says
    ``skip-gfx1250-strict``. Handing that comparison a truncated ``gfx1250`` for
    an agent named ``gfx1250-strict`` therefore does not narrow the selection,
    it reverses it: every base config runs against a strict client and every
    strict config is skipped. ``rocm_agent_enumerator``, which this used to
    call, truncates in exactly that way.

    The target features go, through the same ``cmake_gpu_target`` the build
    spells ``GPU_TARGETS`` with. A detection tool answers with a configuration,
    so amdgpu-arch names a gfx90a agent ``gfx90a:sramecc+:xnack-``, and a mark
    is written for the architecture rather than for one agent's features:
    ``skip-gfx90a`` never matches ``skip-gfx90a:sramecc+:xnack-``, so keeping
    them would unskip every config pinned off this architecture. Only the
    colon-delimited features come off; the hyphenated stepping stays.

    Environment variable priority:
        1. TENSILE_ROCM_PATH (test-specific override)
        2. ROCM_PATH (standard ROCm variable)
        3. /opt/rocm (default)

    Returns:
        List of unique gfx architecture strings (e.g. ["gfx1250-strict"]).
        Returns empty list if no detection tool is found or all of them fail.
    """
    with _rocmPathFromTestOverride():
        return list(dict.fromkeys(map(cmake_gpu_target, detect_gpu_archs())))


def has_arch(target: str) -> bool:
    """Check if a specific GPU architecture is available."""
    return any(target in arch for arch in get_available_archs())


# ---- Unit-test device (HIP; lazy, so `-m "not gpu"` never initialises it) ----

@functools.cache
def _hip():
    try:
        from hip import hip
    except ImportError:
        return None
    return hip


def hip_available() -> bool:
    """True when hip-python is importable."""
    return _hip() is not None


# Runs in a child process. Tests/unit/conftest.py calls detect_gpu_target() at collection,
# before the autouse assign_gpu_to_worker fixture (Tests/conftest.py) sets
# HIP_VISIBLE_DEVICES, and HIP ignores that variable once it is initialised, so an
# in-process probe put every pytest-xdist worker on GPU 0. Prints device 0's arch, or
# nothing when HIP reports no device; any other HIP error exits non-zero with its name.
_PROBE = """
import sys

from hip import hip


def check(result):
    error, *rest = result if isinstance(result, tuple) else (result,)
    if int(error) == int(hip.hipError_t.hipErrorNoDevice):
        raise SystemExit(0)  # no device: print nothing, so the tests skip
    if int(error) != 0:
        sys.exit(f"HIP error {int(error)} ({getattr(error, 'name', error)})")
    return rest[0] if len(rest) == 1 else rest


if int(check(hip.hipGetDeviceCount())) > 0:
    try:
        # hip-python 10 (a wrapper over rocm-bindings) returns the struct.
        props = check(hip.hipGetDeviceProperties(0))
    except TypeError:
        # Earlier releases fill a caller-owned struct instead.
        props = hip.hipDeviceProp_t()
        check(hip.hipGetDeviceProperties(props, 0))
    name = props.gcnArchName
    print(name.decode() if isinstance(name, bytes) else name)
"""


@functools.cache
def detect_gpu_target() -> str | None:
    """Arch of HIP device 0 (e.g. ``gfx1250-strict``), or None without hip-python or a device.

    ``TENSILE_GPU_TARGET`` overrides. HIP, unlike rocm_agent_enumerator, honours
    ROCR_VISIBLE_DEVICES, sees only devices this process can open, and keeps
    the stepping. HIP is queried in a child process (see ``_PROBE``). RuntimeError for any
    other HIP error, or if that process crashes or hangs: a machine whose GPU is broken
    fails its ``gpu`` tests rather than skipping them. HIP also reports ``hipErrorNoDevice``
    when the driver cannot open the GPU, so only ``--require-gpu`` catches that case.
    """
    override = os.environ.get("TENSILE_GPU_TARGET")
    if override:
        return override
    if _hip() is None:
        return None
    # The probe takes under a second on a multi-GPU gfx942 node; the timeout only stops a hung
    # driver from stalling collection.
    try:
        probe = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True,
                               timeout=120, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"GPU probe ({sys.executable}) did not finish in 120 s") from None
    if probe.returncode != 0:
        raise RuntimeError(
            f"GPU probe ({sys.executable}) exited with {probe.returncode}: {probe.stderr.strip()[-500:]}")
    # gcnArchName carries target features (gfx942:sramecc+:xnack-).
    return probe.stdout.strip().split(":")[0] or None


def arch_matches(found: str, wanted: str) -> bool:
    """``gfx1250`` matches ``gfx1250`` and ``gfx1250-strict``; ``gfx1250-strict`` only itself."""
    return found == wanted or found.startswith(wanted + "-")


def check_gpu_marks(item) -> None:
    """One ``gpu`` mark per test, positional arch strings only; raise a usage error otherwise.

    Counted over the whole chain (param, function, class, module). With several,
    the one ``get_closest_marker`` returns depends on the test's shape and the
    pytest version, so the gate could run the test on the wrong arch.
    """
    import pytest

    marks = list(item.iter_markers("gpu"))
    if len(marks) > 1:
        raise pytest.UsageError(
            f"{item.nodeid}: {len(marks)} gpu marks across its param, function, class and module; use one")
    for m in marks:
        if m.kwargs or not all(isinstance(a, str) for a in m.args):
            raise pytest.UsageError(
                f"{item.nodeid}: gpu() takes arch strings only, e.g. gpu(\"gfx950\"); got {m.args} {m.kwargs}")


def gpu_is(*archs: str) -> bool:
    """True if the GPU matches any of *archs*. For non-gate code; tests use ``gpu(...)``."""
    found = detect_gpu_target()
    return found is not None and any(arch_matches(found, a) for a in archs)


# Device tags name a product that shares its gfx architecture with others, which
# the ``skip-<arch>`` marks cannot tell apart. MI300A is gfx942 like MI300X, but
# has 228 CUs (six XCDs) where MI300X has 304 (eight XCDs).
MI300A_CU_COUNT = 228

# ``Name:`` and ``Compute Unit:`` lines of one rocminfo agent block. "Marketing
# Name:" does not match the first pattern, which wants the line to start at Name.
_ROCMINFO_AGENT_START_RE = re.compile(r"^\s*Agent \d+\s*$")
_ROCMINFO_NAME_RE = re.compile(r"^\s+Name:\s+(.+?)\s*$")
_ROCMINFO_COMPUTE_UNIT_RE = re.compile(r"^\s+Compute Unit:\s+(\d+)\s*$")


def device_tags_from_rocminfo(rocminfo_output: str) -> list[str]:
    """Device tags (e.g. ``mi300a``) for the agents in ``rocminfo`` output.

    A tag is only given to an agent whose block carries both its gfx name and its
    CU count, so a CPU agent, which also reports a CU count, never earns one.
    """
    tags = []
    agentName = None
    for line in rocminfo_output.splitlines():
        if _ROCMINFO_AGENT_START_RE.match(line):
            agentName = None
            continue
        nameMatch = _ROCMINFO_NAME_RE.match(line)
        if nameMatch:
            agentName = nameMatch.group(1)
            continue
        cuMatch = _ROCMINFO_COMPUTE_UNIT_RE.match(line)
        if cuMatch and agentName and agentName.split(":")[0] == "gfx942":
            if int(cuMatch.group(1)) == MI300A_CU_COUNT and "mi300a" not in tags:
                tags.append("mi300a")
    return tags


def get_available_device_tags() -> list[str]:
    """Device tags for the GPUs on this machine, or ``[]`` if none can be found.

    ``TENSILE_TEST_DEVICE_TAGS`` (semicolon-separated, e.g. ``mi300a``) replaces
    detection, the way ``--gpu-targets`` replaces arch detection. Set it empty to
    declare that no device tag applies.

    Environment variable priority for the ROCm install that supplies rocminfo:
        1. TENSILE_ROCM_PATH (test-specific override)
        2. ROCM_PATH (standard ROCm variable)
        3. /opt/rocm (default)
    """
    override = os.environ.get("TENSILE_TEST_DEVICE_TAGS")
    if override is not None:
        return [t.strip() for t in override.split(";") if t.strip()]

    with _rocmPathFromTestOverride():
        rocminfo = _tool(_ROCMINFO_RELPATHS)
        output = _run([rocminfo]) if rocminfo else None
    return device_tags_from_rocminfo(output) if output else []


# ---------------------------------------------------------------------------------------------
# DPX: one physical GPU split into partitions that each show up as a HIP device
# ---------------------------------------------------------------------------------------------
#
# The partitions of one GPU share the PCI domain, bus and device number and differ only in the PCI
# function (0001:01:00.0 and 0001:01:00.1 on a DPX gfx1250). Two separate GPUs never share them, so
# this tells DPX from two SPX cards, which cannot reach each other's memory. The KFD topology has
# the same information without loading HIP: ``location_id`` is ``bus << 8 | device << 3 | function``.

_KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _kfd_properties(path):
    """The ``name value`` lines of one KFD node's ``properties`` file as a dict of ints."""
    props = {}
    try:
        with open(os.path.join(path, "properties")) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        props[parts[0]] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return props


def kfd_gpu_nodes(nodes_dir=_KFD_NODES):
    """``(domain, location_id)`` of every GPU node, in node order (the order HIP numbers them)."""
    try:
        names = sorted(os.listdir(nodes_dir), key=lambda n: int(n) if n.isdigit() else -1)
    except OSError:
        return []
    gpus = []
    for name in names:
        props = _kfd_properties(os.path.join(nodes_dir, name))
        if props.get("simd_count", 0) > 0:  # CPU nodes have none
            gpus.append((props.get("domain", 0), props.get("location_id", 0)))
    return gpus


def _visible(nodes, environ):
    """The nodes the process can use, after ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES.

    Both are lists of indexes here. A list of UUIDs cannot be matched to a node without HIP, so
    it makes the answer "none": a skipped test is better than a test on the wrong devices.
    """
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = environ.get(name)
        if value is None:
            continue
        tokens = [t.strip() for t in value.split(",") if t.strip()]
        if not all(t.isdigit() for t in tokens):
            return []
        nodes = [nodes[int(t)] for t in tokens if int(t) < len(nodes)]
    return nodes


def dpx_pairs(nodes_dir=_KFD_NODES, environ=None):
    """The visible HIP device pairs that are the two partitions of one GPU.

    A pair is two devices with the same PCI domain, bus and device and different functions, on a
    GPU that has exactly two visible partitions (DPX; four or eight partitions are another mode).
    The result is a list of ``(i, j)`` HIP device indexes, ``i < j``.
    """
    nodes = _visible(kfd_gpu_nodes(nodes_dir), os.environ if environ is None else environ)
    groups = {}
    for index, (domain, location) in enumerate(nodes):
        groups.setdefault((domain, location >> 3), []).append((location & 7, index))
    pairs = []
    for members in groups.values():
        if len(members) == 2 and members[0][0] != members[1][0]:
            pairs.append(tuple(sorted(index for _, index in members)))
    return sorted(pairs)


def has_dpx(nodes_dir=_KFD_NODES, environ=None):
    """Whether a GPU in DPX mode is visible to this process."""
    return bool(dpx_pairs(nodes_dir, environ))
