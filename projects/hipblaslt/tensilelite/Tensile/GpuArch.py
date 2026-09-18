# Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU architecture detection for Tensile --gpu-targets.

Reports the name the runtime reports, which is also the compiler target its
kernels must be built for. That matters for gfx1250, which ships as two
steppings sharing ISA (12,5,0) but built as separate targets: code objects for
gfx1250 are rejected on gfx1250-strict and vice versa.

Deliberately rocisa-free (no Tensile.Common import) so the invoke build path
never pulls in rocisa just to probe the arch, and it lives inside the packaged
Tensile tree so ROCm test artifacts can exercise it directly; tensilelite's
tasks.py only wraps it in an invoke @task entry point.
"""

import os
import re
import shutil
import subprocess
import sys

# An agent name line from rocminfo, e.g. "  Name:  gfx1250-strict".
_ROCMINFO_AGENT_RE = re.compile(r"^\s+Name:\s+(gfx\S+?)\s*$", re.MULTILINE)
# amdgpu-arch prints one target per line.
_ARCH_LINE_RE = re.compile(r"^(gfx\S+)$", re.MULTILINE)

# A machine with no GPU still enumerates this placeholder.
_PLACEHOLDER_ARCH = "gfx000"

# amdgpu-arch sits under llvm/bin in a stock ROCm install and under lib/llvm/bin
# in the Windows SDK and TheRock layouts.
_AMDGPU_ARCH_RELPATHS = (
    ("llvm", "bin", "amdgpu-arch"),
    ("lib", "llvm", "bin", "amdgpu-arch"),
)

_ROCMINFO_RELPATHS = (("bin", "rocminfo"),)


def _rocm_roots():
    """The ROCm installs ROCM_PATH names, in the order it names them.

    It holds an os.pathsep-separated list often enough that ``validateToolchain``
    splits it too; a box with both a gfx1250 and a gfx1250-strict SDK installed
    is exactly when it does, and exactly when reading only the whole string --
    which matches no directory -- would fall through to PATH and be answered by
    the other install.
    """
    return os.environ.get("ROCM_PATH", "/opt/rocm").split(os.pathsep)


def _inRocmInstall(root, relative_parts):
    """That path under one ROCm install, if it is there and executable."""
    candidate = os.path.join(root, *relative_parts)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _tool(relpaths):
    """An executable at any of ``relpaths``, the ROCm installs before PATH.

    Every install is tried against every layout before PATH is consulted at all.
    Resolving one candidate the whole way to PATH before trying the next would
    let a stray amdgpu-arch beat the one inside the install the caller pointed
    ROCM_PATH at, since the two layouts differ only by a "lib" prefix.
    """
    found = next(
        (
            tool
            for root in _rocm_roots()
            for parts in relpaths
            if (tool := _inRocmInstall(root, parts))
        ),
        None,
    )
    if found:
        return found
    return next((w for parts in relpaths if (w := shutil.which(parts[-1]))), None)


def _run(command):
    """stdout of a successful run, or None on any failure. Never raises."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _real_archs(names):
    """The reported names, minus the GPU-less placeholder, in enumeration order.

    One entry per agent, repeats included: callers index this positionally to
    answer "what is device N", so the four entries of a homogeneous four-GPU box
    have to stay four. Callers asking instead which architectures are present
    de-duplicate it themselves.
    """
    return [name for name in names if name and name != _PLACEHOLDER_ARCH]


def _base_arch(name):
    """``name`` without its stepping suffix or its target features.

    A stepping is hyphenated onto the base name (``gfx1250-strict``) while
    features are colon-delimited and may carry hyphens of their own
    (``gfx950:sramecc+:xnack-``), so the base ends at whichever comes first.
    """
    return name.partition(":")[0].split("-", 1)[0]


def _rocminfo_archs(rocminfo):
    """The agent names rocminfo reports, or ``[]`` if it is absent or fails.

    rocminfo needs read-write /dev/kfd, so a caller outside the render group
    gets the empty list rather than an error -- which is why it supplements
    amdgpu-arch here instead of replacing it.
    """
    if not rocminfo:
        return []
    output = _run([rocminfo])
    if not output:
        return []
    return _real_archs(_ROCMINFO_AGENT_RE.findall(output))


def _restore_steppings(archs, rocminfo_archs):
    """``archs``, with any stepping suffix only rocminfo reports put back.

    amdgpu-arch stopped being able to report a stepping in ROCm 10.2: it is now
    a trampoline that execs offload-arch, which names an agent from the KFD
    node's ``gfx_target_version`` alone and loads neither ROCr nor HIP. Both
    gfx1250 steppings publish 120500 there -- the revision lives in the node's
    ``capability`` bits 25:22, which offload-arch never reads -- so an A0 part
    comes back as a bare "gfx1250". ROCr does apply that rule, and rocminfo
    reports through ROCr, so it still answers "gfx1250-strict".

    Letting rocminfo only ever *lengthen* a name keeps each tool doing what it
    is good for: amdgpu-arch still enumerates, and rocminfo contributes a suffix
    only for a base the two already agree on. A base rocminfo reports under more
    than one spelling is left alone rather than guessed at, since a box holding
    both steppings has no single right answer to substitute.

    A base ``archs`` itself already spells with a stepping is left alone for the
    same reason from the other direction: that answer came from a tool that can
    tell the two apart on this box, so there is nothing to restore, and the
    device rocminfo happens not to be reporting would otherwise be renamed to
    its neighbour's stepping.
    """
    by_base = {}
    for name in rocminfo_archs:
        by_base.setdefault(_base_arch(name), set()).add(name.partition(":")[0])

    told_apart = {
        _base_arch(name) for name in archs if _base_arch(name) != name.partition(":")[0]
    }

    restored = []
    for arch in archs:
        head, separator, features = arch.partition(":")
        unrestorable = head != _base_arch(head) or head in told_apart
        candidates = set() if unrestorable else by_base.get(head, set())
        if len(candidates) == 1:
            (only,) = candidates
            if only.startswith(head + "-"):
                head = only
        restored.append(head + separator + features)
    return restored


def restore_steppings(archs):
    """``archs``, with any stepping suffix only rocminfo reports put back.

    For callers holding an enumeration this module did not produce. Tensile's
    build path is one: it asks a device enumerator first so that a ``target.lst``
    or ``HSA_OVERRIDE_GFX_VERSION`` pin still wins, and every enumerator it can
    be pointed at truncates a stepping the same way amdgpu-arch does, so the
    answer needs the same cross-check before it names a compiler target.
    """
    return _restore_steppings(archs, _rocminfo_archs(_tool(_ROCMINFO_RELPATHS)))


def _probe():
    """``(archs, any_tool_found)`` from the first tool that answers.

    Deliberately does not use rocm_agent_enumerator: it parses rocminfo with a
    capture group that ends at ``gfx\\d+``, so it truncates a suffix and answers
    "gfx1250" for an agent rocminfo names "gfx1250-strict" -- which would build
    the wrong stepping's kernels without a word. amdgpu-arch truncates the same
    way as of ROCm 10.2, so its answer is cross-checked against rocminfo, the
    one local source that applies the revision rule; see ``_restore_steppings``.

    The second element separates "no ROCm here" from "ROCm is here but reported
    nothing usable", which the callers report differently.
    """
    amdgpu_arch = _tool(_AMDGPU_ARCH_RELPATHS)
    rocminfo = _tool(_ROCMINFO_RELPATHS)

    if amdgpu_arch:
        output = _run([amdgpu_arch])
        if output:
            archs = _real_archs(_ARCH_LINE_RE.findall(output))
            if archs:
                return _restore_steppings(archs, _rocminfo_archs(rocminfo)), True

    archs = _rocminfo_archs(rocminfo)
    if archs:
        return archs, True

    return [], bool(amdgpu_arch or rocminfo)


def detect_gpu_archs():
    """One architecture name per GPU the runtime reports, in enumeration order.

    Repeats are kept, so ``[deviceId]`` names that device. Empty when nothing
    could be read, whether or not ROCm is installed; callers that need to tell
    those apart should say so themselves.
    """
    return _probe()[0]


def cmake_gpu_target(name):
    """The ``GPU_TARGETS`` spelling of an architecture name a host reported.

    A detection tool answers with a configuration, not a build target:
    amdgpu-arch names a gfx950 agent ``gfx950:sramecc+:xnack-``, listing the
    target features that agent happens to have. The build validates
    ``GPU_TARGETS`` against a fixed list of targets and appends ``:xnack+``
    itself where a sanitizer build needs it, so a reported feature is not a
    target it accepts -- configure fails on the whole string.

    Only the colon-delimited features come off. A stepping is spelled with a
    hyphen and is part of the name (``gfx1250-strict``), and dropping it would
    build the other stepping's code objects, which the silicon rejects.
    """
    return name.split(":", 1)[0]


def detect_gpu_arch():
    """The architecture name the runtime reports for the first GPU, or None."""
    archs, any_tool_found = _probe()
    if archs:
        return archs[0]

    if not any_tool_found:
        print(
            "Error: neither 'amdgpu-arch' nor 'rocminfo' found. Please install ROCm.",
            file=sys.stderr,
        )
        return None

    print(
        "Failed to detect a valid GPU architecture (gfx target not found).",
        file=sys.stderr,
    )
    return None
