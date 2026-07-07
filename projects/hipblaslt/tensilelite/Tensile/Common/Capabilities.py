################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import rocisa

from typing import List, Dict

from .Types import IsaVersion, IsaInfo

# TEMP DIAGNOSTIC (AIHPBLAS-3877): fires only when TENSILE_CAP_DIAG is set and a
# normally-supported ISA probes as unsupported. Re-runs the exact tryAssembler
# probe via subprocess to capture the real compiler stderr / errno plus process
# context, which pytest surfaces under "Captured stderr setup". To be reverted.
_CAP_DIAG_DUMPED = False


def _capDiagDump(v, cxxCompiler):
    global _CAP_DIAG_DUMPED
    if _CAP_DIAG_DUMPED:
        return
    _CAP_DIAG_DUMPED = True
    import os, sys, subprocess, resource
    ctx = ""
    try:
        fds = len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else -1
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        ctx = (f"isa={tuple(v)} cxx={cxxCompiler} exists={os.path.exists(cxxCompiler)} "
               f"cwd={os.getcwd()} cwd_writable={os.access(os.getcwd(), os.W_OK)} "
               f"fds_open={fds} nofile={soft}/{hard} pid={os.getpid()} "
               f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')!r}")
    except Exception as e:
        ctx = f"(context gather failed: {e!r})"
    probe = ""
    try:
        gfx = "gfx%x%x%x" % (v[0], v[1], v[2])
        cmd = [cxxCompiler, "-x", "assembler", "-target", "amdgcn-amdhsa",
               "-mcpu=" + gfx, "-"]
        r = subprocess.run(cmd, input=b"", capture_output=True)
        probe = (f"subprocess.rc={r.returncode} "
                 f"stderr={r.stderr.decode(errors='replace')[:800]!r}")
    except Exception as e:
        probe = f"subprocess raised {type(e).__name__}: {e}"
    try:
        sys.stderr.write(f"[CAPDIAG] {ctx} {probe}\n")
        sys.stderr.flush()
    except Exception:
        pass


def makeIsaInfoMap(targetIsas: List[IsaVersion], cxxCompiler: str) -> Dict[IsaVersion, IsaInfo]:
    """Computes the supported capabilities for requested ISAs and compiler.

    Given a list of ISAs and a compiler, the ASM, Arch, Register capabilities
    and ASM bugs are computed and stored in a map.

    Args:
        targetIsas: A list of requested ISA versions to inspect.
        cxxCompiler: A string path to a C++ compiler to use when computing capabilities.

    Returns:
        A map of ISA versions to capabilities.
    """
    isaInfoMap = {}
    ti = rocisa.rocIsa.getInstance()
    for v in targetIsas:
        ti.init(v, cxxCompiler, False)
        asmCaps = ti.getIsaInfo(v).asmCaps
        archCaps = ti.getIsaInfo(v).archCaps
        regCaps = ti.getIsaInfo(v).regCaps
        asmBugs = ti.getIsaInfo(v).asmBugs
        # TEMP (AIHPBLAS-3877): gfx942 probing unsupported is always anomalous;
        # dump the real compiler error/context once. Unconditional so it does not
        # depend on env propagation through tox. To be reverted.
        if tuple(v) == (9, 4, 2) and not asmCaps.get("SupportedISA"):
            _capDiagDump(v, cxxCompiler)
        isaInfoMap[v] = IsaInfo(asmCaps, archCaps, regCaps, asmBugs)
    return isaInfoMap
