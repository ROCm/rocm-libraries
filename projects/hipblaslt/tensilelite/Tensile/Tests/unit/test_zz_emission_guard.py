"""Candidate remediation C: a derived-valid decoupled solution must also emit.

Deliberately written as ONE test over the pair space rather than an assertion
bolted onto each existing test, so a new pair shape is covered the day it is
accepted rather than the day someone remembers to add a case.

Emission is driven exactly as test_PrefetchAcrossPersistent._emit_asm drives it
(same three calls, same rocisa state guard), which is the in-tree precedent for
emitting from a unit test on CPU.
"""
import re
import shutil

import pytest

from test_DecouplePGR import _derive, _gp_gfx1250, assembler, gfx1250_iim  # noqa: F401

pytestmark = pytest.mark.unit


def _emit(sol, asmToolchain):
    """(err, asm) for one derived solution.  CPU only; no GPU is touched."""
    import rocisa
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (generateKernelObjectsFromSolutions,
                                                  processKernelSource)
    from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

    with preserve_rocisa_kernel_state():
        kwa = KernelWriterAssembly(asmToolchain, DebugConfig())
        errs, pieces = [], []
        for kernel in generateKernelObjectsFromSolutions([sol]):
            ri = rocisa.rocIsa.getInstance()
            ri.init(tuple(kernel["ISA"]),
                    shutil.which("amdclang++") or "/usr/bin/amdclang++")
            ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
            kernel.duplicate = False
            kernel["BaseName"] = getKernelFileBase(False, kernel)
            res = processKernelSource(kwa, ri.getData(), ri.getOutputOptions(),
                                      False, kernel)
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            pieces.append(src or "")
            errs.append(res.err)
    return errs, "\n".join(pieces)


# Every pair shape the resolver can hand a surviving solution: the two divergent
# pairs, both one-sided autos, both-sided auto, the equal pairs that degenerate
# to the scalar, and the legacy scalar itself.
PAIRS = [
    (1, 2), (2, 1),           # divergent, explicit
    (-1, 1), (1, -1),         # divergent, reached through one-sided auto
    (-1, 2), (2, -1),         # one-sided auto that lands on the equal pair
    (2, 2), (1, 1), (0, 0),   # equal pairs, degenerate to the scalar
    (-1, -1),                 # both-sided auto
]


@pytest.mark.parametrize("pgrA, pgrB", PAIRS)
def test_every_derived_valid_pair_also_emits(_gp_gfx1250, gfx1250_iim, assembler,
                                             capsys, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       PrefetchGlobalRead=2,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    if sol.get("Valid") is not True:
        pytest.skip("(%s, %s) is refused at derivation: %r" % (pgrA, pgrB, out))
    errs, asm = _emit(sol, assembler)
    captured = capsys.readouterr().out
    reason = re.search(r"Tensile::WARNING: [^\n]*(thick-wait|error \d+)[^\n]*",
                       captured)
    assert errs == [0] and asm, (
        "(%s, %s) derives Valid but emits err=%s with %u bytes of source; "
        "resolved pair (%s, %s), _ScheduleIterAlg=%s, _StinkyTofuOptLevel=%s.  %s"
        % (pgrA, pgrB, errs, len(asm), sol.get("PrefetchGlobalReadA"),
           sol.get("PrefetchGlobalReadB"), sol.get("_ScheduleIterAlg"),
           sol.get("_StinkyTofuOptLevel"),
           reason.group(0) if reason else "(no warning captured)"))
