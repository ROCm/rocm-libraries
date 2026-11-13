# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-wave-role mainloop bodies for subtile TDMSingleIssuer (TDMWaveRoleBodies).

The shared mainloop body runs every issuer duty (A/B/MXSA/MXSB TDM, GL2 PF,
cluster signal) behind ``s_cmp_eq_u32 WaveIdx, k; s_cbranch_scc0 L`` gates, so
every wave pays the compare + branch for every duty each iteration. gemm_wt
instead dispatches once into one body per wave with those duties resolved.
``foldWaveIdxGates`` produces such a body from the scheduled shared one.
"""

from rocisa.code import Label, Module
from rocisa.instruction import SCBranchSCC0, SCBranchSCC1, SCmpEQU32, SBranch, SWaitTensorcnt

# Scalar classes that cannot write SCC, so they may sit between a gate's
# compare and its branch.
_SCC_NEUTRAL_SCALAR_PREFIXES = ("SWait", "SNop", "SSetVgprMsb", "SBarrier", "SSetPrior")


def _gateWave(inst):
  """Return k when ``inst`` is ``s_cmp_eq_u32 s[sgprWaveIdx], k``, else None."""
  if not isinstance(inst, SCmpEQU32) or len(inst.srcs) != 2:
    return None
  src0, src1 = inst.srcs
  if str(src0) != "s[sgprWaveIdx]" or not isinstance(src1, int):
    return None
  return src1


def _mayWriteScc(item):
  if isinstance(item, (Label, SBranch, SCBranchSCC0, SCBranchSCC1)):
    return True
  name = type(item).__name__
  return name.startswith("S") and not name.startswith(_SCC_NEUTRAL_SCALAR_PREFIXES)


def foldWaveIdxGates(module, role):
  """Constant-fold every WaveIdx gate in ``module`` for a wave with WaveIdx == role.

  A gate is ``s_cmp_eq_u32 WaveIdx, k`` followed (possibly after SCC-neutral
  instructions such as WMMAs the scheduler slid in) by ``s_cbranch_scc0 L``,
  the gated body, and label ``L``. For k == role the compare, branch and label
  are dropped and the body kept; otherwise the body goes too. Returns a flat
  Module.
  """
  items = list(module.flatitems())
  out = Module(module.name)
  i = 0
  while i < len(items):
    wave = _gateWave(items[i])
    if wave is None:
      out.add(items[i])
      i += 1
      continue
    j = i + 1
    while j < len(items) and not isinstance(items[j], (SCBranchSCC0, SCBranchSCC1)):
      assert not _mayWriteScc(items[j]), \
        "WaveIdx gate: %s between compare and branch" % type(items[j]).__name__
      j += 1
    assert j < len(items) and isinstance(items[j], SCBranchSCC0), \
      "WaveIdx gate must branch on scc0 past its body"
    target = items[j].labelName
    end = j + 1
    while end < len(items) and not (isinstance(items[end], Label)
                                    and items[end].getLabelName() == target):
      assert not isinstance(items[end], (Label, SBranch, SCBranchSCC0, SCBranchSCC1)), \
        "WaveIdx gate body %s holds control flow" % target
      end += 1
    assert end < len(items), "WaveIdx gate label %s not in module" % target
    for item in items[i + 1:j]:
      out.add(item)
    if wave == role:
      for item in items[j + 1:end]:
        out.add(item)
    i = end + 1
  for item in out.flatitems():
    assert not any(str(s) == "s[sgprWaveIdx]" for s in getattr(item, "srcs", [])), \
      "WaveIdx survives in role-%d body" % role
  return out


def relaxScaleIssuerTensorcnt(module, count):
  """Let a scale issuer's READY waits leave ``count`` scale TDMs in flight.

  A role body's tensorcnt counts only that wave's own TDMs. With sc_nbuf scale
  buffers the slice a READY publishes is followed by up to sc_nbuf - 1 younger
  ones, so ``count`` = sc_nbuf - 1; drains (tensorcnt 0) stay.
  """
  out = Module(module.name)
  for item in module.flatitems():
    if isinstance(item, SWaitTensorcnt) and item.getParams()[0] > 0:
      item = SWaitTensorcnt(tensorcnt=count, comment=item.comment)
    out.add(item)
  return out


def emitWaveRoleDispatch(module, loopBegins, comment):
  """Branch waves 1..n-1 to their body; wave 0 falls through into loopBegins[0]."""
  from rocisa.container import sgpr
  for role in range(1, len(loopBegins)):
    module.add(SCmpEQU32(src0=sgpr("WaveIdx"), src1=role,
                         comment="%s: wave %d?" % (comment, role)))
    module.add(SCBranchSCC1(labelName=loopBegins[role].getLabelName(),
                            comment="%s: wave %d body" % (comment, role)))
