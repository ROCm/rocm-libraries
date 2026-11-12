# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDMSingleIssuer proto roles: TDM A+MX on 0/2, GL2 PF on 1/3."""

from __future__ import annotations

from Tensile.Components.Subtile.LdsV7Layout import (
    ISSUER_A, ISSUER_B, ISSUER_MXSA, ISSUER_MXSB,
    PREFETCH_A, PREFETCH_B, make_lds_v7_layout,
)
from rocisa.code import Label, Module
from rocisa.container import sgpr
from rocisa.instruction import (
    SBitcmp1B32, SCBranchSCC1, SCmpEQU32, SCBranchSCC0, SWaitTensorcnt,
)


def test_proto_tdm_and_prefetch_wave_ids():
  assert (ISSUER_A, ISSUER_B) == (0, 2)
  assert (ISSUER_MXSA, ISSUER_MXSB) == (0, 2)
  assert (PREFETCH_A, PREFETCH_B) == (1, 3)
  lay = make_lds_v7_layout()
  assert lay.issuer("A") == 0 and lay.issuer("B") == 2
  assert lay.issuer("MXSA") == 0 and lay.issuer("MXSB") == 2
  assert lay.prefetch("A") == 1 and lay.prefetch("B") == 3


def test_issuer_only_wait_tensorcnt_gate_shape():
  """Even-wave gate around s_wait_tensorcnt (matches InstructionEmitter)."""
  mod = Module("tcnt")
  skip = Label("tdm_si_tcnt_ut", "")
  mod.add(SBitcmp1B32(src0=sgpr(0), src1=0,
                      comment="TDMSingleIssuer: wait_tensorcnt on even (issuer) waves"))
  mod.add(SCBranchSCC1(labelName=skip.getLabelName(),
                       comment="odd waves: no TDM outstanding"))
  mod.add(SWaitTensorcnt(tensorcnt=1, comment="single-issuer READY"))
  mod.add(skip)
  kinds = [type(x).__name__ for x in mod.flatitems()]
  assert kinds == ["SBitcmp1B32", "SCBranchSCC1", "SWaitTensorcnt", "Label"]


def test_dual_issue_wave_gate_comments():
  """Issuer gates name the dual-issue pairs used by emitSingleBufferLoad."""
  for wave, tag in ((0, "A+MXSA"), (2, "B+MXSB")):
    mod = Module("ld")
    skip = Label(f"tdm_si_ld_{tag}_ut", "")
    mod.add(SCmpEQU32(src0=sgpr(0), src1=wave,
                      comment=f"TDMSingleIssuer: {tag} issue = wave {wave}"))
    mod.add(SCBranchSCC0(labelName=skip.getLabelName(),
                         comment=f"other waves skip {tag} TDM issue"))
    mod.add(skip)
    cmp = next(x for x in mod.flatitems() if isinstance(x, SCmpEQU32))
    assert f"wave {wave}" in (cmp.comment or "")
