################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id aabf0d22ae0a02d4ae31ab4f43e838b29f28f582

Predicate : kernel["EdgeType"] == "ShiftPtr" and enableEdge
Site      : Tensile/KernelWriterAssembly.py:16798  (function writeBiasToGlobal)
Solver    : z3 4.16.0  — SAT  (fully-static; two independent witnesses)

Background
----------
enableEdge is a local bool initialised False at line 16790 and set True by
either of two guarded clauses at lines 16794-16797:

  if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialA"])
          and (kernel["ProblemType"]["BiasSrc"] == "A"):
      enableEdge = True
  if not (kernel["BufferLoad"] and kernel["GuaranteeNoPartialB"])
          and (kernel["ProblemType"]["BiasSrc"] == "B"):
      enableEdge = True

EdgeType defaults to "ShiftPtr" (Solution.py:4993) and is overridden to "None"
only when both TDM bits of TDMInst are set (enableTDMA & enableTDMB, i.e.
TDMInst == 3 in the seeded domain {0,3}).

GuaranteeNoPartialA/B are derived from layout bools TLUA/TLUB and the
AF*ElementMultiple % GlobalReadVectorWidth* == 0 modulo test
(Solution.py:4570-4578).

All inputs are validated YAML solution/problem parameters; no GPU state is read.
Classification: fully-static.

Witnesses (z3-derived, re-validated by pure-Python transcription):

  TRUE  example 1: TDMInst=0, BufferLoad=True, AF0EM=1, GRVWA=4, AF1EM=1,
                   GRVWB=4, BiasSrc="B", TLUA=False, TLUB=True
                   -> GNpB=(1%4==0)=False -> (BufferLoad&GNpB)=False ->
                      enableEdge=True (BiasSrc B leg); EdgeType=ShiftPtr. -> True
  TRUE  example 2: TDMInst=0, BufferLoad=False, AF0EM=1, GRVWA=1, AF1EM=1,
                   GRVWB=1, BiasSrc="A", TLUA=True, TLUB=True
                   -> BufferLoad=False -> (BufferLoad&GNpA)=False ->
                      enableEdge=True (BiasSrc A leg); EdgeType=ShiftPtr. -> True
  FALSE example 1: TDMInst=0, BufferLoad=True, AF0EM=4, GRVWA=1, AF1EM=1,
                   GRVWB=4, BiasSrc="D", TLUA=True, TLUB=False
                   -> BiasSrc="D" matches neither A nor B leg -> enableEdge=False -> False
  FALSE example 2: TDMInst=3, BufferLoad=False, AF0EM=1, GRVWA=4, AF1EM=1,
                   GRVWB=4, BiasSrc="A", TLUA=True, TLUB=True
                   -> enableTDMA=True, enableTDMB=True -> EdgeType=None (not ShiftPtr) -> False
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper — transcription of the branch predicate from source
# ---------------------------------------------------------------------------

def shiftptr_edge_taken(
    tdm_inst: int,
    buffer_load: bool,
    af0em: int,
    grvwa: int,
    af1em: int,
    grvwb: int,
    bias_src: str,
    tlua: bool,
    tlub: bool,
) -> bool:
    """
    Pure model of KernelWriterAssembly.py:16798 branch predicate.

    Returns True iff:
      kernel["EdgeType"] == "ShiftPtr"  and  enableEdge

    where:
      - EdgeType == "ShiftPtr" iff NOT (enableTDMA and enableTDMB)
        (enableTDMA = bool(TDMInst & 1); enableTDMB = bool(TDMInst & 2))
      - GuaranteeNoPartialA = (af0em % grvwa == 0) if tlua else True
      - GuaranteeNoPartialB = (af1em % grvwb == 0) if tlub else True
      - enableEdge = True if:
          (not (buffer_load and gnp_a) and bias_src == "A") or
          (not (buffer_load and gnp_b) and bias_src == "B")
    """
    enable_tdma = bool(tdm_inst & 1)
    enable_tdmb = bool(tdm_inst & 2)
    edge_is_shiftptr = not (enable_tdma and enable_tdmb)

    gnp_a = (af0em % grvwa == 0) if tlua else True
    gnp_b = (af1em % grvwb == 0) if tlub else True

    enable_edge = False
    if not (buffer_load and gnp_a) and bias_src == "A":
        enable_edge = True
    if not (buffer_load and gnp_b) and bias_src == "B":
        enable_edge = True

    return edge_is_shiftptr and enable_edge


# ---------------------------------------------------------------------------
# TRUE witnesses — branch is taken
# ---------------------------------------------------------------------------

def test_true_example_biassrc_b_grvwb_partial():
    """
    z3 true witness 1: BiasSrc=B, BufferLoad=True, GNpB=False (1%4!=0)
    -> enableEdge True (B leg); EdgeType=ShiftPtr (TDMInst=0) -> predicate True.
    """
    result = shiftptr_edge_taken(
        tdm_inst=0,
        buffer_load=True,
        af0em=1,
        grvwa=4,
        af1em=1,
        grvwb=4,
        bias_src="B",
        tlua=False,
        tlub=True,
    )
    assert result is True


def test_true_example_biassrc_a_buffload_off():
    """
    z3 true witness 2: BiasSrc=A, BufferLoad=False
    -> (BufferLoad and GNpA)=False -> enableEdge True (A leg); EdgeType=ShiftPtr -> True.
    """
    result = shiftptr_edge_taken(
        tdm_inst=0,
        buffer_load=False,
        af0em=1,
        grvwa=1,
        af1em=1,
        grvwb=1,
        bias_src="A",
        tlua=True,
        tlub=True,
    )
    assert result is True


# ---------------------------------------------------------------------------
# FALSE witnesses — branch is not taken
# ---------------------------------------------------------------------------

def test_false_example_biassrc_d_no_edge():
    """
    z3 false witness 1: BiasSrc=D — neither the A nor B enableEdge clause fires
    -> enableEdge stays False -> predicate False (even though EdgeType=ShiftPtr).
    """
    result = shiftptr_edge_taken(
        tdm_inst=0,
        buffer_load=True,
        af0em=4,
        grvwa=1,
        af1em=1,
        grvwb=4,
        bias_src="D",
        tlua=True,
        tlub=False,
    )
    assert result is False


def test_false_example_tdminst3_edgetype_none():
    """
    z3 false witness 2: TDMInst=3 -> enableTDMA=True AND enableTDMB=True
    -> EdgeType forced to "None" (not "ShiftPtr") -> predicate False
    regardless of enableEdge (which would be True via BiasSrc=A, BufferLoad=False).

    Exercises the EdgeType-override leg of the conjunction.
    """
    result = shiftptr_edge_taken(
        tdm_inst=3,
        buffer_load=False,
        af0em=1,
        grvwa=4,
        af1em=1,
        grvwb=4,
        bias_src="A",
        tlua=True,
        tlub=True,
    )
    assert result is False


# ---------------------------------------------------------------------------
# Boundary pin — TDMInst bit-field semantics
# ---------------------------------------------------------------------------

def test_tdminst_bit_semantics_only_bit0():
    """
    TDMInst=1 -> enableTDMA=True, enableTDMB=False -> EdgeType stays ShiftPtr.
    With BiasSrc=A and BufferLoad=False, enableEdge=True -> predicate True.
    """
    result = shiftptr_edge_taken(
        tdm_inst=1,
        buffer_load=False,
        af0em=1,
        grvwa=1,
        af1em=1,
        grvwb=1,
        bias_src="A",
        tlua=True,
        tlub=True,
    )
    assert result is True


def test_tdminst_bit_semantics_only_bit1():
    """
    TDMInst=2 -> enableTDMA=False, enableTDMB=True -> EdgeType stays ShiftPtr.
    With BiasSrc=B, BufferLoad=False, enableEdge=True -> predicate True.
    """
    result = shiftptr_edge_taken(
        tdm_inst=2,
        buffer_load=False,
        af0em=1,
        grvwa=1,
        af1em=1,
        grvwb=1,
        bias_src="B",
        tlua=True,
        tlub=True,
    )
    assert result is True


def test_guarantee_no_partial_a_suppresses_edge_when_buffered():
    """
    TLUA=True, AF0EM=4, GRVWA=4 -> GNpA=(4%4==0)=True.
    BufferLoad=True and GNpA=True -> (BufferLoad and GNpA)=True -> not(...)=False.
    BiasSrc=A -> A leg does NOT fire.
    BiasSrc=A and no B leg -> enableEdge=False -> predicate False.
    """
    result = shiftptr_edge_taken(
        tdm_inst=0,
        buffer_load=True,
        af0em=4,
        grvwa=4,
        af1em=1,
        grvwb=4,
        bias_src="A",
        tlua=True,
        tlub=False,
    )
    assert result is False
