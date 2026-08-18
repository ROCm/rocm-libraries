################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
################################################################################

"""Registration tests for the ReuseAcrossPersistent (RAP) solution parameter.

RAP keeps the A / MXSA operand resident in VGPRs across persistent-loop
iterations. These tests pin the registries a new solution parameter has to
reach; the naming registry in particular is load-bearing, because a RAP=1 kernel
that hashes to the same name as its RAP=0 twin is silently dropped as a
duplicate.

The runtime problem predicates RAP emits (M, N and the exact K) and the
RAP0 / RAP1 name tag are covered by the LibraryIO and SolutionClass
characterization snapshots, which build real solutions.

The store-neutrality guard is exercised directly here rather than through a
generated kernel: every config tuned so far stays within one store batch, so a
kernel-level test would never reach the rejecting branch.
"""

from types import SimpleNamespace

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.ValidParameters import validParameters
from Tensile.KernelWriter import KernelWriter
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.SolutionStructs.Naming import getParameterNameAbbreviation
from Tensile.SolutionStructs.Solution import validateParameterTypes


def test_rap_is_valid_solution_parameter():
    assert validParameters["ReuseAcrossPersistent"] == [0, 1]
    assert defaultSolution["ReuseAcrossPersistent"] == 0
    assert validateParameterTypes({"ReuseAcrossPersistent": 1}) == []


def test_rap_name_abbreviation_does_not_collide_with_pap():
    assert getParameterNameAbbreviation("ReuseAcrossPersistent") == "RAP"
    assert getParameterNameAbbreviation("PrefetchAcrossPersistent") == "PAP"


class _StoreGuardWriter:
    """Just enough writer to run the two store-neutrality methods standalone."""

    rapStoreWithheldVgprs = KernelWriter.rapStoreWithheldVgprs
    rapCheckStoreNeutrality = KernelWriterAssembly.rapCheckStoreNeutrality

    def __init__(self):
        # ValuA occupies [0, 192) and MXSA [0, 12): 204 vgpr held across the store.
        self.states = SimpleNamespace(
            a=SimpleNamespace(startVgprValu=0),
            b=SimpleNamespace(startVgprValu=192),
            mxsa=SimpleNamespace(startVgprValu=0, numVgprValu=12),
            lastValuMXSAB=12,
            overflowedResources=0,
            rapStoreRejectHint="",
        )

    def isReuseAcrossPersistentEnabled(self, kernel):
        return True


_GUARD_KERNEL = {"_RAPNumResidentKTiles": 3, "DepthU": 256}
_GUARD_SS = SimpleNamespace(numVgprsPerElement=4)
_GUARD_ELEMENTS = [None] * 128


def _runGuard(availForBatching, numBatches, beta=True, edge=False):
    writer = _StoreGuardWriter()
    writer.rapCheckStoreNeutrality(_GUARD_KERNEL, _GUARD_SS, _GUARD_ELEMENTS,
                                   numBatches, availForBatching, beta, edge)
    return writer.states


def test_rap_store_guard_accepts_a_neutral_store():
    # 600 vgpr -> 150 elements/batch, so 128 elements still fit in one batch;
    # without the residency it would be 201, also one batch.
    states = _runGuard(availForBatching=600, numBatches=1)
    assert states.overflowedResources == 0
    assert states.rapStoreRejectHint == ""


def test_rap_store_guard_rejects_an_extra_batch_and_reports_the_k_limit():
    # 400 vgpr -> 100 elements/batch -> 2 batches, but 400+204=604 would give
    # 151 per batch -> 1 batch, so the residency is what split the store.
    states = _runGuard(availForBatching=400, numBatches=2)
    assert states.overflowedResources == 9
    # (604 - 128*4) // (204 // 3) = 1 resident k-tile, i.e. K=256 against K=768.
    assert "largest store-neutral K is 256" in states.rapStoreRejectHint
    assert "needs 768" in states.rapStoreRejectHint


def test_rap_store_guard_only_gates_the_no_edge_beta_variant():
    # RAP's problem predicates already exclude edge tiles, and beta=0 is the
    # cheaper variant, so neither is allowed to reject the solution.
    assert _runGuard(400, 2, beta=True, edge=True).overflowedResources == 0
    assert _runGuard(400, 2, beta=False, edge=False).overflowedResources == 0
