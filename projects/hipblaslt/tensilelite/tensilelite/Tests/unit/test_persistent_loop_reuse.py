# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise resident-tile reuse through the extracted persistent lifecycle."""

from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

from config_harness import emit_kernels_from_config
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
from Tensile.Components.PersistentLoop import PersistentKernelState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mx_a,mx_b", [(False, False), (True, False), (False, True), (True, True)])
def test_reuse_state_restore_preserves_live_registers_and_discards_first_copy_offsets(mx_a, mx_b):
    writer = PersistentKernelState()
    writer.states = SimpleNamespace(
        freeSgprVarPool={"tmp": [4]}, skipLocalWrite=[False], liveNames={"A"},
        numReadsPerIterA=6,
    )
    writer.vgprPool = RegisterPool(16, RegisterType.Vgpr, False)
    writer.sgprPool = RegisterPool(16, RegisterType.Sgpr, False)
    for pool in (writer.vgprPool, writer.sgprPool):
        pool.add(0, 16, "available")
        assert pool.checkOut(4, "live across compute copies") == 0
    writer.sgprs = {"Live": 0}
    kernel = {"ProblemType": {"MXBlockA": mx_a, "MXBlockB": mx_b}}
    a = {"localReadOffset": 8, "MX": {"localReadOffset": 16}}
    b = {"localReadOffset": 24, "MX": {"localReadOffset": 32}}
    tensors = [a, b] + ([a["MX"]] if mx_a else []) + ([b["MX"]] if mx_b else [])
    offsets = [tensor["localReadOffset"] for tensor in tensors]
    snapshot = writer.rapSnapshotEmitterState(kernel, a, b)

    writer.states.freeSgprVarPool["tmp"].append(5)
    writer.states.skipLocalWrite[0] = True
    writer.states.liveNames.add("B")
    writer.states.numReadsPerIterA = 0
    writer.sgprs["Live"] = 12
    for tensor in tensors:
        tensor["localReadOffset"] = 99
        tensor["localWriteSwapByteOffset"] = 4096
    for pool in (writer.vgprPool, writer.sgprPool):
        pool.checkIn(0)
        pool.appendPool(32)

    writer.rapRestoreEmitterState(kernel, a, b, snapshot)
    assert writer.states.freeSgprVarPool == {"tmp": [4]}
    assert writer.states.skipLocalWrite == [False]
    assert writer.states.liveNames == {"A"}
    assert writer.states.numReadsPerIterA == 6
    assert writer.sgprs == {"Live": 0}
    assert [tensor["localReadOffset"] for tensor in tensors] == offsets
    assert all("localWriteSwapByteOffset" not in tensor for tensor in tensors)
    for pool in (writer.vgprPool, writer.sgprPool):
        assert pool.size() == 32
        # The restored allocation remains live, while the first copy's peak
        # size remains available for the second copy's temporary allocations.
        assert pool.checkOut(4, "second compute copy") == 4
        pool.checkIn(4)
        pool.checkIn(0)


@pytest.mark.parametrize("prefetch", [0, 1])
def test_reuse_compute_copies_refill_batches_and_drain_before_store(tmp_path, prefetch):
    # The maintained MX configuration exercises resident A and scale registers.
    # Keep one tile shape so this test emits exactly one complete kernel.
    fixture = Path(__file__).parents[1] / "common/gemm/gfx12/rap_gfx1250.yaml"
    config = yaml.safe_load(fixture.read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    for parameter in config["BenchmarkProblems"][0][1]["ForkParameters"]:
        if "MatrixInstruction" in parameter:
            parameter["MatrixInstruction"] = parameter["MatrixInstruction"][:1]
        elif "ReuseAcrossPersistent" in parameter:
            parameter["ReuseAcrossPersistent"] = [1]
        elif "PrefetchAcrossPersistent" in parameter:
            parameter["PrefetchAcrossPersistent"] = [prefetch]
    path = tmp_path / "reuse.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    kernels = emit_kernels_from_config(path, arch="gfx1250", limit=1)
    assert len(kernels) == 1
    _, source, error = kernels[0]
    (tmp_path / "reuse.s").write_text(source)
    assert error == 0

    # The first tile fills A; later tiles enter the reuse copy. A conditional
    # branch must keep that copy reachable to the wait/barrier insertion pass.
    fill, reuse = source.split("label_RAP_IterN:", 1)
    reuse, store = reuse.split("label_RAP_StoreJoin:", 1)
    assert re.search(r"s_cbranch_scc1\s+label_RAP_StoreJoin", fill)
    batch_check = re.search(r"s_cmp_eq_u32[^\n]*s\[sgprRAPResidentBatch\]", reuse)
    assert batch_check is not None
    refill = reuse[batch_check.start():reuse.index("re-init WaveIdx")]
    assert re.search(r"s_cbranch_scc1\s+label_NoBranch", refill)
    assert re.search(r"s_add_i32[^\n]*label_PersistentLoopStart", refill)
    assert "s_setpc_b64" in refill
    assert re.search(
        r"s_wait_dscnt\s+0\b[^\n]*RAP: drain a small-K exit's unconsumed local reads", store)
    # Both copies must define distinct loop labels after emitter-state restore.
    labels = re.findall(r"^(label_\w+):", source, re.MULTILINE)
    assert len(labels) == len(set(labels))
    assert any("_RAPIterN" in label for label in labels)
