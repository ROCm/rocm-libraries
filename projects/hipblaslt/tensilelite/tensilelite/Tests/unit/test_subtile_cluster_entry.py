# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check the persistent cluster phase on actual subtile preloop control paths."""
from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.Components.Subtile.LogicalScheduler import LogicalScheduler

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).resolve().parents[1] / "common/streamk/gfx1250/core/sk_bf16gemm_tdm_subtile.yaml"


@pytest.fixture(scope="module")
def entries(tmp_path_factory):
    config = yaml.safe_load(_CONFIG.read_text())
    problem = config["BenchmarkProblems"][0]
    group = problem[2]  # Existing clustered MT96x96, including the stalled K8192 case.
    config["BenchmarkProblems"] = [[problem[0], group]]
    group["ForkParameters"] = [{"PrefetchGlobalRead": [0, 1, 2]}, {"ScheduleIterAlg": [0]}]
    path = tmp_path_factory.mktemp("subtile-entry") / "entry.yaml"
    path.write_text(yaml.safe_dump(config))
    captured = {}
    original = LogicalScheduler._emitPreloopPhase

    def capture(self, writer, kernel, *args):
        def template():
            return [str(inst) for part in self._preloop_emitted for slot in part
                    for em in slot for inst in em.instructions]
        before = template()
        module = original(self, writer, kernel, *args)
        assert template() == before, "entry wait mutated the reusable preloop schedule"
        captured[kernel["PrefetchGlobalRead"]] = str(module)
        return module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LogicalScheduler, "_emitPreloopPhase", capture)
        results = emit_kernels_from_config(
            path, limit=3, arch="gfx1250", canonical=False, expected_fork_count=3)
    assert len(results) == 3 and set(captured) == {0, 1, 2}
    for name, source, error in results:
        assert error == 0, (name, error)
        assert_assembles(source, name)
    return captured


def _entry_path(source, iterations):
    """Follow wave zero through the emitted preloop, including the zero-K exit."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = [op for line in source.splitlines() if (op := line.split("//", 1)[0].strip())]
    labels = {op[:-1]: i for i, op in enumerate(code) if op.endswith(":")}
    values = {"s[sgprLoopCounterL]": iterations, "s[sgprWaveIdx]": 0}
    pc, scc = 0, None
    for _ in range(len(code) + 1):
        if pc == len(code):
            return
        op = code[pc]
        pc += 1
        compare = re.fullmatch(r"s_cmp_(eq|le)_u32 (s\[\w+\]), (\d+)", op)
        if compare:
            lhs, rhs = values[compare[2]], int(compare[3])
            scc = lhs == rhs if compare[1] == "eq" else lhs <= rhs
        elif op.startswith("s_cbranch_scc"):
            branch, target = op.split()
            assert scc is not None
            if scc == (branch == "s_cbranch_scc1"):
                if target not in labels:
                    return  # Tail, NLL or NGLL owns the remainder.
                pc = labels[target]
        yield op
    pytest.fail("preloop branch walk did not terminate")


@pytest.mark.parametrize("pgr", [0, 1, 2])
@pytest.mark.parametrize("iterations", [0, 64])
def test_subtile_entry_consumes_persistent_arrival(entries, pgr, iterations):
    pending, joined, loads = True, False, 0  # Prologue/re-entry posts one arrival.
    for op in _entry_path(entries[pgr], iterations):
        if op == "s_barrier_signal -3":
            assert not pending, "second cluster arrival before entry wait"
            assert joined, "new cluster arrival before all-wave rendezvous"
            pending, joined = True, False
        elif op == "s_barrier_wait -3":
            assert pending, "cluster wait without an arrival"
            pending, joined = False, False
        elif op == "s_barrier_wait -1":
            joined = True
        elif op.startswith("tensor_load_to_lds"):
            assert not pending, "tensor load before entry cluster wait"
            loads += 1
    assert not pending, "zero-iteration exit leaves the cluster arrival pending"
    assert bool(loads) == (pgr > 0 and iterations > 0)
