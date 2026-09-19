# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen test: Stream-K partials write must advance its accumulator source.

Under ``LocalSplitU > 1`` the reduced accumulators live in consecutive ValuC
VGPRs, and ``AsmStoreState.setupStoreElementsForBatch``'s ``elementStartIdx``
argument is the only thing that advances the source register base from one store
batch to the next (``elementSumIdx`` is rebuilt per batch). The Stream-K producer
(``StreamK.partialsWriteProcedure`` -> ``partialsWriteBatch``) and the Stream-K
consumer (``fixupStep`` -> ``fixupBatch``) walk the same workspace with the same
addresses, so they must read/write the same accumulators for the same batch
index. If the producer omits ``elementStartIdx``, every batch re-stores batch 0's
accumulators while the consumer sums the correct ones, and the tile owner gets
wrong results.

The check is static, over the emitted assembly for a real Stream-K solution
(``--build-only``, ISA pinned to gfx950 so no GPU is needed):

  * ``Partials Write Batch #N`` must source ValuC registers
    ``N * elementsPerBatch * gwvw`` .. ``+ elementsPerBatch * gwvw - 1``.
  * Those must be exactly the registers ``Fixup Batch #N`` accumulates into.

The ``LocalSplitU == 1`` solution in the same config is the control: it also
stores in several batches, but its accumulators are copied out of AGPRs into the
same scratch VGPRs every batch, so ``elementStartIdx`` has nothing to shift.
"""

import ast
import re
from pathlib import Path

import pytest

from Tensile import Tensile

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).parent / "test_data" / "streamk_lsu_partials_write.yaml"
_ASM_STORE_STATE_PY = Path(__file__).resolve().parents[2] / "AsmStoreState.py"

_VALUC_SET = re.compile(r"^\.set vgprValuC,\s*(\d+)")
_ELEMS_PER_BATCH = re.compile(r"elementsPerBatch=(\d+)")
_BATCH_HEAD = re.compile(r"/\* (Partials Write|Fixup)(?:[A-Za-z ]*?) Batch #(\d+) ")
_GWVW = re.compile(r":vw(\d+)\)")
# Producer: "buffer_store_dword v8, ..." / "buffer_store_dwordx2 v[8:9], ..."
_STORE_SRC = re.compile(r"^buffer_store_dword\w* (?:v(\d+)|v\[(\d+):(\d+)\]),.*// addStore")
# Consumer: "v_add_f32 v[vgprValuC+2], v[vgprValuC+2], v17 // accum partials"
_ACCUM_DST = re.compile(r"^v_\w+ v\[vgprValuC\+(\d+)\].*// accum partials")


def _funcDef(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("%s not found in %s" % (name, path.name))


def _mentionsConst(node: ast.AST, value) -> bool:
    return any(isinstance(n, ast.Constant) and n.value == value for n in ast.walk(node))


class _Batches:
    """ValuC-relative accumulator registers per batch, plus the batch geometry."""

    def __init__(self):
        self.regs = {}            # batchIdx -> [valuC-relative register]
        self.elementsPerBatch = None
        self.gwvw = None


def _parse(text: str):
    """Extract the Partials Write and Fixup batches from one kernel's assembly."""
    partials, fixup = _Batches(), _Batches()
    valuC = 0
    elementsPerBatch = None
    batches = None
    current = None

    for line in text.splitlines():
        line = line.strip()

        m = _VALUC_SET.match(line)
        if m:
            valuC = int(m.group(1))
            continue

        m = _ELEMS_PER_BATCH.search(line)
        if m:
            elementsPerBatch = int(m.group(1))

        m = _BATCH_HEAD.search(line)
        if m:
            batches = partials if m.group(1) == "Partials Write" else fixup
            current = batches.regs.setdefault(int(m.group(2)), [])
            batches.elementsPerBatch = elementsPerBatch
            continue

        if current is None:
            continue

        m = _GWVW.search(line)
        if m:
            batches.gwvw = int(m.group(1))
            continue

        m = _STORE_SRC.match(line)
        if m:
            if m.group(1) is not None:
                current.append(int(m.group(1)) - valuC)
            else:
                current.extend(range(int(m.group(2)) - valuC, int(m.group(3)) - valuC + 1))
            continue

        m = _ACCUM_DST.match(line)
        if m:
            current.append(int(m.group(1)))

    return partials, fixup


def _expected(batches: _Batches, batchIdx: int):
    stride = batches.elementsPerBatch * batches.gwvw
    return list(range(batchIdx * stride, (batchIdx + 1) * stride))


_ASSEMBLY_CACHE = {}


@pytest.fixture
def streamk_assembly(tensile_args: list[str], tmp_path_factory) -> dict:
    """Assembly of the LocalSplitU=4 and LocalSplitU=1 Stream-K solutions.

    ``tensile_args`` is function-scoped, so the (slow) codegen run is memoised
    here rather than by a module-scoped fixture.
    """
    if _ASSEMBLY_CACHE:
        return _ASSEMBLY_CACHE

    output_dir = tmp_path_factory.mktemp("streamk_lsu") / "output"
    Tensile.Tensile([
        str(_CONFIG), str(output_dir), "--build-only",
        "--global-parameters", "KeepBuildTmp=True",
        *tensile_args,
    ])

    kernels = {}
    for path in output_dir.rglob("Cijk_*.s"):
        text = path.read_text()
        if "Partials Write Batch #" not in text:
            continue
        kernels["lsu" if "LocalSplitU Reduction" in text else "nolsu"] = text

    assert "lsu" in kernels, "no LocalSplitU Stream-K kernel was generated"
    assert "nolsu" in kernels, "no LocalSplitU=1 Stream-K kernel was generated"
    _ASSEMBLY_CACHE.update(kernels)
    return kernels


class TestLocalSplitUPartialsWrite:
    def test_partials_write_spans_several_batches(self, streamk_assembly):
        # Guards the premise of every other assertion here: a single-batch store
        # would make the accumulator advance unobservable.
        partials, fixup = _parse(streamk_assembly["lsu"])
        assert len(partials.regs) > 1, "the config no longer produces a multi-batch store"
        assert sorted(partials.regs) == sorted(fixup.regs)

    def test_partials_write_advances_accumulators_per_batch(self, streamk_assembly):
        # The bug: without elementStartIdx every batch re-stores batch 0's
        # accumulators, so batch #N stores the same registers as batch #0.
        partials, _ = _parse(streamk_assembly["lsu"])
        for batchIdx, regs in sorted(partials.regs.items()):
            assert regs == _expected(partials, batchIdx), (
                "Partials Write Batch #%u sources ValuC %s, expected %s"
                % (batchIdx, regs, _expected(partials, batchIdx))
            )

    def test_partials_write_matches_fixup(self, streamk_assembly):
        # Producer and consumer address the same workspace region per batch, so
        # they must also agree on which accumulator that region holds.
        partials, fixup = _parse(streamk_assembly["lsu"])
        for batchIdx, regs in sorted(partials.regs.items()):
            assert regs == fixup.regs[batchIdx], (
                "Partials Write Batch #%u sources ValuC %s but Fixup Batch #%u "
                "accumulates into %s" % (batchIdx, regs, batchIdx, fixup.regs[batchIdx])
            )


class TestNoLocalSplitUIsUnaffected:
    def test_partials_write_reuses_the_same_registers(self, streamk_assembly):
        # Without LocalSplitU the accumulators are copied out of AGPRs into the
        # same scratch VGPRs every batch, so the store source does not move --
        # elementStartIdx has nothing to shift.
        partials, _ = _parse(streamk_assembly["nolsu"])
        assert len(partials.regs) > 1, "the control solution no longer stores in batches"
        distinct = {tuple(regs) for regs in partials.regs.values()}
        assert len(distinct) == 1, (
            "LocalSplitU=1 partials write must source one fixed register set, got %s"
            % sorted(distinct)
        )

    def test_elementStartIdx_is_only_read_under_localsplitu(self):
        # Structural no-op guarantee, independent of any one solution: the only
        # reader of elementStartIdx in setupStoreElementsForBatch sits inside the
        # LocalSplitU > 1 branch.
        fn = _funcDef(_ASM_STORE_STATE_PY, "setupStoreElementsForBatch")
        guarded = {
            id(node)
            for branch in fn.body
            for outer in ast.walk(branch)
            if isinstance(outer, ast.If) and _mentionsConst(outer.test, "LocalSplitU")
            for stmt in outer.body
            for node in ast.walk(stmt)
        }
        uses = [n for n in ast.walk(fn)
                if isinstance(n, ast.Name) and n.id == "elementStartIdx"
                and isinstance(n.ctx, ast.Load)]
        assert uses, "setupStoreElementsForBatch never reads elementStartIdx"
        assert all(id(n) in guarded for n in uses), (
            "elementStartIdx is read outside the LocalSplitU > 1 branch"
        )
