# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen checks for ScalarLoadB kernels (B read with scalar loads, used as the scalar FMA operand).

The main check follows the emitted assembly symbolically: it tracks which element of B every
ring register holds and requires each FMA to read B[k][column] for the summation iteration k
and the column of the accumulator it updates.
"""
from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).resolve().parents[1] / "common/sgemm_scalar_load_b_asm.yaml"
_ARCH = "gfx1151"
# Problem groups of _CONFIG: (index, number of forked solutions)
_GROUPS = {"TN": (0, 12), "TN prefetch": (1, 2), "TT": (2, 12), "TT prefetch": (3, 1), "NN": (4, 4), "NT": (5, 4)}


def _emit(group):
    index, forks = _GROUPS[group]
    results = emit_kernels_from_config(_CONFIG, limit=forks, arch=_ARCH, canonical=False,
                                       problem_index=index, expected_fork_count=forks)
    assert len(results) == forks, group
    for name, _, error in results:
        assert error == 0, (name, error)
    return results


@pytest.fixture(scope="module", params=sorted(_GROUPS))
def kernels(request):
    return _emit(request.param)


def _instructions(source):
    lines = [line.split("//", 1)[0].strip() for line in source.splitlines()]
    return [line for line in lines if line and not line.startswith("/*")]


_LOAD = re.compile(r"s_load_b(\d+) s\[sgprScalarB(\d+)(?::[^\]]*)?\], s\[sgprScalarPtrB:sgprScalarPtrB\+1\], (\S+)")
_ADVANCE = re.compile(r"s_add_u32 s\[sgprScalarPtrB\+0\], s\[sgprScalarPtrB\+0\], (\S+)")
_COLOFF = re.compile(r"s_mul_i32 s\[sgprScalarColOffB(\d+)\], s\d+, (\d+)")
_MAC = re.compile(r"v_(?:dual_)?fmac_f32 v\[vgprValuC\+(\d+)\], s\[sgprScalarB(\d+)\+(\d+)\], v\[vgprValuA_X\d+_I0\+(\d+)\]")


class _Ring:
    """Symbolic contents of the B SGPRs while walking the summation code of one kernel."""

    def __init__(self, source):
        tt = re.search(r"/\* ThreadTile= (\d+) x (\d+) \*/", source)
        sg = re.search(r"/\* SubGroup= (\d+) x (\d+) \*/", source)
        self.tt0, self.tt1 = int(tt.group(1)), int(tt.group(2))
        self.sg1 = int(sg.group(2))
        self.vwb = int(re.search(r"/\* VectorWidthB=(\d+) \*/", source).group(1))
        self.tlu = "Bjlk" in re.search(r"\.globl (\S+)", source).group(1)
        self.k = 0             # summation iteration the stream pointer is at
        self.regs = {}         # (group, offset) -> (k, column)
        self.colOff = {}       # ScalarColOffB index -> column
        self.macs = 0          # FMA products so far
        self.loadedTo = 0      # summation iterations fetched so far (exclusive)
        self.seen = set()
        self.inFlight = set()  # registers loaded since the last full SMEM/LDS drain

    def column(self, j):
        return (j // self.vwb) * self.sg1 * self.vwb + j % self.vwb

    def step(self, line):
        m = _COLOFF.match(line)
        if m:
            self.colOff[int(m.group(1))] = int(m.group(2))
            return
        m = _LOAD.match(line)
        if m:
            width, group, offset = int(m.group(1)) // 32, int(m.group(2)), m.group(3)
            if offset.startswith("s[sgprScalarColOffB"):
                col = self.colOff[int(re.search(r"ColOffB(\d+)", offset).group(1))]
            else:
                col = int(offset, 0) // 4
            for i in range(width):
                # contiguous along the tile index: adjacent columns; otherwise adjacent iterations
                self.regs[(group, i)] = (self.k, col + i) if self.tlu else (self.k + i, col)
                self.inFlight.add((group, i))
            self.loadedTo = max(self.loadedTo, self.k + (1 if self.tlu else width))
            return
        if line.startswith("s_waitcnt lgkmcnt(0)"):
            # SMEM returns out of order with LDS, so only a full drain covers a scalar load
            self.inFlight.clear()
            return
        m = _ADVANCE.match(line)
        if m:
            step = m.group(1)
            self.k += 1 if step.startswith("s[sgprScalarStrideB") else int(step, 0) // 4
            return
        for m in _MAC.finditer(line.replace("::", "\n")) if "fmac_f32" in line else ():
            c, group, offset, a = (int(x) for x in m.groups())
            i, j = c % self.tt0, c // self.tt0
            k = self.macs // (self.tt0 * self.tt1)
            assert a == i, line
            assert self.regs.get((group, offset)) == (k, self.column(j)), \
                "iteration %u column %u reads %s: %s" % (k, self.column(j), self.regs.get((group, offset)), line)
            assert (group, offset) not in self.inFlight, "read before its load was waited for: " + line
            assert (k, c) not in self.seen, line
            self.seen.add((k, c))
            self.macs += 1


def _sections(insts):
    """(setup, unrolled loop body, drain) of the summation."""
    begin = insts.index("label_LoopBeginL:")
    back = next(i for i in range(begin, len(insts)) if insts[i] == "s_cbranch_scc0 label_LoopBeginL")
    end = next(i for i in range(back, len(insts)) if insts[i].startswith("label_Summation_End"))
    return insts[:begin], insts[begin:back], insts[back:end]


def _loop_blocks(source, loop):
    """Unrolled blocks in one trip of the loop: ExpandPointerSwap unrolls two."""
    ring = _Ring(source)
    loopIters = int(re.search(r"\.set DepthU, (\d+)", source).group(1))
    blocks = sum(len(_MAC.findall(line)) for line in loop) // (ring.tt0 * ring.tt1 * loopIters)
    assert blocks in (1, 2), blocks
    return blocks


def test_kernels_assemble(kernels):
    for name, source, _ in kernels:
        assert_assembles(source, name)


def test_every_fma_reads_its_b_element(kernels):
    for name, source, _ in kernels:
        setup, loop, drain = _sections(_instructions(source))
        loopIters = int(re.search(r"\.set DepthU, (\d+)", source).group(1))
        loopBlocks = _loop_blocks(source, loop)
        drained = set()
        for trips in (1, 3):
            ring = _Ring(source)
            for line in setup + loop * trips + drain:
                ring.step(line)
            iterations = ring.macs // (ring.tt0 * ring.tt1)
            # every accumulator once per iteration, in whole unrolled blocks
            assert ring.macs == iterations * ring.tt0 * ring.tt1, name
            assert iterations % loopIters == 0, (name, iterations)
            drained.add(iterations // loopIters - trips * loopBlocks)
            # scalar loads fault past the end of B: nothing may be fetched beyond the summation range
            assert ring.loadedTo == iterations, (name, ring.loadedTo, iterations)
        # the blocks after the loop are the PrefetchGlobalRead ones that run without global reads
        assert drained in ({1}, {2}), (name, drained)


def test_every_block_issues_one_warming_load():
    # Waits count the warming load, so no block may skip it.
    for group in ("TN prefetch", "TT prefetch"):
        for name, source, _ in _emit(group):
            if "flat_load_b32" not in source:
                continue
            _, loop, drain = _sections(_instructions(source))
            blocks = _loop_blocks(source, loop)
            assert sum(line.startswith("flat_load_b32") for line in loop) == blocks, name
            assert not any(line.startswith("flat_load_b32") for line in drain), name


def test_b_never_touches_vgprs_or_lds(kernels):
    for name, source, _ in kernels:
        for symbol in ("vgprValuB", "vgprG2LB", "vgprLocalReadAddrB", "vgprLocalWriteAddrB", "vgprGlobalReadOffsetB"):
            for line in _instructions(source):
                assert symbol not in line or line.startswith(".set"), (name, line)


def test_register_budget_keeps_eight_waves_per_simd():
    # The 512x128 tile runs 16 waves per workgroup; 192 VGPRs is what lets two of them share a
    # gfx1151 workgroup processor.
    for name, source, _ in _emit("TN") + _emit("TT"):
        assert int(re.search(r"\.amdhsa_next_free_vgpr (\d+)", source).group(1)) <= 192, name
        assert int(re.search(r"\.amdhsa_next_free_sgpr (\d+)", source).group(1)) <= 106, name


def test_parked_sgprs_are_undefined_during_the_summation():
    # The ring reuses the store arguments' registers; a use in between must not assemble.
    name, source, _ = _emit("TN")[-1]
    insts = _instructions(source)
    for sgpr in ("Alpha", "Beta", "AddressD", "StridesD", "StrideD1J"):
        undef = insts.index(".set sgpr%s, UNDEF" % sgpr)
        redef = next(i for i in range(undef, len(insts)) if insts[i].startswith(".set sgpr%s, " % sgpr)
                     and "UNDEF" not in insts[i])
        assert undef < insts.index("label_LoopBeginL:") < redef, sgpr
        assert not any("sgpr%s" % sgpr in line for line in insts[undef + 1:redef]), sgpr


def _config_with(tmp_path, **forks):
    """One-problem config: the TN group with fork parameters replaced."""
    config = yaml.safe_load(_CONFIG.read_text())
    problem = config["BenchmarkProblems"][0]
    params = problem[1]["ForkParameters"]
    base = {"ScalarLoadB": [8], "UseDualFMAC": [True], "PrefetchGlobalRead": [1]}
    for key, value in {**base, **forks}.items():
        for entry in params:
            if key in entry:
                entry[key] = value
                break
        else:
            params.append({key: value})
    config["BenchmarkProblems"] = [problem]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _valid(path):
    return [s for s in solutions_from_config(path, arch=_ARCH) if s["Valid"]]


def test_baseline_config_is_accepted(tmp_path):
    assert len(_valid(_config_with(tmp_path))) == 1


@pytest.mark.parametrize("forks", [
    {"WorkGroup": [[16, 16, 1]]},                 # B differs between the lanes of a wave
    {"AssertSummationElementMultiple": [1]},      # would need a tail loop
    {"AssertFree1ElementMultiple": [1]},          # would need partial tiles along B
    {"StaggerU": [8]},                            # B is read in summation order
    {"GlobalSplitU": [2]},                        # the summation range has to be contiguous
    {"OptNoLoadLoop": [1]},                       # its checks read the parked store arguments
    {"ScalarLoadB": [16]},                        # does not divide DepthU
    {"PrefetchLocalRead": [0]},
], ids=lambda forks: next(iter(forks)))
def test_unsupported_configs_are_rejected(tmp_path, forks):
    assert _valid(_config_with(tmp_path, **forks)) == []
