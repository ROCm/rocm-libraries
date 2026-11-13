# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen checks for PrefetchGlobalRead=2 with two sets of G2L registers.

With ExpandPointerSwap and unscheduled global reads/local writes, the two copies of the unroll
loop alternate between two G2L sets: a block's reads are issued at the top of a block and stored
near the end of the next one. The main check executes the vector-memory instructions of the
emitted summation symbolically for several K: it tracks which block every G2L set holds and which
loads a wait has covered, and requires every local write to store the next block from a set whose
read has landed.
"""
from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config, solutions_from_config
from test_scalar_load_b_codegen import _CONFIG as _SCALAR_B_CONFIG, _instructions

pytestmark = pytest.mark.unit
_CONFIG = Path(__file__).resolve().parents[1] / "common/sgemm_pgr2_double_g2l_asm.yaml"
_ARCH = "gfx1151"
# (config, problem index, forked solutions)
_GROUPS = {"NN": (_CONFIG, 0, 4), "TN": (_CONFIG, 1, 4), "TN scalar B": (_SCALAR_B_CONFIG, 6, 2)}

_READ = re.compile(r"buffer_load_b\d+ v\[vgprG2L([AB]2?)\+")
_WRITE = re.compile(r"ds_store_b\d+ v\[vgprLocalWriteAddr[AB]\+\d+\], v\[vgprG2L([AB]2?)\+")
_MOVE = re.compile(r"v_mov_b32 v\[vgprG2L([AB])\+\d+\], v\[vgprG2L([AB]2)\+\d+\]")
# after the first read every add to the read pointer steps one block (also with StaggerU)
_ADVANCE = re.compile(r"s_add_u32 s\[sgprSrd([AB])\+0\], s\[sgprSrd\1\+0\], ")
_WAIT = re.compile(r"s_waitcnt .*vmcnt\((\d+)\)")
_COUNTER = re.compile(r"s_cmp_(eq|le|lt|gt|ge)_[ui]32 s\[sgprLoopCounterL\], (0x[0-9a-fA-F]+|\d+)$")
_COMPARE = {"eq": int.__eq__, "le": int.__le__, "lt": int.__lt__, "gt": int.__gt__, "ge": int.__ge__}


@pytest.fixture(scope="module", params=sorted(_GROUPS))
def kernels(request):
    config, index, forks = _GROUPS[request.param]
    results = emit_kernels_from_config(config, limit=forks, arch=_ARCH, canonical=False,
                                       problem_index=index, expected_fork_count=forks)
    assert len(results) == forks
    assert all(error == 0 for _, _, error in results)
    return results


class _Summation:
    """Symbolic run of the summation of one kernel for a K of `blocks` unrolled blocks."""

    def __init__(self, insts, blocks):
        self.insts = insts
        self.labels = {line[:-1]: i for i, line in enumerate(insts) if line.endswith(":")}
        self.counter = blocks
        self.scc = None
        self.block = {"A": 0, "B": 0}      # block the read pointer of a tensor is at
        self.inFlight = []                 # issued vector loads, oldest first: (G2L set or None, block)
        self.holds = {}                    # G2L set -> block
        self.batches = {"A": 0, "B": 0}    # local-write batches so far
        self.writing = None                # tensor of the local-write batch being walked

    def _tracked(self, line):
        return any(p.match(line) for p in (_READ, _WRITE, _MOVE, _ADVANCE, _WAIT)) or line.startswith("flat_load_b")

    def _target(self, pc, line):
        """Index to continue at if the branch at pc is taken, else None."""
        op, target = line.split()[0], self.labels[line.split()[1]]
        if op == "s_branch":
            return target
        if self.scc is None:
            # a branch on something else (e.g. GlobalSplitU): it must not skip anything tracked here
            assert target > pc and not any(self._tracked(l) for l in self.insts[pc:target]), line
            return None
        return target if self.scc == (op == "s_cbranch_scc1") else None

    def run(self):
        pc = next(i for i, line in enumerate(self.insts) if _READ.match(line))
        end = self.labels["label_toPGR1"]   # the loops after it neither read nor local-write
        while pc != end:
            line = self.insts[pc]
            pc += 1
            if line.startswith(("s_cbranch_scc", "s_branch")):
                target = self._target(pc - 1, line)
                pc = pc if target is None else target
                continue
            counter = _COUNTER.match(line)
            self.scc = _COMPARE[counter.group(1)](self.counter, int(counter.group(2), 0)) if counter else None
            if line.startswith("s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1"):
                self.counter -= 1
            self._step(line)
        return self

    def _step(self, line):
        write = _WRITE.match(line)
        if write:
            regs = write.group(1)
            if self.writing != regs[0]:
                self.batches[regs[0]] += 1
            # the n-th batch stores block n-1: the block the next unrolled block multiplies
            assert self.holds.get(regs) == self.batches[regs[0]] - 1, \
                "batch %u stores block %s: %s" % (self.batches[regs[0]], self.holds.get(regs), line)
            assert all(dst != regs for dst, _ in self.inFlight), "stored while a read of it is in flight: " + line
        self.writing = write.group(1)[0] if write else None
        read, move, advance, wait = _READ.match(line), _MOVE.match(line), _ADVANCE.match(line), _WAIT.match(line)
        if read:
            self.inFlight.append((read.group(1), self.block[read.group(1)[0]]))
            self.holds.pop(read.group(1), None)
        elif line.startswith("flat_load_b"):
            self.inFlight.append((None, None))
        elif move:
            assert all(dst != move.group(2) for dst, _ in self.inFlight), "moved while its read is in flight: " + line
            self.holds[move.group(1)] = self.holds[move.group(2)]
        elif advance:
            self.block[advance.group(1)] += 1
        elif wait:
            # vector loads return in order
            while len(self.inFlight) > int(wait.group(1)):
                regs, block = self.inFlight.pop(0)
                if regs is not None:
                    self.holds[regs] = block


def test_kernels_assemble(kernels):
    for name, source, _ in kernels:
        assert_assembles(source, name)


def test_every_local_write_stores_a_landed_read_of_the_next_block(kernels):
    for name, source, _ in kernels:
        insts = _instructions(source)
        tensors = sorted({m.group(1)[0] for m in map(_READ.match, insts) if m})
        # 1 and 2 blocks never enter the loop; 3 and 5 leave it after the first copy, 4 and 6 after the second
        for blocks in range(1, 7):
            run = _Summation(insts, blocks).run()
            for tc in tensors:
                # every block was stored once and the read pointer stands behind the last block
                assert run.batches[tc] == blocks, (name, blocks, tc, run.batches)
                assert run.block[tc] == blocks, (name, blocks, tc, run.block)
            # nothing may still write a register after the summation
            assert run.inFlight == [], (name, blocks, run.inFlight)


def test_reads_stay_in_flight_across_the_local_write(kernels):
    # The point of the second set: the wait before a local write leaves this block's reads outstanding.
    for name, source, _ in kernels:
        insts = _instructions(source)
        begin = insts.index("label_LoopBeginL:")
        store = next(i for i in range(begin, len(insts)) if _WRITE.match(insts[i]))
        wait = next(i for i in range(store, begin, -1) if _WAIT.match(insts[i]))
        reads = sum(bool(_READ.match(line)) for line in insts[begin:store])
        assert reads > 0, name
        assert int(_WAIT.match(insts[wait]).group(1)) >= reads, (name, insts[wait])


def _plain_config(tmp_path, **forks):
    """The NN group of _CONFIG with fork parameters replaced."""
    config = yaml.safe_load(_CONFIG.read_text())
    problem = config["BenchmarkProblems"][0]
    params = problem[1]["ForkParameters"]
    for key, value in {"UseDualFMAC": [False], "ThreadTile": [[4, 4]], **forks}.items():
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


def _expands(tmp_path, **forks):
    solutions = [s for s in solutions_from_config(_plain_config(tmp_path, **forks), arch=_ARCH) if s["Valid"]]
    assert len(solutions) == 1
    return solutions[0]["ExpandPointerSwap"], solutions[0]["PGR2DoubleG2L"]


def test_unscheduled_pgr2_keeps_expand_pointer_swap(tmp_path):
    assert _expands(tmp_path) == (True, True)


@pytest.mark.parametrize("forks", [
    {"ScheduleGlobalRead": [1]},      # each read is issued after its local write: one set suffices
    {"ScheduleLocalWrite": [1]},
    {"ExpandPointerSwap": [False]},   # one loop copy cannot alternate
], ids=lambda forks: next(iter(forks)))
def test_other_pgr2_kernels_keep_one_set(tmp_path, forks):
    # as before this change: ExpandPointerSwap is dropped for f32 PrefetchGlobalRead=2
    assert _expands(tmp_path, **forks) == (False, False)
