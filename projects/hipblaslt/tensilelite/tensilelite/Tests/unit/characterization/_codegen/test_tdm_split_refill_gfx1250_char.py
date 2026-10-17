################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 split TDM refill on emitted loop bodies.

tdm_split_refill.yaml derives three kernels the split applies to: TDMFuse=1 with MX on both tensors,
HalfPLR 1 (B leads) and HalfPLR 2 (A leads), and TDMFuse=2 without MX, HalfPLR 0 (B leads); ClusterDim
[2, 4] multicasts every set. In the sub-iteration that reads a buffer last, the lead set reads and refills
behind its own fence, then the trail set; the next sub-iteration reads the other buffer in the same order,
each set behind its own RAW barrier.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = os.path.join(os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250",
                       "tdm_split_refill.yaml")
# Descriptor sets by owner; each set's reads and refills carry its owner's LDS tokens, one per buffer.
_SETS = {1: {"A": ("A", "MXSA"), "B": ("MXSB", "B")}, 2: {"A": ("A",), "B": ("B",)}}
_SIDE_TOKENS = {"A": (0, 1), "B": (2, 3)}
_LEAD = {(1, 1): "B", (1, 2): "A", (2, 0): "B"}
_FILL = re.compile(r"^\s*tensor_load_to_lds\s+s\[sgprtdm(A|B)Group0\b")
_READ = re.compile(r"^\s*ds_load\S*\s.*\bvgprLocalReadAddr(A|B|MXSA|MXSB)\b")
_DS_OP = re.compile(r"^ds_\w+\s")
_DSCNT = re.compile(r"^s_wait_dscnt\s+(\d+)$")
_TOKEN = re.compile(r"\bsync LDS(\d+)\b")
_NAME = re.compile(r"^\.amdhsa_kernel\s+(\S+)", re.M)
_TAG = re.compile(r"_HPLR(\d)_.*_TDMF(\d)_")


def _tokens(line):
    return {int(t) for t in _TOKEN.findall(line.partition("//")[2])}


def _code(line):
    return line.partition("//")[0].strip()


def _fills(body):
    return [(i, m.group(1), _tokens(body[i])) for i, line in enumerate(body) for m in [_FILL.match(line)] if m]


def _reads(body, start, end, members, tokens):
    return [j for j in range(start, end)
            if (r := _READ.match(body[j])) and r.group(1) in members and _tokens(body[j]) & tokens]


def _otherBuffer(side, tokens):
    return {_SIDE_TOKENS[side][1 - _SIDE_TOKENS[side].index(t)] for t in tokens}


def _fenceIssue(body, fill, members, tokens):
    """Why the refill at `fill` is not fenced from its set's last read of the same LDS tokens, or None."""
    last = max(_reads(body, 0, fill, members, tokens), default=None)
    if last is None:
        return "no read of %s on LDS%s ahead of the refill" % ("/".join(members), sorted(tokens))
    dsAfter, retired, signalled, waited, clusterWait = 0, False, False, False, False
    for line in body[last + 1:fill]:
        code = _code(line)
        dscnt = _DSCNT.match(code)
        if dscnt:
            retired = retired or int(dscnt.group(1)) <= dsAfter
        elif _DS_OP.match(code):
            dsAfter += 1
        elif code == "s_barrier_wait -3":
            clusterWait = clusterWait or not signalled
        elif code == "s_barrier_signal -1":
            signalled = signalled or retired
        elif code == "s_barrier_wait -1":
            waited = waited or signalled
    if not waited:
        return "no s_wait_dscnt and workgroup barrier after the read at %d" % last
    if not clusterWait:
        return "no cluster wait ahead of the workgroup barrier"
    return None


@pytest.fixture(scope="module")
def loops():
    results = [r for index in (0, 1)
               for r in emit_kernels_from_config(_CONFIG, limit=4, arch="gfx1250", problem_index=index)]
    assert all(err == 0 for (_b, _s, err) in results), [(b, e) for (b, _s, e) in results if e]
    out = {}
    for base, src, _err in results:
        assert_assembles(src, base)
        halfPLR, fuse = map(int, _TAG.search(_NAME.search(src).group(1)).groups())
        lines = src.splitlines()
        out[(fuse, halfPLR)] = lines[lines.index("label_LoopBeginL:"):lines.index("label_LoopEndL:")]
    assert sorted(out) == sorted(_LEAD)
    return out


@pytest.mark.parametrize("kernel", sorted(_LEAD))
def test_the_sets_refill_in_turn_starting_with_the_lead(loops, kernel):
    order = [tc for _i, tc, _tokens in _fills(loops[kernel])]
    assert order[:1] == [_LEAD[kernel]], order
    assert all(a != b for a, b in zip(order, order[1:])), order


@pytest.mark.parametrize("kernel", sorted(_LEAD))
def test_each_refill_waits_for_its_own_sets_reads(loops, kernel):
    body, sets = loops[kernel], _SETS[kernel[0]]
    for fill, tc, tokens in _fills(body):
        assert _fenceIssue(body, fill, sets[tc], tokens) is None, (fill, tc)


@pytest.mark.parametrize("kernel", sorted(_LEAD))
def test_the_trail_set_reads_the_buffer_after_the_lead_refills_it(loops, kernel):
    body, lead, sets = loops[kernel], _LEAD[kernel], _SETS[kernel[0]]
    trail = "A" if lead == "B" else "B"
    fills = _fills(body)
    for n, (fill, tc, tokens) in enumerate(fills):
        if tc != lead:
            continue
        trailTokens = {_SIDE_TOKENS[trail][_SIDE_TOKENS[lead].index(t)] for t in tokens}
        trailFill = next((i for i, t, _x in fills[n + 1:] if t == trail), len(body))
        assert _reads(body, fill + 1, trailFill, sets[trail], trailTokens), \
            "no %s read of the buffer between the %s refill at %d and the %s refill" % (trail, lead, fill, trail)


@pytest.mark.parametrize("kernel", sorted(_LEAD))
def test_the_next_buffer_is_read_lead_first_behind_one_raw_barrier_per_set(loops, kernel):
    body, lead, sets = loops[kernel], _LEAD[kernel], _SETS[kernel[0]]
    trail = "A" if lead == "B" else "B"
    for fill, tc, tokens in _fills(body):
        if tc != trail:
            continue
        leadNext = _otherBuffer(lead, {_SIDE_TOKENS[lead][_SIDE_TOKENS[trail].index(t)] for t in tokens})
        trailNext = _otherBuffer(trail, tokens)
        leadRead = min(_reads(body, fill + 1, len(body), sets[lead], leadNext), default=None)
        trailRead = min(_reads(body, fill + 1, len(body), sets[trail], trailNext), default=None)
        if leadRead is None and trailRead is None:
            continue
        assert leadRead is not None and trailRead is not None and leadRead < trailRead, (fill, leadRead, trailRead)
        rawBarriers = [j for j in range(fill + 1, trailRead) if _code(body[j]) == "s_barrier_wait -1"]
        assert any(_tokens(body[j]) & leadNext for j in rawBarriers if j < leadRead), (fill, "lead RAW")
        assert any(_tokens(body[j]) & trailNext for j in rawBarriers if j > leadRead), (fill, "trail RAW")
