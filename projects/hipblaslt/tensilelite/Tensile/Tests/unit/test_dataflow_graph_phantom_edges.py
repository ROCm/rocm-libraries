################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Regression fixture for the phantom-edge bug (wx9.4).

The dataflow resolver yields an edge from the consumer to *every* prior
writer of each register the consumer reads, instead of just the latest
writer in stream order. On real TF32 captures this produces ~24,688
false-positive cross-side OrderInverted failures because Pack instructions
reuse a small pool of scratch vgprs (e.g. v133) across PackA0 and PackB0
chains.

This file pins the bug-fix-plus-no-regression invariant. The resolver
rewrite (wx9.4.2 / Sub-B) makes both assertions pass; if a future
change re-introduces the over-yield, these tests fail loudly.

# Why a synthetic fixture instead of the real TF32 capture

The TF32 corpus (Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml,
custom_xfp32.yaml) is integration-level and slow; it also requires
_GenericALURule to be in the production registry, which is itself the
final step of the phantom-edge thread (wx9.4.4 / Sub-D). This fixture
provisionally enables a generic ALU rule via a context manager, so the
production registry stays clean while the bug is reproduced unit-side.
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from rocisa.container import RegisterContainer

from Tensile.Components import ScheduleCapture
from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    SLOT_KIND_MFMA,
    SlotKey,
    TaggedInstruction,
    build_dataflow_graph,
)

from dataflow_fixtures import _vrange, make_capture, make_mfma

# Reuse the same _wrap helper pattern as test_dataflow_graph_register_gaps.
from test_dataflow_graph_register_gaps import _wrap


# =============================================================================
# Synthetic Pack-style instruction + test-scope ALU rule
# =============================================================================
# The fixture needs an instruction that publishes (reads, writes) via the
# rule registry. Real Pack rocisa classes (VCvtPkF32toBF16 etc.) work but
# require the production registry to gain _GenericALURule — that's wx9.4.4.
# Use a fake instruction shape + injectable rule so this regression test
# is independent of that work.


@dataclass
class _FakePack:
    """Stand-in for a Pack-style ALU instruction. Reads scratch_in and
    src; writes dst. dst==scratch_out lets the same scratch vgpr appear
    on both the write side and on a later instruction's read side."""
    dst: RegisterContainer
    src: Optional[RegisterContainer] = None
    scratch_in: Optional[RegisterContainer] = None

    def __str__(self):
        parts = [f"dst={self.dst}"]
        if self.src is not None:
            parts.append(f"src={self.src}")
        if self.scratch_in is not None:
            parts.append(f"scratch_in={self.scratch_in}")
        return f"fake_pack({', '.join(parts)})"


class _FakePackRule:
    """Test-scope operand rule for _FakePack. Mirrors what _GenericALURule
    will do for production Pack classes (wx9.4.4): publish dst as the
    written resource and src/scratch_in as reads."""
    def applies(self, inst, category=None):
        return isinstance(inst, _FakePack)

    def extract(self, inst, category=None):
        writes = (inst.dst,) if isinstance(inst.dst, RegisterContainer) else ()
        reads = tuple(r for r in (inst.src, inst.scratch_in)
                      if isinstance(r, RegisterContainer))
        return reads, writes


@contextmanager
def using_pack_rule():
    """Inject _FakePackRule at the END of _OPERAND_RULES for the test scope.

    Order matters: more-specific rules (DSLoad, DSStore, etc.) run first;
    _NoDataflowRule absorbs SWait/SBarrier/SNop; _FakePackRule catches
    anything labeled as our _FakePack.
    """
    saved = ScheduleCapture._OPERAND_RULES
    ScheduleCapture._OPERAND_RULES = saved + (_FakePackRule(),)
    try:
        yield
    finally:
        ScheduleCapture._OPERAND_RULES = saved


def _tag_pack(inst, *, category: str, mfma_index: int, sequence: int) -> TaggedInstruction:
    return TaggedInstruction(
        inst=inst,
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=mfma_index, sequence=sequence),
    )


# =============================================================================
# Helpers for edge counting
# =============================================================================


def _edges_keyed_by_consumer_and_resource(graph):
    """Return {(consumer.identity, resource_key): [edges]} for the graph.

    resource_key is (regType, regIdx, regNum) for numeric registers, or
    a stable tuple for symbolic ones. Two writers feeding the same
    consumer's read of the same byte-range collide here — that's
    exactly the phantom-edge symptom.
    """
    by_key = defaultdict(list)
    for edge in graph.edges:
        res = edge.resource
        if isinstance(res, RegisterContainer):
            res_key = (res.regType, res.regIdx, res.regNum)
        else:
            res_key = ("memory", id(res))  # MemoryRegion, not the focus here
        by_key[(edge.consumer.identity, res_key)].append(edge)
    return by_key


# =============================================================================
# Phantom-edge regression — Pack scratch-vgpr reuse
# =============================================================================


class TestPhantomPackScratchReuse:
    """Reproduce the v133-scratch-reuse phantom-edge case."""

    @staticmethod
    def _build_capture():
        """Three writers + three readers of scratch v133 across PackA0/PackB0.

        Each writer reads a DIFFERENT src vgpr so their render-strings (and
        thus identities) are distinct — same render-string would collide in
        nodes_by_identity and silently mask the bug.

        Stream order (slot.mfma_index, slot.sequence):
          (0, 0)  PackA0[0]: write v133 from v8   (scratch start)
          (0, 1)  PackA0[1]: read v133, write v40 (scratch consumer #1)
          (0, 2)  PackA0[2]: write v133 from v9   (scratch reuse)
          (0, 3)  PackA0[3]: read v133, write v41 (scratch consumer #2)
          (0, 4)  PackB0[0]: write v133 from v10  (scratch reuse — cross-side)
          (0, 5)  PackB0[1]: read v133, write v50 (scratch consumer #3)
        """
        v133 = _vrange(133, 1)
        v40 = _vrange(40, 1)
        v41 = _vrange(41, 1)
        v50 = _vrange(50, 1)
        v8 = _vrange(8, 1)
        v9 = _vrange(9, 1)
        v10 = _vrange(10, 1)

        return make_capture(BODY_LABEL_ML, [
            _tag_pack(_FakePack(dst=v133, src=v8),
                      category="PackA0", mfma_index=0, sequence=0),
            _tag_pack(_FakePack(dst=v40, scratch_in=v133),
                      category="PackA0", mfma_index=0, sequence=1),
            _tag_pack(_FakePack(dst=v133, src=v9),
                      category="PackA0", mfma_index=0, sequence=2),
            _tag_pack(_FakePack(dst=v41, scratch_in=v133),
                      category="PackA0", mfma_index=0, sequence=3),
            _tag_pack(_FakePack(dst=v133, src=v10),
                      category="PackB0", mfma_index=0, sequence=4),
            _tag_pack(_FakePack(dst=v50, scratch_in=v133),
                      category="PackB0", mfma_index=0, sequence=5),
            # MFMA consumes v40, v41, v50 so they appear as graph identities.
            make_mfma(c_dst_start=60, a_src_start=40, b_src_start=50,
                      slot=1, a_src_count=2, b_src_count=1),
        ])

    def test_each_v133_reader_has_exactly_one_incoming_edge(self):
        """Tight assertion: with per-byte latest-writer (post-Sub-B), each
        v133 reader gets ONE incoming edge — to the most recent v133 writer
        in stream order. Today the resolver yields every prior writer;
        readers later in the chain accumulate 2-3 phantom edges.
        """
        with using_pack_rule():
            graph = build_dataflow_graph(_wrap(self._build_capture()))

        by_key = _edges_keyed_by_consumer_and_resource(graph)

        v133_collisions = {
            consumer_id: edges
            for (consumer_id, res_key), edges in by_key.items()
            if res_key[0] == "v" and res_key[1] == 133 and len(edges) > 1
        }

        assert not v133_collisions, (
            f"Phantom edges on v133: {len(v133_collisions)} consumer(s) "
            f"received multiple edges. Each consumer should receive exactly "
            f"one edge, to the latest writer in stream order. "
            f"Collisions: {[(cid, len(es)) for cid, es in v133_collisions.items()]}"
        )

    def test_total_v133_edge_count_is_linear_in_consumers(self):
        """Loose assertion: the total number of v133 edges scales with the
        number of v133 readers (one each), not with the cross-product of
        readers * prior writers. Three v133 readers => 3 v133 edges.
        Today: 1 + 2 + 3 = 6 edges from the cross product."""
        with using_pack_rule():
            graph = build_dataflow_graph(_wrap(self._build_capture()))

        v133_edges = [
            e for e in graph.edges
            if isinstance(e.resource, RegisterContainer)
            and e.resource.regType == "v" and e.resource.regIdx == 133
        ]
        assert len(v133_edges) == 3, (
            f"Expected 3 v133 edges (one per reader to its latest writer); "
            f"got {len(v133_edges)}. Phantom edges from prior-writer "
            f"over-yield are likely the cause."
        )
