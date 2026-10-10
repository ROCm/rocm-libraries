# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""ForceUnrollSubIter next-loop local-read WAR gate (KernelWriter._makeSubIterSchedule).

With ForceUnrollSubIter and a single VGPR buffer, the last sub-iteration of the
main loop refills sub-tile 0 of ValuA/B_X0 for the next loop while its MFMAs
still read sub-tile 1. The reads may only be interleaved with those MFMAs when
each read stays inside one sub-tile (lrvwTile <= MIWaveTile / numSubTiles);
otherwise they must be deferred until after the last MFMA.

MIWaveTile is [4, 4] (sub-tiles of 2 tile elements), so VectorWidth 4 makes a
read span both sub-tiles and VectorWidth 2 does not.
"""

import itertools
import os
import re

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "fusi_next_loop_read_war.yaml",
)

_VARIANTS = [(sia, vwa, vwb) for sia, vwa, vwb in itertools.product((1, 3), (2, 4), (2, 4))]

def _variant_id(variant):
    sia, vwa, vwb = variant
    return f"sia{sia}-vwa{vwa}-vwb{vwb}"


_READ_DST = re.compile(r"ds_read_b\d+ v\[vgprValu([AB])_X0_I0\+(\d+):vgprValu\1_X0_I0\+\2\+(\d+)\]")
_MFMA_SRC = re.compile(r"v\[vgprValu([AB])_X0_I0\+(\d+)\+0\+0\]")


@pytest.fixture(scope="module")
def kernels():
    results = emit_kernels_from_config(_CONFIG, limit=len(_VARIANTS), arch="gfx950",
                                       expected_fork_count=len(_VARIANTS))
    kernels = {}
    for _base, source, error in results:
        assert error == 0
        name = re.search(r"\.amdhsa_kernel (\S+)", source).group(1)
        sia = int(re.search(r"_SIA(\d)_", name).group(1))
        vwa, vwb = map(int, re.search(r"_VWA(\d+)_VWB(\d+)_", name).groups())
        kernels[(sia, vwa, vwb)] = source
    assert set(kernels) == set(_VARIANTS)
    return kernels


def _last_subiter(source):
    loop_begin = source.index("label_LoopBeginL:")
    loop_end = source.index("label_LoopEndL", loop_begin)
    body = source[loop_begin:loop_end]
    last = body[body.rindex("/* subiter "):]
    return [line.strip() for line in last.splitlines()
            if line.strip().startswith(("ds_read", "v_mfma"))]


@pytest.mark.parametrize("variant", _VARIANTS, ids=_variant_id)
def test_next_loop_reads_never_clobber_pending_mfma_operands(kernels, variant):
    insts = _last_subiter(kernels[variant])
    for i, inst in enumerate(insts):
        m = _READ_DST.match(inst)
        if not m:
            continue
        tc, start, extra = m.group(1), int(m.group(2)), int(m.group(3))
        written = set(range(start, start + extra + 1))
        for later in insts[i + 1:]:
            if later.startswith("v_mfma"):
                read = {int(idx) for t, idx in _MFMA_SRC.findall(later) if t == tc}
                assert not written & read, f"{inst!r} overwrites a register read by later {later!r}"


@pytest.mark.parametrize("variant", _VARIANTS, ids=_variant_id)
def test_next_loop_reads_deferred_only_when_a_read_spans_sub_tiles(kernels, variant):
    _sia, vwa, vwb = variant
    sub_tile = 4 // 2
    deferred = vwa > sub_tile or vwb > sub_tile
    insts = _last_subiter(kernels[variant])
    last_mfma = max(i for i, inst in enumerate(insts) if inst.startswith("v_mfma"))
    reads = [i for i, inst in enumerate(insts) if inst.startswith("ds_read")]
    assert reads
    assert all(i > last_mfma for i in reads) == deferred
