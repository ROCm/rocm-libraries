################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from types import SimpleNamespace

import pytest
from rocisa.code import Module, TextBlock

from Tensile.Components.SIA import noSchedGlobalRead

pytestmark = pytest.mark.unit


def _marker_module(name, marker):
    module = Module(name)
    module.add(TextBlock("%s\n" % marker))
    return module


def _make_writer(loop_iters=8, num_iters_plr=1):
    codes = SimpleNamespace(
        perIterGlobalRead=[Module("perIterGlobalRead%u" % i) for i in range(loop_iters)],
        unrollLoopHeader=Module("unrollLoopHeader"),
        dtlsM0UpdateA=_marker_module("dtlsM0UpdateA", "dtls_m0_update_a"),
        dtlsM0UpdateB=_marker_module("dtlsM0UpdateB", "dtls_m0_update_b"),
        dtlsM0UpdateMXSA=Module("dtlsM0UpdateMXSA"),
        dtlsM0UpdateMXSB=Module("dtlsM0UpdateMXSB"),
        globalReadA=_marker_module("globalReadA", "buffer_load_a_to_vgprG2LA"),
        globalReadB=_marker_module("globalReadB", "buffer_load_b_to_vgprG2LB"),
        globalReadMXSA=Module("globalReadMXSA"),
        globalReadMXSB=Module("globalReadMXSB"),
        globalReadMetadata=Module("globalReadMetadata"),
        gl2PrefetchIncrement=Module("gl2PrefetchIncrement"),
        gl2Prefetch=Module("gl2Prefetch"),
    )
    return SimpleNamespace(
        states=SimpleNamespace(numItersPLR=num_iters_plr),
        codes=codes,
    )


def _make_kernel(schedule_iter_alg):
    return {
        "_ScheduleIterAlg": schedule_iter_alg,
        "PrefetchGlobalRead": 2,
        "LoopIters": 8,
        "NoLdsWriteCode": False,
        "enableTDMA": False,
        "enableTDMB": False,
        "ProblemType": {
            "Sparse": False,
        },
    }


def _emit_no_sched_global_read(schedule_iter_alg):
    writer = _make_writer()
    kernel = _make_kernel(schedule_iter_alg)
    globalReadIncA = _marker_module("globalReadIncA", "global_read_inc_a")
    globalReadIncB = _marker_module("globalReadIncB", "global_read_inc_b")

    noSchedGlobalRead(writer, kernel, globalReadIncA, globalReadIncB)
    return writer


def test_sia0_pgr2_global_read_waits_for_local_write_g2l_use():
    writer = _emit_no_sched_global_read(schedule_iter_alg=0)

    assert writer.codes.perIterGlobalRead[0].itemsSize() == 0

    localWriteEndIter = 6
    scheduled = str(writer.codes.perIterGlobalRead[localWriteEndIter])
    assert "global_read_inc_a" in scheduled
    assert "global_read_inc_b" in scheduled
    assert "buffer_load_a_to_vgprG2LA" in scheduled
    assert "buffer_load_b_to_vgprG2LB" in scheduled


def test_non_sia0_pgr2_no_schedule_keeps_original_global_read_iter():
    writer = _emit_no_sched_global_read(schedule_iter_alg=1)

    scheduled = str(writer.codes.perIterGlobalRead[0])
    assert "global_read_inc_a" in scheduled
    assert "global_read_inc_b" in scheduled
    assert "buffer_load_a_to_vgprG2LA" in scheduled
    assert "buffer_load_b_to_vgprG2LB" in scheduled
    assert writer.codes.perIterGlobalRead[6].itemsSize() == 0
