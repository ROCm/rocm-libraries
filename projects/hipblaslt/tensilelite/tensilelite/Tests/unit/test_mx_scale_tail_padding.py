# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from test_segment_interleave_policy import (
    _execute_scale_addresses,
    _gp_gfx1250,
    derive,
)


@pytest.mark.parametrize("source_swap", [False, True])
@pytest.mark.parametrize("interleave", [0, 1])
@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_mx_scale_tail_crosses_padding_blocks(
        derive, assembler, monkeypatch, source_swap, interleave, tc):
    from codegen_harness import emit_one_kernel
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    from tensilelite.create_library.run import generateKernelObjectsFromSolutions

    # One K step covers half a padding block, so the second increment must
    # include padding even when the generated tail body has only one iteration.
    stride = 128
    block = 2 * stride
    sol, out = derive(
        interleave, mi=[16, 16, 128, 1, 1, 2 if interleave else 1, 1, 2, 2],
        VectorWidthA=2 if interleave else 1, VectorWidthB=1, DepthU=512, HalfPLR=0,
        SourceSwap=source_swap, WaveTileContiguousA=False,
        LDSSegmentBRotate=0, NumElementsPerBatchStore=16,
        ProblemType={"DataType": "F8", "DestDataType": "S",
                     "MXBlockA": 32, "MXBlockB": 32},
        **{"LdsPad" + tc: 16, "LdsBlockSizePerPad" + tc: block})
    assert sol["Valid"], out
    increments = []
    original = KernelWriterAssembly.localReadInc

    def capture(self, kernel, iui, tp):
        module = original(self, kernel, iui, tp)
        if self.states.inTailLoop and tp["tensorChar"] == tc:
            increments.append(str(module))
        return module

    monkeypatch.setattr(KernelWriterAssembly, "localReadInc", capture)
    kernel, = generateKernelObjectsFromSolutions([sol])
    writer = KernelWriterAssembly(assembler, DebugConfig())
    name, _, error = emit_one_kernel(writer, kernel, canonical=False)
    assert error == 0, name
    assert len(increments) == 1
    num_addresses = getattr(writer.states, tc.lower()).numVgprLocalReadAddr
    offsets = [0] * num_addresses
    for step in range(3):
        initial = f"s_mov_b32 s[sgprOrigLoopCounter], {step * 128}\n"
        for idx, offset in enumerate(offsets):
            initial += f"v_mov_b32 v[vgprLocalReadAddr{tc}+{idx}], {offset}\n"
        registers, normalized = _execute_scale_addresses(initial + increments[0], 0)
        offsets = [registers[normalized(f"v[vgprLocalReadAddr{tc}+{idx}]")]
                   for idx in range(num_addresses)]
        # Independent writer layout: insert 16 bytes after each full block.
        raw = (step + 1) * stride
        assert offsets == [raw + raw // block * 16] * num_addresses
