# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import pytest
import yaml
from config_harness import derive_states, emit_kernels_from_config
from rocisa import getMFMAs
from rocisa.instruction import SNop
from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).parent / "data/test_data/_designed/gfx942/complex_b_read_wait.yaml"


@pytest.mark.parametrize(
    "conjugate_a,conjugate_b",
    [(False, False), (False, True), (True, False), (True, True)],
)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["C", "Z"])
@pytest.mark.parametrize("delay_product", [False, True])
def test_complex_b_reads_complete_before_use(
    arch, dtype, conjugate_a, conjugate_b, delay_product, monkeypatch, tmp_path
):
    if delay_product:
        original = KernelWriterAssembly.mfmaIter

        def delayed_product(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            mfmas = getMFMAs(result)
            if mfmas:
                mfmas[0].parent.add(SNop(waitState=0, comment="extra complex product instruction"))
            return result

        monkeypatch.setattr(KernelWriterAssembly, "mfmaIter", delayed_product)

    config = yaml.safe_load(_CONFIG.read_text())
    problem, params = config["BenchmarkProblems"][0]
    problem["ComplexConjugateA"] = conjugate_a
    problem["ComplexConjugateB"] = conjugate_b
    for key in ("DataType", "DestDataType", "ComputeDataType"):
        problem[key] = dtype
    wave_tile = [7, 4] if dtype == "C" else [7 if arch == "gfx950" else 3, 2]
    if dtype == "Z":
        # The larger gfx950 tile exposes the late double-complex wait; gfx942
        # needs a smaller tile to fit its LDS capacity.
        for param in params["ForkParameters"]:
            if "MatrixInstruction" in param:
                param["MatrixInstruction"][0][5:7] = wave_tile
            if "GlobalReadVectorWidthB" in param:
                param["GlobalReadVectorWidthB"] = [1]
            for key in ("LdsBlockSizePerPadA", "LdsBlockSizePerPadB"):
                if key in param:
                    param[key] = [256]
    path = tmp_path / "complex.yaml"
    path.write_text(yaml.safe_dump(config))

    states = derive_states(path, arch=arch, limit_solutions=1)
    assert len(states) == 1
    state = states[0]
    assert state["UnrollMajorLDSB"]
    assert state["MIWaveTile"] == wave_tile
    assert state["PrefetchLocalRead"] == 1
    assert state["ClusterLocalRead"] == 1

    results = emit_kernels_from_config(path, arch=arch, limit=1)
    assert len(results) == 1
    _name, source, error = results[0]
    assert error == 0

    open_loop = source.index("label_openLoopL:")
    prefetch = source[source.index("/* local read prefetch b */") : open_loop]
    b_reads = re.findall(r"^ds_read_b\d+ v\[([^:]+):", prefetch, re.M)
    assert len(b_reads) == (4 if dtype == "C" else 2)

    first_waits = list(re.finditer(r"^s_waitcnt[^\n]*for iteration == 0[^\n]*", source, re.M))
    assert len(first_waits) == (4 if dtype == "C" else 5)
    for index, wait in enumerate(first_waits):
        assert wait[0].startswith(f"s_waitcnt lgkmcnt({len(b_reads) - 1})")
        end = first_waits[index + 1].start() if index + 1 < len(first_waits) else len(source)
        body = source[wait.end() : end]
        first_later_b_use = re.search(r"^v_mfma_.*v\[" + re.escape(b_reads[1]) + r"\+", body, re.M)
        assert first_later_b_use is not None
        before_later_b = body[: first_later_b_use.start()]
        first_product = list(
            re.finditer(
                r"^v_mfma_.*v\[" + re.escape(b_reads[0]) + r"\+[^\n]*",
                before_later_b,
                re.M,
            )
        )
        assert first_product
        product_end = first_product[-1].end()
        if delay_product:
            extra = re.search(
                r"^s_nop[^\n]*extra complex product instruction", before_later_b, re.M
            )
            assert extra is not None
            product_end = extra.end()

        # The last prefetched B read must be complete before B[1] is consumed.
        # At a wait, its younger LDS operations determine the safe counter value.
        younger_ds = 0
        b_ready = False
        offset = 0
        for line in before_later_b.splitlines(keepends=True):
            if re.match(r"ds_(?:read|write)_", line):
                younger_ds += 1
            second_wait = re.match(r"s_waitcnt lgkmcnt\((\d+)\)", line)
            if second_wait and offset >= product_end and int(second_wait[1]) <= younger_ds:
                b_ready = True
            offset += len(line)
        assert b_ready
