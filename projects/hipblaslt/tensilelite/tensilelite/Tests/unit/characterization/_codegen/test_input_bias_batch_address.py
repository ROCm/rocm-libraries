# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched bias stores use a 64-bit base, without changing the vector bound."""

from pathlib import Path
import re
import pytest
import yaml
from tensilelite import __file__ as tensile_file
from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source", ["A", "B"])
@pytest.mark.parametrize("bias_type,bpe_shift", [("s", 2), ("h", 1)])
def test_direct_bias_batch_address_assembles(tmp_path, source, bias_type, bpe_shift):
    root = Path(tensile_file).parent
    config = yaml.safe_load((root / "Tests/common/gradient/gfx1250/hhs_bgrada_gfx1250.yaml").read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    problem, group = config["BenchmarkProblems"][0]
    problem.update(BiasSrc=source, BiasDataTypeList=[bias_type], SetConstStrideBias=[[2, 64]])
    values = {key: value[:1] for entry in group["ForkParameters"] for key, value in entry.items()}
    values.update(TileProcessingStrategy=["DataParallel"], WorkAssignment=["StaticGrid"],
                  GlobalSplitU=[0], ClusterDim=[[1, 1]], DepthU=[64],
                  PrefetchAcrossPersistent=[0], StaggerU=[0], ScheduleIterAlg=[0])
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    group["BenchmarkFinalParameters"] = [
        {"ProblemSizes": [{"Exact": [33, 49, 2, 65]}]}, {"BiasTypeArgs": [bias_type]}]
    path = tmp_path / "bias_batch.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        (tmp_path / "kernel.s").write_text(assembly)
        begin = re.search(r"s_mul_hi_u32[^\n]*bias batch offset in elements", assembly).start()
        end = assembly.index("// global store bias", begin)
        address = assembly[begin:end]
        assert re.search(r"s_mul_hi_u32.*sgprWorkGroup2.*sgprBiasStride", address)
        assert re.search(r"s_mul_i32.*sgprWorkGroup2.*sgprBiasStride", address)
        assert re.search(rf"s_lshl_b64.*(?:0x{bpe_shift:x}|{bpe_shift})\s*// bias batch offset in bytes", address)
        assert re.search(r"s_add_u32.*sgprSrdBias.*bias batch base", address)
        assert re.search(r"s_addc_u32.*sgprSrdBias\+1.*bias batch base carry", address)
        assert address.index("bias batch base carry") < address.index("Local bias address")
        assert not re.search(r"s_(?:add|addc)_u32 s\[sgprAddressBias", address)
