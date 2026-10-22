# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Input-bias LDS stores must complete before publication to another wave."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config
from Tensile import __file__ as tensile_file

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source", ["A", "B"])
def test_bias_lds_stores_complete_before_publication(tmp_path, source):
    root = Path(tensile_file).parent
    config = yaml.safe_load((root / "Tests/common/gradient/gfx1250/hhs_bgrada_gfx1250.yaml").read_text())
    config["BenchmarkProblems"] = config["BenchmarkProblems"][:1]
    problem, group = config["BenchmarkProblems"][0]
    problem["BiasSrc"] = source
    values = {key: value[:1] for entry in group["ForkParameters"] for key, value in entry.items()}
    values.update(MatrixInstruction=[[16, 16, 32, 1, 1, 2, 2, 2, 2]],
                  TileProcessingStrategy=["DataParallel"], WorkAssignment=["StaticGrid"],
                  GlobalSplitU=[0], ClusterDim=[[1, 1]], PrefetchAcrossPersistent=[0],
                  ForceDisableShadowInit=[True], ScheduleIterAlg=[0], DepthU=[64])
    group["ForkParameters"] = [{key: value} for key, value in values.items()]
    group["BenchmarkFinalParameters"] = [
        {"ProblemSizes": [{"Exact": [128, 128, 1, 1024]}]}, {"BiasTypeArgs": ["s"]}]
    path = tmp_path / "bias_publication.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(str(path), limit=1, arch="gfx1250", canonical=False)
    assert len(results) == 1
    for name, assembly, error in results:
        assert error == 0
        assert_assembles(assembly, name)
        write = assembly.index("global store bias")
        partial = assembly.rindex("local store bias", 0, write)
        publication = assembly.index("s_barrier_signal -1", partial)
        assert re.search(r"s_wait_dscnt 0[^\n]*finish bias LDS stores before publication",
                         assembly[partial:publication])
