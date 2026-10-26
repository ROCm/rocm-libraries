# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""PGR2 + ScheduleLocalWrite=0 must still prefetch the next tile.

With PrefetchGlobalRead>=2, prepareGRInstToSched defers the next tile's global
reads until the local writes have consumed their G2L vgprs. Only schedLocalWrite
used to issue them, so with ScheduleLocalWrite=0 (the only 1LDSBuffer option for
SIA1) the unrolled loop had no global reads and the result was wrong for more
than PGR K-tiles.
"""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_CONFIG = (Path(__file__).parent / ".." / "common" / "gemm" / "gfx11"
           / "fp16_tn_1lds_noslw_gfx11.yaml").resolve()


@pytest.mark.parametrize("sia", [1, 2])
def test_noschedlw_pgr2_main_loop_has_global_reads(tmp_path, sia):
    config = yaml.safe_load(_CONFIG.read_text())
    params = config["BenchmarkProblems"][0][1]["ForkParameters"]
    for parameter in params:
        if "ScheduleIterAlg" in parameter:
            parameter["ScheduleIterAlg"] = [sia]

    path = tmp_path / ("sia%u.yaml" % sia)
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx1100")

    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0

    loop = re.search(r"^label_LoopBeginL:(.*?)^label_LoopEndL:", assembly, re.S | re.M)
    assert loop, "no unrolled main loop"
    body = loop.group(1)
    numLocalWrites = body.count("ds_store")
    assert numLocalWrites > 0
    assert body.count("buffer_load") == numLocalWrites, \
        "unrolled loop does not prefetch the next tile"
    assert body.index("buffer_load") > body.index("ds_store"), \
        "deferred global reads must follow the local writes that free their vgprs"
