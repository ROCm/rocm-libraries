# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU tests (gfx1250) of the generalized unfused Winograd pipeline: small
shapes with dilation, padding, W tails,
segmented long filters, 2-D 5x5 and 7x1 filters. Each case runs the
benchmark driver with ``--verify`` (poisoned outputs, rocke metric < 5e-2)
and one short timing repeat."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _gfx1250():
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) == "gfx1250"
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _gfx1250(), reason="needs a gfx1250 GPU")

_T = "-F 1 -u 1 -v 1 -l 1 -g 1 -m conv -I NHWC -O NHWC -f NHWC"
_COMMON = [
    "--verify",
    "--repeats",
    "1",
    "--iters",
    "1",
    "--warmup",
    "1",
    "--cooldown",
    "0",
]


def _cmd(n, c, k, h, w, y, x, p, q, dh, dw):
    return (
        f"convfp16 -n {n} -c {c} -k {k} -H {h} -W {w} -y {y} -x {x} -p {p} -q {q} "
        f"-j {dw} {_T}".replace("-l 1", f"-l {dh}")
    )


GEN_CASES = [
    # (cmd, variants)
    (_cmd(2, 32, 32, 1, 203, 1, 11, 0, 15, 1, 3), "4s3,6s2,2"),
    (_cmd(1, 64, 32, 1, 301, 1, 7, 0, 15, 1, 5), "4,4s2,2"),
    (_cmd(2, 32, 48, 1, 97, 1, 3, 0, 1, 1, 1), "4,6,8"),
    (_cmd(2, 48, 64, 13, 11, 5, 5, 2, 2, 1, 1), "2x2,3x3"),
    (_cmd(2, 32, 32, 17, 17, 7, 1, 3, 0, 1, 1), "4,2"),
]


@pytest.mark.parametrize("case", range(len(GEN_CASES)))
def test_gen_pipeline(case):
    from benchmarks.common import benchmark_winograd_gen as B

    cmd, variants = GEN_CASES[case]
    rc = B.main(
        [
            "--miopen-cmd",
            cmd,
            "--variant",
            variants,
            "--k3-sweep",
            "tiny",
            "--jobs",
            "1",
        ]
        + _COMMON
    )
    assert rc == 0
