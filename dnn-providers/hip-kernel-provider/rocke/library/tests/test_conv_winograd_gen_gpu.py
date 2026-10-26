# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU tests (gfx1250) of the generalized unfused Winograd pipeline and the
fused 1-D Winograd kernel: small shapes with dilation, padding, W tails,
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


FUSED_CASES = [
    (
        _cmd(2, 32, 32, 1, 203, 1, 11, 0, 15, 1, 3),
        "4s3",
        "dict(q=2, fk=2, grid_blocks=4)",
    ),
    (
        _cmd(2, 64, 64, 1, 517, 1, 11, 0, 25, 1, 5),
        "4s3,6s2",
        "dict(q=2, fk=2, grid_blocks=6, xf='wmma', prefetch=True, m_f16=True,"
        " xf_batch=8, merge=True)",
    ),
    (
        _cmd(3, 128, 64, 1, 333, 1, 7, 0, 3, 1, 1),
        "4s2,4,6",
        "dict(q=2, fk=2, grid_blocks=5, prefetch=True, m_f16=True, merge=True)",
    ),
    (
        _cmd(2, 32, 96, 1, 250, 1, 3, 0, 1, 1, 1),
        "4,6",
        "dict(q=1, fk=1, grid_blocks=6, xf='wmma', prefetch=True, m_f16=True,"
        " merge=True, waves=4)",
    ),
    (
        _cmd(2, 128, 64, 1, 517, 1, 11, 0, 25, 1, 5),
        "4s3",
        "dict(q=2, fk=2, grid_blocks=10, xf='wmma', prefetch=True, m_f16=True,"
        " xf_batch=4, merge=True, waves=4, waves_per_eu=2)",
    ),
]


@pytest.mark.parametrize("case", range(len(FUSED_CASES)))
def test_fused_1d(case):
    from benchmarks.common import benchmark_winograd_fused_1d as B

    cmd, variants, cfg = FUSED_CASES[case]
    rc = B.main(
        [
            "--miopen-cmd",
            cmd,
            "--variant",
            variants,
            "--cfgs",
            f"[{cfg}]",
            "--jobs",
            "1",
        ]
        + _COMMON
    )
    assert rc == 0
