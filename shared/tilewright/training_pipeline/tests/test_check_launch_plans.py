# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

import check_launch_plans as clp
from lib import features as fs
from lib import hipblaslt_launch as hl
from lib.hardware import DATATYPE_VALUE

HYBRID = dict(
    tile_processing_strategy=2, work_assignment=2,
    work_group_x=256, work_group_y=1, work_group_z=1,
)  # fmt: skip
KERNEL = dict(
    sol_idx_global=41, mt_m=128, mt_n=128, mt_k=64, occupancy=2, attributes=HYBRID
)
BF16 = DATATYPE_VALUE["bfloat16"]


def line(pos, sol, m, n, k, b, cu, sched, plan, kernel=(128, 128, 64, 2)):
    dyn, grid, red = plan
    fields = [pos, sol, m, n, k, b, BF16, cu, sched, int(dyn), grid, red, *kernel]
    return " ".join(str(f) for f in fields)


def mirror(m, n, k, b, cu, schedule):
    prob = dict(m=m, n=n, k=k, batch=b, mi_dtype="bfloat16")
    ctx = fs.ExecutionContext(cu_budget=cu, schedule=schedule)
    p = hl.launch_plan(prob, KERNEL, HYBRID, ctx, 256, "gfx950")
    return (p.dynamic_queue, p.grid_wgs, p.partial_reduction)


def test_matching_plans_pass():
    lines = [
        "# header",
        line(
            0,
            41,
            4096,
            4096,
            512,
            1,
            224,
            0,
            mirror(4096, 4096, 512, 1, 224, "default"),
        ),
        line(0, 41, 256, 256, 4096, 1, 0, 1, mirror(256, 256, 4096, 1, 0, "dynamic")),
        line(0, 41, 300, 77, 9, 4, 96, 2, mirror(300, 77, 9, 4, 96, "auto")),
    ]
    rows = clp.parse_plans(lines)
    assert [r.context.schedule for r in rows] == ["default", "dynamic", "auto"]
    result = clp.check_plans(rows, [KERNEL], 256, "gfx950")
    assert result.ok and result.rows == 3 and result.kernels == 1


def test_differing_fields_and_kernels_are_reported():
    good = mirror(4096, 4096, 512, 1, 224, "default")
    rows = clp.parse_plans(
        [
            line(
                0, 41, 4096, 4096, 512, 1, 224, 0, (not good[0], good[1] + 1, good[2])
            ),
            line(0, 42, 4096, 4096, 512, 1, 224, 0, good),
            line(0, 41, 4096, 4096, 512, 1, 224, 0, good, kernel=(64, 128, 64, 2)),
            line(1, 41, 4096, 4096, 512, 1, 224, 0, good),
        ]
    )
    result = clp.check_plans(rows, [KERNEL], 256, "gfx950")
    assert not result.ok
    assert result.mismatched_rows == 1
    assert result.field_mismatches == {
        "dynamic_queue": 1,
        "grid_wgs": 1,
        "partial_reduction": 0,
    }
    assert len(result.kernel_mismatches) == 3


def test_matching_keys_pass_and_plans_skip_key_lines():
    key = hl.launch_key(KERNEL, HYBRID)
    lines = [f"key 0 41 {key}", line(0, 41, 300, 77, 9, 4, 96, 2, (False, 1, "none"))]
    assert len(clp.parse_plans(lines)) == 1
    result = clp.check_keys(clp.parse_keys(lines), [KERNEL])
    assert result.ok and result.kernels == 1


def test_differing_and_missing_keys_are_reported():
    key = hl.launch_key(KERNEL, HYBRID)
    pool = [KERNEL, dict(KERNEL, sol_idx_global=7), dict(KERNEL, sol_idx_global=8)]
    keys = clp.parse_keys(
        [f"key 0 41 {key ^ 1}", f"key 1 9 {key}", f"key 1 7 {key}", f"key 5 41 {key}"]
    )
    result = clp.check_keys(keys, pool)
    assert not result.ok
    assert len(result.mismatches) == 5
    assert result.mismatches[-1] == "pool kernels without a key: 1"


def test_bad_key_lines_are_rejected():
    with pytest.raises(ValueError):
        clp.parse_keys(["key 0 41"])


@pytest.mark.parametrize(
    "bad",
    [
        "0 41 1 1 1 1",
        line(0, 41, 1, 1, 1, 1, 0, 0, (False, 1, "none")).replace(
            f" {BF16} ", " 9999 ", 1
        ),
    ],
)
def test_bad_lines_are_rejected(bad):
    with pytest.raises(ValueError):
        clp.parse_plans([bad])


def test_pool_file_names_the_logic_files_and_the_pool(tmp_path):
    (tmp_path / "Lib_a.dat.zlib").write_bytes(b"")
    (tmp_path / "Lib_b.dat").write_bytes(b"")
    pool = clp.Pool(["Lib_a", "Lib_b"], [dict(KERNEL), dict(KERNEL, sol_idx_global=7)])
    text = clp.pool_file(pool, tmp_path)
    assert text.splitlines() == [
        f"library {tmp_path.resolve() / 'Lib_a'}",
        f"library {tmp_path.resolve() / 'Lib_b'}",
        "pool 41 7",
    ]
