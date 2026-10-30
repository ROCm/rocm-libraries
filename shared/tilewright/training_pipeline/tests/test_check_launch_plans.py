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


def mirror(m, n, k, b, cu, schedule, version=None):
    prob = dict(m=m, n=n, k=k, batch=b, mi_dtype="bfloat16")
    ctx = fs.ExecutionContext(cu_budget=cu, schedule=schedule)
    p = hl.launch_plan(prob, KERNEL, HYBRID, ctx, 256, "gfx950", version=version)
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
    assert result.known_rows == 0


def _other_grid(m, n, k, b, cu, schedule="default"):
    """A static plan of another grid than rule version 1's, with partial
    tiles."""
    good = mirror(m, n, k, b, cu, schedule, version=1)
    assert not good[0]
    return (False, good[1] - 1, "tree")


def test_one_k_iteration_static_stream_k_differences_are_a_known_class():
    """Under rule version 1, TensileLite takes the grid Origami's
    select_grid_size chooses for a static Stream-K launch of batch 1 with one
    K iteration per tile and more tiles than the CUs it may use: such a
    difference is counted apart and does not fail the check; outside that
    class every difference fails, and under rule version 2 every one."""
    known = [
        # 1600 tiles on 256 CUs, K one depth.
        line(0, 41, 5120, 5120, 64, 1, 0, 0, _other_grid(5120, 5120, 64, 1, 0)),
        # K below the depth; 256 tiles on a budget of 64 CUs.
        line(0, 41, 2048, 2048, 17, 1, 64, 0, _other_grid(2048, 2048, 17, 1, 64)),
    ]
    rows = clp.parse_plans(known)
    result = clp.check_plans(rows, [KERNEL], 256, "gfx950", 1)
    assert result.ok and result.mismatched_rows == 0
    assert result.known_rows == 2
    assert result.known_field_mismatches == {
        "dynamic_queue": 0,
        "grid_wgs": 2,
        "partial_reduction": 2,
    }
    for version in (2, None):
        result = clp.check_plans(rows, [KERNEL], 256, "gfx950", version)
        assert not result.ok and result.known_rows == 0
        assert result.mismatched_rows == 2
    assert clp.check_plans(rows, [KERNEL], 256, "gfx1250").ok
    outside = [
        # Two K iterations per tile.
        line(0, 41, 5120, 5120, 128, 1, 0, 0, _other_grid(5120, 5120, 128, 1, 0)),
        # A batch of 2.
        line(0, 41, 5120, 5120, 64, 2, 0, 0, _other_grid(5120, 5120, 64, 2, 0)),
        # 256 tiles on the device's 256 CUs.
        line(0, 41, 2048, 2048, 64, 1, 0, 0, _other_grid(2048, 2048, 64, 1, 0)),
        # A dynamic launch on one side.
        line(0, 41, 5120, 5120, 64, 1, 0, 0, (True, 768, "none")),
    ]
    result = clp.check_plans(
        clp.parse_plans(known + outside), [KERNEL], 256, "gfx950", 1
    )
    assert not result.ok
    assert result.known_rows == 2 and result.mismatched_rows == len(outside)
    data_parallel = dict(KERNEL, attributes=dict(HYBRID, tile_processing_strategy=1))
    data_parallel["attributes"]["work_assignment"] = 0
    rows = clp.parse_plans(
        [line(0, 41, 5120, 5120, 64, 1, 0, 0, (False, 1599, "none"))]
    )
    result = clp.check_plans(rows, [data_parallel], 256, "gfx950", 1)
    assert not result.ok and result.known_rows == 0


def test_tensilelite_plans_of_one_k_iteration_match_rule_version_2():
    """What TensileLite plans for 1600 tiles of one K iteration on 256 CUs:
    Origami's streaming grid, with no partial reduction for a model of rule
    version 2 and the reduction rule version 1 reported for one of version
    1. Version 2's plan equals the mirror's; version 1's is of the known
    class."""
    static = dict(HYBRID, work_assignment=0)
    kernel = dict(KERNEL, occupancy=1, attributes=static)
    v2 = [line(0, 41, 5120, 5120, 64, 1, 0, 0, (False, 246, "none"), (128, 128, 64, 1))]
    v1 = [line(0, 41, 5120, 5120, 64, 1, 0, 0, (False, 246, "tree"), (128, 128, 64, 1))]
    result = clp.check_plans(clp.parse_plans(v2), [kernel], 256, "gfx950", 2)
    assert result.ok and result.mismatched_rows == 0 and result.known_rows == 0
    result = clp.check_plans(clp.parse_plans(v1), [kernel], 256, "gfx950", 1)
    assert result.ok and result.mismatched_rows == 0 and result.known_rows == 1
    assert not clp.check_plans(clp.parse_plans(v1), [kernel], 256, "gfx950", 2).ok


def test_the_launch_line_names_the_launch_model_and_the_cus():
    lines = [
        "# launch hipblaslt:gfx1250:1 cu_count 300",
        "# pos solution",
        "key 0 41 7",
    ]
    assert clp.parse_launch(lines) == ("hipblaslt:gfx1250:1", 300)
    assert clp.parse_launch(lines[1:]) is None
    for bad in (
        "# launch hipblaslt:gfx950:2",
        "# launch x cus 300",
        "# launch x cu_count y",
    ):
        with pytest.raises(ValueError):
            clp.parse_launch([bad])


def test_compare_needs_the_dump_of_the_launch_model_and_cus_it_checks(
    tmp_path, monkeypatch, capsys
):
    """compare checks the plans under the rule version given (default the
    arch's), of a dump whose launch line names that launch model and CU
    count: any other dump is bad input."""
    static = dict(HYBRID, work_assignment=0)
    kernel = dict(KERNEL, occupancy=1, attributes=static)
    monkeypatch.setattr(clp, "library_pool", lambda args: clp.Pool(["Lib"], [kernel]))
    key = hl.launch_key(kernel, static)
    body = [
        f"key 0 41 {key}",
        line(0, 41, 5120, 5120, 64, 1, 0, 0, (False, 246, "none"), (128, 128, 64, 1)),
    ]
    common = ["--library-dir", str(tmp_path), "--library-stem", "Lib"]

    def compare(launch, *extra):
        dump = tmp_path / "plans"
        dump.write_text("\n".join(([launch] if launch else []) + body) + "\n")
        argv = ["compare", str(dump), *common, *extra]
        rc = clp.main(argv)
        return rc, capsys.readouterr()

    gfx950 = ["--processor", "gfx950", "--cu-count", "256"]
    rc, out = compare("# launch hipblaslt:gfx950:2 cu_count 256", *gfx950)
    assert rc == 0 and "launch model: hipblaslt:gfx950:2 on 256 CUs" in out.out
    rc, out = compare("# launch hipblaslt:gfx950:1 cu_count 256", *gfx950)
    assert rc == 2 and "hipblaslt:gfx950:1" in out.err
    rc, out = compare(
        "# launch hipblaslt:gfx950:1 cu_count 256", *gfx950, "--rule-version", "1"
    )
    assert rc == 0 and f"({clp.KNOWN_CLASS}): 1\n" in out.out
    for launch in (None, "# launch hipblaslt:gfx950:2 cu_count 192"):
        assert compare(launch, *gfx950)[0] == 2
    rc, _ = compare(
        "# launch hipblaslt:gfx950:2 cu_count 256", *gfx950, "--rule-version", "3"
    )
    assert rc == 2


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
