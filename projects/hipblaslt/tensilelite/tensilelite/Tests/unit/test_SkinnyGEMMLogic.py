# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the skinny GEMM logic file generator (KernelGenerators.SkinnyGEMMLogic)."""

import pytest
import yaml

pytestmark = pytest.mark.unit

from Tensile.KernelGenerators import SkinnyGEMM, SkinnyGEMMLogic

RING_PARAMS = {"bm": 1, "tn": 1, "lanesK": 16, "waves": 1, "prefetch": 1, "upIters": 0, "partialSteps": 1,
               "group": 1, "numXcds": 1, "grid3d": True}
RING_KNOBS = SkinnyGEMMLogic.to_knobs(dict(RING_PARAMS, useBeta=False))


@pytest.fixture(scope="module")
def pool():
    return SkinnyGEMMLogic.buildPool()


@pytest.fixture(scope="module")
def pickedShapes():
    return {shape: SkinnyGEMMLogic.pickedKnobs(*shape) for shape in SkinnyGEMMLogic.measuredShapes()}


def writeRows(path, rows):
    path.write_text(yaml.safe_dump(rows, default_flow_style=None), encoding="utf-8")
    return path


def validRow(**overrides):
    row = {"range": [1, 4096, 1, -1, 1, 1, 1, 8192], "params": dict(RING_PARAMS)}
    row.update(overrides)
    return row


class TestPool:
    def test_deterministic_and_sorted(self, pool):
        again = SkinnyGEMMLogic.buildPool()
        assert again == pool
        assert list(again) == sorted(again)

    def test_size_and_validity(self, pool):
        assert len(pool) <= SkinnyGEMMLogic.MAX_POOL
        for knobs in pool:
            SkinnyGEMM.validateParams(SkinnyGEMMLogic.asParams(knobs))
        assert all(reasons for reasons in pool.values())

    def test_both_variants_of_every_knob_set(self, pool):
        falseVariants = [knobs for knobs in pool if not SkinnyGEMMLogic.usesBeta(knobs)]
        assert len(falseVariants) * 2 == len(pool)
        for knobs in falseVariants:
            twin = SkinnyGEMMLogic.withBeta(knobs, True)
            assert twin in pool
            assert pool[twin] == pool[knobs]
            assert list(pool).index(twin) == list(pool).index(knobs) + 1

    def test_cap_is_twice_the_knob_set_cap(self):
        assert SkinnyGEMMLogic.MAX_POOL == 2 * SkinnyGEMMLogic.MAX_KNOB_SETS

    def test_contains_picks_and_baseline(self, pool, pickedShapes):
        for shape, knobs in pickedShapes.items():
            assert knobs in pool, shape
        for knobs in SkinnyGEMMLogic.BASELINE_PARAMS:
            assert knobs in pool

    def test_ring_twins(self, pool, pickedShapes):
        for knobs in list(pickedShapes.values()) + list(SkinnyGEMMLogic.BASELINE_PARAMS):
            assert SkinnyGEMMLogic._ringTwin(knobs) in pool

    def test_persistent_entries_come_from_pickConfig(self, pool, pickedShapes):
        allowed = set(pickedShapes.values()) | {SkinnyGEMMLogic._ringTwin(k) for k in pickedShapes.values()}
        persistent = [knobs for knobs in pool if not knobs[SkinnyGEMMLogic._GRID3D]]
        assert persistent
        assert all(SkinnyGEMMLogic.withBeta(knobs, False) in allowed for knobs in persistent)

    def test_up_front_windows_hold_a_measured_k(self, pool):
        ks = {k for _, _, k in SkinnyGEMMLogic.measuredShapes()}
        for knobs in pool:
            upIters = knobs[SkinnyGEMMLogic._UP_ITERS]
            if upIters == 0:
                continue
            step = SkinnyGEMMLogic.kernelStep(knobs)
            assert any((upIters - 1) * step < k <= upIters * step for k in ks), knobs

    def test_size(self, pool):
        assert len(pool) == 172


class TestKCoverage:
    def test_ring_partial_steps(self):
        assert SkinnyGEMMLogic.kCoverage((1, 1, 16, 1, 1, 0, 1, 1, 1, True, False)) == "(0, 2^20]"

    def test_ring_whole_steps(self):
        assert SkinnyGEMMLogic.kCoverage((1, 1, 16, 1, 1, 0, 0, 1, 1, True, True)) == "(0, 2^20], K % 1024 == 0"

    def test_up_front(self):
        assert SkinnyGEMMLogic.kCoverage((1, 1, 64, 4, 2, 2, 0, 1, 1, True, False)) == "(4096, 8192], K % 4096 == 0"


class TestHeader:
    def originamiHeader(self):
        lines = SkinnyGEMMLogic.ORIGAMI_LOGIC.read_text(encoding="utf-8").splitlines()
        return lines[: lines.index("DefaultSolution:")]

    def test_copies_the_leading_lines(self):
        assert SkinnyGEMMLogic.headerLines(SkinnyGEMMLogic.ORIGAMI_LOGIC) == self.originamiHeader()

    def test_missing_key_raises(self, tmp_path):
        path = tmp_path / "logic.yaml"
        path.write_text("MinimumRequiredVersion: 5.0.0\nScheduleName: gfx950\nArchitectureName: gfx950\n"
                        "CUCount: null\nDeviceNames: [Device 75a0]\nSolutions: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="ProblemType"):
            SkinnyGEMMLogic.headerLines(path)

    def test_other_keys_are_not_copied(self, tmp_path):
        path = tmp_path / "logic.yaml"
        path.write_text("MinimumRequiredVersion: 5.0.0\nScheduleName: gfx950\nArchitectureName: gfx950\n"
                        "CUCount: null\nDeviceNames: [Device 75a0]\nDefaultSolution:\n  Other: 1\n"
                        "ProblemType:\n  A: 1\nSolutions:\n- SolutionIndex: 0\n", encoding="utf-8")
        lines = SkinnyGEMMLogic.headerLines(path)
        assert lines[-2:] == ["ProblemType:", "  A: 1"]
        assert not any("Other" in line or "SolutionIndex" in line for line in lines)


class TestRender:
    @pytest.fixture(scope="class")
    def rendered(self, pool):
        rows = [([1, 10, 1, -1, 1, 1, 1, -1], 3), ([11, -1, 1, 5, 1, 1, 1, 100], 0)]
        text = SkinnyGEMMLogic.renderLogic(SkinnyGEMMLogic.headerLines(SkinnyGEMMLogic.ORIGAMI_LOGIC), pool, rows)
        return yaml.safe_load(text), rows

    def test_header_and_structure(self, rendered):
        logic, _ = rendered
        origami = yaml.safe_load(
            "\n".join(SkinnyGEMMLogic.headerLines(SkinnyGEMMLogic.ORIGAMI_LOGIC)))
        assert logic["ProblemType"] == origami["ProblemType"]
        assert "DefaultSolution" not in logic
        assert logic["LibraryType"] == "Range"
        assert logic["RangeLogic"] is None

    def test_solutions(self, rendered, pool):
        logic, _ = rendered
        assert len(logic["Solutions"]) == len(pool)
        for index, (solution, knobs) in enumerate(zip(logic["Solutions"], pool)):
            assert solution["SolutionIndex"] == index
            assert solution["CustomKernel"]["generator"] == "SkinnyGEMM"
            assert solution["CustomKernel"]["generatorParams"] == SkinnyGEMMLogic.asParams(knobs)

    def test_rows(self, rendered, pool):
        logic, routing = rendered
        rows = logic["ExactLogic"]
        assert [(key, index) for key, (index, _) in rows[: len(routing)]] == [(k, i) for k, i in routing]
        enumeration = rows[len(routing):]
        assert [key for key, _ in enumeration] == [SkinnyGEMMLogic.EMPTY_RANGE] * len(pool)
        assert [value[0] for _, value in enumeration] == list(range(len(pool)))

    def test_params_carry_use_beta(self, rendered, pool):
        logic, _ = rendered
        flags = [solution["CustomKernel"]["generatorParams"]["useBeta"] for solution in logic["Solutions"]]
        assert flags == [SkinnyGEMMLogic.usesBeta(knobs) for knobs in pool]
        assert set(flags) == {False, True}

    def test_render_is_deterministic(self, pool):
        header = SkinnyGEMMLogic.headerLines(SkinnyGEMMLogic.ORIGAMI_LOGIC)
        rows = [([1, 10, 1, -1, 1, 1, 1, -1], 0)]
        assert SkinnyGEMMLogic.renderLogic(header, pool, rows) == SkinnyGEMMLogic.renderLogic(header, pool, rows)

    def test_routing_params_have_no_use_beta(self):
        assert "useBeta" not in SkinnyGEMMLogic.render_routing_params(RING_KNOBS)
        assert SkinnyGEMMLogic.render_params(RING_KNOBS).endswith("useBeta: false}")


class TestRows:
    def test_valid_file(self, tmp_path, pool):
        row = validRow()
        other = validRow(params=dict(RING_PARAMS, waves=2))
        path = writeRows(tmp_path / "rows.yaml", [row, other])
        indexOf = {knobs: index for index, knobs in enumerate(pool)}
        routing = SkinnyGEMMLogic.readRoutingRows(path, pool)
        assert routing == [
            (row["range"], indexOf[RING_KNOBS]),
            (other["range"], indexOf[SkinnyGEMMLogic.to_knobs(dict(other["params"], useBeta=False))]),
        ]

    def test_rows_route_to_the_use_beta_false_variant(self, tmp_path, pool):
        path = writeRows(tmp_path / "rows.yaml", [validRow()])
        ((_, index),) = SkinnyGEMMLogic.readRoutingRows(path, pool)
        knobs = list(pool)[index]
        assert not SkinnyGEMMLogic.usesBeta(knobs)
        assert knobs == RING_KNOBS

    def test_committed_rows_route_to_the_use_beta_false_variant(self, pool):
        routing = SkinnyGEMMLogic.readRoutingRows(SkinnyGEMMLogic.DEFAULT_ROWS, pool)
        assert routing
        assert not any(SkinnyGEMMLogic.usesBeta(list(pool)[index]) for _, index in routing)

    @pytest.mark.parametrize("useBeta", [False, True])
    def test_rows_with_use_beta_are_rejected(self, tmp_path, pool, useBeta):
        path = writeRows(tmp_path / "rows.yaml", [validRow(params=dict(RING_PARAMS, useBeta=useBeta))])
        with pytest.raises(ValueError, match="must not set useBeta"):
            SkinnyGEMMLogic.readRoutingRows(path, pool)

    @pytest.mark.parametrize("change, message", [
        ({"params": dict(RING_PARAMS, lanesK=4)}, "not in the pool"),
        ({"range": [1, 4096, 1, -1, 1, 1, 1]}, "list of 8 ints"),
        ({"range": [1, 4096, 1, -1, 1, 2, 1, 8192]}, "batch bounds"),
        ({"range": [10, 5, 1, -1, 1, 1, 1, 8192]}, "below its lower bound"),
        ({"range": [0, 5, 1, -1, 1, 1, 1, 8192]}, "at least 1"),
        ({"params": dict(RING_PARAMS, unknown=1)}, "unknown keys"),
    ])
    def test_errors(self, tmp_path, pool, change, message):
        path = writeRows(tmp_path / "rows.yaml", [validRow(**change)])
        with pytest.raises(ValueError, match=message):
            SkinnyGEMMLogic.readRoutingRows(path, pool)

    def test_error_names_the_row(self, tmp_path, pool):
        path = writeRows(tmp_path / "rows.yaml", [validRow(), validRow(range=[1, 2, 3])])
        with pytest.raises(ValueError, match="routing row 1"):
            SkinnyGEMMLogic.readRoutingRows(path, pool)


class TestMain:
    def test_regeneration_is_deterministic(self, tmp_path):
        first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
        SkinnyGEMMLogic.main(["--output", str(first)])
        SkinnyGEMMLogic.main(["--output", str(second)])
        assert first.read_bytes() == second.read_bytes()

    def test_list_prints_one_line_per_entry(self, tmp_path, pool, capsys):
        output = tmp_path / "x.yaml"
        SkinnyGEMMLogic.main(["--list", "--output", str(output)])
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == len(pool) + 2
        assert not output.exists()

