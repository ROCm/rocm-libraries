# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from bench_fakes import BANNER, Gemm, measured_lines, preamble, skip, winner
from lib import bench_log
from lib.bench_log import LogStats, ProblemBoundaryTracker, iter_problems

G1 = Gemm(256, 512, 1024)
G2 = Gemm(64, 32, 4096, 4, "N", "T")


def _full_problem(g=G1, **kw):
    return (
        preamble(4, adaptive=kw.get("adaptive", False))
        + measured_lines(0, g, 10, 20.0, **kw)
        + measured_lines(1, g, 11, 12.5, **kw)
        + [skip(2, 12, "1.2e+02", best="12")]
        + [skip(3, 13, "456.79")]
        + winner(1, g, 11, 12.5, **kw)
    )


def test_tested_rows_are_mapped_by_their_header():
    stats = LogStats()
    probs = list(
        iter_problems(BANNER + _full_problem(split_k=True, adaptive=True), stats)
    )
    assert len(probs) == 1
    p = probs[0]
    assert p.complete and p.n_supported == 4
    assert [t.rank for t in p.tested] == [0, 1]
    row = p.tested[1]
    assert row.sol_index == 11
    assert row.fields["us"] == "12.5"
    assert row.fields["rotating_buffer"] == "16"
    assert row.fields["splitK"] == "2"
    assert row.fields["status"] == "converged"
    assert row.fields["samples"] == "50"
    assert row.sol_name.startswith("Cijk_") and row.kernel_name == row.sol_name
    assert p.winner is not None and p.winner.sol_index == 11
    assert p.shape() == (256, 512, 1024, 1)
    assert stats.problems == 1 and stats.malformed_rows == 0


@pytest.mark.parametrize(
    "warm, expected",
    [("1.2e+02", 120.0), ("456.79", 456.79), ("99", 99.0), ("inf", None)]
    + [("-nan", None), (".5", 0.5)],
)
def test_skip_warm_up_formats(warm, expected):
    (p,) = list(
        iter_problems(preamble(2) + measured_lines(0, G1, 1, 5.0) + [skip(1, 7, warm)])
    )
    assert p.skipped[0].warm_up_us == expected
    assert p.skipped[0].sol_index == 7
    assert p.complete


def test_skip_without_index_and_unparsed_skip_lines():
    stats = LogStats()
    lines = (
        preamble(3)
        + measured_lines(0, G1, 1, 5.0)
        + [skip(1, None, "8.0"), "Skip solution: 2 (garbled output)"]
    )
    (p,) = list(iter_problems(lines, stats))
    assert p.skipped[0].sol_index is None
    assert stats.skip_lines == 2 and stats.unparsed_skip_lines == 1
    assert stats.unparsed_skip_examples == ["Skip solution: 2 (garbled output)"]
    assert not p.complete


def test_single_candidate_problem_is_complete_without_winner():
    (p,) = list(iter_problems(preamble(1) + measured_lines(0, G1, 1, 5.0)))
    assert p.complete and not p.winner_seen


def test_crash_leaves_an_incomplete_problem():
    lines = (
        BANNER
        + _full_problem(G1)
        + preamble(4)
        + measured_lines(0, G2, 20, 3.0)
        + ["", "*** SKIPPED CRASHED GEMM at block-offset 1 (yaml line 2): M=64 ***"]
        + BANNER
        + _full_problem(G1)
    )
    probs = list(iter_problems(lines))
    assert [p.complete for p in probs] == [True, False, True]
    assert probs[1].shape() == (64, 32, 4096, 4)


def test_no_solution_problem_and_rotation_off():
    lines = (
        preamble(2, rotating=False)
        + measured_lines(0, G1, 1, 5.0)
        + measured_lines(1, G1, 2, 4.0)
        + winner(1, G1, 2, 4.0)
        + preamble(2)[:1]
        + ["error: NO solution found! at testing_matmul.hpp:1"]
        + preamble(1, rotating=False)
        + measured_lines(0, G2, 3, 1.0)
    )
    probs = list(iter_problems(lines))
    assert len(probs) == 3
    assert probs[0].complete
    assert probs[1].no_solution and probs[1].finished and not probs[1].complete
    assert probs[2].complete


def test_malformed_tested_row_is_counted():
    stats = LogStats()
    lines = preamble(1) + ["[0]:transA,transB,m", "    T,N", "Winner: "]
    (p,) = list(iter_problems(lines, stats))
    assert stats.malformed_rows == 1 and not p.tested and p.complete


def test_tracker_agrees_with_parser():
    lines = (
        BANNER
        + _full_problem(G1, adaptive=True)
        + preamble(2)[:1]
        + ["error: NO solution found! at x"]
        + preamble(1, rotating=False)
        + measured_lines(0, G2, 3, 1.0)
        + ["*** KILLED [stall]: no output for 300s ***"]
        + measured_lines(0, G2, 4, 1.0)
        + _full_problem(G2)
    )
    tracker = ProblemBoundaryTracker()
    starts = [i for i, ln in enumerate(lines, 1) if tracker.feed(ln)]
    assert starts == [p.line_no for p in iter_problems(lines)]
    assert len(starts) == 5


def test_count_finished_problems():
    one = preamble(3) + measured_lines(0, G1, 1, 5.0) + measured_lines(1, G1, 2, 4.0)
    assert bench_log.count_finished_problems(BANNER) == 0
    assert bench_log.count_finished_problems(BANNER + one) == 0
    assert bench_log.count_finished_problems(BANNER + one + winner(1, G1, 2, 4.0)) == 1
    all_printed = (
        preamble(2) + measured_lines(0, G1, 1, 5.0) + measured_lines(1, G1, 2, 4.0)
    )
    assert bench_log.count_finished_problems(all_printed) == 1
    two = _full_problem(G1) + preamble(1) + measured_lines(0, G2, 3, 1.0)
    assert bench_log.count_finished_problems(two) == 2
    crashed_setup = _full_problem(G1) + preamble(3)[:1]
    assert bench_log.count_finished_problems(crashed_setup) == 1
    no_solution = _full_problem(G1) + preamble(2)[:1] + ["NO solution found!"]
    assert bench_log.count_finished_problems(no_solution) == 2


def test_parse_log_file(tmp_path):
    path = tmp_path / "block_0000_lines1-1_gpu0.log"
    path.write_text("\n".join(BANNER + _full_problem()) + "\n")
    (p,) = bench_log.parse_log_file(path)
    assert p.complete and len(p.skipped) == 2
