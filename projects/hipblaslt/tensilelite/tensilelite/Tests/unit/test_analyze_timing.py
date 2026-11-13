# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for scripts/analyze_timing.py

These tests verify the timing analysis script's parsing and analysis logic
"""

import pytest

pytestmark = pytest.mark.unit
import gzip
import zlib
import re
import sys
import os
from pathlib import Path
from io import StringIO

# Add scripts directory to path for importing analyze_timing
scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(scripts_dir))

from analyze_timing import (
    parse_timing_line,
    parse_context_line,
    analyze_timing_file,
    build_hierarchy,
    timings_to_tree,
    tree_to_timings,
    print_summary,
    TimingSeries,
    build_purpose_hierarchy,
    build_staged_hierarchy,
    STAGES,
    ROOT_NAME,
    TIMING_HIERARCHY,
    PURPOSE_OF,
    G_REFERENCE,
    G_INPUT,
    G_VALIDATION,
    G_KERNEL,
    G_LAUNCH,
    G_UNATTRIBUTED,
    TimingRecord,
    ProblemTiming,
    ALL_KNOWN_CATEGORIES,
    SOLUTION_GENERATION_PHASES,
    SOLUTION_GENERATION_PARENT,
    BENCH_POSTPROCESS_PHASES,
    BENCH_POSTPROCESS_PARENT,
)


# Sample timing output that mimics real TensileLite output
SAMPLE_TIMING_LOG = """\
Some initial output that should be ignored
TIMING:library_loading:45.123
TIMING:code_object_loading:12.456
TIMING:lazy_loading_init:3.789
TIMING_CONTEXT:M=1024,N=1024,K=512,batch=1,typeA=f32,typeD=f32
TIMING:cpu_data_init:156.234
TIMING:gpu_input_preparation:23.456
TIMING:cpu_reference_gemm:1234.567
TIMING:gpu_kernel_execution:5.678
TIMING_CONTEXT:M=2048,N=2048,K=1024,batch=1,typeA=f32,typeD=f32
TIMING:cpu_data_init:312.468
TIMING:gpu_input_preparation:45.678
TIMING:cpu_reference_gemm:9876.543
TIMING:gpu_kernel_execution:12.345
TIMING:python_solution_generation:500.0
TIMING:python_kernel_compilation:2000.0
TIMING:python_client_execution:15000.0
TIMING:python_benchmark_problems:20000.0
Some trailing output
"""


class TestParseTimingLine:
    def test_record_after_a_pytest_test_id(self):
        line = ("tensilelite/Tests/common/test_config_run.py::test_config_run[x.yaml] "
                "TIMING:python_artifact_extract:735.372\n")
        rec = parse_timing_line(line)
        assert rec is not None
        assert rec.category == "python_artifact_extract"
        assert rec.duration_ms == pytest.approx(735.372)

    def test_record_glued_to_other_text_is_ignored(self):
        assert parse_timing_line("fooTIMING:x:1.0\n") is None

    def test_pytest_captured_log_copy_is_ignored(self):
        # With -rP pytest prints each record twice: the captured stderr line (counted) and the
        # captured log line (ignored), so the record is counted once.
        line = "INFO     tensile.timing:TimingInstrumentation.py:88 TIMING:python_artifact_compress:12.5\n"
        assert parse_timing_line(line) is None
        assert parse_timing_line("TIMING:python_artifact_compress:12.5\n") is not None


    """Tests for parse_timing_line function."""

    def test_valid_timing_line(self):
        record = parse_timing_line("TIMING:cpu_data_init:123.456")
        assert record is not None
        assert record.category == "cpu_data_init"
        assert record.duration_ms == 123.456

    def test_valid_timing_line_integer(self):
        record = parse_timing_line("TIMING:library_loading:100")
        assert record is not None
        assert record.category == "library_loading"
        assert record.duration_ms == 100.0

    def test_valid_timing_line_zero(self):
        record = parse_timing_line("TIMING:warmup_runs:0.0")
        assert record is not None
        assert record.category == "warmup_runs"
        assert record.duration_ms == 0.0

    def test_invalid_line_no_timing_prefix(self):
        assert parse_timing_line("not a timing line") is None

    def test_invalid_line_wrong_format(self):
        assert parse_timing_line("TIMING:missing_value") is None

    def test_invalid_line_empty(self):
        assert parse_timing_line("") is None

    def test_valid_timing_line_scientific_notation_positive(self):
        record = parse_timing_line("TIMING:cpu_reference_gemm:1.5e+03")
        assert record is not None
        assert record.category == "cpu_reference_gemm"
        assert record.duration_ms == 1500.0

    def test_valid_timing_line_scientific_notation_negative(self):
        record = parse_timing_line("TIMING:gpu_kernel_execution:3.2e-04")
        assert record is not None
        assert record.category == "gpu_kernel_execution"
        assert record.duration_ms == pytest.approx(0.00032)

    def test_timing_line_with_surrounding_text(self):
        # A record after other text matches only when whitespace separates them (pytest -v puts
        # the test id before the first record, see test_record_after_a_pytest_test_id); trailing
        # text after the value does not match.
        assert parse_timing_line("prefixTIMING:category:123.0") is None
        assert parse_timing_line("TIMING:category:123.0 trailing") is None


class TestParseContextLine:
    """Tests for parse_context_line function."""

    def test_valid_context_line(self):
        ctx = parse_context_line("TIMING_CONTEXT:M=1024,N=2048,K=512,batch=1,typeA=f32,typeD=f32")
        assert ctx is not None
        assert ctx["M"] == "1024"
        assert ctx["N"] == "2048"
        assert ctx["K"] == "512"
        assert ctx["batch"] == "1"
        assert ctx["typeA"] == "f32"
        assert ctx["typeD"] == "f32"

    def test_context_line_partial(self):
        ctx = parse_context_line("TIMING_CONTEXT:M=100,N=200")
        assert ctx is not None
        assert ctx["M"] == "100"
        assert ctx["N"] == "200"
        assert "K" not in ctx

    def test_invalid_context_line(self):
        assert parse_context_line("not a context line") is None

    def test_empty_context(self):
        ctx = parse_context_line("TIMING_CONTEXT:")
        assert ctx is not None
        assert len(ctx) == 0


class TestAnalyzeTimingFile:
    """Tests for analyze_timing_file function."""

    @pytest.fixture
    def sample_log_file(self, tmp_path):
        """Create a sample log file for testing."""
        log_file = tmp_path / "timing.log"
        log_file.write_text(SAMPLE_TIMING_LOG)
        return log_file

    def test_parses_all_timing_categories(self, sample_log_file):
        timings, _ = analyze_timing_file(str(sample_log_file))

        # Check that expected categories are present
        assert "library_loading" in timings
        assert "cpu_data_init" in timings
        assert "cpu_reference_gemm" in timings
        assert "gpu_kernel_execution" in timings
        assert "python_benchmark_problems" in timings

    def test_timing_values_correct(self, sample_log_file):
        timings, _ = analyze_timing_file(str(sample_log_file))

        assert timings["library_loading"] == [45.123]
        assert timings["cpu_data_init"] == [156.234, 312.468]
        assert timings["cpu_reference_gemm"] == [1234.567, 9876.543]

    def test_problem_timings_extracted(self, sample_log_file):
        _, problem_timings = analyze_timing_file(str(sample_log_file))

        assert len(problem_timings) == 2

        # First problem
        assert problem_timings[0].context["M"] == "1024"
        assert problem_timings[0].cpu_data_init_ms == 156.234
        assert problem_timings[0].cpu_reference_gemm_ms == 1234.567
        assert problem_timings[0].gpu_kernel_execution_ms == 5.678

        # Second problem
        assert problem_timings[1].context["M"] == "2048"
        assert problem_timings[1].cpu_data_init_ms == 312.468
        assert problem_timings[1].cpu_reference_gemm_ms == 9876.543
        assert problem_timings[1].gpu_kernel_execution_ms == 12.345

    def test_ignores_non_timing_lines(self, sample_log_file):
        timings, _ = analyze_timing_file(str(sample_log_file))

        # Should not have any categories from non-timing lines
        assert "Some" not in timings
        assert "initial" not in timings

    def test_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.log"
        empty_file.write_text("")

        timings, problem_timings = analyze_timing_file(str(empty_file))

        assert len(timings) == 0
        assert len(problem_timings) == 0


class TestTimingNesting:
    """Tests to verify understanding of timing nesting relationships."""

    @pytest.fixture
    def nested_timing_log(self, tmp_path):
        """Create a log that demonstrates the nesting structure."""
        # This represents the actual nesting:
        # python_benchmark_problems contains the detail phases
        # python_client_execution contains all C++ phases
        log_content = """\
TIMING:python_solution_generation:1000.0
TIMING:python_kernel_compilation:2000.0
TIMING:library_loading:50.0
TIMING:code_object_loading:25.0
TIMING_CONTEXT:M=1024,N=1024,K=512,batch=1,typeA=f32,typeD=f32
TIMING:cpu_data_init:100.0
TIMING:cpu_reference_gemm:500.0
TIMING:gpu_kernel_execution:10.0
TIMING:python_client_execution:1000.0
TIMING:python_benchmark_problems:5000.0
"""
        log_file = tmp_path / "nested.log"
        log_file.write_text(log_content)
        return log_file

    def test_nested_timing_totals(self, nested_timing_log):
        """
        Verify that we understand the nesting:
        - python_benchmark_problems (5000ms) contains:
          - python_solution_generation (1000ms)
          - python_kernel_compilation (2000ms)
          - python_client_execution (1000ms) which contains C++ phases

        The sum of detail phases (4000ms) < python_benchmark_problems (5000ms)
        because there's uninstrumented overhead.
        """
        timings, _ = analyze_timing_file(str(nested_timing_log))

        python_total = timings["python_benchmark_problems"][0]
        python_detail_sum = (
            timings["python_solution_generation"][0]
            + timings["python_kernel_compilation"][0]
            + timings["python_client_execution"][0]
        )

        # Detail phases should sum to less than or equal to total
        # (less than due to uninstrumented overhead)
        assert python_detail_sum <= python_total

        # C++ phases should sum to less than python_client_execution
        cpp_sum = (
            timings["library_loading"][0]
            + timings["code_object_loading"][0]
            + timings["cpu_data_init"][0]
            + timings["cpu_reference_gemm"][0]
        )
        assert cpp_sum <= timings["python_client_execution"][0]


class TestAllKnownCategories:
    """Tests that ALL_KNOWN_CATEGORIES includes the new sub-timing phases."""

    def test_solution_generation_phases_in_known(self):
        for phase in SOLUTION_GENERATION_PHASES:
            assert phase in ALL_KNOWN_CATEGORIES, f"{phase} not in ALL_KNOWN_CATEGORIES"

    def test_bench_postprocess_phases_in_known(self):
        for phase in BENCH_POSTPROCESS_PHASES:
            assert phase in ALL_KNOWN_CATEGORIES, f"{phase} not in ALL_KNOWN_CATEGORIES"


class TestSolutionGenerationHierarchy:
    """Tests for solution generation sub-phase hierarchy."""

    @pytest.fixture
    def solgen_timing_log(self, tmp_path):
        log_content = """\
TIMING:python_solgen_fork_permutations:50.0
TIMING:python_solgen_forked_solutions:400.0
TIMING:python_solgen_custom_kernels:30.0
TIMING:python_solution_generation:500.0
TIMING:python_kernel_compilation:2000.0
TIMING:python_client_execution:1000.0
TIMING:python_benchmark_problems:5000.0
"""
        log_file = tmp_path / "solgen.log"
        log_file.write_text(log_content)
        return log_file

    def test_solgen_subphases_sum_within_parent(self, solgen_timing_log):
        timings, _ = analyze_timing_file(str(solgen_timing_log))
        parent = timings[SOLUTION_GENERATION_PARENT][0]
        child_sum = sum(timings[p][0] for p in SOLUTION_GENERATION_PHASES if p in timings)
        assert child_sum <= parent

    def test_solgen_hierarchy_built(self, solgen_timing_log):
        timings, _ = analyze_timing_file(str(solgen_timing_log))
        nodes = build_hierarchy(timings)
        # Find the python_benchmark_problems node
        pbp = next(n for n in nodes if n.name == "python_benchmark_problems")
        # Find python_solution_generation child
        solgen = next(c for c in pbp.children if c.name == SOLUTION_GENERATION_PARENT)
        # It should have children (the sub-phases)
        child_names = {c.name for c in solgen.children}
        for phase in SOLUTION_GENERATION_PHASES:
            assert phase in child_names, f"{phase} not in solgen children"


class TestBenchPostprocessHierarchy:
    """Tests for bench postprocess sub-phase hierarchy."""

    @pytest.fixture
    def benchpost_timing_log(self, tmp_path):
        log_content = """\
TIMING:python_benchpost_naming:10.0
TIMING:python_benchpost_lib_construction:50.0
TIMING:python_benchpost_library_write:30.0
TIMING:python_benchpost_client_config:15.0
TIMING:python_kernel_bench_postprocess:120.0
TIMING:python_kernel_compilation:2000.0
TIMING:python_solution_generation:500.0
TIMING:python_client_execution:1000.0
TIMING:python_benchmark_problems:5000.0
"""
        log_file = tmp_path / "benchpost.log"
        log_file.write_text(log_content)
        return log_file

    def test_benchpost_subphases_sum_within_parent(self, benchpost_timing_log):
        timings, _ = analyze_timing_file(str(benchpost_timing_log))
        parent = timings[BENCH_POSTPROCESS_PARENT][0]
        child_sum = sum(timings[p][0] for p in BENCH_POSTPROCESS_PHASES if p in timings)
        assert child_sum <= parent

    def test_benchpost_hierarchy_built(self, benchpost_timing_log):
        timings, _ = analyze_timing_file(str(benchpost_timing_log))
        nodes = build_hierarchy(timings)
        # Find python_benchmark_problems -> python_kernel_compilation -> python_kernel_bench_postprocess
        pbp = next(n for n in nodes if n.name == "python_benchmark_problems")
        kc = next(c for c in pbp.children if c.name == "python_kernel_compilation")
        bp = next(c for c in kc.children if c.name == BENCH_POSTPROCESS_PARENT)
        child_names = {c.name for c in bp.children}
        for phase in BENCH_POSTPROCESS_PHASES:
            assert phase in child_names, f"{phase} not in benchpost children"


class TestGzipLogs:
    """analyze_timing_file reads .gz logs the same as plain text."""

    def test_gz_matches_plain(self, tmp_path):
        plain = tmp_path / "run.log"
        plain.write_text(SAMPLE_TIMING_LOG)
        gz = tmp_path / "run.log.gz"
        with gzip.open(gz, "wt") as f:
            f.write(SAMPLE_TIMING_LOG)

        plain_timings, plain_problems = analyze_timing_file(str(plain))
        gz_timings, gz_problems = analyze_timing_file(str(gz))

        assert dict(gz_timings) == dict(plain_timings)
        assert [p.context for p in gz_problems] == [p.context for p in plain_problems]

    def test_truncated_gz_keeps_the_records_read(self, tmp_path, capsys):
        full = tmp_path / "full.log.gz"
        with gzip.open(full, "wt") as f:
            f.write(SAMPLE_TIMING_LOG)
        truncated = tmp_path / "truncated.log.gz"
        truncated.write_bytes(full.read_bytes()[:-8])  # no CRC/length trailer: EOFError on read

        full_timings, _ = analyze_timing_file(str(full))
        timings, _ = analyze_timing_file(str(truncated))

        assert "ended early" in capsys.readouterr().err
        assert timings
        assert all(timings[cat] == full_timings[cat][:len(timings[cat])] for cat in timings)

    @pytest.mark.parametrize("with_prefix", [False, True])
    def test_corrupt_deflate_warns_and_keeps_parsed_records(self, tmp_path, capsys, with_prefix):
        corrupt = bytearray(gzip.compress(b"unused\n", mtime=0))
        # The gzip header is 10 bytes; DEFLATE block type 3 is reserved and invalid.
        corrupt[10] = (corrupt[10] & ~0x06) | 0x06
        with pytest.raises(zlib.error):
            gzip.decompress(corrupt)

        prefix = b""
        if with_prefix:
            log = "TIMING_CONTEXT:M=64,N=64,K=64\n" + "TIMING:cpu_reference_gemm:20.0\n" * 1000
            # Padding exceeds text read-ahead so all timing lines reach the parser
            # before decompression encounters the following corrupt member.
            prefix = gzip.compress((log + "\n" * 16384).encode(), mtime=0)
        path = tmp_path / "corrupt.log.gz"
        path.write_bytes(prefix + corrupt)

        timings, problems = analyze_timing_file(str(path))

        warning = capsys.readouterr().err
        assert str(path) in warning
        assert "ended early" in warning
        assert "using the records read before that" in warning
        if with_prefix:
            assert dict(timings) == {"cpu_reference_gemm": [20.0] * 1000}
            assert len(problems) == 1
            assert problems[0].context == {"M": "64", "N": "64", "K": "64"}
            assert problems[0].cpu_reference_gemm_ms == 20.0
        else:
            assert not timings
            assert not problems


class TestClientStamps:
    """TIMING_TS stamps turn into process_startup / main_teardown_locals / process_exit."""

    @staticmethod
    def _analyze(tmp_path, stamps):
        log = tmp_path / "run.log"
        log.write_text("".join(f"TIMING_TS:{name}:{int(ms * 1e6)}\n" for name, ms in stamps))
        timings, _ = analyze_timing_file(str(log))
        return timings

    def test_one_client(self, tmp_path):
        timings = self._analyze(tmp_path, [
            ("py_popen_start", 0), ("main_start", 30), ("main_end", 130),
            ("main_locals_destroyed", 140), ("py_communicate_end", 150)])
        assert timings["process_startup"] == [30.0]
        assert timings["main_teardown_locals"] == [10.0]
        assert timings["process_exit"] == [10.0]

    def test_second_client_of_one_popen_does_not_recount_the_first_run(self, tmp_path):
        # One Popen, two clients (the run script starts one per pass).
        timings = self._analyze(tmp_path, [
            ("py_popen_start", 0),
            ("main_start", 30), ("main_end", 130), ("main_locals_destroyed", 140),
            ("main_start", 200), ("main_end", 300), ("main_locals_destroyed", 310),
            ("py_communicate_end", 320)])
        assert timings["process_startup"] == [30.0]
        assert timings["main_teardown_locals"] == [10.0, 10.0]
        assert timings["process_exit"] == [10.0]


@pytest.mark.parametrize("parent_path, check, readback, scan", [
    (("synchronizer_check",), "synchronizer_check", "synchronizer_readback", "synchronizer_scan"),
    (("benchmark_runs", "benchmark_validate_enqueues", "validate_enqueues_listener.SynchronizerValidator", "synchronizer_check_after_enqueues"),
     "synchronizer_check_after_enqueues", "synchronizer_readback_after_enqueues", "synchronizer_scan_after_enqueues"),
    (("post_solution", "synchronizer_check_post_solution"),
     "synchronizer_check_post_solution", "synchronizer_readback_post_solution", "synchronizer_scan_post_solution"),
    (("post_solution", "synchronizer_flag_check_post_solution"),
     "synchronizer_flag_check_post_solution", "synchronizer_flag_readback_post_solution", "synchronizer_flag_scan_post_solution"),
])
def test_synchronizer_contexts_have_one_parent_and_valid_reports(
    parent_path, check, readback, scan, capsys
):
    hierarchy = TIMING_HIERARCHY["python_benchmark_problems"]["python_client_execution"]
    for category in parent_path:
        assert category in hierarchy
        hierarchy = hierarchy[category]
    assert readback in hierarchy and scan in hierarchy

    timings = {"python_benchmark_problems": [10.0], "python_client_execution": [10.0]}
    timings.update({category: [10.0] for category in parent_path[:-1]})
    timings.update({check: [6.0], readback: [5.0], scan: [1.0]})
    print_summary(timings, [])
    report = capsys.readouterr().out
    assert "HIERARCHY VALIDATION: OK" in report
    assert "Unknown Categories" not in report
    assert "children exceed" not in report
    assert sum(group.total_ms for group in build_purpose_hierarchy(timings)) == pytest.approx(10.0)


class TestTimingTree:
    """timings_to_tree / tree_to_timings: the tiered dict of a run's results."""

    def _timings(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(SAMPLE_TIMING_LOG)
        timings, _ = analyze_timing_file(str(log))
        return timings

    def test_follows_hierarchy_with_raw_count_and_total(self, tmp_path):
        tree = timings_to_tree(self._timings(tmp_path))

        top = tree["phases"]["python_benchmark_problems"]
        assert top["count"] == 1 and top["total_ms"] == 20000.0
        client = top["children"]["python_client_execution"]
        assert client["count"] == 1 and client["total_ms"] == 15000.0
        # Two problems -> two cpu_reference_gemm records, summed.
        ref = client["children"]["pre_problem"]["children"]["cpu_reference_gemm"]
        assert ref["count"] == 2
        assert ref["total_ms"] == pytest.approx(1234.567 + 9876.543)

    def test_unlogged_categories_are_left_out(self, tmp_path):
        tree = timings_to_tree(self._timings(tmp_path))
        top = tree["phases"]["python_benchmark_problems"]
        assert "python_cache_check" not in top["children"]
        assert tree["overlapping_totals"] == {}
        assert tree["unknown_categories"] == {}

    def test_unknown_category_is_kept_separately(self):
        tree = timings_to_tree({"made_up_category": [1.0, 2.0]})
        assert tree["phases"] == {}
        assert tree["unknown_categories"] == {"made_up_category": {"count": 2, "total_ms": 3.0}}

    def test_roundtrip_preserves_count_and_total(self, tmp_path):
        timings = self._timings(tmp_path)
        back = tree_to_timings(timings_to_tree(timings))

        assert set(back) == set(timings)
        for category, values in timings.items():
            assert len(back[category]) == len(values)
            assert sum(back[category]) == pytest.approx(sum(values))


class TestColumnAlignment:
    """The numeric columns of the phase table stay aligned however large the values get."""

    def _table_numeric_edges(self, out):
        lines = out.split("\n")
        header_i = next(i for i, l in enumerate(lines) if l.strip().startswith("Category"))
        end_i = next(i for i, l in enumerate(lines)
                     if "ESTIMATED WALL CLOCK" in l or l.startswith(("HIERARCHY VALIDATION",
                                                                      "TIME BREAKDOWN")))
        header = lines[header_i]
        allowed = {header.index(k) + len(k)
                   for k in ("Count", "Total (ms)", "Mean (ms)", "% of Parent", "% of Wall")}
        checked = 0
        for line in lines[header_i + 1:end_i]:
            ends = [m.end() for m in re.finditer(r"[\d,]+(?:\.\d+)?%?(?=\s|$)", line)
                    if m.start() > header.index("Count") - 14]
            for end in ends:
                assert end in allowed, f"misaligned: {line!r}"
            checked += bool(ends)
        return checked

    def test_huge_counts_and_totals(self, capsys):
        # Counts in the millions and totals in the billions of ms, as when many runs are summed.
        timings = {
            "python_benchmark_problems": TimingSeries(184, 9_876_543_210.12),
            "python_client_execution": TimingSeries(855, 9_800_000_000.5),
            "pre_problem": TimingSeries(28_641_123, 1_035_798_421.42),
            "cpu_reference_gemm": TimingSeries(28_596_001, 928_712_973.97),
            "solve_cpu_fast": TimingSeries(26_656_000, 200_931_330.0),
            "made_up_category": TimingSeries(12_345_678, 5_000_000.0),
            "main_total": TimingSeries(855, 8_000_000_000.0),
        }
        print_summary(timings, [])
        out = capsys.readouterr().out

        assert self._table_numeric_edges(out) >= 6
        # The rule lines are as wide as the table.
        widths = {len(l) for l in out.split("\n")[:3] if l.strip() and set(l.strip()) <= set("=-")}
        assert len(widths) == 1


class TestPurposeHierarchy:
    """The default tree is grouped by what the time is spent on, not by where the timer is called."""

    TIMINGS = {
        "python_benchmark_problems": [1000.0],
        "python_client_execution": [900.0],
        "hip_initialization": [50.0],
        "pre_problem": [300.0],
        "cpu_data_init": [100.0],
        "cpu_reference_gemm": [150.0],
        "solve_cpu_fast": [140.0],
        "pre_solution": [200.0],
        "pre_solution_listener.ReferenceValidator": [120.0],
        "cpu_reference_gemm_per_solution_inputs_changed": [110.0],
        "solve_cpu_fast_per_solution_inputs_changed": [105.0],
        "pre_solution_listener.DataInitialization": [60.0],
        "validate_warmups": [180.0],
        "validate_warmups_listener.SynchronizerValidator": [100.0],
        "validate_warmups_listener.BenchmarkTimer": [20.0],
        "validate_gpu_sync": [15.0],
    }

    def _groups(self):
        return {g.name: g for g in build_purpose_hierarchy(dict(self.TIMINGS))}

    @staticmethod
    def _names(node):
        out = set()
        for child in node.children:
            out.add(child.name)
            out |= TestPurposeHierarchy._names(child)
        return out

    def test_categories_move_to_their_purpose_group(self):
        groups = self._groups()

        # The reference GEMM is one group even though it is called from two different places.
        assert {"cpu_reference_gemm", "solve_cpu_fast", "pre_solution_listener.ReferenceValidator",
                "cpu_reference_gemm_per_solution_inputs_changed"} <= self._names(groups[G_REFERENCE])
        assert "cpu_data_init" in self._names(groups[G_INPUT])
        assert "pre_solution_listener.DataInitialization" in self._names(groups[G_INPUT])
        assert "validate_warmups_listener.SynchronizerValidator" in self._names(groups[G_VALIDATION])
        # The warmup sync wait belongs to kernel execution, although it is called from validation.
        assert "validate_warmups_listener.BenchmarkTimer" in self._names(groups[G_KERNEL])
        assert "hip_initialization" in self._names(groups[G_LAUNCH])

    def test_containers_get_no_row_and_every_category_is_shown_once(self):
        groups = self._groups()
        shown = [n for g in groups.values() for n in self._names(g) if not n.startswith("(")]
        assert len(shown) == len(set(shown))
        for container in ("python_benchmark_problems", "python_client_execution", "pre_problem",
                          "pre_solution", "validate_warmups"):
            assert container not in shown

    def test_groups_add_up_to_the_nesting_tree(self):
        nesting_total = sum(n.total_ms for n in build_hierarchy(dict(self.TIMINGS)))
        purpose_total = sum(g.total_ms for g in self._groups().values())
        assert purpose_total == pytest.approx(nesting_total, abs=0.05)

    def test_containers_report_what_is_left_over_as_unattributed(self):
        timings = {
            "python_benchmark_problems": [100.0], "python_client_execution": [100.0],
            "validate_warmups": [100.0],
            "validate_warmups_listener.BenchmarkTimer": [30.0], "validate_gpu_sync": [30.0],
            "validate_warmups_listener.SynchronizerValidator": [60.0],
        }
        groups = {g.name: g for g in build_purpose_hierarchy(timings)}
        assert groups[G_VALIDATION].total_ms == pytest.approx(60.0)
        assert groups[G_KERNEL].total_ms == pytest.approx(30.0)
        # validate_warmups itself accounted for 10 ms that no sub-timer covers: reported, not lost.
        assert groups[G_UNATTRIBUTED].total_ms == pytest.approx(10.0)

    def test_a_row_loses_the_time_of_children_that_moved_elsewhere(self, monkeypatch):
        # benchmark_runs is a row of the kernel group; move one of its timers to another group.
        monkeypatch.setitem(PURPOSE_OF, "benchmark_pre_syncs", G_LAUNCH)
        timings = {
            "python_benchmark_problems": [100.0], "python_client_execution": [100.0],
            "benchmark_runs": [100.0], "benchmark_pre_syncs": [25.0], "benchmark_post_syncs": [5.0],
        }
        groups = {g.name: g for g in build_purpose_hierarchy(timings)}
        runs = next(c for c in groups[G_KERNEL].children if c.name == "benchmark_runs")
        assert runs.total_ms == pytest.approx(75.0)                   # 100 - the 25 that moved
        assert groups[G_LAUNCH].total_ms == pytest.approx(25.0)
        assert sum(g.total_ms for g in groups.values()) == pytest.approx(100.0)

    def test_every_direct_child_of_a_phase_has_an_explicit_purpose(self):
        # A new timer under these must be given a purpose on purpose, not by inheriting.
        top = TIMING_HIERARCHY["python_benchmark_problems"]["python_client_execution"]
        for name in list(TIMING_HIERARCHY) + list(top):
            assert name in PURPOSE_OF, f"{name} has no purpose group"

    def test_report_has_the_root_row_then_purpose_then_nesting_table(self, capsys):
        print_summary(dict(self.TIMINGS), [])
        out = capsys.readouterr().out
        lines = out.split("\n")

        # The first row of the first table is the total runtime, at 100% of the wall clock.
        purpose_start = next(i for i, l in enumerate(lines) if l.startswith("TIMING BY PURPOSE"))
        root = next(l for l in lines[purpose_start:] if l.strip().startswith(ROOT_NAME))
        assert root.rstrip().endswith("100.0%")
        # The purpose table comes first and the old (timer-nested) table below it.
        assert out.index("TIMING BY PURPOSE") < out.index("TIMING BY TIMER NESTING")
        assert out.index(G_REFERENCE) < out.index("Data Preparation")
        # The columns of the two tables line up.
        headers = [l for l in lines if l.strip().startswith("Category")]
        assert len(headers) == 2 and headers[0] == headers[1]


class TestStagedHierarchy:
    """One root (the total runtime) over RUN / BUILD stages, for build-only, run-only and both."""

    BUILD = {
        "python_benchmark_problems": [1000.0],
        "python_cache_check": [2.0],
        "python_kernel_compilation": [900.0],
        "python_write_cache": [3.0],
    }
    RUN = {
        "python_benchmark_problems": [500.0],
        "python_cache_check": [1.0],
        "python_client_execution": [450.0],
        "cpu_reference_gemm": [300.0],
        "pre_problem": [320.0],
        "hip_initialization": [100.0],
    }

    @staticmethod
    def _tree(timings):
        (root,) = build_staged_hierarchy(timings)
        return root, {c.name: c for c in root.children}

    def test_build_only_log_has_only_the_build_stage(self):
        root, top = self._tree(dict(self.BUILD))
        assert "BUILD: solution generation & compile" in top
        assert "RUN: benchmark client" not in top

    def test_run_only_log_has_only_the_run_stage(self):
        root, top = self._tree(dict(self.RUN))
        assert "RUN: benchmark client" in top
        assert "BUILD: solution generation & compile" not in top

    def test_combined_logs_have_both_stages_and_sum_their_processes(self):
        # A combined run is two processes, so their records are concatenated (summed).
        combined = {k: list(self.BUILD.get(k, [])) + list(self.RUN.get(k, []))
                    for k in set(self.BUILD) | set(self.RUN)}
        root, top = self._tree(combined)
        assert "RUN: benchmark client" in top
        assert "BUILD: solution generation & compile" in top
        assert root.total_ms == pytest.approx(1500.0)       # both python_benchmark_problems

    def test_root_is_the_wall_clock_and_equals_its_children(self):
        for timings in (self.BUILD, self.RUN):
            root, _ = self._tree(dict(timings))
            assert root.total_ms == pytest.approx(sum(timings["python_benchmark_problems"]), abs=0.05)
            assert sum(c.total_ms for c in root.children) == pytest.approx(root.total_ms)

    def test_stage_rows_gather_their_groups(self):
        _, top = self._tree(dict(self.RUN))
        assert {c.name for c in top["RUN: benchmark client"].children} >= {G_REFERENCE, G_LAUNCH}
        assert G_REFERENCE not in top        # nested under the stage, not next to it

    def test_every_stage_group_is_a_known_purpose_group(self):
        known = set(PURPOSE_OF.values())
        for _stage, members in STAGES:
            assert set(members) <= known

    def test_artifact_packing_is_build_time_in_the_total(self):
        # Packing runs after the Tensile invocation, so it is a top-level phase of its own.
        timings = dict(self.BUILD, python_artifact_compress=[250.0])
        root, top = self._tree(timings)
        assert root.total_ms == pytest.approx(1250.0, abs=0.05)
        build = top["BUILD: solution generation & compile"]
        assert "Artifact packing (test harness, .tar.gz)" in {c.name for c in build.children}

    def test_artifact_extraction_is_run_time_in_the_total(self):
        timings = dict(self.RUN, python_artifact_extract=[40.0])
        root, top = self._tree(timings)
        assert root.total_ms == pytest.approx(540.0, abs=0.05)
        run = top["RUN: benchmark client"]
        assert "Artifact extraction (test harness, .tar.gz)" in {c.name for c in run.children}
