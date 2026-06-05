# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for scripts/analyze_timing.py

These tests verify the timing analysis script's parsing and analysis logic
"""

import pytest

pytestmark = pytest.mark.unit
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
    TimingRecord,
    ProblemTiming,
    ALL_KNOWN_CATEGORIES,
    SOLUTION_GENERATION_PHASES,
    SOLUTION_GENERATION_PARENT,
    BENCH_POSTPROCESS_PHASES,
    BENCH_POSTPROCESS_PARENT,
)


# Sample timing output that mimics real Tensile output
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
        # Should not match if TIMING is not at start
        assert parse_timing_line("prefix TIMING:category:123.0") is None


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
            + timings["gpu_kernel_execution"][0]
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
