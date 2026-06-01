################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
import os
import tempfile
import yaml
from Tensile.BenchmarkSplitter import BenchmarkSplitter


@pytest.fixture
def sample_benchmark_config():
    """Sample benchmark YAML config with multiple problems and sizes"""
    return {
        "GlobalParameters": {
            "MinimumRequiredVersion": "4.0.0"
        },
        "BenchmarkProblems": [
            # Problem 1
            [
                # Problem type group
                {
                    "OperationType": "GEMM",
                    "DataType": "f32"
                },
                # Benchmark group 1
                {
                    "BenchmarkFinalParameters": [
                        {
                            "ProblemSizes": [
                                [64, 64, 64],
                                [128, 128, 128],
                                [256, 256, 256]
                            ]
                        }
                    ]
                },
                # Benchmark group 2
                {
                    "BenchmarkFinalParameters": [
                        {
                            "ProblemSizes": [
                                [512, 512, 512],
                                [1024, 1024, 1024]
                            ]
                        }
                    ]
                }
            ],
            # Problem 2
            [
                {
                    "OperationType": "GEMM",
                    "DataType": "f16"
                },
                {
                    "BenchmarkFinalParameters": [
                        {
                            "ProblemSizes": [
                                [32, 32, 32],
                                [64, 64, 64]
                            ]
                        }
                    ]
                }
            ]
        ]
    }


@pytest.fixture
def single_problem_config():
    """Sample benchmark YAML config with single problem"""
    return {
        "GlobalParameters": {
            "MinimumRequiredVersion": "4.0.0"
        },
        "BenchmarkProblems": [
            [
                {
                    "OperationType": "GEMM",
                    "DataType": "f32"
                },
                {
                    "BenchmarkFinalParameters": [
                        {
                            "ProblemSizes": [
                                [64, 64, 64],
                                [128, 128, 128],
                                [256, 256, 256],
                                [512, 512, 512],
                                [1024, 1024, 1024]
                            ]
                        }
                    ]
                }
            ]
        ]
    }


@pytest.fixture
def temp_config_file(sample_benchmark_config):
    """Create a temporary YAML config file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(sample_benchmark_config, f)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestBenchmarkSplitterReadConfig:
    """Test reading configuration files"""

    def test_read_config_file(self, temp_config_file, sample_benchmark_config):
        """Test reading a YAML config file"""
        # Use private method via name mangling
        data = BenchmarkSplitter._BenchmarkSplitter__readConfigFile(temp_config_file)

        assert data is not None
        assert "GlobalParameters" in data
        assert "BenchmarkProblems" in data
        assert data["GlobalParameters"]["MinimumRequiredVersion"] == "4.0.0"
        assert len(data["BenchmarkProblems"]) == 2

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist"""
        with pytest.raises(FileNotFoundError):
            BenchmarkSplitter._BenchmarkSplitter__readConfigFile("nonexistent.yaml")


class TestBenchmarkSplitterSplitByProblem:
    """Test splitting benchmarks by problem"""

    def test_split_by_problem_multiple(self, sample_benchmark_config):
        """Test splitting config with multiple problems"""
        result = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(sample_benchmark_config)

        # Should split into 2 separate configs (one per problem)
        assert len(result) == 2

        # Each result should have GlobalParameters preserved
        for r in result:
            assert "GlobalParameters" in r
            assert r["GlobalParameters"]["MinimumRequiredVersion"] == "4.0.0"

        # Each result should have exactly 1 BenchmarkProblem
        assert len(result[0]["BenchmarkProblems"]) == 1
        assert len(result[1]["BenchmarkProblems"]) == 1

        # Check first problem has f32 DataType
        assert result[0]["BenchmarkProblems"][0][0]["DataType"] == "f32"

        # Check second problem has f16 DataType
        assert result[1]["BenchmarkProblems"][0][0]["DataType"] == "f16"

    def test_split_by_problem_preserves_structure(self, sample_benchmark_config):
        """Test that splitting preserves the structure of each problem"""
        result = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(sample_benchmark_config)

        # First problem should have 3 groups (problem type + 2 benchmark groups)
        assert len(result[0]["BenchmarkProblems"][0]) == 3

        # Second problem should have 2 groups (problem type + 1 benchmark group)
        assert len(result[1]["BenchmarkProblems"][0]) == 2


class TestBenchmarkSplitterSplitByBenchmarkGroup:
    """Test splitting benchmarks by benchmark group"""

    def test_split_by_benchmark_group(self, single_problem_config):
        """Test splitting single problem by benchmark groups"""
        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(single_problem_config)

        # Should have 1 result (problem has 1 problem type + 1 benchmark group)
        assert len(result) == 1

        # Each result should have the problem type + one benchmark group
        for r in result:
            assert len(r["BenchmarkProblems"][0]) == 2
            # First should be OperationType
            assert "OperationType" in r["BenchmarkProblems"][0][0]
            # Second should be BenchmarkFinalParameters
            assert "BenchmarkFinalParameters" in r["BenchmarkProblems"][0][1]

    def test_split_by_benchmark_group_multiple_groups(self, sample_benchmark_config):
        """Test splitting when there are multiple benchmark groups"""
        # First split by problem to get single problem configs
        problems = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(sample_benchmark_config)

        # Split first problem by benchmark group
        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(problems[0])

        # Should have 2 results (2 benchmark groups in first problem)
        assert len(result) == 2

        # Each should have problem type + one benchmark group
        for r in result:
            assert len(r["BenchmarkProblems"][0]) == 2

    def test_split_by_benchmark_group_assertion_multiple_problems(self, sample_benchmark_config):
        """Test that assertion fails when config has multiple problems"""
        with pytest.raises(AssertionError, match="must have one BenchmarkProblems group"):
            BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(sample_benchmark_config)

    def test_split_by_benchmark_group_assertion_no_operation_type(self):
        """Test that assertion fails when no OperationType found"""
        bad_config = {
            "BenchmarkProblems": [
                [
                    {"SomeOtherKey": "value"},
                    {"BenchmarkFinalParameters": [{"ProblemSizes": []}]}
                ]
            ]
        }
        with pytest.raises(AssertionError, match="Could not find problem type group"):
            BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(bad_config)


class TestBenchmarkSplitterSplitByBenchmarkSizes:
    """Test splitting benchmarks by sizes"""

    def test_split_by_benchmark_sizes_single(self, single_problem_config):
        """Test splitting by sizes with numSizes=1"""
        # First prepare config with single problem and single group
        problems = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(single_problem_config)
        groups = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(problems[0])

        # Split by sizes (1 size per file)
        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(groups[0], numSizes=1)

        # Should have 5 results (5 problem sizes)
        assert len(result) == 5

        # Each result should have exactly 1 problem size
        for r in result:
            sizes = r["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
            assert len(sizes) == 1

    def test_split_by_benchmark_sizes_multiple(self, single_problem_config):
        """Test splitting by sizes with numSizes=2"""
        problems = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(single_problem_config)
        groups = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(problems[0])

        # Split by sizes (2 sizes per file)
        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(groups[0], numSizes=2)

        # Should have 3 results (ceiling(5/2) = 3)
        assert len(result) == 3

        # First two should have 2 sizes each
        assert len(result[0]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]) == 2
        assert len(result[1]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]) == 2

        # Last should have 1 size (remainder)
        assert len(result[2]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]) == 1

    def test_split_by_benchmark_sizes_preserves_sizes(self, single_problem_config):
        """Test that actual problem sizes are preserved correctly"""
        problems = BenchmarkSplitter._BenchmarkSplitter__splitByProblem(single_problem_config)
        groups = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkGroup(problems[0])

        # Split by sizes
        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(groups[0], numSizes=2)

        # Check first file has first two sizes
        sizes_0 = result[0]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
        assert sizes_0[0] == [64, 64, 64]
        assert sizes_0[1] == [128, 128, 128]

        # Check second file has next two sizes
        sizes_1 = result[1]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
        assert sizes_1[0] == [256, 256, 256]
        assert sizes_1[1] == [512, 512, 512]

        # Check third file has last size
        sizes_2 = result[2]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
        assert sizes_2[0] == [1024, 1024, 1024]

    def test_split_by_benchmark_sizes_assertion_multiple_problems(self, sample_benchmark_config):
        """Test assertion fails with multiple problems"""
        with pytest.raises(AssertionError, match="must have one BenchmarkProblems group"):
            BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(sample_benchmark_config)

    def test_split_by_benchmark_sizes_assertion_wrong_structure(self):
        """Test assertion fails when structure is wrong"""
        bad_config = {
            "BenchmarkProblems": [
                [
                    {"OperationType": "GEMM"}
                    # Missing benchmark group
                ]
            ]
        }
        with pytest.raises(AssertionError, match="must have one ProblemType group and one Benchmark group"):
            BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(bad_config)

    def test_split_by_benchmark_sizes_assertion_no_problem_sizes(self):
        """Test assertion fails when ProblemSizes is missing"""
        bad_config = {
            "BenchmarkProblems": [
                [
                    {"OperationType": "GEMM"},
                    {"BenchmarkFinalParameters": [{"SomeOtherKey": "value"}]}
                ]
            ]
        }
        with pytest.raises(AssertionError, match="must have non-empty ProblemSizes"):
            BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(bad_config)


class TestBenchmarkSplitterFileNameUtils:
    """Test filename manipulation utilities"""

    def test_append_file_name_suffix_default(self):
        """Test appending suffix with default formatting"""
        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "benchmark.yaml", 0
        )
        assert result == "benchmark_00.yaml"

        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "benchmark.yaml", 5
        )
        assert result == "benchmark_05.yaml"

        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "benchmark.yaml", 42
        )
        assert result == "benchmark_42.yaml"

    def test_append_file_name_suffix_custom_separator(self):
        """Test appending suffix with custom separator"""
        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "benchmark.yaml", 3, separator="-"
        )
        assert result == "benchmark-03.yaml"

    def test_append_file_name_suffix_custom_format(self):
        """Test appending suffix with custom formatting"""
        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "benchmark.yaml", 7, formatting="{:04}"
        )
        assert result == "benchmark_0007.yaml"

        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "benchmark.yaml", 99, formatting="{:d}"
        )
        assert result == "benchmark_99.yaml"

    def test_append_file_name_suffix_with_path(self):
        """Test appending suffix when file has path"""
        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "/path/to/benchmark.yaml", 10
        )
        assert result == "/path/to/benchmark_10.yaml"

    def test_append_file_name_suffix_no_extension(self):
        """Test appending suffix to file without extension"""
        result = BenchmarkSplitter._BenchmarkSplitter__appendFileNameSuffix(
            "benchmark", 5
        )
        assert result == "benchmark_05"


class TestBenchmarkSplitterEndToEnd:
    """Test end-to-end splitting functionality"""

    def test_split_benchmark_by_sizes_basic(self, temp_config_file, temp_output_dir):
        """Test basic end-to-end splitting"""
        BenchmarkSplitter.splitBenchmarkBySizes(
            temp_config_file,
            temp_output_dir,
            numSizes=1
        )

        # Check that output files were created
        output_files = os.listdir(temp_output_dir)
        assert len(output_files) > 0

        # Verify files are valid YAML
        for fname in output_files:
            fpath = os.path.join(temp_output_dir, fname)
            with open(fpath, 'r') as f:
                data = yaml.safe_load(f)
                assert data is not None
                assert "BenchmarkProblems" in data

    def test_split_benchmark_by_sizes_custom_base_name(self, temp_config_file, temp_output_dir):
        """Test splitting with custom base filename"""
        BenchmarkSplitter.splitBenchmarkBySizes(
            temp_config_file,
            temp_output_dir,
            numSizes=2,
            baseFileName="custom_benchmark.yaml"
        )

        output_files = os.listdir(temp_output_dir)

        # All files should start with "custom_benchmark"
        for fname in output_files:
            assert fname.startswith("custom_benchmark_")
            assert fname.endswith(".yaml")

    def test_split_benchmark_by_sizes_custom_separator(self, temp_config_file, temp_output_dir):
        """Test splitting with custom separator"""
        BenchmarkSplitter.splitBenchmarkBySizes(
            temp_config_file,
            temp_output_dir,
            numSizes=1,
            separator="-"
        )

        output_files = os.listdir(temp_output_dir)

        # Files should use - separator
        for fname in output_files:
            assert "-" in fname

    def test_split_benchmark_by_sizes_custom_suffix_format(self, temp_config_file, temp_output_dir):
        """Test splitting with custom suffix format"""
        BenchmarkSplitter.splitBenchmarkBySizes(
            temp_config_file,
            temp_output_dir,
            numSizes=1,
            suffixFormat="{:04}"
        )

        output_files = os.listdir(temp_output_dir)

        # Files should have 4-digit suffixes
        for fname in output_files:
            # Extract the number part
            parts = fname.replace(".yaml", "").split("_")
            suffix = parts[-1]
            assert len(suffix) == 4
            assert suffix.isdigit()

    def test_split_benchmark_by_sizes_default_base_name(self, temp_config_file, temp_output_dir):
        """Test that default base name uses config file name"""
        BenchmarkSplitter.splitBenchmarkBySizes(
            temp_config_file,
            temp_output_dir,
            numSizes=1,
            baseFileName=""  # Use default
        )

        output_files = os.listdir(temp_output_dir)
        config_basename = os.path.basename(temp_config_file)
        config_name_no_ext = os.path.splitext(config_basename)[0]

        # Files should start with the config file's base name
        for fname in output_files:
            assert fname.startswith(config_name_no_ext)

    def test_split_benchmark_validates_output_content(self, temp_config_file, temp_output_dir):
        """Test that output files have correct structure and content"""
        BenchmarkSplitter.splitBenchmarkBySizes(
            temp_config_file,
            temp_output_dir,
            numSizes=2
        )

        # Read one of the output files and validate structure
        output_files = sorted(os.listdir(temp_output_dir))
        first_file = os.path.join(temp_output_dir, output_files[0])

        with open(first_file, 'r') as f:
            data = yaml.safe_load(f)

        # Should have GlobalParameters
        assert "GlobalParameters" in data

        # Should have exactly 1 BenchmarkProblems entry
        assert len(data["BenchmarkProblems"]) == 1

        # Should have exactly 2 groups (problem type + benchmark group)
        assert len(data["BenchmarkProblems"][0]) == 2

        # First group should have OperationType
        assert "OperationType" in data["BenchmarkProblems"][0][0]

        # Second group should have BenchmarkFinalParameters with ProblemSizes
        assert "BenchmarkFinalParameters" in data["BenchmarkProblems"][0][1]
        assert "ProblemSizes" in data["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]

        # Number of sizes should be <= numSizes (2 in this case)
        sizes = data["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
        assert len(sizes) <= 2
        assert len(sizes) > 0


class TestBenchmarkSplitterEdgeCases:
    """Test edge cases and error conditions"""

    def test_split_by_sizes_single_size(self):
        """Test splitting when there's only one problem size"""
        config = {
            "BenchmarkProblems": [
                [
                    {"OperationType": "GEMM"},
                    {
                        "BenchmarkFinalParameters": [
                            {"ProblemSizes": [[64, 64, 64]]}
                        ]
                    }
                ]
            ]
        }

        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(config, numSizes=1)

        # Should have 1 result
        assert len(result) == 1

        # Should preserve the single size
        sizes = result[0]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
        assert len(sizes) == 1
        assert sizes[0] == [64, 64, 64]

    def test_split_by_sizes_numSizes_larger_than_total(self):
        """Test splitting when numSizes > total problem sizes"""
        config = {
            "BenchmarkProblems": [
                [
                    {"OperationType": "GEMM"},
                    {
                        "BenchmarkFinalParameters": [
                            {"ProblemSizes": [[64, 64, 64], [128, 128, 128]]}
                        ]
                    }
                ]
            ]
        }

        result = BenchmarkSplitter._BenchmarkSplitter__splitByBenchmarkSizes(config, numSizes=10)

        # Should have 1 result (ceiling(2/10) = 1)
        assert len(result) == 1

        # Should have all 2 sizes in one file
        sizes = result[0]["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
        assert len(sizes) == 2
