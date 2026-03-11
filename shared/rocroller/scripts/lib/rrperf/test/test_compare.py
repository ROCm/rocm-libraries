# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rrperf.compare module with focus on resource tracking."""

import io
import sys
from pathlib import Path

import yaml
from rrperf.compare import compare
from rrperf.problems import GEMMResult

repo_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(repo_dir / "scripts" / "lib"))

FILE_DIR = Path(__file__).parent.resolve()


def write_sample_yaml(target_dir: Path, source_file: Path, m_override: int = None):
    target_dir.mkdir(parents=True, exist_ok=True)
    result = yaml.safe_load(source_file.read_text())
    if m_override is not None:
        problem = result.get("problem")
        if not isinstance(problem, dict):
            raise ValueError(
                "Expected nested GEMM schema with a 'problem' mapping in sample YAML."
            )
        problem["M"] = m_override
    (target_dir / "gemm-000000.yaml").write_text(
        yaml.safe_dump(result, sort_keys=False)
    )


class TestResourceMarkdown:
    def test_any_increase_resource_usage(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1_increase_usage"
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="resource_md",
            output=result_io,
        )
        result = result_io.getvalue()
        assert "- SGPR: 102 -> 104 (+2) | VGPR: 206 -> 205 (-1)" in result

    def test_decrease_resource_usage(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1_decrease_usage"
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="resource_md",
            output=result_io,
        )
        result = result_io.getvalue()
        assert "+ VGPR: 206 -> 199 (-7) | LDS: 131072 -> 131070 bytes (-2)" in result

    def test_same_dir(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1"  # same dir
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="resource_md",
            output=result_io,
        )
        result_lower = result_io.getvalue().lower()
        assert "sgpr" not in result_lower
        assert "vgpr" not in result_lower


class TestHTML:
    def test_decrease_resource_usage(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        modified_dir = samples_dir / "1_decrease_usage"
        assert original_dir.exists()
        assert modified_dir.exists()
        compare(
            directories=[str(original_dir), str(modified_dir)],
            format="html",
            output=result_io,
        )
        result = result_io.getvalue()
        # These are the GPR and LDS value pairs from samples
        expected_values = """
                    <td> 102 </td>
                    <td> 102 </td>
                    <td> 206 </td>
                    <td> 199 </td>
                    <td> 256 </td>
                    <td> 256 </td>
                    <td> 131,072 </td>
                    <td> 131,070 </td>
        """
        for line in expected_values.splitlines():
            line = line.strip()
            if line:
                assert line in result

    def test_comparison_summary_counts(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        compare(
            directories=[str(original_dir), str(original_dir)],
            format="html",
            output=result_io,
        )
        result = result_io.getvalue()
        assert "<h2>Comparison Summary</h2>" in result
        assert "<li>Compared: 1</li>" in result
        assert "<li>Significant diffs: 0</li>" in result
        assert "<li>Insignificant diffs: 1</li>" in result
        assert "<li>Not compared (total): 0</li>" in result


class TestComparisonSummaryMarkdown:
    def test_counts_for_matching_runs(self):
        result_io = io.StringIO()
        samples_dir = FILE_DIR / "samples"
        original_dir = samples_dir / "1"
        compare(
            directories=[str(original_dir), str(original_dir)],
            format="md",
            output=result_io,
        )
        result = result_io.getvalue()
        assert "- Compared: 1" in result
        assert "- Significant diffs: 0" in result
        assert "- Insignificant diffs: 1" in result
        assert "- Not compared (reference-only): 0" in result
        assert "- Not compared (candidate-only): 0" in result
        assert "- Not compared (total): 0" in result

    def test_counts_for_non_matching_runs(self, tmp_path):
        samples_dir = FILE_DIR / "samples"
        source_file = samples_dir / "1" / "gemm-000000.yaml"

        reference_dir = tmp_path / "reference"
        candidate_dir = tmp_path / "candidate"
        write_sample_yaml(reference_dir, source_file, m_override=4096)
        write_sample_yaml(candidate_dir, source_file, m_override=8192)

        result_io = io.StringIO()
        compare(
            directories=[str(reference_dir), str(candidate_dir)],
            format="md",
            output=result_io,
        )
        result = result_io.getvalue()
        assert "- Compared: 0" in result
        assert "- Significant diffs: 0" in result
        assert "- Insignificant diffs: 0" in result
        assert "- Not compared (reference-only): 1" in result
        assert "- Not compared (candidate-only): 1" in result
        assert "- Not compared (total): 2" in result


class TestRunInvariantToken:
    def test_gemm_token_ignores_version_metadata(self):
        baseline = GEMMResult(
            resultType="GEMM",
            path=Path("baseline.yaml"),
            kernelGenerate=1,
            kernelAssemble=1,
            kernelExecute=[1],
            version="abc12345",
        )
        candidate = GEMMResult(
            resultType="GEMM",
            path=Path("candidate.yaml"),
            kernelGenerate=1,
            kernelAssemble=1,
            kernelExecute=[1],
            version="def67890",
        )
        assert baseline.run_invariant_token == candidate.run_invariant_token
