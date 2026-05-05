# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for the convert-miopen-shapes CLI."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    entry_point = shutil.which("dnn-convert-shapes")
    if entry_point:
        cmd = [entry_point, *args]
    else:
        cmd = [
            sys.executable,
            "-c",
            "from dnn_benchmarking.tools.convert_miopen_shapes import main; "
            "import sys; sys.exit(main())",
            *args,
        ]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestInlineArgs:
    """Tests for --args / -A inline conversion."""

    def test_conv_inline_produces_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "conv.json"
        result = _run_cli(
            "-A",
            "convbfp16 -n 2 -c 64 -H 8 -W 8 -k 128 -y 3 -x 3 -F 1",
            "--output",
            str(out),
        )
        assert result.returncode == 0
        graph = json.loads(out.read_text())
        assert graph["io_data_type"] == "bfloat16"
        assert graph["nodes"][0]["type"] == "ConvolutionFwdAttributes"

    def test_bnorm_inline_produces_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "bnorm.json"
        result = _run_cli(
            "-A",
            "bnormbfp16 -n 4 -c 32 -H 16 -W 16 --forw 2",
            "--output",
            str(out),
        )
        assert result.returncode == 0
        graph = json.loads(out.read_text())
        assert graph["io_data_type"] == "bfloat16"
        assert graph["nodes"][0]["type"] == "BatchnormInferenceAttributes"

    def test_empty_args_shows_help(self) -> None:
        result = _run_cli("-A", "")
        assert result.returncode == 1

    def test_unsupported_operation_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "bad.json"
        result = _run_cli("-A", "matmul -n 1", "--output", str(out))
        assert result.returncode == 1
        assert not out.exists()

    def test_f0_inline_produces_three_files(self, tmp_path: Path) -> None:
        result = _run_cli(
            "-A",
            "convbfp16 -n 2 -c 64 -H 8 -W 8 -k 128 -y 3 -x 3 -F 0",
        )
        assert result.returncode == 0
        assert result.stdout.count("Written:") == 3

    def test_f0_inline_with_output_fails(self, tmp_path: Path) -> None:
        out = tmp_path / "conv.json"
        result = _run_cli(
            "-A",
            "convbfp16 -n 2 -c 64 -H 8 -W 8 -k 128 -y 3 -x 3 -F 0",
            "--output",
            str(out),
        )
        assert result.returncode == 1


class TestFileProcessing:
    """Tests for shape file conversion."""

    def test_single_file_produces_json_per_line(self, tmp_path: Path) -> None:
        shapes = tmp_path / "shapes.txt"
        shapes.write_text(
            "convbfp16 -n 2 -c 32 -H 8 -W 8 -k 64 -y 1 -x 1 -F 1\n"
            "convbfp16 -n 4 -c 64 -H 16 -W 16 -k 128 -y 3 -x 3 -F 1\n"
        )
        outdir = tmp_path / "out"
        result = _run_cli(str(shapes), "--outdir", str(outdir))
        assert result.returncode == 0
        jsons = list(outdir.glob("*.json"))
        assert len(jsons) == 2
        for j in jsons:
            graph = json.loads(j.read_text())
            assert "nodes" in graph
            assert "tensors" in graph

    def test_comments_and_blanks_skipped(self, tmp_path: Path) -> None:
        shapes = tmp_path / "shapes.txt"
        shapes.write_text(
            "# comment line\n"
            "\n"
            "convbfp16 -n 1 -c 8 -H 4 -W 4 -k 16 -y 1 -x 1 -F 1\n"
            "   \n"
        )
        outdir = tmp_path / "out"
        result = _run_cli(str(shapes), "--outdir", str(outdir))
        assert result.returncode == 0
        assert len(list(outdir.glob("*.json"))) == 1

    def test_duplicate_shapes_get_deduplicated_names(self, tmp_path: Path) -> None:
        line = "convbfp16 -n 1 -c 8 -H 4 -W 4 -k 16 -y 1 -x 1 -F 1\n"
        shapes = tmp_path / "shapes.txt"
        shapes.write_text(line * 3)
        outdir = tmp_path / "out"
        result = _run_cli(str(shapes), "--outdir", str(outdir))
        assert result.returncode == 0
        jsons = sorted(outdir.glob("*.json"))
        assert len(jsons) == 3
        stems = [j.stem for j in jsons]
        assert len(set(stems)) == 3

    def test_missing_file_fails(self) -> None:
        result = _run_cli("/nonexistent/shapes.txt")
        assert result.returncode == 1

    def test_invalid_line_warns_but_succeeds(self, tmp_path: Path) -> None:
        shapes = tmp_path / "shapes.txt"
        shapes.write_text(
            "convbfp16 -n 1 -c 8 -H 4 -W 4 -k 16 -y 1 -x 1 -F 1\n" "matmul -n 1 -c 8\n"
        )
        outdir = tmp_path / "out"
        result = _run_cli(str(shapes), "--outdir", str(outdir))
        assert result.returncode == 0
        assert "WARNING" in result.stderr
        assert len(list(outdir.glob("*.json"))) == 1

    def test_f0_file_produces_three_jsons_per_line(self, tmp_path: Path) -> None:
        shapes = tmp_path / "shapes.txt"
        shapes.write_text("convbfp16 -n 1 -c 8 -H 4 -W 4 -k 16 -y 1 -x 1 -F 0\n")
        outdir = tmp_path / "out"
        result = _run_cli(str(shapes), "--outdir", str(outdir))
        assert result.returncode == 0
        jsons = sorted(outdir.glob("*.json"))
        assert len(jsons) == 3
        node_types = set()
        for j in jsons:
            graph = json.loads(j.read_text())
            node_types.add(graph["nodes"][0]["type"])
        assert node_types == {
            "ConvolutionFwdAttributes",
            "ConvolutionBwdAttributes",
            "ConvolutionWrwAttributes",
        }

    def test_executable_prefix_stripped(self, tmp_path: Path) -> None:
        shapes = tmp_path / "shapes.txt"
        shapes.write_text(
            "./bin/MIOpenDriver convbfp16 -n 2 -c 32 -H 8 -W 8 -k 64 -y 1 -x 1 -F 1\n"
        )
        outdir = tmp_path / "out"
        result = _run_cli(str(shapes), "--outdir", str(outdir))
        assert result.returncode == 0
        assert len(list(outdir.glob("*.json"))) == 1


class TestArgConflicts:
    """Tests for CLI argument conflicts."""

    def test_args_and_file_conflict(self, tmp_path: Path) -> None:
        shapes = tmp_path / "shapes.txt"
        shapes.write_text("convbfp16 -n 1 -c 8 -H 4 -W 4 -k 16 -y 1 -x 1 -F 1\n")
        result = _run_cli("-A", "convbfp16 -n 1", str(shapes))
        assert result.returncode == 1

    def test_no_args_prints_help(self) -> None:
        result = _run_cli()
        assert result.returncode == 1
        assert "usage" in result.stderr.lower() or "help" in result.stderr.lower()
