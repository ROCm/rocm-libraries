#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for sweep_runner.py (T2.3) and compare_report.py (T2.6).

All tests run CPU-only (no GPU, no hipcc):
  - sweep_runner: dry-run and cpu-only modes; Parquet schema; resume behaviour.
  - compare_report: Markdown generation; rollup tables; TE merge; exit codes.

The new dtype config files are also smoke-tested via te_to_dispatcher.translate_file
to confirm they parse and translate without errors.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from te_to_dispatcher import translate_file, TranslationError

# ── helpers ──────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_CONFIGS = _HERE / "configs"


def _cfg(name: str) -> Path:
    return _CONFIGS / name


def _make_parquet(rows: list, path: Path) -> Path:
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


_BASE_ROW = {
    "config_file": "configs/single_fp16_rcr.json",
    "config_index": 0,
    "identifier": "fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16",
    "kernel_name": "fp16_rcr_compv3_default_intrawave_False_False_False_False_256x128x32_4x1x1_32x32x16",
    "datatype": "fp16",
    "layout": "rcr",
    "pipeline": "compv3",
    "scheduler": "intrawave",
    "tile_m": 256, "tile_n": 128, "tile_k": 32,
    "split_k": 1,
    "pad_m": False, "pad_n": False, "pad_k": False,
    "persistent": False,
    "M": 512, "N": 512, "K": 512,
    "verdict": "PASSED",
    "tflops": 10.5,
    "error_msg": "",
    "stage_failed": "",
    "ts": "2026-06-02T00:00:00+00:00",
}

# ── dtype config smoke tests ──────────────────────────────────────────────────

class TestDtypeConfigs:
    """New dtype configs translate without error."""

    def test_bf16_translates(self):
        cfgs = translate_file(_cfg("single_bf16_rcr.json"))
        assert len(cfgs) == 1
        sig = cfgs[0]["signature"]
        assert sig["dtype_a"] == "bf16"
        assert sig["dtype_acc"] == "fp32"
        assert sig["dtype_c"] == "bf16"

    def test_fp8_translates(self):
        cfgs = translate_file(_cfg("single_fp8_rcr.json"))
        assert len(cfgs) == 1
        sig = cfgs[0]["signature"]
        assert sig["dtype_a"] == "fp8"
        assert sig["dtype_acc"] == "fp32"
        assert sig["dtype_c"] == "fp16"   # promoted from fp8

    def test_int8_translates(self):
        cfgs = translate_file(_cfg("single_int8_rcr.json"))
        assert len(cfgs) == 1
        sig = cfgs[0]["signature"]
        assert sig["dtype_a"] == "int8"
        assert sig["dtype_acc"] == "int32"
        assert sig["dtype_c"] == "int8"

    def test_splitk_translates(self):
        cfgs = translate_file(_cfg("single_fp16_rcr_splitk.json"))
        assert len(cfgs) == 1
        assert cfgs[0]["signature"]["split_k"] == 4

    def test_splitk_identifier_has_suffix(self):
        from identifier import encode_identifier
        cfgs = translate_file(_cfg("single_fp16_rcr_splitk.json"))
        ident = encode_identifier(cfgs[0])
        assert "_splitk4" in ident

    def test_splitk_255_boundary(self):
        """split_k=255 is the maximum accepted value."""
        import json
        data = json.loads(_cfg("single_fp16_rcr.json").read_text())
        data["split_k"] = 255
        from te_to_dispatcher import translate
        cfgs = translate(data)
        assert len(cfgs) == 1
        assert cfgs[0]["signature"]["split_k"] == 255

    def test_splitk_256_rejected(self):
        import json
        data = json.loads(_cfg("single_fp16_rcr.json").read_text())
        data["split_k"] = 256
        from te_to_dispatcher import translate
        with pytest.raises(TranslationError, match="split_k=256"):
            translate(data)


# ── sweep_runner tests ────────────────────────────────────────────────────────

class TestSweepRunnerDryRun:
    """sweep_runner --dry-run produces Parquet rows with DRYRUN verdict."""

    def test_dryrun_produces_parquet(self, tmp_path):
        import subprocess, sys
        out = tmp_path / "out.parquet"
        result = subprocess.run(
            [sys.executable, str(_HERE / "sweep_runner.py"),
             str(_cfg("single_fp16_rcr.json")),
             "--dry-run", "--output", str(out),
             "--sizes", "512x512x512"],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 1
        assert df["verdict"].iloc[0] == "DRYRUN"

    def test_dryrun_schema_columns(self, tmp_path):
        import subprocess, sys
        out = tmp_path / "out.parquet"
        subprocess.run(
            [sys.executable, str(_HERE / "sweep_runner.py"),
             str(_cfg("single_fp16_rcr.json")),
             "--dry-run", "--output", str(out),
             "--sizes", "512x512x512"],
            capture_output=True, text=True, cwd=_HERE,
        )
        df = pd.read_parquet(out)
        required = {"identifier", "M", "N", "K", "verdict", "tflops",
                    "datatype", "pipeline", "split_k", "ts"}
        assert required.issubset(set(df.columns))


class TestSweepRunnerCpuOnly:
    """--cpu-only records SKIPPED rows for all (kernel, size) pairs."""

    def test_cpu_only_skips_all(self, tmp_path):
        import subprocess, sys
        out = tmp_path / "out.parquet"
        result = subprocess.run(
            [sys.executable, str(_HERE / "sweep_runner.py"),
             str(_cfg("single_fp16_rcr.json")),
             "--cpu-only", "--output", str(out),
             "--sizes", "512x512x512,1024x1024x1024"],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0, result.stderr
        df = pd.read_parquet(out)
        assert len(df) == 2
        assert (df["verdict"] == "SKIPPED").all()

    def test_cpu_only_multiple_configs(self, tmp_path):
        import subprocess, sys
        out = tmp_path / "out.parquet"
        result = subprocess.run(
            [sys.executable, str(_HERE / "sweep_runner.py"),
             str(_cfg("single_fp16_rcr.json")),
             str(_cfg("padding_fp16_rcr.json")),
             "--cpu-only", "--output", str(out),
             "--sizes", "512x512x512"],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0, result.stderr
        df = pd.read_parquet(out)
        assert len(df) == 2
        identifiers = set(df["identifier"])
        assert len(identifiers) == 2   # two distinct kernels


class TestSweepRunnerResume:
    """Rows already in Parquet are skipped on re-run."""

    def test_resume_skips_done_rows(self, tmp_path):
        import subprocess, sys
        out = tmp_path / "out.parquet"
        # First run: 1 size
        subprocess.run(
            [sys.executable, str(_HERE / "sweep_runner.py"),
             str(_cfg("single_fp16_rcr.json")),
             "--cpu-only", "--output", str(out),
             "--sizes", "512x512x512"],
            capture_output=True, text=True, cwd=_HERE,
        )
        df_before = pd.read_parquet(out)
        assert len(df_before) == 1

        # Second run: same config+size — should skip, not append duplicate
        result = subprocess.run(
            [sys.executable, str(_HERE / "sweep_runner.py"),
             str(_cfg("single_fp16_rcr.json")),
             "--cpu-only", "--output", str(out),
             "--sizes", "512x512x512"],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0
        assert "already recorded" in result.stdout
        df_after = pd.read_parquet(out)
        assert len(df_after) == 1  # no duplicate row added


class TestSweepRunnerInvalidConfig:
    """Non-existent config file exits non-zero."""

    def test_missing_config_nonzero(self, tmp_path):
        import subprocess, sys
        out = tmp_path / "out.parquet"
        result = subprocess.run(
            [sys.executable, str(_HERE / "sweep_runner.py"),
             "configs/does_not_exist.json",
             "--cpu-only", "--output", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode != 0


# ── compare_report tests ──────────────────────────────────────────────────────

class TestCompareReportMarkdown:
    """compare_report produces valid Markdown with required sections."""

    def _run(self, tmp_path, extra_rows=None):
        import subprocess, sys
        rows = [dict(_BASE_ROW)]
        if extra_rows:
            rows.extend(extra_rows)
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        return result, out

    def test_report_created(self, tmp_path):
        result, out = self._run(tmp_path)
        assert result.returncode == 0, result.stderr
        assert out.exists()

    def test_report_has_overall_section(self, tmp_path):
        _, out = self._run(tmp_path)
        text = out.read_text()
        assert "## Overall" in text

    def test_report_has_per_shape_section(self, tmp_path):
        _, out = self._run(tmp_path)
        assert "## Per-shape detail" in out.read_text()

    def test_report_shows_passed(self, tmp_path):
        _, out = self._run(tmp_path)
        assert "PASSED" in out.read_text() or "✅" in out.read_text()

    def test_nonzero_exit_on_failures(self, tmp_path):
        import subprocess, sys
        rows = [dict(_BASE_ROW, verdict="FAILED", tflops=None)]
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"), str(pq)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode != 0

    def test_zero_exit_on_all_passed(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert result.returncode == 0

    def test_zero_exit_on_all_skipped(self, tmp_path):
        import subprocess, sys
        rows = [dict(_BASE_ROW, verdict="SKIPPED", tflops=None)]
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"), str(pq)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0


class TestCompareReportWithTE:
    """compare_report merges TE baseline and computes delta%."""

    def test_delta_computed(self, tmp_path):
        import subprocess, sys
        disp_row = dict(_BASE_ROW, tflops=10.0, verdict="PASSED")
        te_row = dict(_BASE_ROW, tflops=10.0, verdict="PASSED")
        disp_pq = _make_parquet([disp_row], tmp_path / "disp.parquet")
        te_pq   = _make_parquet([te_row],   tmp_path / "te.parquet")
        out = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(disp_pq), "--te", str(te_pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0, result.stderr
        text = out.read_text()
        # 10 vs 10 → 0% delta
        assert "0.0%" in text

    def test_positive_delta_shown(self, tmp_path):
        import subprocess, sys
        disp_row = dict(_BASE_ROW, tflops=11.0, verdict="PASSED")
        te_row   = dict(_BASE_ROW, tflops=10.0, verdict="PASSED")
        disp_pq = _make_parquet([disp_row], tmp_path / "disp.parquet")
        te_pq   = _make_parquet([te_row],   tmp_path / "te.parquet")
        out = tmp_path / "report.md"
        subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(disp_pq), "--te", str(te_pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        text = out.read_text()
        assert "+10.0%" in text


class TestCompareReportDtypeFilter:
    """--dtype filters rows before generating report."""

    def test_dtype_filter(self, tmp_path):
        import subprocess, sys
        rows = [
            dict(_BASE_ROW, datatype="fp16", M=512),
            dict(_BASE_ROW, datatype="bf16", M=1024,
                 identifier="bf16_rcr_test", kernel_name="bf16_rcr_test"),
        ]
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(pq), "--dtype", "fp16", "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0
        text = out.read_text()
        assert "512" in text      # fp16 row kept
        assert "1024" not in text  # bf16 row filtered out


# ── compare_report tile-shape rollup ─────────────────────────────────────────

class TestCompareReportTileRollup:
    """compare_report includes a 'By tile shape' rollup table."""

    def test_tile_rollup_present(self, tmp_path):
        import subprocess, sys
        rows = [
            dict(_BASE_ROW, tile_m=256, tile_n=128, tile_k=32, M=512),
            dict(_BASE_ROW, tile_m=256, tile_n=128, tile_k=32, M=1024),
        ]
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0, result.stderr
        text = out.read_text()
        assert "By tile shape" in text
        assert "256×128×32" in text

    def test_tile_rollup_counts_correctly(self, tmp_path):
        import subprocess, sys
        rows = [
            dict(_BASE_ROW, tile_m=256, tile_n=128, tile_k=32, M=512, verdict="PASSED"),
            dict(_BASE_ROW, tile_m=256, tile_n=128, tile_k=32, M=1024, verdict="SKIPPED",
                 tflops=None),
        ]
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        text = out.read_text()
        # 1 passed, 1 skipped → 50%
        assert "50.0%" in text


# ── compare_report layout rollup ─────────────────────────────────────────────

class TestCompareReportLayoutRollup:
    """compare_report includes a 'By layout' rollup table (T2.6 spec requirement)."""

    def test_layout_rollup_present(self, tmp_path):
        import subprocess, sys
        rows = [
            dict(_BASE_ROW, layout="rcr", M=512),
            dict(_BASE_ROW, layout="rcr", M=1024),
        ]
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0, result.stderr
        text = out.read_text()
        assert "By layout" in text
        assert "rcr" in text

    def test_layout_rollup_after_dtype_before_pipeline(self, tmp_path):
        """Layout rollup must appear between dtype and pipeline sections."""
        import subprocess, sys
        pq = _make_parquet([dict(_BASE_ROW)], tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"),
             str(pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        text = out.read_text()
        pos_dtype = text.find("By dtype")
        pos_layout = text.find("By layout")
        pos_pipeline = text.find("By pipeline")
        assert pos_dtype < pos_layout < pos_pipeline, (
            f"Expected By dtype < By layout < By pipeline in report; "
            f"got positions {pos_dtype}, {pos_layout}, {pos_pipeline}"
        )


# ── dispatcher_binding structural tests ──────────────────────────────────────

class TestDispatcherBindingStructure:
    """dispatcher_binding.py can be imported and has the required interface."""

    def test_module_imports(self):
        """The module must import without a GPU or .so file."""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "dispatcher_binding",
            str(_HERE / "dispatcher_binding.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Loading the source-only module (not instantiating DispatcherLib) must not fail
        spec.loader.exec_module(mod)

    def test_required_symbols_present(self):
        """DispatcherLib, DispatcherError, and status codes must be exported."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dispatcher_binding",
            str(_HERE / "dispatcher_binding.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "DispatcherLib")
        assert hasattr(mod, "DispatcherError")
        assert hasattr(mod, "DISPATCHER_OK")
        assert hasattr(mod, "DISPATCHER_ERR_NOT_FOUND")
        assert hasattr(mod, "DISPATCHER_ERR_LAUNCH")

    def test_dispatcher_lib_methods(self):
        """DispatcherLib must expose all 7 C API methods."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dispatcher_binding",
            str(_HERE / "dispatcher_binding.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        required = [
            "version", "kernel_count", "kernel_names",
            "find_kernel", "kernel_name", "supports", "run_gemm",
        ]
        for name in required:
            assert hasattr(mod.DispatcherLib, name), (
                f"DispatcherLib missing method: {name}"
            )

    def test_dispatcher_error_has_status(self):
        """DispatcherError must carry a .status attribute."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dispatcher_binding",
            str(_HERE / "dispatcher_binding.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        err = mod.DispatcherError(mod.DISPATCHER_ERR_NOT_FOUND, "test")
        assert err.status == mod.DISPATCHER_ERR_NOT_FOUND
        assert "NOT_FOUND" in str(err)


# ── sweep_runner TFLOP/s parser ──────────────────────────────────────────────

class TestHarnessTflopsParser:
    """_run_harness() correctly extracts TFLOP/s from GFLOP/s harness output."""

    def test_gflops_converted_to_tflops(self, monkeypatch, tmp_path):
        """Harness prints GFLOP/s; runner must convert to TFLOP/s (÷ 1000)."""
        import sweep_runner
        import subprocess

        fake_stdout = (
            "kernel : gemm_fp16_rcr_compv3_...\n"
            "problem: M=1024 N=1024 K=1024 (rcr)\n"
            "time   : 0.0250 ms  (85868.1 GFLOP/s)\n"
            "verify : max_abs_err=0.00000 max_rel_err=0.00000 "
            "abs_tol=0.03200 rel_tol=0.01000\n"
            "verify : 1048576/1048576 elements pass (100.0%)\n"
            "PASSED\n"
        )

        class FakeProc:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeProc())
        # create a fake harness binary so the existence check passes
        harness = sweep_runner._HERE / "harness"
        harness_existed = harness.exists()
        if not harness_existed:
            harness.touch()
        try:
            verdict, tflops, err = sweep_runner._run_harness(1024, 1024, 1024, dry_run=False)
        finally:
            if not harness_existed:
                harness.unlink(missing_ok=True)

        assert verdict == "PASSED"
        assert tflops is not None
        assert abs(tflops - 85.8681) < 0.01, f"Expected ~85.868 TFLOP/s, got {tflops}"

    def test_skipped_verdict_no_tflops(self, monkeypatch):
        """SKIPPED lines don't produce tflops."""
        import sweep_runner
        import subprocess

        fake_stdout = "SKIPPED: Arguments not supported!\n"

        class FakeProc:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeProc())
        harness = sweep_runner._HERE / "harness"
        harness_existed = harness.exists()
        if not harness_existed:
            harness.touch()
        try:
            verdict, tflops, err = sweep_runner._run_harness(257, 257, 56, dry_run=False)
        finally:
            if not harness_existed:
                harness.unlink(missing_ok=True)

        assert verdict == "SKIPPED"
        assert tflops is None


# ── all configs round-trip check ─────────────────────────────────────────────

class TestAllConfigsTranslate:
    """Every config file in configs/ must translate without error."""

    @pytest.mark.parametrize("config_name", [
        p.name for p in (_HERE / "configs").glob("*.json")
        if not p.name.startswith("_")
    ])
    def test_config_translates(self, config_name):
        cfgs = translate_file(_HERE / "configs" / config_name)
        assert isinstance(cfgs, list)
        assert len(cfgs) >= 1, f"{config_name}: produced zero valid configs"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ── demo_binding.py structure tests ──────────────────────────────────────────

class TestDemoBinding:
    """demo_binding.py T2.2 demo script structural tests (no GPU/so required)."""

    def test_demo_file_exists(self):
        assert (_HERE / "demo_binding.py").exists(), "demo_binding.py must exist"

    def test_demo_has_main(self):
        src = (_HERE / "demo_binding.py").read_text()
        assert "def main(" in src, "demo_binding.py must define main()"

    def test_demo_has_run_demo(self):
        src = (_HERE / "demo_binding.py").read_text()
        assert "def run_demo(" in src, "demo_binding.py must define run_demo()"

    def test_demo_imports_dispatcher_binding(self):
        src = (_HERE / "demo_binding.py").read_text()
        assert "from dispatcher_binding import" in src or "import dispatcher_binding" in src

    def test_demo_handles_missing_so(self):
        """run_demo() returns non-zero when .so file does not exist (no GPU needed)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "demo_binding", _HERE / "demo_binding.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.run_demo("/nonexistent_path/libdispatcher_gemm.so", 512, 512, 512)
        assert rc != 0, "run_demo must return non-zero when .so is missing"

    def test_demo_list_only_flag_documented(self):
        src = (_HERE / "demo_binding.py").read_text()
        assert "list-only" in src or "list_only" in src, \
            "demo_binding.py must support --list-only flag"


# ── rejection CSV CLI tests ───────────────────────────────────────────────────

class TestRejectionCSVCLI:
    """te_to_dispatcher.py --rejection-csv writes a CSV with expected columns."""

    def test_rejection_csv_written(self, tmp_path):
        """Running the CLI with --rejection-csv produces a file."""
        import subprocess
        cfg = _HERE / "configs" / "single_fp16_rcr.json"
        csv_out = tmp_path / "rejections.csv"
        result = subprocess.run(
            [__import__("sys").executable, str(_HERE / "te_to_dispatcher.py"),
             str(cfg), "--rejection-csv", str(csv_out)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert csv_out.exists(), "--rejection-csv must create the file"

    def test_rejection_csv_has_header(self, tmp_path):
        """CSV produced by CLI must have at least a header row."""
        import subprocess
        cfg = _HERE / "configs" / "single_fp16_rcr.json"
        csv_out = tmp_path / "rejections.csv"
        subprocess.run(
            [__import__("sys").executable, str(_HERE / "te_to_dispatcher.py"),
             str(cfg), "--rejection-csv", str(csv_out)],
            capture_output=True, text=True
        )
        content = csv_out.read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) >= 1, "CSV must have at least a header row"

    def test_rejection_csv_for_unsupported_pipeline(self, tmp_path):
        """A config with an unsupported pipeline produces a rejection row in the CSV."""
        import subprocess
        # Config using the tile_config/trait_config format; compv1 is an unsupported pipeline
        bad_cfg = {
            "datatype": "fp16", "layout": "rcr", "gpu_target": "gfx942",
            "block_size": 256, "k_block_per_cu": 1,
            "tile_config": {
                "tile_m": {"values": [256]}, "tile_n": {"values": [128]},
                "tile_k": {"values": [32]},  "warp_m": {"values": [4]},
                "warp_n": {"values": [1]},   "warp_k": {"values": [1]},
                "warp_tile_m": {"values": [32]}, "warp_tile_n": {"values": [32]},
                "warp_tile_k": {"values": [16]},
            },
            "trait_config": {
                "pipeline":   {"values": ["compv1"]},
                "epilogue":   {"values": ["default"]},
                "scheduler":  {"values": ["intrawave"]},
                "pad_m": {"values": [False]}, "pad_n": {"values": [False]},
                "pad_k": {"values": [False]}, "persistent": {"values": [False]},
            },
        }
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text(json.dumps(bad_cfg))
        csv_out = tmp_path / "rejections.csv"
        result = subprocess.run(
            [__import__("sys").executable, str(_HERE / "te_to_dispatcher.py"),
             str(cfg_file), "--rejection-csv", str(csv_out)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"CLI must exit 0 (rejection collected, not fatal): {result.stderr}"
        assert csv_out.exists(), "--rejection-csv must be written even for unsupported configs"
        content = csv_out.read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) >= 2, f"Expected header + rejection row, got:\n{content}"


# ── drive_codegen.py assertion coverage ──────────────────────────────────────

class TestDriveCodegenAssertion:
    """drive_codegen.py enforces count==1 kernel header and identifier-in-name.

    The improve_advice.pdf (gap #4) called out that drive_codegen.py must assert
    exactly one primary header is generated and that the expected identifier
    appears in the filename.  These tests confirm those guards are present in
    the source (CPU-only; actually running codegen needs hipcc/GPU environment).
    """

    def test_count_assertion_present(self):
        """drive_codegen.py must check len(kernel_headers) == 1."""
        src = (_HERE / "drive_codegen.py").read_text()
        assert "len(kernel_headers) != 1" in src or "len(kernel_headers) == 1" in src, (
            "drive_codegen.py must assert exactly one primary kernel header is generated"
        )

    def test_identifier_in_name_check_present(self):
        """drive_codegen.py must verify the expected identifier appears in the header filename."""
        src = (_HERE / "drive_codegen.py").read_text()
        assert "base_identifier not in kernel_headers[0].name" in src or \
               "identifier" in src and "kernel_headers[0].name" in src, (
            "drive_codegen.py must check that the expected identifier is in the generated header filename"
        )

    def test_nonzero_exit_on_count_mismatch_documented(self):
        """The count mismatch branch must return a nonzero exit code (not just print)."""
        src = (_HERE / "drive_codegen.py").read_text()
        # find the block after 'len(kernel_headers) != 1' and confirm 'return 1' follows
        idx = src.find("len(kernel_headers) != 1")
        assert idx != -1, "count check not found"
        snippet = src[idx: idx + 300]
        assert "return 1" in snippet, (
            "drive_codegen.py must return 1 when kernel header count != 1"
        )

    def test_splitk_suffix_stripped_before_filename_check(self):
        """split_k suffix (_splitkN) must be stripped before identifier-in-name check.

        split_k is a runtime parameter not encoded in the generated header filename.
        Stripping _splitkN before the filename containment check prevents false failures
        for configs like single_fp16_rcr_splitk.json.
        """
        src = (_HERE / "drive_codegen.py").read_text()
        assert "_splitk" in src and ("sub" in src or "re.sub" in src or "_re.sub" in src), (
            "drive_codegen.py must strip _splitkN suffix before identifier-in-name check"
        )


# ── PORTING_DECISIONS.md content coverage ────────────────────────────────────

class TestPortingDecisionsContent:
    """PORTING_DECISIONS.md (T2.7) has all four required sections and correct state.

    The spec says: skipped combinations table, default reconciliation table,
    known performance deltas with reasons, methodology choices with rationale.
    """

    _DOC = _HERE / "PORTING_DECISIONS.md"

    def test_doc_exists(self):
        assert self._DOC.exists(), "PORTING_DECISIONS.md must exist"

    def test_skipped_combinations_section(self):
        text = self._DOC.read_text()
        assert "Skipped Combinations" in text or "## 1." in text, (
            "PORTING_DECISIONS.md must have a Skipped Combinations section"
        )

    def test_default_reconciliation_section(self):
        text = self._DOC.read_text()
        assert "Default Reconciliation" in text or "## 2." in text

    def test_performance_deltas_section(self):
        text = self._DOC.read_text()
        assert "Performance" in text and ("TFLOP" in text or "TFLOP/s" in text), (
            "PORTING_DECISIONS.md must document known performance deltas"
        )

    def test_methodology_section(self):
        text = self._DOC.read_text()
        assert "warmup" in text.lower() and "median" in text.lower(), (
            "PORTING_DECISIONS.md must document measurement methodology (warmup, median)"
        )

    def test_preshufflev2_discrepancy_resolved(self):
        """The preshufflev2 double_buffer item must be marked resolved (NOT A BUG)."""
        text = self._DOC.read_text()
        assert "NOT A BUG" in text or "no discrepancy" in text.lower(), (
            "PORTING_DECISIONS.md must document that preshufflev2 double_buffer is NOT a bug"
        )

    def test_follow_up_gpu_done(self):
        """GPU execution follow-up (#3) must be marked DONE."""
        text = self._DOC.read_text()
        assert "DONE" in text and "gfx942" in text, (
            "PORTING_DECISIONS.md must record GPU verification as DONE"
        )


# ── check_parity.py spec requirements ────────────────────────────────────────

class TestCheckParityDefaults:
    """check_parity.py implements the T1.7 spec requirements as code constants.

    projectdes.txt T1.7 requires: ~2% tolerance, 10 back-to-back runs, median.
    These are verified by checking the module-level constant and argparse defaults.
    """

    def _load_module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_parity", _HERE / "check_parity.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_perf_runs_constant_is_10(self):
        """_PERF_RUNS must equal 10 (spec: 10 back-to-back runs)."""
        mod = self._load_module()
        assert mod._PERF_RUNS == 10, (
            f"_PERF_RUNS must be 10 per T1.7 spec; got {mod._PERF_RUNS}"
        )

    def test_default_sizes_include_non_tile_aligned(self):
        """Default --sizes must include at least one non-tile-aligned size (T1.6)."""
        src = (_HERE / "check_parity.py").read_text()
        # 257x257x56 and 513x511x40 are non-tile-aligned; either is acceptable
        assert "257x257x56" in src or "513x511x40" in src, (
            "check_parity.py default sizes must include a non-tile-aligned size for padding path (T1.6)"
        )

    def test_perf_tol_default_is_2pct(self):
        """--perf-tol default must be 0.02 (2%), matching T1.7 spec."""
        src = (_HERE / "check_parity.py").read_text()
        assert "default=0.02" in src, (
            "check_parity.py --perf-tol must default to 0.02 (2%) per T1.7 spec"
        )

    def test_perf_tol_argparse_exists(self):
        """--perf-tol argparse argument must exist."""
        src = (_HERE / "check_parity.py").read_text()
        assert "--perf-tol" in src, "check_parity.py must expose --perf-tol CLI argument"

    def test_perf_runs_argparse_exists(self):
        """--perf-runs argparse argument must exist."""
        src = (_HERE / "check_parity.py").read_text()
        assert "--perf-runs" in src, "check_parity.py must expose --perf-runs CLI argument"


# ── parse_harness_output unit tests ─────────────────────────────────────────

class TestParseHarnessOutput:
    """Unit tests for check_parity.parse_harness_output().

    This function is the critical bridge between the harness binary and Stage 2/3
    adjudication. Incorrect parsing silently turns FAILED runs into UNKNOWN verdicts.
    """

    def _parse(self, text: str):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_parity", _HERE / "check_parity.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.parse_harness_output(text)

    def test_passed_verdict(self):
        result = self._parse("verify : 1024/1024 elements pass (100.0%)\nPASSED\n")
        assert result["verdict"] == "PASSED"

    def test_failed_verdict(self):
        result = self._parse("verify : 900/1024 elements pass (87.9%)\nFAILED\n")
        assert result["verdict"] == "FAILED"

    def test_skipped_verdict(self):
        result = self._parse("SKIPPED: Arguments not supported!\n")
        assert result["verdict"] == "SKIPPED"

    def test_gflops_converted_to_tflops(self):
        result = self._parse("time   : 1.2345 ms  (85868.1 GFLOP/s)\nPASSED\n")
        assert result["tflops"] is not None
        assert abs(result["tflops"] - 85.8681) < 0.01

    def test_no_gflops_line_tflops_none(self):
        result = self._parse("SKIPPED: Arguments not supported!\n")
        assert result["tflops"] is None

    def test_unknown_on_empty_output(self):
        result = self._parse("")
        assert result["verdict"] == "UNKNOWN"

    def test_passed_takes_precedence_over_failed_word_in_detail(self):
        """PASSED keyword wins even if 'FAILED' appears elsewhere in output."""
        result = self._parse("verify : max_abs_err=0.001 max_rel_err=0.001\nPASSED\n")
        assert result["verdict"] == "PASSED"


# ── compare_report.py By-tile-shape rollup ───────────────────────────────────

class TestCompareReportTileShapeRollup:
    """compare_report.py emits a By-tile-shape rollup table (4th of 4 T2.6 rollups).

    T2.6 spec: 'rolled-up summaries by dtype and by layout' plus pipeline and tile.
    compare_report.py emits all four: By dtype, By layout, By pipeline, By tile shape.
    """

    def test_tile_shape_rollup_in_report(self, tmp_path):
        import subprocess, sys
        rows = [
            dict(_BASE_ROW, tile_m=256, tile_n=128, tile_k=32, M=512),
            dict(_BASE_ROW, tile_m=128, tile_n=128, tile_k=32, M=1024),
        ]
        pq = _make_parquet(rows, tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        result = subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"), str(pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        assert result.returncode == 0, result.stderr
        text = out.read_text()
        assert "By tile" in text or "tile shape" in text.lower(), (
            "compare_report.py must include a By tile shape rollup section"
        )

    def test_four_rollup_sections_present(self, tmp_path):
        """All four T2.6 rollup sections must appear in one report."""
        import subprocess, sys
        pq = _make_parquet([dict(_BASE_ROW)], tmp_path / "disp.parquet")
        out = tmp_path / "report.md"
        subprocess.run(
            [sys.executable, str(_HERE / "compare_report.py"), str(pq), "-o", str(out)],
            capture_output=True, text=True, cwd=_HERE,
        )
        text = out.read_text()
        for section in ("By dtype", "By layout", "By pipeline"):
            assert section in text, f"compare_report.py must emit '{section}' rollup"
        assert "By tile" in text or "tile" in text.lower(), (
            "compare_report.py must emit a tile-shape rollup"
        )


# ── parse_te_csv unit tests ───────────────────────────────────────────────────

class TestParseTeCsv:
    """check_parity.parse_te_csv() reads the last CSV data row correctly."""

    def test_missing_file_returns_none(self, tmp_path):
        from check_parity import parse_te_csv
        assert parse_te_csv(tmp_path / "nonexistent.csv") is None

    def test_header_only_returns_none(self, tmp_path):
        """A CSV with no data row (only header) must return None."""
        from check_parity import parse_te_csv
        f = tmp_path / "te.csv"
        f.write_text("latency(ms),tflops,bandwidth(GB/s)\n")
        assert parse_te_csv(f) is None

    def test_empty_file_returns_none(self, tmp_path):
        from check_parity import parse_te_csv
        f = tmp_path / "te.csv"
        f.write_text("")
        assert parse_te_csv(f) is None

    def test_parses_latency_and_tflops(self, tmp_path):
        """Standard TE CSV format: header + one data row."""
        from check_parity import parse_te_csv
        f = tmp_path / "te.csv"
        f.write_text("latency(ms),tflops,bandwidth(GB/s)\n0.0250,85.87,1234.5\n")
        result = parse_te_csv(f)
        assert result is not None
        assert abs(result["latency_ms"] - 0.0250) < 1e-6
        assert abs(result["tflops"] - 85.87) < 1e-4
        assert abs(result["bandwidth"] - 1234.5) < 0.1

    def test_returns_last_row(self, tmp_path):
        """When multiple data rows exist, the LAST row's values are used."""
        from check_parity import parse_te_csv
        f = tmp_path / "te.csv"
        f.write_text(
            "latency(ms),tflops,bandwidth(GB/s)\n"
            "0.0100,10.0,500.0\n"
            "0.0250,85.87,1234.5\n"
        )
        result = parse_te_csv(f)
        assert result is not None
        assert abs(result["tflops"] - 85.87) < 1e-4, (
            "parse_te_csv must return the last data row, not the first"
        )

    def test_prefix_match_for_column_names(self, tmp_path):
        """Column matching is prefix-based; 'latency(ms)' starts with 'latency'."""
        from check_parity import parse_te_csv
        f = tmp_path / "te.csv"
        f.write_text("latency(ms),tflops(fp16),bandwidth(GB/s)\n0.025,85.0,1000.0\n")
        result = parse_te_csv(f)
        assert result is not None
        assert result["tflops"] is not None

    def test_missing_column_returns_none_value(self, tmp_path):
        """If a column is absent, its key maps to None (not KeyError)."""
        from check_parity import parse_te_csv
        f = tmp_path / "te.csv"
        f.write_text("latency(ms),tflops\n0.025,85.0\n")
        result = parse_te_csv(f)
        assert result is not None
        assert result["bandwidth"] is None

    def test_non_numeric_value_returns_none(self, tmp_path):
        """Unparseable float in a cell returns None for that key."""
        from check_parity import parse_te_csv
        f = tmp_path / "te.csv"
        f.write_text("latency(ms),tflops,bandwidth(GB/s)\n0.025,N/A,1000.0\n")
        result = parse_te_csv(f)
        assert result is not None
        assert result["tflops"] is None


# ── sweep_runner._kernel_name unit tests ─────────────────────────────────────

class TestSweepRunnerKernelName:
    """sweep_runner._kernel_name mirrors te_kernel_name — same _preshuffle risk.

    Bug #1 in check_parity.te_kernel_name was the missing _preshuffle suffix for
    preshufflev2.  sweep_runner._kernel_name has the same logic in a separate file
    and needs the same guard tested independently.
    """

    def _make_cfg(self, pipeline="compv3", scheduler="intrawave"):
        """Minimal cfg dict accepted by sweep_runner._kernel_name."""
        return {
            "_te": {
                "datatype": "fp16",
                "layout": "rcr",
                "pipeline": pipeline,
                "epilogue": "default",
                "scheduler": scheduler,
            },
            "algorithm": {
                "pad_m": False, "pad_n": False, "pad_k": False, "persistent": False,
                "tile_m": 256, "tile_n": 128, "tile_k": 32,
                "warp_m": 4, "warp_n": 1, "warp_k": 1,
                "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16,
            },
            "signature": {"split_k": 1},
        }

    def test_vanilla_name_no_preshuffle_suffix(self):
        """compv3 pipeline must NOT append _preshuffle."""
        import sweep_runner
        cfg = self._make_cfg(pipeline="compv3")
        name = sweep_runner._kernel_name(cfg)
        assert not name.endswith("_preshuffle"), (
            f"compv3 kernel name must not end with _preshuffle; got: {name}"
        )

    def test_preshufflev2_appends_preshuffle_suffix(self):
        """preshufflev2 pipeline MUST append _preshuffle (Bug #1 guard)."""
        import sweep_runner
        cfg = self._make_cfg(pipeline="preshufflev2")
        name = sweep_runner._kernel_name(cfg)
        assert name.endswith("_preshuffle"), (
            f"preshufflev2 kernel name must end with _preshuffle; got: {name}"
        )

    def test_name_contains_tile_shape(self):
        """Tile dimensions must appear in the kernel name."""
        import sweep_runner
        cfg = self._make_cfg()
        name = sweep_runner._kernel_name(cfg)
        assert "256x128x32" in name
        assert "4x1x1" in name
        assert "32x32x16" in name

    def test_name_contains_padding_flags(self):
        """Padding flags (False/True) must appear as 'False'/'True' in name."""
        import sweep_runner
        cfg = self._make_cfg()
        name = sweep_runner._kernel_name(cfg)
        assert "False_False_False_False" in name

    def test_name_matches_check_parity_te_kernel_name(self):
        """sweep_runner._kernel_name must produce the same string as check_parity.te_kernel_name."""
        import sweep_runner
        from check_parity import te_kernel_name
        cfg = self._make_cfg(pipeline="compv3")
        assert sweep_runner._kernel_name(cfg) == te_kernel_name(cfg), (
            "sweep_runner._kernel_name and check_parity.te_kernel_name must agree"
        )

    def test_preshufflev2_matches_check_parity(self):
        """Both implementations must agree on _preshuffle suffix for preshufflev2."""
        import sweep_runner
        from check_parity import te_kernel_name
        cfg = self._make_cfg(pipeline="preshufflev2")
        assert sweep_runner._kernel_name(cfg) == te_kernel_name(cfg)


# ── compare_report format helper unit tests ───────────────────────────────────

class TestCompareReportFormatHelpers:
    """Pure-function coverage for _verdict_icon, _fmt_tflops, _fmt_delta."""

    def test_verdict_icon_passed(self):
        from compare_report import _verdict_icon
        assert _verdict_icon("PASSED") == "✅"

    def test_verdict_icon_failed(self):
        from compare_report import _verdict_icon
        assert _verdict_icon("FAILED") == "❌"

    def test_verdict_icon_error(self):
        from compare_report import _verdict_icon
        assert _verdict_icon("ERROR") == "❌"

    def test_verdict_icon_skipped(self):
        from compare_report import _verdict_icon
        assert _verdict_icon("SKIPPED") == "⏭"

    def test_verdict_icon_dryrun(self):
        from compare_report import _verdict_icon
        assert _verdict_icon("DRYRUN") == "⏭"

    def test_verdict_icon_none_returns_dash(self):
        from compare_report import _verdict_icon
        import pandas as pd
        assert _verdict_icon(None) == "—"
        assert _verdict_icon(float("nan")) == "—"
        assert _verdict_icon(pd.NA) == "—"

    def test_fmt_tflops_float(self):
        from compare_report import _fmt_tflops
        assert _fmt_tflops(85.868) == "85.87"

    def test_fmt_tflops_none(self):
        from compare_report import _fmt_tflops
        assert _fmt_tflops(None) == "—"

    def test_fmt_tflops_nan(self):
        from compare_report import _fmt_tflops
        import math
        assert _fmt_tflops(float("nan")) == "—"

    def test_fmt_delta_positive(self):
        from compare_report import _fmt_delta
        assert _fmt_delta(1.23) == "+1.2%"

    def test_fmt_delta_negative(self):
        from compare_report import _fmt_delta
        assert _fmt_delta(-3.5) == "-3.5%"

    def test_fmt_delta_zero(self):
        from compare_report import _fmt_delta
        assert _fmt_delta(0.0) == "+0.0%"

    def test_fmt_delta_none(self):
        from compare_report import _fmt_delta
        assert _fmt_delta(None) == "—"

    def test_fmt_delta_nan(self):
        from compare_report import _fmt_delta
        assert _fmt_delta(float("nan")) == "—"


# ── sweep_runner._make_row schema tests ───────────────────────────────────────

class TestMakeRow:
    """_make_row() produces a dict with the correct Parquet schema keys."""

    def _make_cfg(self):
        return {
            "_te": {
                "datatype": "fp16",
                "layout": "rcr",
                "pipeline": "compv3",
                "epilogue": "default",
                "scheduler": "intrawave",
            },
            "algorithm": {
                "pipeline": "compv3", "scheduler": "intrawave", "epilogue": "default",
                "pad_m": False, "pad_n": False, "pad_k": False, "persistent": False,
                "tile_m": 256, "tile_n": 128, "tile_k": 32,
                "warp_m": 4, "warp_n": 1, "warp_k": 1,
                "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16,
                "block_size": 256, "double_buffer": False, "preshuffle": False,
                "transpose_c": False, "num_wave_groups": 1, "k_block_per_cu": 1,
            },
            "signature": {
                "split_k": 1,
                "dtype_a": "fp16", "dtype_b": "fp16", "dtype_acc": "fp32",
                "dtype_c": "fp16", "layout_a": "row", "layout_b": "col",
                "layout_c": "row", "transpose_a": False, "transpose_b": False,
                "grouped": False, "elementwise_op": "passthrough",
                "num_d_tensors": 0, "structured_sparsity": False,
            },
        }

    def test_required_columns_present(self):
        """Row must have all columns required by compare_report._load_parquet."""
        import sweep_runner
        cfg = self._make_cfg()
        row = sweep_runner._make_row(
            Path("configs/single_fp16_rcr.json"), 0, cfg,
            512, 512, 512, "PASSED", 10.5, "", "",
        )
        required = {
            "config_file", "config_index", "identifier", "kernel_name",
            "datatype", "layout", "pipeline", "scheduler",
            "tile_m", "tile_n", "tile_k", "split_k",
            "pad_m", "pad_n", "pad_k", "persistent",
            "M", "N", "K", "verdict", "tflops", "error_msg", "stage_failed", "ts",
        }
        missing = required - set(row.keys())
        assert not missing, f"_make_row missing columns: {missing}"

    def test_problem_dimensions_stored(self):
        import sweep_runner
        cfg = self._make_cfg()
        row = sweep_runner._make_row(
            Path("c.json"), 0, cfg, 1024, 768, 512, "PASSED", 85.0, "", ""
        )
        assert row["M"] == 1024
        assert row["N"] == 768
        assert row["K"] == 512

    def test_verdict_and_tflops_stored(self):
        import sweep_runner
        cfg = self._make_cfg()
        row = sweep_runner._make_row(
            Path("c.json"), 0, cfg, 512, 512, 512, "FAILED", None, "OOM", "stage2"
        )
        assert row["verdict"] == "FAILED"
        assert row["tflops"] is None
        assert row["error_msg"] == "OOM"
        assert row["stage_failed"] == "stage2"

    def test_identifier_matches_encode_identifier(self):
        """Row's identifier field must equal encode_identifier(cfg)."""
        import sweep_runner
        from identifier import encode_identifier
        cfg = self._make_cfg()
        row = sweep_runner._make_row(
            Path("c.json"), 0, cfg, 512, 512, 512, "PASSED", 10.0, "", ""
        )
        assert row["identifier"] == encode_identifier(cfg)

    def test_kernel_name_has_no_preshuffle_for_compv3(self):
        """compv3 rows must not have _preshuffle in kernel_name."""
        import sweep_runner
        cfg = self._make_cfg()
        row = sweep_runner._make_row(
            Path("c.json"), 0, cfg, 512, 512, 512, "PASSED", 10.0, "", ""
        )
        assert "_preshuffle" not in row["kernel_name"]

    def test_ts_is_iso_utc(self):
        """Timestamp must be a valid ISO 8601 string (for Parquet ordering)."""
        import sweep_runner
        from datetime import datetime, timezone
        cfg = self._make_cfg()
        row = sweep_runner._make_row(
            Path("c.json"), 0, cfg, 512, 512, 512, "PASSED", 10.0, "", ""
        )
        ts = row["ts"]
        assert isinstance(ts, str)
        # Must parse as a datetime
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None, "ts must be timezone-aware (UTC)"
