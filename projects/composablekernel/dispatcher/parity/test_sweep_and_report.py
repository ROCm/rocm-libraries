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
