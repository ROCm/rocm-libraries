# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko import search


def test_normalize_compute_type_prefixing() -> None:
    assert search._normalize_compute_type("f32_r") == "c_f32_r"
    assert search._normalize_compute_type("c_f32_r") == "c_f32_r"


def test_load_processed_df_missing_file_returns_empty(tmp_path: Path) -> None:
    out = search._load_processed_df(tmp_path / "missing.csv")
    assert out.empty


def test_load_processed_df_invalid_schema_deletes_file(tmp_path: Path) -> None:
    processed = tmp_path / "processed.csv"
    pd.DataFrame({"bad": [1]}).to_csv(processed, index=False)

    out = search._load_processed_df(processed)
    assert out.empty
    assert not processed.exists()


def test_configure_rejects_empty_df() -> None:
    with pytest.raises(ValueError, match="No GEMM operations"):
        search.configure(pd.DataFrame())


def test_configure_sets_skip_ratio_from_latency() -> None:
    df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 32,
                "n": 32,
                "k": 32,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "scaleA": 1,
                "scaleB": 1,
                "us": 10.0,
            },
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 4096,
                "n": 4096,
                "k": 2048,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "scaleA": 1,
                "scaleB": 1,
                "us": 30.0,
            },
        ]
    )

    out = search.configure(df)
    assert out[0]["skip_slow_solution_ratio"] == 0.3
    assert out[1]["skip_slow_solution_ratio"] == 0.6


def test_run_raises_for_missing_hipblaslt_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        search.run(tmp_path / "nohip", [{"M": 1}], tmp_path / "out", devices=[0])


def test_run_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    with pytest.raises(ValueError, match="max_chunk_size"):
        search.run(
            hip,
            [
                {
                    "transA": "N",
                    "transB": "N",
                    "batch_count": 1,
                    "M": 32,
                    "N": 32,
                    "K": 32,
                    "a_type": "f16_r",
                    "b_type": "f16_r",
                    "c_type": "f16_r",
                    "d_type": "f16_r",
                    "compute_type": "f32_r",
                }
            ],
            tmp_path / "out",
            devices=[0],
            max_chunk_size=0,
        )


def test_run_raises_for_unbuilt_custom_library_before_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()

    cfg = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 32,
        "N": 32,
        "K": 32,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
    }

    custom_lib = tmp_path / "custom"
    custom_lib.mkdir()

    def _runner_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("Runner should not be launched for unbuilt custom_lib_dir")

    monkeypatch.setattr(search, "Runner", _runner_should_not_be_called)

    with pytest.raises(ValueError, match="is not built"):
        search.run(
            hip,
            [cfg],
            tmp_path / "out",
            custom_lib_dir=custom_lib,
            devices=[0],
        )


def test_run_empty_configs_returns_empty_df(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    out = search.run(hip, [], tmp_path / "out", devices=[0])
    assert out.empty


def test_run_uses_existing_data_when_all_configs_processed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify complete GEMM keys, including scale fields, skip processed configs.

    Args:
        monkeypatch: Pytest fixture used to prevent benchmark runner creation.
        tmp_path: Temporary directory used for search output files.

    Returns:
        None.

    Raises:
        AssertionError: If the processed config is incorrectly retried.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"
    data_dir = outdir / "data"
    data_dir.mkdir(parents=True)

    def _runner_should_not_be_called(*_args: object, **_kwargs: object) -> None:
        """Fail if a processed configuration reaches the benchmark runner.

        Args:
            *_args: Positional runner constructor arguments.
            **_kwargs: Keyword runner constructor arguments.

        Returns:
            None.

        Raises:
            AssertionError: Always, because no retry should be attempted.
        """
        raise AssertionError("Runner should not retry an already-processed config")

    monkeypatch.setattr(search, "Runner", _runner_should_not_be_called)

    cfg = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 32,
        "N": 32,
        "K": 32,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
        "scaleA": 1,
        "scaleB": 1,
    }

    processed = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 32,
                "n": 32,
                "k": 32,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
                "scaleA": 1,
                "scaleB": 1,
            }
        ]
    )
    processed.to_csv(outdir / "processed.csv", index=False)

    pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 32,
                "n": 32,
                "k": 32,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
                "scaleA": 1,
                "scaleB": 1,
                "us": 1.0,
            }
        ]
    ).to_csv(data_dir / "existing.csv", index=False)

    out = search.run(hip, [cfg], outdir, devices=[0])
    assert len(out) == 1
    assert out.iloc[0]["m"] == 32


def test_run_persists_winners_with_stubbed_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    cfg = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 64,
        "N": 64,
        "K": 64,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
        "scaleA": 1,
        "scaleB": 1,
    }

    monkeypatch.setattr(search.bench.log, "dump", lambda data, p: yaml.safe_dump(data, Path(p).open("w"), sort_keys=False))

    def _fake_bench_run(_hip, _bench_file, log_file, **_kwargs):
        Path(log_file).write_text("Winner\n")
        return pd.DataFrame()

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    parse_df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 64,
                "n": 64,
                "k": 64,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "c_f32_r",
                "scaleA": 1,
                "scaleB": 1,
                "us": 2.0,
            }
        ]
    )
    monkeypatch.setattr(search.bench.utils, "parse_benchmark_output", lambda _p: parse_df)

    class _PassRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self._items = items
            self._worker_impl = worker_impl
            self._devices = devices

        def __call__(self, _workdir):
            for item in self._items:
                w = self._worker_impl(item, self._devices[0], 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return self._items

    monkeypatch.setattr(search, "Runner", _PassRunner)

    out = search.run(hip, [cfg], outdir, devices=[0])
    assert len(out) == 1
    assert (outdir / "processed.csv").is_file()


def test_run_emits_failed_gemms_after_retry_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    cfg = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 64,
        "N": 64,
        "K": 64,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
    }

    monkeypatch.setattr(search.bench.log, "dump", lambda data, p: yaml.safe_dump(data, Path(p).open("w"), sort_keys=False))

    def _fake_bench_run(_hip, _bench_file, _log_file, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    class _FailRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            self._items = items
            self._worker_impl = worker_impl
            self._devices = devices

        def __call__(self, _workdir):
            for item in self._items:
                w = self._worker_impl(item, self._devices[0], 0, None, None)
                w.setup()
                w.run()
                w.teardown()
            return self._items

    monkeypatch.setattr(search, "Runner", _FailRunner)

    out = search.run(hip, [cfg], outdir, devices=[0])
    assert out.empty
    assert (outdir / "failed_gemms.yaml").is_file()


def _gemm_config(**overrides: object) -> dict:
    """Return one dense-search GEMM configuration.

    Args:
        **overrides: Configuration fields that replace the defaults.

    Returns:
        A configuration dictionary accepted by ``search.run``.

    Raises:
        None.
    """
    config = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "M": 32,
        "N": 32,
        "K": 32,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
        "scaleA": 1,
        "scaleB": 1,
    }
    config.update(overrides)
    return config


def _processed_row(config: dict) -> dict:
    """Convert a search configuration into a processed.csv row.

    Args:
        config: Dense-search configuration using uppercase problem sizes.

    Returns:
        A row whose columns match ``GEMM_FIELDS``, with a normalized compute type.

    Raises:
        KeyError: If config is missing a GEMM field.
    """
    compute_type = str(config["compute_type"])
    if not compute_type.startswith("c_"):
        compute_type = f"c_{compute_type}"
    return {
        "transA": config["transA"],
        "transB": config["transB"],
        "batch_count": config["batch_count"],
        "m": config["M"],
        "n": config["N"],
        "k": config["K"],
        "a_type": config["a_type"],
        "b_type": config["b_type"],
        "c_type": config["c_type"],
        "d_type": config["d_type"],
        "compute_type": compute_type,
        "scaleA": config["scaleA"],
        "scaleB": config["scaleB"],
    }


class _InlineRunner:
    """Execute search workers immediately in the calling thread.

    Args:
        items: Work items supplied by ``search.run``.
        worker_impl: Worker class constructed for each item.
        devices: Device ids available to the workers.
        n_slots: Slot count accepted for compatibility with ``Runner``.

    Raises:
        None.
    """

    def __init__(
        self, items: list, worker_impl: type, devices: list[int], n_slots: int = 1
    ) -> None:
        """Store the runner inputs needed to execute workers inline.

        Args:
            items: Work items supplied by ``search.run``.
            worker_impl: Worker class constructed for each item.
            devices: Device ids available to the workers.
            n_slots: Slot count accepted for compatibility with ``Runner``.

        Returns:
            None.

        Raises:
            None.
        """
        self._items = items
        self._worker_impl = worker_impl
        self._devices = devices
        self._n_slots = n_slots

    def __call__(self, _workdir: Path) -> list:
        """Run setup, work, and teardown for every item.

        Args:
            _workdir: Output directory accepted for compatibility with ``Runner``.

        Returns:
            The items supplied when the runner was constructed.

        Raises:
            Exception: Any exception raised by a worker.
        """
        for item in self._items:
            worker = self._worker_impl(item, self._devices[0], 0, None, None)
            worker.setup()
            worker.run()
            worker.teardown()
        return self._items


def test_configure_sets_skip_ratio_from_problem_size() -> None:
    """Use problem size when the input has no measured latency.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If either size-based skip ratio is wrong.
    """
    df = pd.DataFrame(
        [
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 32,
                "n": 32,
                "k": 32,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "scaleA": 1,
                "scaleB": 1,
            },
            {
                "transA": "N",
                "transB": "N",
                "batch_count": 1,
                "m": 4096,
                "n": 4096,
                "k": 2048,
                "a_type": "f16_r",
                "b_type": "f16_r",
                "c_type": "f16_r",
                "d_type": "f16_r",
                "compute_type": "f32_r",
                "scaleA": 1,
                "scaleB": 1,
            },
        ]
    )

    out = search.configure(df)
    assert out[0]["skip_slow_solution_ratio"] == 0.3
    assert out[1]["skip_slow_solution_ratio"] == 0.6


def test_run_defaults_to_eight_devices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use devices 0 through 7 when the caller does not select devices.

    Args:
        monkeypatch: Pytest fixture used to observe device parsing.
        tmp_path: Temporary directory used for the hipBLASLt and output paths.

    Returns:
        None.

    Raises:
        AssertionError: If the default device list is not 0 through 7.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    seen: dict[str, list[int]] = {}

    def _parse_devices(devices: list[int]) -> list[int]:
        """Record the device list selected by ``search.run``.

        Args:
            devices: Device ids passed to ``parse_devices``.

        Returns:
            The first device, which is sufficient for an empty config list.

        Raises:
            None.
        """
        seen["devices"] = list(devices)
        return [devices[0]]

    monkeypatch.setattr(search, "parse_devices", _parse_devices)

    out = search.run(hip, [], tmp_path / "out")
    assert out.empty
    assert seen["devices"] == list(range(8))


def test_run_accepts_built_custom_library(tmp_path: Path) -> None:
    """Continue when the custom library contains a compiled Tensile artifact.

    Args:
        tmp_path: Temporary directory used for the library and output paths.

    Returns:
        None.

    Raises:
        ValueError: If the built-library check incorrectly rejects the artifact.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    artifact = tmp_path / "custom" / "library" / "gfx942" / "TensileLibrary_lazy_gfx942.dat"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("built")

    out = search.run(hip, [], tmp_path / "out", custom_lib_dir=tmp_path / "custom", devices=[0])
    assert out.empty


def test_run_returns_empty_when_processed_results_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Return no results when every config was processed but no CSV remains.

    Args:
        monkeypatch: Pytest fixture used to prevent benchmark runner creation.
        tmp_path: Temporary directory used for search output files.

    Returns:
        None.

    Raises:
        AssertionError: If the processed config is retried or a result is returned.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"
    outdir.mkdir()
    config = _gemm_config()
    pd.DataFrame([_processed_row(config)]).to_csv(outdir / "processed.csv", index=False)

    def _runner_should_not_be_called(*_args: object, **_kwargs: object) -> None:
        """Fail if a fully processed configuration reaches the benchmark runner.

        Args:
            *_args: Positional runner constructor arguments.
            **_kwargs: Keyword runner constructor arguments.

        Returns:
            None.

        Raises:
            AssertionError: Always, because the config should not be retried.
        """
        raise AssertionError("Runner should not retry an already-processed config")

    monkeypatch.setattr(search, "Runner", _runner_should_not_be_called)

    out = search.run(hip, [config], outdir, devices=[0])
    assert out.empty


def test_run_treats_missing_winner_log_as_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail a chunk whose benchmark exits without writing a winner log.

    Args:
        monkeypatch: Pytest fixture used to stub benchmarking and the runner.
        tmp_path: Temporary directory used for search output files.

    Returns:
        None.

    Raises:
        AssertionError: If the missing log is treated as a successful search.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"
    previous_level = search.logger.level
    search.logger.setLevel(logging.DEBUG)
    monkeypatch.setattr(
        search.bench.log,
        "dump",
        lambda data, path: yaml.safe_dump(data, Path(path).open("w"), sort_keys=False),
    )
    monkeypatch.setattr(search.bench, "run", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(search, "Runner", _InlineRunner)

    try:
        out = search.run(hip, [_gemm_config(), _gemm_config(M=64)], outdir, devices=[0])
    finally:
        search.logger.setLevel(previous_level)

    assert out.empty
    assert (outdir / "failed_gemms.yaml").is_file()
    assert search.logger.level == previous_level


def test_run_warns_when_persisting_winners_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Log a warning when a winning benchmark cannot be saved.

    Args:
        monkeypatch: Pytest fixture used to stub benchmarking and the runner.
        tmp_path: Temporary directory used for search output files.
        caplog: Pytest fixture that captures GEKO log records.

    Returns:
        None.

    Raises:
        AssertionError: If the persistence failure is not reported.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    def _write_winner_log(_hip: Path, _bench_file: Path, log_file: Path, **_kwargs: object) -> pd.DataFrame:
        """Write a winner marker without producing a parsed result.

        Args:
            _hip: Unused hipBLASLt installation path.
            _bench_file: Unused benchmark configuration path.
            log_file: Log path expected by the search worker.
            **_kwargs: Unused benchmark keyword arguments.

        Returns:
            An empty DataFrame.

        Raises:
            None.
        """
        Path(log_file).write_text("Winner\n")
        return pd.DataFrame()

    def _fail_to_parse(_path: Path) -> pd.DataFrame:
        """Raise the persistence error exercised by this test.

        Args:
            _path: Unused benchmark log path.

        Returns:
            This function does not return.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("parse failed")

    monkeypatch.setattr(
        search.bench.log,
        "dump",
        lambda data, path: yaml.safe_dump(data, Path(path).open("w"), sort_keys=False),
    )
    monkeypatch.setattr(search.bench, "run", _write_winner_log)
    monkeypatch.setattr(search.bench.utils, "parse_benchmark_output", _fail_to_parse)
    monkeypatch.setattr(search, "Runner", _InlineRunner)

    with caplog.at_level(logging.WARNING, logger="GEKO"):
        out = search.run(hip, [_gemm_config()], outdir, devices=[0])

    assert out.empty
    assert "Failed to persist" in caplog.text
