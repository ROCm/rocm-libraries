# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections import defaultdict
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


def _adaptive_like_row() -> dict:
    # A representative bf16 TN GEMM row for adaptive-timing assertions.
    return {
        "transA": "T",
        "transB": "N",
        "batch_count": 1,
        "m": 9504,
        "n": 7248,
        "k": 368,
        "a_type": "bf16_r",
        "b_type": "bf16_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "compute_type": "f32_r",
        "scaleA": 0,
        "scaleB": 0,
    }


def test_configure_fast_profile_timing() -> None:
    """profile="fast" emits the "fast" adaptive-timing config.

    measure_time=5 (ms) is the measurement floor; warmup_time, max_measure_time
    and sample_time are derived from it via the "fast" profile's
    warmup_mult/ceil_mult/min_samples (0.4x, 3x, 1/10).
    """
    out = search.configure(pd.DataFrame([_adaptive_like_row()]), profile="fast")
    cfg = out[0]

    assert cfg["adaptive"] is True
    assert cfg["measure_time"] == 5.0
    assert cfg["warmup_time"] == 2.0
    assert cfg["max_measure_time"] == 15.0
    assert cfg["sample_time"] == 0.5
    assert cfg["noise_threshold"] == 0.01
    assert cfg["min_iters"] == 3
    assert cfg["max_iters"] == 10000
    assert cfg["stability_threshold"] == 0.0
    assert cfg["stability_window"] == 32
    assert cfg["stability_interval"] == 16


def test_configure_thorough_profile_timing() -> None:
    """The "thorough" profile's derived timing knobs track its own measure_time."""
    out = search.configure(pd.DataFrame([_adaptive_like_row()]), profile="thorough")
    cfg = out[0]

    assert cfg["measure_time"] == 20.0
    assert cfg["warmup_time"] == 40.0
    assert cfg["max_measure_time"] == 320.0
    assert cfg["sample_time"] == 2.0
    assert cfg["noise_threshold"] == 0.005
    assert cfg["stability_threshold"] == 0.03


def test_configure_defaults_to_thorough_profile() -> None:
    """Default configure() (no profile arg) matches the "thorough" profile."""
    out = search.configure(pd.DataFrame([_adaptive_like_row()]))
    cfg = out[0]

    assert cfg["measure_time"] == 20.0
    assert cfg["warmup_time"] == 40.0
    assert cfg["max_measure_time"] == 320.0
    assert cfg["sample_time"] == 2.0
    assert cfg["noise_threshold"] == 0.005
    assert cfg["stability_threshold"] == 0.03
    assert cfg["stability_window"] == 6
    assert cfg["stability_interval"] == 3


def _flat_profiles() -> list[dict]:
    flat = []
    for cfg in search.SEARCH_PROFILES.values():
        flat.extend([cfg["stage_a"], cfg["stage_b"]] if "stage_a" in cfg else [cfg])
    return flat


@pytest.mark.parametrize("cfg", _flat_profiles())
def test_profile_knobs_expand_to_expected_adaptive_fields(cfg: dict) -> None:
    """Every flat profile dict unpacks into adaptive_fields and yields the expected block."""
    fields = search.bench.log.adaptive_fields(**cfg)
    mt = cfg["measure_time"]

    assert fields == {
        "adaptive": True,
        "measure_time": mt,
        "warmup_time": cfg["warmup_mult"] * mt,
        "max_measure_time": cfg["ceil_mult"] * mt,
        "sample_time": mt / cfg["min_samples"],
        "noise_threshold": cfg["noise_threshold"],
        "min_iters": cfg["min_iters"],
        "max_iters": cfg["max_iters"],
        "stability_threshold": cfg["stability_threshold"],
        "stability_window": cfg["stability_window"],
        "stability_interval": cfg["stability_interval"],
    }


def test_configure_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown search profile"):
        search.configure(pd.DataFrame([_adaptive_like_row()]), profile="bogus")


def test_configure_balanced_profile_uses_stage_a() -> None:
    """The two-stage "balanced" profile's configure() output matches its "stage_a" (== "fast") knobs."""
    out = search.configure(pd.DataFrame([_adaptive_like_row()]), profile="balanced")
    cfg = out[0]

    assert cfg["measure_time"] == 5.0
    assert cfg["warmup_time"] == 2.0
    assert cfg["max_measure_time"] == 15.0
    assert cfg["sample_time"] == 0.5
    assert cfg["noise_threshold"] == 0.01
    assert cfg["stability_threshold"] == 0.0
    assert cfg["stability_window"] == 32
    assert cfg["stability_interval"] == 16


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


def test_run_rejects_unknown_profile(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    with pytest.raises(ValueError, match="Unknown search profile"):
        search.run(hip, [{"M": 1}], tmp_path / "out", devices=[0], profile="bogus")


def test_run_rejects_stage_b_profile_without_k(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    monkeypatch.setitem(
        search.SEARCH_PROFILES,
        "no_k",
        {"stage_a": search.SEARCH_PROFILES["fast"], "stage_b": search.SEARCH_PROFILES["thorough"]},
    )
    with pytest.raises(ValueError, match="positive 'k'"):
        search.run(hip, [{"M": 1}], tmp_path / "out", devices=[0], profile="no_k")


def test_run_two_stage_profile_reverifies_with_stage_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """profile="balanced" re-verifies stage_a's shortlisted top-k candidates
    with stage_b's own adaptive config, and the FINAL winner (persisted by
    the unmodified teardown()) comes exclusively from stage_b's own result.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    # Shortlist to the top 2 candidates to keep the test small.
    monkeypatch.setitem(search.SEARCH_PROFILES["balanced"], "k", 2)

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

    gemm_row = {
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
    }

    # stage_a (screen) sees 3 candidates; only the best 2 (by "us") are
    # shortlisted for stage_b's re-verify pass.
    stage_a_df = pd.DataFrame(
        [
            {**gemm_row, "us": 5.0, "solutionIdx": "30"},
            {**gemm_row, "us": 1.0, "solutionIdx": "10"},
            {**gemm_row, "us": 3.0, "solutionIdx": "20"},
        ]
    )
    # stage_b (re-verify) measures only the shortlisted candidates; its own
    # result (not stage_a's) must become the final persisted winner.
    stage_b_df = pd.DataFrame(
        [
            {**gemm_row, "us": 0.9, "solutionIdx": "10"},
            {**gemm_row, "us": 1.2, "solutionIdx": "20"},
        ]
    )

    dumped: list[list[dict]] = []
    monkeypatch.setattr(
        search.bench.log,
        "dump",
        lambda data, p: (dumped.append(data), yaml.safe_dump(data, Path(p).open("w"), sort_keys=False))[-1],
    )

    bench_run_calls: list[Path] = []

    def _fake_bench_run(_hip, _bench_file, log_file, **_kwargs):
        bench_run_calls.append(Path(log_file))
        if len(bench_run_calls) == 1:
            Path(log_file).write_text("Winner\n")
        else:
            Path(log_file).write_text("stage_b output\n")
        return pd.DataFrame()

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    def _fake_parse(p):
        p = Path(p)
        return stage_a_df if p.name.endswith("_stage_a.out") else stage_b_df

    monkeypatch.setattr(search.bench.utils, "parse_benchmark_output", _fake_parse)

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

    out = search.run(hip, [cfg], outdir, devices=[0], profile="balanced")

    # bench.run was called twice: stage_a (screen), then stage_b (re-verify).
    assert len(bench_run_calls) == 2
    # stage_a's own output is preserved under its own name...
    assert (outdir / "gemms_0_stage_a.out").is_file()
    # ...and the chunk's original log filename ends up holding stage_b's output.
    assert bench_run_calls[0] == bench_run_calls[1] == outdir / "gemms_0.out"

    # The second bench.log.dump() call built one algo_method=2 row per
    # shortlisted candidate (k=2), referencing the screened solution indices.
    stage_b_rows = dumped[1]
    assert len(stage_b_rows) == 2
    assert {r["solution_index"] for r in stage_b_rows} == {10, 20}
    assert all(r["algo_method"] == 2 for r in stage_b_rows)
    assert all("requested_solution_num" not in r for r in stage_b_rows)
    # stage_b rows carry stage_b's own (not stage_a's) adaptive timing.
    assert all(r["measure_time"] == search.SEARCH_PROFILES["thorough"]["measure_time"] for r in stage_b_rows)

    # The final persisted winner is stage_b's own best candidate (us=0.9,
    # solutionIdx=10), never compared against stage_a's.
    assert len(out) == 1
    assert out.iloc[0]["us"] == 0.9
    assert int(out.iloc[0]["solutionIdx"]) == 10


def test_run_two_stage_shortlist_sorts_dedupes_and_ranks_per_gemm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dedicated test for the stage_a -> stage_b shortlisting step itself:
    candidates are ranked by ascending "us" (not input order), a duplicate
    solutionIdx for the same GEMM is only counted once toward "k", and each
    GEMM in the chunk gets its own independent top-k ranking.
    """
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    monkeypatch.setitem(search.SEARCH_PROFILES["balanced"], "k", 2)

    def _cfg(k: int) -> dict:
        return {
            "transA": "N",
            "transB": "N",
            "batch_count": 1,
            "M": 64,
            "N": 64,
            "K": k,
            "a_type": "f16_r",
            "b_type": "f16_r",
            "c_type": "f16_r",
            "d_type": "f16_r",
            "compute_type": "f32_r",
            "scaleA": 1,
            "scaleB": 1,
        }

    def _gemm_row(k: int, us: float, sol: str) -> dict:
        return {
            "transA": "N",
            "transB": "N",
            "batch_count": 1,
            "m": 64,
            "n": 64,
            "k": k,
            "a_type": "f16_r",
            "b_type": "f16_r",
            "c_type": "f16_r",
            "d_type": "f16_r",
            "compute_type": "c_f32_r",
            "scaleA": 1,
            "scaleB": 1,
            "us": us,
            "solutionIdx": sol,
        }

    cfg1, cfg2 = _cfg(64), _cfg(128)

    # GEMM1 (K=64): presented out of order, with a duplicate solutionIdx "10"
    # row that must not count twice toward the k=2 shortlist. Sorted by us:
    # 10 (1.0), 10 (dup, skipped), 20 (3.0), 30 (5.0) -> shortlist = [10, 20].
    # GEMM2 (K=128): independent ranking, lowest two are 40 and 50 -> [40, 50].
    stage_a_df = pd.DataFrame(
        [
            _gemm_row(64, 5.0, "30"),
            _gemm_row(64, 1.0, "10"),
            _gemm_row(64, 1.0, "10"),  # duplicate candidate, must be deduped
            _gemm_row(64, 3.0, "20"),
            _gemm_row(128, 2.0, "50"),
            _gemm_row(128, 0.5, "40"),
            _gemm_row(128, 4.0, "60"),
        ]
    )
    stage_b_df = pd.DataFrame(
        [
            _gemm_row(64, 0.9, "10"),
            _gemm_row(64, 1.1, "20"),
            _gemm_row(128, 0.4, "40"),
            _gemm_row(128, 0.6, "50"),
        ]
    )

    dumped: list[list[dict]] = []
    monkeypatch.setattr(
        search.bench.log,
        "dump",
        lambda data, p: (dumped.append(data), yaml.safe_dump(data, Path(p).open("w"), sort_keys=False))[-1],
    )

    bench_run_calls: list[Path] = []

    def _fake_bench_run(_hip, _bench_file, log_file, **_kwargs):
        bench_run_calls.append(Path(log_file))
        if len(bench_run_calls) == 1:
            Path(log_file).write_text("Winner\nWinner\n")
        else:
            Path(log_file).write_text("stage_b output\n")
        return pd.DataFrame()

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    def _fake_parse(p):
        p = Path(p)
        return stage_a_df if p.name.endswith("_stage_a.out") else stage_b_df

    monkeypatch.setattr(search.bench.utils, "parse_benchmark_output", _fake_parse)

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

    # Explicit max_chunk_size: this test cares about both GEMMs landing in
    # the same chunk (so stage_a produces one shortlist covering both),
    # not about the default chunk-size heuristic.
    out = search.run(hip, [cfg1, cfg2], outdir, devices=[0], max_chunk_size=10, profile="balanced")

    stage_b_rows = dumped[1]
    # 2 GEMMs x k=2 shortlisted candidates each = 4 rows; the duplicate
    # solutionIdx "10" row for GEMM1 did not consume a second shortlist slot.
    assert len(stage_b_rows) == 4

    by_k = defaultdict(set)
    for row in stage_b_rows:
        by_k[row["K"]].add(row["solution_index"])

    assert by_k[64] == {10, 20}
    assert by_k[128] == {40, 50}

    # Final per-GEMM winners come from stage_b's own best candidate.
    assert len(out) == 2
    winners = {int(row["k"]): (row["us"], int(row["solutionIdx"])) for _, row in out.iterrows()}
    assert winners[64] == (0.9, 10)
    assert winners[128] == (0.4, 40)


def test_run_two_stage_retries_single_config_after_stage_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If stage_a fails for a chunk, its config is requeued into
    failed_configs and retried as a single-config item (see
    test_run_emits_failed_gemms_after_retry_failure for the all-fail case).
    When that retry runs under profile="balanced" it must still perform the
    FULL two-stage (stage_a then stage_b) dance and persist stage_b's own
    winner -- not just stop after stage_a, and not skip stage_b."""
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    monkeypatch.setitem(search.SEARCH_PROFILES["balanced"], "k", 2)

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
    gemm_row = {
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
    }

    # The retry's own stage_a screens 2 candidates; stage_b re-verifies both
    # (k=2) and its own best candidate (solutionIdx 10) is the final winner.
    stage_a_df = pd.DataFrame(
        [
            {**gemm_row, "us": 1.0, "solutionIdx": "10"},
            {**gemm_row, "us": 3.0, "solutionIdx": "20"},
        ]
    )
    stage_b_df = pd.DataFrame(
        [
            {**gemm_row, "us": 0.9, "solutionIdx": "10"},
            {**gemm_row, "us": 1.2, "solutionIdx": "20"},
        ]
    )

    monkeypatch.setattr(
        search.bench.log, "dump", lambda data, p: yaml.safe_dump(data, Path(p).open("w"), sort_keys=False)
    )

    bench_run_calls: list[Path] = []

    def _fake_bench_run(_hip, _bench_file, log_file, **_kwargs):
        bench_run_calls.append(Path(log_file))
        call_index = len(bench_run_calls)
        if call_index == 1:
            # Initial attempt's stage_a: zero "Winner" lines for 1 config -> fails.
            Path(log_file).write_text("no winner here\n")
        elif call_index == 2:
            # Retry attempt's own stage_a: passes.
            Path(log_file).write_text("Winner\n")
        else:
            # Retry attempt's own stage_b.
            Path(log_file).write_text("stage_b output\n")
        return pd.DataFrame()

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    def _fake_parse(p):
        p = Path(p)
        return stage_a_df if p.name.endswith("_stage_a.out") else stage_b_df

    monkeypatch.setattr(search.bench.utils, "parse_benchmark_output", _fake_parse)

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

    out = search.run(hip, [cfg], outdir, devices=[0], profile="balanced")

    # 1 failed stage_a attempt + the retry's own stage_a + stage_b = 3 calls.
    assert len(bench_run_calls) == 3
    # The retry redid stage_a from scratch and preserved its own raw output.
    assert (outdir / "gemms_0_stage_a.out").is_file()

    # Final result is stage_b's own best candidate, re-verified by the retry.
    assert len(out) == 1
    assert out.iloc[0]["us"] == 0.9
    assert int(out.iloc[0]["solutionIdx"]) == 10
    assert not (outdir / "failed_gemms.yaml").is_file()


def test_run_two_stage_retries_single_config_after_stage_b_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If stage_a passes but stage_b fails for a MULTI-config chunk, the
    chunk's own stage_a output survives on disk as "chunk_<i>_stage_a.out"
    (teardown only cleans up self.log_file/self.bench_file, not that
    renamed-away stage_a artifact). The single-config retry must redo BOTH
    stages from scratch under its own "gemms_<id>" naming, never colliding
    with or accidentally reusing that stale chunk-level file."""
    hip = tmp_path / "hip"
    hip.mkdir()
    outdir = tmp_path / "out"

    monkeypatch.setitem(search.SEARCH_PROFILES["balanced"], "k", 1)

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
    filler_cfg = {**cfg, "K": 32}

    def _row(k: int) -> dict:
        return {
            "transA": "N",
            "transB": "N",
            "batch_count": 1,
            "m": 64,
            "n": 64,
            "k": k,
            "a_type": "f16_r",
            "b_type": "f16_r",
            "c_type": "f16_r",
            "d_type": "f16_r",
            "compute_type": "c_f32_r",
            "scaleA": 1,
            "scaleB": 1,
        }

    # Per-retry-id dataframes keep each retried GEMM's shortlist independent
    # (mirroring what the real parsed stage_a/stage_b logs would contain).
    chunk_stage_a_df = pd.DataFrame(
        [
            {**_row(64), "us": 1.0, "solutionIdx": "10"},
            {**_row(32), "us": 1.0, "solutionIdx": "10"},
        ]
    )
    gemms0_stage_a_df = pd.DataFrame([{**_row(64), "us": 1.0, "solutionIdx": "10"}])
    gemms0_stage_b_df = pd.DataFrame([{**_row(64), "us": 0.9, "solutionIdx": "10"}])
    gemms1_stage_a_df = pd.DataFrame([{**_row(32), "us": 1.0, "solutionIdx": "10"}])
    gemms1_stage_b_df = pd.DataFrame([{**_row(32), "us": 0.9, "solutionIdx": "10"}])

    monkeypatch.setattr(
        search.bench.log, "dump", lambda data, p: yaml.safe_dump(data, Path(p).open("w"), sort_keys=False)
    )

    bench_run_calls: list[Path] = []

    def _fake_bench_run(_hip, _bench_file, log_file, **_kwargs):
        bench_run_calls.append(Path(log_file))
        call_index = len(bench_run_calls)
        if call_index == 1:
            Path(log_file).write_text("Winner\nWinner\n")  # chunk's own stage_a: both configs pass
        elif call_index == 2:
            raise RuntimeError("stage_b crashed")  # chunk's own stage_b: crashes
        else:
            # Retries: each retried config does its own stage_a (Winner) then
            # stage_b (output), strictly alternating in processing order.
            if (call_index - 3) % 2 == 0:
                Path(log_file).write_text("Winner\n")
            else:
                Path(log_file).write_text("stage_b output\n")
        return pd.DataFrame()

    monkeypatch.setattr(search.bench, "run", _fake_bench_run)

    def _fake_parse(p):
        p = Path(p)
        if p.name == "chunk_0_stage_a.out":
            return chunk_stage_a_df
        if p.stem.startswith("gemms_0"):
            return gemms0_stage_a_df if p.name.endswith("_stage_a.out") else gemms0_stage_b_df
        if p.stem.startswith("gemms_1"):
            return gemms1_stage_a_df if p.name.endswith("_stage_a.out") else gemms1_stage_b_df
        raise AssertionError(f"unexpected parse target: {p}")

    monkeypatch.setattr(search.bench.utils, "parse_benchmark_output", _fake_parse)

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

    # max_chunk_size=10 with 2 configs on 1 device packs both into ONE
    # "chunk_0" (multi-config) chunk, so its stage_a output uses "chunk_<i>"
    # naming -- the scenario the stale-file concern is actually about.
    out = search.run(hip, [cfg, filler_cfg], outdir, devices=[0], max_chunk_size=10, profile="balanced")

    # Stale artifact from the chunk's own (crashed) attempt: never cleaned up.
    assert (outdir / "chunk_0_stage_a.out").is_file()
    # The retry's own fresh stage_a output uses a DIFFERENT name -- no collision.
    assert (outdir / "gemms_0_stage_a.out").is_file()

    winners = {int(row["k"]): (row["us"], int(row["solutionIdx"])) for _, row in out.iterrows()}
    assert winners[64] == (0.9, 10)
    assert winners[32] == (0.9, 10)
    assert not (outdir / "failed_gemms.yaml").is_file()


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


def test_run_balances_chunks_by_flops_and_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chunks should be capacity-capped (equal-ish item counts) and
    flops-balanced (heaviest GEMMs spread across different chunks)."""
    hip = tmp_path / "hip"
    hip.mkdir()

    # flops == K here since M=N=batch_count=1; 3 heavy + 3 light GEMMs.
    flop_values = [100_000, 90_000, 80_000, 10, 9, 8]
    configs = [_gemm_config(M=1, N=1, K=k, batch_count=1) for k in flop_values]

    captured: dict = {}

    class _CaptureRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            captured["items"] = items

        def __call__(self, _workdir):
            return []

    monkeypatch.setattr(search, "Runner", _CaptureRunner)

    search.run(hip, configs, tmp_path / "out", devices=[0, 1], max_chunk_size=10)

    items = captured["items"]
    assert len(items) == 2

    # Capacity cap: chunks stay equally "full" (differ by at most one item).
    sizes = sorted(len(item["configs"]) for item in items)
    assert sizes == [3, 3]

    def _chunk_of(original_id: int) -> int:
        for i, item in enumerate(items):
            if any(cfg[1] == original_id for cfg in item["configs"]):
                return i
        raise AssertionError(f"original id {original_id} missing from chunks")

    # The two heaviest GEMMs (ids 0 and 1, flops 100_000/90_000) are the
    # first two placed by the greedy algorithm and must land in different
    # chunks -- that's the flops-balancing guarantee.
    assert _chunk_of(0) != _chunk_of(1)


def test_run_names_multi_and_single_config_items_differently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multi-config chunks are named "chunk_<i>"; single-config items (a
    lone chunk, or a retry) keep "gemms_<id>" so retries never clash with
    an existing chunk's output files."""
    hip = tmp_path / "hip"
    hip.mkdir()

    captured: dict = {}

    class _CaptureRunner:
        def __init__(self, items, worker_impl, devices, n_slots=1):
            captured["items"] = items
            captured["worker_impl"] = worker_impl
            captured["devices"] = devices

        def __call__(self, _workdir):
            return []

    monkeypatch.setattr(search, "Runner", _CaptureRunner)

    # 4 configs on 1 device => a single chunk holding all 4 (multi-config).
    configs = [_gemm_config(M=1, N=1, K=k, batch_count=1) for k in [4, 3, 2, 1]]
    search.run(hip, configs, tmp_path / "out", devices=[0], max_chunk_size=10)

    items = captured["items"]
    worker_impl = captured["worker_impl"]
    devices = captured["devices"]
    assert len(items) == 1
    assert len(items[0]["configs"]) == 4

    worker = worker_impl(items[0], devices[0], 0, None, None)
    worker.setup()
    assert worker.chunk_id == "chunk_0"
    assert worker.bench_file.name == "chunk_0.yaml"
    assert worker.log_file.name == "chunk_0.out"

    # A single config on its own device => a lone (single-config) chunk.
    captured.clear()
    search.run(hip, [configs[0]], tmp_path / "out2", devices=[0], max_chunk_size=10)
    worker = captured["worker_impl"](captured["items"][0], captured["devices"][0], 0, None, None)
    worker.setup()
    assert worker.chunk_id == "gemms_0"
    assert worker.bench_file.name == "gemms_0.yaml"

    # A single-config retry item (built directly, as search.run() does
    # internally) must use the same "gemms_<id>" naming.
    retry_item = {"configs": [(configs[0], 7)], "keep_failed_logs": True}
    retry_worker = worker_impl(retry_item, devices[0], 0, None, None)
    retry_worker.setup()
    assert retry_worker.chunk_id == "gemms_7"
    assert retry_worker.bench_file.name == "gemms_7.yaml"


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
