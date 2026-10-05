# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the CSV-to-Parquet boundary, where RFC 0019.13 §8.3's checks apply."""

from __future__ import annotations

import pytest

# Skip at module scope so a missing pandas is a skip, not a collection error.
pd = pytest.importorskip("pandas")

from uhd_gen.dataset.publish import (  # noqa: E402  (deliberately after the skip)
    ValidationError,
    build_dataset,
    expand_descriptors,
    load_csvs,
    resolve_duplicates,
    write_parquet,
)


def rows(**overrides) -> pd.DataFrame:
    base = {
        "q.M": [1024, 1024],
        "q.N": [1024, 1024],
        "q.K": [1024, 1024],
        "q.dtype": ["fp32", "fp32"],
        # Engine-published costs (RFC 0019 13.6): 2*M*N*K and three 1024^2 fp32 tensors.
        "q.flops": [2 * 1024**3, 2 * 1024**3],
        "q.bytes": [3 * 1024**2 * 4, 3 * 1024**2 * 4],
        "kernel.tile_m": [64, 128],
        "device.cu_count": [80, 80],
        "minTimeMs": [1.0, 2.0],
        "avgTimeMs": [1.1, 2.1],
        "stddevMs": [0.01, 0.01],
        "iters": [10, 10],
        "error": ["", ""],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_metrics_are_added_and_collection_bookkeeping_is_dropped():
    frame = rows(shard_id=[3, 3])
    out = build_dataset(frame)

    assert "tflops" in out.columns and "gbs" in out.columns
    assert out["tflops"].iloc[0] == pytest.approx(2 * 1024**3 / 1.1e-3 / 1e12)
    assert "shard_id" not in out.columns


def test_the_published_rate_names_the_winner_the_calibrated_label_names():
    """Rates come from `avgTimeMs`, the statistic a calibrated label is defined on."""
    out = build_dataset(rows(minTimeMs=[1.0, 1.5], avgTimeMs=[2.0, 1.6]))
    assert out.loc[out["tflops"].idxmax(), "kernel.tile_m"] == 128
    assert out["tflops"].tolist() == pytest.approx(
        [2 * 1024**3 / t / 1e9 for t in (2.0, 1.6)]
    )


def test_absent_optional_columns_take_their_defaults():
    """A foreign CSV carries q.*, kernel.*, device.* and a measurement, and nothing else."""
    out = build_dataset(rows())
    assert out["problem_complete"].all()


def test_a_failed_row_keeps_null_metrics_and_downgrades_its_problem():
    """A failed candidate keeps its row, but its problem is no longer complete."""
    frame = rows(
        minTimeMs=[1.0, None],
        avgTimeMs=[1.1, None],
        stddevMs=[0.01, None],
        iters=[10, None],
        error=["", "HIP error 700"],
        problem_complete=[True, True],
    )
    out = build_dataset(frame)

    assert pd.isna(out["tflops"].iloc[1]) and pd.isna(out["gbs"].iloc[1])
    assert not out[
        "problem_complete"
    ].any(), "the errored candidate left the problem complete"


def test_a_row_claiming_both_a_measurement_and_an_error_is_rejected():
    frame = rows(error=["", "HIP error 700"])
    with pytest.raises(ValidationError, match="both"):
        build_dataset(frame)


def test_a_row_with_neither_is_rejected():
    """An unattempted pair is a filtered configuration, recorded in the run report."""
    frame = rows(minTimeMs=[1.0, None], avgTimeMs=[1.1, None])
    with pytest.raises(ValidationError, match="neither"):
        build_dataset(frame)


def test_a_complete_problem_spanning_two_candidate_sets_is_rejected():
    """Two complete collections of one problem that repeat a configuration conflict."""
    frame = pd.concat([rows(), rows()], ignore_index=True)
    frame["problem_complete"] = True
    with pytest.raises(ValidationError, match="candidate sets"):
        build_dataset(frame)


def test_inconsistent_completeness_across_one_problem_is_rejected():
    frame = rows(problem_complete=[True, False])
    with pytest.raises(ValidationError, match="disagrees"):
        build_dataset(frame)


def test_min_above_avg_is_rejected():
    with pytest.raises(ValidationError, match="minTimeMs"):
        build_dataset(rows(minTimeMs=[9.0, 2.0]))


def test_a_corpus_that_identifies_no_problem_is_rejected():
    frame = rows().drop(columns=["q.M", "q.N", "q.K", "q.dtype", "q.flops", "q.bytes"])
    with pytest.raises(ValidationError, match="no problem columns"):
        build_dataset(frame)


def test_the_problem_namespace_is_the_operations_own_name():
    """Any namespace other than `kernel.*`/`device.*` is the problem namespace."""
    frame = rows().rename(
        columns=lambda c: c.replace("q.", "matmul.", 1) if c.startswith("q.") else c
    )
    out = build_dataset(frame)

    assert "matmul.M" in out.columns and not any(
        c.startswith("q.") for c in out.columns
    )
    assert out["tflops"].iloc[0] == pytest.approx(2 * 1024**3 / 1.1e-3 / 1e12)


def test_shards_concatenate_and_round_trip_through_parquet(tmp_path):
    """Appending is why collection stays CSV; skips without pyarrow."""
    pytest.importorskip("pyarrow")
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    rows().to_csv(first, index=False)
    rows(**{"q.M": [512, 512]}).to_csv(second, index=False)

    out = build_dataset(load_csvs([first, second]))
    assert len(out) == 4

    destination = tmp_path / "out" / "results.parquet"
    write_parquet(out, destination)
    back = pd.read_parquet(destination)

    # pyarrow normalises object columns to a string dtype; compare values only.
    pd.testing.assert_frame_equal(back, out, check_dtype=False)

    # A null must survive the round trip rather than become "" or 0.
    nulled = build_dataset(
        rows(
            minTimeMs=[1.0, None],
            avgTimeMs=[1.1, None],
            stddevMs=[0.01, None],
            iters=[10, None],
            error=["", "HIP error 700"],
        ),
    )
    write_parquet(nulled, destination)
    reread = pd.read_parquet(destination)
    assert reread["tflops"].isna().tolist() == [False, True]
    assert reread["minTimeMs"].isna().tolist() == [False, True]


def test_an_empty_csv_field_reads_back_as_a_null_metric(tmp_path):
    """The encoding §8.3 specifies: null is the empty field, and it survives to the dataset."""
    path = tmp_path / "partial.csv"
    path.write_text(
        "q.M,q.N,q.K,q.dtype,kernel.tile_m,device.cu_count,"
        "minTimeMs,avgTimeMs,stddevMs,iters,error\n"
        "1024,1024,1024,fp32,64,80,,,,,HIP error 700\n"
    )
    out = build_dataset(load_csvs([path]))
    assert pd.isna(out["tflops"].iloc[0])
    assert out["error"].iloc[0] == "HIP error 700"


def test_a_solver_name_bound_to_two_ids_is_refused():
    """One solver under two ids, as in a corpus merged across engine versions."""
    frame = rows(
        **{
            "kernel.solver": ["ConvBinWinoRxS", "ConvBinWinoRxS"],
            "kernel.solver_id": [37, 53],
        }
    )
    with pytest.raises(ValidationError, match="ambiguous"):
        build_dataset(frame)


def test_a_solver_id_bound_to_two_names_is_refused():
    """The other direction: a reused id."""
    frame = rows(
        **{
            "kernel.solver": ["ConvBinWinogradRxSf3x2", "ConvBinWinoRxS<3-2>"],
            "kernel.solver_id": [37, 37],
        }
    )
    with pytest.raises(ValidationError, match="ambiguous"):
        build_dataset(frame)


def test_the_agreeing_case_passes():
    frame = rows(
        **{
            "kernel.solver": ["ConvBinWinogradRxSf3x2", "ConvBinWinogradRxSf2x3"],
            "kernel.solver_id": [37, 53],
        }
    )
    assert len(build_dataset(frame)) == 2


def test_a_corpus_without_ids_is_left_alone():
    """Producers other than the MIOpen adapter carry no id, and must not be made to."""
    frame = rows(**{"kernel.solver": ["SomeEngineKernel", "SomeEngineKernel"]})
    assert len(build_dataset(frame)) == 2


def test_the_pairing_is_a_convention_not_a_column_list():
    """`X`/`X_id` from any producer, not one engine's spelling."""
    frame = rows(**{"kernel.variant": ["fast", "fast"], "kernel.variant_id": [7, 9]})
    with pytest.raises(ValidationError, match="ambiguous"):
        build_dataset(frame)


def test_an_unpaired_column_is_not_checked():
    """`kernel.tile_m` has no `kernel.tile_m_id`, so there is nothing to agree with."""
    assert len(build_dataset(rows())) == 2


# ---------------------------------------------------------------------------------------
# Configuration expansion and duplicate resolution
# ---------------------------------------------------------------------------------------


def test_expansion_makes_two_configurations_of_one_kernel_distinguishable():
    """Unexpanded, a model sees one feature row for every configuration of a kernel."""
    frame = rows(**{"kernel.descriptor": ["t,64,4", "t,128,4"]})
    out = expand_descriptors(build_dataset(frame), ["kernel.descriptor"])

    assert out["kernel.descriptor.cfg0"].tolist() == [64, 128]
    assert out["kernel.descriptor.cfg1"].tolist() == [4, 4]
    # The source column is kept as the readable identity of a configuration.
    assert "kernel.descriptor" in out.columns
    # The word shape stays text; the training tool numbers it (RFC 0019 §6.5).
    assert out["kernel.descriptor.variant"].tolist() == ["t", "t"]


def test_expanding_a_column_the_corpus_lacks_is_refused():
    """Named but absent is a mistake in the invocation, not an empty result to carry forward."""
    with pytest.raises(ValidationError, match="does not carry"):
        expand_descriptors(build_dataset(rows()), ["kernel.nope"])


def test_resolution_keeps_the_latest_occasion_per_problem():
    """Per problem, not per file: an older sweep's problems must survive."""
    frame = pd.DataFrame(
        {
            "q.M": [1024, 1024, 2048],
            "q.N": [1024, 1024, 2048],
            "q.K": [1024, 1024, 2048],
            "q.dtype": ["fp32"] * 3,
            "kernel.tile_m": [64, 64, 64],
            "device.cu_count": [80, 80, 80],
            "minTimeMs": [5.0, 1.0, 7.0],
            "avgTimeMs": [5.1, 1.1, 7.1],
            "stddevMs": [0.01] * 3,
            "iters": [10] * 3,
            "error": [""] * 3,
            "date_run": ["2026-01-01", "2026-02-01", "2026-01-01"],
        }
    )

    out = resolve_duplicates(frame, "date_run", "minTimeMs")

    assert sorted(out["q.M"].tolist()) == [1024, 2048]
    assert out.loc[out["q.M"] == 1024, "minTimeMs"].tolist() == [1.0]
    assert out.loc[out["q.M"] == 2048, "minTimeMs"].tolist() == [7.0]


def test_a_repeat_within_one_occasion_keeps_the_fastest():
    """Repeats differ by contention and clocks, not by the kernel."""
    frame = pd.DataFrame(
        {
            "q.M": [1024, 1024],
            "q.N": [1024, 1024],
            "q.K": [1024, 1024],
            "q.dtype": ["fp32", "fp32"],
            # What the engine published beside the shape (RFC 0019 13.6), and the only thing the
            # metrics are derived from: 2*M*N*K, and three 1024^2 fp32 tensors.
            "q.flops": [2 * 1024**3, 2 * 1024**3],
            "q.bytes": [3 * 1024**2 * 4, 3 * 1024**2 * 4],
            "kernel.tile_m": [64, 64],
            "device.cu_count": [80, 80],
            "minTimeMs": [5.0, 2.0],
            "avgTimeMs": [5.1, 2.1],
            "stddevMs": [0.01, 0.01],
            "iters": [10, 10],
            "error": ["", ""],
            "date_run": ["2026-01-01", "2026-01-01"],
        }
    )

    out = resolve_duplicates(frame, "date_run", "minTimeMs")
    assert out["minTimeMs"].tolist() == [2.0]


def test_resolution_settles_what_validation_would_otherwise_reject():
    """A re-measured problem fails validation until resolution keeps one occasion."""
    frame = pd.DataFrame(
        {
            "q.M": [1024, 1024],
            "q.N": [1024, 1024],
            "q.K": [1024, 1024],
            "q.dtype": ["fp32", "fp32"],
            "q.flops": [2 * 1024**3, 2 * 1024**3],
            "q.bytes": [3 * 1024**2 * 4, 3 * 1024**2 * 4],
            "kernel.tile_m": [64, 64],
            "device.cu_count": [80, 80],
            "minTimeMs": [5.0, 1.0],
            "avgTimeMs": [5.1, 1.1],
            "stddevMs": [0.01, 0.01],
            "iters": [10, 10],
            "error": ["", ""],
            "problem_complete": [True, True],
            "date_run": ["2026-01-01", "2026-02-01"],
        }
    )

    with pytest.raises(ValidationError, match="same kernel configuration twice"):
        build_dataset(frame)

    build_dataset(resolve_duplicates(frame, "date_run", "minTimeMs"))


def test_resolution_keeps_each_boards_problem_apart():
    """One shape on two boards is two problems, each with its own newest occasion."""
    frame = pd.DataFrame(
        {
            "benchmark": ["g0"] * 4,
            "device": ["boardA", "boardA", "boardB", "boardB"],
            "q.M": [1024] * 4,
            "q.N": [1024] * 4,
            "q.K": [1024] * 4,
            "q.dtype": ["fp32"] * 4,
            "kernel.tile_m": [64, 128, 64, 128],
            "device.cu_count": [256, 256, 304, 304],
            "minTimeMs": [1.0, 2.0, 3.0, 0.5],
            "avgTimeMs": [1.1, 2.1, 3.1, 0.6],
            "stddevMs": [0.01] * 4,
            "iters": [10] * 4,
            "error": [""] * 4,
            "date_run": ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"],
        }
    )

    for dated in (frame, frame.assign(date_run="2026-01-01")):
        out = resolve_duplicates(dated, "date_run", "avgTimeMs")
        assert sorted(zip(out["device"], out["kernel.tile_m"])) == [
            ("boardA", 64),
            ("boardA", 128),
            ("boardB", 64),
            ("boardB", 128),
        ]


def test_scoping_gives_each_group_its_own_positions():
    """Descriptor positions are scoped per group when the schema varies by kernel."""
    frame = rows(
        **{
            "kernel.solver_id": [107, 137],
            "kernel.descriptor": ["a,64,4", "b,7"],
        }
    )
    out = expand_descriptors(
        build_dataset(frame), ["kernel.descriptor"], scope_by="kernel.solver_id"
    )

    assert out["kernel.descriptor.s107_f0"].tolist() == [64, -1]
    assert out["kernel.descriptor.s107_f1"].tolist() == [4, -1]
    assert out["kernel.descriptor.s137_f0"].tolist() == [-1, 7]
    assert not [c for c in out.columns if c.startswith("kernel.descriptor.cfg")]


def test_a_row_outside_its_group_takes_the_absent_value():
    """Absent (-1), not zero: zero is a legal tuning value."""
    frame = rows(
        **{
            "kernel.solver_id": [107, 137],
            "kernel.descriptor": ["a,0", "b,7"],
        }
    )
    out = expand_descriptors(
        build_dataset(frame), ["kernel.descriptor"], scope_by="kernel.solver_id"
    )
    assert out["kernel.descriptor.s107_f0"].tolist() == [
        0,
        -1,
    ], "a real 0 was confused with absent"


def test_scoping_by_a_column_the_corpus_lacks_is_refused():
    with pytest.raises(ValidationError, match="scope-by"):
        expand_descriptors(
            build_dataset(rows(**{"kernel.descriptor": ["a,1", "b,2"]})),
            ["kernel.descriptor"],
            scope_by="kernel.nope",
        )


def test_unscoped_expansion_still_shares_one_set_of_positions():
    """The default, for an engine whose schema does not vary."""
    frame = rows(**{"kernel.descriptor": ["a,64,4", "a,128,4"]})
    out = expand_descriptors(build_dataset(frame), ["kernel.descriptor"])
    assert out["kernel.descriptor.cfg0"].tolist() == [64, 128]


def test_two_boards_measuring_one_shape_import_as_two_problems():
    """A problem is (graph, device): one shape on two boards is two candidate sets."""
    frame = pd.concat(
        [rows(device=["a", "a"]), rows(device=["b", "b"])], ignore_index=True
    )
    frame["problem_complete"] = True

    out = build_dataset(frame)

    assert len(out) == 4
    assert out["problem_complete"].all()


def test_a_fault_on_one_board_does_not_downgrade_the_other_boards_problem():
    """`problem_complete` is per board, so a healthy board's regret stays exact."""
    healthy = rows(device=["a", "a"])
    faulted = rows(
        device=["b", "b"],
        minTimeMs=[1.0, None],
        avgTimeMs=[1.1, None],
        stddevMs=[0.01, None],
        iters=[10, None],
        error=["", "HIP error 700"],
    )

    out = build_dataset(pd.concat([healthy, faulted], ignore_index=True))

    assert out.loc[out["device"] == "a", "problem_complete"].all()
    assert not out.loc[out["device"] == "b", "problem_complete"].any()


def test_the_collectors_spelling_of_a_failure_is_republished_as_an_error():
    """`export-benchmarks` writes `is_valid=False` and `skip_reason`, no `error`."""
    frame = rows(
        minTimeMs=[1.0, None],
        avgTimeMs=[1.1, None],
        stddevMs=[0.01, None],
        iters=[10, None],
        is_valid=["True", "False"],
        skip_reason=["", "hip error 700"],
    ).drop(columns=["error"])

    out = build_dataset(frame)

    assert out["error"].tolist() == ["", "hip error 700"]
    # §8.3 records a failure once, as `error`; the validity flag is not republished.
    assert "is_valid" not in out.columns and "skip_reason" not in out.columns
    assert not out["problem_complete"].any()


def test_a_row_marked_failed_that_still_carries_a_timing_is_still_rejected():
    """A row marked failed that still reports a time is a producer bug."""
    frame = rows(is_valid=["True", "False"], skip_reason=["", "hip error 700"]).drop(
        columns=["error"]
    )
    with pytest.raises(ValidationError, match="both"):
        build_dataset(frame)


@pytest.mark.parametrize(
    "timing",
    [
        # The bench reports a time even when the verdict is wrong.
        {"minTimeMs": [1.0, 0.011], "avgTimeMs": [1.1, 0.011], "stddevMs": [0.01, 0.0]},
        # `generate` has already suppressed the timing.
        {"minTimeMs": [1.0, None], "avgTimeMs": [1.1, None], "stddevMs": [0.01, None]},
    ],
    ids=["bench", "generated"],
)
def test_a_numerically_wrong_row_is_published_as_a_failure_with_its_verdict(
    tmp_path, timing
):
    """RFC 0019 §13.2: a wrong candidate keeps its row and marker, never its time."""
    path = tmp_path / "collected.csv"
    rows(
        **timing,
        numerically_valid=[True, False],
        validation=["agrees_with_catalog: 2 of 2", "output_mismatch: tensor 'Y'"],
    ).to_csv(path, index=False)

    out = build_dataset(load_csvs([path]))

    good, wrong = out.iloc[0], out.iloc[1]
    assert good["tflops"] > 0 and good["error"] == ""
    for label in ("minTimeMs", "avgTimeMs", "stddevMs", "tflops", "gbs"):
        assert pd.isna(wrong[label]), label
    assert wrong["error"] == "output_mismatch: tensor 'Y'"
    assert not wrong["numerically_valid"]
    assert out["numerically_valid"].tolist() == [True, False]
    assert wrong["validation"] == "output_mismatch: tensor 'Y'"
    assert not out["problem_complete"].any()


def test_a_numeric_looking_identity_is_read_as_a_name_not_a_number(tmp_path):
    """Identity columns stay strings, so `0123` is not read as 123."""
    path = tmp_path / "shard.csv"
    rows(benchmark=["0123", "0123"], device=["0007", "0007"]).to_csv(path, index=False)

    out = build_dataset(load_csvs([path]))

    assert out["benchmark"].tolist() == ["0123", "0123"]
    assert out["device"].tolist() == ["0007", "0007"]


def test_a_collected_sdpa_corpus_imports_and_is_given_its_metrics(tmp_path):
    """An SDPA corpus through the real read path, with engine-published causal costs."""
    path = tmp_path / "sdpa.csv"
    pd.DataFrame(
        {
            "benchmark": ["prefill", "decode"],
            "device": ["gfx942-0", "gfx942-0"],
            "q.batch": [2, 2],
            "q.heads": [16, 16],
            "q.seqlen_q": [1024, 1],
            "q.seqlen_k": [1024, 4096],
            "q.head_dim": [128, 128],
            "q.is_causal": [1, 1],
            "q.dtype": ["fp16", "fp16"],
            "q.flops": [
                4 * 2 * 16 * 128 * (1024 * 1024 - 1024 * 1023 / 2),
                4 * 2 * 16 * 128 * 1 * 4096,
            ],
            "q.bytes": [
                2 * 2 * 16 * 128 * (2 * 1024 + 2 * 1024),
                2 * 2 * 16 * 128 * (2 * 1 + 2 * 4096),
            ],
            "kernel.tile_m": [64, 128],
            "device.cu_count": [304, 304],
            "minTimeMs": [1.0, 0.5],
            "avgTimeMs": [1.1, 0.55],
            "stddevMs": [0.01, 0.01],
            "iters": [20, 20],
        }
    ).to_csv(path, index=False)

    out = build_dataset(load_csvs([path]))

    scale = 4 * 2 * 16 * 128
    assert out["tflops"].iloc[0] == pytest.approx(
        scale * (1024 * 1024 - 1024 * 1023 / 2) / 1.1e-3 / 1e12
    )
    # A single query sees a full row of the causal mask, not half of it.
    assert out["tflops"].iloc[1] == pytest.approx(scale * 1 * 4096 / 5.5e-4 / 1e12)
    assert out["gbs"].notna().all()
