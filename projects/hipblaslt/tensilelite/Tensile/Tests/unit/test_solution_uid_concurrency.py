# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Concurrent uniqueness tests for :mod:`Tensile.Common.SolutionIdGen`."""

from __future__ import annotations

import multiprocessing as mp
from typing import List

import pytest

from Tensile.Common.SolutionIdGen import (
    EPOCH_MS,
    MS_MASK,
    RANDOM_MASK,
    RANDOM_SHIFT,
    _ms_since_epoch,
    decode_solution_id,
    decode_solution_uid,
    encode_solution_uid,
    ensure_solution_uid,
    generate_solution_id,
    read_solution_uid,
    regenerate_solution_uid,
)

pytestmark = pytest.mark.unit

PROCESS_COUNT = 8
IDS_PER_PROCESS = 8


def _generate_after_barrier(
    barrier: mp.synchronize.Barrier,
    result_queue: mp.Queue,
) -> None:
    """Generate UIDs once all workers have reached the barrier.

    Args:
        barrier: Shared process barrier sized to the worker count.
        result_queue: Queue used to return generated UIDs to the parent process.
    """
    barrier.wait()
    result_queue.put([generate_solution_id() for _ in range(IDS_PER_PROCESS)])


def _run_concurrent_generation(
    process_count: int,
    ids_per_process: int,
) -> List[int]:
    """Run concurrent SolutionUID generation across multiple processes.

    Args:
        process_count: Number of worker processes to spawn.
        ids_per_process: UIDs each worker generates after synchronizing.

    Returns:
        Flat list of all generated SolutionUID values.
    """
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(process_count)
    result_queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_generate_after_barrier,
            args=(barrier, result_queue),
        )
        for _ in range(process_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0, f"worker exited with code {process.exitcode}"

    batches: List[List[int]] = []
    for _ in range(process_count):
        batches.append(result_queue.get(timeout=30))
    return [solution_uid for batch in batches for solution_uid in batch]


def test_8_processes_generate_unique_solution_uids() -> None:
    """8 processes generating UIDs concurrently must not collide."""
    solution_uids = _run_concurrent_generation(PROCESS_COUNT, IDS_PER_PROCESS)
    expected_count = PROCESS_COUNT * IDS_PER_PROCESS
    assert len(solution_uids) == expected_count

    unique_uids = set(solution_uids)
    duplicates = len(solution_uids) - len(unique_uids)
    assert duplicates == 0, (
        f"Expected all {expected_count} SolutionUID values to be unique, "
        f"but found {duplicates} collision(s)"
    )


def test_generate_solution_id_is_unique_across_sequential_calls() -> None:
    """Sequential calls on one process should also remain unique."""
    solution_uids = {generate_solution_id() for _ in range(1000)}
    assert len(solution_uids) == 1000


def test_decode_solution_id_splits_40_plus_24_layout() -> None:
    """``decode_solution_id`` extracts 40-bit ms and 24-bit random fields."""
    uid = generate_solution_id(now_ms=EPOCH_MS + 12345)
    ms_part, random_part = decode_solution_id(uid)
    assert ms_part == 12345
    assert 0 <= random_part <= RANDOM_MASK
    assert uid == (ms_part << RANDOM_SHIFT) | random_part


def test_read_solution_uid_reads_solution_uid_field() -> None:
    """SolutionUID in YAML state should be returned as-is."""
    assert read_solution_uid({"SolutionUID": "0u5"}) == 5


def test_read_solution_uid_generates_when_missing() -> None:
    """Missing SolutionUID should trigger ephemeral UID generation."""
    generated = read_solution_uid({})
    assert generated > 0


def test_ensure_solution_uid_assigns_solution_uid() -> None:
    """Missing UID fields should be assigned as SolutionUID."""
    solution: dict[str, str] = {}
    ensure_solution_uid(solution)
    assert "SolutionUID" in solution
    assert solution["SolutionUID"].startswith("0u")
    assert decode_solution_uid(solution["SolutionUID"]) > 0


def test_multi_winner_batch_assigns_unique_solution_uids() -> None:
    """Multi-winner LibraryLogic scale (10) should stay unique without retry."""
    import time

    now_ms = int(time.time() * 1000)
    winner_count = 10
    solutions = [{} for _ in range(winner_count)]
    for solution in solutions:
        ensure_solution_uid(solution)

    uids = [decode_solution_uid(solution["SolutionUID"]) for solution in solutions]
    assert len(set(uids)) == winner_count
    expected_ms = now_ms - EPOCH_MS
    assert all(decode_solution_id(uid)[0] >= expected_ms for uid in uids)


def test_ms_since_epoch_rejects_time_before_epoch() -> None:
    """Timestamps before the SolutionUID epoch are out of range.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a pre-epoch timestamp is accepted.
    """
    with pytest.raises(ValueError, match="out of range"):
        _ms_since_epoch(EPOCH_MS - 1)


def test_ms_since_epoch_rejects_time_after_40_bit_limit() -> None:
    """Timestamps that overflow the 40-bit millisecond field are rejected.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a post-mask timestamp is accepted.
    """
    with pytest.raises(ValueError, match="out of range"):
        generate_solution_id(now_ms=EPOCH_MS + MS_MASK + 1)


def test_ms_since_epoch_accepts_zero_and_max_offset() -> None:
    """Epoch and the last representable millisecond both encode.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If either bound is rejected or mis-encoded.
    """
    assert _ms_since_epoch(EPOCH_MS) == 0
    assert _ms_since_epoch(EPOCH_MS + MS_MASK) == MS_MASK


def test_ensure_solution_uid_preserves_existing_uid() -> None:
    """``ensure_solution_uid`` is a no-op when ``SolutionUID`` is already set.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If an existing UID is overwritten.
    """
    solution = {"SolutionUID": "0ug"}
    ensure_solution_uid(solution)
    assert solution["SolutionUID"] == "0ug"


def test_regenerate_solution_uid_replaces_existing_value() -> None:
    """``regenerate_solution_uid`` always writes a new UID.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the previous UID is kept.
    """
    solution = {"SolutionUID": "0ug"}
    new_uid = regenerate_solution_uid(solution)
    assert new_uid != 42
    assert solution["SolutionUID"] == encode_solution_uid(new_uid)
