# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Helpers for GPU tests that check paged attention never reads outside a sequence.

A kernel's last KV tile spans keys past the sequence. They are masked to P = 0, but 0 * NaN is NaN,
so their K/V has to come from the sequence's own blocks. ``poison`` surrounds each sequence with
hostile data: NaN in the cache slots after its last key, and block-table entries past its own
blocks (in every row, also of shorter sequences) that point to a NaN block or to a block id far
outside the cache. A fault poisons the HIP context and an out-of-bounds read can also hang, so
``spawn_results`` runs every case in its own fresh process under one deadline.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import time

import pytest

FAR_BLOCK = 100_000  # outside any cache here, but within i32 element addressing
RESULTS_TIMEOUT_S = 300  # the cases take seconds


def own_blocks(kv_lens, bs, num_blocks):
    """Block table giving each sequence its own blocks; the cache's last block stays unused."""
    import torch

    need = [(kl + bs - 1) // bs for kl in kv_lens]
    perm = torch.randperm(num_blocks - 1, device="cuda").int()
    table = torch.zeros(len(kv_lens), max(need), dtype=torch.int32, device="cuda")
    start = 0
    for i, n in enumerate(need):
        table[i, :n] = perm[start : start + n]
        start += n
    return table


def poison(k, v, table, kv_lens, bs, mode):
    """Return the block table padded with hostile entries ("nan": the NaN-filled spare block;
    "far": a block id outside the cache), after filling the unused slots after each sequence's
    last key with NaN ("nan" only)."""
    import torch

    spare = k.shape[0] - 1
    if mode == "nan":
        k[spare] = float("nan")
        v[spare] = float("nan")
        for i, kl in enumerate(kv_lens):
            if kl % bs:
                blk = int(table[i, (kl - 1) // bs])
                k[blk, kl % bs :] = float("nan")
                v[blk, kl % bs :] = float("nan")
    pad = spare if mode == "nan" else FAR_BLOCK
    table = table.clone()
    for i, kl in enumerate(kv_lens):
        table[i, (kl + bs - 1) // bs :] = pad
    extra = torch.full((table.shape[0], 8), pad, dtype=table.dtype, device=table.device)
    return torch.cat([table, extra], 1).contiguous()


def spawn_results(jobs, env=None):
    """Submit ``{key: (fn, case)}`` each to its own spawned process; yield ``(futures, deadline)``
    (read them with ``result``), then stop the workers. ``env``: variables the workers start
    with. Use from a module-scoped fixture."""
    saved = {name: os.environ.get(name) for name in env or {}}
    os.environ.update(
        env or {}
    )  # submit() starts each worker, which copies the environment
    ctx = mp.get_context("spawn")  # fork after the parent touched the GPU is unsafe
    pools, futures = [], {}
    try:
        for key, (fn, case) in jobs.items():
            pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
            pools.append(pool)
            futures[key] = pool.submit(fn, case)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    yield futures, time.monotonic() + RESULTS_TIMEOUT_S
    for pool in pools:
        # A hung worker would block a waiting shutdown forever.
        for proc in getattr(pool, "_processes", {}).values():
            if proc.is_alive():
                proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)


def result(results, key):
    """The worker's return value; a fault in the worker re-raises here (a hard abort raises
    ``BrokenProcessPool``), and a worker past the deadline fails the test."""
    futures, deadline = results
    try:
        return futures[key].result(timeout=max(0.0, deadline - time.monotonic()))
    except cf.TimeoutError:
        pytest.fail(
            f"no result within {RESULTS_TIMEOUT_S} s: the kernel likely read out of bounds"
        )
