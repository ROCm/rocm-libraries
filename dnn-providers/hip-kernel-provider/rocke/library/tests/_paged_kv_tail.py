# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Helpers for GPU tests that check paged attention never reads outside a sequence.

A kernel's last KV tile spans keys past the sequence. They are masked to P = 0, but 0 * NaN is NaN,
so their K/V has to come from the sequence's own blocks. ``poison`` surrounds each sequence with
hostile data: NaN in the cache slots after its last key, and block-table entries past its own
blocks (in every row, also of shorter sequences) that point to a NaN block or to a block id far
outside the cache. A fault poisons the HIP context and an out-of-bounds read can also hang, so
``spawn_results`` runs every case in its own fresh process with its own deadline, a few at a time
(each process holds its own torch and HIP context).
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import threading
import time
import traceback

import pytest

TOL = 4e-2  # repo-wide bf16 attention gate
FAR_BLOCK = 100_000  # outside any cache here
CASE_TIMEOUT_S = 300  # a case takes seconds
MAX_WORKERS = 8


class CaseTimeout(Exception):
    """A case gave no result in time: its kernel likely hangs on an out-of-bounds read."""


def own_blocks(kv_lens, bs, num_blocks):
    """Block table giving each sequence its own blocks; the cache's last block stays unused."""
    import torch

    need = [(kl + bs - 1) // bs for kl in kv_lens]
    assert sum(need) < num_blocks, "the cache has too few blocks for the sequences"
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
    # Large pages: keep the far block's byte offset below 2**31, where i32 offsets would wrap.
    far = min(FAR_BLOCK, ((1 << 31) - 1) // (k[0].numel() * k.element_size()))
    pad = spare if mode == "nan" else far
    table = table.clone()
    for i, kl in enumerate(kv_lens):
        table[i, (kl + bs - 1) // bs :] = pad
    extra = torch.full((table.shape[0], 8), pad, dtype=table.dtype, device=table.device)
    return torch.cat([table, extra], 1).contiguous()


def _run_case(conn, fn, case, env):
    # Set before anything imports torch: the allocator reads its configuration at init.
    os.environ.update(env or {})
    try:
        conn.send(("ok", fn(case)))
    except BaseException:  # noqa: BLE001 - reported to the parent
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def spawn_results(jobs, env=None):
    """Run ``{key: (fn, case)}`` each in its own fresh spawned process, at most ``MAX_WORKERS``
    at a time, and yield ``{key: future}`` (read them with ``result``). A case gets
    ``CASE_TIMEOUT_S`` from its start; then its process is killed. ``env``: variables each
    process sets for itself. Use from a module-scoped fixture."""
    ctx = mp.get_context("spawn")  # fork after the parent touched the GPU is unsafe
    futures = {key: cf.Future() for key in jobs}
    pending = list(jobs.items())
    running = {}  # key -> (process, connection, deadline)
    stop = threading.Event()

    def finish(key, error=None, value=None):
        proc, conn, _ = running.pop(key)
        if proc.is_alive():
            proc.join(timeout=10)
        if proc.is_alive():  # hung while exiting
            proc.kill()
        conn.close()
        if error is None:
            futures[key].set_result(value)
        else:
            futures[key].set_exception(error)

    def schedule():
        while (pending or running) and not stop.is_set():
            while pending and len(running) < MAX_WORKERS:
                key, (fn, case) = pending.pop(0)
                parent_end, child_end = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_run_case, args=(child_end, fn, case, env))
                proc.start()
                child_end.close()
                running[key] = (proc, parent_end, time.monotonic() + CASE_TIMEOUT_S)
            for key, (proc, conn, deadline) in list(running.items()):
                if conn.poll():
                    try:
                        status, value = conn.recv()
                    except EOFError:  # died without a result: a fault or an abort
                        finish(key, RuntimeError(f"worker exited ({proc.exitcode})"))
                        continue
                    if status == "ok":
                        finish(key, value=value)
                    else:
                        finish(key, RuntimeError(f"worker raised:\n{value}"))
                elif not proc.is_alive():
                    finish(key, RuntimeError(f"worker exited ({proc.exitcode})"))
                elif time.monotonic() > deadline:
                    proc.kill()
                    finish(key, CaseTimeout(f"no result within {CASE_TIMEOUT_S} s"))
            time.sleep(0.05)

    thread = threading.Thread(target=schedule, daemon=True)
    thread.start()
    try:
        yield futures
    finally:
        stop.set()
        thread.join()
        for proc, conn, _ in running.values():
            proc.kill()
            conn.close()
        for future in futures.values():
            future.cancel()


def spawn_selected(request, fn, env=None):
    """``spawn_results`` of ``fn`` over the ``case`` parameters of the requesting module's selected
    tests, keyed by the case's ``id``. Use from a module-scoped fixture."""
    cases = [
        item.callspec.params["case"]
        for item in request.session.items
        if item.module is request.module
        and "case" in getattr(getattr(item, "callspec", None), "params", {})
    ]
    yield from spawn_results({c["id"]: (fn, c) for c in cases}, env)


def result(results, key):
    """The case's return value; a fault in its process re-raises here, and a case past its
    deadline fails the test."""
    try:
        return results[key].result()
    except CaseTimeout as e:
        pytest.fail(f"{e}: the kernel likely read out of bounds")
