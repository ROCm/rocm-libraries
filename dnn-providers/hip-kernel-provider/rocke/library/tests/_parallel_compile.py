# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run COMGR on many lowered kernels at once, in worker processes.

COMGR compiles one module at a time in a process: threads calling it do not
overlap. A test that needs many code objects therefore lowers its kernels in
its own process (``rocke.helpers.compile.lower_kernel_for_comgr``) and hands the
lowered inputs to :func:`compile_in_workers`, which runs
``compile_comgr_input`` on them in worker processes, one per CPU this process
may use, up to :data:`MAX_COMPILE_WORKERS`. A worker gives the code object
``compile_kernel`` would.
"""

from __future__ import annotations

import multiprocessing
import pickle
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from ._worker_process import available_cpus

__all__ = ["MAX_COMPILE_WORKERS", "compile_in_workers"]

#: The most worker processes one batch compiles in; each loads its own COMGR.
MAX_COMPILE_WORKERS = 8


def compile_in_workers(
    comgr_inputs: Sequence,
) -> list[tuple[bytes | None, BaseException | None]]:
    """``(code object, None)`` or ``(None, error)`` for each of ``comgr_inputs``
    (``ComgrInput`` values), in order; one input's failure is its own result.
    With one usable CPU, or one input, they compile in this process."""
    workers = min(available_cpus(), MAX_COMPILE_WORKERS, len(comgr_inputs))
    if workers <= 1:
        return [_code_object_or_error(comgr_input) for comgr_input in comgr_inputs]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(pool.map(_code_object_or_error, comgr_inputs))


def _code_object_or_error(comgr_input) -> tuple[bytes | None, BaseException | None]:
    """``compile_comgr_input`` of one lowered kernel, or the error it raised (as
    sent back from a worker: the original when it pickles, else its text)."""
    from rocke.helpers.compile import compile_comgr_input

    try:
        return compile_comgr_input(comgr_input), None
    except Exception as error:  # noqa: BLE001 - reported on its kernel
        try:
            pickle.dumps(error)
        except Exception:  # noqa: BLE001 - any pickling failure
            error = RuntimeError(f"{type(error).__name__}: {error}")
        return None, error
