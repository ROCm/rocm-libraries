################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Shared glue for the per-arch codegen-emit suites (Phase 1+).

Not a test module. Each ``test_emit_<arch>_char.py`` calls
:func:`digests_for_dir` on its ``data/<arch>/`` directory of curated logic YAMLs
and snapshots the returned list. The curated inputs are small copies of valid
tuning logic files (one or a few per arch/dtype) — they drive the real emit and
thereby cover the arch-specific paths in ``KernelWriterAssembly`` /
``KernelWriter`` / ``Components/*``.
"""

import glob
import os

from codegen_harness import emit_kernels_from_logic


def data_dir(arch):
    return os.path.join(os.path.dirname(__file__), "data", arch)


def logic_files(arch):
    """Sorted list of (relname, abspath) logic YAMLs for an arch."""
    d = data_dir(arch)
    files = sorted(glob.glob(os.path.join(d, "**", "*.yaml"), recursive=True))
    return [(os.path.relpath(f, d), f) for f in files]


def digests_for_dir(arch):
    """Emit every kernel from every logic file under ``data/<arch>/`` and return
    a compact, **order-invariant** digest list suitable for snapshotting.

    Each entry: {file, kernels: [{basename, err}, ...]}.

    The golden records kernel identity (``basename`` = the parameter-hash kernel
    name, independent of emit state) and the emit return code — both stable
    regardless of how many kernels were emitted earlier in the process. The full
    assembly *text* is deliberately NOT hashed here: the emitter accumulates
    process-global MMA-scheduler state (e.g. register-reuse) that changes which
    instructions are emitted depending on session history, so a text hash would
    be order-coupled and flaky in the full suite. The value of this suite is the
    coverage from running the real emit end-to-end (which still happens); the
    golden's job is to pin that every curated config emits successfully with a
    stable identity. See ``resistance.md``.
    """
    out = []
    for relname, path in logic_files(arch):
        kernels = []
        for base, src, err in emit_kernels_from_logic(path):
            kernels.append({"basename": base, "err": err})
        out.append({"file": relname, "kernels": kernels})
    return out
