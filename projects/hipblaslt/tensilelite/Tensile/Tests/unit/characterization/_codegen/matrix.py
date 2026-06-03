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
import hashlib
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
    a compact, deterministic digest list suitable for snapshotting.

    Each entry: {file, kernels: [{basename, err, n_lines, sha256}, ...]}.
    """
    out = []
    for relname, path in logic_files(arch):
        kernels = []
        for base, src, err in emit_kernels_from_logic(path):
            kernels.append(
                {
                    "basename": base,
                    "err": err,
                    "n_lines": len(src.splitlines()) if src else 0,
                    "sha256": hashlib.sha256(src.encode()).hexdigest() if src else None,
                }
            )
        out.append({"file": relname, "kernels": kernels})
    return out
