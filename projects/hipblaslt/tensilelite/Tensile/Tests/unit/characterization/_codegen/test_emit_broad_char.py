################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 3 — broad operation-layout / data-type sweep.

A curated, de-duplicated set of small tuned logic files spanning every arch
family and many (operation-layout × data-type) combinations not covered by the
per-arch dtype suites — e.g. the Ailk_Bljk / Alik_Bljk transpose layouts, int8
(I8II / I8BH) on the WMMA archs, and BSS / B8HS / GradB variants. The aim is the
long tail of address-calculation, global-read and pack/convert branches in
``KernelWriterAssembly`` / ``KernelWriter`` / ``Components/*``. Order-invariant
golden ({basename, err}); see ``target.md``.
"""

import glob
import os

import pytest

from codegen_harness import emit_kernels_from_logic

pytestmark = pytest.mark.unit

_BROAD = os.path.join(os.path.dirname(__file__), "data", "broad")


def _cases():
    files = sorted(glob.glob(os.path.join(_BROAD, "**", "*.yaml"), recursive=True))
    return [(os.path.relpath(f, _BROAD), f) for f in files]


@pytest.mark.parametrize("rel,path", _cases(), ids=[c[0] for c in _cases()])
def test_broad_emit(rel, path, snapshot):
    results = emit_kernels_from_logic(path)
    assert results
    assert all(e == 0 for _b, _s, e in results)
    digest = [{"basename": b, "err": e} for (b, _s, e) in results]
    assert digest == snapshot
