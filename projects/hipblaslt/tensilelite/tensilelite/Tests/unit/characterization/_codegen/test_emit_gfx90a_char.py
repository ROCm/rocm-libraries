################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 1 — gfx90a (CDNA2 / MFMA) codegen-emit goldens.

Drives the real assembly emitter over curated valid gfx90a logic files spanning
data-type families, exercising the arch-specific paths in
``KernelWriterAssembly`` / ``KernelWriter`` / ``Components/*``.
Goldens are compact digests; see ``target.md``.
"""

import pytest

from matrix import digests_for_dir

pytestmark = pytest.mark.unit

_ARCH = "gfx90a"


def test_gfx90a_emit_digests(snapshot):
    digests = digests_for_dir(_ARCH)
    assert digests
    assert all(k["err"] == 0 for f in digests for k in f["kernels"])
    assert digests == snapshot
