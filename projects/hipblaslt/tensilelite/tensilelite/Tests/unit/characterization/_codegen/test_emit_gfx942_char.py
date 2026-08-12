################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Phase 1 — gfx942 (aquavanjaram, CDNA3/MFMA) codegen-emit goldens.

Drives the real assembly emitter over a curated set of valid gfx942 logic files
spanning data-type families (HSS, HHS, BBS, SB/F32, DB/F64, F8N) and operation
layouts. The emit exercises the arch-specific MFMA paths in
``KernelWriterAssembly`` / ``KernelWriter`` and the data-type ``Components/*``
(MAC_*, PackData, conversion). Goldens are compact digests; see ``target.md``.
"""

import pytest

from matrix import digests_for_dir

pytestmark = pytest.mark.unit

_ARCH = "gfx942"


def test_gfx942_emit_digests(snapshot):
    digests = digests_for_dir(_ARCH)
    assert digests  # the curated corpus is present
    # every curated kernel emits without error
    assert all(k["err"] == 0 for f in digests for k in f["kernels"])
    assert digests == snapshot
