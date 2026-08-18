# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocKE-AOT family: compile the .co from family.json + the co-located producer at
# build time (see ../../CMakeLists.txt). rocKE is optional; skipped if unavailable.
#
# The producer imports `kernels.gfx950.attention_dense` from rocKE's `library` root,
# which the build pyenv editable-installs alongside `platform`, so no PYTHONPATH
# surgery is needed. The kernel is gfx950-only (supports_attention_dense rejects
# every other arch), and this folder only configures when gfx950 is in GPU_TARGETS.
rocke_add_aot_family(
    ARCH     gfx950
    FAMILY   attention_dense_prefill
    PRODUCER produce_attention_dense_co.py
)

rocke_add_aot_family_test(
    ARCH    gfx950
    SOURCES TestSdpaDenseNumericParityGfx950.cpp
)
