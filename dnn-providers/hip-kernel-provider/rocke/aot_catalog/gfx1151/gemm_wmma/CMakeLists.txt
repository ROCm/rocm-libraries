# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocKE-AOT family: compile the .co from family.json + the co-located producer at
# build time (see ../../CMakeLists.txt). rocKE is optional; skipped if unavailable.
rocke_add_aot_family(
    ARCH     gfx1151
    FAMILY   gemm_wmma
    PRODUCER produce_gemm_wmma_co.py
)

# This family's own numeric-parity tests (self-contained; see ../../CMakeLists.txt).
rocke_add_aot_family_test(
    ARCH    gfx1151
    SOURCES TestGemmWmmaNumericParity.cpp
)
