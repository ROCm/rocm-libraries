# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocKE-AOT family: compile the .co from family.json + the co-located producer at
# build time (see ../../CMakeLists.txt). rocKE is optional; skipped if unavailable.
rocke_add_aot_family(
    ARCH     gfx1151
    FAMILY   fmha_wmma_fwd
    PRODUCER produce_fmha_fwd_co.py
)

# This family's own numeric-parity test (self-contained; see ../../CMakeLists.txt).
rocke_add_aot_family_test(
    ARCH    gfx1151
    SOURCES TestSdpaNumericParity.cpp
)
