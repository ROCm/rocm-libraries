# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocKE-AOT family: compile the .co from family.json + the co-located producer at
# build time (see ../../CMakeLists.txt). rocKE is optional; skipped if unavailable.
aot_add_rocke_family(
    ARCH     gfx1151
    FAMILY   gemm_wmma
    PRODUCER produce_gemm_wmma_co.py
)
