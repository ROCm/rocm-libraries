# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocKE-AOT family: compile the .co from family.json + the co-located producer at
# build time (see ../../CMakeLists.txt). rocKE is optional; skipped if unavailable.
aot_add_rocke_family(
    ARCH     gfx1151
    FAMILY   fmha_wmma_fwd
    PRODUCER produce_fmha_fwd_co.py
)
