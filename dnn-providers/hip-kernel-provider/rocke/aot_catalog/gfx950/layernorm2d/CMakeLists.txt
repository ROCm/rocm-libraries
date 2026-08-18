# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# rocKE-AOT family: compile the .co from family.json + the co-located producer at
# build time (see ../../CMakeLists.txt). rocKE is optional; skipped if unavailable.
rocke_add_aot_family(
    ARCH     gfx950
    FAMILY   layernorm2d
    PRODUCER produce_layernorm2d_co.py
)
