# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Enroll an architecture after publishing its operation-specific archive and lock.
# Every operation package needs a list; one with nothing published uses
# set(ROCKE_PUBLISHED_<OP>_ARCHITECTURES "") because a bare set() unsets it.
set(ROCKE_PUBLISHED_SDPA_ARCHITECTURES gfx942 gfx950 gfx1151 gfx1201)
set(ROCKE_PUBLISHED_CONV_ARCHITECTURES gfx942 gfx950)
