# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# StinkyTofu Architecture List
# This file defines all supported architectures

# What "all" means. Gfx1250 (v1) and Gfx1250v0 (v0) are two steppings of the same chip: they
# report the same {12,5,0} ISA triple but are distinct GfxArchID identities. They ship together in
# one library so a single binary can serve both steppings, with the identity (not the triple)
# selecting the per-arch cost table. A build that wants only one stepping can still ask for it by
# name via -DSTINKYTOFU_ARCHS_TO_BUILD=Gfx1250 (or =Gfx1250v0).
set(STINKYTOFU_ALL_ARCHS
    Gfx1250
    Gfx1250v0
)
