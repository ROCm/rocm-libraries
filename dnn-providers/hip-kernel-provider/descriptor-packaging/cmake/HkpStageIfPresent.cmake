# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Copy HKP_FROM into HKP_TO, but only if HKP_FROM exists.
#
# Run in script mode from the staging command, not at configure time: a source
# root's output directory is created by the pack step in the same build, so
# whether it exists is only knowable once that step has run.
#
# A root that packed nothing is normal, not a failure. The packer writes no arch
# folder when nothing in that root targets the arch being built -- a tree whose
# descriptors all name gfx942 produces no output at all on a gfx90a build -- and
# `cmake -E copy_directory` treats a missing source as a hard error.

if(NOT DEFINED HKP_FROM OR NOT DEFINED HKP_TO)
    message(FATAL_ERROR "HkpStageIfPresent: set both HKP_FROM and HKP_TO")
endif()

if(NOT IS_DIRECTORY "${HKP_FROM}")
    return()
endif()

file(COPY "${HKP_FROM}/" DESTINATION "${HKP_TO}")
