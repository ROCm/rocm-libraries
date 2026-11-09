# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Build-time copy of tilewright weights. Globbing happens here, not at
# configure time, so manifests fetched after configure are still copied.
# cmake -DSRC=<tilewright arch dir> -DDST=<Tensile library arch dir> -P OrigamiNNColocate.cmake

if(NOT DEFINED SRC OR NOT DEFINED DST)
    message(FATAL_ERROR "OrigamiNNColocate.cmake requires -DSRC and -DDST")
endif()

if(NOT EXISTS "${SRC}/origami_nn_index")
    message(STATUS "origami tilewright weights not found (skip): ${SRC}")
    return()
endif()

file(MAKE_DIRECTORY "${DST}")
file(COPY_FILE "${SRC}/origami_nn_index" "${DST}/origami_nn_index" ONLY_IF_DIFFERENT)

file(GLOB _weights "${SRC}/*.tilewright.yaml" "${SRC}/*.tilewright.wts.yaml")
foreach(_file IN LISTS _weights)
    cmake_path(GET _file FILENAME _name)
    file(COPY_FILE "${_file}" "${DST}/${_name}" ONLY_IF_DIFFERENT)
endforeach()
