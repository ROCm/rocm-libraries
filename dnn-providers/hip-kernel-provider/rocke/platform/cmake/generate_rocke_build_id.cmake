# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Recompute the portable-IR bundle provenance header as a build step. Hashing
# the engine/CLI source and build-recipe inputs prevents an incremental build
# from stamping either artifact with a configure-time ID.

if(NOT DEFINED ROCKE_SOURCE_ROOT OR NOT DEFINED ROCKE_BUILD_ID_HEADER OR
   NOT DEFINED ROCKE_ENGINE_VERSION)
  message(FATAL_ERROR "ROCKE_SOURCE_ROOT, ROCKE_BUILD_ID_HEADER and ROCKE_ENGINE_VERSION are required")
endif()

include("${ROCKE_SOURCE_ROOT}/cmake/rocke_build_id.cmake")
rocke_compute_build_id(_rocke_build_id "${ROCKE_SOURCE_ROOT}")

string(CONCAT _content
       "#ifndef ROCKE_GENERATED_BUILD_ID_VALUES_H\n"
       "#define ROCKE_GENERATED_BUILD_ID_VALUES_H\n"
       "#define ROCKE_BUILD_ID ${_rocke_build_id}\n"
       "#define ROCKE_ENGINE_VERSION ${ROCKE_ENGINE_VERSION}\n"
       "#endif\n")
get_filename_component(_header_dir "${ROCKE_BUILD_ID_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${_header_dir}")
set(_tmp "${ROCKE_BUILD_ID_HEADER}.tmp")
file(WRITE "${_tmp}" "${_content}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                        "${_tmp}" "${ROCKE_BUILD_ID_HEADER}"
                RESULT_VARIABLE _copy_result)
file(REMOVE "${_tmp}")
if(NOT _copy_result EQUAL 0)
  message(FATAL_ERROR "failed to publish ${ROCKE_BUILD_ID_HEADER}: ${_copy_result}")
endif()
