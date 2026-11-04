# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# The embedding provider owns these inputs independently of SDPA graph assets.
if(NOT DEFINED ROCKE_HKP_PACKS_DIR)
  set(ROCKE_HKP_PACKS_DIR "" CACHE PATH
      "Provider pack headers to stage with installed attention contract tests")
endif()

set(_ROCKE_HKP_PACK_HEADERS_STAGED OFF)
if("${ROCKE_HKP_PACKS_DIR}" STREQUAL "")
  return()
endif()
if(NOT IS_DIRECTORY "${ROCKE_HKP_PACKS_DIR}")
  message(FATAL_ERROR "ROCKE_HKP_PACKS_DIR must name an existing directory")
endif()

set(_rocke_pack_headers Gfx950AttentionDenseGeometry.hpp
                        Gfx950AttentionUnified3DGeometry.hpp)
foreach(_header IN LISTS _rocke_pack_headers)
  if(NOT EXISTS "${ROCKE_HKP_PACKS_DIR}/${_header}"
     OR IS_DIRECTORY "${ROCKE_HKP_PACKS_DIR}/${_header}")
    message(FATAL_ERROR "ROCKE_HKP_PACKS_DIR must contain the file ${_header}")
  endif()
endforeach()
foreach(_header IN LISTS _rocke_pack_headers)
  install(FILES "${ROCKE_HKP_PACKS_DIR}/${_header}"
          DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/library/tests/hkp_packs")
endforeach()
set(_ROCKE_HKP_PACK_HEADERS_STAGED ON)
