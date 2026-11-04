# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# External test inputs belong to the embedding project, not the SDK layout.
# Respect ordinary parent variables as well as explicit cache overrides.
if(NOT DEFINED ROCKE_HIPDNN_SDPA_TEMPLATE)
  set(ROCKE_HIPDNN_SDPA_TEMPLATE "" CACHE FILEPATH
      "hipDNN SDPA graph template to stage with installed contract tests")
endif()
if(NOT DEFINED ROCKE_HIPDNN_SDPA_BUNDLES_DIR)
  set(ROCKE_HIPDNN_SDPA_BUNDLES_DIR "" CACHE PATH
      "hipDNN integration-test bundle root to stage SDPA golden graphs from")
endif()

set(_ROCKE_SDPA_HIPDNN_ASSETS OFF)
if("${ROCKE_HIPDNN_SDPA_TEMPLATE}" STREQUAL ""
   AND "${ROCKE_HIPDNN_SDPA_BUNDLES_DIR}" STREQUAL "")
  return()
endif()
if("${ROCKE_HIPDNN_SDPA_TEMPLATE}" STREQUAL ""
   OR "${ROCKE_HIPDNN_SDPA_BUNDLES_DIR}" STREQUAL "")
  message(FATAL_ERROR
          "ROCKE_HIPDNN_SDPA_TEMPLATE and ROCKE_HIPDNN_SDPA_BUNDLES_DIR must be set together")
endif()
if(NOT EXISTS "${ROCKE_HIPDNN_SDPA_TEMPLATE}"
   OR IS_DIRECTORY "${ROCKE_HIPDNN_SDPA_TEMPLATE}")
  message(FATAL_ERROR "ROCKE_HIPDNN_SDPA_TEMPLATE must name an existing file")
endif()
if(NOT IS_DIRECTORY "${ROCKE_HIPDNN_SDPA_BUNDLES_DIR}")
  message(FATAL_ERROR "ROCKE_HIPDNN_SDPA_BUNDLES_DIR must name an existing directory")
endif()

file(GLOB_RECURSE _rocke_sdpa_graphs CONFIGURE_DEPENDS
     RELATIVE "${ROCKE_HIPDNN_SDPA_BUNDLES_DIR}"
     "${ROCKE_HIPDNN_SDPA_BUNDLES_DIR}/*.json")
# Keep the same selection as the installed SDPA asset reader.
list(FILTER _rocke_sdpa_graphs INCLUDE REGEX "^[^/]+/SdpaFwd[^/]*/.+\\.json$")
list(FILTER _rocke_sdpa_graphs EXCLUDE REGEX "(meta|support|sweep)\\.json$")
list(FILTER _rocke_sdpa_graphs EXCLUDE REGEX "(^|/)Default/")
if(_rocke_sdpa_graphs)
  set(_ROCKE_SDPA_HIPDNN_DEST "${ROCKE_TEST_INSTALL_DIR}/tests/library/tests/sdpa/hipdnn")
  install(FILES "${ROCKE_HIPDNN_SDPA_TEMPLATE}"
          DESTINATION "${_ROCKE_SDPA_HIPDNN_DEST}"
          RENAME attention_sdpa_fwd.graph.json)
  foreach(_rocke_sdpa_graph IN LISTS _rocke_sdpa_graphs)
    get_filename_component(_rocke_sdpa_graph_dir "${_rocke_sdpa_graph}" DIRECTORY)
    install(FILES "${ROCKE_HIPDNN_SDPA_BUNDLES_DIR}/${_rocke_sdpa_graph}"
            DESTINATION "${_ROCKE_SDPA_HIPDNN_DEST}/integration-test-bundles/${_rocke_sdpa_graph_dir}")
  endforeach()
  set(_ROCKE_SDPA_HIPDNN_ASSETS ON)
endif()
