# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Resolves the hipDNN SDK (header-only data_sdk + plugin_sdk, and optionally the
# frontend/backend for integration tests) as IMPORTED targets from a prebuilt
# hipDNN tree pointed to by HIPDNN_ROOT. Unlike the CK FMHA provider this needs
# no CK dispatcher / ck_host -- the only kernel backend is ck_dsl_runtime
# (HIP + libamd_comgr), built from the bundled runtime/ subdirectory.

find_package(hip REQUIRED)

if(NOT TARGET hipdnn_plugin_sdk OR NOT TARGET hipdnn_data_sdk)
    if(NOT DEFINED HIPDNN_ROOT)
        message(FATAL_ERROR "Set HIPDNN_ROOT to a built hipDNN source tree")
    endif()
    if(NOT DEFINED HIPDNN_BUILD_DIR)
        if(EXISTS "${HIPDNN_ROOT}/build/release")
            set(HIPDNN_BUILD_DIR "${HIPDNN_ROOT}/build/release")
        else()
            set(HIPDNN_BUILD_DIR "${HIPDNN_ROOT}/build")
        endif()
    endif()

    set(HIPDNN_FB_INCLUDE   "${HIPDNN_BUILD_DIR}/_deps/flatbuffers-src/include")
    set(HIPDNN_JSON_INCLUDE "${HIPDNN_BUILD_DIR}/_deps/json-src/include")

    # Helper: append a path to a list only if it exists on disk (INTERFACE
    # IMPORTED targets reject non-existent INTERFACE_INCLUDE_DIRECTORIES).
    macro(_add_if_exists _listvar _dir)
        if(EXISTS "${_dir}")
            list(APPEND ${_listvar} "${_dir}")
        endif()
    endmacro()

    add_library(hipdnn_data_sdk INTERFACE IMPORTED)

    if(EXISTS "${HIPDNN_ROOT}/flatbuffers_sdk/include")
        # --- Newer layout (rocm-libraries): the flatbuffer data_objects +
        # flatbuffer_utilities live in a separate hipdnn_flatbuffers_sdk; the
        # data_sdk keeps only the non-generated utilities (EngineNames, ...). ---
        set(_DS_INC "")
        _add_if_exists(_DS_INC "${HIPDNN_ROOT}/data_sdk/include")
        _add_if_exists(_DS_INC "${HIPDNN_BUILD_DIR}/data_sdk/include")
        target_include_directories(hipdnn_data_sdk INTERFACE ${_DS_INC})

        set(_FB_INC "")
        _add_if_exists(_FB_INC "${HIPDNN_ROOT}/flatbuffers_sdk/include")
        _add_if_exists(_FB_INC "${HIPDNN_BUILD_DIR}/flatbuffers_sdk/include")
        _add_if_exists(_FB_INC "${HIPDNN_FB_INCLUDE}")
        add_library(hipdnn_flatbuffers_sdk INTERFACE IMPORTED)
        target_include_directories(hipdnn_flatbuffers_sdk INTERFACE ${_FB_INC})
    else()
        # --- Older layout (rocm-lib-copy): generated data_objects live under
        # data_sdk/.../data_objects/v*; alias hipdnn_flatbuffers_sdk to data_sdk. ---
        file(GLOB _VDIRS LIST_DIRECTORIES TRUE
            "${HIPDNN_ROOT}/data_sdk/include/hipdnn_data_sdk/data_objects/v*")
        if(_VDIRS)
            list(SORT _VDIRS)
            list(GET _VDIRS -1 HIPDNN_DATA_SDK_GENERATED_DIR)
        endif()
        target_include_directories(hipdnn_data_sdk INTERFACE
            ${HIPDNN_ROOT}/data_sdk/include
            ${HIPDNN_DATA_SDK_GENERATED_DIR}
            ${HIPDNN_BUILD_DIR}/data_sdk/include
            ${HIPDNN_FB_INCLUDE})
        add_library(hipdnn_flatbuffers_sdk INTERFACE IMPORTED)
        target_link_libraries(hipdnn_flatbuffers_sdk INTERFACE hipdnn_data_sdk)
    endif()

    add_library(hipdnn_plugin_sdk INTERFACE IMPORTED)
    target_include_directories(hipdnn_plugin_sdk INTERFACE
        ${HIPDNN_ROOT}/plugin_sdk/include)
    target_link_libraries(hipdnn_plugin_sdk INTERFACE hipdnn_data_sdk hipdnn_flatbuffers_sdk)

    # frontend/backend only needed by the integration demo (optional).
    if(EXISTS "${HIPDNN_BUILD_DIR}/lib/libhipdnn_backend.so")
        add_library(hipdnn_frontend INTERFACE IMPORTED)
        target_include_directories(hipdnn_frontend INTERFACE
            ${HIPDNN_ROOT}/frontend/include
            ${HIPDNN_BUILD_DIR}/frontend/include
            ${HIPDNN_JSON_INCLUDE})
        target_link_libraries(hipdnn_frontend INTERFACE hipdnn_data_sdk)

        add_library(hipdnn_backend SHARED IMPORTED)
        set_target_properties(hipdnn_backend PROPERTIES
            IMPORTED_LOCATION ${HIPDNN_BUILD_DIR}/lib/libhipdnn_backend.so)
        target_include_directories(hipdnn_backend INTERFACE
            ${HIPDNN_ROOT}/backend/include
            ${HIPDNN_BUILD_DIR}/backend/src/backend/include)
    endif()
endif()
