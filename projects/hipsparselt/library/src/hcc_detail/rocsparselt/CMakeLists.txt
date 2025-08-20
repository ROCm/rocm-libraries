# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../library/src/hcc_detail/rocsparselt")

target_compile_definitions(hipsparselt PRIVATE ROCSPARSELT_TENSILE_LAZY_LOAD=1)

target_sources(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/handle.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/Debug.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/status.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/utility.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/rocsparselt_auxiliary.cpp"
)

target_sources(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_compress.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_prune.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/spmm/rocsparselt_spmm.cpp"
)

target_include_directories(hipsparselt PRIVATE
    "${_CMAKE_CURRENT_SOURCE_DIR}/include"
    "${_CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

target_sources(hipsparselt PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/src/tensile_host.cpp")
target_include_directories(hipsparselt PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/src/Tensile")

set(tensile_source_files "${_CMAKE_CURRENT_SOURCE_DIR}/src/tensile_host.cpp")
list(APPEND hipsparselt_source_backend ${tensile_source_files})

target_link_libraries(hipsparselt PRIVATE tensilelite::tensilelite-host)
