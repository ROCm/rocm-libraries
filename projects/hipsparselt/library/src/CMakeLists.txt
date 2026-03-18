# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(
  hipsparselt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_ostream.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary.cpp"
)

target_include_directories(hipsparselt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(HIPSPARSELT_ENABLE_HIP)
  # target_include_directories(hipsparselt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail")
  add_subdirectory(hcc_detail)
elseif(HIPSPARSELT_ENABLE_CUDA)
  # target_include_directories(hipsparselt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail")
  add_subdirectory(nvcc_detail)
else()
  message(
    FATAL_ERROR
      "Unsupported backend, either HIPSPARSELT_ENABLE_CUDA or HIPSPARSELT_ENABLE_HIP must be set to ON"
  )
endif()

source_group("Source Files\\Common" FILES ${hipsparselt_source_common})
source_group("Source Files\\Backend" FILES ${hipsparselt_source_backend})

if(HIPSPARSELT_ENABLE_TENSILELITE AND NOT HIPSPARSELT_ENABLE_CUDA)
  source_group("Source Files\\TensileLite" FILES ${tensile_source_files})
endif()
