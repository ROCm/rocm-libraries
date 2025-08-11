# Copyright Advanced Micro Devices, Inc., or its affiliates. SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../clients")

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(cblas REQUIRED)
find_package(LAPACK REQUIRED)

if(HIPSPARSELT_ENABLE_BLIS)
  find_package(BLIS QUIET)
endif()

configure_file(
  "${_CMAKE_CURRENT_SOURCE_DIR}/include/hipsparselt_common.yaml"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsparselt_common.yaml" COPYONLY
)

configure_file(
  "${_CMAKE_CURRENT_SOURCE_DIR}/include/hipsparselt_template.yaml"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsparselt_template.yaml" COPYONLY
)

configure_file(
  "${_CMAKE_CURRENT_SOURCE_DIR}/common/hipsparselt_gentest.py"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsparselt_gentest.py" COPYONLY
)

add_library(hipsparselt-clients-common OBJECT)

target_compile_features(hipsparselt-clients-common PRIVATE cxx_std_17)

target_compile_definitions(hipsparselt-clients-common PUBLIC HIPSPARSELT_INTERNAL_API)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(hipsparselt-clients-common PUBLIC -mf16c)
endif()

target_sources(
  hipsparselt-clients-common
  PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/common/singletons.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/common/utility.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/common/cblas_interface.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/common/argument_model.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/common/hipsparselt_parse_data.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/common/hipsparselt_arguments.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/common/hipsparselt_random.cpp"
)

target_include_directories(
  hipsparselt-clients-common
  PUBLIC $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/include/spmm>
         $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}/../library/include>
         # Ideally, for the build interface, we can just look into the source tree
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(HIPSPARSELT_ENABLE_CUDA)
  target_compile_definitions(hipsparselt-clients-common PUBLIC __HIP_PLATFORM_NVIDIA__)
  target_include_directories(hipsparselt-clients-common PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(hipsparselt-clients-common PUBLIC CUDA::cudart CUDA::cuda_driver)
else()
  target_compile_definitions(
    hipsparselt-clients-common PUBLIC __HIP_PLATFORM_AMD__ ROCM_USE_FLOAT16
  )
  target_link_libraries(hipsparselt-clients-common PUBLIC hip::host hip::device)
endif()

target_link_libraries(hipsparselt-clients-common PRIVATE Threads::Threads PUBLIC roc::hipsparselt)

if(TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(hipsparselt-clients-common PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add subdirectories
# ============================================================================
if(HIPSPARSELT_ENABLE_SAMPLES)
  add_subdirectory(samples)
endif()

if(HIPSPARSELT_ENABLE_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(HIPSPARSELT_BUILD_TESTING OR BUILD_TESTING)
  add_subdirectory(test)
endif()
