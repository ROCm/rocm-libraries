# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(hipsparse-bench)

target_sources(hipsparse-bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_arguments_config.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bench.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bench_app.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bench_cmdlines.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_routine.cpp"
)

target_compile_options(hipsparse-bench PRIVATE -Wall)

if(NOT HIPSPARSE_ENABLE_HOST)
    target_include_directories(hipsparse-bench PRIVATE "${hipsparse_INCLUDE_DIRS}/hipsparse")
endif()

target_link_libraries(hipsparse-bench PRIVATE hipsparse::clients-common)

# Ensure pthread is linked on Unix platforms
if(NOT WIN32)
    target_link_libraries(hipsparse-bench PRIVATE pthread)
endif()

if(HIPSPARSE_ENABLE_OPENMP)
    target_link_libraries(hipsparse-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-bench PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_link_libraries(hipsparse-bench PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

# Set output directory - Jenkins expects clients/staging
set_target_properties(hipsparse-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
)

rocm_install(TARGETS hipsparse-bench COMPONENT benchmarks)
