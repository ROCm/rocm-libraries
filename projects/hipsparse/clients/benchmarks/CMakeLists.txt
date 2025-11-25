# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Create benchmark executable
add_executable(hipsparse-bench)

# Add benchmark sources
target_sources(hipsparse-bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_arguments_config.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bench.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bench_app.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_bench_cmdlines.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_routine.cpp"
)

# Additional compile options specific to benchmarks
target_compile_options(hipsparse-bench PRIVATE -Wno-deprecated)

# Additional includes when building clients only
if(HIPSPARSE_BUILD_CLIENTS_ONLY)
    target_include_directories(hipsparse-bench PRIVATE ${hipsparse_INCLUDE_DIRS}/hipsparse)
endif()

# Link dependencies - gets common properties transitively from helper
target_link_libraries(hipsparse-bench PRIVATE hipsparse-clients-common)

# Link OpenMP if enabled
if(HIPSPARSE_ENABLE_OPENMP)
    target_link_libraries(hipsparse-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

# Backend-specific settings
if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-bench PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(hipsparse-bench PRIVATE ${HIP_INCLUDE_DIRS})
    target_link_libraries(hipsparse-bench PRIVATE ${CUDA_LIBRARIES})
endif()

# Set output directory - backward compatible with Jenkins expectations
# Jenkins expects build/{release|debug}/clients/staging/
# Note: PROJECT_BINARY_DIR already includes the build type directory from install.sh
set_target_properties(hipsparse-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
)

# Installation
rocm_install(TARGETS hipsparse-bench COMPONENT benchmarks)
