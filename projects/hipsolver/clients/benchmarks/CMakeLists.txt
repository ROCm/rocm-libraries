# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(HIPSOLVER_ENABLE_HIPSPARSE_TESTS)
    find_package(hipsparse REQUIRED CONFIG)
endif()

# --- Benchmark executable ---

add_executable(hipsolver-bench)

target_compile_features(hipsolver-bench PRIVATE cxx_std_17)

target_sources(hipsolver-bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp"
)

target_compile_definitions(hipsolver-bench PRIVATE HIPSOLVER_BENCH ROCM_USE_FLOAT16)

# --- Includes ---

target_include_directories(hipsolver-bench PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_include_directories(hipsolver-bench SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# --- Linking ---

target_link_libraries(hipsolver-bench PRIVATE
    ${LAPACK_LIBRARIES}
    hipsolver-clients-common
    roc::hipsolver
)

if(HIPSOLVER_ENABLE_HIPSPARSE_TESTS)
    set_source_files_properties(client.cpp PROPERTIES COMPILE_DEFINITIONS HAVE_HIPSPARSE)
    target_link_libraries(hipsolver-bench PRIVATE roc::hipsparse)
endif()

if(HIPSOLVER_ENABLE_FORTRAN)
    target_link_libraries(hipsolver-bench PRIVATE hipsolver_fortran_client)
endif()

# --- ASAN flags ---

if(HIPSOLVER_ENABLE_ASAN)
    target_compile_options(hipsolver-bench PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(hipsolver-bench PRIVATE -fsanitize=address -shared-libasan -fuse-ld=lld)
endif()

# --- Armor ---

add_armor_flags(hipsolver-bench "${HIPSOLVER_ARMOR_LEVEL}")

# --- Platform-specific linking ---

if(NOT HIPSOLVER_ENABLE_CUDA)
    target_link_libraries(hipsolver-bench PRIVATE hip::host)

    if(CUSTOM_TARGET)
        target_link_libraries(hipsolver-bench PRIVATE hip::${CUSTOM_TARGET})
    endif()

    if(UNIX AND CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
        target_link_libraries(hipsolver-bench PRIVATE -lpthread -lm)
    endif()
else()
    target_compile_definitions(hipsolver-bench PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(hipsolver-bench PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(hipsolver-bench PRIVATE ${CUDA_LIBRARIES} Threads::Threads)
endif()

# --- Output and install ---

set_target_properties(hipsolver-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

rocm_install(TARGETS hipsolver-bench COMPONENT benchmarks)
