# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# --- C sample (C99 compatibility test) ---

add_executable(example-c-basic)
target_sources(example-c-basic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_basic.c"
)
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/example_basic.c"
    PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-xc -std=c99"
)
set_target_properties(example-c-basic PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

target_include_directories(example-c-basic SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
target_include_directories(example-c-basic PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
target_link_libraries(example-c-basic PRIVATE roc::hipsolver)

if(NOT HIPSOLVER_ENABLE_CUDA)
    target_link_libraries(example-c-basic PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(example-c-basic PRIVATE hip::${CUSTOM_TARGET})
    endif()
else()
    target_compile_definitions(example-c-basic PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(example-c-basic PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(example-c-basic PRIVATE ${CUDA_LIBRARIES})
endif()

# --- C++ sample (C++11 compatibility test) ---

add_executable(example-cpp-basic)
target_sources(example-cpp-basic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/example_basic.cpp"
)
set_target_properties(example-cpp-basic PROPERTIES
    CXX_STANDARD 11
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

target_include_directories(example-cpp-basic SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)
target_include_directories(example-cpp-basic PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)
target_link_libraries(example-cpp-basic PRIVATE roc::hipsolver)

if(NOT HIPSOLVER_ENABLE_CUDA)
    target_link_libraries(example-cpp-basic PRIVATE hip::host)
    if(CUSTOM_TARGET)
        target_link_libraries(example-cpp-basic PRIVATE hip::${CUSTOM_TARGET})
    endif()
else()
    target_compile_definitions(example-cpp-basic PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories(example-cpp-basic PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )
    target_link_libraries(example-cpp-basic PRIVATE ${CUDA_LIBRARIES})
endif()
