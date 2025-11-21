# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Create common helper target for all clients
add_library(hipsparse-clients-common OBJECT)

# Compile features
target_compile_features(hipsparse-clients-common PRIVATE cxx_std_17)

# Add common sources
target_sources(hipsparse-clients-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/arg_check.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_parse_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_template_specialization.cpp"
)

# Common include directories
target_include_directories(hipsparse-clients-common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# Additional includes when building clients only
if(HIPSPARSE_BUILD_CLIENTS_ONLY)
    target_include_directories(hipsparse-clients-common PRIVATE ${hipsparse_INCLUDE_DIRS}/hipsparse)
endif()

# Common compile options
target_compile_options(hipsparse-clients-common PUBLIC
    -Wno-unused-command-line-argument
    -Wall
)

# Common compile definitions
target_compile_definitions(hipsparse-clients-common PUBLIC GOOGLE_TEST)

# GTest is required for arg_check.cpp and unit.cpp
find_package(GTest REQUIRED)

# Link common dependencies
target_link_libraries(hipsparse-clients-common PUBLIC 
    roc::hipsparse
    GTest::gtest
)

if(TARGET Threads::Threads)
    target_link_libraries(hipsparse-clients-common PUBLIC Threads::Threads)
endif()

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(hipsparse-clients-common PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link against stdc++fs for std::experimental::filesystem support
if(NOT WIN32)
    target_link_libraries(hipsparse-clients-common PUBLIC stdc++fs)
endif()

# Backend-specific settings
if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-clients-common PUBLIC __HIP_PLATFORM_NVIDIA__)
    target_include_directories(hipsparse-clients-common PUBLIC ${HIP_INCLUDE_DIRS})
    target_link_libraries(hipsparse-clients-common PUBLIC ${CUDA_LIBRARIES})
elseif(HIPSPARSE_ENABLE_HIP)
    target_link_libraries(hipsparse-clients-common PUBLIC hip::host)
endif()

# Enable POSITION_INDEPENDENT_CODE for OBJECT library if building shared libs
if(HIPSPARSE_BUILD_SHARED_LIBS)
    set_target_properties(hipsparse-clients-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Lightweight OBJECT library for samples only (contains only utility.cpp)
add_library(hipsparse-samples-common OBJECT)

target_compile_features(hipsparse-samples-common PUBLIC cxx_std_17)

target_sources(hipsparse-samples-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

target_include_directories(hipsparse-samples-common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_link_libraries(hipsparse-samples-common PUBLIC roc::hipsparse)

# Link against stdc++fs for std::experimental::filesystem support
if(NOT WIN32)
    target_link_libraries(hipsparse-samples-common PUBLIC stdc++fs)
endif()

if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-samples-common PUBLIC __HIP_PLATFORM_NVIDIA__)
    target_include_directories(hipsparse-samples-common PUBLIC ${HIP_INCLUDE_DIRS})
    target_link_libraries(hipsparse-samples-common PUBLIC ${CUDA_LIBRARIES})
elseif(HIPSPARSE_ENABLE_HIP)
    target_link_libraries(hipsparse-samples-common PUBLIC hip::host)
endif()

# Enable POSITION_INDEPENDENT_CODE for OBJECT library if building shared libs
if(HIPSPARSE_BUILD_SHARED_LIBS)
    set_target_properties(hipsparse-samples-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
