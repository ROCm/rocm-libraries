# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Create common helper target for all clients
add_library(hipsparse-clients-common STATIC)
add_library(hipsparse::clients-common ALIAS hipsparse-clients-common)

target_compile_features(hipsparse-clients-common PUBLIC cxx_std_17)

target_sources(hipsparse-clients-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/arg_check.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_parse_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_template_specialization.cpp"
)

target_include_directories(hipsparse-clients-common PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
)

target_compile_options(hipsparse-clients-common PUBLIC -Wall)

target_compile_definitions(hipsparse-clients-common PRIVATE GOOGLE_TEST)

# GTest is required for arg_check.cpp and unit.cpp
find_package(GTest REQUIRED)

target_link_libraries(hipsparse-clients-common 
    PUBLIC 
        roc::hipsparse
    PRIVATE
        GTest::gtest
)

if(NOT WIN32)
    target_link_libraries(hipsparse-clients-common PRIVATE stdc++fs)
endif()

if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-clients-common PUBLIC __HIP_PLATFORM_NVIDIA__)
    target_link_libraries(hipsparse-clients-common PUBLIC hip::host CUDA::cudart CUDA::cuda_driver)
elseif(HIPSPARSE_ENABLE_HIP)
    target_link_libraries(hipsparse-clients-common PUBLIC hip::host)
endif()

if(HIPSPARSE_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    set_target_properties(hipsparse-clients-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


# Lightweight STATIC library for samples only
add_library(hipsparse-samples-common STATIC)
add_library(hipsparse::samples-common ALIAS hipsparse-samples-common)

target_compile_features(hipsparse-samples-common PUBLIC cxx_std_17)

target_sources(hipsparse-samples-common PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

target_include_directories(hipsparse-samples-common PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
)

target_link_libraries(hipsparse-samples-common PUBLIC roc::hipsparse)

if(NOT WIN32)
    target_link_libraries(hipsparse-samples-common PRIVATE stdc++fs)
endif()

if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-samples-common PUBLIC __HIP_PLATFORM_NVIDIA__)
    target_link_libraries(hipsparse-samples-common PUBLIC hip::host CUDA::cudart CUDA::cuda_driver)
elseif(HIPSPARSE_ENABLE_HIP)
    target_link_libraries(hipsparse-samples-common PUBLIC hip::host)
endif()

if(HIPSPARSE_BUILD_SHARED_LIBS OR BUILD_SHARED_LIBS)
    set_target_properties(hipsparse-samples-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


# Interface library for Fortran samples - reduces code repetition
# Links against samples-common and adds Fortran-specific C++ runtime dependencies
if(HIPSPARSE_ENABLE_FORTRAN AND HIPSPARSE_ENABLE_HOST)
    add_library(hipsparse-fortran-samples-common INTERFACE)
    add_library(hipsparse::fortran-samples-common ALIAS hipsparse-fortran-samples-common)
    
    target_link_libraries(hipsparse-fortran-samples-common 
        INTERFACE 
            hipsparse::samples-common
    )
    
    # Fortran linking C++: Need explicit C++ runtime dependencies
    if(NOT WIN32)
        target_link_libraries(hipsparse-fortran-samples-common 
            INTERFACE 
                m 
                ${CMAKE_DL_LIBS}
        )
    endif()
endif()
