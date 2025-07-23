# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Boost COMPONENTS program_options filesystem REQUIRED)
add_executable(tensilelite-client "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(tensilelite-client
    PRIVATE
        hip::device
        hip::host
        tensilelite::tensilelite-host
        rocisa::rocisa-cpp
        Boost::program_options
        Boost::filesystem
        OpenMP::OpenMP_CXX
)
if(NOT WIN32)
    find_package(rocm_smi REQUIRED)
else()
    find_package(rocm_smi)
endif()
if(rocm_smi_FOUND)
    target_link_libraries(tensilelite-client PRIVATE rocm_smi64)
endif()
target_include_directories(tensilelite-client PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(tensilelite-client PRIVATE TENSILE_DEFAULT_SERIALIZATION)

set_target_properties(tensilelite-client
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(tensilelite-client PRIVATE)
endif()

add_subdirectory(include)
add_subdirectory(src)
