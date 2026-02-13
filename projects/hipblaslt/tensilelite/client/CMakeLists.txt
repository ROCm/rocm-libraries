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

if(TENSILELITE_ENABLE_ROCROLLER)
    if(NOT ROCM_LIBS_SUPERBUILD)
        if(HIPBLASLT_ENABLE_THEROCK)
            find_package(mxDataGenerator REQUIRED)
        else()
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../shared/mxdatagenerator"
                           "${CMAKE_CURRENT_BINARY_DIR}/mxdatagenerator")
        endif()
    endif()
    target_compile_definitions(tensilelite-client PRIVATE HIPBLASLT_USE_ROCROLLER)
    target_link_libraries(tensilelite-client PRIVATE roc::mxDataGenerator)
    target_compile_features(tensilelite-client PRIVATE cxx_std_20)
endif()

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

if(NOT TENSILELITE_ENABLE_ROCROLLER)
    set_target_properties(tensilelite-client
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(tensilelite-client PRIVATE)
endif()

add_subdirectory(include)
add_subdirectory(src)
