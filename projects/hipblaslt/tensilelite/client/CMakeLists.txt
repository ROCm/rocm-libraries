# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT


find_package(Boost COMPONENTS program_options filesystem REQUIRED)

add_executable(tensilelite-client "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

add_executable(cpu-gemm-driver "${CMAKE_CURRENT_SOURCE_DIR}/cpu_gemm_driver.cpp")

set(COMMON_CLIENT_LIBS
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
    list(APPEND COMMON_CLIENT_LIBS rocm_smi64)
endif()

foreach(target tensilelite-client cpu-gemm-driver)
    target_link_libraries(${target} PRIVATE ${COMMON_CLIENT_LIBS})

    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_definitions(${target} PRIVATE TENSILE_DEFAULT_SERIALIZATION)

    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    if(HIPBLASLT_ENABLE_ASAN)
        hipblaslt_target_configure_sanitizers(${target} PRIVATE)
    endif()
endforeach()

add_subdirectory(include)
add_subdirectory(src)

