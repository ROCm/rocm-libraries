# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(tensilelite-client "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

add_executable(cpu-gemm-driver "${CMAKE_CURRENT_SOURCE_DIR}/cpu_gemm_driver.cpp")

set(COMMON_CLIENT_LIBS
    hip::device
    hip::host
    tensilelite::tensilelite-host
    rocisa::rocisa-cpp
    OpenMP::OpenMP_CXX
)

if(NOT WIN32)
    find_package(amd_smi REQUIRED)
else()
    find_package(amd_smi)
endif()

if(amd_smi_FOUND)
    list(APPEND COMMON_CLIENT_LIBS amd_smi)
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

option(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK "Enable in-process ROCprofiler-SDK counters" OFF)
if (TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)
  add_compile_definitions(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK=1)
  find_package(rocprofiler-sdk)

  if (rocprofiler-sdk_FOUND)
    target_link_libraries(tensilelite-client PRIVATE rocprofiler-sdk)
  else()
    message(FATAL_ERROR "ROCprofiler-SDK not found.")
  endif()
else()
  add_compile_definitions(TENSILELITE_CLIENT_ENABLE_ROCPROFSDK=0)
endif()

add_subdirectory(include)
add_subdirectory(src)
