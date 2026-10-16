# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# HIP transport shared by the hipBLASLt and TensileLite clients. The numerical
# algorithm belongs to HostNumerics; this target has no dependency on either
# client's argument, allocation, reference, or stress-test infrastructure.
add_library(hipblaslt-client-freivalds STATIC FreivaldsDevice.cpp)
target_include_directories(hipblaslt-client-freivalds PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(hipblaslt-client-freivalds
    PUBLIC roc::hostnumerics hip::host
    PRIVATE hip::device)
if(COMMAND hipblaslt_target_configure_sanitizers)
    hipblaslt_target_configure_sanitizers(hipblaslt-client-freivalds PUBLIC)
endif()
