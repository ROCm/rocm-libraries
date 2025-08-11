# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(common-test-utilities OBJECT)
add_library(rocroller::common-test-utilities ALIAS common-test-utilities)

target_link_libraries(common-test-utilities
    PUBLIC
        roc::rocroller-interface
        roc::mxDataGenerator
        hip::host
        OpenMP::OpenMP_CXX
        ${CBLAS_LIBRARIES}
)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(common-test-utilities PRIVATE)
endif()

if(ROCROLLER_ENABLE_TIMERS)
    target_compile_definitions(common-test-utilities PUBLIC ROCROLLER_ENABLE_TIMERS)
    target_compile_options(common-test-utilities PUBLIC -pg -fno-omit-frame-pointer)
endif()

target_include_directories(common-test-utilities
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_sources(common-test-utilities
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/CommonGraphs.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/CommonGraphs_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/F8Values.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/GEMMProblem.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/mxDataGen.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/SourceMatcher.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/TestValues.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/Utilities.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/common/WidenAddrExprTo64bit.hpp"
)

add_subdirectory(src)
