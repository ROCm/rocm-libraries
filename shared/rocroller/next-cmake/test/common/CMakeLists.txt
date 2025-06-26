
################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_TEST_DIR}/common")

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

if(ROCROLLER_ENABLE_TIMERS)
    target_compile_definitions(common-test-utilities PUBLIC ROCROLLER_ENABLE_TIMERS)
    target_compile_options(common-test-utilities PUBLIC -pg -fno-omit-frame-pointer)
endif()

target_include_directories(common-test-utilities
    PUBLIC
        $<BUILD_INTERFACE:${_CMAKE_CURRENT_SOURCE_DIR}>
)

target_sources(common-test-utilities
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/CommonGraphs.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/CommonGraphs_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/F8Values.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/GEMMProblem.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/mxDataGen.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/SourceMatcher.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/TestValues.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/Utilities.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/common/WidenAddrExprTo64bit.hpp"
)

add_subdirectory(src)
