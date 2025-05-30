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

include(CTest)

find_package(mxDataGenerator)
if(NOT mxDataGenerator_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            mxDataGenerator
            GIT_REPOSITORY "https://github.com/ROCm/mxDataGenerator.git"
            GIT_TAG e74b2c1b6f357c053b27958177ca454ba21918b7
        )
        FetchContent_MakeAvailable(mxDataGenerator)
    else()
        message(FATAL_ERROR "Failed to find mxDataGenerator")
    endif()
endif()

#set(CMAKE_FIND_DEBUG_MODE ON)
find_package(Catch2 REQUIRED)
if(catch2_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG devel
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        message(FATAL_ERROR "Failed to find Catch2")
    endif()
endif()

find_package(GTest REQUIRED)
if(gtest_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
        )
        FetchContent_MakeAvailable(googletest)
    else()
        message(FATAL_ERROR "Failed to find googletest")
    endif()
endif()

add_executable(rocroller-tests)
add_executable(rocroller-tests-catch)

find_package(BLAS REQUIRED)

if(ROCROLLER_ENABLE_YAML_CPP)
    target_compile_definitions(rocroller-tests PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
endif()


add_subdirectory(common)
add_subdirectory(catch)
add_subdirectory(unit)

target_link_libraries(rocroller-tests
    PRIVATE
        rocroller::common-test-utilities
        ${BLAS_LIBRARIES}
        GTest::gtest_main
        GTest::gmock
        libdivide::libdivide
        ${llvm_libs}
        ${llvm_yaml_libs}
        lldELF
)

target_link_libraries(rocroller-tests-catch
    PRIVATE
        rocroller::common-test-utilities
        Catch2::Catch2WithMain
        ${BLAS_LIBRARIES}
)
