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

find_package(GTest)
if(NOT GTest_FOUND)
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

find_package(mxDataGenerator)
if(NOT mxDataGenerator_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            mxDataGenerator
            GIT_REPOSITORY "https://github.com/ROCm/mxDataGenerator.git"
            GIT_TAG b99e329c644d771be3ca026f4df5bb635714bd40
        )
        FetchContent_MakeAvailable(mxDataGenerator)
    else()
        message(FATAL_ERROR "Failed to find mxDataGenerator")
    endif()
endif()

add_executable(rocroller-tests)
if(ROCROLLER_ENABLE_CATCH AND (ROCROLLER_BUILD_TESTING OR BUILD_TESTING))
    add_executable(rocroller-tests-catch)
endif()

find_package(BLAS REQUIRED)
find_package(cblas REQUIRED)
find_package(OpenMP REQUIRED)

if(ROCROLLER_ENABLE_YAML_CPP)
    target_compile_definitions(rocroller-tests PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
endif()

if(ROCROLLER_ENABLE_CATCH)
    add_subdirectory(catch)
endif()
add_subdirectory(common)
add_subdirectory(unit)

# Component stand-alone test
add_executable(component-test "${ROCROLLER_TEST_DIR}/standalone/ComponentTest.cpp")
target_link_libraries(component-test PRIVATE roc::rocroller ${msgpack_libs})

if(ROCROLLER_ENABLE_ARCH_GEN_TEST)
    find_package(Pytest REQUIRED) # requires pip install -r requirements.txt
    pytest_discover_tests(
        GEMMClientTests
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../client"
        ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}"
    )
    add_executable(arch-gen-tests "${ROCROLLER_TEST_DIR}/unit/GPUArchitectureGeneratorTest.cpp")
    target_include_directories(
        arch-gen-tests
            PRIVATE
                "${ROCROLLER_GPUARCH_DIR}/include"
    )
    target_link_libraries(
        arch-gen-tests
            PRIVATE
                roc::rocroller
                rocroller::common-test-utilities
                rocroller::isa-spec-manager
                GTest::gtest_main
                GTest::gmock
                ${BLAS_LIBRARIES}
    )
endif()
