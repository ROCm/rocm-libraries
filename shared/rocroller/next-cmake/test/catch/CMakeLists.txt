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

find_package(Catch2)
if(NOT Catch2_FOUND)
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

target_link_libraries(rocroller-tests-catch
    PRIVATE
        rocroller::common-test-utilities
        Catch2::Catch2WithMain
        ${BLAS_LIBRARIES}
)

set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_TEST_DIR}/catch")
target_sources(rocroller-tests-catch
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/CustomAssertions.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TestKernels.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddPrefetchTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AliasDataFlowTagsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AnnotateInstructionsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ArgumentTracerTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BinaryExpressionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BranchGeneratorTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CommandArgumentValueTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CommandTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ConnectWorkgroupsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CustomAssertionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CustomAssertions.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CustomMatcherTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypesTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionContainsSubExpressionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionIdenticalTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionSerializationTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionToStringTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionTransformationTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/FastArithmeticTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/FastDivisionTransformationTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GlobalLoadStoreTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HazardObserverTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HypergraphTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/IdentifyParallelDimensionsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/InstructionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelArgumentsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelGraphColouringTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelGraphConstraintsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelGraphRemoveDuplicatesTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelGraphSimplifyTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LDSAllocationTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LargerLDSTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LoadPackedTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MEMObserverTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MultipleLDSAllocationTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RandomNumberExpressionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RandomNumberGenerationTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RegisterTagManagerTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RegisterTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScalarExpressionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SettingsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SubDwordExpressionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TernaryExpressionTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TestContext.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TestContextTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TestKernels.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TransposeLoadsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UnrollLoopsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UtilsTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WMMAHazardObserverTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WMMAObserverUnitTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WidenAddrExprTo64bitTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/mxDataGeneratorTest.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScaleUtilsTest.cpp"
)
