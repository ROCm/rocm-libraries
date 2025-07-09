# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

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
        ${msgpack_libs}
)

target_sources(rocroller-tests-catch
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/CustomAssertions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestKernels.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddDeallocateTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AddPrefetchTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AliasDataFlowTagsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AnnotateInstructionsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ArgumentTracerTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/BinaryExpressionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/BranchGeneratorTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CommandArgumentValueTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CommandTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ConnectWorkgroupsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CustomAssertionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CustomAssertions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CustomMatcherTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypesTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExpressionContainsSubExpressionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExpressionIdenticalTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExpressionSerializationTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExpressionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExpressionToStringTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExpressionTransformationTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FastArithmeticTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FastDivisionTransformationTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GlobalLoadStoreTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HazardObserverTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HypergraphTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/IdentifyParallelDimensionsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/InstructionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelArgumentsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraphColouringTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraphConstraintsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraphRemoveDuplicatesTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelGraphSimplifyTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LDSAllocationTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LargerLDSTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LoadPackedTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MEMObserverTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MultipleLDSAllocationTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RandomNumberExpressionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RandomNumberGenerationTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RegisterTagManagerTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RegisterTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ScalarExpressionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SettingsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SubDwordExpressionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TernaryExpressionTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestContext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestContextTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestKernels.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TransposeLoadsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UnrollLoopsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UtilsTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/WMMAHazardObserverTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/WMMAObserverUnitTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/WidenAddrExprTo64bitTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/mxDataGeneratorTest.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ScaleUtilsTest.cpp"
)
