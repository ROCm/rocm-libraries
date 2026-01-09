# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Common test main entry point
set(TEST_MAIN
    StinkyTofuTestMain.cpp
)
list(TRANSFORM TEST_MAIN PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Common include directories for all tests
set(TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipblaslt/tensilelite/rocisa/rocisa/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../hardware/include
    ${TABLEGEN_OUTPUT_DIR}
)

# Common link libraries for all tests
set(TEST_LINK_LIBS
    GTest::gtest
    Threads::Threads
    stinkytofu
)

# =============================================================================
# Unit Tests
# =============================================================================
set(UNIT_TEST_FILES
    unit/PassManagerTest.cpp
    unit/IRBuilderTest.cpp
    unit/IRModuleTest.cpp
    unit/IRToAsmPipelineTest.cpp
    unit/IntrusiveListTest.cpp
    unit/ConfigurableWaitCntPassTest.cpp
    unit/StinkyAsmEmitterTest.cpp
    unit/VirtualRegisterTest.cpp
    unit/ErrorHandlingTest.cpp
    unit/PeepholeOptimizationPassTest.cpp
    unit/DeadCodeEliminationPassTest.cpp
    unit/DuplicateEliminationPassTest.cpp
    unit/DelayAluInsertionPassTest.cpp
    unit/OptimizationPipelineTest.cpp
    unit/IntrinsicFlowTest.cpp
    unit/HexLiteralTest.cpp
)
list(TRANSFORM UNIT_TEST_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_executable(unit_tests ${TEST_MAIN} ${UNIT_TEST_FILES})
target_link_libraries(unit_tests PUBLIC ${TEST_LINK_LIBS})
target_include_directories(unit_tests PRIVATE ${TEST_INCLUDE_DIRS})
add_dependencies(unit_tests tablegen_generated)

gtest_discover_tests(unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    PROPERTIES LABELS "unit_tests"
)

# =============================================================================
# All Tests Target (convenience target to build all test executables)
# =============================================================================
add_custom_target(tests)
add_dependencies(tests unit_tests)

# Note: Future test executables can be added here, e.g.:
# add_executable(integration_tests ...)
# add_dependencies(tests integration_tests)
