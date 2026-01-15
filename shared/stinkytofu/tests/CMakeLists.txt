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
    ${CMAKE_CURRENT_SOURCE_DIR}/unit  # For TestHelpers.hpp
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

# Assembly IR Tests
set(UNIT_ASM_TEST_FILES
    unit/asm/ConfigurableWaitCntPassTest.cpp
    unit/asm/DeadCodeEliminationPassTest.cpp
    unit/asm/DelayAluInsertionPassTest.cpp
    unit/asm/RedundantMovEliminationPassTest.cpp
    unit/asm/OptimizationPipelineTest.cpp
    unit/asm/PeepholeOptimizationPassTest.cpp
    unit/asm/StinkyAsmEmitterTest.cpp
    unit/asm/VirtualRegisterTest.cpp
)

# High-Level IR Tests
set(UNIT_LOGICAL_TEST_FILES
    unit/logical/LogicalCommutativeTest.cpp
    unit/logical/LogicalHexLiteralTest.cpp
    unit/logical/LogicalPeepholePassTest.cpp
    unit/logical/LogicalBuilderTest.cpp
    unit/logical/LogicalModuleTest.cpp
    unit/logical/LogicalModuleOwnershipTest.cpp
    unit/logical/LogicalOpcodeTest.cpp
    unit/logical/LogicalToAsmPipelineTest.cpp
    unit/logical/LogicalIntrinsicFlowTest.cpp
    unit/logical/LogicalPassManagerTest.cpp
)

# Common/Shared Tests
set(UNIT_COMMON_TEST_FILES
    unit/ErrorHandlingTest.cpp
    unit/IntrusiveListTest.cpp
    unit/ir/IntrinsicExpansionPassTest.cpp
)

# Combine all unit tests
set(UNIT_TEST_FILES
    ${UNIT_ASM_TEST_FILES}
    ${UNIT_LOGICAL_TEST_FILES}
    ${UNIT_COMMON_TEST_FILES}
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
