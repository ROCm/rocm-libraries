# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

add_executable(unittests)

set(UNITTESTS_FILES
    StinkyTofuTestMain.cpp
    PassManagerTest.cpp
    IntrusiveListTest.cpp
    ConfigurableWaitCntPassTest.cpp
)
list(TRANSFORM UNITTESTS_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

target_sources(unittests
    PRIVATE
        "${UNITTESTS_FILES}"
)

target_link_libraries(unittests
    PUBLIC
        GTest::gtest
        Threads::Threads
        stinkytofu
)

target_include_directories(unittests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipblaslt/tensilelite/rocisa/rocisa/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../hardware/include
        ${TABLEGEN_OUTPUT_DIR}
)

# Make unittests depend on the generated ISA definition files
add_dependencies(unittests tablegen_generated)

gtest_discover_tests(unittests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TIMEOUT 60)
