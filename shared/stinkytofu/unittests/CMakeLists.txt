# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest REQUIRED)

add_executable(unittests)

set(UNITTESTS_FILES
    StinkyTofuTestMain.cpp
    PassManagerTest.cpp
    IntrusiveListTest.cpp
)
list(TRANSFORM UNITTESTS_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

target_sources(unittests
    PRIVATE
        "${UNITTESTS_FILES}"
)

target_link_libraries(unittests
    PUBLIC
        GTest::gtest
        stinkytofu
)

target_include_directories(unittests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipblaslt/tensilelite/rocisa/rocisa/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

gtest_discover_tests(unittests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TIMEOUT 60)
