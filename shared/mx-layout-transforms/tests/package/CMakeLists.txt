# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCMxLayoutTransformsPackageTest LANGUAGES CXX)

enable_testing()
find_package(ROCMxLayoutTransforms CONFIG REQUIRED)

add_executable(mx-layout-transforms-package-test main.cpp)
target_link_libraries(
    mx-layout-transforms-package-test
    PRIVATE
        roc::mx-layout-transforms
)
add_test(NAME mx-layout-transforms-package-consumer COMMAND mx-layout-transforms-package-test)
