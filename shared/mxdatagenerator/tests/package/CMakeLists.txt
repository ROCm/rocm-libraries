# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(mxDataGeneratorCompatibilityPackageTest LANGUAGES NONE)

find_package(mxDataGenerator CONFIG REQUIRED)

if(NOT TARGET roc::mxDataGenerator)
    message(FATAL_ERROR "mxDataGenerator compatibility target was not imported.")
endif()

add_library(mxdatagenerator-compatibility-consumer INTERFACE)
target_link_libraries(
    mxdatagenerator-compatibility-consumer
    INTERFACE
        roc::mxDataGenerator
)
