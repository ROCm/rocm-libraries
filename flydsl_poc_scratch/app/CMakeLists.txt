# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Standalone app that drives hipDNN's public API and proves flyDSL kernel selection.
# Links hipdnn_frontend (INTERFACE -> pulls hipdnn_backend + hipdnn_data_sdk). Does NOT
# link the provider; the provider .so is dlopen'd at runtime from HIPDNN_PLUGIN_DIR.

cmake_minimum_required(VERSION 3.21)
project(flydsl_hipdnn_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(hip REQUIRED)
find_package(Threads REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)

add_executable(flydsl_hipdnn_app flydsl_hipdnn_app.cpp)
target_link_libraries(flydsl_hipdnn_app
    PRIVATE
        hip::host
        Threads::Threads
        hipdnn_frontend)
