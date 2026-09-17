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

# M2: RMSNorm family (one HSACO per hidden size N), selected per-shape.
add_executable(flydsl_rmsnorm_app flydsl_rmsnorm_app.cpp)
target_link_libraries(flydsl_rmsnorm_app
    PRIVATE
        hip::host
        Threads::Threads
        hipdnn_frontend)

# M2 swap: the REAL FlyDSL build_rmsnorm_module bf16 family, N in {4096, 8192}.
add_executable(flydsl_rmsnorm_bf16_app flydsl_rmsnorm_bf16_app.cpp)
target_link_libraries(flydsl_rmsnorm_bf16_app
    PRIVATE
        hip::host
        Threads::Threads
        hipdnn_frontend)

# M3: the REAL FlyDSL flash-attention bf16 prefill family, per head-config {H=8, H=16}.
add_executable(flydsl_attention_bf16_app flydsl_attention_bf16_app.cpp)
target_link_libraries(flydsl_attention_bf16_app
    PRIVATE
        hip::host
        Threads::Threads
        hipdnn_frontend)

# M4 de-risk: enumerate ALL applicable SDPA engines (no pin) for the prefill shape.
add_executable(flydsl_attention_bakeoff_app flydsl_attention_bakeoff_app.cpp)
target_link_libraries(flydsl_attention_bakeoff_app
    PRIVATE
        hip::host
        Threads::Threads
        hipdnn_frontend)

# M4 execute+numerics: run EACH enumerated engine on the same non-causal shape, check output.
add_executable(flydsl_attention_bakeoff_exec_app flydsl_attention_bakeoff_exec_app.cpp)
target_link_libraries(flydsl_attention_bakeoff_exec_app
    PRIVATE
        hip::host
        Threads::Threads
        hipdnn_frontend)

# M4 autotune/bench: time EACH enumerated engine per shape, print ranked winner table.
add_executable(flydsl_attention_autotune_app flydsl_attention_autotune_app.cpp)
target_link_libraries(flydsl_attention_autotune_app
    PRIVATE
        hip::host
        Threads::Threads
        hipdnn_frontend)
