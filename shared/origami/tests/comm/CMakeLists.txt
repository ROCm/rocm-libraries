# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# origami::comm byte-identity suite.
#
# Links the compiled roc::origami-comm library (no HIP), so this runner builds
# and runs without a ROCm toolchain. Catch2 is provided by the parent
# tests/CMakeLists.txt (find_package or FetchContent).

add_executable(origami-comm-tests)

target_sources(
    origami-comm-tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test_types.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_hardware.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_heuristics.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_primitives.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_algorithms.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_latency.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_collective.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_byte_identity.cpp"
)

target_link_libraries(
    origami-comm-tests PRIVATE roc::origami-comm Catch2::Catch2WithMain
)

# Byte-identity corpora are read at runtime relative to this directory.
catch_discover_tests(origami-comm-tests
    DISCOVERY_MODE PRE_TEST
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
