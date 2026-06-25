# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# By-layer C++ test registration for the rocKE engine. Included via
# add_subdirectory(tests) from the top-level rocKE/CMakeLists.txt. All targets
# link the engine archive rocke_core (built by the parent). Paths are relative to
# this file (tests/), keeping the tree copy-able.

# --- core layer ---------------------------------------------------------------
# Optional smoke test: cmake --build build --target rocke_smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/smoke.cpp")
  add_executable(rocke_smoke core/smoke.cpp)
  target_link_libraries(rocke_smoke PRIVATE rocke_core)
endif()

if(BUILD_TESTING)
  # ck.dsl.ir/v1 serializer round-trip + verifier test. Proves
  # lower(parse(serialize(k))) == lower(k) byte-for-byte, serialization
  # idempotence, and a clean verifier pass for the buildable GEMM configs.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/ir_serialize_roundtrip.cpp")
    add_executable(rocke_ir_serialize_roundtrip core/ir_serialize_roundtrip.cpp)
    target_link_libraries(rocke_ir_serialize_roundtrip PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_ir_serialize_roundtrip PRIVATE m)
    endif()
    add_test(NAME rocke_ir_serialize_roundtrip
             COMMAND rocke_ir_serialize_roundtrip)
    set_tests_properties(rocke_ir_serialize_roundtrip
                         PROPERTIES LABELS "ckc;unit;host")
    install(TARGETS rocke_ir_serialize_roundtrip
            RUNTIME DESTINATION ${ROCKE_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_ir_serialize_roundtrip)
  endif()

  # --- instances layer --------------------------------------------------------
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/tiled_attention_2d_reentrancy.cpp")
    add_executable(rocke_tiled_attention_2d_reentrancy
                   instances/tiled_attention_2d_reentrancy.cpp)
    target_link_libraries(rocke_tiled_attention_2d_reentrancy PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_tiled_attention_2d_reentrancy PRIVATE m)
    endif()
    add_test(NAME rocke_tiled_attention_2d_reentrancy
             COMMAND rocke_tiled_attention_2d_reentrancy)
    set_tests_properties(rocke_tiled_attention_2d_reentrancy
                         PROPERTIES LABELS "ckc;unit;host")
    install(TARGETS rocke_tiled_attention_2d_reentrancy
            RUNTIME DESTINATION ${ROCKE_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_tiled_attention_2d_reentrancy)
  endif()
endif()
