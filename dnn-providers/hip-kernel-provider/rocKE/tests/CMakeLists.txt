# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# By-layer C++ test registration for the rocKE engine. Included via
# add_subdirectory(tests) from the top-level rocKE/CMakeLists.txt. All targets
# link the engine archive ckc_core (built by the parent). Paths are relative to
# this file (tests/), keeping the tree copy-able.

# --- core layer ---------------------------------------------------------------
# Optional smoke test: cmake --build build --target ckc_smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/smoke.cpp")
  add_executable(ckc_smoke core/smoke.cpp)
  target_link_libraries(ckc_smoke PRIVATE ckc_core)
endif()

if(BUILD_TESTING)
  # ck.dsl.ir/v1 serializer round-trip + verifier test. Proves
  # lower(parse(serialize(k))) == lower(k) byte-for-byte, serialization
  # idempotence, and a clean verifier pass for the buildable GEMM configs.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/ir_serialize_roundtrip.cpp")
    add_executable(ckc_ir_serialize_roundtrip core/ir_serialize_roundtrip.cpp)
    target_link_libraries(ckc_ir_serialize_roundtrip PRIVATE ckc_core)
    if(UNIX)
      target_link_libraries(ckc_ir_serialize_roundtrip PRIVATE m)
    endif()
    add_test(NAME ckc_ir_serialize_roundtrip
             COMMAND ckc_ir_serialize_roundtrip)
    set_tests_properties(ckc_ir_serialize_roundtrip
                         PROPERTIES LABELS "ckc;unit;host")
    install(TARGETS ckc_ir_serialize_roundtrip
            RUNTIME DESTINATION ${CKC_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY CKC_INSTALL_TEST_TARGETS
                 ckc_ir_serialize_roundtrip)
  endif()

  # --- instances layer --------------------------------------------------------
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/tiled_attention_2d_reentrancy.cpp")
    add_executable(ckc_tiled_attention_2d_reentrancy
                   instances/tiled_attention_2d_reentrancy.cpp)
    target_link_libraries(ckc_tiled_attention_2d_reentrancy PRIVATE ckc_core)
    if(UNIX)
      target_link_libraries(ckc_tiled_attention_2d_reentrancy PRIVATE m)
    endif()
    add_test(NAME ckc_tiled_attention_2d_reentrancy
             COMMAND ckc_tiled_attention_2d_reentrancy)
    set_tests_properties(ckc_tiled_attention_2d_reentrancy
                         PROPERTIES LABELS "ckc;unit;host")
    install(TARGETS ckc_tiled_attention_2d_reentrancy
            RUNTIME DESTINATION ${CKC_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY CKC_INSTALL_TEST_TARGETS
                 ckc_tiled_attention_2d_reentrancy)
  endif()
endif()
