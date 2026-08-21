# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# By-layer C++ test registration for the rocKE engine. Included via
# add_subdirectory(tests) from the top-level rocke/platform/CMakeLists.txt. All targets
# link the engine archive rocke_core (built by the parent). Paths are relative to
# this file (tests/), keeping the tree copy-able.

# --- core layer ---------------------------------------------------------------
# Optional smoke test: cmake --build build --target rocke_smoke
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/smoke.cpp")
  add_executable(rocke_smoke core/smoke.cpp)
  target_link_libraries(rocke_smoke PRIVATE rocke_core)
endif()

if(BUILD_TESTING)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/device_print.cpp")
    add_executable(rocke_device_print core/device_print.cpp)
    target_link_libraries(rocke_device_print PRIVATE rocke_core)
    add_test(NAME rocke_device_print COMMAND rocke_device_print)
    set_tests_properties(rocke_device_print
                         PROPERTIES LABELS "ckc;unit;host")
  endif()

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
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_ir_serialize_roundtrip)
    set_target_properties(rocke_ir_serialize_roundtrip
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_ir_serialize_roundtrip
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_ir_serialize_roundtrip)
  endif()

  # C++ parity for Python TestNewTargetIntrinsics.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/future_intrinsic_lowering.cpp")
    add_executable(rocke_future_intrinsic_lowering core/future_intrinsic_lowering.cpp)
    target_link_libraries(rocke_future_intrinsic_lowering PRIVATE rocke_core)
    add_test(NAME rocke_future_intrinsic_lowering
             COMMAND rocke_future_intrinsic_lowering)
    set_tests_properties(rocke_future_intrinsic_lowering
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_future_intrinsic_lowering)
    set_target_properties(rocke_future_intrinsic_lowering
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_future_intrinsic_lowering
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_future_intrinsic_lowering)
  endif()

  # Holds the ISA-layer and lower_llvm-layer flavor->datalayout tables together.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/flavor_datalayout_ssot.cpp")
    add_executable(rocke_flavor_datalayout_ssot core/flavor_datalayout_ssot.cpp)
    target_link_libraries(rocke_flavor_datalayout_ssot PRIVATE rocke_core)
    add_test(NAME rocke_flavor_datalayout_ssot
             COMMAND rocke_flavor_datalayout_ssot)
    set_tests_properties(rocke_flavor_datalayout_ssot
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_flavor_datalayout_ssot)
    set_target_properties(rocke_flavor_datalayout_ssot
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_flavor_datalayout_ssot
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_flavor_datalayout_ssot)
  endif()

  # Host unit test for the IR-layer MMA frag-length / accumulator-dtype tables
  # that rocke_b_mma consults to size a tile.mma result vector.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/mma_frag_ssot.cpp")
    add_executable(rocke_mma_frag_ssot core/mma_frag_ssot.cpp)
    target_link_libraries(rocke_mma_frag_ssot PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_mma_frag_ssot PRIVATE m)
    endif()
    add_test(NAME rocke_mma_frag_ssot
             COMMAND rocke_mma_frag_ssot)
    set_tests_properties(rocke_mma_frag_ssot
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_mma_frag_ssot)
    set_target_properties(rocke_mma_frag_ssot
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_mma_frag_ssot
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_mma_frag_ssot)
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
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_tiled_attention_2d_reentrancy)
    set_target_properties(rocke_tiled_attention_2d_reentrancy
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_tiled_attention_2d_reentrancy
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_tiled_attention_2d_reentrancy)
  endif()

  # Apply YAML-driven quick/standard/comprehensive/full tier labels to the
  # three tests above. Must run in THIS directory scope: CMake (pre-3.28,
  # which this project's cmake_minimum_required predates) only allows
  # set_property(TEST ...)/set_tests_properties() on a test from the
  # directory that add_test()'d it, so calling this from a parent scope
  # (e.g. hip-kernel-provider/src/CMakeLists.txt) would silently do
  # nothing for these three. Guarded because rocke/platform also builds
  # standalone (outside hip-kernel-provider), where neither the shared
  # parser (apply_ctest_category_labels) nor
  # ROCKE_ENGINE_CTEST_CATEGORIES_YAML/HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
  # exist. Without this, these tests carry only the hardcoded
  # "ckc;unit;host" label and silently drop out of `ctest -L <tier>`
  # selection -- the mechanism TheRock CI's per-tier
  # test_hipkernelprovider.py relies on.
  if(COMMAND apply_ctest_category_labels
     AND HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
     AND ROCKE_ENGINE_CTEST_CATEGORIES_YAML)
    apply_ctest_category_labels("${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}")
  endif()
endif()
