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
  # Build-tree artifact-reproduction gate, deliberately separate from pytest.
  # It re-probes the full intrinsic/architecture matrix and therefore has a
  # different timeout, resource, and skip contract from Python unit tests. The
  # installed test manifest registers the same logical gate against its staged
  # generator and relocated package data.
  if(Python3_Interpreter_FOUND)
    add_test(
      NAME rocke_arch_domain_reproducibility
      COMMAND "${Python3_EXECUTABLE}" -u
              "${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_arch_domain.py"
              --check --unvalidated-exit-code 77)
    set_tests_properties(
      rocke_arch_domain_reproducibility
      PROPERTIES
        LABELS "ckc;python;host;integration;toolchain;generated-artifact;standard"
        RUN_SERIAL TRUE
        SKIP_RETURN_CODE 77
        TIMEOUT 1800
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")
  else()
    message(WARNING
            "rocKE: Python3 interpreter not found; arch-domain reproducibility "
            "CTest is not registered")
  endif()

  add_executable(rocke_optimization_barrier core/test_optimization_barrier.cpp)
  target_link_libraries(rocke_optimization_barrier PRIVATE rocke_core)
  add_test(NAME rocke_optimization_barrier COMMAND rocke_optimization_barrier)
  set_tests_properties(rocke_optimization_barrier
                      PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name
                              rocke_optimization_barrier)
  set_target_properties(rocke_optimization_barrier
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_optimization_barrier
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_optimization_barrier)

  add_executable(rocke_tf32 core/test_tf32.cpp)
  target_link_libraries(rocke_tf32 PRIVATE rocke_core)
  add_test(NAME rocke_tf32 COMMAND rocke_tf32)
  set_tests_properties(rocke_tf32 PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_tf32)
  set_target_properties(rocke_tf32 PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_tf32 RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_tf32)

  add_executable(rocke_fp8_scale_conversion_gate core/test_fp8_scale_conversion_gate.cpp)
  target_link_libraries(rocke_fp8_scale_conversion_gate PRIVATE rocke_core)
  add_test(NAME rocke_fp8_scale_conversion_gate COMMAND rocke_fp8_scale_conversion_gate)
  set_tests_properties(rocke_fp8_scale_conversion_gate PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_fp8_scale_conversion_gate)
  set_target_properties(rocke_fp8_scale_conversion_gate
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_fp8_scale_conversion_gate
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_fp8_scale_conversion_gate)

  add_executable(rocke_storage core/test_storage.cpp)
  target_link_libraries(rocke_storage PRIVATE rocke_core)
  add_test(NAME rocke_storage COMMAND rocke_storage)
  set_tests_properties(rocke_storage PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_storage)
  set_target_properties(rocke_storage PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_storage RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_storage)

  add_executable(rocke_ir_types core/test_ir_types.cpp)
  target_link_libraries(rocke_ir_types PRIVATE rocke_core)
  add_test(NAME rocke_ir_types COMMAND rocke_ir_types)
  set_tests_properties(rocke_ir_types PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_ir_types)
  set_target_properties(rocke_ir_types PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_ir_types RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_ir_types)

  add_executable(rocke_dtypes core/test_dtypes.cpp)
  target_link_libraries(rocke_dtypes PRIVATE rocke_core)
  add_test(NAME rocke_dtypes COMMAND rocke_dtypes)
  set_tests_properties(rocke_dtypes PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_dtypes)
  set_target_properties(rocke_dtypes
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_dtypes
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_dtypes)

  add_executable(rocke_mma_catalog_lookup core/test_mma_catalog_lookup.cpp)
  find_package(Threads REQUIRED)
  target_link_libraries(rocke_mma_catalog_lookup PRIVATE rocke_core Threads::Threads)
  target_include_directories(rocke_mma_catalog_lookup PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/core/arch")
  add_test(NAME rocke_mma_catalog_lookup COMMAND rocke_mma_catalog_lookup)
  set_tests_properties(rocke_mma_catalog_lookup PROPERTIES LABELS
    "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_mma_catalog_lookup)
  set_target_properties(rocke_mma_catalog_lookup
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_mma_catalog_lookup
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_mma_catalog_lookup)

  add_executable(rocke_arch_target_identity core/test_arch_target_identity.cpp)
  target_link_libraries(rocke_arch_target_identity PRIVATE rocke_core)
  add_test(NAME rocke_arch_target_identity COMMAND rocke_arch_target_identity)
  set_tests_properties(rocke_arch_target_identity PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_arch_target_identity)
  set_target_properties(rocke_arch_target_identity
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_arch_target_identity
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_arch_target_identity)

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

  # The C++ engine's arch-domain lane -- mode parsing, the message, and the
  # refusal itself -- without the rocke_engine extension.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_arch_domain_refusal.cpp")
    add_executable(rocke_arch_domain_refusal core/test_arch_domain_refusal.cpp)
    target_link_libraries(rocke_arch_domain_refusal PRIVATE rocke_core)
    add_test(NAME rocke_arch_domain_refusal COMMAND rocke_arch_domain_refusal)
    set_tests_properties(rocke_arch_domain_refusal
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_arch_domain_refusal)
    set_target_properties(rocke_arch_domain_refusal
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_arch_domain_refusal
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_arch_domain_refusal)
  endif()

  # QkScaleSpec validation in the C++ engine: the Python messages, bad value included.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_qk_scale_spec_validate.cpp")
    add_executable(rocke_qk_scale_spec_validate core/test_qk_scale_spec_validate.cpp)
    target_link_libraries(rocke_qk_scale_spec_validate PRIVATE rocke_core)
    add_test(NAME rocke_qk_scale_spec_validate
             COMMAND rocke_qk_scale_spec_validate)
    set_tests_properties(rocke_qk_scale_spec_validate
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_qk_scale_spec_validate)
    set_target_properties(rocke_qk_scale_spec_validate
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_qk_scale_spec_validate
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_qk_scale_spec_validate)
  endif()

  # temporal_hint (rocke_mem_opts_t) on global_load_vN/global_store_vN and the
  # io helpers, through the C++ builders and HIP/LLVM lowerers.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_nontemporal_hip.cpp")
    add_executable(rocke_nontemporal_hip core/test_nontemporal_hip.cpp)
    target_link_libraries(rocke_nontemporal_hip PRIVATE rocke_core)
    add_test(NAME rocke_nontemporal_hip COMMAND rocke_nontemporal_hip)
    set_tests_properties(rocke_nontemporal_hip PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_nontemporal_hip)
    set_target_properties(rocke_nontemporal_hip
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_nontemporal_hip
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_nontemporal_hip)
  endif()

  # The buffer descriptor word (DWORD3) the HIP lowerer emits per target, and
  # its agreement with the LLVM lowerer's word.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_buffer_rsrc_hip.cpp")
    add_executable(rocke_buffer_rsrc_hip core/test_buffer_rsrc_hip.cpp)
    target_link_libraries(rocke_buffer_rsrc_hip PRIVATE rocke_core)
    add_test(NAME rocke_buffer_rsrc_hip COMMAND rocke_buffer_rsrc_hip)
    set_tests_properties(rocke_buffer_rsrc_hip PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_buffer_rsrc_hip)
    set_target_properties(rocke_buffer_rsrc_hip
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_buffer_rsrc_hip
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_buffer_rsrc_hip)
  endif()

  # A buffer descriptor refused as a loop-carried value by the C++ builder and
  # verifier.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_loop_carried_descriptor.cpp")
    add_executable(rocke_loop_carried_descriptor core/test_loop_carried_descriptor.cpp)
    target_link_libraries(rocke_loop_carried_descriptor PRIVATE rocke_core)
    add_test(NAME rocke_loop_carried_descriptor COMMAND rocke_loop_carried_descriptor)
    set_tests_properties(rocke_loop_carried_descriptor PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_loop_carried_descriptor)
    set_target_properties(rocke_loop_carried_descriptor
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_loop_carried_descriptor
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_loop_carried_descriptor)
  endif()

  # Direct f32 tanh lowering and narrow-type rejection in the C++ engine.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_tanh_lowering.cpp")
    add_executable(rocke_tanh_lowering core/test_tanh_lowering.cpp)
    target_link_libraries(rocke_tanh_lowering PRIVATE rocke_core)
    add_test(NAME rocke_tanh_lowering COMMAND rocke_tanh_lowering)
    set_tests_properties(rocke_tanh_lowering
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_tanh_lowering)
    set_target_properties(rocke_tanh_lowering
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_tanh_lowering
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_tanh_lowering)
  endif()

  # Fenced barriers (fence attr of tile.sync / sync_lds_only / s_barrier_bare).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_barrier_fence.cpp")
    add_executable(rocke_barrier_fence core/test_barrier_fence.cpp)
    target_link_libraries(rocke_barrier_fence PRIVATE rocke_core)
    add_test(NAME rocke_barrier_fence COMMAND rocke_barrier_fence)
    set_tests_properties(rocke_barrier_fence
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_barrier_fence)
    set_target_properties(rocke_barrier_fence
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_barrier_fence
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_barrier_fence)
  endif()

  # BF16x3 expansion of the emulated XF32 atoms, lowered from serialized IR.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_xf32_bf16x3_lowering.cpp")
    add_executable(rocke_xf32_bf16x3_lowering core/test_xf32_bf16x3_lowering.cpp)
    target_link_libraries(rocke_xf32_bf16x3_lowering PRIVATE rocke_core)
    add_test(NAME rocke_xf32_bf16x3_lowering COMMAND rocke_xf32_bf16x3_lowering)
    set_tests_properties(rocke_xf32_bf16x3_lowering
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_xf32_bf16x3_lowering)
    set_target_properties(rocke_xf32_bf16x3_lowering
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_xf32_bf16x3_lowering
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_xf32_bf16x3_lowering)
  endif()

  # Unsigned compare, no-unsigned-wrap add and range marker: C API builders,
  # both C++ lowerers, the verifier and the constant folder.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_unsigned_int_ops.cpp")
    add_executable(rocke_unsigned_int_ops core/test_unsigned_int_ops.cpp)
    target_link_libraries(rocke_unsigned_int_ops PRIVATE rocke_core)
    add_test(NAME rocke_unsigned_int_ops COMMAND rocke_unsigned_int_ops)
    set_tests_properties(rocke_unsigned_int_ops
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_unsigned_int_ops)
    set_target_properties(rocke_unsigned_int_ops
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_unsigned_int_ops
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_unsigned_int_ops)
  endif()

  # f64 op rejection and f64 vector arithmetic in both C++ lowerers.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_f64_lowering.cpp")
    add_executable(rocke_f64_lowering core/test_f64_lowering.cpp)
    target_link_libraries(rocke_f64_lowering PRIVATE rocke_core)
    add_test(NAME rocke_f64_lowering COMMAND rocke_f64_lowering)
    set_tests_properties(rocke_f64_lowering
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_f64_lowering)
    set_target_properties(rocke_f64_lowering
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_f64_lowering
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_f64_lowering)
  endif()

  # Pins every opcode's value, name and purity: rocke_opcode_t is C ABI.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_opcode_values.cpp")
    add_executable(rocke_opcode_values core/test_opcode_values.cpp)
    target_link_libraries(rocke_opcode_values PRIVATE rocke_core)
    add_test(NAME rocke_opcode_values COMMAND rocke_opcode_values)
    set_tests_properties(rocke_opcode_values
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_opcode_values)
    set_target_properties(rocke_opcode_values
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_opcode_values
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_opcode_values)
  endif()

  # The E8M0 scale decoder reads its byte unsigned in both C++ lowerers.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_mx_scale_decode.cpp")
    add_executable(rocke_mx_scale_decode core/test_mx_scale_decode.cpp)
    target_link_libraries(rocke_mx_scale_decode PRIVATE rocke_core)
    add_test(NAME rocke_mx_scale_decode COMMAND rocke_mx_scale_decode)
    set_tests_properties(rocke_mx_scale_decode
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_mx_scale_decode)
    set_target_properties(rocke_mx_scale_decode
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_mx_scale_decode
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_mx_scale_decode)
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

  # --- portable-IR layer ------------------------------------------------------
  # Decoders + recipe VM for the replay path (cpp/portable_ir/). Both tests are
  # hermetic -- no artifact files, no Python -- so they run wherever the engine
  # builds. Wider coverage of the same code (every production kernel, both
  # archs, byte-compared against the Python lowerer) lives in the Python driver
  # python/rocke/portable_ir/drivers/parity_matrix.py, which needs a Python
  # front end and so cannot be a ctest.
  foreach(_pir_test dom_decoders recipe_vm_replay)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/portable_ir/${_pir_test}.cpp")
      set(_pir_target "rocke_portable_ir_${_pir_test}")
      add_executable(${_pir_target} portable_ir/${_pir_test}.cpp)
      target_link_libraries(${_pir_target} PRIVATE rocke_core)
      if(UNIX)
        target_link_libraries(${_pir_target} PRIVATE m)
      endif()
      add_test(NAME ${_pir_target} COMMAND ${_pir_target})
      set_tests_properties(${_pir_target} PROPERTIES LABELS "ckc;unit;host")
      rocke_cxx_test_installed_name(_rocke_installed_name ${_pir_target})
      set_target_properties(${_pir_target}
                            PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
      install(TARGETS ${_pir_target}
              RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
      set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS ${_pir_target})
    endif()
  endforeach()

  # Standalone artifact -> .ll tool. Not an add_test target: it needs an
  # artifact to chew on, and the tests that produce one are Python-side. Build
  # it with `cmake --build build --target rocke_portable_ir_replay_cli`.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/portable_ir/replay_cli.cpp")
    add_executable(rocke_portable_ir_replay_cli portable_ir/replay_cli.cpp)
    target_link_libraries(rocke_portable_ir_replay_cli PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_portable_ir_replay_cli PRIVATE m)
    endif()
    # Python recipe tests need the native VM in relocated test artifacts.
    install(TARGETS rocke_portable_ir_replay_cli
            RUNTIME DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/portable_ir")
    install(FILES "${ROCKE_BUILD_ID_MANIFEST}"
            DESTINATION "${ROCKE_TEST_INSTALL_DIR}/tests/portable_ir")
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

  # The three scalar-attention _lower_to_llvm wrappers take no target, so they
  # lower for a fixed one while the lowerer itself rejects a NULL arch. Nothing
  # else calls them, so this is the only thing that notices when they stop
  # working.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/unified_attention_scalar_lower_to_llvm.cpp")
    add_executable(rocke_unified_attention_scalar_lower_to_llvm
                   instances/unified_attention_scalar_lower_to_llvm.cpp)
    target_link_libraries(rocke_unified_attention_scalar_lower_to_llvm PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_unified_attention_scalar_lower_to_llvm PRIVATE m)
    endif()
    add_test(NAME rocke_unified_attention_scalar_lower_to_llvm
             COMMAND rocke_unified_attention_scalar_lower_to_llvm)
    set_tests_properties(rocke_unified_attention_scalar_lower_to_llvm
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name
                                  rocke_unified_attention_scalar_lower_to_llvm)
    set_target_properties(rocke_unified_attention_scalar_lower_to_llvm
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_unified_attention_scalar_lower_to_llvm
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_unified_attention_scalar_lower_to_llvm)
  endif()

  # Spec-validation parity for the two C++ fused-MoE mega builders: the
  # whole-kernel LDS budget, the scaled-f8f6f4 arch gate and the fp8 wave-size
  # gate. The byte-identity gate only compares specs both engines accept, so an
  # acceptance divergence from the Python validators is invisible to it.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/moe_fused_mega_spec_guards.cpp")
    add_executable(rocke_moe_fused_mega_spec_guards
                   instances/moe_fused_mega_spec_guards.cpp)
    target_link_libraries(rocke_moe_fused_mega_spec_guards PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_moe_fused_mega_spec_guards PRIVATE m)
    endif()
    add_test(NAME rocke_moe_fused_mega_spec_guards
             COMMAND rocke_moe_fused_mega_spec_guards)
    set_tests_properties(rocke_moe_fused_mega_spec_guards
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_moe_fused_mega_spec_guards)
    set_target_properties(rocke_moe_fused_mega_spec_guards
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_moe_fused_mega_spec_guards
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_moe_fused_mega_spec_guards)
  endif()

  # Host-side stream-K partition math: the launch grid it sizes is invisible to
  # the byte-identity gate, so the C port is pinned to the Python reference
  # vectors here (mirrors tests/instances/test_streamk_iter_partition.py).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/streamk_iter_partition.cpp")
    add_executable(rocke_streamk_iter_partition instances/streamk_iter_partition.cpp)
    target_link_libraries(rocke_streamk_iter_partition PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_streamk_iter_partition PRIVATE m)
    endif()
    add_test(NAME rocke_streamk_iter_partition COMMAND rocke_streamk_iter_partition)
    set_tests_properties(rocke_streamk_iter_partition PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_streamk_iter_partition)
    set_target_properties(rocke_streamk_iter_partition
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_streamk_iter_partition
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_streamk_iter_partition)
  endif()

  # Stream-K wgrad host helpers and validator gates (auto CTA pool per arch,
  # grid, workspace layout, reject reasons): also invisible to the byte-identity
  # gate, and the Python-vs-C pytest needs the pybind module CI does not build,
  # so the C port is pinned to Python reference vectors here.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/wgrad_streamk_host.cpp")
    add_executable(rocke_wgrad_streamk_host instances/wgrad_streamk_host.cpp)
    target_link_libraries(rocke_wgrad_streamk_host PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_wgrad_streamk_host PRIVATE m)
    endif()
    add_test(NAME rocke_wgrad_streamk_host COMMAND rocke_wgrad_streamk_host)
    set_tests_properties(rocke_wgrad_streamk_host PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_wgrad_streamk_host)
    set_target_properties(rocke_wgrad_streamk_host
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_wgrad_streamk_host
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_wgrad_streamk_host)
  endif()

  # RDNA split-KV WMMA decode host helpers (split count, pack groups, row
  # tiles, grids, workspace layout, spec resolution, kernel names, LDS budget):
  # the C API mirrors the Python launch plan for C callers, and the
  # byte-identity gate never sees them, so the C port is pinned to Python
  # reference vectors here.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/wmma_fmha_decode_host.cpp")
    add_executable(rocke_wmma_fmha_decode_host instances/wmma_fmha_decode_host.cpp)
    target_link_libraries(rocke_wmma_fmha_decode_host PRIVATE rocke_core)
    if(UNIX)
      target_link_libraries(rocke_wmma_fmha_decode_host PRIVATE m)
    endif()
    add_test(NAME rocke_wmma_fmha_decode_host COMMAND rocke_wmma_fmha_decode_host)
    set_tests_properties(rocke_wmma_fmha_decode_host PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_wmma_fmha_decode_host)
    set_target_properties(rocke_wmma_fmha_decode_host
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_wmma_fmha_decode_host
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_wmma_fmha_decode_host)
  endif()

  # Apply YAML-driven quick/standard/comprehensive/full tier labels to the
  # tests above. Must run in THIS directory scope: CMake (pre-3.28,
  # which this project's cmake_minimum_required predates) only allows
  # set_property(TEST ...)/set_tests_properties() on a test from the
  # directory that add_test()'d it, so calling this from a parent scope
  # (e.g. hip-kernel-provider/src/CMakeLists.txt) would silently do
  # nothing for them. Guarded because rocke/platform also builds
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
