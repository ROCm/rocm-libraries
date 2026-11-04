# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Core-layer C++ test registration for the rocKE engine. Included from the
# parent tests/CMakeLists.txt to preserve its CMake directory scope. All targets
# link the engine archive rocke_core (built by the parent). Paths are relative
# to tests/, keeping the tree copy-able.
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

    # The committed column of compiler_options.json for the loaded compiler,
    # re-measured. The installed manifest registers the same gate.
    add_test(
      NAME rocke_compiler_options_reproducibility
      COMMAND "${Python3_EXECUTABLE}" -u
              "${CMAKE_CURRENT_SOURCE_DIR}/../tools/gen_compiler_options.py"
              --check)
    set_tests_properties(
      rocke_compiler_options_reproducibility
      PROPERTIES
        LABELS "ckc;python;host;toolchain;generated-artifact;standard"
        TIMEOUT 600
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/..")

    # The arch-table parity test reads cpp/core/arch/data.cpp, which the
    # installed test tree does not carry, so CMakeLists.txt keeps it out of the
    # installed pytest cohort; run it here, from the source tree, instead.
    add_test(
      NAME rocke_arch_table_engine_parity
      COMMAND "${Python3_EXECUTABLE}" -m pytest
              "${CMAKE_CURRENT_SOURCE_DIR}/core/test_arch_table_engine_parity.py" -q)
    set_tests_properties(
      rocke_arch_table_engine_parity
      PROPERTIES
        LABELS "ckc;python;host"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python")

    # The GEMM spec-dict parity test reads cpp/bindings/rocke_engine.cpp, which
    # the installed test tree does not carry either; it checks that the binding
    # reads every key the Python spec dicts write, and only those, without
    # needing the optional rocke_engine build.
    add_test(
      NAME rocke_gemm_spec_dict_binding_parity
      COMMAND "${Python3_EXECUTABLE}" -m pytest
              "${CMAKE_CURRENT_SOURCE_DIR}/core/test_gemm_spec_dict_binding_parity.py" -q)
    set_tests_properties(
      rocke_gemm_spec_dict_binding_parity
      PROPERTIES
        LABELS "ckc;python;host"
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../python")
  else()
    message(WARNING
            "rocKE: Python3 interpreter not found; the Python-driven build-tree "
            "CTests (reproducibility gates and engine-parity checks) are not "
            "registered")
  endif()

  # The binding consumes Python-built IR; this binary exercises the independent
  # C builder as well, including compiler-dependent argument evaluation order.
  set_source_files_properties(instances/parity/optimization_barrier_emit.c
                              PROPERTIES LANGUAGE CXX)
  add_executable(rocke_optimization_barrier_emit
                 instances/parity/optimization_barrier_emit.c)
  target_link_libraries(rocke_optimization_barrier_emit PRIVATE rocke_core)
  # The supported CMake minimum predates language flags for a .c file.
  if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(rocke_optimization_barrier_emit PRIVATE /TP)
  else()
    target_compile_options(rocke_optimization_barrier_emit PRIVATE -x c++)
  endif()
  rocke_cxx_test_installed_name(_rocke_installed_name
                              rocke_optimization_barrier_emit)
  set_target_properties(rocke_optimization_barrier_emit
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_optimization_barrier_emit
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  if(Python3_Interpreter_FOUND)
    set(_rocke_core_tests "${CMAKE_CURRENT_SOURCE_DIR}/core")
    set(_rocke_python_root "${CMAKE_CURRENT_SOURCE_DIR}/../python")
    set(_rocke_parity_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/instances/parity")
    add_test(
      NAME rocke_optimization_barrier_hip_pair
      COMMAND "${Python3_EXECUTABLE}"
              "${_rocke_core_tests}/test_optimization_barrier_cli.py"
              --native "$<TARGET_FILE:rocke_optimization_barrier_emit>"
              --launcher
              "${_rocke_python_root}/rocke/tests/subprocess_launcher.py"
              --emitter
              "${_rocke_parity_dir}/optimization_barrier_emit.py"
              --count 17)
    set_tests_properties(rocke_optimization_barrier_hip_pair
                         PROPERTIES LABELS "ckc;python;host;quick" TIMEOUT 120)
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

  add_executable(rocke_f64_mma core/test_f64_mma.cpp)
  target_link_libraries(rocke_f64_mma PRIVATE rocke_core)
  target_include_directories(rocke_f64_mma PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/instances/gfx942")
  add_test(NAME rocke_f64_mma COMMAND rocke_f64_mma)
  set_tests_properties(rocke_f64_mma PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_f64_mma)
  set_target_properties(rocke_f64_mma PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_f64_mma RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_f64_mma)

  add_executable(rocke_tf32 core/test_tf32.cpp)
  target_link_libraries(rocke_tf32 PRIVATE rocke_core)
  target_include_directories(rocke_tf32 PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/instances/gfx942")
  add_test(NAME rocke_tf32 COMMAND rocke_tf32)
  set_tests_properties(rocke_tf32 PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_tf32)
  set_target_properties(rocke_tf32 PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_tf32 RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_tf32)

  add_executable(rocke_mfma_operand_contract core/test_mfma_operand_contract.cpp)
  target_link_libraries(rocke_mfma_operand_contract PRIVATE rocke_core)
  add_test(NAME rocke_mfma_operand_contract COMMAND rocke_mfma_operand_contract)
  set_tests_properties(rocke_mfma_operand_contract PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_mfma_operand_contract)
  set_target_properties(rocke_mfma_operand_contract
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_mfma_operand_contract
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_mfma_operand_contract)

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

  # The C++ copy of arch_specs.json's generic targets, compared with the file.
  # It reads the file from the source tree, so it runs from the build tree
  # only and is not installed.
  add_executable(rocke_arch_data_mirror core/arch_data_mirror.cpp)
  target_link_libraries(rocke_arch_data_mirror PRIVATE rocke_core)
  add_test(NAME rocke_arch_data_mirror
           COMMAND rocke_arch_data_mirror
                   "${CMAKE_CURRENT_SOURCE_DIR}/../python/rocke/core/arch/data")
  set_tests_properties(rocke_arch_data_mirror PROPERTIES LABELS "ckc;unit;host")

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

  # The buffer address space of the C++ tensor view: the Python error kind and
  # message for each refused call (tests/core/test_tensor_view_buffer_errors.py
  # pins the Python side).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_tensor_view_buffer_errors.cpp")
    add_executable(rocke_tensor_view_buffer_errors core/test_tensor_view_buffer_errors.cpp)
    target_link_libraries(rocke_tensor_view_buffer_errors PRIVATE rocke_core)
    add_test(NAME rocke_tensor_view_buffer_errors
             COMMAND rocke_tensor_view_buffer_errors)
    set_tests_properties(rocke_tensor_view_buffer_errors
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_tensor_view_buffer_errors)
    set_target_properties(rocke_tensor_view_buffer_errors
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_tensor_view_buffer_errors
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_tensor_view_buffer_errors)
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
  # its agreement with the LLVM lowerer's word; plus tile.buffer_atomic_fadd_f32:
  # its LLVM lowering with the cache policy, and the refusal of a policy
  # outside 0..31 by the builder and of a bad cachepolicy attr by the lowerer.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_buffer_rsrc_hip.cpp")
    add_executable(rocke_buffer_rsrc_hip core/test_buffer_rsrc_hip.cpp
                                         core/test_buffer_atomic_fadd.cpp)
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

  # The C++ builder refuses an scf.for whose induction variable or carried value
  # reuses a name that is taken.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_scf_for_name_reuse.cpp")
    add_executable(rocke_scf_for_name_reuse core/test_scf_for_name_reuse.cpp)
    target_link_libraries(rocke_scf_for_name_reuse PRIVATE rocke_core)
    add_test(NAME rocke_scf_for_name_reuse COMMAND rocke_scf_for_name_reuse)
    set_tests_properties(rocke_scf_for_name_reuse PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_scf_for_name_reuse)
    set_target_properties(rocke_scf_for_name_reuse
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_scf_for_name_reuse
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_scf_for_name_reuse)
  endif()

  # A failed LLVM lowering gives the values of an scf.for body their names back.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_scf_for_lowering_restores_names.cpp")
    add_executable(rocke_scf_for_lowering_restores_names
                   core/test_scf_for_lowering_restores_names.cpp)
    target_link_libraries(rocke_scf_for_lowering_restores_names PRIVATE rocke_core)
    add_test(NAME rocke_scf_for_lowering_restores_names
             COMMAND rocke_scf_for_lowering_restores_names)
    set_tests_properties(rocke_scf_for_lowering_restores_names
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_scf_for_lowering_restores_names)
    set_target_properties(rocke_scf_for_lowering_restores_names
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_scf_for_lowering_restores_names
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_scf_for_lowering_restores_names)
  endif()

  # The C++ engine lowers an unrolled scf.for whose body holds nested regions,
  # each trip defining names of its own.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_unrolled_for_nested_regions.cpp")
    add_executable(rocke_unrolled_for_nested_regions
                   core/test_unrolled_for_nested_regions.cpp)
    target_link_libraries(rocke_unrolled_for_nested_regions PRIVATE rocke_core)
    add_test(NAME rocke_unrolled_for_nested_regions
             COMMAND rocke_unrolled_for_nested_regions)
    set_tests_properties(rocke_unrolled_for_nested_regions
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_unrolled_for_nested_regions)
    set_target_properties(rocke_unrolled_for_nested_regions
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_unrolled_for_nested_regions
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_unrolled_for_nested_regions)
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

  # Scalar typed global_load GEP index takes the index operand's width.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_global_load_index_width.cpp")
    add_executable(rocke_global_load_index_width core/test_global_load_index_width.cpp)
    target_link_libraries(rocke_global_load_index_width PRIVATE rocke_core)
    add_test(NAME rocke_global_load_index_width COMMAND rocke_global_load_index_width)
    set_tests_properties(rocke_global_load_index_width
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_global_load_index_width)
    set_target_properties(rocke_global_load_index_width
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_global_load_index_width
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_global_load_index_width)
  endif()

  # The C++ verifier, if/else rule and both lowerings refuse a misplaced
  # scf.yield and a malformed if/else, and check a result's type packed-only
  # first.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_if_else_yield_rule.cpp")
    add_executable(rocke_if_else_yield_rule core/test_if_else_yield_rule.cpp)
    target_link_libraries(rocke_if_else_yield_rule PRIVATE rocke_core)
    add_test(NAME rocke_if_else_yield_rule COMMAND rocke_if_else_yield_rule)
    set_tests_properties(rocke_if_else_yield_rule PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_if_else_yield_rule)
    set_target_properties(rocke_if_else_yield_rule
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_if_else_yield_rule
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_if_else_yield_rule)
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

  # MfmaAtom.lane_to_output: every lane and slot against atoms.py's layouts.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_mfma_atom_lane_to_output.cpp")
    add_executable(rocke_mfma_atom_lane_to_output core/test_mfma_atom_lane_to_output.cpp)
    target_link_libraries(rocke_mfma_atom_lane_to_output PRIVATE rocke_core)
    add_test(NAME rocke_mfma_atom_lane_to_output COMMAND rocke_mfma_atom_lane_to_output)
    set_tests_properties(rocke_mfma_atom_lane_to_output
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_mfma_atom_lane_to_output)
    set_target_properties(rocke_mfma_atom_lane_to_output
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_mfma_atom_lane_to_output
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_mfma_atom_lane_to_output)
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

  # Facts on a kernel parameter's declaration (cpp/core/ir/param_facts.cpp). With
  # --ir it is also the C++ side of tests/core/test_param_facts.py, which pins its
  # print_ir, verifier messages and canonicalize against the Python engine.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_param_facts.cpp")
    add_executable(rocke_param_facts core/test_param_facts.cpp)
    target_link_libraries(rocke_param_facts PRIVATE rocke_core)
    add_test(NAME rocke_param_facts COMMAND rocke_param_facts)
    set_tests_properties(rocke_param_facts PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_param_facts)
    set_target_properties(rocke_param_facts PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_param_facts RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_param_facts)
  endif()

  # The integer-expression evaluator (cpp/support/intexpr.cpp). With --eval it is
  # also the C++ side of tests/core/test_intexpr.py, which pins it against the
  # Python mirror.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_intexpr.cpp")
    add_executable(rocke_intexpr core/test_intexpr.cpp)
    target_link_libraries(rocke_intexpr PRIVATE rocke_core)
    add_test(NAME rocke_intexpr COMMAND rocke_intexpr)
    set_tests_properties(rocke_intexpr PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_intexpr)
    set_target_properties(rocke_intexpr PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_intexpr RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_intexpr)
  endif()

  # The GPU check (cpp/core/ir/gpu_check.cpp) and the range marker builder. With
  # --cases and --verify it is also the C++ side of tests/core/test_gpu_checked_value.py,
  # which pins its emitted IR, refusal texts and marker verifier messages
  # against the Python engine.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_gpu_checked_value.cpp")
    add_executable(rocke_gpu_checked_value core/test_gpu_checked_value.cpp)
    target_link_libraries(rocke_gpu_checked_value PRIVATE rocke_core)
    add_test(NAME rocke_gpu_checked_value COMMAND rocke_gpu_checked_value)
    set_tests_properties(rocke_gpu_checked_value PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_gpu_checked_value)
    set_target_properties(rocke_gpu_checked_value PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_gpu_checked_value RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS rocke_gpu_checked_value)
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

  # The exact workgroup_size kernel attr: the one block-size accessor, both
  # C++ lowerers and the verifier's refusal of both block-size attrs.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_workgroup_size.cpp")
    add_executable(rocke_workgroup_size core/test_workgroup_size.cpp)
    target_link_libraries(rocke_workgroup_size PRIVATE rocke_core)
    add_test(NAME rocke_workgroup_size COMMAND rocke_workgroup_size)
    set_tests_properties(rocke_workgroup_size PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_workgroup_size)
    set_target_properties(rocke_workgroup_size
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_workgroup_size
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_workgroup_size)
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

  # The gfx1250 fp8 GEMM's 16x16x128 WMMA atom lowers to HIP in the C++ engine.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/test_dense_fp8_k128_hip.cpp")
    add_executable(rocke_dense_fp8_k128_hip core/test_dense_fp8_k128_hip.cpp)
    target_link_libraries(rocke_dense_fp8_k128_hip PRIVATE rocke_core)
    add_test(NAME rocke_dense_fp8_k128_hip COMMAND rocke_dense_fp8_k128_hip)
    set_tests_properties(rocke_dense_fp8_k128_hip
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_dense_fp8_k128_hip)
    set_target_properties(rocke_dense_fp8_k128_hip
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_dense_fp8_k128_hip
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_dense_fp8_k128_hip)
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

  # Holds both C++ backend registries to one row per target with facts of its own.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/isa_backend_targets.cpp")
    add_executable(rocke_isa_backend_targets core/isa_backend_targets.cpp)
    target_link_libraries(rocke_isa_backend_targets PRIVATE rocke_core)
    add_test(NAME rocke_isa_backend_targets
             COMMAND rocke_isa_backend_targets)
    set_tests_properties(rocke_isa_backend_targets
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_isa_backend_targets)
    set_target_properties(rocke_isa_backend_targets
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_isa_backend_targets
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_isa_backend_targets)
  endif()

  # Holds the intrinsic declaration chain (per declaration ABI generation) to its rule.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/flavor_decl_chain.cpp")
    add_executable(rocke_flavor_decl_chain core/flavor_decl_chain.cpp)
    target_link_libraries(rocke_flavor_decl_chain PRIVATE rocke_core)
    add_test(NAME rocke_flavor_decl_chain
             COMMAND rocke_flavor_decl_chain)
    set_tests_properties(rocke_flavor_decl_chain
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_flavor_decl_chain)
    set_target_properties(rocke_flavor_decl_chain
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_flavor_decl_chain
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_flavor_decl_chain)
  endif()

  # Matrix intrinsic calls described as data: forms, schema and declarations.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/matrix_call_lowering.cpp")
    add_executable(rocke_matrix_call_lowering core/matrix_call_lowering.cpp)
    target_link_libraries(rocke_matrix_call_lowering PRIVATE rocke_core)
    add_test(NAME rocke_matrix_call_lowering
             COMMAND rocke_matrix_call_lowering)
    set_tests_properties(rocke_matrix_call_lowering
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_matrix_call_lowering)
    set_target_properties(rocke_matrix_call_lowering
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_matrix_call_lowering
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_matrix_call_lowering)
  endif()

  # The CDNA MFMA catalog rows and lane maps beyond f16 / bf16 / fp8 / bf8.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/cdna_mfma_ops.cpp")
    add_executable(rocke_cdna_mfma_ops core/cdna_mfma_ops.cpp)
    target_link_libraries(rocke_cdna_mfma_ops PRIVATE rocke_core)
    add_test(NAME rocke_cdna_mfma_ops
             COMMAND rocke_cdna_mfma_ops)
    set_tests_properties(rocke_cdna_mfma_ops
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_cdna_mfma_ops)
    set_target_properties(rocke_cdna_mfma_ops
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_cdna_mfma_ops
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_cdna_mfma_ops)
  endif()

  # The gfx12 WMMA catalog rows and lane maps beyond f16 / bf16 -> f32.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/gfx12_wmma_ops.cpp")
    add_executable(rocke_gfx12_wmma_ops core/gfx12_wmma_ops.cpp)
    target_link_libraries(rocke_gfx12_wmma_ops PRIVATE rocke_core)
    add_test(NAME rocke_gfx12_wmma_ops
             COMMAND rocke_gfx12_wmma_ops)
    set_tests_properties(rocke_gfx12_wmma_ops
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_gfx12_wmma_ops)
    set_target_properties(rocke_gfx12_wmma_ops
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_gfx12_wmma_ops
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_gfx12_wmma_ops)
  endif()

  # The gfx11 WMMA catalog rows and lane maps with a 16-bit result.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/gfx11_wmma_ops.cpp")
    add_executable(rocke_gfx11_wmma_ops core/gfx11_wmma_ops.cpp)
    target_link_libraries(rocke_gfx11_wmma_ops PRIVATE rocke_core)
    add_test(NAME rocke_gfx11_wmma_ops
             COMMAND rocke_gfx11_wmma_ops)
    set_tests_properties(rocke_gfx11_wmma_ops
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_gfx11_wmma_ops)
    set_target_properties(rocke_gfx11_wmma_ops
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_gfx11_wmma_ops
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_gfx11_wmma_ops)
  endif()

  # The gfx1250 WMMA catalog rows with a 16-bit or integer result and the
  # f32 16x16x4 lane maps.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/gfx1250_wmma_ops.cpp")
    add_executable(rocke_gfx1250_wmma_ops core/gfx1250_wmma_ops.cpp)
    target_link_libraries(rocke_gfx1250_wmma_ops PRIVATE rocke_core)
    add_test(NAME rocke_gfx1250_wmma_ops
             COMMAND rocke_gfx1250_wmma_ops)
    set_tests_properties(rocke_gfx1250_wmma_ops
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_gfx1250_wmma_ops)
    set_target_properties(rocke_gfx1250_wmma_ops
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_gfx1250_wmma_ops
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_gfx1250_wmma_ops)
  endif()

  # The gfx1250 f8f6f4 16x16x128 and f4 32x16x128 WMMA catalog rows, calls and
  # lane maps, and the f8f6f4 operand layout.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/gfx1250_f8f6f4_wmma_ops.cpp")
    add_executable(rocke_gfx1250_f8f6f4_wmma_ops core/gfx1250_f8f6f4_wmma_ops.cpp)
    target_link_libraries(rocke_gfx1250_f8f6f4_wmma_ops PRIVATE rocke_core)
    add_test(NAME rocke_gfx1250_f8f6f4_wmma_ops
             COMMAND rocke_gfx1250_f8f6f4_wmma_ops)
    set_tests_properties(rocke_gfx1250_f8f6f4_wmma_ops
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_gfx1250_f8f6f4_wmma_ops)
    set_target_properties(rocke_gfx1250_f8f6f4_wmma_ops
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_gfx1250_f8f6f4_wmma_ops
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_gfx1250_f8f6f4_wmma_ops)
  endif()

  # The gfx950 f8f6f4 16x16x128 and 32x32x64 MFMA catalog rows, calls and lane
  # maps.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/gfx950_f8f6f4_mfma_ops.cpp")
    add_executable(rocke_gfx950_f8f6f4_mfma_ops core/gfx950_f8f6f4_mfma_ops.cpp)
    target_link_libraries(rocke_gfx950_f8f6f4_mfma_ops PRIVATE rocke_core)
    add_test(NAME rocke_gfx950_f8f6f4_mfma_ops
             COMMAND rocke_gfx950_f8f6f4_mfma_ops)
    set_tests_properties(rocke_gfx950_f8f6f4_mfma_ops
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_gfx950_f8f6f4_mfma_ops)
    set_target_properties(rocke_gfx950_f8f6f4_mfma_ops
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_gfx950_f8f6f4_mfma_ops
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_gfx950_f8f6f4_mfma_ops)
  endif()

  # The MMA result dtype where it differs from the accumulator: catalog rows,
  # queries and the tile.mma result type.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/core/mma_result_dtype.cpp")
    add_executable(rocke_mma_result_dtype core/mma_result_dtype.cpp)
    target_link_libraries(rocke_mma_result_dtype PRIVATE rocke_core)
    add_test(NAME rocke_mma_result_dtype
             COMMAND rocke_mma_result_dtype)
    set_tests_properties(rocke_mma_result_dtype
                         PROPERTIES LABELS "ckc;unit;host")
    rocke_cxx_test_installed_name(_rocke_installed_name rocke_mma_result_dtype)
    set_target_properties(rocke_mma_result_dtype
                          PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
    install(TARGETS rocke_mma_result_dtype
            RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
                 rocke_mma_result_dtype)
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

endif()
