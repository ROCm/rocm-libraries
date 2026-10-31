# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# C++ parity emitters checked against the differential golden. Included from
# the parent tests/CMakeLists.txt to preserve its CMake directory scope.
#
# run_diff.py compiles each parity emitter on demand, which needs a compiler
# and the engine source, so CI's installed test tree cannot compare C++ with
# Python. These emitters are built here instead and installed as test
# binaries; tests/instances/differential/check_native_golden.py then checks
# their .ll against the golden that rocke_differential_golden_ll checks the
# Python reference against. Each family needs blessed golden entries.

set(_ROCKE_NATIVE_GOLDEN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../library/tests/parity")
set(_ROCKE_NATIVE_GOLDEN_FAMILIES
    attention_bwd_aux
    attention_bwd_band
    attention_bwd_body
    attention_bwd_counted
    attention_bwd_dq
    attention_bwd_frag
    attention_bwd_layout
    attention_bwd_lds
    attention_bwd_main
    attention_bwd_paged
    attention_bwd_sched
    attention_bwd_sched_ir
    attention_bwd_sched_params
    attention_bwd_sinks
    attention_bwd_spec
    attention_bwd_spec_extra
    attention_bwd_unified_main
    attention_bwd_valu
    attention_dense_bwd_dq
    attention_dense_bwd_main)

set(_rocke_native_golden_args "")
foreach(_family ${_ROCKE_NATIVE_GOLDEN_FAMILIES})
  set(_src "${_ROCKE_NATIVE_GOLDEN_SRC_DIR}/${_family}_emit.c")
  if(NOT EXISTS "${_src}")
    continue()
  endif()
  set(_target "rocke_${_family}_emit")
  # The emitters are C++ in a .c file, as run_diff.py compiles them.
  set_source_files_properties("${_src}" PROPERTIES LANGUAGE CXX)
  add_executable(${_target} "${_src}")
  target_link_libraries(${_target} PRIVATE rocke_core)
  # Test drivers, not engine code: run_diff.py builds them with no warning
  # flags, so warnings stay visible here but do not fail the build.
  if(NOT MSVC)
    target_compile_options(${_target} PRIVATE -Wno-error)
  endif()
  if(UNIX)
    target_link_libraries(${_target} PRIVATE m)
  endif()
  rocke_cxx_test_installed_name(_rocke_installed_name ${_target})
  set_target_properties(${_target} PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS ${_target} RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  # Not ROCKE_INSTALL_TEST_TARGETS: that list registers each binary as a
  # ctest of its own, and an emitter needs a config index.
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_NATIVE_GOLDEN_EMITTERS
               "${_family}=${_target}")
  list(APPEND _rocke_native_golden_args "${_family}=$<TARGET_FILE:${_target}>")
endforeach()

if(_rocke_native_golden_args AND Python3_Interpreter_FOUND)
  add_test(NAME rocke_native_golden_ll
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/instances/differential/check_native_golden.py"
                   --flavor llvm20,llvm22,llvm23,llvm24 ${_rocke_native_golden_args})
  set_tests_properties(rocke_native_golden_ll
                       PROPERTIES LABELS "ckc;python;golden;host;quick" TIMEOUT 600)
endif()
