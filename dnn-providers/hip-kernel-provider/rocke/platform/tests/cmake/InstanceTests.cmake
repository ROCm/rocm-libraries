# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Instance-layer C++ test registration. Included from the parent
# tests/CMakeLists.txt to preserve its CMake directory scope.

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

# UnifiedAttentionProblem C port: field defaults, the @property helpers, and
# the explicit-arch save/restore in the tiled-2D spec-from-problem selector.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/attention_unified_problem_test.cpp")
  add_executable(rocke_attention_unified_problem
                 instances/attention_unified_problem_test.cpp)
  target_link_libraries(rocke_attention_unified_problem PRIVATE rocke_core)
  if(UNIX)
    target_link_libraries(rocke_attention_unified_problem PRIVATE m)
  endif()
  add_test(NAME rocke_attention_unified_problem
           COMMAND rocke_attention_unified_problem)
  set_tests_properties(rocke_attention_unified_problem
                       PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name
                                rocke_attention_unified_problem)
  set_target_properties(rocke_attention_unified_problem
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_attention_unified_problem
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_attention_unified_problem)
endif()

# Tiled-2D selectors pinned to the values the Python selectors return.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/attention_unified_selectors_test.cpp")
  add_executable(rocke_attention_unified_selectors
                 instances/attention_unified_selectors_test.cpp)
  target_link_libraries(rocke_attention_unified_selectors PRIVATE rocke_core)
  if(UNIX)
    target_link_libraries(rocke_attention_unified_selectors PRIVATE m)
  endif()
  add_test(NAME rocke_attention_unified_selectors
           COMMAND rocke_attention_unified_selectors)
  set_tests_properties(rocke_attention_unified_selectors
                       PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name
                                rocke_attention_unified_selectors)
  set_target_properties(rocke_attention_unified_selectors
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_attention_unified_selectors
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_attention_unified_selectors)
endif()

# Unified-attention 2D/3D routing and the scalar / tiled-3D coverage gates.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/attention_unified_routing_test.cpp")
  add_executable(rocke_attention_unified_routing
                 instances/attention_unified_routing_test.cpp)
  target_link_libraries(rocke_attention_unified_routing PRIVATE rocke_core)
  if(UNIX)
    target_link_libraries(rocke_attention_unified_routing PRIVATE m)
  endif()
  add_test(NAME rocke_attention_unified_routing
           COMMAND rocke_attention_unified_routing)
  set_tests_properties(rocke_attention_unified_routing
                       PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name
                                rocke_attention_unified_routing)
  set_target_properties(rocke_attention_unified_routing
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_attention_unified_routing
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_attention_unified_routing)
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

# gfx942 tiled-attention spec guards: same acceptance-divergence blind spot as
# above, for the 2D/3D __post_init__ ports and supports_tiled_2d.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/attention_tiled_2d_gfx942_spec_guards.cpp")
  add_executable(rocke_attention_tiled_2d_gfx942_spec_guards
                 instances/attention_tiled_2d_gfx942_spec_guards.cpp)
  target_link_libraries(rocke_attention_tiled_2d_gfx942_spec_guards PRIVATE rocke_core)
  if(UNIX)
    target_link_libraries(rocke_attention_tiled_2d_gfx942_spec_guards PRIVATE m)
  endif()
  add_test(NAME rocke_attention_tiled_2d_gfx942_spec_guards
           COMMAND rocke_attention_tiled_2d_gfx942_spec_guards)
  set_tests_properties(rocke_attention_tiled_2d_gfx942_spec_guards
                       PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_attention_tiled_2d_gfx942_spec_guards)
  set_target_properties(rocke_attention_tiled_2d_gfx942_spec_guards
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_attention_tiled_2d_gfx942_spec_guards
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_attention_tiled_2d_gfx942_spec_guards)
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

# Python-vs-C++ parity of the unified-attention selectors, routing gates and
# the binding's spec fields, run through the rocke_engine module. Needs that
# module and the sibling library tree; a platform-only copy still configures.
set(_rocke_library_tests "${CMAKE_CURRENT_SOURCE_DIR}/../../library/tests")
if(ROCKE_BUILD_PYBIND
   AND EXISTS "${_rocke_library_tests}/test_attention_selector_cpp_parity.py")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import pytest"
                  RESULT_VARIABLE _rocke_has_pytest
                  OUTPUT_QUIET ERROR_QUIET)
  if(_rocke_has_pytest EQUAL 0)
    # The pytest files skip without the module; this entry fails instead, so
    # a missing or unbuilt module can't pass as a fully skipped run.
    add_test(NAME rocke_engine_import
             COMMAND "${Python3_EXECUTABLE}" -c "import rocke_engine")
    add_test(NAME rocke_attention_cpp_parity_pytest
             COMMAND "${Python3_EXECUTABLE}" -m pytest
                     "${_rocke_library_tests}/test_attention_selector_cpp_parity.py"
                     "${_rocke_library_tests}/test_attention_routing_cpp_parity.py"
                     "${_rocke_library_tests}/test_attention_binding_spec_fields.py"
                     "${_rocke_library_tests}/test_attention_spec_cpp_parity.py"
                     "${_rocke_library_tests}/test_attention_4warp_gqa_cpp_parity.py"
                     -q -rsx -p no:cacheprovider
             WORKING_DIRECTORY "${_rocke_library_tests}")
    # The library conftest puts the library and platform/python on sys.path.
    set_tests_properties(rocke_engine_import rocke_attention_cpp_parity_pytest
                         PROPERTIES LABELS "ckc;python;attention;host"
                                    ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:rocke_engine>")
    set_tests_properties(rocke_attention_cpp_parity_pytest PROPERTIES TIMEOUT 600)
  else()
    message(STATUS "rocKE: ${Python3_EXECUTABLE} cannot import pytest; "
                   "rocke_attention_cpp_parity_pytest is not registered")
  endif()
endif()
