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

# Shared dense attention spec (validator, preflight, kernel name): the reject
# reasons never reach the emitted IR, so the byte-identity gate cannot see a
# divergence in them. Both engines are pinned to one vectors file. It is the C
# ABI's contract, so platform owns it (tests/golden/); the library test
# test_attention_dense_spec_vectors.py regenerates it and checks the Python
# against it (library -> platform, the allowed direction). The file is embedded
# into this binary at configure time so the installed test needs no data path.
set(_rocke_dense_spec_vectors "${CMAKE_CURRENT_SOURCE_DIR}/golden/attention_dense_spec_vectors.json")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/instances/test_attention_dense_spec.cpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rocke_dense_spec_vectors}")
  file(READ "${_rocke_dense_spec_vectors}" _rocke_dense_spec_hex HEX)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _rocke_dense_spec_bytes
                       "${_rocke_dense_spec_hex}")
  string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)"
                       "\\1\n" _rocke_dense_spec_bytes "${_rocke_dense_spec_bytes}")
  # configure_file rewrites the .inc only when the bytes change, so a
  # reconfigure does not force a recompile.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/attention_dense_spec_vectors.inc.tmp"
       "${_rocke_dense_spec_bytes}\n")
  configure_file("${CMAKE_CURRENT_BINARY_DIR}/generated/attention_dense_spec_vectors.inc.tmp"
                 "${CMAKE_CURRENT_BINARY_DIR}/generated/attention_dense_spec_vectors.inc"
                 COPYONLY)
  add_executable(rocke_attention_dense_spec instances/test_attention_dense_spec.cpp)
  target_include_directories(rocke_attention_dense_spec
                             PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_link_libraries(rocke_attention_dense_spec PRIVATE rocke_core)
  if(UNIX)
    target_link_libraries(rocke_attention_dense_spec PRIVATE m)
  endif()
  add_test(NAME rocke_attention_dense_spec COMMAND rocke_attention_dense_spec)
  set_tests_properties(rocke_attention_dense_spec PROPERTIES LABELS "ckc;unit;host")
  rocke_cxx_test_installed_name(_rocke_installed_name rocke_attention_dense_spec)
  set_target_properties(rocke_attention_dense_spec
                        PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
  install(TARGETS rocke_attention_dense_spec
          RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
  set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
               rocke_attention_dense_spec)
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
