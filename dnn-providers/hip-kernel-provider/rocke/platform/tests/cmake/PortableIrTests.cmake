# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Portable-IR C++ test and replay-tool registration. Included from the parent
# tests/CMakeLists.txt to preserve its CMake directory scope.

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
