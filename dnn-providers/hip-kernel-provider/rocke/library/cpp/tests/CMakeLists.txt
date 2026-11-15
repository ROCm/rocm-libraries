# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Host-only C++ tests of the library families' C++ mirrors. Registered and
# installed exactly like the engine's own C++ tests (platform/tests), so the
# installed CTest file lists them with the rest.

# Stream-K wgrad host helpers and validator gates (auto CTA pool per arch,
# grid, workspace layout, reject reasons): also invisible to the byte-identity
# gate, and the Python-vs-C pytest needs the pybind module CI does not build,
# so the C port is pinned to Python reference vectors here.
add_executable(rocke_wgrad_streamk_host wgrad_streamk_host.cpp)
target_link_libraries(rocke_wgrad_streamk_host PRIVATE rocke_library_core)
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

# Forward implicit-GEMM gates around the gfx950 MFMA split-K (parity configs
# 66..85 of conv_implicit_gemm_emit, whose spec table it compiles in): the C++
# verdicts and reasons against Python's. CI checks those configs with the
# Python engine only, and the pybind engine-agreement tests skip there.
add_executable(rocke_conv_implicit_gemm_spec_guards conv_implicit_gemm_spec_guards.cpp)
target_include_directories(rocke_conv_implicit_gemm_spec_guards
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/parity")
target_link_libraries(rocke_conv_implicit_gemm_spec_guards PRIVATE rocke_library_core)
if(UNIX)
  target_link_libraries(rocke_conv_implicit_gemm_spec_guards PRIVATE m)
endif()
add_test(NAME rocke_conv_implicit_gemm_spec_guards
         COMMAND rocke_conv_implicit_gemm_spec_guards)
set_tests_properties(rocke_conv_implicit_gemm_spec_guards PROPERTIES LABELS "ckc;unit;host")
rocke_cxx_test_installed_name(_rocke_installed_name rocke_conv_implicit_gemm_spec_guards)
set_target_properties(rocke_conv_implicit_gemm_spec_guards
                      PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
install(TARGETS rocke_conv_implicit_gemm_spec_guards
        RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
             rocke_conv_implicit_gemm_spec_guards)

# Depthwise dgrad entries the byte-identity gate cannot see: the lowering's
# failure path when the kernel name overflows, the pad_w < -1 refusals and
# the per-axis padding in the tiled dgrad kernel name.
add_executable(rocke_conv_direct_dw_dgrad_host conv_direct_dw_dgrad_host.cpp)
target_link_libraries(rocke_conv_direct_dw_dgrad_host PRIVATE rocke_library_core)
if(UNIX)
  target_link_libraries(rocke_conv_direct_dw_dgrad_host PRIVATE m)
endif()
add_test(NAME rocke_conv_direct_dw_dgrad_host COMMAND rocke_conv_direct_dw_dgrad_host)
set_tests_properties(rocke_conv_direct_dw_dgrad_host PROPERTIES LABELS "ckc;unit;host")
rocke_cxx_test_installed_name(_rocke_installed_name rocke_conv_direct_dw_dgrad_host)
set_target_properties(rocke_conv_direct_dw_dgrad_host
                      PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
install(TARGETS rocke_conv_direct_dw_dgrad_host
        RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
             rocke_conv_direct_dw_dgrad_host)

# The C++ validator of the gfx950 ping-pong direct dgrad: CI does not build
# the rocke_library_engine extension, so its refusal texts are pinned to
# Python's here.
add_executable(rocke_conv_direct_dgrad_pp_mfma_host conv_direct_dgrad_pp_mfma_host.cpp)
target_link_libraries(rocke_conv_direct_dgrad_pp_mfma_host PRIVATE rocke_library_core)
if(UNIX)
  target_link_libraries(rocke_conv_direct_dgrad_pp_mfma_host PRIVATE m)
endif()
add_test(NAME rocke_conv_direct_dgrad_pp_mfma_host
         COMMAND rocke_conv_direct_dgrad_pp_mfma_host)
set_tests_properties(rocke_conv_direct_dgrad_pp_mfma_host PROPERTIES LABELS "ckc;unit;host")
rocke_cxx_test_installed_name(_rocke_installed_name rocke_conv_direct_dgrad_pp_mfma_host)
set_target_properties(rocke_conv_direct_dgrad_pp_mfma_host
                      PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
install(TARGETS rocke_conv_direct_dgrad_pp_mfma_host
        RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
             rocke_conv_direct_dgrad_pp_mfma_host)

# Every refusal of the gfx1250 direct wgrad specs (dense wave32,
# image-per-block, small_c, depthwise) in the C++ validators, by reason. The
# byte-identity gate never sees a refused spec; this mirrors the Python cases
# in library/tests/test_direct_wgrad_refusals.py. Both read the refusal texts
# from library/tests/direct_wgrad_refusal_reasons.inc.
add_executable(rocke_conv_direct_wgrad_refusals_host conv_direct_wgrad_refusals_host.cpp)
target_include_directories(rocke_conv_direct_wgrad_refusals_host
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../tests")
target_link_libraries(rocke_conv_direct_wgrad_refusals_host PRIVATE rocke_library_core)
if(UNIX)
  target_link_libraries(rocke_conv_direct_wgrad_refusals_host PRIVATE m)
endif()
add_test(NAME rocke_conv_direct_wgrad_refusals_host
         COMMAND rocke_conv_direct_wgrad_refusals_host)
set_tests_properties(rocke_conv_direct_wgrad_refusals_host PROPERTIES LABELS "ckc;unit;host")
rocke_cxx_test_installed_name(_rocke_installed_name rocke_conv_direct_wgrad_refusals_host)
set_target_properties(rocke_conv_direct_wgrad_refusals_host
                      PROPERTIES OUTPUT_NAME "${_rocke_installed_name}")
install(TARGETS rocke_conv_direct_wgrad_refusals_host
        RUNTIME DESTINATION ${ROCKE_CXX_TEST_INSTALL_DIR})
set_property(GLOBAL APPEND PROPERTY ROCKE_INSTALL_TEST_TARGETS
             rocke_conv_direct_wgrad_refusals_host)

# YAML tier labels (ROCKE_ENGINE_test_categories_external.yaml lists every
# test above). The labeler only reaches tests add_test()'d in the calling
# directory, so platform/tests' own include does not cover them.
include("${rocke_SOURCE_DIR}/tests/cmake/TestCategoryLabels.cmake")
