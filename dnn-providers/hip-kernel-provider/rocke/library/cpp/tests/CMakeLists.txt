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
