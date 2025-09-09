# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(component-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ComponentTest.cpp")
target_link_libraries(component-test PRIVATE roc::rocroller-interface ${msgpack_libs})
