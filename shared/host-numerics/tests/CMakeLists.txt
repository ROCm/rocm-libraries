# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(host-numerics-version-test version_test.cpp)
target_link_libraries(host-numerics-version-test PRIVATE roc::host-numerics)
add_test(NAME host-numerics-version COMMAND host-numerics-version-test)
