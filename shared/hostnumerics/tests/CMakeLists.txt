# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(hostnumerics-version-test version_test.cpp)
target_link_libraries(hostnumerics-version-test PRIVATE roc::hostnumerics)
add_test(NAME hostnumerics-version COMMAND hostnumerics-version-test)

add_executable(hostnumerics-scalar-type-test scalar_type_test.cpp)
target_link_libraries(hostnumerics-scalar-type-test PRIVATE roc::hostnumerics-core)
add_test(NAME hostnumerics-scalar-type COMMAND hostnumerics-scalar-type-test)
