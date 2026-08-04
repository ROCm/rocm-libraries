# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(host-validation-tests host_validation_test.cpp)
target_link_libraries(host-validation-tests PRIVATE roc::host-validation)
add_test(NAME host-validation-tests COMMAND host-validation-tests)
