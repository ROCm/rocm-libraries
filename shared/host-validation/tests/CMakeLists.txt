# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(host-validation-tests host_validation_test.cpp)
target_link_libraries(host-validation-tests PRIVATE roc::host-validation)
add_test(NAME host-validation-tests COMMAND host-validation-tests)

add_executable(host-validation-core-tests tensor_core_test.cpp)
target_link_libraries(host-validation-core-tests PRIVATE roc::host-validation-core)
add_test(NAME host-validation-core-tests COMMAND host-validation-core-tests)

add_executable(host-validation-scalar-codec-tests scalar_codec_test.cpp)
target_link_libraries(host-validation-scalar-codec-tests PRIVATE roc::host-validation-core)
add_test(NAME host-validation-scalar-codec-tests COMMAND host-validation-scalar-codec-tests)

if(TARGET host-validation-blas)
    add_executable(host-validation-blas-tests blas_backend_test.cpp)
    target_link_libraries(
        host-validation-blas-tests
        PRIVATE
            roc::host-validation-blas
    )
    add_test(NAME host-validation-blas-tests COMMAND host-validation-blas-tests)
endif()

add_test(
    NAME host-validation-core-boundary
    COMMAND
        "${CMAKE_COMMAND}"
        "-DCORE_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/../include/roc/host_validation/tensor.hpp"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_core_boundary.cmake"
)
