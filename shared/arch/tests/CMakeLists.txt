# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
add_executable(rocm-arch-constexpr catalog.cpp)
target_link_libraries(rocm-arch-constexpr PRIVATE rocm::arch)
set_target_properties(rocm-arch-constexpr PROPERTIES
    CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
)
add_test(NAME rocm-arch-constexpr COMMAND rocm-arch-constexpr)
add_test(NAME rocm-arch-python
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_catalog.py"
)
add_test(NAME rocm-arch-cmake
    COMMAND "${CMAKE_COMMAND}"
        "-DCATALOG=${PROJECT_BINARY_DIR}/generated/rocm-arch-catalog.cmake"
        "-DFUNCTIONS=${PROJECT_SOURCE_DIR}/cmake/rocm-arch-functions.cmake"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/catalog.cmake"
)
if(ROCM_ARCH_BUILD_INTEGRATION_TESTING)
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/../../projects/rocfft/CMakeLists.txt")
        message(FATAL_ERROR "Consumer integration tests need a monorepo checkout")
    endif()
    add_test(NAME rocm-arch-fft-consumers
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_consumers.py"
    )
endif()
