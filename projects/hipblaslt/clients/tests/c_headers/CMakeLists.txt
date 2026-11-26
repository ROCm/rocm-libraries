# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Builds the public extern "C" headers as C. A header that only C++ accepts
# fails this build. Not registered with ctest (see ../../CMakeLists.txt for why
# extra ctest entries break TheRock sharding); building it is the check.
add_executable(hipblaslt-c-headers-check
    ${CMAKE_CURRENT_SOURCE_DIR}/c_headers_a_test.c
    ${CMAKE_CURRENT_SOURCE_DIR}/c_headers_b_test.c
)
set_target_properties(hipblaslt-c-headers-check PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    LINKER_LANGUAGE C
)
if(NOT MSVC)
    target_compile_options(hipblaslt-c-headers-check PRIVATE -Wall -Wextra -Werror)
endif()
target_link_libraries(hipblaslt-c-headers-check PRIVATE roc::hipblaslt hip::host)

add_dependencies(hipblaslt-test hipblaslt-c-headers-check)
