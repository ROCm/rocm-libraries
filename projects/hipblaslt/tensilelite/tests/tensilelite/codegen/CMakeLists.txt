# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(tensilelite-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/IRBuilder_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Registers_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RocisaCore_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Target_test.cpp"
)

target_link_libraries(tensilelite-tests
    PRIVATE
        roc::tensilelite-codegen
        rocisa::rocisa-core
)

# rocisa probes the assembler for ISA capabilities at init.
target_compile_definitions(tensilelite-tests
    PRIVATE
        ROCISA_CORE_TEST_ASSEMBLER="${TENSILELITE_CODEGEN_ASSEMBLER}"
)
