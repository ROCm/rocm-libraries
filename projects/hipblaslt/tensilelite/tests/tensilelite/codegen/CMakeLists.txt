# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(tensilelite-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Arith_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/BufferResource_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Compare_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/IRBuilder_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Move_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Registers_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RocisaCore_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Select_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Target_test.cpp"
)

# Assembling and launching needs the JIT assembler.
if(TENSILELITE_ENABLE_JIT)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/ArithDeviceTest.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/ArithDevice_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/BufferResourceDevice_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/CompareDevice_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/KernelLaunch_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/MoveDevice_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/Saxpy_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/SelectDevice_test.cpp"
    )
endif()

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
