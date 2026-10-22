# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# rocisa probes this assembler for ISA capabilities when a Target is first used.
set(TENSILELITE_CODEGEN_ASSEMBLER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
    "Assembler that rocisa probes for ISA capabilities in the codegen library.")
mark_as_advanced(TENSILELITE_CODEGEN_ASSEMBLER)

add_library(tensilelite-codegen STATIC)
add_library(roc::tensilelite-codegen ALIAS tensilelite-codegen)

set_target_properties(tensilelite-codegen
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)

target_compile_features(tensilelite-codegen PUBLIC cxx_std_20)

target_sources(tensilelite-codegen
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Arith.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/BufferResource.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Gfx9Legality.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/IRBuilder.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Kernel.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Move.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/OperandUtils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Registers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Target.cpp"
)

target_include_directories(tensilelite-codegen PUBLIC "${PROJECT_SOURCE_DIR}/include")

# Public because the codegen headers expose rocisa types.
target_link_libraries(tensilelite-codegen PUBLIC rocisa::rocisa-core)

target_compile_definitions(tensilelite-codegen
    PRIVATE
        TENSILELITE_CODEGEN_ASSEMBLER="${TENSILELITE_CODEGEN_ASSEMBLER}"
)
