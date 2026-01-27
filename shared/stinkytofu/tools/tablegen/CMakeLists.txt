# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(SOURCES
    tablegen.cpp
    GenIsa.cpp
    GenRocisaHwMapping.cpp
    GenPatterns.cpp
    GenLogicalIR.cpp
    # Pattern parser sources (needed for GenPatterns)
    ../../src/ir/asm/IRLexer.cpp
    ../../src/ir/asm/PatternParser.cpp
)

add_executable(tablegen "${SOURCES}")

target_link_libraries(tablegen PRIVATE gfxisa)

target_include_directories(tablegen PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/ir/asm"  # For IRLexer.hpp
)

# Define output directory for generated files
# Use absolute normalized path to avoid issues with ../.. in dependencies
get_filename_component(TABLEGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../.." ABSOLUTE)

# Define the list of generated files
set(TABLEGEN_GENERATED_FILES "")

# Depend on the C++ files that define instruction mappings and costs
set(HARDWARE_SOURCE_FILES "")

foreach(arch ${STINKYTOFU_ARCHS_TO_BUILD})
    list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/hardware/${arch}Isa.inc")
    list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/ir/rocisa/Rocisa${arch}Mappings.inc")

    list(APPEND HARDWARE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/src/gfx/${arch}.cpp")
endforeach()

# Add pattern-generated files
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/PeepholePatterns.inc")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/LogicalIRPatterns.inc")
set(PATTERN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Dialect/Transforms/PeepholePatterns.pattern")
set(HLIR_PATTERN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/Dialect/Transforms/LogicalIR/LogicalIRPatterns.pattern")

# Add high-level IR generated files
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/LogicalOpcodes_generated.inc")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/ir/logical/LogicalOpcode.cpp")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/ir/logical/LogicalInstructions_generated.hpp")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/StinkyBuilder_decls_generated.inc")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/StinkyBuilder_impls_generated.inc")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/ir/IRMnemonics_generated.inc")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/PythonBindings_generated.inc")

# Custom command to run tablegen and generate .inc files
# Using $<TARGET_FILE:tablegen> allows CMake to:
# 1. Know the exact executable path
# 2. Automatically track the dependency
# 3. Check timestamps correctly
add_custom_command(
    OUTPUT ${TABLEGEN_GENERATED_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/hardware"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/ir"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/ir/logical"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/ir/rocisa"
    COMMAND $<TARGET_FILE:tablegen> "${TABLEGEN_OUTPUT_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware"
    DEPENDS tablegen ${HARDWARE_SOURCE_FILES} ${PATTERN_FILE} ${HLIR_PATTERN_FILE}
    COMMENT "Generating ISA definitions, IR classes, and pattern matchers with tablegen..."
    VERBATIM
)

# Custom target that depends on the generated files
add_custom_target(tablegen_generated
    DEPENDS ${TABLEGEN_GENERATED_FILES}
)

# Export variables for parent scope
set(TABLEGEN_OUTPUT_DIR "${TABLEGEN_OUTPUT_DIR}" PARENT_SCOPE)
set(TABLEGEN_GENERATED_FILES "${TABLEGEN_GENERATED_FILES}" PARENT_SCOPE)
