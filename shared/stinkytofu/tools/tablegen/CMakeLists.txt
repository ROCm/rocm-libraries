# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(SOURCES
    tablegen.cpp
    GenRocisaHwMapping.cpp
    GenPatterns.cpp
    GenLogicalIR.cpp
    GenInstructions.cpp  # NEW: Instruction metadata generator (proof-of-concept)
    # Pattern parser sources (needed for GenPatterns)
    ../../src/ir/asm/IRLexer.cpp
    ../../src/ir/asm/PatternParser.cpp
)

add_executable(tablegen "${SOURCES}")

target_link_libraries(tablegen PRIVATE gfxisa)

# Standalone instruction generator (no gfxisa dependency) to break build cycle:
# gfxisa needs generated .inc files, but full tablegen links gfxisa.
add_executable(tablegen_inst_gen
    GenInstructions.cpp
    GenInstructionsMain.cpp
)
# No link to gfxisa - this executable only parses .def files and writes .inc

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

# Per-arch Isa.inc and gfxIsa.inc are produced by tablegen_inst_gen (instruction_generated), not full tablegen.
foreach(arch ${STINKYTOFU_ARCHS_TO_BUILD})
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
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/ir/IRMnemonics_generated.inc")
list(APPEND TABLEGEN_GENERATED_FILES "${TABLEGEN_OUTPUT_DIR}/PythonBindings_generated.inc")

# Instruction metadata and ISA .inc from .def files (tablegen_inst_gen, no gfxisa).
# Single run produces: hardware/generated/*_costs.inc, *_init.inc, *_operands.inc and
# hardware/*Isa.inc, hardware/gfxIsa.inc so the full tablegen does not need gfxisa for ISA.
get_filename_component(INSTRUCTION_DEF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/defs" ABSOLUTE)
set(INSTRUCTION_GEN_DIR "${TABLEGEN_OUTPUT_DIR}/hardware/generated")
set(INSTRUCTION_GEN_FILES "")
foreach(arch Gfx1250 Gfx942 Gfx950)
    list(APPEND INSTRUCTION_GEN_FILES "${INSTRUCTION_GEN_DIR}/${arch}_costs.inc")
    list(APPEND INSTRUCTION_GEN_FILES "${INSTRUCTION_GEN_DIR}/${arch}_init.inc")
    list(APPEND INSTRUCTION_GEN_FILES "${INSTRUCTION_GEN_DIR}/${arch}_operands.inc")
    list(APPEND INSTRUCTION_GEN_FILES "${TABLEGEN_OUTPUT_DIR}/hardware/${arch}Isa.inc")
endforeach()
list(APPEND INSTRUCTION_GEN_FILES "${TABLEGEN_OUTPUT_DIR}/hardware/gfxIsa.inc")
list(APPEND TABLEGEN_GENERATED_FILES ${INSTRUCTION_GEN_FILES})
set(INSTRUCTION_DEF_FILES
    "${INSTRUCTION_DEF_DIR}/Gfx1250Formats.def"
    "${INSTRUCTION_DEF_DIR}/Gfx1250Instructions.def"
    "${INSTRUCTION_DEF_DIR}/Gfx942Formats.def"
    "${INSTRUCTION_DEF_DIR}/Gfx942Instructions.def"
    "${INSTRUCTION_DEF_DIR}/Gfx950Formats.def"
    "${INSTRUCTION_DEF_DIR}/Gfx950Instructions.def"
)

# One tablegen_inst_gen run: all archs + ISA .inc (single build, no two-phase).
add_custom_command(
    OUTPUT ${INSTRUCTION_GEN_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/hardware/generated"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/hardware"
    COMMAND $<TARGET_FILE:tablegen_inst_gen> --input-dir="${INSTRUCTION_DEF_DIR}" --output-dir="${TABLEGEN_OUTPUT_DIR}"
    DEPENDS tablegen_inst_gen ${INSTRUCTION_DEF_FILES}
    COMMENT "Generating instruction metadata and ISA from .def files..."
    VERBATIM
)
add_custom_target(instruction_generated DEPENDS ${INSTRUCTION_GEN_FILES})

# Remaining tablegen outputs (ISA, patterns, etc.); tablegen links gfxisa
set(TABLEGEN_OTHER_FILES ${TABLEGEN_GENERATED_FILES})
list(REMOVE_ITEM TABLEGEN_OTHER_FILES ${INSTRUCTION_GEN_FILES})

add_custom_command(
    OUTPUT ${TABLEGEN_OTHER_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/hardware"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/ir"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/ir/logical"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/ir/rocisa"
    COMMAND $<TARGET_FILE:tablegen> "${TABLEGEN_OUTPUT_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware"
    DEPENDS tablegen instruction_generated ${HARDWARE_SOURCE_FILES} ${PATTERN_FILE} ${HLIR_PATTERN_FILE}
    COMMENT "Generating ISA definitions, IR classes, and pattern matchers with tablegen..."
    VERBATIM
)

# Custom target that depends on all generated files
add_custom_target(tablegen_generated
    DEPENDS ${TABLEGEN_GENERATED_FILES}
)

# Export variables for parent scope
set(TABLEGEN_OUTPUT_DIR "${TABLEGEN_OUTPUT_DIR}" PARENT_SCOPE)
set(TABLEGEN_GENERATED_FILES "${TABLEGEN_GENERATED_FILES}" PARENT_SCOPE)
