# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(SOURCES
    tablegen.cpp
    GenIsa.cpp
    GenRocisaHwMapping.cpp
)

add_executable(tablegen "${SOURCES}")

target_link_libraries(tablegen PRIVATE gfxisa)

target_include_directories(tablegen PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

# Define output directory for generated files
# Use absolute normalized path to avoid issues with ../.. in dependencies
get_filename_component(TABLEGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/../.." ABSOLUTE)

# Define the list of generated files
set(TABLEGEN_GENERATED_FILES
    "${TABLEGEN_OUTPUT_DIR}/hardware/gfx942Isa.inc"
    "${TABLEGEN_OUTPUT_DIR}/hardware/gfx950Isa.inc"
    "${TABLEGEN_OUTPUT_DIR}/hardware/gfx1250Isa.inc"
    "${TABLEGEN_OUTPUT_DIR}/ir/rocisa/Rocisagfx942Mappings.inc"
    "${TABLEGEN_OUTPUT_DIR}/ir/rocisa/Rocisagfx950Mappings.inc"
    "${TABLEGEN_OUTPUT_DIR}/ir/rocisa/Rocisagfx1250Mappings.inc"
)

set(HARDWARE_YAML_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/data/gfx942.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/data/gfx950.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware/data/gfx1250.yaml"
)

# Custom command to run tablegen and generate .inc files
# Using $<TARGET_FILE:tablegen> allows CMake to:
# 1. Know the exact executable path
# 2. Automatically track the dependency
# 3. Check timestamps correctly
add_custom_command(
    OUTPUT ${TABLEGEN_GENERATED_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/hardware"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TABLEGEN_OUTPUT_DIR}/ir/rocisa"
    COMMAND $<TARGET_FILE:tablegen> "${TABLEGEN_OUTPUT_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../hardware"
    DEPENDS $<TARGET_FILE:tablegen> ${HARDWARE_YAML_FILES}
    COMMENT "Generating ISA definition files with tablegen..."
    VERBATIM
)

# Custom target that depends on the generated files
add_custom_target(tablegen_generated
    DEPENDS ${TABLEGEN_GENERATED_FILES}
)

# Export variables for parent scope
set(TABLEGEN_OUTPUT_DIR "${TABLEGEN_OUTPUT_DIR}" PARENT_SCOPE)
set(TABLEGEN_GENERATED_FILES "${TABLEGEN_GENERATED_FILES}" PARENT_SCOPE)
