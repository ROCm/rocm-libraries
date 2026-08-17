# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# TEMPLATE ADAPTATION: Replace the kernel filenames below with your GPU kernel source files. Files
# with .cpp extension are embedded as kernel sources (compiled by HIPRTC). Files with .h/.hpp
# extension are embedded as kernel headers (available to #include at runtime). The embedding system
# (cmake/EmbedKernelSources.cmake) handles the rest automatically.
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/relu/ReluForward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv/ConvForwardNaive.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/IndexType.hpp
)

# All generated output files produced by the embedding script
set(KERNEL_GENERATED_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.hpp
)

# Replace CMake list separator (;) with pipe (|) so the list survives as a
# single COMMAND argument through VERBATIM shell escaping. The -P script
# converts back to semicolons before iterating.
string(REPLACE ";" "|" KERNEL_FILES_CMD_STRING "${KERNEL_FILES}")

# Generate embedded kernel sources at build time.
# Modifying any kernel source, template, or the embedding script itself
# triggers regeneration without requiring a full reconfigure.
# See kernel_sources.hpp.in for the full embedding overview.
add_custom_command(
    OUTPUT ${KERNEL_GENERATED_FILES}
    DEPENDS
        ${KERNEL_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_sources.hpp.in
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_sources.cpp.in
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_includes.hpp.in
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/kernel_includes.cpp.in
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedKernelSources.cmake
    COMMAND
        ${CMAKE_COMMAND}
        "-DKERNEL_FILES=${KERNEL_FILES_CMD_STRING}"
        -DTEMPLATE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/templates
        -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedKernelSources.cmake
    COMMENT "Embedding kernel sources as C++ string literals"
    VERBATIM
)

add_custom_target(example_provider_kernel_embed
    DEPENDS ${KERNEL_GENERATED_FILES}
    COMMENT "Updating embedded kernel sources"
)

# Export generated sources for src/ to compile
set(KERNEL_GENERATED_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.cpp
    PARENT_SCOPE
)
