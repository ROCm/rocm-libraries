# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Function to embed GPU reference kernel sources as C++ strings at configure time
function(embed_gpu_ref_kernel_sources OUTPUT_CPP OUTPUT_HPP KERNEL_FILES)
    set(KERNEL_DECLARATIONS "")
    set(KERNEL_DEFINITIONS "")
    set(KERNEL_MAP_ENTRIES "")
    set(HEADER_DECLARATIONS "")
    set(HEADER_DEFINITIONS "")
    set(HEADER_MAP_ENTRIES "")
    set(HEADER_FILENAMES "")

    foreach(KERNEL_FILE ${KERNEL_FILES})
        file(READ ${KERNEL_FILE} KERNEL_CONTENT)
        get_filename_component(KERNEL_NAME ${KERNEL_FILE} NAME_WE)
        get_filename_component(KERNEL_FILENAME ${KERNEL_FILE} NAME)
        get_filename_component(KERNEL_EXT ${KERNEL_FILE} EXT)
        string(TOUPPER ${KERNEL_NAME} KERNEL_VAR_NAME)

        if(KERNEL_EXT STREQUAL ".h" OR KERNEL_EXT STREQUAL ".hpp")
            string(APPEND HEADER_DECLARATIONS
                   "extern const char* const GPU_REF_${KERNEL_VAR_NAME}_SOURCE;\n")

            string(APPEND HEADER_DEFINITIONS
                   "const char* const GPU_REF_${KERNEL_VAR_NAME}_SOURCE = R\"KERNEL_SOURCE(\n")
            string(APPEND HEADER_DEFINITIONS "${KERNEL_CONTENT}")
            string(APPEND HEADER_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")

            string(APPEND HEADER_MAP_ENTRIES
                   "        {\"${KERNEL_FILENAME}\", GPU_REF_${KERNEL_VAR_NAME}_SOURCE},\n")

            string(APPEND HEADER_FILENAMES "        \"${KERNEL_FILENAME}\",\n")
        else()
            string(APPEND KERNEL_DECLARATIONS
                   "extern const char* const GPU_REF_${KERNEL_VAR_NAME}_SOURCE;\n")

            string(APPEND KERNEL_DEFINITIONS
                   "const char* const GPU_REF_${KERNEL_VAR_NAME}_SOURCE = R\"KERNEL_SOURCE(\n")
            string(APPEND KERNEL_DEFINITIONS "${KERNEL_CONTENT}")
            string(APPEND KERNEL_DEFINITIONS "\n)KERNEL_SOURCE\";\n\n")

            string(APPEND KERNEL_MAP_ENTRIES
                   "        {\"${KERNEL_FILENAME}\", GPU_REF_${KERNEL_VAR_NAME}_SOURCE},\n")
        endif()
    endforeach()

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/GpuRefKernelSources.hpp.in ${OUTPUT_HPP} @ONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/templates/GpuRefKernelSources.cpp.in ${OUTPUT_CPP} @ONLY
    )
endfunction()

# List kernel files to embed
set(GPU_REF_KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution/GpuRefConvFwd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/validator/GpuRefValidator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefConvArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefValidatorArgs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/types/GpuRefTypes.h
)

# Force CMake reconfigure when kernel files change
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GPU_REF_KERNEL_FILES})

# Generate embedded kernel sources at configure time
embed_gpu_ref_kernel_sources(
    ${CMAKE_CURRENT_BINARY_DIR}/GpuRefKernelSources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/GpuRefKernelSources.hpp
    "${GPU_REF_KERNEL_FILES}"
)

# Export generated source files to parent scope
set(GPU_REF_KERNEL_GENERATED_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/GpuRefKernelSources.cpp PARENT_SCOPE)
