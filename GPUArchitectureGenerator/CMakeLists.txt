
################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

# ArchitectureGenerator

# Not visible to library users, so not necessary to use system packages
rocroller_add_dependency(
    isa_spec_manager HASH 1c6fcc8072c7a94ceedb734cf06fd8a08487c4b3
)
if(NOT ROCROLLER_USE_PREGENERATED_ARCH_DEF)
    rocroller_add_dependency(mrisa_xml VERSION 2024_08_22)
endif()

add_executable(
    GPUArchitectureGenerator
    ${CMAKE_CURRENT_SOURCE_DIR}/source/GPUArchitectureGenerator.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/GPUArchitecture.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/GPUArchitectureLibrary.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/Assemblers/Assembler.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/Costs/Cost.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/DataTypes_F6_Utils.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/DataTypes_F8_Utils.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/DataTypes_BF16_Utils.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/Error.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/Logging.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/Scheduler.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/Utilities/Settings.cpp
    ${PROJECT_SOURCE_DIR}/lib/source/Timer.cpp
)

if(NOT ROCROLLER_USE_PREGENERATED_ARCH_DEF AND DEFINED INTERNAL_MRISAS AND INTERNAL_MRISAS)
    foreach(MRISA_FILE IN LISTS INTERNAL_MRISAS)
        if(EXISTS ${MRISA_FILE})
            get_filename_component(mrisa_file_name "${MRISA_FILE}" NAME)
            add_custom_target(
                copy_${mrisa_file_name}
                COMMAND ${CMAKE_COMMAND} -E copy "${MRISA_FILE}"
                        "${mrisa_xml_SOURCE_DIR}"
                BYPRODUCTS "${mrisa_xml_SOURCE_DIR}/${mrisa_file_name}"
                DEPENDS ${MRISA_FILE}
            )
            list(APPEND INTERNAL_MRISA_FILES
                 "${mrisa_xml_SOURCE_DIR}/${mrisa_file_name}"
            )
            list(APPEND INTERNAL_MRISA_TARGETS copy_${mrisa_file_name})
        else()
            message(WARNING "MRISA file ${MRISA_FILE} could not be found.")
        endif()
    endforeach(MRISA_FILE IN LISTS INTERNAL_MRISAS)
endif()

target_compile_options(
    GPUArchitectureGenerator PUBLIC ${COROUTINES_COMPILE_OPTION}
                                    ${EXTRA_COMPILE_OPTIONS}
)
target_link_options(GPUArchitectureGenerator PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  GPUArchitectureGenerator PUBLIC
  ${llvm_libs}
  msgpackc-cxx
  isa_spec_manager
  spdlog::spdlog_header_only
)
target_include_directories(
    GPUArchitectureGenerator PRIVATE include ${PROJECT_SOURCE_DIR}/lib/include
)

target_include_directories(
    GPUArchitectureGenerator SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS}
)

target_compile_definitions(
    GPUArchitectureGenerator PRIVATE ROCROLLER_NO_EMBED_ARCH_DEF
)

if(YAML_BACKEND STREQUAL "YAML_CPP")
    target_link_libraries(GPUArchitectureGenerator PUBLIC yaml-cpp)
endif()

if(ROCROLLER_USE_HIP)
    target_link_libraries(GPUArchitectureGenerator PUBLIC hip::host)
    target_compile_definitions(
        GPUArchitectureGenerator PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__
    )
endif()

set(ARCH_FILE_DIR ${PROJECT_BINARY_DIR}/share/rocRoller)

file(MAKE_DIRECTORY ${ARCH_FILE_DIR})

if(NOT ROCROLLER_USE_PREGENERATED_ARCH_DEF)

    file(MAKE_DIRECTORY ${ARCH_FILE_DIR}/split_yamls)

    add_custom_command(
        OUTPUT ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml
        COMMAND
            GPUArchitectureGenerator
            ${ARCH_FILE_DIR}/split_yamls/GPUArchitecture_def.yaml -Y --split --xml_dir
            ${mrisa_xml_SOURCE_DIR}
        COMMAND
            GPUArchitectureGenerator
            ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml -Y --xml_dir
            ${mrisa_xml_SOURCE_DIR}
        DEPENDS ${INTERNAL_MRISA_TARGETS}
    )

else()

    file(GLOB YAML_FILES "${PROJECT_SOURCE_DIR}/GPUArchitectureGenerator/pregenerated/*.yaml")
    set(YAML_ARGS "")
    foreach(YAML_FILE ${YAML_FILES})
        list(APPEND YAML_ARGS --yaml "${YAML_FILE}")
    endforeach()

    add_custom_command(
        OUTPUT ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml
        COMMAND
            GPUArchitectureGenerator
            ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml -Y ${YAML_ARGS}
        DEPENDS "${YAML_FILES}"
    )

endif()

add_custom_command(
    OUTPUT ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack
    COMMAND
        GPUArchitectureGenerator
        ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack
        --yaml ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml
    DEPENDS ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml
)

if(INTERNAL_MRISA_FILES)
    set(delete_internal_command COMMAND ${CMAKE_COMMAND} -E remove
                                ${INTERNAL_MRISA_FILES}
    )
endif()

if(ROCROLLER_EMBED_ARCH_DEF)
    # Not exposed to users of the library, so not necessary to use system package
    rocroller_add_dependency(cmrc VERSION 2.0.1)

    cmrc_add_resource_library(
        GPUArchitecture_def_library
        ALIAS rocRoller::GPUArchitecture_def
        NAMESPACE rocRoller
        WHENCE ${ARCH_FILE_DIR}
        PREFIX resources
        ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack
    )

    set_target_properties(
        GPUArchitecture_def_library PROPERTIES POSITION_INDEPENDENT_CODE True
    )

    set(resource_library GPUArchitecture_def_library)
endif()

add_custom_target(
    GPUArchitecture_def ALL
    ${delete_internal_command}
    DEPENDS ${ARCH_FILE_DIR}/GPUArchitecture_def.yaml
            ${ARCH_FILE_DIR}/GPUArchitecture_def.msgpack ${resource_library}
)
