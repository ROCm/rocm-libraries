# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

function(hipblaslt_create_device_library)
    set(_opts "")
    set(_one
        TARGET LOGIC_PATH OUTPUT_DIR CXX_COMPILER OFFLOAD_BUNDLER JOBS LOGIC_FILTER
        ASAN YAML_FORMAT NO_COMPRESS EXPERIMENTAL LAZY_LOAD ASM_COMMENTS KEEP_BUILD_TMP ASM_DEBUG)
    set(_multi ARCHES)
    cmake_parse_arguments(_cdl "${_opts}" "${_one}" "${_multi}" ${ARGN})

    if(_cdl_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "hipblaslt_create_device_library: unexpected arguments: ${_cdl_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT _cdl_LOGIC_PATH)
        message(FATAL_ERROR "hipblaslt_create_device_library: LOGIC_PATH is required")
    endif()
    if(NOT _cdl_OUTPUT_DIR)
        message(FATAL_ERROR "hipblaslt_create_device_library: OUTPUT_DIR is required")
    endif()
    if(NOT HIPBLASLT_PYTHON_COMMAND)
        message(FATAL_ERROR "hipblaslt_create_device_library: HIPBLASLT_PYTHON_COMMAND is not set")
    endif()

    get_filename_component(_codegen_dir "${CMAKE_CURRENT_LIST_DIR}/../tensilelite" ABSOLUTE)

    if(NOT _cdl_TARGET)
        set(_cdl_TARGET "tensilelite-device-libraries")
    endif()
    if(NOT _cdl_ARCHES)
        set(_cdl_ARCHES ${GPU_TARGETS})
    endif()
    if(NOT _cdl_ARCHES)
        message(FATAL_ERROR "hipblaslt_create_device_library: no ARCHES given and GPU_TARGETS is empty")
    endif()
    if(NOT _cdl_CXX_COMPILER)
        set(_cdl_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
    endif()
    if(NOT DEFINED _cdl_LAZY_LOAD)
        set(_cdl_LAZY_LOAD ON)
    endif()

    file(MAKE_DIRECTORY "${_cdl_OUTPUT_DIR}/library")

    list(JOIN _cdl_ARCHES "$<SEMICOLON>" _arches_semi)
    set(_opts_list "--architecture=${_arches_semi}" "--cxx-compiler=${_cdl_CXX_COMPILER}")
    if(_cdl_OFFLOAD_BUNDLER)
        list(APPEND _opts_list "--offload-bundler=${_cdl_OFFLOAD_BUNDLER}")
    endif()
    if(_cdl_ASAN)
        list(APPEND _opts_list "--address-sanitizer")
    endif()
    if(_cdl_JOBS)
        list(APPEND _opts_list "--jobs=${_cdl_JOBS}")
    endif()
    if(_cdl_KEEP_BUILD_TMP)
        list(APPEND _opts_list "--keep-build-tmp")
    endif()
    if(_cdl_ASM_DEBUG)
        list(APPEND _opts_list "--asm-debug")
    endif()
    if(_cdl_YAML_FORMAT)
        list(APPEND _opts_list "--library-format=yaml")
    endif()
    if(_cdl_LOGIC_FILTER)
        list(APPEND _opts_list "--logic-filter=${_cdl_LOGIC_FILTER}")
    endif()
    if(_cdl_NO_COMPRESS)
        list(APPEND _opts_list "--no-compress")
    endif()
    if(_cdl_EXPERIMENTAL)
        list(APPEND _opts_list "--experimental")
    endif()
    if(NOT _cdl_LAZY_LOAD)
        list(APPEND _opts_list "--no-lazy-library-loading")
    endif()
    if(NOT _cdl_ASM_COMMENTS)
        list(APPEND _opts_list "--disable-asm-comments")
    endif()

    # Keep the resource as an explicit dependency so quarantine edits invalidate
    # the validation stamp; TensileLogic itself resolves the resource at runtime.
    set(_known_bugs_resource "${_codegen_dir}/Tensile/TensileLogic/known_bugs.yaml")
    set(_logic_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}-TensileLogic.stamp")
    add_custom_command(
        OUTPUT "${_logic_stamp}"
        COMMENT "Validating library logic (TensileLogic --check-all) for ${_cdl_TARGET} ..."
        COMMAND ${HIPBLASLT_PYTHON_COMMAND}
            "${_codegen_dir}/Tensile/bin/TensileLogic"
            "${_cdl_LOGIC_PATH}"
            --architecture
            "${_arches_semi}"
            --use-bundled-known-bugs
            --check-all
        COMMAND ${CMAKE_COMMAND} -E touch "${_logic_stamp}"
        DEPENDS ${HIPBLASLT_PYTHON_DEPS} "${_known_bugs_resource}"
        VERBATIM
        USES_TERMINAL
    )

    set(_output_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_cdl_TARGET}.stamp")
    set(_tcl_command
        ${HIPBLASLT_PYTHON_COMMAND} -m Tensile.TensileCreateLibrary
        ${_opts_list}
        "${_cdl_LOGIC_PATH}"
        "${_cdl_OUTPUT_DIR}"
        HIP
    )
    add_custom_command(
        OUTPUT "${_output_stamp}"
        COMMENT "Building device libraries to ${_cdl_OUTPUT_DIR} ..."
        COMMAND ${_tcl_command}
        COMMAND ${CMAKE_COMMAND} -E touch "${_output_stamp}"
        DEPENDS ${HIPBLASLT_PYTHON_DEPS} "${_logic_stamp}"
        VERBATIM
        USES_TERMINAL
    )

    block(SCOPE_FOR VARIABLES)
        list(JOIN _tcl_command " " _formatted_tcl)
        message(STATUS "Device lib build command (${_cdl_TARGET}): ${_formatted_tcl}")
    endblock()

    add_custom_target(${_cdl_TARGET} ALL
        DEPENDS "${_output_stamp}"
    )
endfunction()
