# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(ENABLE_CLANG_TIDY)
    get_filename_component(CLANG_TIDY_EXE_HINT "${CMAKE_CXX_COMPILER}" PATH)
    find_program(CLANG_TIDY_EXE
        NAMES
            clang-tidy
        HINTS
            ${CLANG_TIDY_EXE_HINT}
        PATH_SUFFIXES
            compiler/bin
        PATHS
            ${ROCM_PATH}/llvm/bin
            /usr/local/opt/llvm/bin
    )

    function(find_clang_tidy_version VAR)
        execute_process(COMMAND ${CLANG_TIDY_EXE} -version OUTPUT_VARIABLE VERSION_OUTPUT)
        separate_arguments(VERSION_OUTPUT_LIST NATIVE_COMMAND "${VERSION_OUTPUT}")
        list(FIND VERSION_OUTPUT_LIST "version" VERSION_INDEX)
        if(VERSION_INDEX GREATER 0)
            math(EXPR VERSION_INDEX "${VERSION_INDEX} + 1")
            list(GET VERSION_OUTPUT_LIST ${VERSION_INDEX} VERSION)
            set(${VAR} ${VERSION} PARENT_SCOPE)
        else()
            set(${VAR} "0.0" PARENT_SCOPE)
        endif()

    endfunction()

    if( NOT CLANG_TIDY_EXE )
        message( STATUS "Clang tidy not found" )
        set(CLANG_TIDY_VERSION "0.0")
    else()
        find_clang_tidy_version(CLANG_TIDY_VERSION)
        message( STATUS "Clang tidy found: ${CLANG_TIDY_VERSION}. Path:${CLANG_TIDY_EXE}")
    endif()

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    set(CLANG_TIDY_COMMAND 
        ${CLANG_TIDY_EXE}
        --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
        -p ${CMAKE_BINARY_DIR}
    )
endif()

function(clang_tidy_check TARGET)
    if(ENABLE_CLANG_TIDY)
        set_target_properties(${TARGET} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
    endif()
endfunction()
