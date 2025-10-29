# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)

if(ENABLE_CLANG_TIDY)
    findAndCheckClangTidy()

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
