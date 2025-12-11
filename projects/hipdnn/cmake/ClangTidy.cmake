# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)
include(ProcessorCount)

if(ENABLE_CLANG_TIDY)
    findandcheckclangtidy()

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    set(CLANG_TIDY_HIP_ARGS -extra-arg=-D__HIP_PLATFORM_AMD__ -extra-arg=-D__HIPCC__
                            -extra-arg=-isystem -extra-arg=${ROCM_PATH}/include
    )

    set(CLANG_TIDY_COMMAND ${CLANG_TIDY_EXE} -config-file=${CMAKE_SOURCE_DIR}/.clang-tidy -p
                           ${CMAKE_BINARY_DIR} ${CLANG_TIDY_HIP_ARGS}
    )

    # Alternatively, this could be separate from the ENABLE_CLANG_TIDY flag
    if(RUN_CLANG_TIDY_EXE)
        processorcount(N)
        if(NOT N EQUAL 0)
            set(CLANG_TIDY_JOBS ${N})
        else()
            set(CLANG_TIDY_JOBS 1)
        endif()

        add_custom_target(
            tidy
            COMMAND
                ${RUN_CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
                -config-file=${CMAKE_SOURCE_DIR}/.clang-tidy -source-filter "^(?!.*_deps/).*" -quiet
                -j ${CLANG_TIDY_JOBS} ${CLANG_TIDY_HIP_ARGS}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT
                "Running clang-tidy on all source files and headers (${CLANG_TIDY_JOBS} parallel jobs)..."
            VERBATIM
        )
    else()
        message(WARNING "run-clang-tidy-20 not found. The 'tidy' target will not be available.")
    endif()
endif()

# Enable clang-tidy checks for a specific target during compilation.
#
# @param TARGET target to enable clang-tidy checks for
function(clang_tidy_check TARGET)
    if(ENABLE_CLANG_TIDY)
        set_target_properties(${TARGET} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
        set_target_properties(${TARGET} PROPERTIES HIP_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
        set_target_properties(${TARGET} PROPERTIES C_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
    endif()
endfunction()
