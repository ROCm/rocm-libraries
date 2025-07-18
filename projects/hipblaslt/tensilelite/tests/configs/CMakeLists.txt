# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT


if(TENSILE_USE_LLVM)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/Kernels.yaml"
        )
endif()

if(TENSILE_USE_MSGPACK)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/Kernels.dat"
        )
endif()

set(SOLUTION_LIBRARY_FILES
    ${SOLUTION_LIBRARY_FILES_}
    PARENT_SCOPE)
