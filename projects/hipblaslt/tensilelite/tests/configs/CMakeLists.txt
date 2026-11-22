# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT


set(SOLUTION_LIBRARY_FILES_
    ${SOLUTION_LIBRARY_FILES_}
    "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/Kernels_gfx950.yaml"
    )

set(SOLUTION_LIBRARY_FILES_
    ${SOLUTION_LIBRARY_FILES_}
    "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/Kernels_gfx950.dat"
    )

set(SOLUTION_LIBRARY_FILES_
    ${SOLUTION_LIBRARY_FILES_}
    "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLite.dat"
    )

list(APPEND SOLUTION_LIBRARY_FILES_
    "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/geko_bias_contract.dat"
    "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/geko_bias_contract.yaml")

set(SOLUTION_LIBRARY_FILES
    ${SOLUTION_LIBRARY_FILES_}
    PARENT_SCOPE)
