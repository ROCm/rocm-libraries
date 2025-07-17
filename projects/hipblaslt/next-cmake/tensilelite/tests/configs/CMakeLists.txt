# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(CONFIGS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite/HostLibraryTests/configs")

# set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
# file(MAKE_DIRECTORY "${TEST_DATA_DIR}")
# set(SOLUTION_LIBRARY_DIR "${CONFIGS_SRC_DIR}/SolutionLibraries")

if(TENSILE_USE_LLVM)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CONFIGS_SRC_DIR}/SolutionLibraries/Kernels.yaml"
        )
endif()

if(TENSILE_USE_MSGPACK)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CONFIGS_SRC_DIR}/SolutionLibraries/Kernels.dat"
        )
endif()

set(SOLUTION_LIBRARY_FILES
    ${SOLUTION_LIBRARY_FILES_}
    PARENT_SCOPE)
