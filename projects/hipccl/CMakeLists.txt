# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# hipCCL layout selector.
#
# This file exists purely to redirect a single `cmake` invocation to one of
# two layouts living side by side under this directory:
#   - hipccl3/ (default): the unified hipCCL superbuild - one version, one
#     install layout, one CPack package, one find_package(hipccl) entry
#     point. See hipccl3/CMakeLists.txt and README.md for details.
#   - hipccl2/: the legacy layout - rocPRIM, hipCUB, and rocThrust built and
#     packaged as three independent projects, matching how they build today
#     outside of any hipccl unification effort.
#
# This file does NOT change how either layout works when built directly
# (e.g. `cd hipccl3 && cmake ..`, or `cd hipccl2/rocprim && cmake ..`) - it is
# purely an optional, additional entry point.
cmake_minimum_required(VERSION 3.25.2)
project(hipccl-selector LANGUAGES NONE)

option(HIPCCL_BUILD_LEGACY
    "Build the legacy hipccl2 layout (rocPRIM, hipCUB, rocThrust as independent projects) instead of the unified hipccl3 project"
    OFF)

if(HIPCCL_BUILD_LEGACY)
    message(STATUS "hipccl: building the legacy hipccl2 layout (rocPRIM, hipCUB, rocThrust as independent projects)")

    # NOTE: packaging (`make package`) is skipped by default for all three
    # projects when reached this way - see hipccl2/rocprim, hipcub, and
    # rocthrust's own CMakeLists.txt for the ROCPRIM_PROJECT_IS_TOP_LEVEL /
    # HIPCUB_PROJECT_IS_TOP_LEVEL / ROCTHRUST_PROJECT_IS_TOP_LEVEL guards.
    # This is intentional: all three call rocm_create_package() (which
    # triggers include(CPack)), and CPack only supports one such call per
    # configure. Configure, build, and install all work normally regardless.
    #
    # To get packaging back for exactly one of the three when building via
    # this router, force its flag on the command line, e.g.:
    #   cmake -S projects/hipccl -B build -DHIPCCL_BUILD_LEGACY=ON -DROCPRIM_PROJECT_IS_TOP_LEVEL=ON
    add_subdirectory(hipccl2/rocprim)
    add_subdirectory(hipccl2/hipcub)
    add_subdirectory(hipccl2/rocthrust)
else()
    message(STATUS "hipccl: building the unified hipccl3 project")
    add_subdirectory(hipccl3)
endif()
