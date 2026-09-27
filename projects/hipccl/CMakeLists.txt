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
#   - hipccl2/: the legacy layout - rocPRIM, hipCUB, and rocThrust as a
#     frozen compatibility snapshot, not actively kept in sync with upstream.
#     Its own root CMakeLists.txt mirrors hipccl3's unified build/version/
#     install/packaging behavior (same HIPCCL_BUILD_ROCPRIM/HIPCUB/ROCTHRUST/
#     ALL flags, same unified "hipccl" package and find_package(hipccl)
#     config) so CI can build either layout through this selector with an
#     identical CMake variable surface.
#
# This file does NOT change how either layout works when built directly
# (e.g. `cd hipccl3 && cmake ..`, or `cd hipccl2 && cmake ..`) - it is purely
# an optional, additional entry point.
cmake_minimum_required(VERSION 3.25.2)
project(hipccl-selector LANGUAGES NONE)

option(HIPCCL_BUILD_LEGACY
    "Build the legacy hipccl2 layout (rocPRIM, hipCUB, rocThrust as independent projects) instead of the unified hipccl3 project"
    OFF)

if(HIPCCL_BUILD_LEGACY)
    message(STATUS "hipccl: building the legacy hipccl2 layout")

    # NOTE: hipccl2 has its own root CMakeLists.txt mirroring hipccl3's
    # unified build/version/install/packaging behavior (same
    # HIPCCL_BUILD_ROCPRIM/HIPCUB/ROCTHRUST/ALL flags, same unified "hipccl"
    # package and find_package(hipccl) config) - see hipccl2/CMakeLists.txt.
    # The only difference is which copies of rocPRIM/hipCUB/rocThrust get
    # built: hipccl2's are a frozen pre-CCCL 3.0 compatibility snapshot, not actively kept
    # in sync with upstream the way hipccl3's are intended to be.
    add_subdirectory(hipccl2)
else()
    message(STATUS "hipccl: building the unified hipccl3 project")
    add_subdirectory(hipccl3)
endif()
