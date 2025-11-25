# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# If building clients only, find the hipsparse package
if(HIPSPARSE_BUILD_CLIENTS_ONLY)
    find_package(hipsparse 4.0.0 REQUIRED)
endif()

# Add common subdirectory for shared client code
add_subdirectory(common)

# Add client subdirectories based on options
if(HIPSPARSE_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

if(HIPSPARSE_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(HIPSPARSE_BUILD_TESTING OR BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
