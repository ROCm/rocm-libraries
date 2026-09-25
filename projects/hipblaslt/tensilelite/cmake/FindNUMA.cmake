# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Find module for libnuma, providing the numa::numa imported target.
#
# This exists for two reasons, one local and one to work around a defect in the
# hsakmt package config shipped by ROCm 7.x.
#
# 1. Local: hsakmt::hsakmt's exported link interface names numa::numa, so that
#    target has to exist before the export is loaded.
#
# 2. Workaround: rocm-systems commit 0adba114d0e2 ("fix(rocr): Fix exported
#    hsakmt NUMA dependency", 2026-07-21) added a bare
#
#        find_dependency(NUMA)
#
#    to hsakmt-config.cmake.in, guarded by HSAKMT_FIND_NUMA_DEPENDENCY -- which
#    is ON exactly when the distribution bundles libnuma as an imported target,
#    as TheRock does. Nothing installs a NUMA package config, though: libnuma-dev
#    ships only numa.h/libnuma.so/numa.pc, CMake has no builtin FindNUMA, and
#    hsakmt does not install its own. So on such a build every consumer of
#    find_package(hsakmt) dies at configure time with
#
#        By not providing "FindNUMA.cmake" in CMAKE_MODULE_PATH this project has
#        asked CMake to find a package configuration file provided by "NUMA" ...
#
#    Putting this module on CMAKE_MODULE_PATH before find_package(hsakmt)
#    satisfies that find_dependency in module mode.
#
#    Upstream fixed the defect in 4d81cbd3496f ("fix(rocr): Resolve NUMA
#    dependency in exported hsakmt config (ROCM-21395)", 2026-07-29) by forcing
#    CONFIG mode against the bundled config under lib/rocm_sysdeps. A ROCm built
#    from that commit or later ignores this module entirely -- CONFIG mode does
#    not consult CMAKE_MODULE_PATH -- so this stays correct either way and can be
#    deleted once every ROCm we build against postdates the fix.
#
# Sets NUMA_FOUND, NUMA_LIBRARIES, NUMA_INCLUDE_DIR and defines numa::numa.

include(FindPackageHandleStandardArgs)

# hsakmt-config.cmake re-enters this module via find_dependency(NUMA) after the
# client has already called find_package(NUMA), so everything below has to be
# safe to run twice. The find_* results are cached and the target creation is
# guarded, which is what makes that so.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

# ROCm 7.x vendors libnuma into ${ROCM_PATH}/lib/rocm_sysdeps/lib (renamed
# librocm_sysdeps_numa.so.1, with a libnuma.so symlink beside it) and is NOT on
# the default library search path, so a bare find_library misses it. Search
# there first: hsakmt itself links the vendored copy, and loading a second
# system libnuma into the same process alongside it is worth avoiding.
find_library(NUMA_LIBRARY
    NAMES numa
    HINTS "${ROCM_PATH}/lib/rocm_sysdeps/lib"
    PATHS "${ROCM_PATH}/lib/rocm_sysdeps/lib")

find_path(NUMA_INCLUDE_DIR
    NAMES numa.h
    HINTS "${ROCM_PATH}/lib/rocm_sysdeps/include"
    PATHS "${ROCM_PATH}/lib/rocm_sysdeps/include")

find_package_handle_standard_args(NUMA
    REQUIRED_VARS NUMA_LIBRARY NUMA_INCLUDE_DIR
    REASON_FAILURE_MESSAGE
        "Searched the ROCm-vendored path '${ROCM_PATH}/lib/rocm_sysdeps/lib' and \
the default system paths. Install libnuma-dev, set ROCM_PATH if your ROCm is \
elsewhere, or point NUMA_LIBRARY/NUMA_INCLUDE_DIR at the libnuma to use.")

if(NUMA_FOUND)
    set(NUMA_LIBRARIES "${NUMA_LIBRARY}")
    if(NOT TARGET numa::numa)
        add_library(numa::numa UNKNOWN IMPORTED)
        set_target_properties(numa::numa PROPERTIES
            IMPORTED_LOCATION "${NUMA_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(NUMA_LIBRARY NUMA_INCLUDE_DIR)
