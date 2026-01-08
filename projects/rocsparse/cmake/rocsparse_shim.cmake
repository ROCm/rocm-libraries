# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Backward compatibility shim for legacy build option names
# This file maps old option names to modern ROCSPARSE_* prefixed names
# and provides deprecation warnings to guide users to the new names.

# shim_mapping(old_var new_var type description [NO_DEPRECATION])
# Maps a legacy option to a modern option name with optional deprecation warning.
#   old_var       - The legacy option name
#   new_var       - The modern option name
#   type          - Cache variable type (BOOL or STRING)
#   description   - Cache variable description
#   NO_DEPRECATION - Optional flag to suppress deprecation warning (for standard CMake vars)
macro(shim_mapping old_var new_var type description)
    # Parse optional NO_DEPRECATION argument
    set(_shim_no_deprecation FALSE)
    if("${ARGN}" STREQUAL "NO_DEPRECATION")
        set(_shim_no_deprecation TRUE)
    endif()

    if(DEFINED ${old_var})
        # Check for conflicting values
        if(DEFINED ${new_var})
            if(NOT "${${old_var}}" STREQUAL "${${new_var}}")
                message(FATAL_ERROR
                    "Conflicting options detected:\n"
                    "  ${old_var}=${${old_var}}\n"
                    "  ${new_var}=${${new_var}}\n"
                    "Please use only '${new_var}' going forward."
                )
            endif()
        else()
            # Set modern variable from legacy
            set(${new_var} ${${old_var}} CACHE ${type} "${description}" FORCE)
            if(NOT _shim_no_deprecation)
                message(DEPRECATION
                    "Option '${old_var}' is deprecated and will be removed in a future release.\n"
                    "  Please use '-D${new_var}=${${old_var}}' instead.\n"
                    "  To suppress this warning, use: -DCMAKE_WARN_DEPRECATED=OFF"
                )
            endif()
        endif()
        # Cleanup old variable
        unset(${old_var} CACHE)
    endif()
    unset(_shim_no_deprecation)
endmacro()

# shim_obsolete(old_var message)
# Warns about an obsolete option that has no replacement.
macro(shim_obsolete old_var warning_message)
    if(DEFINED ${old_var})
        message(WARNING "${warning_message}")
        unset(${old_var} CACHE)
    endif()
endmacro()

# =============================================================================
# Option Mappings
# =============================================================================

# Deprecated options → Modern ROCSPARSE_* names
shim_mapping(BUILD_CLIENTS_TESTS      ROCSPARSE_BUILD_TESTING            BOOL "Build tests (requires googletest)")
shim_mapping(BUILD_CLIENTS_BENCHMARKS ROCSPARSE_ENABLE_BENCHMARKS        BOOL "Build benchmarks")
shim_mapping(BUILD_CLIENTS_SAMPLES    ROCSPARSE_ENABLE_SAMPLES           BOOL "Build examples")
shim_mapping(BUILD_CODE_COVERAGE      ROCSPARSE_BUILD_COVERAGE           BOOL "Build with code coverage enabled")
shim_mapping(BUILD_ADDRESS_SANITIZER  ROCSPARSE_ENABLE_ASAN              BOOL "Build with address sanitizer enabled")
shim_mapping(BUILD_MEMSTAT            ROCSPARSE_ENABLE_MEMSTAT           BOOL "Build with memory statistics enabled")
shim_mapping(BUILD_ROCSPARSE_ILP64    ROCSPARSE_ENABLE_ILP64             BOOL "Build with rocsparse_int equal to int64_t")
shim_mapping(BUILD_COMPRESSED_DBG     ROCSPARSE_ENABLE_COMPRESSED_DBG    BOOL "Enable compressed debug symbols")
shim_mapping(BUILD_WITH_ROCBLAS       ROCSPARSE_ENABLE_ROCBLAS           BOOL "Enable building rocSPARSE with rocBLAS")
shim_mapping(BUILD_WITH_ROCTX         ROCSPARSE_ENABLE_MARKER            BOOL "Enable rocTracer marker support")
shim_mapping(BUILD_FORTRAN_CLIENTS    ROCSPARSE_ENABLE_FORTRAN           BOOL "Build Fortran clients")
shim_mapping(BUILD_DOCS               ROCSPARSE_BUILD_DOCS               BOOL "Build documentation")
shim_mapping(BUILD_WITH_OFFLOAD_COMPRESS ROCSPARSE_ENABLE_OFFLOAD_COMPRESS BOOL "Enable offload compression during compilation")
shim_mapping(AMDGPU_TARGETS           GPU_TARGETS                        STRING "AMD GFX targets to cross-compile")

# Standard CMake variables - supported without deprecation warning
shim_mapping(BUILD_TESTING     ROCSPARSE_BUILD_TESTING      BOOL "Build tests (requires googletest)" NO_DEPRECATION)
shim_mapping(BUILD_SHARED_LIBS ROCSPARSE_BUILD_SHARED_LIBS  BOOL "Build rocSPARSE as a shared library" NO_DEPRECATION)

# Obsolete options with no replacement
shim_obsolete(BUILD_CLIENTS_ONLY
    "Option 'BUILD_CLIENTS_ONLY' is obsolete and has no effect in the modernized build system.\n"
    "  Use individual options like ROCSPARSE_BUILD_TESTING, ROCSPARSE_ENABLE_BENCHMARKS, etc."
)

shim_obsolete(BUILD_VERBOSE
    "Option 'BUILD_VERBOSE' has been removed as it violates CMake best practices.\n"
    "  Use CMAKE_VERBOSE_MAKEFILE or VERBOSE=1 with make instead."
)
