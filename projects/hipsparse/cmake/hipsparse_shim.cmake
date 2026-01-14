# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# Backward Compatibility Shims for hipSPARSE Build Options
# ==============================================================================
# This file maps legacy build option names to modern project-specific options.
# It provides a transition period for users to update their build scripts while
# maintaining backward compatibility.
#
# Users can suppress deprecation warnings with: -DCMAKE_WARN_DEPRECATED=OFF
#
# Legacy → Modern Mappings:
# ------------------------
# BUILD_CLIENTS_TESTS       → HIPSPARSE_BUILD_TESTING
# BUILD_CLIENTS_BENCHMARKS  → HIPSPARSE_ENABLE_BENCHMARKS
# BUILD_CLIENTS_SAMPLES     → HIPSPARSE_ENABLE_SAMPLES
# BUILD_CLIENTS_ONLY        → HIPSPARSE_BUILD_CLIENTS_ONLY
# USE_CUDA                  → HIPSPARSE_ENABLE_CUDA
# BUILD_CUDA                → HIPSPARSE_ENABLE_CUDA (deprecated in favor of USE_CUDA)
# BUILD_CODE_COVERAGE       → HIPSPARSE_ENABLE_COVERAGE
# BUILD_ADDRESS_SANITIZER   → HIPSPARSE_ENABLE_ASAN
# BUILD_VERBOSE             → CMAKE_VERBOSE_MAKEFILE
# BUILD_DOCS                → HIPSPARSE_BUILD_DOCS
#
# Modern Options (no legacy equivalent):
# --------------------------------------
# HIPSPARSE_ENABLE_HIP      - Build hipSPARSE with HIP backend (default ON)
# CMAKE_VERBOSE_MAKEFILE    - Enable verbose Makefile output
# ==============================================================================

# Helper macro for deprecation warnings using native CMake mechanism
macro(_hipsparse_deprecation_warning old_var new_var)
    message(DEPRECATION
        "The option '${old_var}' is deprecated and will be removed in a future release.\n"
        "Please use '${new_var}' instead.\n"
        "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
endmacro()

# Helper macro for conflict detection
macro(_hipsparse_check_conflict old_var new_var)
    # If new variable is defined, unset any cached legacy variable to avoid conflicts
    # This allows smooth migration from legacy to modern options without requiring build dir removal
    if(DEFINED ${new_var})
        unset(${old_var} CACHE)
    endif()

    # Only check for actual conflicts if both are actively being set
    if(DEFINED ${old_var} AND DEFINED ${new_var})
        if(NOT "${${old_var}}" STREQUAL "${${new_var}}")
            message(FATAL_ERROR
                "Conflicting options detected:\n"
                "  ${old_var}=${${old_var}} (deprecated)\n"
                "  ${new_var}=${${new_var}}\n"
                "Please remove ${old_var} from your build configuration and use only ${new_var}.")
        endif()
    endif()
endmacro()

# Consolidated function for mapping legacy options to modern equivalents
# Usage: shim_mapping(<legacy_var> <modern_var> <description> [<type>])
# Type defaults to BOOL if not specified
function(shim_mapping legacy_var modern_var description)
    set(var_type "${ARGV3}")
    if(NOT var_type)
        set(var_type "BOOL")
    endif()

    if(DEFINED ${legacy_var})
        _hipsparse_check_conflict(${legacy_var} ${modern_var})
        if(NOT DEFINED ${modern_var})
            set(${modern_var} ${${legacy_var}} CACHE ${var_type} "${description}" FORCE)
            _hipsparse_deprecation_warning(${legacy_var} ${modern_var})
            list(APPEND _HIPSPARSE_LEGACY_OPTIONS_USED "${legacy_var}=${${legacy_var}}")
            list(APPEND _HIPSPARSE_CURRENT_OPTIONS "${modern_var}=${${modern_var}}")
            # Propagate lists to parent scope
            set(_HIPSPARSE_LEGACY_OPTIONS_USED "${_HIPSPARSE_LEGACY_OPTIONS_USED}" PARENT_SCOPE)
            set(_HIPSPARSE_CURRENT_OPTIONS "${_HIPSPARSE_CURRENT_OPTIONS}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

# ==============================================================================
# Apply Legacy Option Mappings
# ==============================================================================

# Initialize tracking lists for migration guidance
set(_HIPSPARSE_LEGACY_OPTIONS_USED "")
set(_HIPSPARSE_CURRENT_OPTIONS "")

# Apply mappings using consolidated function
shim_mapping(BUILD_CLIENTS_TESTS HIPSPARSE_BUILD_TESTING "Build test client; master switch.")
shim_mapping(BUILD_CLIENTS_BENCHMARKS HIPSPARSE_ENABLE_BENCHMARKS "Build benchmark client.")
shim_mapping(BUILD_CLIENTS_SAMPLES HIPSPARSE_ENABLE_SAMPLES "Build client samples.")
shim_mapping(BUILD_CODE_COVERAGE HIPSPARSE_ENABLE_COVERAGE "Build with code coverage enabled.")
shim_mapping(BUILD_ADDRESS_SANITIZER HIPSPARSE_ENABLE_ASAN "Build with address sanitizer enabled.")
shim_mapping(BUILD_VERBOSE CMAKE_VERBOSE_MAKEFILE "Enable verbose output from Makefile builds.")
shim_mapping(BUILD_DOCS HIPSPARSE_BUILD_DOCS "Build documentation.")

# Special case: BUILD_CLIENTS_ONLY → HIPSPARSE_ENABLE_HOST (inverted)
# When building clients only, we're NOT building the host library
if(DEFINED BUILD_CLIENTS_ONLY)
    if(NOT DEFINED HIPSPARSE_ENABLE_HOST)
        if(BUILD_CLIENTS_ONLY)
            set(HIPSPARSE_ENABLE_HOST OFF CACHE BOOL "Build hipSPARSE library." FORCE)
            set(HIPSPARSE_BUILD_CLIENTS_ONLY ON CACHE BOOL "Build only clients." FORCE)
        else()
            set(HIPSPARSE_ENABLE_HOST ON CACHE BOOL "Build hipSPARSE library." FORCE)
            set(HIPSPARSE_BUILD_CLIENTS_ONLY OFF CACHE BOOL "Build only clients." FORCE)
        endif()
        _hipsparse_deprecation_warning(BUILD_CLIENTS_ONLY HIPSPARSE_ENABLE_HOST)
        list(APPEND _HIPSPARSE_LEGACY_OPTIONS_USED "BUILD_CLIENTS_ONLY=${BUILD_CLIENTS_ONLY}")
        list(APPEND _HIPSPARSE_CURRENT_OPTIONS "HIPSPARSE_ENABLE_HOST=${HIPSPARSE_ENABLE_HOST}")
    endif()
endif()

# Special case: USE_CUDA → HIPSPARSE_ENABLE_CUDA
# Note: In the old build system, USE_CUDA controlled the backend selection
if(DEFINED USE_CUDA)
    _hipsparse_check_conflict(USE_CUDA HIPSPARSE_ENABLE_CUDA)
    if(NOT DEFINED HIPSPARSE_ENABLE_CUDA)
        set(HIPSPARSE_ENABLE_CUDA ${USE_CUDA} CACHE BOOL "Build hipSPARSE with CUDA backend." FORCE)
        _hipsparse_deprecation_warning(USE_CUDA HIPSPARSE_ENABLE_CUDA)
        list(APPEND _HIPSPARSE_LEGACY_OPTIONS_USED "USE_CUDA=${USE_CUDA}")
        list(APPEND _HIPSPARSE_CURRENT_OPTIONS "HIPSPARSE_ENABLE_CUDA=${HIPSPARSE_ENABLE_CUDA}")
        
        # Auto-disable HIP when CUDA is enabled (unless user explicitly set HIP)
        if(HIPSPARSE_ENABLE_CUDA AND NOT DEFINED HIPSPARSE_ENABLE_HIP)
            set(HIPSPARSE_ENABLE_HIP OFF CACHE BOOL "Build hipSPARSE with HIP backend." FORCE)
        endif()
    endif()
endif()

# Special case: BUILD_CUDA → HIPSPARSE_ENABLE_CUDA
# This was already deprecated in favor of USE_CUDA, now maps to HIPSPARSE_ENABLE_CUDA
if(DEFINED BUILD_CUDA)
    _hipsparse_check_conflict(BUILD_CUDA HIPSPARSE_ENABLE_CUDA)
    if(NOT DEFINED HIPSPARSE_ENABLE_CUDA)
        set(HIPSPARSE_ENABLE_CUDA ${BUILD_CUDA} CACHE BOOL "Build hipSPARSE with CUDA backend." FORCE)
        message(DEPRECATION
            "The option 'BUILD_CUDA' is deprecated and will be removed in a future release.\n"
            "It was previously deprecated in favor of 'USE_CUDA'.\n"
            "Please use 'HIPSPARSE_ENABLE_CUDA' instead.\n"
            "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
        list(APPEND _HIPSPARSE_LEGACY_OPTIONS_USED "BUILD_CUDA=${BUILD_CUDA}")
        list(APPEND _HIPSPARSE_CURRENT_OPTIONS "HIPSPARSE_ENABLE_CUDA=${HIPSPARSE_ENABLE_CUDA}")
        
        # Auto-disable HIP when CUDA is enabled (unless user explicitly set HIP)
        if(HIPSPARSE_ENABLE_CUDA AND NOT DEFINED HIPSPARSE_ENABLE_HIP)
            set(HIPSPARSE_ENABLE_HIP OFF CACHE BOOL "Build hipSPARSE with HIP backend." FORCE)
        endif()
    endif()
endif()

# ==============================================================================
# Platform-Specific Option Validation
# ==============================================================================

# Windows does not support Fortran clients - force disable if enabled
if(WIN32 AND HIPSPARSE_ENABLE_FORTRAN)
    message(DEPRECATION
        "HIPSPARSE_ENABLE_FORTRAN is not supported on Windows.\n"
        "Fortran support is automatically disabled on Windows.\n"
        "Remove -DHIPSPARSE_ENABLE_FORTRAN=ON from your build configuration.\n"
        "To suppress these warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...")
    set(HIPSPARSE_ENABLE_FORTRAN OFF CACHE BOOL "Enable Fortran client support." FORCE)
endif()

# ==============================================================================
# Display Migration Guidance
# ==============================================================================

if(_HIPSPARSE_LEGACY_OPTIONS_USED)
    # Format the current options with -D prefix for each
    string(REPLACE ";" " -D" _formatted_current_options "${_HIPSPARSE_CURRENT_OPTIONS}")
    set(_formatted_current_options "-D${_formatted_current_options}")

    message(DEPRECATION
        "\n"
        "  Legacy build options detected:\n"
        "    ${_HIPSPARSE_LEGACY_OPTIONS_USED}\n"
        "\n"
        "  To use current options, run:\n"
        "    cmake ${_formatted_current_options} ..\n"
        "\n"
        "  To suppress warnings: cmake -DCMAKE_WARN_DEPRECATED=OFF ...\n"
    )
endif()
