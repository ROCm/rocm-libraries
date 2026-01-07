# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Backward compatibility shim for legacy build option names
# This file maps old option names to modern ROCSPARSE_* prefixed names
# and provides deprecation warnings to guide users to the new names.

# Helper macros for deprecation warnings and conflict detection
macro(_rocsparse_deprecation_warning old_var new_var)
    message(DEPRECATION 
        "Option '${old_var}' is deprecated and will be removed in a future release.\n"
        "  Please use '-D${new_var}=${${old_var}}' instead.\n"
        "  To suppress this warning, use: -DCMAKE_WARN_DEPRECATED=OFF"
    )
endmacro()

macro(_rocsparse_check_conflict old_var new_var)
    if(DEFINED ${old_var} AND DEFINED ${new_var})
        if(NOT "${${old_var}}" STREQUAL "${${new_var}}")
            message(FATAL_ERROR 
                "Conflicting options detected:\n"
                "  ${old_var}=${${old_var}}\n"
                "  ${new_var}=${${new_var}}\n"
                "Please use only '${new_var}' going forward."
            )
        endif()
    endif()
endmacro()

# BUILD_CLIENTS_TESTS → ROCSPARSE_BUILD_TESTING
if(DEFINED BUILD_CLIENTS_TESTS)
    _rocsparse_check_conflict(BUILD_CLIENTS_TESTS ROCSPARSE_BUILD_TESTING)
    if(NOT DEFINED ROCSPARSE_BUILD_TESTING)
        set(ROCSPARSE_BUILD_TESTING ${BUILD_CLIENTS_TESTS} CACHE BOOL 
            "Build tests (requires googletest)" FORCE)
        _rocsparse_deprecation_warning(BUILD_CLIENTS_TESTS ROCSPARSE_BUILD_TESTING)
    endif()
endif()

# BUILD_TESTING (standard CMake variable) → ROCSPARSE_BUILD_TESTING
# BUILD_TESTING is the standard CMake variable used by CTest, so support it without deprecation
if(DEFINED BUILD_TESTING AND BUILD_TESTING)
    if(NOT DEFINED ROCSPARSE_BUILD_TESTING)
        set(ROCSPARSE_BUILD_TESTING ${BUILD_TESTING} CACHE BOOL 
            "Build tests (requires googletest)" FORCE)
    endif()
endif()

# BUILD_CLIENTS_BENCHMARKS → ROCSPARSE_ENABLE_BENCHMARKS
if(DEFINED BUILD_CLIENTS_BENCHMARKS)
    _rocsparse_check_conflict(BUILD_CLIENTS_BENCHMARKS ROCSPARSE_ENABLE_BENCHMARKS)
    if(NOT DEFINED ROCSPARSE_ENABLE_BENCHMARKS)
        set(ROCSPARSE_ENABLE_BENCHMARKS ${BUILD_CLIENTS_BENCHMARKS} CACHE BOOL 
            "Build benchmarks" FORCE)
        _rocsparse_deprecation_warning(BUILD_CLIENTS_BENCHMARKS ROCSPARSE_ENABLE_BENCHMARKS)
    endif()
endif()

# BUILD_CLIENTS_SAMPLES → ROCSPARSE_ENABLE_SAMPLES
if(DEFINED BUILD_CLIENTS_SAMPLES)
    _rocsparse_check_conflict(BUILD_CLIENTS_SAMPLES ROCSPARSE_ENABLE_SAMPLES)
    if(NOT DEFINED ROCSPARSE_ENABLE_SAMPLES)
        set(ROCSPARSE_ENABLE_SAMPLES ${BUILD_CLIENTS_SAMPLES} CACHE BOOL 
            "Build examples" FORCE)
        _rocsparse_deprecation_warning(BUILD_CLIENTS_SAMPLES ROCSPARSE_ENABLE_SAMPLES)
    endif()
endif()

# BUILD_CLIENTS_ONLY → Warn and ignore (obsolete in modern structure)
if(DEFINED BUILD_CLIENTS_ONLY)
    message(WARNING 
        "Option 'BUILD_CLIENTS_ONLY' is obsolete and has no effect in the modernized build system.\n"
        "  Use individual options like ROCSPARSE_BUILD_TESTING, ROCSPARSE_ENABLE_BENCHMARKS, etc."
    )
endif()

# BUILD_VERBOSE → Warn (anti-pattern, removed)
if(DEFINED BUILD_VERBOSE)
    message(WARNING 
        "Option 'BUILD_VERBOSE' has been removed as it violates CMake best practices.\n"
        "  Use CMAKE_VERBOSE_MAKEFILE or VERBOSE=1 with make instead."
    )
endif()

# BUILD_CODE_COVERAGE → ROCSPARSE_BUILD_COVERAGE
if(DEFINED BUILD_CODE_COVERAGE)
    _rocsparse_check_conflict(BUILD_CODE_COVERAGE ROCSPARSE_BUILD_COVERAGE)
    if(NOT DEFINED ROCSPARSE_BUILD_COVERAGE)
        set(ROCSPARSE_BUILD_COVERAGE ${BUILD_CODE_COVERAGE} CACHE BOOL 
            "Build with code coverage enabled" FORCE)
        _rocsparse_deprecation_warning(BUILD_CODE_COVERAGE ROCSPARSE_BUILD_COVERAGE)
    endif()
endif()

# BUILD_ADDRESS_SANITIZER → ROCSPARSE_ENABLE_ASAN
if(DEFINED BUILD_ADDRESS_SANITIZER)
    _rocsparse_check_conflict(BUILD_ADDRESS_SANITIZER ROCSPARSE_ENABLE_ASAN)
    if(NOT DEFINED ROCSPARSE_ENABLE_ASAN)
        set(ROCSPARSE_ENABLE_ASAN ${BUILD_ADDRESS_SANITIZER} CACHE BOOL 
            "Build with address sanitizer enabled" FORCE)
        _rocsparse_deprecation_warning(BUILD_ADDRESS_SANITIZER ROCSPARSE_ENABLE_ASAN)
    endif()
endif()

# BUILD_MEMSTAT → ROCSPARSE_ENABLE_MEMSTAT
if(DEFINED BUILD_MEMSTAT)
    _rocsparse_check_conflict(BUILD_MEMSTAT ROCSPARSE_ENABLE_MEMSTAT)
    if(NOT DEFINED ROCSPARSE_ENABLE_MEMSTAT)
        set(ROCSPARSE_ENABLE_MEMSTAT ${BUILD_MEMSTAT} CACHE BOOL 
            "Build with memory statistics enabled" FORCE)
        _rocsparse_deprecation_warning(BUILD_MEMSTAT ROCSPARSE_ENABLE_MEMSTAT)
    endif()
endif()

# BUILD_ROCSPARSE_ILP64 → ROCSPARSE_ENABLE_ILP64
if(DEFINED BUILD_ROCSPARSE_ILP64)
    _rocsparse_check_conflict(BUILD_ROCSPARSE_ILP64 ROCSPARSE_ENABLE_ILP64)
    if(NOT DEFINED ROCSPARSE_ENABLE_ILP64)
        set(ROCSPARSE_ENABLE_ILP64 ${BUILD_ROCSPARSE_ILP64} CACHE BOOL 
            "Build with rocsparse_int equal to int64_t" FORCE)
        _rocsparse_deprecation_warning(BUILD_ROCSPARSE_ILP64 ROCSPARSE_ENABLE_ILP64)
    endif()
endif()

# BUILD_COMPRESSED_DBG → ROCSPARSE_ENABLE_COMPRESSED_DBG
if(DEFINED BUILD_COMPRESSED_DBG)
    _rocsparse_check_conflict(BUILD_COMPRESSED_DBG ROCSPARSE_ENABLE_COMPRESSED_DBG)
    if(NOT DEFINED ROCSPARSE_ENABLE_COMPRESSED_DBG)
        set(ROCSPARSE_ENABLE_COMPRESSED_DBG ${BUILD_COMPRESSED_DBG} CACHE BOOL 
            "Enable compressed debug symbols" FORCE)
        _rocsparse_deprecation_warning(BUILD_COMPRESSED_DBG ROCSPARSE_ENABLE_COMPRESSED_DBG)
    endif()
endif()

# BUILD_WITH_ROCBLAS → ROCSPARSE_ENABLE_ROCBLAS
if(DEFINED BUILD_WITH_ROCBLAS)
    _rocsparse_check_conflict(BUILD_WITH_ROCBLAS ROCSPARSE_ENABLE_ROCBLAS)
    if(NOT DEFINED ROCSPARSE_ENABLE_ROCBLAS)
        set(ROCSPARSE_ENABLE_ROCBLAS ${BUILD_WITH_ROCBLAS} CACHE BOOL 
            "Enable building rocSPARSE with rocBLAS" FORCE)
        _rocsparse_deprecation_warning(BUILD_WITH_ROCBLAS ROCSPARSE_ENABLE_ROCBLAS)
    endif()
endif()

# BUILD_WITH_ROCTX → ROCSPARSE_ENABLE_MARKER
if(DEFINED BUILD_WITH_ROCTX)
    _rocsparse_check_conflict(BUILD_WITH_ROCTX ROCSPARSE_ENABLE_MARKER)
    if(NOT DEFINED ROCSPARSE_ENABLE_MARKER)
        set(ROCSPARSE_ENABLE_MARKER ${BUILD_WITH_ROCTX} CACHE BOOL 
            "Enable rocTracer marker support" FORCE)
        _rocsparse_deprecation_warning(BUILD_WITH_ROCTX ROCSPARSE_ENABLE_MARKER)
    endif()
endif()

# BUILD_FORTRAN_CLIENTS → ROCSPARSE_ENABLE_FORTRAN
if(DEFINED BUILD_FORTRAN_CLIENTS)
    _rocsparse_check_conflict(BUILD_FORTRAN_CLIENTS ROCSPARSE_ENABLE_FORTRAN)
    if(NOT DEFINED ROCSPARSE_ENABLE_FORTRAN)
        set(ROCSPARSE_ENABLE_FORTRAN ${BUILD_FORTRAN_CLIENTS} CACHE BOOL 
            "Build Fortran clients" FORCE)
        _rocsparse_deprecation_warning(BUILD_FORTRAN_CLIENTS ROCSPARSE_ENABLE_FORTRAN)
    endif()
endif()

# BUILD_DOCS → ROCSPARSE_BUILD_DOCS
if(DEFINED BUILD_DOCS)
    _rocsparse_check_conflict(BUILD_DOCS ROCSPARSE_BUILD_DOCS)
    if(NOT DEFINED ROCSPARSE_BUILD_DOCS)
        set(ROCSPARSE_BUILD_DOCS ${BUILD_DOCS} CACHE BOOL 
            "Build documentation" FORCE)
        _rocsparse_deprecation_warning(BUILD_DOCS ROCSPARSE_BUILD_DOCS)
    endif()
endif()

# BUILD_WITH_OFFLOAD_COMPRESS → Maintain as-is (not prefixed in legacy)
if(DEFINED BUILD_WITH_OFFLOAD_COMPRESS)
    _rocsparse_check_conflict(BUILD_WITH_OFFLOAD_COMPRESS ROCSPARSE_ENABLE_OFFLOAD_COMPRESS)
    if(NOT DEFINED ROCSPARSE_ENABLE_OFFLOAD_COMPRESS)
        set(ROCSPARSE_ENABLE_OFFLOAD_COMPRESS ${BUILD_WITH_OFFLOAD_COMPRESS} CACHE BOOL 
            "Enable offload compression during compilation" FORCE)
        _rocsparse_deprecation_warning(BUILD_WITH_OFFLOAD_COMPRESS ROCSPARSE_ENABLE_OFFLOAD_COMPRESS)
    endif()
endif()

# AMDGPU_TARGETS → GPU_TARGETS
if(DEFINED AMDGPU_TARGETS)
    _rocsparse_check_conflict(AMDGPU_TARGETS GPU_TARGETS)
    if(NOT DEFINED GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS} CACHE STRING 
            "AMD GFX targets to cross-compile" FORCE)
        _rocsparse_deprecation_warning(AMDGPU_TARGETS GPU_TARGETS)
    endif()
endif()

# BUILD_SHARED_LIBS → ROCSPARSE_BUILD_SHARED_LIBS
# Note: BUILD_SHARED_LIBS is a standard CMake variable, so we support it without deprecation
if(DEFINED BUILD_SHARED_LIBS)
    if(NOT DEFINED ROCSPARSE_BUILD_SHARED_LIBS)
        set(ROCSPARSE_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS} CACHE BOOL 
            "Build rocSPARSE as a shared library" FORCE)
    endif()
endif()

# Cleanup: Unset old variables to prevent accidental usage downstream
# This ensures the modern names are authoritative
unset(BUILD_CLIENTS_TESTS CACHE)
unset(BUILD_CLIENTS_BENCHMARKS CACHE)
unset(BUILD_CLIENTS_SAMPLES CACHE)
unset(BUILD_CLIENTS_ONLY CACHE)
unset(BUILD_VERBOSE CACHE)
unset(BUILD_CODE_COVERAGE CACHE)
unset(BUILD_ADDRESS_SANITIZER CACHE)
unset(BUILD_MEMSTAT CACHE)
unset(BUILD_ROCSPARSE_ILP64 CACHE)
unset(BUILD_COMPRESSED_DBG CACHE)
unset(BUILD_WITH_ROCBLAS CACHE)
unset(BUILD_WITH_ROCTX CACHE)
unset(BUILD_FORTRAN_CLIENTS CACHE)
unset(BUILD_DOCS CACHE)
unset(BUILD_WITH_OFFLOAD_COMPRESS CACHE)
unset(AMDGPU_TARGETS CACHE)

