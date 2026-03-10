# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Backward-compatibility shim for renamed hipSOLVER cache variables.
# Maps old BUILD_* / USE_* names to new HIPSOLVER_* names.
# Include this BEFORE defining options so user-provided old names take effect.

macro(_hipsolver_deprecate_option old_name new_name)
    if(DEFINED ${old_name} AND NOT DEFINED ${new_name})
        message(DEPRECATION
            "${old_name} is deprecated. Use ${new_name} instead."
        )
        set(${new_name} "${${old_name}}" CACHE BOOL "" FORCE)
    elseif(DEFINED ${old_name} AND DEFINED ${new_name})
        message(WARNING
            "Both ${old_name} and ${new_name} are defined. "
            "Using ${new_name}=${${new_name}}, ignoring ${old_name}=${${old_name}}."
        )
    endif()
endmacro()

macro(_hipsolver_deprecate_string_option old_name new_name)
    if(DEFINED ${old_name} AND NOT DEFINED ${new_name})
        message(DEPRECATION
            "${old_name} is deprecated. Use ${new_name} instead."
        )
        set(${new_name} "${${old_name}}" CACHE STRING "" FORCE)
    elseif(DEFINED ${old_name} AND DEFINED ${new_name})
        message(WARNING
            "Both ${old_name} and ${new_name} are defined. "
            "Using ${new_name}=${${new_name}}, ignoring ${old_name}=${${old_name}}."
        )
    endif()
endmacro()

# Boolean options
_hipsolver_deprecate_option(BUILD_ADDRESS_SANITIZER   HIPSOLVER_ENABLE_ASAN)
_hipsolver_deprecate_option(BUILD_CODE_COVERAGE       HIPSOLVER_BUILD_COVERAGE)
_hipsolver_deprecate_option(BUILD_HIPBLAS_TESTS       HIPSOLVER_ENABLE_HIPBLAS_TESTS)
_hipsolver_deprecate_option(BUILD_HIPSPARSE_TESTS     HIPSOLVER_ENABLE_HIPSPARSE_TESTS)
_hipsolver_deprecate_option(BUILD_WITH_SPARSE         HIPSOLVER_ENABLE_SPARSE)
_hipsolver_deprecate_option(BUILD_VERBOSE             HIPSOLVER_ENABLE_VERBOSE)
_hipsolver_deprecate_option(USE_CUDA                  HIPSOLVER_ENABLE_CUDA)
_hipsolver_deprecate_option(BUILD_FORTRAN_BINDINGS    HIPSOLVER_ENABLE_FORTRAN)
_hipsolver_deprecate_option(BUILD_CLIENTS_TESTS       HIPSOLVER_BUILD_TESTING)
_hipsolver_deprecate_option(BUILD_CLIENTS_BENCHMARKS  HIPSOLVER_ENABLE_BENCHMARKS)
_hipsolver_deprecate_option(BUILD_CLIENTS_SAMPLES     HIPSOLVER_ENABLE_SAMPLES)

# String/numeric options
_hipsolver_deprecate_string_option(ARMOR_LEVEL        HIPSOLVER_ARMOR_LEVEL)
