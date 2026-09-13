# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

#[=======================================================================[.rst:
FindCBLAS
---------

Finds a provider-neutral LP64 CBLAS implementation suitable for
ROCHostNumerics.

The module honors the standard FindBLAS inputs, including ``BLA_VENDOR``,
``BLA_STATIC``, ``BLA_PREFER_PKGCONFIG``, and ``BLA_PKGCONFIG_BLAS``. An
explicit ``BLA_SIZEOF_INTEGER`` must be ``4`` or ``ANY`` because the
host-numerics backend uses 32-bit CBLAS dimensions.

Result variables:

``CBLAS_FOUND``
  True when the header, ABI, and required GEMM symbols are available.
``CBLAS_INCLUDE_DIRS``
  Directory containing ``cblas.h``.
``CBLAS_LIBRARIES``
  Complete link interface used by ``CBLAS::CBLAS``.
``CBLAS_INTEGER_SIZE``
  The validated CBLAS integer size, always ``4``.
``CBLAS_BUILD_PROVIDER``
  Human-readable description of the resolved link interface.

Imported targets:

``CBLAS::CBLAS``
  The validated CBLAS include and link interface.
#]=======================================================================]

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
include(FindPackageHandleStandardArgs)

set(_CBLAS_requested_vendor "${BLA_VENDOR}")
if(NOT _CBLAS_requested_vendor)
    set(_CBLAS_requested_vendor "All")
endif()
set(_CBLAS_requested_static "${BLA_STATIC}")
if(NOT DEFINED BLA_STATIC)
    set(_CBLAS_requested_static OFF)
endif()

set(_CBLAS_integer_request_valid TRUE)
if(DEFINED BLA_SIZEOF_INTEGER
   AND NOT "${BLA_SIZEOF_INTEGER}" STREQUAL ""
   AND NOT "${BLA_SIZEOF_INTEGER}" STREQUAL "ANY"
   AND NOT "${BLA_SIZEOF_INTEGER}" STREQUAL "4")
    set(_CBLAS_integer_request_valid FALSE)
    string(
        CONCAT
        _CBLAS_failure_reason
        "ROCHostNumerics requires the LP64 CBLAS ABI "
        "(BLA_SIZEOF_INTEGER=4), but BLA_SIZEOF_INTEGER was set to "
        "\"${BLA_SIZEOF_INTEGER}\"."
    )
endif()

if(_CBLAS_integer_request_valid)
    set(_CBLAS_had_integer_request FALSE)
    if(DEFINED BLA_SIZEOF_INTEGER)
        set(_CBLAS_had_integer_request TRUE)
        set(_CBLAS_saved_integer_request "${BLA_SIZEOF_INTEGER}")
    endif()

    set(BLA_SIZEOF_INTEGER 4)
    find_package(BLAS QUIET)

    if(_CBLAS_had_integer_request)
        set(BLA_SIZEOF_INTEGER "${_CBLAS_saved_integer_request}")
    else()
        unset(BLA_SIZEOF_INTEGER)
    endif()
endif()

# FindBLAS has provided BLAS::BLAS since CMake 3.18. Normalize custom or older
# FindBLAS modules that return only raw variables so the exported
# host-numerics target can depend on one stable semantic target.
if(BLAS_FOUND AND NOT TARGET BLAS::BLAS AND BLAS_LIBRARIES)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
    set_target_properties(
        BLAS::BLAS
        PROPERTIES
            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
    )
    if(BLAS_LINKER_FLAGS)
        set_property(
            TARGET BLAS::BLAS
            PROPERTY INTERFACE_LINK_OPTIONS "${BLAS_LINKER_FLAGS}"
        )
    endif()
endif()

set(_CBLAS_header_hints ${BLAS_INCLUDE_DIRS})
if(TARGET BLAS::BLAS)
    get_target_property(
        _CBLAS_blas_target_includes
        BLAS::BLAS
        INTERFACE_INCLUDE_DIRECTORIES
    )
    if(_CBLAS_blas_target_includes)
        list(APPEND _CBLAS_header_hints ${_CBLAS_blas_target_includes})
    endif()
endif()
find_path(
    CBLAS_INCLUDE_DIR
    NAMES cblas.h
    HINTS ${_CBLAS_header_hints}
)

# The exact function-pointer signatures validate the ABI without relying on a
# provider-specific integer typedef such as MKL_INT, blasint, or CBLAS_INT.
# Global pointer initializers also force all four symbols into the link probe.
set(
    _CBLAS_probe_source
    [=[
#include <cblas.h>

using Layout = decltype(CblasColMajor);
using Transpose = decltype(CblasNoTrans);

using Sgemm = void (*)(Layout,
                       Transpose,
                       Transpose,
                       int,
                       int,
                       int,
                       float,
                       const float*,
                       int,
                       const float*,
                       int,
                       float,
                       float*,
                       int);
using Dgemm = void (*)(Layout,
                       Transpose,
                       Transpose,
                       int,
                       int,
                       int,
                       double,
                       const double*,
                       int,
                       const double*,
                       int,
                       double,
                       double*,
                       int);
using Cgemm = void (*)(Layout,
                       Transpose,
                       Transpose,
                       int,
                       int,
                       int,
                       const void*,
                       const void*,
                       int,
                       const void*,
                       int,
                       const void*,
                       void*,
                       int);
using Zgemm = void (*)(Layout,
                       Transpose,
                       Transpose,
                       int,
                       int,
                       int,
                       const void*,
                       const void*,
                       int,
                       const void*,
                       int,
                       const void*,
                       void*,
                       int);

Sgemm sgemm = &cblas_sgemm;
Dgemm dgemm = &cblas_dgemm;
Cgemm cgemm = &cblas_cgemm;
Zgemm zgemm = &cblas_zgemm;

int main()
{
    return sgemm && dgemm && cgemm && zgemm ? 0 : 1;
}
]=]
)

# Compile and link the CBLAS ABI probe against one candidate link interface.
function(_cblas_check_link_interface result_variable)
    set(_CBLAS_check_libraries)
    foreach(_CBLAS_check_library IN LISTS ARGN)
        if(TARGET "${_CBLAS_check_library}")
            get_target_property(
                _CBLAS_aliased_target
                "${_CBLAS_check_library}"
                ALIASED_TARGET
            )
            if(_CBLAS_aliased_target)
                list(APPEND _CBLAS_check_libraries "${_CBLAS_aliased_target}")
            else()
                list(APPEND _CBLAS_check_libraries "${_CBLAS_check_library}")
            endif()
        else()
            list(APPEND _CBLAS_check_libraries "${_CBLAS_check_library}")
        endif()
    endforeach()

    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_QUIET TRUE)
    set(CMAKE_REQUIRED_INCLUDES "${CBLAS_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES ${_CBLAS_check_libraries})
    unset(_CBLAS_link_interface_works CACHE)
    check_cxx_source_compiles(
        "${_CBLAS_probe_source}"
        _CBLAS_link_interface_works
    )
    cmake_pop_check_state()
    set(
        "${result_variable}"
        "${_CBLAS_link_interface_works}"
        PARENT_SCOPE
    )
endfunction()

set(CBLAS_LINK_INTERFACE)
set(CBLAS_INTEGER_ABI_OK FALSE)
if(_CBLAS_integer_request_valid
   AND CBLAS_INCLUDE_DIR
   AND TARGET BLAS::BLAS)
    _cblas_check_link_interface(_CBLAS_blas_provides_cblas BLAS::BLAS)
    if(_CBLAS_blas_provides_cblas)
        set(CBLAS_LINK_INTERFACE BLAS::BLAS)
        set(CBLAS_INTEGER_ABI_OK TRUE)
    endif()
endif()

# Some environments package the CBLAS wrapper separately from the Fortran BLAS
# implementation. Accept that arrangement without selecting a provider in
# host-numerics itself.
if(_CBLAS_integer_request_valid
   AND CBLAS_INCLUDE_DIR
   AND TARGET BLAS::BLAS
   AND NOT CBLAS_LINK_INTERFACE)
    find_library(CBLAS_LIBRARY NAMES cblas)
    if(CBLAS_LIBRARY)
        _cblas_check_link_interface(
            _CBLAS_separate_library_provides_cblas
            "${CBLAS_LIBRARY}"
            BLAS::BLAS
        )
        if(_CBLAS_separate_library_provides_cblas)
            set(CBLAS_LINK_INTERFACE "${CBLAS_LIBRARY};BLAS::BLAS")
            set(CBLAS_SEPARATE_LIBRARY "${CBLAS_LIBRARY}")
            set(CBLAS_INTEGER_ABI_OK TRUE)
        endif()
    endif()
endif()

set(CBLAS_LIBRARIES "${CBLAS_LINK_INTERFACE}")
set(CBLAS_INCLUDE_DIRS "${CBLAS_INCLUDE_DIR}")
set(CBLAS_INTEGER_SIZE 4)

if(NOT _CBLAS_failure_reason)
    if(NOT BLAS_FOUND)
        set(
            _CBLAS_failure_reason
            "FindBLAS did not locate an LP64 BLAS implementation."
        )
    elseif(NOT TARGET BLAS::BLAS)
        set(
            _CBLAS_failure_reason
            "FindBLAS did not provide BLAS::BLAS or BLAS_LIBRARIES."
        )
    elseif(NOT CBLAS_INCLUDE_DIR)
        set(
            _CBLAS_failure_reason
            "The selected BLAS environment does not provide cblas.h."
        )
    elseif(NOT CBLAS_INTEGER_ABI_OK)
        string(
            CONCAT
            _CBLAS_failure_reason
            "The selected BLAS/CBLAS interface does not provide LP64 "
            "cblas_sgemm, cblas_dgemm, cblas_cgemm, and cblas_zgemm symbols."
        )
    endif()
endif()

find_package_handle_standard_args(
    CBLAS
    REQUIRED_VARS
        CBLAS_INCLUDE_DIR
        CBLAS_LIBRARIES
        CBLAS_INTEGER_ABI_OK
    REASON_FAILURE_MESSAGE "${_CBLAS_failure_reason}"
)

if(CBLAS_FOUND AND NOT TARGET CBLAS::CBLAS)
    add_library(CBLAS::CBLAS INTERFACE IMPORTED)
    set_target_properties(
        CBLAS::CBLAS
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CBLAS_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${CBLAS_LIBRARIES}"
    )
endif()

set(_CBLAS_provider_parts)
if(CBLAS_SEPARATE_LIBRARY)
    list(APPEND _CBLAS_provider_parts "${CBLAS_SEPARATE_LIBRARY}")
endif()
if(BLAS_LIBRARIES)
    list(APPEND _CBLAS_provider_parts ${BLAS_LIBRARIES})
elseif(TARGET BLAS::BLAS)
    list(APPEND _CBLAS_provider_parts "BLAS::BLAS")
endif()
list(REMOVE_DUPLICATES _CBLAS_provider_parts)
string(JOIN " " CBLAS_BUILD_PROVIDER ${_CBLAS_provider_parts})

if(CBLAS_FOUND AND NOT CBLAS_FIND_QUIETLY)
    message(STATUS "ROCHostNumerics CBLAS:")
    message(STATUS "  requested BLAS vendor: ${_CBLAS_requested_vendor}")
    message(STATUS "  static linkage requested: ${_CBLAS_requested_static}")
    message(STATUS "  header: ${CBLAS_INCLUDE_DIR}/cblas.h")
    message(STATUS "  link interface: ${CBLAS_BUILD_PROVIDER}")
    message(STATUS "  integer ABI: LP64 (32-bit)")
endif()

mark_as_advanced(CBLAS_INCLUDE_DIR CBLAS_LIBRARY)
