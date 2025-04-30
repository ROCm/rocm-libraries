# ########################################################################
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

#[=======================================================================[.rst:
FindCHOLMOD
----------

Find the SuiteSparse CHOLMOD library

Imported targets
^^^^^^^^^^^^^^^^

This module defines the :prop_tgt:`IMPORTED` target if CHOLMOD is found:

``SuiteSparse::CHOLMOD``

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``CHOLMOD_INCLUDE_DIR``
``CHOLMOD_LIBRARY``
``CHOLMOD_LIBRARIES``
``CHOLMOD_FOUND``

#]=======================================================================]

find_path(CHOLMOD_INCLUDE_DIR
    NAMES cholmod.h
    PATH_SUFFIXES suitesparse
    )
find_library(CHOLMOD_LIBRARY cholmod)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD CHOLMOD_INCLUDE_DIR CHOLMOD_LIBRARY)

find_package(SuiteSparse_config QUIET)

if(CHOLMOD_FOUND)
  if(NOT DEFINED CHOLMOD_LIBRARIES)
    if(TARGET SuiteSparse::SuiteSparse_config)
      set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY} SuiteSparse::SuiteSparse_config)
    else()
      set(CHOLMOD_LIBRARIES ${CHOLMOD_LIBRARY})
    endif()
  endif()

  if(NOT TARGET SuiteSparse::CHOLMOD)
    add_library(SuiteSparse::CHOLMOD UNKNOWN IMPORTED)

    set_target_properties(SuiteSparse::CHOLMOD PROPERTIES
      IMPORTED_LOCATION "${CHOLMOD_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CHOLMOD_INCLUDE_DIR}"
    )
    if(TARGET SuiteSparse::SuiteSparse_config)
      set_target_properties(SuiteSparse::CHOLMOD PROPERTIES
        INTERFACE_LINK_LIBRARIES SuiteSparse::SuiteSparse_config
      )
    endif()
  endif()
endif()
