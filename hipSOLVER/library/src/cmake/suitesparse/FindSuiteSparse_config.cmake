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
FindSuiteSparse_config
----------

Find the SuiteSparse SuiteSparse_config library

Imported targets
^^^^^^^^^^^^^^^^

This module defines the :prop_tgt:`IMPORTED` target if SuiteSparse_config is found:

``SuiteSparse::SuiteSparseConfig``

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SUITESPARSE_CONFIG_INCLUDE_DIR``
``SUITESPARSE_CONFIG_LIBRARY``
``SUITESPARSE_CONFIG_LIBRARIES``
``SUITESPARSE_CONFIG_FOUND``

#]=======================================================================]

find_path(SUITESPARSE_CONFIG_INCLUDE_DIR
    NAMES SuiteSparse_config.h
    PATH_SUFFIXES suitesparse
    )
find_library(SUITESPARSE_CONFIG_LIBRARY suitesparseconfig)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuiteSparse_config SUITESPARSE_CONFIG_INCLUDE_DIR SUITESPARSE_CONFIG_LIBRARY)

if(SUITESPARSE_CONFIG_FOUND)
  if(NOT DEFINED SUITESPARSE_CONFIG_LIBRARIES)
    set(SUITESPARSE_CONFIG_LIBRARIES ${SUITESPARSE_CONFIG_LIBRARY})
  endif()

  if(NOT TARGET SuiteSparse::SuiteSparseConfig)
    add_library(SuiteSparse::SuiteSparseConfig UNKNOWN IMPORTED)

    set_target_properties(SuiteSparse::SuiteSparseConfig PROPERTIES
      IMPORTED_LOCATION "${SUITESPARSE_CONFIG_LIBRARY}"
    )
    set_target_properties(SuiteSparse::SuiteSparseConfig PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${SUITESPARSE_CONFIG_INCLUDE_DIR}"
    )
  endif()
endif()
