# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
Findroctx
----------

Find the roctx library

Imported targets
^^^^^^^^^^^^^^^^

This module defines the :prop_tgt:`IMPORTED` target if roctx is found:

``roctx::roctx``

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ROCTX_INCLUDE_DIR``
``ROCTX_LIBRARY``
``ROCTX_LIBRARIES``
``ROCTX_FOUND``

#]=======================================================================]

find_path(ROCTX_INCLUDE_DIR
    NAMES roctx.h
    PATH_SUFFIXES roctracer
    )
find_library(ROCTX_LIBRARY roctx64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(roctx ROCTX_INCLUDE_DIR ROCTX_LIBRARY)

if(ROCTX_FOUND)
  if(NOT DEFINED ROCTX_LIBRARIES)
    set(ROCTX_LIBRARIES ${ROCTX_LIBRARY})
  endif()

  if(NOT TARGET roctx::roctx)
    add_library(roctx::roctx UNKNOWN IMPORTED)

    set_target_properties(roctx::roctx PROPERTIES
      IMPORTED_LOCATION "${ROCTX_LIBRARY}"
    )
    set_target_properties(roctx::roctx PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ROCTX_INCLUDE_DIR}"
    )
  endif()
endif()
