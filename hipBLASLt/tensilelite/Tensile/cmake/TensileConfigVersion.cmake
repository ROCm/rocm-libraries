################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

# hardcoded tensile version; also in Tensile/__init__.py
set(TENSILE_VERSION_MAJOR  4)
set(TENSILE_VERSION_MINOR 33)
set(TENSILE_VERSION_PATCH  0)

# export version
set(PACKAGE_VERSION "${TENSILE_VERSION_MAJOR}.${TENSILE_VERSION_MINOR}.${TENSILE_VERSION_PATCH}")

if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
else()
    set(PACKAGE_VERSION_EXACT FALSE)
endif()

if(PACKAGE_VERSION_EXACT} OR (PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION))
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
else()
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()

if(PACKAGE_VERSION_COMPATIBLE)
    set(PACKAGE_VERSION_UNSUTABLE FALSE)
else()
    set(PACKAGE_VERSION_UNSUTABLE TRUE)
endif()

if(false)
# set to compatible, and switch to false below if necessary
set(PACKAGE_VERSION_EXACT      TRUE)
set(PACKAGE_VERSION_COMPATIBLE TRUE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)

# if major doesn't match
if (NOT PACKAGE_FIND_VERSION_MAJOR STREQUAL TENSILE_VERSION_MAJOR)

  set(PACKAGE_VERSION_EXACT      FALSE)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
  set(PACKAGE_VERSION_UNSUITABLE TRUE)
  return()
endif()

# if minor insufficient
if (PACKAGE_FIND_VERSION_MINOR STRGREATER TENSILE_VERSION_MINOR)

  set(PACKAGE_VERSION_EXACT      FALSE)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
  set(PACKAGE_VERSION_UNSUITABLE TRUE)
  return()
endif()

# if minor==minor but patch insufficient
if (PACKAGE_FIND_VERSION_MINOR STREQUAL TENSILE_VERSION_MINOR)
  if (PACKAGE_FIND_VERSION_PATCH STRGREATER TENSILE_VERSION_PATCH)

    set(PACKAGE_VERSION_EXACT      FALSE)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
    return()
  endif()
endif()

# check exactness
if (NOT (PACKAGE_FIND_VERSION_MINOR STREQUAL TENSILE_VERSION_MINOR
    AND PACKAGE_FIND_VERSION_MINOR STREQUAL TENSILE_VERSION_MINOR) )

  set(PACKAGE_VERSION_EXACT FALSE)
endif()
endif()

