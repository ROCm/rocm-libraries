################################################################################
# 
# MIT License
# 
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
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
find_path(LIBTAR_INCLUDE_DIRS
    NAMES libtar.h
    HINTS
    $ENV{LIBTAR_PATH}/include
    PATHS
    /usr/include
    /usr/local/include
)
mark_as_advanced(LIBTAR_INCLUDE_DIRS)

find_library(LIBTAR_LIBRARIES
        NAMES libtar.a tar libtar
        HINTS
        $ENV{LIBTAR_PATH}/lib
        $ENV{LIBTAR_PATH}/lib64
        PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBTAR_PATH} "/usr/local" "/usr/lib"
        PATH_SUFFIXES lib lib64)

mark_as_advanced(LIBTAR_LIBRARIES)

if(LIBTAR_LIBRARIES AND LIBTAR_INCLUDE_DIRS)
    message("-- ${White}Using Libtar -- \n\tLibraries:${LIBTAR_LIBRARIES} \n\tIncludes:${LIBTAR_INCLUDE_DIRS}${ColourReset}")
    set(LIBTAR_FOUND TRUE)
else()
    message( "-- ${Yellow}NOTE: FindLibTar failed to find -- LibTar${ColourReset}" )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibTar
    FOUND_VAR  LIBTAR_FOUND 
    REQUIRED_VARS
        LIBTAR_LIBRARIES
        LIBTAR_INCLUDE_DIRS
)

set(LIBTAR_FOUND ${LIBTAR_FOUND} CACHE INTERNAL "")
set(LIBTAR_LIBRARIES ${LIBTAR_LIBRARIES} CACHE INTERNAL "")
set(LIBTAR_INCLUDE_DIRS ${LIBTAR_INCLUDE_DIRS} CACHE INTERNAL "")
