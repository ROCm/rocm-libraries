################################################################################
#
# MIT License
#
# Copyright (c) 2017 - 2025 Advanced Micro Devices, Inc.
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
if(APPLE)
    set(SHARED_LIB_TYPE ".dylib")
else()
    set(SHARED_LIB_TYPE ".so")
endif()

find_path(TurboJpeg_INCLUDE_DIRS
    NAMES turbojpeg.h
    HINTS
    $ENV{TURBO_JPEG_PATH}/include
    PATHS
    ${TURBO_JPEG_PATH}/include
    /usr/include
    /opt/libjpeg-turbo/include
)
mark_as_advanced(TurboJpeg_INCLUDE_DIRS)

find_library(TurboJpeg_LIBRARIES
    NAMES libturbojpeg${SHARED_LIB_TYPE}
    HINTS
    $ENV{TURBO_JPEG_PATH}/lib
    $ENV{TURBO_JPEG_PATH}/lib64
    PATHS
    ${TURBO_JPEG_PATH}/lib
    ${TURBO_JPEG_PATH}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /opt/libjpeg-turbo/lib
)
mark_as_advanced(TurboJpeg_LIBRARIES)

find_path(TurboJpeg_LIBRARIES_DIRS
    NAMES libturbojpeg${SHARED_LIB_TYPE}
    HINTS
    $ENV{TURBO_JPEG_PATH}/lib
    $ENV{TURBO_JPEG_PATH}/lib64
    PATHS
    ${TURBO_JPEG_PATH}/lib
    ${TURBO_JPEG_PATH}/lib64
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /opt/libjpeg-turbo/lib
)
mark_as_advanced(TurboJpeg_LIBRARIES_DIRS)

if(TurboJpeg_LIBRARIES AND TurboJpeg_INCLUDE_DIRS)
    set(TurboJpeg_FOUND TRUE)
endif( )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TurboJpeg
    FOUND_VAR
        TurboJpeg_FOUND
    REQUIRED_VARS
        TurboJpeg_LIBRARIES
        TurboJpeg_INCLUDE_DIRS
        TurboJpeg_LIBRARIES_DIRS
)

set(TurboJpeg_FOUND ${TurboJpeg_FOUND} CACHE INTERNAL "")
set(TurboJpeg_LIBRARIES ${TurboJpeg_LIBRARIES} CACHE INTERNAL "")
set(TurboJpeg_INCLUDE_DIRS ${TurboJpeg_INCLUDE_DIRS} CACHE INTERNAL "")
set(TurboJpeg_LIBRARIES_DIRS ${TurboJpeg_LIBRARIES_DIRS} CACHE INTERNAL "")

if(TurboJpeg_FOUND)
    message("-- ${White}Using Turbo JPEG -- \n\tLibraries:${TurboJpeg_LIBRARIES} \n\tIncludes:${TurboJpeg_INCLUDE_DIRS}${ColourReset}")
else()
    if(TurboJpeg_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindTurboJpeg -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindTurboJpeg failed to find -- turbojpeg${ColourReset}" )
endif()
