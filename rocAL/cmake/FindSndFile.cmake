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
find_path(SNDFILE_INCLUDE_DIRS
    NAMES sndfile.h
    HINTS
    $ENV{SNDFILE_PATH}/include
    PATHS
    /usr/local/include
    /usr/include
)
mark_as_advanced(SNDFILE_INCLUDE_DIRS)

find_library(SNDFILE_LIBRARIES
    NAMES sndfile libsndfile
    HINTS
    $ENV{SNDFILE_PATH}/lib
    $ENV{SNDFILE_PATH}/lib64
    PATHS
    ${CMAKE_SYSTEM_PREFIX_PATH}
    ${SNDFILE_PATH}
    /usr/local/
    PATH_SUFFIXES lib lib64
)
mark_as_advanced(SNDFILE_LIBRARIES)

if(SNDFILE_LIBRARIES AND SNDFILE_INCLUDE_DIRS)
    set(SNDFILE_FOUND TRUE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SndFile 
    FOUND_VAR  SNDFILE_FOUND 
    REQUIRED_VARS
        SNDFILE_LIBRARIES
        SNDFILE_INCLUDE_DIRS
)

set(SNDFILE_FOUND ${SNDFILE_FOUND} CACHE INTERNAL "")
set(SNDFILE_LIBRARIES ${SNDFILE_LIBRARIES} CACHE INTERNAL "")
set(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIRS} CACHE INTERNAL "")

if(SNDFILE_FOUND)
    message("-- ${White}Using SndFile -- \n\tLibraries:${SNDFILE_LIBRARIES} \n\tIncludes:${SNDFILE_INCLUDE_DIRS}${ColourReset}")   
else()
    message( "-- ${Yellow}NOTE: FindSndFile failed to find -- SndFile${ColourReset}" )
endif()
