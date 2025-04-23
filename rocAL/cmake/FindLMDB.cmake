################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2023 Advanced Micro Devices, Inc.
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
find_path(LMDB_INCLUDE_DIRS
    NAMES lmdb.h
    HINTS
    $ENV{LMDB_DIR}/include
    PATHS
    ${LMDB_DIR}/include
    /usr/include
    /usr/local/include
)
mark_as_advanced(LMDB_INCLUDE_DIRS)

find_library(LMDB_LIBRARIES
    NAMES lmdb
    HINTS
    $ENV{LMDB_DIR}/lib
    $ENV{LMDB_DIR}/lib64
    PATHS
    ${LMDB_DIR}/lib
    ${LMDB_DIR}/lib64
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
)
mark_as_advanced(LMDB_LIBRARIES)

find_path(LMDB_LIBRARIES_DIRS
    NAMES lmdb
    HINTS
    $ENV{LMDB_DIR}/lib
    $ENV{LMDB_DIR}/lib64
    PATHS
    ${LMDB_DIR}/lib
    ${LMDB_DIR}/lib64
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
)
mark_as_advanced(LMDB_LIBRARIES_DIRS)

if(LMDB_LIBRARIES AND LMDB_INCLUDE_DIRS)
    set(LMDB_FOUND TRUE)
endif( )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( LMDB 
    FOUND_VAR  LMDB_FOUND 
    REQUIRED_VARS
        LMDB_LIBRARIES 
        LMDB_INCLUDE_DIRS 
)

set(LMDB_FOUND ${LMDB_FOUND} CACHE INTERNAL "")
set(LMDB_LIBRARIES ${LMDB_LIBRARIES} CACHE INTERNAL "")
set(LMDB_INCLUDE_DIRS ${LMDB_INCLUDE_DIRS} CACHE INTERNAL "")
set(LMDB_LIBRARIES_DIRS ${LMDB_LIBRARIES_DIRS} CACHE INTERNAL "")

if(LMDB_FOUND)
    message("-- ${White}Using LMDB -- \n\tLibraries:${LMDB_LIBRARIES} \n\tIncludes:${LMDB_INCLUDE_DIRS}${ColourReset}")    
else()
    if(LMDB_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindLMDB -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindLMDB failed to find -- LMDB${ColourReset}" )
endif()
