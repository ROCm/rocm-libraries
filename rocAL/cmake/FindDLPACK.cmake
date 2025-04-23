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
find_path(DLPACK_INCLUDE_DIRS
    NAMES dlpack/dlpack.h
    HINTS
    $ENV{DLPACK_DIR}/include
    $ENV{ROCM_PATH}/include
    PATHS
    ${DLPACK_DIR}/include
    /usr/include
    /usr/local/include
    ${ROCM_PATH}/include
)
mark_as_advanced(DLPACK_INCLUDE_DIRS)

if(DLPACK_INCLUDE_DIRS)
    set(DLPACK_FOUND TRUE)
endif( )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( DLPACK 
    FOUND_VAR  DLPACK_FOUND 
    REQUIRED_VARS
        DLPACK_INCLUDE_DIRS 
)

set(DLPACK_FOUND ${DLPACK_FOUND} CACHE INTERNAL "")
set(DLPACK_INCLUDE_DIRS ${DLPACK_INCLUDE_DIRS} CACHE INTERNAL "")

if(DLPACK_FOUND)
    message("-- ${White}Using DLPACK -- \n\tIncludes:${DLPACK_INCLUDE_DIRS}${ColourReset}")    
else()
    if(DLPACK_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindDLPack -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindDLPack failed to find -- dlpack.h${ColourReset}" )
endif()
