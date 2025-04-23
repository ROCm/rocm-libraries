################################################################################
# 
# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
find_path(HALF_INCLUDE_DIRS
    NAMES half/half.hpp
    HINTS
    $ENV{HALF_DIR}/include
    $ENV{ROCM_PATH}/include
    PATHS
    ${HALF_DIR}/include
    /usr/include
    /usr/local/include
    ${ROCM_PATH}/include
)
mark_as_advanced(HALF_INCLUDE_DIRS)

if(HALF_INCLUDE_DIRS)
    set(HALF_FOUND TRUE)
endif( )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( HALF 
    FOUND_VAR  HALF_FOUND 
    REQUIRED_VARS
        HALF_INCLUDE_DIRS 
)

set(HALF_FOUND ${HALF_FOUND} CACHE INTERNAL "")
set(HALF_INCLUDE_DIRS ${HALF_INCLUDE_DIRS} CACHE INTERNAL "")

if(HALF_FOUND)
    message("-- ${White}Using HALF -- \n\tIncludes:${HALF_INCLUDE_DIRS}${ColourReset}")    
else()
    if(HALF_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindHALF -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindHALF failed to find -- half.hpp${ColourReset}" )
endif()
