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
################################################################################################################################################################
# - Try to find rocDecode libraries and headers
# Once done this will define
#
# ROCDECODE_FOUND - system has rocDecode
# ROCDECODE_INCLUDE_DIR - the rocDecode include directory
# ROCDECODE_LIBRARY - Link these to use rocDecode
################################################################################

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# find rocDecode - library and headers
find_path(ROCDECODE_INCLUDE_DIR NAMES rocdecode.h PATHS ${ROCM_PATH}/include/rocdecode)
find_library(ROCDECODE_LIBRARY NAMES rocdecode HINTS ${ROCM_PATH}/lib)
mark_as_advanced(ROCDECODE_INCLUDE_DIR)
mark_as_advanced(ROCDECODE_LIBRARY)

if(ROCDECODE_INCLUDE_DIR AND ROCDECODE_LIBRARY)
    message("-- ${White}FindrocDecode -- Using rocDecode: \n\tIncludes:${ROCDECODE_INCLUDE_DIR}\n\tLib:${ROCDECODE_LIBRARY}${ColourReset}")
    set(ROCDECODE_FOUND TRUE)
else()
    if(rocDecode_FIND_REQUIRED)
        message(FATAL_ERROR "FindrocDecode -- Failed to find rocDecode Library")
    endif()
    message( "-- ${Yellow}NOTE: FindrocDecode failed to find rocDecode -- INSTALL rocDecode${ColourReset}" )
endif()

if(ROCDECODE_FOUND)
    # Find rocDecode Version
    file(READ "${ROCDECODE_INCLUDE_DIR}/rocdecode_version.h" ROCDECODE_VERSION_FILE)
    string(REGEX MATCH "ROCDECODE_MAJOR_VERSION ([0-9]*)" _ ${ROCDECODE_VERSION_FILE})
    set(ROCDECODE_VER_MAJOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "ROCDECODE_MINOR_VERSION ([0-9]*)" _ ${ROCDECODE_VERSION_FILE})
    set(ROCDECODE_VER_MINOR ${CMAKE_MATCH_1})
    string(REGEX MATCH "ROCDECODE_MICRO_VERSION ([0-9]*)" _ ${ROCDECODE_VERSION_FILE})
    set(ROCDECODE_VER_MICRO ${CMAKE_MATCH_1})
    message("-- ${White}Found rocDecode Version: ${ROCDECODE_VER_MAJOR}.${ROCDECODE_VER_MINOR}.${ROCDECODE_VER_MICRO}${ColourReset}")
    mark_as_advanced(ROCDECODE_VER_MAJOR)
    mark_as_advanced(ROCDECODE_VER_MINOR)
    mark_as_advanced(ROCDECODE_VER_MICRO)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocDecode
    FOUND_VAR  
        ROCDECODE_FOUND 
    REQUIRED_VARS
        ROCDECODE_INCLUDE_DIR
        ROCDECODE_LIBRARY
)

set(ROCDECODE_FOUND ${ROCDECODE_FOUND} CACHE INTERNAL "")
set(ROCDECODE_INCLUDE_DIR ${ROCDECODE_INCLUDE_DIR} CACHE INTERNAL "")
set(ROCDECODE_LIBRARY ${ROCDECODE_LIBRARY} CACHE INTERNAL "")
set(ROCDECODE_VER_MAJOR ${ROCDECODE_VER_MAJOR} CACHE INTERNAL "")
set(ROCDECODE_VER_MINOR ${ROCDECODE_VER_MINOR} CACHE INTERNAL "")
set(ROCDECODE_VER_MICRO ${ROCDECODE_VER_MICRO} CACHE INTERNAL "")
