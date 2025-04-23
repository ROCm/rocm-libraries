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
if(APPLE)
    set(SHARED_LIB_TYPE ".dylib")
else()
    set(SHARED_LIB_TYPE ".so")
endif()

find_path(MIVisionX_INCLUDE_DIRS
    NAMES vx_ext_amd.h
    HINTS
    $ENV{MIVisionX_PATH}/include/mivisionx
    PATHS
    ${MIVisionX_PATH}/include/mivisionx
    /usr/include
    ${ROCM_PATH}/include/mivisionx
)
mark_as_advanced(MIVisionX_INCLUDE_DIRS)

# OpenVX
find_library(OPENVX_LIBRARIES
    NAMES libopenvx${SHARED_LIB_TYPE}
    HINTS
    $ENV{MIVisionX_PATH}/lib
    PATHS
    ${MIVisionX_PATH}/lib
    /usr/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(OPENVX_LIBRARIES)

# VX_RPP
find_library(VXRPP_LIBRARIES
    NAMES libvx_rpp${SHARED_LIB_TYPE}
    HINTS
    $ENV{MIVisionX_PATH}/lib
    PATHS
    ${MIVisionX_PATH}/lib
    /usr/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(VXRPP_LIBRARIES)

if(OPENVX_LIBRARIES AND MIVisionX_INCLUDE_DIRS)
    set(MIVisionX_FOUND TRUE)
endif( )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( MIVisionX 
    FOUND_VAR  MIVisionX_FOUND 
    REQUIRED_VARS
        OPENVX_LIBRARIES
        VXRPP_LIBRARIES
        MIVisionX_INCLUDE_DIRS
)

set(MIVisionX_FOUND ${MIVisionX_FOUND} CACHE INTERNAL "")
set(OPENVX_LIBRARIES ${OPENVX_LIBRARIES} CACHE INTERNAL "")
set(VXRPP_LIBRARIES ${VXRPP_LIBRARIES} CACHE INTERNAL "")
set(MIVisionX_INCLUDE_DIRS ${MIVisionX_INCLUDE_DIRS} CACHE INTERNAL "")

if(MIVisionX_FOUND)
    if(VXRPP_LIBRARIES)
        if(EXISTS "${MIVisionX_INCLUDE_DIRS}/vx_ext_rpp_version.h")
            # Find RPP Version
            file(READ "${MIVisionX_INCLUDE_DIRS}/vx_ext_rpp_version.h" VX_EXT_RPP_VERSION_FILE)
            string(REGEX MATCH "VX_EXT_RPP_VERSION_MAJOR ([0-9]*)" _ ${VX_EXT_RPP_VERSION_FILE})
            set(VX_EXT_RPP_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
            string(REGEX MATCH "VX_EXT_RPP_VERSION_MINOR ([0-9]*)" _ ${VX_EXT_RPP_VERSION_FILE})
            set(VX_EXT_RPP_VERSION_MINOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
            string(REGEX MATCH "VX_EXT_RPP_VERSION_PATCH ([0-9]*)" _ ${VX_EXT_RPP_VERSION_FILE})
            set(VX_EXT_RPP_VERSION_PATCH ${CMAKE_MATCH_1} CACHE INTERNAL "")
            message("-- ${White}Found VX RPP Version: ${VX_EXT_RPP_VERSION_MAJOR}.${VX_EXT_RPP_VERSION_MINOR}.${VX_EXT_RPP_VERSION_PATCH}${ColourReset}")
            message("-- ${White}Using MIVisionX -- \n\tLibraries:${OPENVX_LIBRARIES} \n\tIncludes:${MIVisionX_INCLUDE_DIRS}${ColourReset}")
        else()
            set(VX_EXT_RPP_VERSION_MAJOR 0)
            set(VX_EXT_RPP_VERSION_MINOR 0)
            set(VX_EXT_RPP_VERSION_PATCH 0)
        endif()
    else()
        message("-- ${Yellow}VX RPP - Not Found${ColourReset}")
    endif()
else()
    if(MIVisionX_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindMIVisionX -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindMIVisionX failed to find -- openvx${ColourReset}" )
endif()