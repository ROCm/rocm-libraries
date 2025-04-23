################################################################################
# 
# MIT License
# 
# Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc.
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
find_path(AMDRPP_INCLUDE_DIRS
    NAMES rpp.h
    HINTS
    $ENV{AMDRPP_PATH}/include/rpp
    PATHS
    ${AMDRPP_PATH}/include/rpp
    /usr/local/include/rpp
    ${ROCM_PATH}/include/rpp
)
mark_as_advanced(AMDRPP_INCLUDE_DIRS)

find_library(AMDRPP_LIBRARIES
    NAMES rpp
    HINTS
    $ENV{AMDRPP_PATH}/lib
    $ENV{AMDRPP_PATH}/lib64
    PATHS
    ${AMDRPP_PATH}/lib
    ${AMDRPP_PATH}/lib64
    /usr/local/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(AMDRPP_LIBRARIES)

find_path(AMDRPP_LIBRARIES_DIRS
    NAMES rpp
    HINTS
    $ENV{AMDRPP_PATH}/lib
    $ENV{AMDRPP_PATH}/lib64
    PATHS
    ${AMDRPP_PATH}/lib
    ${AMDRPP_PATH}/lib64
    /usr/local/lib
    ${ROCM_PATH}/lib
)
mark_as_advanced(AMDRPP_LIBRARIES_DIRS)

if(AMDRPP_LIBRARIES AND AMDRPP_INCLUDE_DIRS)
    set(AMDRPP_FOUND TRUE)
endif( )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    AMDRPP 
    FOUND_VAR
        AMDRPP_FOUND 
    REQUIRED_VARS
        AMDRPP_LIBRARIES 
        AMDRPP_INCLUDE_DIRS 
)

set(AMDRPP_FOUND ${AMDRPP_FOUND} CACHE INTERNAL "")
set(AMDRPP_LIBRARIES ${AMDRPP_LIBRARIES} CACHE INTERNAL "")
set(AMDRPP_INCLUDE_DIRS ${AMDRPP_INCLUDE_DIRS} CACHE INTERNAL "")
set(AMDRPP_LIBRARIES_DIRS ${AMDRPP_LIBRARIES_DIRS} CACHE INTERNAL "")

if(AMDRPP_FOUND)
    # Find RPP Version
    if (EXISTS "${AMDRPP_INCLUDE_DIRS}/rpp_version.h")
        file(READ "${AMDRPP_INCLUDE_DIRS}/rpp_version.h" RPP_VERSION_FILE)
        string(REGEX MATCH "RPP_VERSION_MAJOR ([0-9]*)" _ ${RPP_VERSION_FILE})
        set(RPP_VERSION_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
        string(REGEX MATCH "RPP_VERSION_MINOR ([0-9]*)" _ ${RPP_VERSION_FILE})
        set(RPP_VERSION_MINOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
        string(REGEX MATCH "RPP_VERSION_PATCH ([0-9]*)" _ ${RPP_VERSION_FILE})
        set(RPP_VERSION_PATCH ${CMAKE_MATCH_1} CACHE INTERNAL "")

        # Check for RPP 2.0.0
        if(RPP_VERSION_MAJOR EQUAL 2 AND RPP_VERSION_MINOR EQUAL 0 AND RPP_VERSION_PATCH EQUAL 0)
            message("-- ${White}Found RPP Version: ${RPP_VERSION_MAJOR}.${RPP_VERSION_MINOR}.${RPP_VERSION_PATCH}${ColourReset}")
            message("-- ${White}Using AMD RPP -- \n\tLibraries:${AMDRPP_LIBRARIES} \n\tIncludes:${AMDRPP_INCLUDE_DIRS}${ColourReset}")
        else()
            message("-- ${Yellow}NOTE: FindAMDRPP failed to find -- rpp2.0.0${ColourReset}")
            set(AMDRPP_FOUND 0)
        endif()
    else()
        message( "-- ${Yellow}NOTE: FindAMDRPP failed to find -- ${AMDRPP_INCLUDE_DIRS}/rpp_version.h${ColourReset}" )
        set(AMDRPP_FOUND 0)
    endif()
else()
    if(AMDRPP_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindAMDRPP -- NOT FOUND${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindAMDRPP failed to find -- rpp${ColourReset}" )
endif()
