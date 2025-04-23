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

# ROCM Path
if(ROCM_PATH)
    message("-- FindMIVISIONX: ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# find OpenVX - Core MIVisionX library
find_path(OPENVX_INCLUDE_DIR NAMES VX/vx.h PATHS ${ROCM_PATH}/include/mivisionx)
find_library(OPENVX_LIBRARY NAMES openvx HINTS ${ROCM_PATH}/lib)
find_library(OPENVXU_LIBRARY NAMES vxu HINTS ${ROCM_PATH}/lib)
mark_as_advanced(OPENVX_INCLUDE_DIR)
mark_as_advanced(OPENVX_LIBRARY)
mark_as_advanced(OPENVXU_LIBRARY)

if(OPENVX_INCLUDE_DIR AND OPENVX_LIBRARY AND OPENVXU_LIBRARY)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- Using OpenVX: \n\tIncludes:${OPENVX_INCLUDE_DIR}\n\tLib:${OPENVX_LIBRARY}${ColourReset}")
    endif()
    set(MIVISIONX_FOUND TRUE)
    set(MIVISIONX_INCLUDE_DIR ${OPENVX_INCLUDE_DIR})
    set(MIVISIONX_LIBRARIES ${OPENVX_LIBRARY} ${OPENVXU_LIBRARY})
else()
    if(MIVISIONX_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindMIVISIONX -- Failed to find OPENVX${ColourReset}")
    endif()
    message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find OpenVX -- INSTALL MIVisionX${ColourReset}" )
endif()

# find vx_amd_media
find_path(VX_AMD_MEDIA_INCLUDE_DIR NAMES vx_amd_media.h PATHS ${ROCM_PATH}/include/mivisionx)
find_library(VX_AMD_MEDIA_LIBRARY NAMES vx_amd_media HINTS ${ROCM_PATH}/lib)
mark_as_advanced(VX_AMD_MEDIA_LIBRARY)
if(VX_AMD_MEDIA_INCLUDE_DIR AND VX_AMD_MEDIA_LIBRARY)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- Using VX_AMD_MEDIA: \n\tLib:${VX_AMD_MEDIA_LIBRARY}${ColourReset}")
    endif()
    set(MIVISIONX_LIBRARIES ${MIVISIONX_LIBRARIES} ${VX_AMD_MEDIA_LIBRARY})
else()
    if(MIVISIONX_FIND_REQUIRED)
        message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find VX_AMD_MEDIA${ColourReset}" )
    endif()
endif()

# find vx_nn
find_path(VX_NN_INCLUDE_DIR NAMES vx_amd_nn.h PATHS ${ROCM_PATH}/include/mivisionx)
find_library(VX_NN_LIBRARY NAMES vx_nn HINTS ${ROCM_PATH}/lib)
mark_as_advanced(VX_NN_LIBRARY)
if(VX_NN_INCLUDE_DIR AND VX_NN_LIBRARY)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- Using VX_NN: \n\tLib:${VX_NN_LIBRARY}${ColourReset}")
    endif()
    set(MIVISIONX_LIBRARIES ${MIVISIONX_LIBRARIES} ${VX_NN_LIBRARY})
else()
    if(MIVISIONX_FIND_REQUIRED)
        message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find VX_NN${ColourReset}" )
    endif()
endif()

# find vx_rpp
find_path(VX_RPP_INCLUDE_DIR NAMES vx_ext_rpp.h PATHS ${ROCM_PATH}/include/mivisionx)
find_library(VX_RPP_LIBRARY NAMES vx_rpp HINTS ${ROCM_PATH}/lib)
mark_as_advanced(VX_RPP_LIBRARY)
if(VX_RPP_INCLUDE_DIR AND VX_RPP_LIBRARY)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- Using VX_RPP: \n\tLib:${VX_RPP_LIBRARY}${ColourReset}")
    endif()
    set(MIVISIONX_LIBRARIES ${MIVISIONX_LIBRARIES} ${VX_RPP_LIBRARY})
else()
    if(MIVISIONX_FIND_REQUIRED)
        message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find VX_RPP${ColourReset}" )
    endif()
endif()

# find vx_opencv
find_path(VX_OPENCV_INCLUDE_DIR NAMES vx_ext_opencv.h PATHS ${ROCM_PATH}/include/mivisionx)
find_library(VX_OPENCV_LIBRARY NAMES vx_opencv HINTS ${ROCM_PATH}/lib)
mark_as_advanced(VX_OPENCV_LIBRARY)
if(VX_OPENCV_LIBRARY AND VX_OPENCV_INCLUDE_DIR)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- Using VX_OPENCV: \n\tLib:${VX_OPENCV_LIBRARY}${ColourReset}")
    endif()
    set(MIVISIONX_LIBRARIES ${MIVISIONX_LIBRARIES} ${VX_OPENCV_LIBRARY})
else()
    if(MIVISIONX_FIND_REQUIRED)
        message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find VX_OPENCV${ColourReset}" )
    endif()
endif()

# find vx_amd_migraphx
find_path(VX_AMD_MIGRAPHX_INCLUDE_DIR NAMES vx_amd_migraphx.h PATHS ${ROCM_PATH}/include/mivisionx)
find_library(VX_AMD_MIGRAPHX_LIBRARY NAMES vx_amd_migraphx HINTS ${ROCM_PATH}/lib)
mark_as_advanced(VX_AMD_MIGRAPHX_LIBRARY)
if(VX_AMD_MIGRAPHX_LIBRARY AND VX_AMD_MIGRAPHX_INCLUDE_DIR)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- Using VX_AMD_MIGRAPHX: \n\tLib:${VX_AMD_MIGRAPHX_LIBRARY}${ColourReset}")
    endif()
    set(MIVISIONX_LIBRARIES ${MIVISIONX_LIBRARIES} ${VX_AMD_MIGRAPHX_LIBRARY})
else()
    if(MIVISIONX_FIND_REQUIRED)
        message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find VX_AMD_MIGRAPHX${ColourReset}" )
    endif()
endif()


# find runvx exe
find_program(RUNVX_EXECUTABLE NAMES runvx PATHS ${ROCM_PATH}/bin)
mark_as_advanced(RUNVX_EXECUTABLE)
if(RUNVX_EXECUTABLE)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- Using RunVX: \n\tEXE:${RUNVX_EXECUTABLE}${ColourReset}")
    endif()
else()
    if(MIVISIONX_FIND_REQUIRED)
        message( "-- ${Yellow}NOTE: FindMIVISIONX failed to find RUNVX_EXECUTABLE${ColourReset}" )
    endif()
endif()

# find backend
if(MIVISIONX_FOUND)
    if (EXISTS "${MIVISIONX_INCLUDE_DIR}/openvx_backend.h")
        file(READ "${MIVISIONX_INCLUDE_DIR}/openvx_backend.h" MIVISIONX_BACKEND_FILE)
        string(REGEX MATCH "ENABLE_OPENCL ([0-9]*)" _ ${MIVISIONX_BACKEND_FILE})
        set(MIVISIONX_OPENCL_BACKEND ${CMAKE_MATCH_1} CACHE INTERNAL "")
        string(REGEX MATCH "ENABLE_HIP ([0-9]*)" _ ${MIVISIONX_BACKEND_FILE})
        set(MIVISIONX_HIP_BACKEND ${CMAKE_MATCH_1} CACHE INTERNAL "")
    endif()
    # set mivisionx backend
    if(MIVISIONX_HIP_BACKEND)
        set(MIVISIONX_BACKEND "HIP")
    elseif(MIVISIONX_OPENCL_BACKEND)
        set(MIVISIONX_BACKEND "OPENCL")
    else()
        set(MIVISIONX_BACKEND "CPU")
    endif()

    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}FindMIVISIONX -- MIVISIONX_BACKEND: ${MIVISIONX_BACKEND}${ColourReset}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    MIVISIONX
    FOUND_VAR  
        MIVISIONX_FOUND 
    REQUIRED_VARS
        OPENVX_INCLUDE_DIR
        OPENVX_LIBRARY
        RUNVX_EXECUTABLE
        MIVISIONX_INCLUDE_DIR
        MIVISIONX_LIBRARIES
        MIVISIONX_BACKEND
)

set(MIVISIONX_FOUND ${MIVISIONX_FOUND} CACHE INTERNAL "")
set(OPENVX_INCLUDE_DIR ${OPENVX_INCLUDE_DIR} CACHE INTERNAL "")
set(OPENVX_LIBRARIES ${OPENVX_LIBRARIES} CACHE INTERNAL "")
set(VX_AMD_MEDIA_LIBRARY ${VX_AMD_MEDIA_LIBRARY} CACHE INTERNAL "")
set(VX_AMD_MIGRAPHX_LIBRARY ${VX_AMD_MIGRAPHX_LIBRARY} CACHE INTERNAL "")
set(VX_NN_LIBRARY ${VX_NN_LIBRARY} CACHE INTERNAL "")
set(VX_OPENCV_LIBRARY ${VX_OPENCV_LIBRARY} CACHE INTERNAL "")
set(VX_RPP_LIBRARY ${VX_RPP_LIBRARY} CACHE INTERNAL "")
set(RUNVX_EXECUTABLE ${RUNVX_EXECUTABLE} CACHE INTERNAL "")
set(MIVISIONX_INCLUDE_DIR ${MIVISIONX_INCLUDE_DIR} CACHE INTERNAL "")
set(MIVISIONX_LIBRARIES ${MIVISIONX_LIBRARIES} CACHE INTERNAL "")
set(MIVISIONX_BACKEND ${MIVISIONX_BACKEND} CACHE INTERNAL "")

if(MIVISIONX_FOUND)
    if(MIVISIONX_FIND_REQUIRED)
        message("-- ${White}Using MIVISIONX: \n\tMIVisionX Libraries:${MIVISIONX_LIBRARIES} \n\tMIVisionX Includes:${MIVISIONX_INCLUDE_DIR}${ColourReset}")
    endif()
else()
    if(MIVISIONX_FIND_REQUIRED)
        message(FATAL_ERROR "{Red}FindMIVISIONX -- NOT FOUND${ColourReset}")
    endif()
endif()
