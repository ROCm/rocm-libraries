# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.10)
project(utilities)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
#set(ROCM_PATH /opt/rocm CACHE PATH " default ROCm installation path")

if(GPU_SUPPORT)
    find_package(FFmpeg QUIET)
    if("${BACKEND}" STREQUAL "OPENCL")
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            set(BUILD_OPENCL true)
        else()
            set(BUILD_OPENCL false)
            set(GPU_SUPPORT false)
        endif()
    elseif ("${BACKEND}" STREQUAL "HIP")
        if(NOT DEFINED HIP_PATH)
            if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
            else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            endif()
        endif()
        list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
        find_package(HIP QUIET)
        SET(BUILD_OPENCL false)
    endif()
endif()

add_subdirectory(runvx)

if(OpenCL_FOUND AND BUILD_OPENCL)
    message("-- ${Green}Utilities -- runvx module added with OPENCL support${ColourReset}")
    add_subdirectory(runcl)
    if(LOOM)
        add_subdirectory(loom_shell)
        message("-- ${Green}Loom Shell -- loom shell script interpreter module added${ColourReset}")
    else()
        message("-- ${Cyan}Loom Shell Module turned OFF by user option -D LOOM=OFF ${ColourReset}")
    endif()
    if(FFMPEG_FOUND)
        add_subdirectory(mv_deploy)
        message("-- ${Green}Utilities -- runvx, runcl, loom_shell, & mv_deploy modules added${ColourReset}")
    else()
        message("-- ${Green}Utilities -- runcl modules added${ColourReset}")
        message("-- ${Red}WARNING: FFMPEG Not Found -- mv_deploy module excluded${ColourReset}")
    endif(FFMPEG_FOUND)
elseif (HIP_FOUND)
    if(FFMPEG_FOUND)
        add_subdirectory(mv_deploy)
        message("-- ${Green}Utilities -- runvx and mv_deploy modules added${ColourReset}")
    else()
        message("-- ${Green}Utilities -- runvx module added with HIP support${ColourReset}")
        message("-- ${Red}WARNING: FFMPEG Not Found -- mv_deploy module excluded${ColourReset}")
    endif(FFMPEG_FOUND)

else()
    message("-- ${Green}Utilities -- runvx module added with CPU support${ColourReset}")
    message("-- ${Red}WARNING: Utilities -- runcl, loom_shell, & mv_deploy modules excluded${ColourReset}")
endif()
