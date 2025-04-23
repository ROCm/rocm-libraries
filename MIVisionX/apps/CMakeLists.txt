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
project(mivisionx_applications)

set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm Installation Path")
#Added Default libdir
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

if(EXISTS ${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}/libopenvx.so)
    message("-- MIVisionX Applications -- OpenVX Found, adding CV Apps")
    add_subdirectory(bubble_pop)
    add_subdirectory(optical_flow)
else()
    message(FATAL_ERROR "MIVisionX Not Installed - ${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}/libopenvx.so missing")
endif()

if(EXISTS ${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}/libvx_nn.so)
    message("-- MIVisionX Applications -- OpenVX NN Extension Found, adding ML Apps")
    add_subdirectory(cloud_inference/server_app)
    add_subdirectory(dg_test)
    add_subdirectory(mivisionx_openvx_classifier)
else()
    message("-- MIVisionX Applications -- OpenVX NN Extension Not Found at ${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}")
endif()
