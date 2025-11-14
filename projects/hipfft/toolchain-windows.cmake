# #############################################################################
# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# #############################################################################

if (DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
elseif (DEFINED ENV{HIP_DIR})
  file(TO_CMAKE_PATH "$ENV{HIP_DIR}" HIP_DIR)
  set(rocm_bin "${HIP_DIR}/bin")
else()
  set(HIP_DIR "C:/hip")
  set(rocm_bin "C:/hip/bin")
endif()

if( NOT DEFINED CMAKE_CXX_COMPILER )
  set(CMAKE_CXX_COMPILER "${rocm_bin}/clang++.exe")
endif()
if( NOT DEFINED CMAKE_C_COMPILER )
  set(CMAKE_C_COMPILER "${rocm_bin}/clang.exe")
endif()

# our usage flags
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -DWIN32 -D_CRT_SECURE_NO_WARNINGS")

# flags for clang direct use

# -Wno-ignored-attributes to avoid warning: __declspec attribute 'dllexport' is not supported [-Wignored-attributes] which is used by msvc compiler
# -Wno-unknown-attributes to avoid warning: unknown attribute '__dllimport__' ignored [-Wunknown-attributes], in boost
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -Wno-ignored-attributes -Wno-unknown-attributes")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -DHIP_CLANG_HCC_COMPAT_MODE=1 -DBOOST_USE_WINDOWS_H -DNOMINMAX")

# args also in hipcc.bat
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -fms-extensions -fms-compatibility -D__HIP_ROCclr__=1")

if (DEFINED ENV{VCPKG_PATH})
  file(TO_CMAKE_PATH "$ENV{VCPKG_PATH}" VCPKG_PATH)
else()
  set(VCPKG_PATH "C:/github/vcpkg")
endif()
include("${VCPKG_PATH}/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_STATIC_LIBRARY_PREFIX "static_")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_SHARED_LIBRARY_PREFIX "")
