# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################


# This is incremented when the ABI to the library changes
set(hipsolver_SOVERSION 1.0)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip ${HIP_PATH}/cmake)

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print out compiler flags for viewing/debug
if(BUILD_VERBOSE)
  message(STATUS "hipsolver_VERSION: " ${hipsolver_VERSION})
  message(STATUS "\t==>CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
  message(STATUS "\t==>BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS})
  message(STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX})
  message(STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH})
  message(STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH})

  message(STATUS "\t==>CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS})
  message(STATUS "\t==>CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG})
  message(STATUS "\t==>CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE})
  message(STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  message(STATUS "\t==>CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS})
endif()

if(NOT USE_CUDA)
  find_package(hip REQUIRED PATHS ${ROCM_PATH} /opt/rocm ${HIP_PATH})
else()
  find_package(HIP REQUIRED)
endif()

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/internal/hipsolver-version.h.in"
	"${PROJECT_BINARY_DIR}/include/hipsolver/internal/hipsolver-version.h")

set(hipsolver_headers_public
  include/hipsolver.h
  include/internal/hipsolver-types.h
  include/internal/hipsolver-functions.h
  include/internal/hipsolver-compat.h
  include/internal/hipsolver-dense.h
  include/internal/hipsolver-dense64.h
  include/internal/hipsolver-refactor.h
  include/internal/hipsolver-sparse.h
  ${PROJECT_BINARY_DIR}/include/hipsolver/internal/hipsolver-version.h
)

source_group("Header Files\\Public" FILES ${hipsolver_headers_public})

set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/hipsolver/internal)

# Build into subdirectories
add_subdirectory(src)

# The following code is setting variables to control the behavior of CPack to generate our
# if(WIN32)
#     set(CPACK_SOURCE_GENERATOR "ZIP")
#     set(CPACK_GENERATOR "ZIP")
# else()
#     set(CPACK_SOURCE_GENERATOR "TGZ")
#     set(CPACK_GENERATOR "DEB;RPM" CACHE STRING "cpack list: 7Z, DEB, IFW, NSIS, NSIS64, RPM, STGZ, TBZ2, TGZ, TXZ, TZ, ZIP")
# endif()
