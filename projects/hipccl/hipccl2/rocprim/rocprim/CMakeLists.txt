# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

# rocPRIM header-only library

# Configure a header file to pass the rocPRIM version
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/rocprim/rocprim_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/rocprim/rocprim_version.hpp"
  @ONLY
)

# Only header target, does not include dependencies
add_library(rocprim INTERFACE)
target_include_directories(rocprim
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/rocprim>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:rocprim/include/>
)

# This target links against HIP library
add_library(rocprim_hip INTERFACE)
target_link_libraries(rocprim_hip INTERFACE rocprim hip::device)


# Installation

include(GNUInstallDirs)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# We need to install headers manually as rocm_install_targets
# does not support header-only libraries (INTERFACE targets)
rocm_install_targets(
  TARGETS rocprim rocprim_hip
  PREFIX rocprim
)

install(
  DIRECTORY
    "include/"
    "${PROJECT_BINARY_DIR}/rocprim/include/"
  DESTINATION rocprim/include/
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

# Export targets
rocm_export_targets(
  TARGETS roc::rocprim roc::rocprim_hip
  PREFIX rocprim
  NAMESPACE roc::
)

# Create symlinks
rocm_install_symlink_subdir(rocprim)
