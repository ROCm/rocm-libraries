# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

# rocThrust header-only library

# Configure a header file to pass the rocThrust version
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocthrust_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/rocthrust_version.hpp"
  @ONLY
)

# Only header target, does not include dependencies
add_library(rocthrust INTERFACE)
target_include_directories(rocthrust
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:rocthrust/include/>
)

target_link_libraries(rocthrust
  INTERFACE
    roc::rocprim_hip
)

# Installation

include(GNUInstallDirs)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# We need to install headers manually as rocm_install_targets
# does not support header-only libraries (INTERFACE targets)
rocm_install_targets(
  TARGETS rocthrust
  PREFIX rocthrust
)
if(CMAKE_VERSION VERSION_LESS 3.7)
  # Workaround: old versions of CMake do not support set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
  # The default archive type is "paxr" which generates corrupted DEB packages
  # if there are long file names.
  set(EXCLUDE_PATTERNS PATTERN "./system/cuda/detail/cub/*" EXCLUDE)
endif()
install(
  DIRECTORY
    "./"
    "${PROJECT_BINARY_DIR}/thrust/include/"
  DESTINATION rocthrust/include/thrust
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cuh"
  PATTERN "*.hpp"
  PATTERN "*.in"
  PATTERN "*.inl"
  ${EXCLUDE_PATTERNS}
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

# Export targets
rocm_export_targets(
  TARGETS rocthrust
  PREFIX rocthrust
)

# Create symlinks
rocm_install_symlink_subdir(rocthrust)
