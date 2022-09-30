# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
# ########################################################################

# rocThrust header-only library

# Configure a header file to pass the rocThrust version
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocthrust_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/rocthrust_version.hpp"
  @ONLY
)

#Creat wrapper for generated version file
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_file(
      "rocthrust_version.hpp"
      HEADER_LOCATION include/thrust
      WRAPPER_LOCATIONS rocthrust/${CMAKE_INSTALL_INCLUDEDIR}/thrust
      OUTPUT_LOCATIONS rocthrust/wrapper/include/thrust
  )
endif()

# Only header target, does not include dependencies
add_library(rocthrust INTERFACE)
target_include_directories(rocthrust
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

target_link_libraries(rocthrust
  INTERFACE
    roc::rocprim_hip
)

# Installation

# We need to install headers manually as rocm_install_targets
# does not support header-only libraries (INTERFACE targets)
rocm_install_targets(
  TARGETS rocthrust
)
if(CMAKE_VERSION VERSION_LESS 3.7)
  # Workaround: old versions of CMake do not support set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")
  # The default archive type is "paxr" which generates corrupted DEB packages
  # if there are long file names.
  set(EXCLUDE_PATTERNS PATTERN "./system/cuda/detail/cub/*" EXCLUDE)
endif()
rocm_install(
  DIRECTORY
    "./"
    "${PROJECT_BINARY_DIR}/thrust/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/thrust
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cuh"
  PATTERN "*.hpp"
  PATTERN "*.in"
  PATTERN "*.inl"
  ${EXCLUDE_PATTERNS}
  PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

#Install the wrapper to rocthrust folder. 
#So wrapper would be in /opt/rocm-xxx/rocthrust/include/thrust
if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_install(
    DIRECTORY
    "${PROJECT_BINARY_DIR}/rocthrust/wrapper/"
    DESTINATION rocthrust/ 
  )
endif()

include(ROCMExportTargetsHeaderOnly)
# Export targets
rocm_export_targets_header_only(
  TARGETS roc::rocthrust
  DEPENDS PACKAGE rocprim
  NAMESPACE roc::
)

