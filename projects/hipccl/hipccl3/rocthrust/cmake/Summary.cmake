# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
# ########################################################################

function(print_configuration_summary)
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  HIP ROOT              : ${HIP_ROOT_DIR}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
  string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS_STRIP}")
  message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
  message(STATUS "  Install prefix        : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "")
  message(STATUS "  BUILD_TEST            : ${BUILD_TEST}")
endfunction()
