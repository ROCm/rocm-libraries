# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc.
# ########################################################################

# This file is intended to be used in two ways; independently in a stand alone PROJECT
# and as part of a superbuild.  If the file is included in a stand alone project, the
# variables are not expected to be preset, and this will produce options() in the GUI
# for the user to examine.  If this file is included in a superbuild, the options will be
# presented in the superbuild GUI, but then passed into the ExternalProject as -D
# parameters, which would already define them.

if( NOT BUILD_CLIENTS_TESTS )
  option( BUILD_CLIENTS_TESTS "Build hipBLAS unit tests" OFF )
endif( )

if( NOT BUILD_CLIENTS_BENCHMARKS )
  option( BUILD_CLIENTS_BENCHMARKS "Build hipBLAS benchmarks" OFF )
endif( )

if( NOT BUILD_CLIENTS_SAMPLES )
  option( BUILD_CLIENTS_SAMPLES "Build hipBLAS samples" OFF )
endif( )

if( HIP_PLATFORM STREQUAL nvidia )
  option( LINK_BLIS "Link AOCL Blis reference library" OFF )
else()
  option( LINK_BLIS "Link AOCL Blis reference library" ON )
endif()


