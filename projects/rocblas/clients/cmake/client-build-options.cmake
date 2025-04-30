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

# This file is intended to be used in two ways; independently in a stand alone PROJECT
# and as part of a superbuild.  If the file is included in a stand alone project, the
# variables are not expected to be preset, and this will produce options() in the GUI
# for the user to examine.  If this file is included in a superbuild, the options will be
# presented in the superbuild GUI, but then passed into the ExternalProject as -D
# parameters, which would already define them.

# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
if( NOT BUILD_WITH_TENSILE )
  option( BUILD_WITH_TENSILE "Build rocBLAS with Tensile or not" ON )
endif( )

# Clients utilize rocblas fortran API and a fortran compiler
if( NOT BUILD_FORTRAN_CLIENTS )
  option( BUILD_FORTRAN_CLIENTS "Build rocBLAS clients requiring Fortran capabilities" ON )
endif( )

# Samples have no other dependencies except for rocblas, so are enabled by default
if( NOT BUILD_CLIENTS_SAMPLES )
  option( BUILD_CLIENTS_SAMPLES "Build rocBLAS samples" OFF )
endif( )

if( NOT BUILD_CLIENTS_TESTS )
  option( BUILD_CLIENTS_TESTS "Build rocBLAS unit tests" OFF )
endif( )

if( NOT BUILD_CLIENTS_BENCHMARKS )
  option( BUILD_CLIENTS_BENCHMARKS "Build rocBLAS benchmarks" OFF )
endif( )

if( NOT LINK_BLIS )
  option( LINK_BLIS "rocBLAS clients link AOCL BLIS library for CPU reference BLAS" ON )
endif( )
