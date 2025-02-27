# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

include(CheckCXXCompilerFlag)

option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build rocBLAS as a shared library" ON )

# Building tensile can add significant compile time; this option allows to build
# library without tensile to allow for rapid iteration without GEMM functionality
option( BUILD_WITH_TENSILE "Build full functionality which requires tensile?" ON )

include(clients/cmake/client-build-options.cmake)

if (WIN32)
  # not supported on windows so set off
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

# this file is intended to be loaded by toolchain or early as sets global compiler flags
# rocm-cmake checks will throw warnings if set later as cmake watchers installed


option(BUILD_OFFLOAD_COMPRESS "Build rocBLAS with offload compression" ON)
if (BUILD_OFFLOAD_COMPRESS)
  check_cxx_compiler_flag("--offload-compress" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if (NOT CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    message( STATUS "WARNING: BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option." )
  endif()
endif()

# FOR OPTIONAL CODE COVERAGE
option(BUILD_CODE_COVERAGE "Build rocBLAS with code coverage enabled" OFF)

# FOR OPTIONAL ADDRESS SANITIZER
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)

# FOR OPTIONAL HEADER TESTING
option(RUN_HEADER_TESTING "Post build header compatibility testing" OFF)

if( BUILD_WITH_TENSILE )

  set( Tensile_CPU_THREADS "" CACHE STRING "Number of threads for Tensile parallel build")

  set( Tensile_LOGIC "asm_full" CACHE STRING "Tensile to use which logic?")
  set( Tensile_CODE_OBJECT_VERSION "default" CACHE STRING "Tensile code_object_version")
  set( Tensile_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "Tensile compiler")
  set( Tensile_LIBRARY_FORMAT "msgpack" CACHE STRING "Tensile library format")

  set_property( CACHE Tensile_LOGIC PROPERTY STRINGS aldebaran asm_full asm_lite asm_miopen hip_lite other )
  set_property( CACHE Tensile_CODE_OBJECT_VERSION PROPERTY STRINGS default V4 V5 )
  set_property( CACHE Tensile_COMPILER PROPERTY STRINGS amdclang++ hipcc clang++)
  set_property( CACHE Tensile_LIBRARY_FORMAT PROPERTY STRINGS msgpack yaml)

  option( Tensile_MERGE_FILES "Tensile to merge kernels and solutions files?" ON )
  option( Tensile_SHORT_FILENAMES "Tensile to use short file names? Use if compiler complains they're too long." OFF )
  option( Tensile_PRINT_DEBUG "Tensile to print runtime debug info?" OFF )
  option( Tensile_SEPARATE_ARCHITECTURES "Tensile to use GPU architecture specific files?" ON )
  option( Tensile_LAZY_LIBRARY_LOADING "Tensile to load kernels on demand?" ON )

  if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
    option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
    option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." OFF)
  else()
    option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." OFF)
    option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." ON)
  endif()

  option( TENSILE_VENV_UPGRADE_PIP "Upgrade pip in Tensile virtual environment" OFF)
  option( BUILD_WITH_PIP "Use pip to install Python dependencies" ON)

endif()


# FOR HANDLING ENABLE/DISABLE OPTIONAL BACKWARD COMPATIBILITY for FILE/FOLDER REORG
option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY "Build with file/folder reorg with backward compatibility enabled" OFF)
