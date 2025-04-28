# ########################################################################
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Set up Tensile Dependency
  # If we want to build a shared rocblaslt lib, force Tensile to build as a static lib to absorb into rocblaslt
  if( BUILD_SHARED_LIBS )
    set( ROCBLASLT_SHARED_LIBS ON )
    set( BUILD_SHARED_LIBS OFF )
  else( )
    set( ROCBLASLT_SHARED_LIBS OFF )
  endif( )

  set( Tensile_RUNTIME_LANGUAGE "HIP" )
  #set( Tensile_LIBRARY_FORMAT "yaml" )

  #TODO update when this feature has been validated
  #set( PACKAGE_TENSILE_LIBRARY ON )
  set( PACKAGE_TENSILE_LIBRARY OFF )

  # Build options list
  if(Tensile_SHORT_FILENAMES)
    set(Tensile_Options ${Tensile_Options} SHORT_FILE_NAMES)
  endif()
  if(Tensile_PRINT_DEBUG)
    set(Tensile_Options ${Tensile_Options} PRINT_DEBUG)
  endif()
  if(PACKAGE_TENSILE_LIBRARY)
    set(Tensile_Options ${Tensile_Options} GENERATE_PACKAGE)
  endif()
  if(Tensile_NO_LAZY_LIBRARY_LOADING)
    set(Tensile_Options ${Tensile_Options} NO_LAZY_LIBRARY_LOADING)
  endif()
  if(Tensile_ASAN_BUILD)
    set(Tensile_Options ${Tensile_Options} ASAN_BUILD)
  endif()
  if(Tensile_KEEP_BUILD_TMP)
    set(Tensile_Options ${Tensile_Options} KEEP_BUILD_TMP)
  endif()
  if(Tensile_NO_COMPRESS)
    set(Tensile_Options ${Tensile_Options} NO_COMPRESS)
  endif()
  if(Tensile_EXPERIMENTAL)
    set(Tensile_Options ${Tensile_Options} EXPERIMENTAL)
  endif()

  # Add a build target for Tensile kernel library
  # Runtime language is HIP by default
  # warning our Tensile_ variables may shadow variable in TensileCreateLibraryFiles
  # thus bypassing the function argument parameter system (mainly the options list) and CPU_THREADS
  if(Tensile_CPU_THREADS MATCHES "^[0-9]+$")
    # only including threads argument if number
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      "TENSILE_LIBRARY_TARGET"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      CPU_THREADS         ${Tensile_CPU_THREADS}
      ${Tensile_Options}
    )
  else()
    TensileCreateLibraryFiles(
      "${CMAKE_CURRENT_SOURCE_DIR}/Tensile/Logic/${Tensile_LOGIC}"
      "${PROJECT_BINARY_DIR}/Tensile"
      "TENSILE_LIBRARY_TARGET"
      ARCHITECTURE        ${Tensile_ARCHITECTURE}
      CODE_OBJECT_VERSION ${Tensile_CODE_OBJECT_VERSION}
      COMPILER            ${Tensile_COMPILER}
      LIBRARY_FORMAT      ${Tensile_LIBRARY_FORMAT}
      ${Tensile_Options}
    )
  endif()

  add_dependencies(TENSILE_LIBRARY_TARGET rocisa)

  if(NOT Tensile_SKIP_BUILD)
    add_subdirectory(extops)
  endif()

  # Create a unique name for TensileHost compiled for rocBLAS
  set_target_properties( TensileHost PROPERTIES OUTPUT_NAME rocblaslt-tensile CXX_EXTENSIONS NO )
  set_target_properties( TensileHost PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties( TensileHost PROPERTIES C_VISIBILITY_PRESET hidden)

  # Tensile host depends on libs build target
  if(NOT Tensile_SKIP_BUILD)
    add_dependencies( TensileHost TENSILE_LIBRARY_TARGET )
  endif()

  if( ROCBLASLT_SHARED_LIBS )
    set( BUILD_SHARED_LIBS ON )
    set_target_properties( TensileHost PROPERTIES POSITION_INDEPENDENT_CODE ON )
  endif()

# Run gcc --version and capture the first line of the output
execute_process(
    COMMAND gcc --version
    OUTPUT_VARIABLE GCC_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract just the version number from the first line using regular expressions
string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+)" GCC_VERSION "${GCC_OUTPUT}")

# Print the extracted GCC version number
message(STATUS "Extracted GCC Version: ${GCC_VERSION}")

# Compare GCC version and set the flag conditionally if it's 7.5.0 or less
if(${GCC_VERSION} VERSION_LESS "7.6.0")
    # Apply -ftemplate-depth=2048 to tensile_host.cpp if GCC_VERSION is less than 7.6.0 to avoid template recursion
    set_source_files_properties(src/amd_detail/rocblaslt/src/tensile_host.cpp PROPERTIES COMPILE_FLAGS "-ftemplate-depth=2048")
endif()

set_source_files_properties(tensile_host.cpp PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-std=c++17")

if(USE_ROCROLLER)
  target_sources(hipblaslt
    PRIVATE
      rocroller_host.cpp
  )
endif()

if(NOT Tensile_SKIP_BUILD)
    add_subdirectory(kernels)
endif()

# rocBLASLt source
target_sources(
  hipblaslt
      PRIVATE
          Debug.cpp
          handle.cpp
          status.cpp
          rocblaslt_auxiliary.cpp
          rocblaslt_mat.cpp
          utility.cpp
          rocblaslt_transform.cpp
          UserDrivenTuningParser.cpp
          tensile_host.cpp
)
