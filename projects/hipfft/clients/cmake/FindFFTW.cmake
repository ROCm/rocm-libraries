# #############################################################################
# Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#if( FFTW_FIND_VERSION VERSION_LESS "3" )
#    message( FFTW_FIND_VERION is ${FFTW_FIND_VERSION})
#    message( FATAL_ERROR "FindFFTW can not configure versions less than FFTW 3.0.0" )
#endif( )

find_path(FFTW_INCLUDE_DIRS
    NAMES fftw3.h
    HINTS
        ${FFTW_ROOT}/include
        $ENV{FFTW_ROOT}/include
    PATHS
        /usr/include
        /usr/local/include
)
mark_as_advanced( FFTW_INCLUDE_DIRS )

# message( STATUS "FFTW_FIND_COMPONENTS: ${FFTW_FIND_COMPONENTS}" )
# message( STATUS "FFTW_FIND_REQUIRED_FLOAT: ${FFTW_FIND_REQUIRED_FLOAT}" )
# message( STATUS "FFTW_FIND_REQUIRED_DOUBLE: ${FFTW_FIND_REQUIRED_DOUBLE}" )

set( FFTW_LIBRARIES "" )
if( FFTW_FIND_REQUIRED_FLOAT OR FFTW_FIND_REQUIRED_SINGLE )
  find_library( FFTW_LIBRARIES_SINGLE
      NAMES fftw3f fftw3f-3 fftw3 fftw3-3
      HINTS
          ${FFTW_ROOT}/lib
          $ENV{FFTW_ROOT}/lib
      PATHS
          /usr/lib
          /usr/local/lib
      PATH_SUFFIXES
          x86_64-linux-gnu
      DOC "FFTW dynamic library single"
  )
  mark_as_advanced( FFTW_LIBRARIES_SINGLE )
  list( APPEND FFTW_LIBRARIES ${FFTW_LIBRARIES_SINGLE} )

  # Look for omp (preferred) or thread libraries.  These are not a
  # hard requirement, but are nice to have to make FFTW run faster.
  find_library( FFTWF_OMP_LIBRARY fftw3f_omp )
  find_library( FFTWF_THREADS_LIBRARY fftw3f_threads )
  if( FFTWF_OMP_LIBRARY )
    list( APPEND FFTW_LIBRARIES ${FFTWF_OMP_LIBRARY} )
    set( FFTW_MULTITHREAD TRUE )
  elseif( FFTWF_THREADS_LIBRARY )
    list( APPEND FFTW_LIBRARIES ${FFTWF_THREADS_LIBRARY} )
    set( FFTW_MULTITHREAD TRUE )
  endif()
endif( )

if( FFTW_FIND_REQUIRED_DOUBLE )
  find_library( FFTW_LIBRARIES_DOUBLE
      NAMES fftw3
      HINTS
          ${FFTW_ROOT}/lib
          $ENV{FFTW_ROOT}/lib
      PATHS
          /usr/lib
          /usr/local/lib
      PATH_SUFFIXES
          x86_64-linux-gnu
      DOC "FFTW dynamic library double"
  )
  mark_as_advanced( FFTW_LIBRARIES_DOUBLE )
  list( APPEND FFTW_LIBRARIES ${FFTW_LIBRARIES_DOUBLE} )

  # Look for omp (preferred) or thread libraries.  These are not a
  # hard requirement, but are nice to have to make FFTW run faster.
  find_library( FFTW_OMP_LIBRARY fftw3_omp )
  find_library( FFTW_THREADS_LIBRARY fftw3_threads )
  if( FFTW_OMP_LIBRARY )
    list( APPEND FFTW_LIBRARIES ${FFTW_OMP_LIBRARY} )
    set( FFTW_MULTITHREAD TRUE )
  elseif( FFTW_THREADS_LIBRARY )
    list( APPEND FFTW_LIBRARIES ${FFTW_THREADS_LIBRARY} )
    set( FFTW_MULTITHREAD TRUE )
  endif()
endif( )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( FFTW
    REQUIRED_VARS FFTW_INCLUDE_DIRS FFTW_LIBRARIES )

# assume the threads feature is always enabled on Windows, since it's
# not a separate library there
if( FFTW_FOUND AND WIN32 )
  set( FFTW_MULTITHREAD TRUE )
endif()

if( NOT FFTW_FOUND )
    message( STATUS "FindFFTW could not find all of the following fftw libraries" )
    message( STATUS "${FFTW_FIND_COMPONENTS}" )
else( )
    message(STATUS "FindFFTW configured variables:" )
    message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW_INCLUDE_DIRS}" )
    message(STATUS "FFTW_LIBRARIES: ${FFTW_LIBRARIES}" )
endif()
