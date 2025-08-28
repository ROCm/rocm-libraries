# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

message( STATUS "Configuring lapack external dependency" )
include( ExternalProject )

# set( lapack_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package )
set( PREFIX_LAPACK ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where lapack should install, defaults to /usr/local" )
set( lapack_cmake_args -DCMAKE_INSTALL_PREFIX=${PREFIX_LAPACK} -DBUILD_SHARED_LIBS=OFF )
append_cmake_cli_arguments( lapack_cmake_args lapack_cmake_args )

set( lapack_git_repository "https://github.com/Reference-LAPACK/lapack-release" CACHE STRING "URL to download lapack from" )
set( lapack_git_tag "lapack-3.7.1" CACHE STRING "git branch" )

# message( STATUS "lapack_make ( " ${lapack_make} " ) " )
# message( STATUS "lapack_cmake_args ( " ${lapack_cmake_args} " ) " )

enable_language( Fortran )
include( GNUInstallDirs )

# lapack cmake exports has a bug on debian architectures, they do not take into account the
# lib/<machine> paths
# if CMAKE_INSTALL_LIBDIR is of the form above, strip the machine
# Match against a '/' in CMAKE_INSTALL_LIBDIR, i.e. lib/x86_64-linux-gnu
if( ${CMAKE_INSTALL_LIBDIR} MATCHES "lib/.*" )
  list( APPEND lapack_cmake_args "-DCMAKE_INSTALL_LIBDIR=lib" )
endif( )

ExternalProject_Add(
  lapack
  PREFIX ${CMAKE_BINARY_DIR}/lapack
  GIT_REPOSITORY ${lapack_git_repository}
  GIT_TAG ${lapack_git_tag}
  CMAKE_ARGS ${lapack_cmake_args} -DCBLAS=ON -DLAPACKE=OFF -DBUILD_TESTING=OFF
  LOG_BUILD 1
  INSTALL_COMMAND ""
  LOG_INSTALL 1
)

ExternalProject_Get_Property( lapack source_dir )

set_property( TARGET lapack PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( lapack install_dir )
ExternalProject_Get_Property( lapack binary_dir )

# For use by the user of ExternalGtest.cmake
set( LAPACK_INSTALL_ROOT ${install_dir} )
set( LAPACK_BINARY_ROOT ${binary_dir} )