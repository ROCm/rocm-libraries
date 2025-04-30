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

message( STATUS "Configuring boost external dependency" )
include( ExternalProject )
set( PREFIX_BOOST ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where boost should install, defaults to /usr/local" )

# We need to detect the compiler the user is attempting to invoke with CMake,
# we do our best to translate cmake parameters into bjam parameters
enable_language( CXX )
include( build-bitness )

# TODO:  Options should be added to allow downloading Boost straight from github

# This file is used to add Boost as a library dependency to another project
# This sets up boost to download from sourceforge, and builds it as a cmake
# ExternalProject

# Change this one line to upgrade to newer versions of boost
set( ext.Boost_VERSION "1.64.0" CACHE STRING "Boost version to download/use" )
mark_as_advanced( ext.Boost_VERSION )
string( REPLACE "." "_" ext.Boost_Version_Underscore ${ext.Boost_VERSION} )

message( STATUS "ext.Boost_VERSION: " ${ext.Boost_VERSION} )

if( WIN32 )
  # For newer cmake versions, 7z archives are much smaller to download
  if( CMAKE_VERSION VERSION_LESS "3.1.0" )
    set( Boost_Ext "zip" )
  else( )
    set( Boost_Ext "7z" )
  endif( )
else( )
  set( Boost_Ext "tar.bz2" )
endif( )

if( WIN32 )
  set( Boost.Command b2 --prefix=${PREFIX_BOOST} )
else( )
  set( Boost.Command ./b2 --prefix=${PREFIX_BOOST} )
endif( )

if( CMAKE_COMPILER_IS_GNUCXX )
  list( APPEND Boost.Command cxxflags=-fPIC -std=c++11 )
elseif( XCODE_VERSION OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang") )
  list( APPEND Boost.Command cxxflags=-std=c++11 -stdlib=libc++ linkflags=-stdlib=libc++ )
endif( )

include( ProcessorCount )
ProcessorCount( Cores )
if( NOT Cores EQUAL 0 )
  # Travis can fail to build Boost sporadically; uses 32 cores, reduce stress on VM
  if( DEFINED ENV{TRAVIS} )
    if( Cores GREATER 8 )
      set( Cores 8 )
    endif( )
  endif( )

  # Add build thread in addition to the number of cores that we have
  math( EXPR Cores "${Cores} + 1 " )
else( )
  # If we could not detect # of cores, assume 1 core and add an additional build thread
  set( Cores "2" )
endif( )

message( STATUS "ExternalBoost using ( " ${Cores} " ) cores to build with" )
message( STATUS "ExternalBoost building [ serialization, filesystem, system, regex ] components" )

list( APPEND Boost.Command -j ${Cores} --with-serialization --with-filesystem --with-system --with-regex )

if( BUILD_64 )
  list( APPEND Boost.Command address-model=64 )
else( )
  list( APPEND Boost.Command address-model=32 )
endif( )

if( MSVC10 )
  list( APPEND Boost.Command toolset=msvc-10.0 )
elseif( MSVC11 )
  list( APPEND Boost.Command toolset=msvc-11.0 )
elseif( MSVC12 )
  list( APPEND Boost.Command toolset=msvc-12.0 )
elseif( MSVC14 )
  list( APPEND Boost.Command toolset=msvc-14.0 )
elseif( XCODE_VERSION OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
  list( APPEND Boost.Command toolset=clang )
elseif( CMAKE_COMPILER_IS_GNUCXX )
  list( APPEND Boost.Command toolset=gcc )
endif( )

if( WIN32 AND (ext.Boost_VERSION VERSION_LESS "1.60.0") )
  list( APPEND Boost.Command define=BOOST_LOG_USE_WINNT6_API )
endif( )

if( NOT DEFINED ext.Boost_LINK )
  if( ${BUILD_SHARED_LIBS} MATCHES "ON" )
    set( ext.Boost_LINK "shared" CACHE STRING "Which boost link method?  static | shared | static,shared" )
  else( )
    set( ext.Boost_LINK "static" CACHE STRING "Which boost link method?  static | shared | static,shared" )
  endif( )
endif()
mark_as_advanced( ext.Boost_LINK )

if( WIN32 )
    # Versioned is the default on windows
    set( ext.Boost_LAYOUT "versioned" CACHE STRING "Which boost layout method?  versioned | tagged | system" )

    # For windows, default to build both variants to support the VS IDE
    set( ext.Boost_VARIANT "debug,release" CACHE STRING "Which boost variant?  debug | release | debug,release" )
else( )
    # Tagged builds provide unique enough names to be able to build both variants
    set( ext.Boost_LAYOUT "tagged" CACHE STRING "Which boost layout method?  versioned | tagged | system" )

   # For Linux, typically a build tree only needs one variant
   if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
     set( ext.Boost_VARIANT "debug" CACHE STRING "Which boost variant?  debug | release | debug,release" )
   else( )
     set( ext.Boost_VARIANT "release" CACHE STRING "Which boost variant?  debug | release | debug,release" )
   endif( )
endif( )
mark_as_advanced( ext.Boost_LAYOUT )
mark_as_advanced( ext.Boost_VARIANT )

list( APPEND Boost.Command --layout=${ext.Boost_LAYOUT} link=${ext.Boost_LINK} variant=${ext.Boost_VARIANT} )

message( STATUS "Boost.Command: ${Boost.Command}" )

# If the user has a cached local copy stored somewhere, they can define the full path to the package in a BOOST_URL environment variable
if( DEFINED ENV{BOOST_URL} )
  set( ext.Boost_URL "$ENV{BOOST_URL}" CACHE STRING "URL to download Boost from" )
else( )
  set( ext.Boost_URL "http://sourceforge.net/projects/boost/files/boost/${ext.Boost_VERSION}/boost_${ext.Boost_Version_Underscore}.${Boost_Ext}/download" CACHE STRING "URL to download Boost from" )
endif( )
mark_as_advanced( ext.Boost_URL )

set( Boost.Bootstrap "" )
set( ext.HASH "" )
if( WIN32 )
  set( Boost.Bootstrap "bootstrap.bat" )

  if( CMAKE_VERSION VERSION_LESS "3.1.0" )
    # .zip file
    set( ext.HASH "b99973c805f38b549dbeaf88701c0abeff8b0e8eaa4066df47cac10a32097523" )
  else( )
    # .7z file
    set( ext.HASH "49c6abfeb5b480f6a86119c0d57235966b4690ee6ff9e6401ee868244808d155" )
  endif( )
else( )
  set( Boost.Bootstrap "./bootstrap.sh" )

  # .tar.bz2
  set( ext.HASH "7bcc5caace97baa948931d712ea5f37038dbb1c5d89b43ad4def4ed7cb683332" )

  if( XCODE_VERSION OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
    list( APPEND Boost.Bootstrap --with-toolset=clang )
  endif( )
endif( )

# Below is a fancy CMake command to download, build and install Boost on the users computer
ExternalProject_Add(
  boost
  PREFIX ${CMAKE_BINARY_DIR}/boost
  URL ${ext.Boost_URL}
  URL_HASH SHA256=${ext.HASH}
  UPDATE_COMMAND ${Boost.Bootstrap}
  LOG_UPDATE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${Boost.Command} stage
  BUILD_IN_SOURCE 1
  LOG_BUILD 1
  INSTALL_COMMAND ""
)

set_property( TARGET boost PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( boost install_dir )
ExternalProject_Get_Property( boost binary_dir )

# For use by the user of ExternalGtest.cmake
set( BOOST_INSTALL_ROOT ${install_dir} )
set( BOOST_BINARY_ROOT ${binary_dir} )
