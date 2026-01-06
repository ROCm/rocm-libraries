# ##############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#
# ##############################################################################

cmake_minimum_required(VERSION 3.21.2)
project(tensile VERSION 4.44.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CMakeDependentOption)
include(TensileSupportedArchitectures)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(TENSILE_BUILD_SHARED_LIBS "Build shared library" OFF)
option(TENSILE_BUILD_TESTING "Build host library tests" ON)
option(TENSILE_ENABLE_HOST "Build  host library" ON)
option(TENSILE_ENABLE_DEVICE "Build device libraries" ON)
option(TENSILE_ENABLE_CLIENT "Build client app" ON)
option(TENSILE_ENABLE_MSGPACK "Enable MessagePack support" ON)
option(TENSILE_ENABLE_LLVM "Use llvm yaml library" OFF)
cmake_dependent_option(TENSILE_ENABLE_ROCM_SMI "Use rocm_smi." ON "NOT WIN32" OFF)
set(TENSILE_SOURCE_TREE "${CMAKE_CURRENT_SOURCE_DIR}/../Tensile/Source")
set(TENSILE_CONFIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../HostLibraryTests/configs")
set(TENSILE_CXX_STANDARD
    "17"
    CACHE STRING "C++ standard version to use (14, 17, 20)")
set_property(CACHE TENSILE_CXX_STANDARD PROPERTY STRINGS "14" "17" "20")
set(GPU_TARGETS "" CACHE STRING "AMD GFX targets to cross-compile")

if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
    tensile_get_base_architectures(GPU_TARGETS)
else()
    tensile_validate_gpu_targets("${GPU_TARGETS}")
endif()
message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

find_package(OpenMP REQUIRED)

if(TENSILE_ENABLE_HOST)
  find_package(hip REQUIRED)
  if(TENSILE_BUILD_SHARED_LIBS)
    add_library(tensile-host SHARED)
  else()
    add_library(tensile-host STATIC)
  endif()
  add_library(roc::tensile-host ALIAS tensile-host)
  target_compile_features(tensile-host PUBLIC cxx_std_${TENSILE_CXX_STANDARD})
  set_target_properties(tensile-host PROPERTIES CXX_EXTENSIONS OFF)
  target_link_libraries(tensile-host PUBLIC hip::device)
  if(NOT TENSILE_BUILD_SHARED_LIBS)
    set_target_properties(tensile-host PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  endif()

  if(TENSILE_ENABLE_MSGPACK)
    # See: https://github.com/msgpack/msgpack-c/wiki/Q%26A#how-to-support-both-msgpack-c-c-version-5x-and-6x-
    # Prefer 6.x (msgpack-cxx) as that is what we bundle in the build.
    find_package(msgpack-cxx CONFIG)
    if(msgpack-cxx_FOUND)
        message(STATUS "Found msgpack-cxx (>=6.x)")
        set(msgpack_target msgpack-cxx)
    else()
        find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
        message(STATUS "Found msgpack (<=5.x)")
        set(msgpack_target msgpackc)
    endif()
    get_target_property(msgpack_inc ${msgpack_target} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(tensile-host PRIVATE "${msgpack_inc}")
    target_compile_definitions(tensile-host PRIVATE TENSILE_MSGPACK=1)
  endif()

  target_include_directories(
    tensile-host
    PUBLIC $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/lib/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE "${TENSILE_SOURCE_TREE}/lib/include/Tensile")

  if(TENSILE_ENABLE_LLVM)
    target_compile_definitions(tensile-host PUBLIC -DTENSILE_YAML=1)
    find_package(LLVM REQUIRED)
    find_library(
      LLVMObjectYAML_LIBRARY
      NAMES LLVMObjectYAML
      PATHS ${LLVM_LIBRARY_DIR})
    if(LLVMObjectYAML_LIBRARY)
        target_link_libraries(tensile-host PRIVATE LLVMObjectYAML)
    else()
      target_link_libraries(tensile-host PRIVATE LLVM)
    endif()
    target_include_directories(tensile-host PRIVATE ${LLVM_INCLUDE_DIRS})
  endif()

  if(TENSILE_ENABLE_LLVM OR TENSILE_ENABLE_MSGPACK)
    target_compile_definitions(tensile-host PUBLIC TENSILE_DEFAULT_SERIALIZATION)
  endif()

  if(NOT TENSILE_BUILD_SHARED_LIBS)
    target_compile_definitions(tensile-host PUBLIC TENSILE_STATIC_ONLY)
  endif()

  target_compile_definitions(tensile-host PUBLIC TENSILE_USE_HIP)

  add_subdirectory(host-library)
endif()

if(TENSILE_ENABLE_DEVICE)
  add_subdirectory(device-library)
endif()

if(TENSILE_ENABLE_ROCM_SMI)
  find_package(rocm_smi REQUIRED)
endif()

if(TENSILE_ENABLE_CLIENT)
  find_package(Boost REQUIRED COMPONENTS program_options)
  add_library(tensile-client-lib OBJECT)
  add_library(tensile::tensile-client-lib ALIAS tensile-client-lib)
  target_link_libraries(
    tensile-client-lib PRIVATE Boost::boost Boost::program_options tensile-host OpenMP::OpenMP_CXX)
    target_compile_features(tensile-client-lib PUBLIC cxx_std_${TENSILE_CXX_STANDARD})

  if(rocm_smi_FOUND)
    target_link_libraries(tensile-client-lib PRIVATE rocm_smi64)
  endif()
  target_include_directories(
    tensile-client-lib
    PUBLIC $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/client/include>)

  add_executable(tensile-client)
  target_link_libraries(
    tensile-client PRIVATE roc::tensile-host tensile::tensile-client-lib)
  target_compile_features(tensile-client PUBLIC cxx_std_${TENSILE_CXX_STANDARD})
  target_compile_definitions(
    tensile-client
    PRIVATE BOOST_ALL_NO_LIB BOOST_PROGRAM_OPTIONS_DYN_LINK)
  add_subdirectory(client)
endif()

if(TENSILE_BUILD_TESTING)
  find_package(Boost REQUIRED COMPONENTS filesystem)
  find_package(GTest REQUIRED)

  add_library(tensile-test-lib OBJECT)
  add_library(tensile::tensile-test-lib ALIAS tensile-test-lib)
  target_compile_definitions(
    tensile-test-lib
    PRIVATE BOOST_ALL_NO_LIB BOOST_ERROR_CODE_HEADER_ONLY BOOST_FILESYSTEM_DYN_LINK)
  target_compile_features(tensile-test-lib PRIVATE cxx_std_${TENSILE_CXX_STANDARD})
  target_link_libraries(tensile-test-lib PRIVATE roc::tensile-host)

  target_include_directories(
    tensile-test-lib
    PUBLIC
      $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/../../HostLibraryTests/testlib/include>)

  add_executable(tensile-tests)
  target_link_libraries(
    tensile-tests PRIVATE GTest::gtest roc::tensile-host tensile::tensile-test-lib
                          Boost::filesystem hip::host test-device-kernels
                          tensile-client-lib OpenMP::OpenMP_CXX)
  if(TENSILE_ENABLE_LLVM)
      target_link_libraries(tensile-tests PRIVATE LLVMObjectYAML)
      target_compile_features(tensile-tests PRIVATE cxx_std_${TENSILE_CXX_STANDARD})
  endif()

  target_include_directories(
    tensile-tests
    PRIVATE
      $<BUILD_INTERFACE:${TENSILE_SOURCE_TREE}/client/include>)

  # data dir needs to be parallel to tensile-tests
  # because tensile-tests search for data/ relative
  # to its location
  set(TENSILE_TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/../data")
  file(MAKE_DIRECTORY "${TEST_DATA_DIR}")
  add_subdirectory(tests)
endif()
