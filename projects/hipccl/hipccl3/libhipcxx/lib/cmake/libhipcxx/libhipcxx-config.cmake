# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# find_package(libhipcxx) config file.
#
# Defines a libhipcxx::libhipcxx target that may be linked from user projects to include
# libhipcxx.

if (TARGET libhipcxx::libhipcxx)
  return()
endif()

function(_libhipcxx_declare_interface_alias alias_name ugly_name)
  # 1) Only IMPORTED and ALIAS targets can be placed in a namespace.
  # 2) When an IMPORTED library is linked to another target, its include
  #    directories are treated as SYSTEM includes.
  # 3) nvcc will automatically check the CUDA Toolkit include path *before* the
  #    system includes. This means that the Toolkit libhipcxx will *always* be used
  #    during compilation, and the include paths of an IMPORTED libhipcxx::libhipcxx
  #    target will never have any effect.
  # 4) This behavior can be fixed by setting the property NO_SYSTEM_FROM_IMPORTED
  #    on EVERY target that links to libhipcxx::libhipcxx. This would be a burden and a
  #    footgun for our users. Forgetting this would silently pull in the wrong libhipcxx!
  # 5) A workaround is to make a non-IMPORTED library outside of the namespace,
  #    configure it, and then ALIAS it into the namespace (or ALIAS and then
  #    configure, that seems to work too).
  add_library(${ugly_name} INTERFACE)

  add_library(${alias_name} INTERFACE IMPORTED GLOBAL)
  target_link_libraries(${alias_name} INTERFACE ${ugly_name})
endfunction()

#
# Setup targets
#

_libhipcxx_declare_interface_alias(libhipcxx::libhipcxx _libhipcxx_libhipcxx)
# Pull in the include dir detected by libhipcxx-config-version.cmake
set(_libhipcxx_INCLUDE_DIR "${_libhipcxx_VERSION_INCLUDE_DIR}"
  CACHE INTERNAL "Location of libhipcxx headers."
)
unset(_libhipcxx_VERSION_INCLUDE_DIR CACHE) # Clear tmp variable from cache
target_include_directories(_libhipcxx_libhipcxx INTERFACE "${_libhipcxx_INCLUDE_DIR}")

#
# Standardize version info
#

set(LIBCUDACXX_VERSION ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} CACHE INTERNAL "")
set(LIBCUDACXX_VERSION_MAJOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR} CACHE INTERNAL "")
set(LIBCUDACXX_VERSION_MINOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR} CACHE INTERNAL "")
set(LIBCUDACXX_VERSION_PATCH ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH} CACHE INTERNAL "")
set(LIBCUDACXX_VERSION_TWEAK ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK} CACHE INTERNAL "")
set(LIBCUDACXX_VERSION_COUNT ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
if (NOT libhipcxx_CONFIG)
  set(libhipcxx_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(libhipcxx CONFIG_MODE)
