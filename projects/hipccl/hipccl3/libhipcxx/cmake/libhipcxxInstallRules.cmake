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

option(libhipcxx_ENABLE_INSTALL_RULES
  "Enable installation of libhipcxx" ${LIBCUDACXX_TOPLEVEL_PROJECT}
)

if (NOT libhipcxx_ENABLE_INSTALL_RULES)
  return()
endif()

# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# Libhipcxx headers
install(DIRECTORY "${libhipcxx_SOURCE_DIR}/include/cuda"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN CMakeLists.txt EXCLUDE
)
install(DIRECTORY "${libhipcxx_SOURCE_DIR}/include/nv"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERN CMakeLists.txt EXCLUDE
)

# Copy libhipcxx headers into hip folder additionally
# for minimizing changes of existing dependee packages.
# Note: we can't use symlinks here, as this would
# break builds of packages like hipDF which
# create a Python wheel with setuptools.
install(DIRECTORY "${libhipcxx_SOURCE_DIR}/include/cuda/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hip"
  PATTERN CMakeLists.txt EXCLUDE
)

# Libcudacxx cmake package
install(DIRECTORY "${libhipcxx_SOURCE_DIR}/lib/cmake/libhipcxx"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
  PATTERN *.cmake.in EXCLUDE
)

set(install_location "${CMAKE_INSTALL_LIBDIR}/cmake/libhipcxx")
# Transform to a list of directories, replace each directory with "../"
# and convert back to a string
string(REGEX REPLACE "/" ";" from_install_prefix "${install_location}")
list(TRANSFORM from_install_prefix REPLACE ".+" "../")
list(JOIN from_install_prefix "" from_install_prefix)

# Need to configure a file to store CMAKE_INSTALL_INCLUDEDIR
# since it can be defined by the user. This is common to work around collisions
# with the CTK installed headers.
configure_file("${libhipcxx_SOURCE_DIR}/lib/cmake/libhipcxx/libhipcxx-header-search.cmake.in"
  "${libhipcxx_BINARY_DIR}/lib/cmake/libhipcxx/libhipcxx-header-search.cmake"
  @ONLY
)
install(FILES "${libhipcxx_BINARY_DIR}/lib/cmake/libhipcxx/libhipcxx-header-search.cmake"
  DESTINATION "${install_location}"
)
