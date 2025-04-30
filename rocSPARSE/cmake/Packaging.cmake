# ########################################################################
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

# Client packaging
include(CMakeParseArguments)

function(rocm_create_package_clients)
    set(options)
    set(oneValueArgs LIB_NAME DESCRIPTION SECTION MAINTAINER VERSION)
    set(multiValueArgs DEPENDS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    string(CONCAT PACKAGE_NAME ${PARSE_LIB_NAME} "-clients-" ${PARSE_VERSION} "-Linux.deb")
    string(CONCAT DEB_CONTROL_FILE_CONTENT "Package: " ${PARSE_LIB_NAME} "-clients"
                                           "\nVersion: " ${PARSE_VERSION}
                                           "\nSection: " ${PARSE_SECTION}
                                           "\nPriority: optional"
                                           "\nArchitecture: amd64"
                                           "\nMaintainer: " ${PARSE_MAINTAINER}
                                           "\nDescription: " ${PARSE_DESCRIPTION}
                                           "\nDepends: " ${PARSE_LIB_NAME} "(>=" ${PARSE_VERSION} ")\n\n")

    if(EXISTS "${PROJECT_BINARY_DIR}/package")
        file(REMOVE_RECURSE "${PROJECT_BINARY_DIR}/package")
    endif()
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/package/${ROCM_PATH}/bin/${PARSE_LIB_NAME}")
    file(WRITE "${PROJECT_BINARY_DIR}/package/DEBIAN/control" ${DEB_CONTROL_FILE_CONTENT})

    add_custom_target(package_clients
        COMMAND ${CMAKE_COMMAND} -E remove -f "${PROJECT_BINARY_DIR}/package/${ROCM_PATH}/bin/${PARSE_LIB_NAME}/*"
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/*" "${PROJECT_BINARY_DIR}/package/${ROCM_PATH}/bin/${PARSE_LIB_NAME}/"
        COMMAND dpkg -b "${PROJECT_BINARY_DIR}/package/"  ${PACKAGE_NAME})
endfunction(rocm_create_package_clients)
