#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#####################################################################################
if(COMMAND find_python)
    return()
endif()

macro(py_exec)
    execute_process(${ARGN} RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "Process failed: ${ARGN}")
    endif()
endmacro()

set(PYBIND11_NOPYTHON On)
find_package(pybind11 REQUIRED)
macro(find_python version)
    find_program(PYTHON_CONFIG_${version} python${version}-config)
    if(EXISTS ${PYTHON_CONFIG_${version}})
        py_exec(COMMAND ${PYTHON_CONFIG_${version}} --includes OUTPUT_VARIABLE _python_include_args)
        execute_process(COMMAND ${PYTHON_CONFIG_${version}} --ldflags --embed OUTPUT_VARIABLE _python_ldflags_args RESULT_VARIABLE _python_ldflags_result)
        if(NOT _python_ldflags_result EQUAL 0)
            py_exec(COMMAND ${PYTHON_CONFIG_${version}} --ldflags OUTPUT_VARIABLE _python_ldflags_args)
        endif()
        separate_arguments(_python_includes UNIX_COMMAND "${_python_include_args}")
        separate_arguments(_python_ldflags UNIX_COMMAND "${_python_ldflags_args}")
        string(REPLACE "-I" "" _python_includes "${_python_includes}")
        add_library(python${version}::headers INTERFACE IMPORTED GLOBAL)
        set_target_properties(python${version}::headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_python_includes}"
        )
        add_library(python${version}::runtime INTERFACE IMPORTED GLOBAL)
        set_target_properties(python${version}::runtime PROPERTIES
            INTERFACE_LINK_OPTIONS "${_python_ldflags}"
            INTERFACE_LINK_LIBRARIES python${version}::headers
        )
        py_exec(COMMAND ${PYTHON_CONFIG_${version}} --prefix OUTPUT_VARIABLE _python_prefix)
        string(STRIP "${_python_prefix}" _python_prefix)
        set(PYTHON_${version}_EXECUTABLE "${_python_prefix}/bin/python${version}" CACHE PATH "")
    endif()
endmacro()
function(py_extension name version)
    set(_python_module_extension ".so")
    if(version VERSION_GREATER_EQUAL 3.0)
        py_exec(COMMAND ${PYTHON_CONFIG_${version}} --extension-suffix OUTPUT_VARIABLE _python_module_extension)
        string(STRIP "${_python_module_extension}" _python_module_extension)
    endif()
    set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX "${_python_module_extension}")
endfunction()
function(py_add_module NAME)
    set(options)
    set(oneValueArgs PYTHON_VERSION PYTHON_MODULE)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(PYTHON_VERSION ${PARSE_PYTHON_VERSION})

    add_library(${NAME} MODULE ${PARSE_UNPARSED_ARGUMENTS})
    pybind11_strip(${NAME})
    py_extension(${NAME} ${PYTHON_VERSION})
    target_link_libraries(${NAME} PRIVATE pybind11::module pybind11::lto python${PYTHON_VERSION}::headers)
    set_target_properties(${NAME} PROPERTIES 
        OUTPUT_NAME ${PARSE_PYTHON_MODULE}
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )

endfunction()
set(PYTHON_SEARCH_VERSIONS 3.5 3.6 3.7 3.8 3.9 3.10 3.11 3.12)
set(PYTHON_DISABLE_VERSIONS "" CACHE STRING "")
foreach(PYTHON_DISABLE_VERSION ${PYTHON_DISABLE_VERSIONS})
    list(REMOVE_ITEM PYTHON_SEARCH_VERSIONS ${PYTHON_DISABLE_VERSION})
endforeach()

set(_PYTHON_VERSIONS)
foreach(PYTHON_VERSION ${PYTHON_SEARCH_VERSIONS})
    find_python(${PYTHON_VERSION})
    if(TARGET python${PYTHON_VERSION}::headers)
        message(STATUS "Python ${PYTHON_VERSION} found.")
        list(APPEND _PYTHON_VERSIONS ${PYTHON_VERSION})
    else()
        message(STATUS "Python ${PYTHON_VERSION} not found.")
    endif()
endforeach()

# Make the variable global
set(PYTHON_VERSIONS "${_PYTHON_VERSIONS}" CACHE INTERNAL "" FORCE)
