# Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- rpp: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_CXX_COMPILER clang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

# Set Project Version and Language
project(rocal_pybind VERSION ${VERSION} LANGUAGES CXX)

set(TARGET_NAME rocal_pybind)

set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

set(BUILD_ROCAL_PYBIND true)
set(CMAKE_SKIP_BUILD_RPATH TRUE)

find_package(AMDRPP QUIET)
find_package(MIVisionX QUIET)
find_package(TurboJpeg QUIET)
if (PYTHON_VERSION_SUGGESTED)
    find_package(Python3 "${PYTHON_VERSION_SUGGESTED}" EXACT QUIET COMPONENTS Interpreter Development)
else()
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET COMPONENTS Interpreter Development)
endif()	
find_package(pybind11 QUIET CONFIG)	
find_package(HALF QUIET)
find_package(OpenMP QUIET)

# Backend
if(BUILD_PYPACKAGE)
    if(NOT DEFINED BACKEND)
        set(BACKEND "HIP")   # set default backend to HIP
    else()
        if("${BACKEND}" STREQUAL "OPENCL" OR "${BACKEND}" STREQUAL "OCL" OR "${BACKEND}" STREQUAL "OpenCL")
            set(BACKEND "OPENCL")
        elseif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "hip")
            set(BACKEND "HIP")
        elseif("${BACKEND}" STREQUAL "host" OR "${BACKEND}" STREQUAL "CPU" OR "${BACKEND}" STREQUAL "cpu")
            set(BACKEND "CPU")
            set(GPU_SUPPORT OFF)
        else()
            message("-- ${Red}Warning: rocAL PyBind backend option: ${BACKEND} -- unknown${ColourReset}")
            message("-- ${Red}Warning: rocAL PyBind default backend will be enforced${ColourReset}")
            set(BACKEND "HIP")
        endif()
    endif()
    message("-- ${BoldBlue}rocAL PyBind Backend set to -- ${BACKEND}${ColourReset}")

    # Find HIP Support
    if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
        if(NOT DEFINED HIP_PATH)
            if(NOT DEFINED ENV{HIP_PATH})
                set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
            else()
                set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
            endif()
        endif()
        list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
        find_package(HIP QUIET)
        if(HIP_FOUND)
            message("-- ${White}rocAL PyBind -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
        else()
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Yellow}rocAL PyBind requires HIP, Not Found${ColourReset}")
        endif()
    endif()

    # Find OpenCL Support
    if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            message("-- ${White}rocAL PyBind -- Using OpenCL -- Path:" ${OpenCL_LIBRARIES} "\tVersion:" ${CL_TARGET_OpenCL_VERSION})
        else()
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Yellow}rocAL PyBind requires OpenCL, Not Found${ColourReset}")
        endif()
    endif()
endif()

# Dependency Check
# Find AMD RPP Support
if(NOT AMDRPP_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL PyBind requires AMD RPP, Not Found${ColourReset}")
else()
    #find the RPP backend type
    set(RPP_BACKEND_OPENCL_FOUND 0)
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h)
        file(READ ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_OPENCL ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "${Red}ERROR: ${AMDRPP_INCLUDE_DIRS}/rpp_backend.h file Not Found. AMD RPP Package REQUIRED ${ColourReset}")
    endif()
    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if (NOT RPP_BACKEND_OPENCL_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}ERROR: rocAL PyBind -- RPP with OpenCL backend Not Found${ColourReset}")
        endif()
        if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
            if (NOT RPP_BACKEND_OPENCL_FOUND)
                set(BUILD_ROCAL_PYBIND false)
                message("-- ${Red}ERROR: rocAL PyBind -- RPP with OpenCL backend Not Found${ColourReset}")
            endif()
        elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
            if (NOT RPP_BACKEND_HIP_FOUND)
                set(BUILD_ROCAL_PYBIND false)
                message("-- ${Red}ERROR: rocAL PyBind -- RPP with the HIP backend Not Found${ColourReset}")
            endif()
        endif()
    endif()
endif()
# MIVisionX
if(NOT MIVisionX_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL PyBind library requires AMD's MIVisionX library, Not Found ${ColourReset}")
else()
    #find the MIVisionX backend type
    set(MIVISIONX_BACKEND_OPENCL_FOUND 0)
    set(MIVISIONX_BACKEND_HIP_FOUND 0)
    if(EXISTS ${MIVisionX_INCLUDE_DIRS}/openvx_backend.h)
        file(READ ${MIVisionX_INCLUDE_DIRS}/openvx_backend.h MIVISIONX_BACKEND_FILE)
        string(REGEX MATCH "ENABLE_OPENCL ([0-9]*)" _ ${MIVISIONX_BACKEND_FILE})
        set(MIVISIONX_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
        string(REGEX MATCH "ENABLE_HIP ([0-9]*)" _ ${MIVISIONX_BACKEND_FILE})
        set(MIVISIONX_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "${Red}ERROR: ${MIVisionX_INCLUDE_DIRS}/openvx_backend.h file Not Found. MIVisionX Package REQUIRED ${ColourReset}")
    endif()

    if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
        if (NOT MIVISIONX_BACKEND_OPENCL_FOUND)
            set(BUILD_ROCAL_PYBIND false)
            message("-- ${Red}ERROR: rocAL PyBind -- MIVisionX with OpenCL backend Not Found${ColourReset}")
        endif()

        if ("${BACKEND}" STREQUAL "OPENCL" AND OpenCL_FOUND)
            if (NOT MIVISIONX_BACKEND_OPENCL_FOUND)
                set(BUILD_ROCAL_PYBIND false)
                message("-- ${Red}ERROR: rocAL PyBind -- MIVisionX with OpenCL backend Not Found${ColourReset}")
            endif()
        elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
            if (NOT MIVISIONX_BACKEND_HIP_FOUND)
                set(BUILD_ROCAL_PYBIND false)
                message("-- ${Red}ERROR: rocAL PyBind -- MIVisionX with the HIP backend Not Found${ColourReset}")
            endif()
        endif()
    endif()
endif()
# Turbo JPEG
if(NOT TurboJpeg_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires TurboJpeg, Not Found${ColourReset}")
endif()
# Python3	
if(NOT Python3_FOUND)	
    set(BUILD_ROCAL_PYBIND false)	
    message("-- ${Yellow}NOTE: rocAL pybind requires Python3, Not Found${ColourReset}")
endif()
# PyBind11
if(NOT pybind11_FOUND)	
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires pybind11, Not Found${ColourReset}")
endif()
# Half
if(NOT HALF_FOUND)	
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind requires Half, Not Found${ColourReset}")
endif()
# OpenMP
if(NOT OpenMP_FOUND)
    set(BUILD_ROCAL_PYBIND false)
    message("-- ${Yellow}NOTE: rocAL pybind library requires OpenMP, Not Found ${ColourReset}")
endif()

if(${BUILD_ROCAL_PYBIND})
    # avoid setting the default installation path to /usr/local
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)	
        message("-- ${Green}NOTE: rocAL PyBind default installation path -- ${Python3_STDARCH} ${ColourReset}")	
        set(CMAKE_INSTALL_PREFIX_PYTHON ${Python3_STDARCH} CACHE PATH "rocAL PyBind default installation path" FORCE)	
    endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)	
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # AMD RPP
    include_directories(${AMDRPP_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${AMDRPP_LIBRARIES})
    # MIVisionX
    include_directories(${MIVisionX_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OPENVX_LIBRARIES} ${VXRPP_LIBRARIES})
    # TurboJPEG
    include_directories(${TurboJpeg_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES})
    # rocAL
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocal)
    include_directories(../rocAL/include/api/)
    # Python3	
    include_directories(${Python3_INCLUDE_DIRS})	
    message("-- ${Green}NOTE: rocAL pybind requires Python3, Found --  ${Python3_EXECUTABLE} ${ColourReset}")	
    # PyBind11	
    include_directories(${pybind11_INCLUDE_DIRS})
    # half
    include_directories(${HALF_INCLUDE_DIRS})
    # OpenMP
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenMP_CXX_LIBRARIES})

    file(GLOB_RECURSE pyfiles amd/*.py)
    file(GLOB_RECURSE sources *.cpp)

    pybind11_add_module(rocal_pybind MODULE ${sources})
    target_link_libraries(rocal_pybind PRIVATE ${LINK_LIBRARY_LIST})
    add_dependencies(rocal_pybind rocal)
    message("-- ${White}rocal_pybind -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
    message("-- ${White}rocal_pybind -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")

    #DLPack
    find_package(DLPACK QUIET)
    if(DLPACK_FOUND)
        include_directories(${DLPACK_INCLUDE_DIRS})
        target_compile_definitions(rocal_pybind PUBLIC ENABLE_DLPACK=1)
    else()
        target_compile_definitions(rocal_pybind PUBLIC ENABLE_DLPACK=0)
        message("-- ${Yellow}NOTE: DLPack Not Found - rocAL Tensorflow and Generic plugins built without GPU support!${ColourReset}")
    endif()

    set_target_properties(rocal_pybind PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${CMAKE_INSTALL_LIBDIR}")
    set(TARGET_PYMODULE_NAME "${PYTHON_MODULE_PREFIX}rocal_pybind${PYTHON_MODULE_EXTENSION}")

    CONFIGURE_FILE(amd/__init__.py
                ${CMAKE_BINARY_DIR}/${TARGET_NAME}/__init__.py @ONLY
                )
    CONFIGURE_FILE(setup.py
                ${CMAKE_BINARY_DIR}/setup.py @ONLY
                )
    CONFIGURE_FILE(getrocALwheelname.py
                ${CMAKE_BINARY_DIR}/getrocALwheelname.py @ONLY
                )
    CONFIGURE_FILE(rocal_pybind.cpp
                ${CMAKE_BINARY_DIR}/rocal_pybind.cpp @ONLY
                )

    foreach (filename ${pyfiles})
        get_filename_component(target "${filename}" REALPATH)
        #to maintain folder structure
        file(RELATIVE_PATH ITEM_PATH_REL ${CMAKE_CURRENT_SOURCE_DIR} ${filename})
        message(STATUS "Copying ${filename} to ${TARGET_NAME}/${ITEM_PATH_REL}")
        configure_file("${filename}" "${CMAKE_BINARY_DIR}/${TARGET_NAME}/${ITEM_PATH_REL}" COPYONLY)
    endforeach (filename)

    execute_process(COMMAND "${Python3_EXECUTABLE}" getrocALwheelname.py OUTPUT_VARIABLE PY_WHEELFILE
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    message(STATUS "Detected rocAL Python wheel filename ${PY_WHEELFILE}")
        
    add_custom_target(PyPackageBuild
        COMMAND "${Python3_EXECUTABLE}" -m pip wheel .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Building python wheel package"
        )

    add_dependencies(PyPackageBuild rocal_pybind)

    set_target_properties(PyPackageBuild PROPERTIES Package_Name ${PY_WHEELFILE})
    add_custom_target(PyPackageInstall
            COMMAND "${Python3_EXECUTABLE}" -m pip install --upgrade --force-reinstall --no-deps "${PY_WHEELFILE}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Installing wheel file"
            )
    add_dependencies(PyPackageInstall PyPackageBuild)
        
    #install rocAL pybind libs
    install(FILES ${CMAKE_BINARY_DIR}/${TARGET_NAME}/${CMAKE_INSTALL_LIBDIR}/${TARGET_PYMODULE_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)
    install(DIRECTORY amd DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)
    message("-- ${Green}Radeon Augmentation Library Python Binding - rocal_pybind module added ${ColourReset}")

    # PyBind Test - Installed
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/pybind DESTINATION ${CMAKE_INSTALL_DATADIR}/rocal/test COMPONENT test)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/python_api DESTINATION ${CMAKE_INSTALL_DATADIR}/rocal/test COMPONENT test)
else()
    message("-- ${Red}WARNING: rocAL PyBind Module excluded - Dependency Failure${ColourReset}")
endif()