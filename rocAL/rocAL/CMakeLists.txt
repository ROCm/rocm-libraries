# MIT License
#
# Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc.
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
project(rocal VERSION ${VERSION} LANGUAGES CXX)

set(BUILD_ROCAL true)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# Find Dependency
find_package(AMDRPP QUIET)
find_package(MIVisionX QUIET)
find_package(TurboJpeg QUIET)
find_package(Protobuf QUIET)
find_package(OpenMP QUIET)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads QUIET)
find_package(LMDB QUIET)	
find_package(RapidJSON QUIET)
find_package(StdFilesystem QUIET)
find_package(HALF QUIET)
# optional packages
find_package(FFmpeg QUIET)
find_package(OpenCV QUIET)
find_package(SndFile QUIET)
find_package(LibTar QUIET)
find_package(ROCDECODE QUIET)
find_package(ROCJPEG QUIET)

# HIP Backend
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
        else()
            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)	
    # Set supported GPU Targets
    if(NOT GPU_TARGETS AND NOT AMDGPU_TARGETS)
        find_package(ROCmCMakeBuildTools QUIET)
        if(NOT ROCmCMakeBuildTools_FOUND)
            find_package(ROCM QUIET)
        endif()
        include(ROCMCheckTargetIds OPTIONAL RESULT_VARIABLE HAS_ROCM_CHECK_TARGET_IDS)

        set(OPTIONAL_GPU_TARGETS "gfx950;gfx1200;gfx1201")
        if(HAS_ROCM_CHECK_TARGET_IDS)
            rocm_check_target_ids(OPTIONAL_GPU_TARGETS_AVAILABLE TARGETS ${OPTIONAL_GPU_TARGETS})
        else() # if we don't have rocm_check_target_ids, just assume the targets are available
            set(OPTIONAL_GPU_TARGETS_AVAILABLE "${OPTIONAL_GPU_TARGETS}")
        endif()
        set(DEFAULT_GPU_TARGETS "gfx908;gfx90a;gfx942;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;${OPTIONAL_GPU_TARGETS_AVAILABLE}")
    endif()

    # Set AMD GPU_TARGETS
    if((AMDGPU_TARGETS OR DEFINED ENV{AMDGPU_TARGETS}) AND (NOT GPU_TARGETS))
        message("-- ${Red}${PROJECT_NAME} DEPRECATION -- AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS${ColourReset}")
        if(DEFINED ENV{AMDGPU_TARGETS})
            set(GPU_TARGETS $ENV{AMDGPU_TARGETS} CACHE STRING "List of specific machine types for library to target")
        else()
            set(GPU_TARGETS ${AMDGPU_TARGETS})
        endif()
    endif()

    if(DEFINED ENV{GPU_ARCHS})
        set(GPU_TARGETS $ENV{GPU_ARCHS} CACHE STRING "List of specific machine types for library to target")
    elseif(GPU_TARGETS)
        message("-- ${White}${PROJECT_NAME} -- GPU_TARGETS set with -D option${ColourReset}")
    else()
        set(GPU_TARGETS "${DEFAULT_GPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")
    endif()
    message("-- ${White}${PROJECT_NAME} -- AMD GPU_TARGETS: ${GPU_TARGETS}${ColourReset}")

    find_package(HIP QUIET)
    if(HIP_FOUND)
        SET(BUILD_ROCAL true)
        message("-- ${White}rocAL -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
    else()
        set(BACKEND "CPU")
        message("-- ${Yellow}NOTE: rocAL library requires HIP for BACKEND=HIP, Not Found ${ColourReset}")
        message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
    endif()
endif()

# OpenCL Backend
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(BUILD_ROCAL true)
        message("-- ${White}rocAL -- Using OpenCL -- Path:" ${OpenCL_LIBRARIES} "\tVersion:" ${CL_TARGET_OpenCL_VERSION})
    else()
        set(BACKEND "CPU")
        message("-- ${Yellow}NOTE: rocAL library requires OpenCL for BACKEND=OpenCL, Not Found ${ColourReset}")
        message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
    endif()
endif()
# Dependency Check
# AMD RPP
if(NOT AMDRPP_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires AMD's rpp library, Not Found ${ColourReset}")
else()
    # find the RPP backend type
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
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: RPP with OpenCL backend Not Found (HINT: run setup script with the OCL backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT RPP_BACKEND_HIP_FOUND)
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: RPP with the HIP backend Not Found (HINT: run setup script with the HIP backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    endif()
endif()
# MIVisionX
if(NOT MIVisionX_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires AMD's MIVisionX library, Not Found ${ColourReset}")
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
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: MIVisionX with OpenCL backend Not Found (HINT: run setup script with the OCL backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    elseif("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        if (NOT MIVISIONX_BACKEND_HIP_FOUND)
            set(BACKEND "CPU")
            message("-- ${Red}WARNING: MIVisionX with the HIP backend Not Found (HINT: run setup script with the HIP backend)${ColourReset}")
            message("-- ${Yellow}NOTE: rocAL Backend set to CPU ${ColourReset}")
        endif()
    endif()
endif()
# Turbo JPEG
if(NOT TurboJpeg_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires TurboJpeg, Not Found ${ColourReset}")
endif()
# Google ProtoBuff
if(NOT Protobuf_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires Protobuf, Not Found ${ColourReset}")
endif()
# OpenMP
if(NOT OpenMP_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires OpenMP, Not Found ${ColourReset}")
endif()
# Threads
if(NOT Threads_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires Threads, Not Found ${ColourReset}")
endif()
# LMDB
if(NOT LMDB_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires LMDB, Not Found ${ColourReset}")	
endif()
# RapidJSON
if(NOT RapidJSON_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires RapidJSON, Not Found ${ColourReset}")	
endif()
# C++17 Filesystem
if(NOT FILESYSTEM_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires FileSystem, Not Found ${ColourReset}")	
endif()
# Half.hpp
if(NOT HALF_FOUND)
    set(BUILD_ROCAL false)
    message("-- ${Yellow}NOTE: rocAL library requires HALF, Not Found ${ColourReset}")	
endif()

if(${BUILD_ROCAL})
    # AMD RPP
    include_directories(${AMDRPP_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${AMDRPP_LIBRARIES})
    # MIVisionX
    include_directories(${MIVisionX_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OPENVX_LIBRARIES} ${VXRPP_LIBRARIES})
    # OpenMP
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenMP_CXX_LIBRARIES})
    message("-- ${White}${PROJECT_NAME}: Using OpenMP -- \n\tOpenMP_CXX_FLAGS:${OpenMP_CXX_FLAGS}\n\tOpenMP_CXX_LIBRARIES:${OpenMP_CXX_LIBRARIES}${ColourReset}")
    # Threads
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)
    # TurboJPEG
    include_directories(${TurboJpeg_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES})
    # Protobuf
    include_directories(${PROTOBUF_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${PROTOBUF_LIBRARIES})
    # LMDB
    include_directories(${LMDB_INCLUDE_DIRS})	
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LMDB_LIBRARIES})	
    # RapidJSON
    include_directories(${RapidJSON_INCLUDE_DIRS})
    # Filesystem
    include_directories(${FILESYSTEM_INCLUDE_DIRS})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FILESYSTEM_LIBRARIES})	
    # half
    include_directories(${HALF_INCLUDE_DIRS})
    #rocDecode
    if ("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND AND ROCDECODE_FOUND)
        include_directories(${ROCDECODE_INCLUDE_DIR} ${ROCM_PATH}/share/rocdecode/utils/rocvideodecode)
        include_directories(${ROCM_PATH}/share/rocdecode/utils)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${ROCDECODE_LIBRARY})
        set(ROCDECODE_SRCS ${ROCM_PATH}/share/rocdecode/utils/rocvideodecode/roc_video_dec.cpp)
    endif()
    # rocJPEG
    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND AND ROCJPEG_FOUND)
        include_directories (${ROCJPEG_INCLUDE_DIR})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${ROCJPEG_LIBRARY})
    endif()

    message("-- ${Green}ROCm Augmentation Library -- rocAL dependencies found${ColourReset}")

    include_directories(
                ${CMAKE_CURRENT_BINARY_DIR}
                include
                include/api
    )

    link_directories(${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR})
    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        link_directories(${HIP_PATH}/${CMAKE_INSTALL_LIBDIR})
        include_directories(${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR} rocAL_hip)
        add_subdirectory(rocAL_hip)
    endif()

    protobuf_generate_cpp(TF_PROTO_SRCS TF_PROTO_HEADERS proto/example.proto proto/feature.proto)
    protobuf_generate_cpp(CAFFE2_PROTO_SRCS CAFFE2_PROTO_HEADERS proto/caffe2_protos.proto)
    protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HEADERS proto/caffe_protos.proto)
    link_directories(${AMDRPP_LIBRARIES_DIRS} ${TurboJpeg_LIBRARIES_DIRS} ${PROTOBUF_LIBRARY_DIRS} /usr/local/lib/)

    file(GLOB_RECURSE SOURCES "source/*.cpp")
    add_library(${PROJECT_NAME} SHARED ${SOURCES} ${TF_PROTO_SRCS} ${TF_PROTO_HEADERS} ${CAFFE_PROTO_HEADERS} 
                                ${CAFFE_PROTO_SRCS} ${CAFFE2_PROTO_SRCS} ${CAFFE2_PROTO_HEADERS} ${ROCDECODE_SRCS})

    if("${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
        add_dependencies(${PROJECT_NAME} rocAL_hip)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} $<TARGET_OBJECTS:rocAL_hip>)
        target_compile_definitions(${PROJECT_NAME} PRIVATE __HIP_PLATFORM_AMD__)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=1)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=0)
        if(ROCDECODE_FOUND)
            if(${ROCDECODE_VERSION_MAJOR} VERSION_GREATER "0" OR
               (${ROCDECODE_VERSION_MAJOR} VERSION_EQUAL "0" AND ${ROCDECODE_VERSION_MINOR} VERSION_GREATER_EQUAL "12" AND ${ROCDECODE_VERSION_PATCH} VERSION_GREATER_EQUAL "0"))
                target_compile_definitions(${PROJECT_NAME} PUBLIC -DENABLE_ROCDECODE)
                message("-- ${White}rocAL built with rocDecode video decode capabilities${ColourReset}")
            else()
                message("-- ${Yellow}rocAL -- Requires rocDecode version >= 0.12.0${ColourReset}")
            endif()
        endif()
        if(ROCJPEG_FOUND)
            if(${ROCJPEG_VERSION_MAJOR} VERSION_GREATER "0" OR
               (${ROCJPEG_VERSION_MAJOR} VERSION_EQUAL "0" AND ${ROCJPEG_VERSION_MINOR} VERSION_GREATER_EQUAL "11" AND ${ROCJPEG_VERSION_PATCH} VERSION_GREATER_EQUAL "0"))
                target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_ROCJPEG=1)
                message("-- ${White}rocAL built with rocJpeg hardware decoder${ColourReset}")
            else()
                target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_ROCJPEG=0)
                message("-- ${Yellow}rocAL -- Requires rocJpeg version >= 0.11.0${ColourReset}")
            endif()
        endif()
        message("-- ${White}rocAL built with HIP Backend${ColourReset}")
    elseif("${BACKEND}" STREQUAL "OPENCL" AND OPENCL_FOUND)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=1)
        message("-- ${White}rocAL built with OpenCL Backend${ColourReset}")
    else()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HIP=0)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCL=0)
        message("-- ${White}rocAL built with CPU only Backend${ColourReset}")
    endif()

    target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_SIMD=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBG_TIMING=1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGINFO=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DBGLOG=0 WRNLOG=0)
    target_compile_definitions(${PROJECT_NAME} PUBLIC TFRECORD_ENABLE=0)
    message("-- ${White}rocAL built with ENABLE_SIMD")
    # OpenCV
    if (OpenCV_FOUND)
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=1)
        include_directories(${OpenCV_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenCV_LIBRARIES})
        if(${OpenCV_VERSION_MAJOR} EQUAL 4)
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
        else()
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
        endif()
    else()
        target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_OPENCV=0)
        message("-- ${Yellow}NOTE: rocAL built without OpenCV extension functionality${ColourReset}")
    endif()

    # FFMPEG
    if(NOT FFMPEG_FOUND)
        message("-- ${Yellow}NOTE: rocAL built without FFmpeg video decode functionality${ColourReset}")
    else()
        # FFMPEG multi-version support
        if(_FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100)
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_58_134=0)
        else()
            target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AVCODEC_GREATER_THAN_58_134=1)
        endif()
        include_directories(${AVUTIL_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FFMPEG_LIBRARIES})
        message("-- ${White}rocAL built with FFmpeg video decode functionality${ColourReset}")
        target_compile_definitions(${PROJECT_NAME} PUBLIC -DROCAL_VIDEO)
    endif()
    # Filesystem
    if(STD_FILESYSTEM_PRESENT)
        message("-- ${White}rocAL built with std Filesystem${ColourReset}")
    elseif(EXPERIMENTAL_FILESYSTEM_PRESENT)
        message("-- ${White}rocAL built with std experimental Filesystem${ColourReset}")
    else()
        message(FATAL_ERROR "No filesystem library found.")
    endif()
    
    # Audio features for rocAL
    if(${VX_EXT_RPP_VERSION_MAJOR} VERSION_GREATER_EQUAL "3" AND ${VX_EXT_RPP_VERSION_MINOR} VERSION_GREATER_EQUAL "0" AND ${VX_EXT_RPP_VERSION_PATCH} VERSION_GREATER "0")
        if(NOT SNDFILE_FOUND)
            message("-- ${Yellow}NOTE: rocAL built without SndFile - Audio functionalities will not be supported${ColourReset}")
            set(AUDIO_SUPPORT FALSE)
        else()
            set(AUDIO_SUPPORT TRUE)
        endif()
    else()
        set(AUDIO_SUPPORT FALSE)
        message("-- ${Yellow}rocAL -- Requires MIVisionX vx_rpp version > 3.0.0 for audio${ColourReset}")
    endif()

    # SndFile
    if(AUDIO_SUPPORT)
        include_directories(${SNDFILE_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${SNDFILE_LIBRARIES})
        message("-- ${White}rocAL built with Audio Functionality${ColourReset}")
        target_compile_definitions(${PROJECT_NAME} PUBLIC -DROCAL_AUDIO)
    else()
        message("-- ${Yellow}NOTE: rocAL built without Audio support - Audio Functionalities will not be enabled${ColourReset}")
    endif()

    # LibTar
    if(LIBTAR_FOUND)
        include_directories(${LIBTAR_INCLUDE_DIRS})
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${LIBTAR_LIBRARIES})
        message("-- ${White}NOTE: rocAL built with LibTar support - WebDataset reader enabled${ColourReset}")
        target_compile_definitions(${PROJECT_NAME} PUBLIC -DENABLE_WDS)
    else()
        message("-- ${Yellow}NOTE: rocAL built without LibTar - WebDataset reader will not be supported${ColourReset}")
    endif()

    # -Wall -- Enable most warning messages
    # -mavx2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation
    # -mfma -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and FMA built-in functions and code generation
    # -mf16c -- Support F16C built-in functions and code generation
    # -Wno-deprecated-declarations -- Do not warn about uses of functions, variables, and types marked as deprecated by using the deprecated attribute
    # -std=gnu++17 -- Conform to the ISO 2017 C++ standard with GNU extensions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx2 -mfma -mf16c -Wno-deprecated-declarations -std=gnu++17")
    message("-- ${White}rocAL -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})
    message("-- ${White}rocAL -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")

    set_target_properties(rocal PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

    # install rocAL libs -- {ROCM_PATH)/lib
    install(TARGETS rocal LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
    install(TARGETS rocal LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
    install(TARGETS rocal LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
    # install rocAL include files -- {ROCM_PATH}/include/rocAL/
    install(FILES
            include/api/rocal_api.h
            include/api/rocal_api_info.h
            include/api/rocal_api_augmentation.h
            include/api/rocal_api_data_loaders.h
            include/api/rocal_api_types.h
            include/api/rocal_api_data_transfer.h
            include/api/rocal_api_parameters.h
            include/api/rocal_api_meta_data.h
            include/api/rocal_api_tensor.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocal COMPONENT dev)
else()
    message(FATAL_ERROR "-- ${Red}rocAL dependencies not satisfied${ColourReset}")
endif()
