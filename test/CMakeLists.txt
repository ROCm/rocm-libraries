# ##############################################################################
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
#
# ##############################################################################
cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(White       "${Esc}[37m")
endif()

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

project(rocdecode-test)

# make test with CTest
enable_testing()
include(CTest)

# add find modules
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rocdecode/cmake)

# find rocDecode
find_package(rocDecode QUIET)

if(NOT ROCDECODE_FOUND)
    message("-- ${Yellow}${PROJECT_NAME} requires rocDecode. Install rocDecode before running CTests")
endif()

# 1 - videoDecode HEVC
add_test(
  NAME
    video_decode-HEVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
)

# 2 - videoDecode AVC
add_test(
  NAME
    video_decode-AVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H264.mp4
)

# 3 - videoDecode AV1
add_test(
  NAME
    video_decode-AV1
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-AV1.mp4
)

# 4 - videoDecode VP9
add_test(
  NAME
    video_decode-VP9
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecode"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-VP9.ivf
)

# 5 - videoDecodePerf
add_test(
  NAME
    video_decodePerf-HEVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodePerf"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodePerf"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodeperf"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
)

# 6 - videoDecodeBatch
add_test(
  NAME
    video_decodeBatch
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeBatch"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeBatch"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodebatch"
            -i ${ROCM_PATH}/share/rocdecode/video/ -t 2
)

# 7 - videoDecodeRGB
add_test(
  NAME
    video_decodeRGB-HEVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRGB"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRGB"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodergb"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4 -of rgb
)


# 8 - videoDecodeMem
add_test(
  NAME
    video_decodeMem-HEVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeMem"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeMem"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecodemem"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
)

# 9 - videoDecodeRaw HEVC
add_test(
  NAME
    video_decodeRaw-HEVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H265.265
)

# 10 - videoDecodeRaw AVC
add_test(
  NAME
    video_decodeRaw-AVC
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-H264.264
)

# 11 - videoDecodeRaw AV1
add_test(
  NAME
    video_decodeRaw-AV1
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-AV1.ivf
)

# 12 - videoDecodeRaw VP9
add_test(
  NAME
    video_decodeRaw-VP9
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocdecode/samples/videoDecodeRaw"
                              "${CMAKE_CURRENT_BINARY_DIR}/videoDecodeRaw"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "videodecoderaw"
            -i ${ROCM_PATH}/share/rocdecode/video/AMD_driving_virtual_20-VP9.ivf
)
