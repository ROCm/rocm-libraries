# Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()
set(VERSION "2.0.0")

project(vx_rpp VERSION ${VERSION} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

find_package(AMDRPP REQUIRED)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL REQUIRED)
elseif(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    SET(OpenCL_FOUND FALSE)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP REQUIRED)
endif()

include_directories(../../amd_openvx/openvx/include/ include)
# AMD RPP
include_directories(${AMDRPP_INCLUDE_DIRS})
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${AMDRPP_LIBRARIES})

list(APPEND SOURCES
        source/image/AbsoluteDifferencebatchPD.cpp
        source/image/AccumulatebatchPD.cpp
        source/image/AccumulateSquaredbatchPD.cpp
        source/image/AccumulateWeightedbatchPD.cpp
        source/image/AddbatchPD.cpp
        source/image/BitwiseANDbatchPD.cpp
        source/image/BitwiseNOTbatchPD.cpp
        source/image/BlendbatchPD.cpp
        source/image/BlurbatchPD.cpp
        source/image/BoxFilterbatchPD.cpp
        source/image/BrightnessbatchPD.cpp
        source/image/CannyEdgeDetector.cpp
        source/image/ChannelCombinebatchPD.cpp
        source/image/ChannelExtractbatchPD.cpp
        source/image/ColorTemperaturebatchPD.cpp
        source/image/ColorTwistbatchPD.cpp
        source/image/ContrastbatchPD.cpp
        source/image/CopybatchPD.cpp
        source/image/CropMirrorNormalizePD.cpp
        source/image/CropPD.cpp
        source/image/CustomConvolutionbatchPD.cpp
        source/image/DataObjectCopybatchPD.cpp
        source/image/DilatebatchPD.cpp
        source/image/ErodebatchPD.cpp
        source/image/ExclusiveORbatchPD.cpp
        source/image/ExposurebatchPD.cpp
        source/image/FastCornerDetector.cpp
        source/image/FisheyebatchPD.cpp
        source/image/FlipbatchPD.cpp
        source/image/FogbatchPD.cpp
        source/image/GammaCorrectionbatchPD.cpp
        source/image/GaussianFilterbatchPD.cpp
        source/image/GaussianImagePyramidbatchPD.cpp
        source/image/HarrisCornerDetector.cpp
        source/image/Histogram.cpp
        source/image/HistogramBalancebatchPD.cpp
        source/image/HistogramEqualizebatchPD.cpp
        source/image/HuebatchPD.cpp
        source/image/InclusiveORbatchPD.cpp
        source/image/JitterbatchPD.cpp
        source/image/LaplacianImagePyramid.cpp
        source/image/LensCorrectionbatchPD.cpp
        source/image/LocalBinaryPatternbatchPD.cpp
        source/image/LookUpTablebatchPD.cpp
        source/image/MagnitudebatchPD.cpp
        source/image/MaxbatchPD.cpp
        source/image/MeanStddev.cpp
        source/image/MedianFilterbatchPD.cpp
        source/image/MinbatchPD.cpp
        source/image/MinMaxLoc.cpp
        source/image/MultiplybatchPD.cpp
        source/image/NoisebatchPD.cpp
        source/image/NonLinearFilterbatchPD.cpp
        source/image/NonMaxSupressionbatchPD.cpp
        source/image/NopbatchPD.cpp
        source/image/PhasebatchPD.cpp
        source/image/PixelatebatchPD.cpp
        source/image/RainbatchPD.cpp
        source/image/RandomCropLetterBoxbatchPD.cpp
        source/image/RandomShadowbatchPD.cpp
        source/image/Remap.cpp
        source/image/ResizebatchPD.cpp
        source/image/ResizeCropbatchPD.cpp
        source/image/ResizeCropMirrorPD.cpp
        source/image/ResizeMirrorNormalizeTensor.cpp
        source/image/Resizetensor.cpp
        source/image/RotatebatchPD.cpp
        source/image/SaturationbatchPD.cpp
        source/image/ScalebatchPD.cpp
        source/image/SequenceRearrangebatchPD.cpp
        source/image/SnowbatchPD.cpp
        source/image/SobelbatchPD.cpp
        source/image/SubtractbatchPD.cpp
        source/image/TensorAdd.cpp
        source/image/TensorLookup.cpp
        source/image/TensorMatrixMultiply.cpp
        source/image/TensorMultiply.cpp
        source/image/TensorSubtract.cpp
        source/image/ThresholdingbatchPD.cpp
        source/image/VignettebatchPD.cpp
        source/image/WarpAffinebatchPD.cpp
        source/image/WarpPerspectivebatchPD.cpp
        source/tensor/Blend.cpp
        source/tensor/Blur.cpp
        source/tensor/Brightness.cpp
        source/tensor/ColorTemperature.cpp
        source/tensor/ColorTwist.cpp
        source/tensor/Contrast.cpp
        source/tensor/Copy.cpp
        source/tensor/Crop.cpp
        source/tensor/CropMirrorNormalize.cpp
        source/tensor/Downmix.cpp
        source/tensor/Exposure.cpp
        source/tensor/FishEye.cpp
        source/tensor/Flip.cpp
        source/tensor/Fog.cpp
        source/tensor/GammaCorrection.cpp
        source/tensor/Glitch.cpp
        source/tensor/Hue.cpp
        source/tensor/Jitter.cpp
        source/tensor/LensCorrection.cpp
        source/tensor/MelFilterBank.cpp
        source/tensor/Noise.cpp
        source/tensor/NonSilentRegionDetection.cpp
        source/tensor/Nop.cpp
        source/tensor/Normalize.cpp
        source/tensor/Rotate.cpp
        source/tensor/Pixelate.cpp
        source/tensor/PreemphasisFilter.cpp
        source/tensor/Rain.cpp
        source/tensor/Resample.cpp
        source/tensor/Resize.cpp
        source/tensor/ResizeCrop.cpp
        source/tensor/ResizeCropMirror.cpp
        source/tensor/ResizeMirrorNormalize.cpp
        source/tensor/Saturation.cpp
        source/tensor/SequenceRearrange.cpp
        source/tensor/Snow.cpp
        source/tensor/Spectrogram.cpp
        source/tensor/TensorAddTensor.cpp
        source/tensor/TensorMulScalar.cpp
        source/tensor/ToDecibels.cpp
        source/tensor/Vignette.cpp
        source/tensor/WarpAffine.cpp
        source/tensor/Slice.cpp
        source/tensor/Transpose.cpp
        source/kernel_rpp.cpp
        source/internal_publishKernels.cpp
    )

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL"  AND OpenCL_FOUND)
    message("-- ${Green}vx_rpp -- Building with OpenCL${ColourReset}")
    set(ENABLE_OPENCL 1)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_OPENCL=1)
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenCL_LIBRARIES})
    add_library(vx_rpp SHARED ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_OPENCL=1") # Currently used by rpp. To be removed.
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
    message("-- ${Green}vx_rpp -- Building with HIP${ColourReset}")
    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 1)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -DRPP_BACKEND_HIP=1 -D__HIP_PLATFORM_AMD__)
    include_directories(${ROCM_PATH}/include)
    link_directories(${HIP_PATH}/lib)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
    add_library(vx_rpp SHARED ${SOURCES})
    set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # TBD -- CHECK to FIX
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_HIP=1") # Currently used by rpp. To be removed.
else()
    message("-- ${Green}vx_rpp -- Building with CPU backend${ColourReset}")
    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP})
    add_library(vx_rpp SHARED ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=0" ) # Currently used by rpp. To be removed.
endif()

set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND 0)
find_path(RPP_INCLUDE_DIR NAMES rpp.h PATHS ${ROCM_PATH}/include/rpp)
if(EXISTS ${RPP_INCLUDE_DIR}/rpp_audio_augmentations_support.h)
    file(READ ${RPP_INCLUDE_DIR}/rpp_audio_augmentations_support.h RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE)
    string(REGEX MATCH "RPP_AUDIO_AUGMENTATIONS_SUPPORT ([0-9]*)" _ ${RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE})
    set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND ${CMAKE_MATCH_1})
endif()

# Audio features for VX_RPP
if((${RPP_VERSION_MAJOR} VERSION_GREATER "1") OR (${RPP_VERSION_MAJOR} VERSION_EQUAL "1" AND ${RPP_VERSION_MINOR} VERSION_GREATER_EQUAL "8" AND ${RPP_VERSION_PATCH} VERSION_GREATER_EQUAL "0")
AND RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUDIO_SUPPORT=1")
    message("-- ${White}vx_rpp Audio Features Included${ColourReset}")
    target_compile_definitions(vx_rpp PUBLIC RPP_AUDIO=1)
else()
    target_compile_definitions(vx_rpp PUBLIC RPP_AUDIO=0)
    message("-- ${Yellow}vx_rpp Audio Features Excluded${ColourReset}")
endif()


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenVX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} openvx)
    # -mavx2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX and AVX2 built-in functions and code generation.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

target_link_libraries(vx_rpp ${LINK_LIBRARY_LIST})
message("-- ${White}vx_rpp -- Link Libraries:${LINK_LIBRARY_LIST}${ColourReset}")
message("-- ${White}vx_rpp -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_ext_rpp.h include/vx_ext_rpp_version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/vx_rpp_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)
