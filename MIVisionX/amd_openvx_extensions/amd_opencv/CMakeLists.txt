# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
set(VERSION "1.0.0")

project(amd_opencv VERSION ${VERSION} LANGUAGES CXX)

find_package(OpenCV QUIET)

include_directories(../../amd_openvx/openvx/include
					include
				)

list(APPEND SOURCES
	source/internal_dataTranslator.cpp
	source/internal_publishKernels.cpp
	source/internal_vxNodes.cpp
	source/OpenCV_absDiff.cpp
	source/OpenCV_adaptiveThreshold.cpp
	source/OpenCV_add.cpp
	source/OpenCV_addWeighted.cpp
	source/OpenCV_bilateralFilter.cpp
	source/OpenCV_bitwiseAnd.cpp
	source/OpenCV_bitwiseNot.cpp
	source/OpenCV_bitwiseOr.cpp
	source/OpenCV_bitwiseXor.cpp
	source/OpenCV_blur.cpp
	source/OpenCV_boxFilter.cpp
	source/OpenCV_briskCompute.cpp
	source/OpenCV_briskDetector.cpp
	source/OpenCV_buildOpticalFlowPyramid.cpp
	source/OpenCV_buildPyramid.cpp
	source/OpenCV_canny.cpp
	source/OpenCV_compare.cpp
	source/OpenCV_convertScaleAbs.cpp
	source/OpenCV_cornerHarris.cpp
	source/OpenCV_cornerMinEigenVal.cpp
	source/OpenCV_countNonZero.cpp
	source/OpenCV_cvtColor.cpp
	source/OpenCV_dilate.cpp
	source/OpenCV_distanceTransform.cpp
	source/OpenCV_divide.cpp
	source/OpenCV_erode.cpp
	source/OpenCV_fastFeatureDetector.cpp
	source/OpenCV_fastNlMeansDenoising.cpp
	source/OpenCV_fastNlMeansDenoisingColored.cpp
	source/OpenCV_filter2D.cpp
	source/OpenCV_flip.cpp
	source/OpenCV_gaussianBlur.cpp
	source/OpenCV_goodFeaturesToTrackdetector.cpp
	source/OpenCV_integral.cpp
	source/OpenCV_laplacian.cpp
	source/OpenCV_medianBlur.cpp
	source/OpenCV_morphologyEx.cpp
	source/OpenCV_mserFeatureDetector.cpp
	source/OpenCV_multiply.cpp
	source/OpenCV_norm.cpp
	source/OpenCV_orbCompute.cpp
	source/OpenCV_orbDetector.cpp
	source/OpenCV_pyrDown.cpp
	source/OpenCV_pyrUp.cpp
	source/OpenCV_resize.cpp
	source/OpenCV_scharr.cpp
	source/OpenCV_sepFilter2D.cpp
	source/OpenCV_siftCompute.cpp
	source/OpenCV_siftDetect.cpp
	source/OpenCV_simpleBlobDetector.cpp
	source/OpenCV_sobel.cpp
	source/OpenCV_starFeatureDetector.cpp
	source/OpenCV_subtract.cpp
	source/OpenCV_surfCompute.cpp
	source/OpenCV_surfDetect.cpp
	source/OpenCV_threshold.cpp
	source/OpenCV_transpose.cpp
	source/OpenCV_warpAffine.cpp
	source/OpenCV_warpPerspective.cpp
	)

add_library(vx_opencv SHARED ${SOURCES})

if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
    message("-- ${White}OpenCV Version Supported -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X${ColourReset}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(vx_opencv ${OpenCV_LIBRARIES})
    if(${OpenCV_VERSION_MAJOR} EQUAL 4)
		target_compile_definitions(vx_opencv PUBLIC USE_OPENCV_4=1)
    else()
		target_compile_definitions(vx_opencv PUBLIC USE_OPENCV_4=0)
    endif()
else()
    message(FATAL_ERROR "OpenCV -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
endif()

target_link_libraries(vx_opencv openvx)
target_compile_definitions(vx_opencv PUBLIC USE_OPENCV_CONTRIB=0)
set_target_properties(vx_opencv PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS vx_opencv LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS vx_opencv LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS vx_opencv LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_ext_opencv.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/amd_opencv_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)

if(WIN32)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	target_link_libraries(vx_opencv openvx)
endif()

message("-- ${White}vx_opencv -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
