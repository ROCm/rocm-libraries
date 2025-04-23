#[[
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- rpp_test: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
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

project(rpp_test)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake)

# Set message options
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(BoldBlue    "${Esc}[1;34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
endif()

find_package(OpenCL QUIET)
find_package(OpenCV QUIET)
# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenMP_CXX_LIBRARIES})
message("-- ${White}${PROJECT_NAME}: Using OpenMP -- \n\tOpenMP_CXX_FLAGS:${OpenMP_CXX_FLAGS}\n\tOpenMP_CXX_LIBRARIES:${OpenMP_CXX_LIBRARIES}${ColourReset}")

if (OpenCL_FOUND AND OpenCV_FOUND)
    message("-- ${Green}${PROJECT_NAME} set to build with rpp, OpenCL and OpenCV${ColourReset}")
    # include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers) # commented since OpenCL_INCLUDE_DIRS points to a deprecated /opt/rocm/opencl/include
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
    link_directories(${ROCM_PATH}/lib /usr/local/lib)

    add_executable(BatchPD_ocl_pkd3 BatchPD_ocl_pkd3.cpp)
    add_executable(BatchPD_ocl_pln1 BatchPD_ocl_pln1.cpp)
    add_executable(BatchPD_ocl_pln3 BatchPD_ocl_pln3.cpp)
    # add_executable(Single_ocl Single_ocl.cpp)
    add_executable(uniqueFunctionalities_ocl uniqueFunctionalities_ocl.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_OPENCL=1 -std=gnu++17")
    target_link_libraries(BatchPD_ocl_pkd3 ${OpenCV_LIBS} -lrpp ${OpenCL_LIBRARIES} ${LINK_LIBRARY_LIST})
    target_link_libraries(BatchPD_ocl_pln1 ${OpenCV_LIBS} -lrpp ${OpenCL_LIBRARIES} ${LINK_LIBRARY_LIST})
    target_link_libraries(BatchPD_ocl_pln3 ${OpenCV_LIBS} -lrpp ${OpenCL_LIBRARIES} ${LINK_LIBRARY_LIST})
    # target_link_libraries(Single_ocl ${OpenCV_LIBS} -lrpp ${OpenCL_LIBRARIES} )
    target_link_libraries(uniqueFunctionalities_ocl ${OpenCV_LIBS} -lrpp ${OpenCL_LIBRARIES} ${LINK_LIBRARY_LIST})
else()
    message("-- ${Yellow}Error: OpenCV and OpenCL must be installed to install ${PROJECT_NAME} successfully!${ColourReset}")
endif()