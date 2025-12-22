###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 ###############################################################################

# Find / configure LLVM
find_package(LLVM CONFIG HINTS ${ROCM_DIR}/../../..)

if (NOT LLVM_DIR)
    message(FATAL_ERROR "HipTensor requires LLVM version 7.0 or higher,"
        " which cannot be found. Please install rocm-llvm-dev package first." )
endif()

# Required:
# LLVMObjectYAML for YAML parsing
# LLVMSupport for CL processing
find_library(LLVMObjectYAML_LIBRARY NAMES LLVMObjectYAML PATHS ${LLVM_LIBRARY_DIR})
find_library(LLVMSupport_LIBRARY NAMES LLVMSupport PATHS ${LLVM_LIBRARY_DIR})

message(STATUS "adding hiptensor component: hiptensor_llvm")
message(STATUS "LLVM_LIBRARY_DIR: ${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM_VERSION_MAJOR: ${LLVM_VERSION_MAJOR}")
message(STATUS "LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")
message(STATUS "LLVMSupport_LIBRARY: ${LLVMSupport_LIBRARY}")

if(LLVM_VERSION_MAJOR VERSION_LESS "7")
    message(FATAL_ERROR "The found LLVM version doesn't meet the minimum requirement.  Please "
    "install version 7.0 or above.  It is recommend to install rocm-llvm-dev "
    "(for ROCm versions higher than 6.0).  If you have installed multiple "
    "versions of LLVM, you can specify which version to use by setting "
    "LLVM_ROOT.  For example:\n"
    "        -DLLVM_ROOT=/opt/rocm/llvm-7/lib")
endif()

if(NOT LLVMObjectYAML_LIBRARY)
    message(ERROR "Could not find library: LLVMObjectYAML" )
endif()
if(NOT LLVMSupport_LIBRARY)
    message(ERROR "Could not find library: LLVMSupport")
endif()

# LLVM Libs we need to link against
set(HIPTENSOR_LLVM_LIBS "-L${LLVM_LIBRARY_DIR}"
                        "-fno-rtlib-add-rpath"
                        LLVMObjectYAML
                        LLVMSupport
                        hip::device
                        )

# Includes
set(HIPTENSOR_LLVM_INCLUDES ${LLVM_INCLUDE_DIRS}
                            ${PROJECT_SOURCE_DIR}/library/include
                            ${PROJECT_SOURCE_DIR}/library/src/include
                            ${PROJECT_SOURCE_DIR}/library/src
                            ${PROJECT_SOURCE_DIR}/test
                            )

# Sources for this static object
set(HIPTENSOR_LLVM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/yaml_parser_config.cpp
                           ${CMAKE_CURRENT_SOURCE_DIR}/command_line_parser.cpp)

# Create hiptensor_llvm object target
# All LLVM resources are private to this object.
# Interface will fwd includes of this dir to users of the object.
set(LIBRARY_NAME "hiptensor_llvm")
add_library(${LIBRARY_NAME} OBJECT ${HIPTENSOR_LLVM_SOURCES})
set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${LIBRARY_NAME} PRIVATE ${HIPTENSOR_LLVM_LIBS})
target_include_directories(${LIBRARY_NAME} PRIVATE ${HIPTENSOR_LLVM_INCLUDES})
target_include_directories(${LIBRARY_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(${LIBRARY_NAME} PRIVATE -fno-rtti ${CLANG_DRIVER_MODE})
target_link_options(${LIBRARY_NAME} PRIVATE ${CLANG_DRIVER_MODE})
