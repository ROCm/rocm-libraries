################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED TENSILE_ROOT)
    get_filename_component(TENSILE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
endif()
if(NOT DEFINED TENSILE_SCRIPT_ROOT)
    set(TENSILE_SCRIPT_ROOT "${TENSILE_ROOT}/Tensile")
endif()

set(tensile_sources  ${tensile_sources}
    source/AMDGPU.cpp
    source/ArithmeticUnitTypes.cpp
    source/ContractionProblem.cpp
    source/ContractionSolution.cpp
    source/DataTypes.cpp
    source/Debug.cpp
    source/EmbeddedLibrary.cpp
    source/KernelArguments.cpp
    source/KernelLanguageTypes.cpp
    source/MLFeatures.cpp
    source/PerformanceMetricTypes.cpp
    source/ScalarValueTypes.cpp
    source/TensorDescriptor.cpp
    source/TensorOps.cpp
    source/Tensile.cpp
    source/UserDrivenTuningParser.cpp
    source/Utils.cpp
    )

if(TENSILE_USE_LLVM)
    find_package(LLVM REQUIRED CONFIG)

    set(tensile_sources ${tensile_sources}
        source/llvm/YAML.cpp
        source/llvm/Loading.cpp
    )
endif()

if(TENSILE_USE_MSGPACK)
    set(tensile_sources ${tensile_sources}
        source/msgpack/MessagePack.cpp
    )
endif()

if(TENSILE_USE_HIP)
    set(tensile_sources ${tensile_sources}
        source/hip/HipSolutionAdapter.cpp
        source/hip/HipHardware.cpp
        )
endif()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library (TensileHost STATIC ${tensile_sources})

set_target_properties(TensileHost
                      PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED ON
                      CXX_EXTENSIONS OFF)

target_include_directories(TensileHost INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(TENSILE_USE_LLVM OR TENSILE_USE_MSGPACK)
    target_compile_definitions(TensileHost PUBLIC TENSILE_DEFAULT_SERIALIZATION)
endif()

if(TENSILE_USE_MSGPACK)
    # See: https://github.com/msgpack/msgpack-c/wiki/Q%26A#how-to-support-both-msgpack-c-c-version-5x-and-6x-
    # Prefer 6.x (msgpack-cxx) as that is what we bundle in the build.
    find_package(msgpack-cxx CONFIG)
    if(msgpack-cxx_FOUND)
        message(STATUS "Found msgpack-cxx (>=6.x)")
        set(msgpack_target msgpack-cxx)
    else()
        find_package(msgpackc-cxx CONFIG REQUIRED NAMES msgpackc-cxx msgpack)
        message(STATUS "Found msgpack (<=5.x)")
        set(msgpack_target msgpackc)
    endif()
    get_target_property(msgpack_inc ${msgpack_target} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(TensileHost PRIVATE "${msgpack_inc}")
    target_compile_definitions(TensileHost PRIVATE TENSILE_MSGPACK=1)
endif()

if(TENSILE_USE_LLVM)
    target_compile_definitions(TensileHost PUBLIC -DTENSILE_YAML=1)
    find_library(LLVMObjectYAML_LIBRARY
        NAMES LLVMObjectYAML
        PATHS ${LLVM_LIBRARY_DIR})
    message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")
    #Use LLVM shared libs if LLVMObjectYAML static lib cannot be found
    if(LLVMObjectYAML_LIBRARY)
        target_link_libraries(TensileHost PRIVATE LLVMObjectYAML )
    else()
        target_link_libraries(TensileHost PRIVATE LLVM )
    endif()
    target_include_directories(TensileHost PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_STATIC_ONLY)
    target_compile_definitions(TensileHost PUBLIC TENSILE_STATIC_ONLY)
endif()

if(TENSILE_USE_HIP)
    target_compile_definitions(TensileHost PUBLIC TENSILE_USE_HIP)

    #add_subdirectory(client)
    target_link_libraries( TensileHost PUBLIC ${HSA_LIBRARIES} hip::device )

    if(WIN32)
        target_compile_options( TensileHost PUBLIC -Wno-deprecated-declarations -Wno-ignored-attributes -Wdll-attribute-on-redeclaration -fdelayed-template-parsing )
        target_link_libraries( TensileHost PUBLIC Shlwapi )
    endif()
endif()


if(false)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif()
