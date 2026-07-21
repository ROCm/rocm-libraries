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

if(TENSILE_USE_LLVM)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLite.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteMixed.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteNavi.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsTileLite.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/SampleTensileKernels.yaml"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/rocBLAS_Full.yaml")
endif()

if(TENSILE_USE_MSGPACK)
    set(SOLUTION_LIBRARY_FILES_
        ${SOLUTION_LIBRARY_FILES_}
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLite.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteMixed.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsLiteNavi.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/KernelsTileLite.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/SampleTensileKernels.dat"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionLibraries/rocBLAS_Full.dat")
endif()

set(SOLUTION_LIBRARY_FILES
    ${SOLUTION_LIBRARY_FILES_}
    PARENT_SCOPE)
