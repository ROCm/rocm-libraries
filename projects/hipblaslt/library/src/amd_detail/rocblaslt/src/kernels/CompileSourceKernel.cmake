################################################################################
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
function(CompileSourceKernel source archs buildIdKind outputFolder)
    message("Setup source kernel targets")
    string(REGEX MATCHALL "gfx[a-z0-9]+" archs "${archs}")
    list(REMOVE_DUPLICATES archs)
    list(JOIN archs "," archs)
    message("archs for source kernel compilation: ${archs}")
    add_custom_target(MatrixTransformKernels ALL
                      DEPENDS ${outputFolder}/hipblasltTransform.hsaco
                      VERBATIM)
    add_custom_command(OUTPUT ${outputFolder}/hipblasltTransform.hsaco
                       COMMAND bash  ${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/src/kernels/compile_code_object.sh ${source} ${archs} ${CMAKE_BUILD_TYPE} ${buildIdKind} ${outputFolder}/hipblasltTransform.hsaco
                       COMMENT "Compiling source kernels")
endfunction()