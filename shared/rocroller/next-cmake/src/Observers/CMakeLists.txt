################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the Software), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/source/Observers")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/ACCVGPRReadWrite.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ACCVGPRWriteWrite.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BufferStoreDwordXXRead.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CMPXWriteExec.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DGEMM16x16x4Write.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DGEMM4x4x4Write.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DLWrite.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MEMObserver.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MFMAObserver.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ObserverCreation.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OPSEL94x.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RegisterLivenessObserver.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SupportedInstructionObserver.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUTransWrite94x.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWrite.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteReadlane94x.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteSGPRVCC.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteSGPRVMEM.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteVCCVDIVFMAS.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VCMPXWrite94x.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WaitcntObserver.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WMMAObserver.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WMMAReadSrcD.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WMMAWrite.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WMMAWriteSrcD.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/XDLReadSrcC908.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/XDLReadSrcC90a.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/XDLReadSrcC94x.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/XDLWrite908.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/XDLWrite90a.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/XDLWrite94x.cpp"
)
