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


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/include/rocRoller/Scheduling/Observers/WaitState")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/BufferStoreDwordXXRead.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OPSEL94x.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUTransWrite94x.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteReadlane94x.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteSGPRVCC.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteSGPRVMEM.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VALUWriteVCCVDIVFMAS.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/VCMPXWrite94x.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WaitStateObserver.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WaitStateObserver_impl.hpp"
)

add_subdirectory(WaitState)
add_subdirectory(WMMA)
add_subdirectory(MFMA)
