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


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/include/rocRoller/Utilities")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/Array.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Array_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Comparison.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Component.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Component_Impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Concepts.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/EnumBitset.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/EnumBitset_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Error_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Error.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Error_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Generator.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Generator_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HIPTimer.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HipUtils.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LazySingleton.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Logging.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Random_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Random.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Random_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RTTI.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Settings_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Settings.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Settings_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Timer.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Utils.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Utils_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Version.hpp"
)
