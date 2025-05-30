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


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/include/rocRoller/Scheduling/Costs")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/Cost_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Cost.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Cost_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LinearWeightedCost_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LinearWeightedCost.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MinNopsCost_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MinNopsCost.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/NoneCost_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/NoneCost.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UniformCost_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UniformCost.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WaitCntNopCost_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/WaitCntNopCost.hpp"
)
