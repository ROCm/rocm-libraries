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


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/include/rocRoller/KernelGraph/Transforms")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddComputeIndex.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddConvert.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddDeallocate.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddDirect2LDS.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddF6LDSPadding.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddLDS.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddPrefetch.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddPRNG.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddStreamK.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AliasDataFlowTags_detail.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AliasDataFlowTags.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/All.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CleanArguments.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CleanLoops.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ConnectWorkgroups_detail.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ConnectWorkgroups.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ConstantPropagation.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/FuseExpressions.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/FuseLoops.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GraphTransform_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GraphTransform.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/IdentifyParallelDimensions.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/InlineIncrements.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LoadPacked_detail.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LoadPacked.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LoopOverTileNumbers.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LowerLinear.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LowerTensorContraction.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LowerTile.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OrderEpilogueBlocks.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OrderMemory.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OrderMultiplyNodes.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/PrefetchScale.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RemoveDuplicates.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Simplify.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SwizzleScale.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UnrollLoops.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UpdateParameters.hpp"
)
