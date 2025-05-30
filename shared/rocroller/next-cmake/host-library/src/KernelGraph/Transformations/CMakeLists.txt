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


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/source/KernelGraph/Transformations")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddComputeIndex.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddConvert.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddDeallocate.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddDirect2LDS.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddF6LDSPadding.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddLDS.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddPrefetch.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddPRNG.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AddStreamK.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AliasDataFlowTags.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CleanArguments.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CleanLoops.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ConnectWorkgroups.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ConstantPropagation.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/FuseExpressions.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/FuseLoops.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/IdentifyParallelDimensions.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/InlineIncrements.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelGraph_loop.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LoadPacked.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LoopOverTileNumbers.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LowerLinear.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LowerTensorContraction.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LowerTile.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OrderEpilogueBlocks.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OrderMemory.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/OrderMultiplyNodes.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/PrefetchScale.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RemoveDuplicates.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Simplify.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SwizzleScale.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UnrollLoops.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/UpdateParameters.cpp"
)
