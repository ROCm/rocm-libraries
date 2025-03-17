/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

#include <rocRoller/KernelGraph/Transforms/AddComputeIndex.hpp>
#include <rocRoller/KernelGraph/Transforms/AddConvert.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDeallocate.hpp>
#include <rocRoller/KernelGraph/Transforms/AddDirect2LDS.hpp>
#include <rocRoller/KernelGraph/Transforms/AddF6LDSPadding.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDS.hpp>
#include <rocRoller/KernelGraph/Transforms/AddPRNG.hpp>
#include <rocRoller/KernelGraph/Transforms/AddStreamK.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanArguments.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanLoops.hpp>
#include <rocRoller/KernelGraph/Transforms/ConnectWorkgroups.hpp>
#include <rocRoller/KernelGraph/Transforms/ConstantPropagation.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseExpressions.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops.hpp>
#include <rocRoller/KernelGraph/Transforms/IdentifyParallelDimensions.hpp>
#include <rocRoller/KernelGraph/Transforms/InlineIncrements.hpp>
#include <rocRoller/KernelGraph/Transforms/LoadPacked.hpp>
#include <rocRoller/KernelGraph/Transforms/LoopOverTileNumbers.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerLinear.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTensorContraction.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderEpilogueBlocks.hpp>
#include <rocRoller/KernelGraph/Transforms/OrderMemory.hpp>
#include <rocRoller/KernelGraph/Transforms/RemoveDuplicates.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>
#include <rocRoller/KernelGraph/Transforms/UnrollLoops.hpp>
#include <rocRoller/KernelGraph/Transforms/UpdateParameters.hpp>
