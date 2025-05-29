// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "tree_node_bluestein.h"
#include "../../shared/arithmetic.h"
#include "function_pool.h"
#include "kernel_launch.h"
#include "node_factory.h"
#include <numeric>

size_t BluesteinNode::FindBlue(const function_pool& pool,
                               size_t               len,
                               rocfft_precision     precision,
                               bool                 forcePow2)
{
    if(forcePow2)
    {
        size_t p = 1;
        while(p < len)
            p <<= 1;
        return 2 * p;
    }

    size_t lenPow2 = 1;
    while(lenPow2 < len)
        lenPow2 <<= 1;

    size_t minLenBlue  = 2 * len - 1;
    size_t length      = minLenBlue;
    size_t lenPow2Blue = 2 * lenPow2;

    // We don't want to choose a non-pow2 length that is too close to lenPow2Blue,
    // otherwise using a non-pow2 length may end up being slower than using lenPow2Blue.
    // This ratio has been experimentally verified to yield a non-pow2 length that is at
    // least as fast as its corresponding pow2 length.
    double lenCutOffRatio = .9;

    for(length = minLenBlue; length < lenPow2Blue; ++length)
    {
        auto lenSupported = NodeFactory::NonPow2LengthSupported(pool, precision, length);
        auto lenRatio     = static_cast<double>(length) / static_cast<double>(lenPow2Blue);

        if(lenSupported && lenRatio < lenCutOffRatio)
            break;
    }

    return length;
}

BluesteinType BluesteinNode::DecideBlueType()
{
    bool useSingleKernel = BluesteinSingleNode::SizeFits(pool, length[0], precision);

    // single kernel sticks to pow2 lengthBlue.  the kernel does many
    // other things besides FFTs, so keep radices simple to reduce
    // VGPR usage.
    lengthBlue = FindBlue(pool, length[0], precision, useSingleKernel);

    if(useSingleKernel)
        return BluesteinType::BT_SINGLE_KERNEL;

    NodeMetaData bluePlanData(this);
    bluePlanData.length.push_back(lengthBlue);
    bluePlanData.direction = direction;
    bluePlanData.batch     = batch;

    auto scheme = NodeFactory::DecideNodeScheme(pool, bluePlanData, this);

    if(scheme == CS_L1D_CC)
    {
        // Allow fused Bluestein optimization only for 1D
        // complex forward and complex inverse transforms.
        auto fusedBluesteinAllow = (parent) ? false : true;

        auto type = fusedBluesteinAllow ? BluesteinType::BT_MULTI_KERNEL_FUSED
                                        : BluesteinType::BT_MULTI_KERNEL;

        return type;
    }

    // Handle lengthBlue with its own non-fused FFT sub-plan.  This
    // can be a multi-kernel L1D plan, or a 1-kernel Stockham plan if
    // lengthBlue can be done with a single non-pow2 kernel.
    if(scheme == CS_L1D_CRT || scheme == CS_L1D_TRTRT || scheme == CS_KERNEL_STOCKHAM)
        return BluesteinType::BT_MULTI_KERNEL;

    return BluesteinType::BT_NONE;
}

/*****************************************************
 * CS_BLUESTEIN
 *****************************************************/
void BluesteinNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    // Build a node for a 1D stage using the Bluestein algorithm for
    // general transform lengths.
    typeBlue = DecideBlueType();

    switch(typeBlue)
    {
    case BT_SINGLE_KERNEL:
    {
        // single kernel requires a single lengthBlue FFT on the second
        // half of chirp buffer before we do the rest of the Bluestein
        // steps that kernel

        typeBlue = BluesteinType::BT_SINGLE_KERNEL;

        auto chirpPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CHIRP, this);
        chirpPlan->dimension = 1;
        chirpPlan->length.push_back(length[0]);
        chirpPlan->lengthBlue = lengthBlue;
        chirpPlan->direction  = direction;
        chirpPlan->batch      = 1;
        chirpPlan->large1D    = 2 * length[0];

        NodeMetaData chirpFFTPlanData(this);
        chirpFFTPlanData.dimension = 1;
        chirpFFTPlanData.length.push_back(lengthBlue);
        chirpFFTPlanData.batch   = 1;
        chirpFFTPlanData.iOffset = lengthBlue;
        chirpFFTPlanData.oOffset = lengthBlue;
        auto chirpFFTPlan        = NodeFactory::CreateExplicitNode(chirpFFTPlanData, this);
        chirpFFTPlan->RecursiveBuildTree();

        auto singlePlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_BLUESTEIN_SINGLE, this);
        singlePlan->dimension = 1;
        singlePlan->length    = length;
        singlePlan->lengthBlue = lengthBlue;

        childNodes.emplace_back(std::move(chirpPlan));
        childNodes.emplace_back(std::move(chirpFFTPlan));
        childNodes.emplace_back(std::move(singlePlan));

        break;
    }
    case BT_MULTI_KERNEL_FUSED:
    {
        typeBlue = BluesteinType::BT_MULTI_KERNEL_FUSED;

        // first node: fused chirp + padding + forward fft
        NodeMetaData fftFwdChirpPadPlanData(this);
        fftFwdChirpPadPlanData.dimension = 1;
        fftFwdChirpPadPlanData.length.push_back(lengthBlue);
        fftFwdChirpPadPlanData.batch = 1;
        auto fftFwdChirpPadPlan = NodeFactory::CreateExplicitNode(fftFwdChirpPadPlanData, this);
        fftFwdChirpPadPlan->direction   = direction;
        fftFwdChirpPadPlan->typeBlue    = typeBlue;
        fftFwdChirpPadPlan->fuseBlue    = BluesteinFuseType::BFT_FWD_CHIRP;
        fftFwdChirpPadPlan->lengthBlue  = lengthBlue;
        fftFwdChirpPadPlan->lengthBlueN = length[0];
        fftFwdChirpPadPlan->comments.push_back("Fused chirp + padding w/ fwd FFT");
        fftFwdChirpPadPlan->RecursiveBuildTree();
        for(auto& child : fftFwdChirpPadPlan->childNodes)
            child->comments.push_back("Stockham kernel fused with Bluestein ops");

        // second node: fused chirp / input Hadamard product + padding + forward fft
        NodeMetaData fftFwdChirpMulPadPlanData(this);
        fftFwdChirpMulPadPlanData.dimension = 1;
        fftFwdChirpMulPadPlanData.length.push_back(lengthBlue);
        for(size_t index = 1; index < length.size(); index++)
        {
            fftFwdChirpMulPadPlanData.length.push_back(length[index]);
        }
        auto fftFwdChirpMulPadPlan
            = NodeFactory::CreateExplicitNode(fftFwdChirpMulPadPlanData, this);
        fftFwdChirpMulPadPlan->direction   = direction;
        fftFwdChirpMulPadPlan->lengthBlue  = lengthBlue;
        fftFwdChirpMulPadPlan->lengthBlueN = length[0];
        fftFwdChirpMulPadPlan->typeBlue    = typeBlue;
        fftFwdChirpMulPadPlan->fuseBlue    = BluesteinFuseType::BFT_FWD_CHIRP_MUL;
        fftFwdChirpMulPadPlan->comments.push_back(
            "Fused chirp/input Hadamard prod + padding w/ fwd FFT");
        fftFwdChirpMulPadPlan->RecursiveBuildTree();
        for(auto& child : fftFwdChirpMulPadPlan->childNodes)
            child->comments.push_back("Stockham kernel fused with Bluestein ops");

        // third node: fused convolution Hadamard product + inverse fft + chirp Hadamard product
        NodeMetaData fftInvMulChirpMulPlanData(this);
        fftInvMulChirpMulPlanData.dimension = 1;
        fftInvMulChirpMulPlanData.direction = -direction;
        fftInvMulChirpMulPlanData.length.push_back(lengthBlue);
        for(size_t index = 1; index < length.size(); index++)
        {
            fftInvMulChirpMulPlanData.length.push_back(length[index]);
        }
        auto fftInvMulChirpMulPlan
            = NodeFactory::CreateExplicitNode(fftInvMulChirpMulPlanData, this);
        fftInvMulChirpMulPlan->lengthBlue  = lengthBlue;
        fftInvMulChirpMulPlan->lengthBlueN = length[0];
        fftInvMulChirpMulPlan->typeBlue    = typeBlue;
        fftInvMulChirpMulPlan->fuseBlue    = BluesteinFuseType::BFT_INV_CHIRP_MUL;
        fftInvMulChirpMulPlan->comments.push_back(
            "Fused convolution input Hadamard prod + chirp/output Hadamard prod w/ inv FFT");
        fftInvMulChirpMulPlan->RecursiveBuildTree();
        for(auto& child : fftInvMulChirpMulPlan->childNodes)
            child->comments.push_back("Stockham kernel fused with Bluestein ops");

        childNodes.emplace_back(std::move(fftFwdChirpPadPlan));
        childNodes.emplace_back(std::move(fftFwdChirpMulPadPlan));
        childNodes.emplace_back(std::move(fftInvMulChirpMulPlan));
        break;
    }
    case BT_MULTI_KERNEL:
    {
        auto chirpPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CHIRP, this);
        chirpPlan->dimension = 1;
        chirpPlan->length.push_back(length[0]);
        chirpPlan->lengthBlue = lengthBlue;
        chirpPlan->direction  = direction;
        chirpPlan->batch      = 1;
        chirpPlan->large1D    = 2 * length[0];

        auto padmulPlan        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_PAD_MUL, this);
        padmulPlan->dimension  = 1;
        padmulPlan->length     = length;
        padmulPlan->lengthBlue = lengthBlue;

        NodeMetaData ffticPlanData(this);
        ffticPlanData.dimension = 1;
        ffticPlanData.length.push_back(lengthBlue);
        ffticPlanData.batch *= product(length.begin() + 1, length.end());
        ffticPlanData.batch++;
        ffticPlanData.iOffset = lengthBlue;
        ffticPlanData.oOffset = lengthBlue;
        auto ffticPlan        = NodeFactory::CreateExplicitNode(ffticPlanData, this);
        // FFT nodes must be in-place - were FFTing the second half
        // of chirp as well as the padded user data (via iOffset,
        // oOffset), so if the result goes to a different temp buffer
        // we lose the offset information.
        ffticPlan->allowOutofplace = false;
        ffticPlan->RecursiveBuildTree();

        auto fftmulPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_FFT_MUL, this);
        fftmulPlan->dimension = 1;
        fftmulPlan->length.push_back(lengthBlue);
        for(size_t index = 1; index < length.size(); index++)
        {
            fftmulPlan->length.push_back(length[index]);
        }
        fftmulPlan->lengthBlue = lengthBlue;

        NodeMetaData fftrPlanData(this);
        fftrPlanData.dimension = 1;
        fftrPlanData.length.push_back(lengthBlue);
        for(size_t index = 1; index < length.size(); index++)
        {
            fftrPlanData.length.push_back(length[index]);
        }
        fftrPlanData.direction    = -direction;
        fftrPlanData.iOffset      = 2 * lengthBlue;
        fftrPlanData.oOffset      = 2 * lengthBlue;
        auto fftrPlan             = NodeFactory::CreateExplicitNode(fftrPlanData, this);
        fftrPlan->allowOutofplace = false;
        fftrPlan->RecursiveBuildTree();

        auto resmulPlan        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_RES_MUL, this);
        resmulPlan->dimension  = 1;
        resmulPlan->length     = length;
        resmulPlan->lengthBlue = lengthBlue;

        childNodes.emplace_back(std::move(chirpPlan));
        childNodes.emplace_back(std::move(padmulPlan));
        childNodes.emplace_back(std::move(ffticPlan));
        childNodes.emplace_back(std::move(fftmulPlan));
        childNodes.emplace_back(std::move(fftrPlan));
        childNodes.emplace_back(std::move(resmulPlan));

        break;
    }
    case BT_NONE:
        throw std::runtime_error("Invalid Bluestein type");
    }
}

void BluesteinNode::AssignParams_internal()
{
    switch(typeBlue)
    {
    case BT_SINGLE_KERNEL:
    {
        auto& chirpPlan    = childNodes[0];
        auto& chirpFFTPlan = childNodes[1];
        auto& singlePlan   = childNodes[2];

        chirpPlan->inStride.push_back(1);
        chirpPlan->iDist = chirpPlan->lengthBlue;
        chirpPlan->outStride.push_back(1);
        chirpPlan->oDist = chirpPlan->lengthBlue;

        chirpFFTPlan->inStride  = chirpPlan->outStride;
        chirpFFTPlan->iDist     = chirpPlan->oDist;
        chirpFFTPlan->outStride = chirpFFTPlan->inStride;
        chirpFFTPlan->oDist     = chirpFFTPlan->iDist;
        chirpFFTPlan->AssignParams();

        singlePlan->inStride  = inStride;
        singlePlan->iDist     = iDist;
        singlePlan->outStride = outStride;
        singlePlan->oDist     = oDist;
        singlePlan->AssignParams();

        break;
    }
    case BT_MULTI_KERNEL_FUSED:
    {
        auto& fftFwdChirpPadPlan    = childNodes[0];
        auto& fftFwdChirpMulPadPlan = childNodes[1];
        auto& fftInvMulChirpMulPlan = childNodes[2];

        fftFwdChirpPadPlan->inStride.push_back(1);
        fftFwdChirpPadPlan->inStrideBlue.push_back(1);
        fftFwdChirpPadPlan->iDist     = fftFwdChirpPadPlan->lengthBlueN;
        fftFwdChirpPadPlan->iDistBlue = fftFwdChirpPadPlan->lengthBlue;
        fftFwdChirpPadPlan->outStride.push_back(1);
        fftFwdChirpPadPlan->outStrideBlue.push_back(1);
        fftFwdChirpPadPlan->oDist     = fftFwdChirpPadPlan->lengthBlueN;
        fftFwdChirpPadPlan->oDistBlue = fftFwdChirpPadPlan->lengthBlue;
        fftFwdChirpPadPlan->AssignParams();

        fftFwdChirpMulPadPlan->inStride      = inStride;
        fftFwdChirpMulPadPlan->inStrideBlue  = inStrideBlue;
        fftFwdChirpMulPadPlan->iDist         = iDist;
        fftFwdChirpMulPadPlan->iDistBlue     = iDistBlue;
        fftFwdChirpMulPadPlan->outStride     = outStride;
        fftFwdChirpMulPadPlan->outStrideBlue = outStrideBlue;
        fftFwdChirpMulPadPlan->oDist         = oDist;
        fftFwdChirpMulPadPlan->oDistBlue     = oDistBlue;
        fftFwdChirpMulPadPlan->AssignParams();

        fftInvMulChirpMulPlan->inStride      = inStride;
        fftInvMulChirpMulPlan->inStrideBlue  = inStrideBlue;
        fftInvMulChirpMulPlan->iDist         = iDist;
        fftInvMulChirpMulPlan->iDistBlue     = iDistBlue;
        fftInvMulChirpMulPlan->outStride     = outStride;
        fftInvMulChirpMulPlan->outStrideBlue = outStrideBlue;
        fftInvMulChirpMulPlan->oDist         = oDist;
        fftInvMulChirpMulPlan->oDistBlue     = oDistBlue;
        fftInvMulChirpMulPlan->AssignParams();

        break;
    }
    case BT_MULTI_KERNEL:
    {
        auto& chirpPlan  = childNodes[0];
        auto& padmulPlan = childNodes[1];
        auto& ffticPlan  = childNodes[2];
        auto& fftmulPlan = childNodes[3];
        auto& fftrPlan   = childNodes[4];
        auto& resmulPlan = childNodes[5];

        chirpPlan->inStride.push_back(1);
        chirpPlan->iDist = chirpPlan->lengthBlue;
        chirpPlan->outStride.push_back(1);
        chirpPlan->oDist = chirpPlan->lengthBlue;

        padmulPlan->inStride = inStride;
        padmulPlan->iDist    = iDist;

        padmulPlan->outStride.push_back(1);
        padmulPlan->oDist = padmulPlan->lengthBlue;
        for(size_t index = 1; index < length.size(); index++)
        {
            padmulPlan->outStride.push_back(padmulPlan->oDist);
            padmulPlan->oDist *= length[index];
        }

        ffticPlan->inStride  = chirpPlan->outStride;
        ffticPlan->iDist     = chirpPlan->oDist;
        ffticPlan->outStride = ffticPlan->inStride;
        ffticPlan->oDist     = ffticPlan->iDist;

        ffticPlan->AssignParams();

        fftmulPlan->inStride  = padmulPlan->outStride;
        fftmulPlan->iDist     = padmulPlan->oDist;
        fftmulPlan->outStride = fftmulPlan->inStride;
        fftmulPlan->oDist     = fftmulPlan->iDist;

        fftrPlan->inStride  = fftmulPlan->outStride;
        fftrPlan->iDist     = fftmulPlan->oDist;
        fftrPlan->outStride = fftrPlan->inStride;
        fftrPlan->oDist     = fftrPlan->iDist;

        fftrPlan->AssignParams();

        resmulPlan->inStride  = fftrPlan->outStride;
        resmulPlan->iDist     = fftrPlan->oDist;
        resmulPlan->outStride = outStride;
        resmulPlan->oDist     = oDist;

        break;
    }
    case BT_NONE:
        throw std::runtime_error("Invalid Bluestein type");
    }
}

BluesteinSingleNode::BluesteinSingleNode(TreeNode* p, ComputeScheme s)
    : LeafNode(p, s)
{
    need_twd_table = true;
}

bool BluesteinSingleNode::SizeFits(const function_pool& pool,
                                   size_t               length,
                                   rocfft_precision     precision)
{
    // 2N - 1 must fit into a single kernel, and single-kernel
    // Bluestein only uses pow2 FFT
    return 2 * length - 1 < pool.get_largest_pow2_length(precision);
}

size_t BluesteinSingleNode::GetTwiddleTableLength()
{
    // FFT part of bluestein needs twiddles
    return lengthBlue;
}

void BluesteinSingleNode::GetKernelFactors()
{
    // HACK: for single-kernel bluestein, avoid radix-16 as it uses a
    // lot of VGPRs.  these kernels already do a lot of other stuff
    // besides FFTs, so we need to keep VGPR usage down to get enough
    // occupancy.  fortunately, single-kernel bluestein is always
    // using pow2 <= 4096, and only at length 2048 do we start to
    // want radix-16 anyway.
    if(lengthBlue == 2048)
        kernelFactors = {8, 8, 8, 4};
    else if(lengthBlue == 4096)
        kernelFactors = {8, 8, 8, 8};
    else
        kernelFactors = pool.get_kernel(FMKey(lengthBlue, precision, CS_KERNEL_STOCKHAM)).factors;
}
