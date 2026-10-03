// Copyright (C) 2021 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

void BluesteinNode::ConstructBlueParams()
{
    // single kernel sticks to a pow2 padded length.  the kernel does many other
    // things besides FFTs, so keep radices simple to reduce VGPR usage.
    const bool   useSingleKernel = BluesteinSingleNode::SizeFits(pool, length[0], precision);
    const size_t paddedLength    = FindBlue(pool, length[0], precision, useSingleKernel);

    BluesteinType type = BluesteinType::BT_SINGLE_KERNEL;
    if(!useSingleKernel)
    {
        NodeMetaData bluePlanData(this);
        bluePlanData.length.push_back(paddedLength);
        bluePlanData.direction = direction;
        bluePlanData.batch     = batch;

        auto scheme = NodeFactory::DecideNodeScheme(pool, bluePlanData, this);

        if(scheme == CS_L1D_CC)
            // Fused Bluestein only for a top-level (parentless) 1D
            // complex transform.
            type = parent ? BluesteinType::BT_MULTI_KERNEL : BluesteinType::BT_MULTI_KERNEL_FUSED;
        else if(scheme == CS_L1D_CRT || scheme == CS_L1D_TRTRT || scheme == CS_KERNEL_STOCKHAM)
            // padded length handled by its own non-fused FFT sub-plan
            type = BluesteinType::BT_MULTI_KERNEL;
        else
            type = BluesteinType::BT_NONE;
    }

    blue.emplace(length[0], paddedLength, type);
}

/*****************************************************
 * CS_BLUESTEIN
 *****************************************************/
void BluesteinNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    // Build a node for a 1D stage using the Bluestein algorithm for
    // general transform lengths.
    ConstructBlueParams();

    switch(GetBluesteinType())
    {
    case BT_SINGLE_KERNEL:
    {
        // single kernel requires a single padded-length FFT on the second
        // half of chirp buffer before we do the rest of the Bluestein
        // steps that kernel

        auto chirpPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CHIRP, this);
        chirpPlan->dimension = 1;
        chirpPlan->length.push_back(length[0]);
        chirpPlan->blue      = blue;
        chirpPlan->direction = direction;
        chirpPlan->batch     = 1;
        chirpPlan->large1D   = 2 * length[0];

        NodeMetaData chirpFFTPlanData(this);
        chirpFFTPlanData.dimension = 1;
        chirpFFTPlanData.length.push_back(blue->get_padded_length());
        chirpFFTPlanData.batch   = 1;
        chirpFFTPlanData.iOffset = blue->get_padded_length();
        chirpFFTPlanData.oOffset = blue->get_padded_length();
        auto chirpFFTPlan        = NodeFactory::CreateExplicitNode(chirpFFTPlanData, this);
        chirpFFTPlan->RecursiveBuildTree();

        auto singlePlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_BLUESTEIN_SINGLE, this);
        singlePlan->dimension = 1;
        singlePlan->length    = length;
        singlePlan->blue      = blue;

        childNodes.emplace_back(std::move(chirpPlan));
        childNodes.emplace_back(std::move(chirpFFTPlan));
        childNodes.emplace_back(std::move(singlePlan));

        break;
    }
    case BT_MULTI_KERNEL_FUSED:
    {
        // first node: fused chirp + padding + forward fft
        NodeMetaData fftFwdChirpPadPlanData(this);
        fftFwdChirpPadPlanData.dimension = 1;
        fftFwdChirpPadPlanData.length.push_back(blue->get_padded_length());
        fftFwdChirpPadPlanData.batch = 1;
        auto fftFwdChirpPadPlan = NodeFactory::CreateExplicitNode(fftFwdChirpPadPlanData, this);
        fftFwdChirpPadPlan->direction = direction;
        fftFwdChirpPadPlan->blue      = blue;
        fftFwdChirpPadPlan->blue->set_fuse_type(BluesteinFuseType::BFT_FWD_CHIRP);
        fftFwdChirpPadPlan->comments.push_back("Fused chirp + padding w/ fwd FFT");
        fftFwdChirpPadPlan->RecursiveBuildTree();
        for(auto& child : fftFwdChirpPadPlan->childNodes)
            child->comments.push_back("Stockham kernel fused with Bluestein ops");

        // second node: fused chirp / input Hadamard product + padding + forward fft
        NodeMetaData fftFwdChirpMulPadPlanData(this);
        fftFwdChirpMulPadPlanData.dimension = 1;
        fftFwdChirpMulPadPlanData.length.push_back(blue->get_padded_length());
        for(size_t index = 1; index < length.size(); index++)
        {
            fftFwdChirpMulPadPlanData.length.push_back(length[index]);
        }
        auto fftFwdChirpMulPadPlan
            = NodeFactory::CreateExplicitNode(fftFwdChirpMulPadPlanData, this);
        fftFwdChirpMulPadPlan->direction = direction;
        fftFwdChirpMulPadPlan->blue      = blue;
        fftFwdChirpMulPadPlan->blue->set_fuse_type(BluesteinFuseType::BFT_FWD_CHIRP_MUL);
        fftFwdChirpMulPadPlan->comments.push_back(
            "Fused chirp/input Hadamard prod + padding w/ fwd FFT");
        fftFwdChirpMulPadPlan->RecursiveBuildTree();
        for(auto& child : fftFwdChirpMulPadPlan->childNodes)
            child->comments.push_back("Stockham kernel fused with Bluestein ops");

        // third node: fused convolution Hadamard product + inverse fft + chirp Hadamard product
        NodeMetaData fftInvMulChirpMulPlanData(this);
        fftInvMulChirpMulPlanData.dimension = 1;
        fftInvMulChirpMulPlanData.direction = -direction;
        fftInvMulChirpMulPlanData.length.push_back(blue->get_padded_length());
        for(size_t index = 1; index < length.size(); index++)
        {
            fftInvMulChirpMulPlanData.length.push_back(length[index]);
        }
        auto fftInvMulChirpMulPlan
            = NodeFactory::CreateExplicitNode(fftInvMulChirpMulPlanData, this);
        fftInvMulChirpMulPlan->blue = blue;
        fftInvMulChirpMulPlan->blue->set_fuse_type(BluesteinFuseType::BFT_INV_CHIRP_MUL);
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
        chirpPlan->blue      = blue;
        chirpPlan->direction = direction;
        chirpPlan->batch     = 1;
        chirpPlan->large1D   = 2 * length[0];

        auto padmulPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_PAD_MUL, this);
        padmulPlan->dimension = 1;
        padmulPlan->length    = length;
        padmulPlan->blue      = blue;

        NodeMetaData ffticPlanData(this);
        ffticPlanData.dimension = 1;
        ffticPlanData.length.push_back(blue->get_padded_length());
        ffticPlanData.batch *= product(length.begin() + 1, length.end());
        ffticPlanData.batch++;
        ffticPlanData.iOffset = blue->get_padded_length();
        ffticPlanData.oOffset = blue->get_padded_length();
        auto ffticPlan        = NodeFactory::CreateExplicitNode(ffticPlanData, this);
        // FFT nodes must be in-place - were FFTing the second half
        // of chirp as well as the padded user data (via iOffset,
        // oOffset), so if the result goes to a different temp buffer
        // we lose the offset information.
        ffticPlan->allowOutofplace = false;
        ffticPlan->RecursiveBuildTree();

        auto fftmulPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_FFT_MUL, this);
        fftmulPlan->dimension = 1;
        fftmulPlan->length.push_back(blue->get_padded_length());
        for(size_t index = 1; index < length.size(); index++)
        {
            fftmulPlan->length.push_back(length[index]);
        }
        fftmulPlan->blue = blue;

        NodeMetaData fftrPlanData(this);
        fftrPlanData.dimension = 1;
        fftrPlanData.length.push_back(blue->get_padded_length());
        for(size_t index = 1; index < length.size(); index++)
        {
            fftrPlanData.length.push_back(length[index]);
        }
        fftrPlanData.direction    = -direction;
        fftrPlanData.iOffset      = 2 * blue->get_padded_length();
        fftrPlanData.oOffset      = 2 * blue->get_padded_length();
        auto fftrPlan             = NodeFactory::CreateExplicitNode(fftrPlanData, this);
        fftrPlan->allowOutofplace = false;
        fftrPlan->RecursiveBuildTree();

        auto resmulPlan       = NodeFactory::CreateNodeFromScheme(CS_KERNEL_RES_MUL, this);
        resmulPlan->dimension = 1;
        resmulPlan->length    = length;
        resmulPlan->blue      = blue;

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
    switch(GetBluesteinType())
    {
    case BT_SINGLE_KERNEL:
    {
        auto& chirpPlan    = childNodes[0];
        auto& chirpFFTPlan = childNodes[1];
        auto& singlePlan   = childNodes[2];

        chirpPlan->inStride.push_back(1);
        chirpPlan->iDist = chirpPlan->blue->get_padded_length();
        chirpPlan->outStride.push_back(1);
        chirpPlan->oDist = chirpPlan->blue->get_padded_length();

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
        fftFwdChirpPadPlan->iDist = fftFwdChirpPadPlan->blue->get_transform_length();
        fftFwdChirpPadPlan->outStride.push_back(1);
        fftFwdChirpPadPlan->oDist = fftFwdChirpPadPlan->blue->get_transform_length();
        fftFwdChirpPadPlan->AssignParams();

        fftFwdChirpMulPadPlan->inStride  = inStride;
        fftFwdChirpMulPadPlan->iDist     = iDist;
        fftFwdChirpMulPadPlan->outStride = outStride;
        fftFwdChirpMulPadPlan->oDist     = oDist;
        fftFwdChirpMulPadPlan->AssignParams();

        fftInvMulChirpMulPlan->inStride  = inStride;
        fftInvMulChirpMulPlan->iDist     = iDist;
        fftInvMulChirpMulPlan->outStride = outStride;
        fftInvMulChirpMulPlan->oDist     = oDist;
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
        chirpPlan->iDist = chirpPlan->blue->get_padded_length();
        chirpPlan->outStride.push_back(1);
        chirpPlan->oDist = chirpPlan->blue->get_padded_length();

        padmulPlan->inStride = inStride;
        padmulPlan->iDist    = iDist;

        padmulPlan->outStride.push_back(1);
        padmulPlan->oDist = padmulPlan->blue->get_padded_length();
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
    return blue->get_padded_length();
}

void BluesteinSingleNode::GetKernelFactors()
{
    // HACK: for single-kernel bluestein, avoid radix-16 as it uses a
    // lot of VGPRs.  these kernels already do a lot of other stuff
    // besides FFTs, so we need to keep VGPR usage down to get enough
    // occupancy.  fortunately, single-kernel bluestein is always
    // using pow2 <= 4096, and only at length 2048 do we start to
    // want radix-16 anyway.
    if(blue->get_padded_length() == 2048)
        kernelFactors = {8, 8, 8, 4};
    else if(blue->get_padded_length() == 4096)
        kernelFactors = {8, 8, 8, 8};
    else
        kernelFactors
            = pool.get_kernel(FMKey(blue->get_padded_length(), precision, CS_KERNEL_STOCKHAM))
                  .factors;
}
