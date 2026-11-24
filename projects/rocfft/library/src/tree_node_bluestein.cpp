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
#include "data_set.h"
#include "function_pool.h"
#include "kernel_launch.h"
#include "node_factory.h"

namespace
{
    data_set_t bluestein_1d_layout(
        size_t length, size_t batch, size_t distance, rocfft_precision precision, size_t offset = 0)
    {
        return data_set_t::full_range({length}, {1}, batch, distance)
            .with_element_type(rocfft_array_type_unset, precision, {offset, 0});
    }

    std::vector<size_t> packed_bluestein_strides(const data_set_t& layout, size_t padded_length)
    {
        std::vector<size_t> strides(layout.get_full_rank());
        strides[0] = 1;
        strides[1] = padded_length;
        for(size_t axis = 2; axis < strides.size(); ++axis)
            strides[axis] = strides[axis - 1] * layout.axis_span(axis - 1);
        return strides;
    }

    void set_stage_strides(TreeNode&               stage,
                           std::span<const size_t> input,
                           std::span<const size_t> output)
    {
        stage.set_strides(io_data_label::INPUT, input);
        stage.set_strides(io_data_label::OUTPUT, output);
    }
}

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
    const auto   precision       = get_precision();
    const auto   length          = input_length(0);
    const bool   useSingleKernel = BluesteinSingleNode::SizeFits(pool, length, precision);
    const size_t paddedLength    = FindBlue(pool, length, precision, useSingleKernel);

    BluesteinType type = BluesteinType::BT_SINGLE_KERNEL;
    if(!useSingleKernel)
    {
        const auto probe
            = bluestein_1d_layout(paddedLength, batch_count(), paddedLength, precision);
        NodeMetaData bluePlanData(probe, probe, this);

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

    blue.emplace(length, paddedLength, type);
}

/*****************************************************
 * CS_BLUESTEIN
 *****************************************************/
void BluesteinNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    // Build a node for a 1D stage using the Bluestein algorithm for
    // general transform lengths.
    ConstructBlueParams();

    const auto n         = blue->get_transform_length();
    const auto p         = blue->get_padded_length();
    const auto precision = get_precision();
    // Keep array types provisional until buffer assignment, without inheriting user offsets.
    const auto input  = input_layout().with_element_type(rocfft_array_type_unset, precision);
    const auto output = output_layout().with_element_type(rocfft_array_type_unset, precision);

    switch(GetBluesteinType())
    {
    case BT_SINGLE_KERNEL:
    {
        // single kernel requires a single padded-length FFT on the second
        // half of chirp buffer before we do the rest of the Bluestein
        // steps that kernel

        // Keep the transform-length shape; the chirp kernel writes two padded spans.
        const auto chirp   = bluestein_1d_layout(n, 1, p, precision);
        auto chirpPlan     = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CHIRP, this, chirp, chirp);
        chirpPlan->blue    = blue;
        chirpPlan->large1D = 2 * n;

        const auto   chirpFFT = bluestein_1d_layout(p, 1, p, precision, p);
        NodeMetaData chirpFFTPlanData(chirpFFT, chirpFFT, this);
        auto         chirpFFTPlan = NodeFactory::CreateExplicitNode(chirpFFTPlanData, this);
        chirpFFTPlan->RecursiveBuildTree();

        auto singlePlan
            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_BLUESTEIN_SINGLE, this, input, output);
        singlePlan->blue = blue;

        childNodes.emplace_back(std::move(chirpPlan));
        childNodes.emplace_back(std::move(chirpFFTPlan));
        childNodes.emplace_back(std::move(singlePlan));

        break;
    }
    case BT_MULTI_KERNEL_FUSED:
    {
        // first node: fused chirp + padding + forward fft
        const auto   chirp = bluestein_1d_layout(p, 1, n, precision);
        NodeMetaData fftFwdChirpPadPlanData(chirp, chirp, this);
        auto fftFwdChirpPadPlan  = NodeFactory::CreateExplicitNode(fftFwdChirpPadPlanData, this);
        fftFwdChirpPadPlan->blue = blue;
        fftFwdChirpPadPlan->blue->set_fuse_type(BluesteinFuseType::BFT_FWD_CHIRP);
        fftFwdChirpPadPlan->comments.push_back("Fused chirp + padding w/ fwd FFT");
        fftFwdChirpPadPlan->RecursiveBuildTree();
        for(auto& child : fftFwdChirpPadPlan->childNodes)
            child->comments.push_back("Stockham kernel fused with Bluestein ops");

        // second node: fused chirp / input Hadamard product + padding + forward fft
        const auto   paddedInput  = input.with_axis_span(0, p);
        const auto   paddedOutput = output.with_axis_span(0, p);
        NodeMetaData fftFwdChirpMulPadPlanData(paddedInput, paddedOutput, this);
        auto         fftFwdChirpMulPadPlan
            = NodeFactory::CreateExplicitNode(fftFwdChirpMulPadPlanData, this);
        fftFwdChirpMulPadPlan->blue = blue;
        fftFwdChirpMulPadPlan->blue->set_fuse_type(BluesteinFuseType::BFT_FWD_CHIRP_MUL);
        fftFwdChirpMulPadPlan->comments.push_back(
            "Fused chirp/input Hadamard prod + padding w/ fwd FFT");
        fftFwdChirpMulPadPlan->RecursiveBuildTree();
        for(auto& child : fftFwdChirpMulPadPlan->childNodes)
            child->comments.push_back("Stockham kernel fused with Bluestein ops");

        // third node: fused convolution Hadamard product + inverse fft + chirp Hadamard product
        // Keep the padded FFT shape; the final fused kernel stores only the transform length.
        NodeMetaData fftInvMulChirpMulPlanData(paddedInput, paddedOutput, this);
        fftInvMulChirpMulPlanData.direction = -direction;
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
        // The chirp kernel writes padded storage but has an unpadded logical span.
        const auto chirp   = bluestein_1d_layout(n, 1, p, precision);
        auto chirpPlan     = NodeFactory::CreateNodeFromScheme(CS_KERNEL_CHIRP, this, chirp, chirp);
        chirpPlan->blue    = blue;
        chirpPlan->large1D = 2 * n;

        auto       convolution    = input.with_axis_span(0, p);
        const auto packed_strides = packed_bluestein_strides(convolution, p);
        convolution.set_strides_and_distances(packed_strides);

        // PadMul writes padded output but keeps an unpadded shape for its kernel ABI.
        auto padmulOutput = output;
        padmulOutput.set_strides_and_distances(packed_strides);
        auto padmulPlan
            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_PAD_MUL, this, input, padmulOutput);
        padmulPlan->blue = blue;

        // Fold all the batch axes, then add the FFT of the chirp itself.
        size_t convolution_batch = 1;
        for(size_t axis = 1; axis < convolution.get_full_rank(); ++axis)
            convolution_batch *= convolution.axis_span(axis);
        const auto   fftic = bluestein_1d_layout(p, 1 + convolution_batch, p, precision, p);
        NodeMetaData ffticPlanData(fftic, fftic, this);
        auto         ffticPlan = NodeFactory::CreateExplicitNode(ffticPlanData, this);
        // FFT nodes must be in-place - were FFTing the second half
        // of chirp as well as the padded user data (via iOffset,
        // oOffset), so if the result goes to a different temp buffer
        // we lose the offset information.
        ffticPlan->allowOutofplace = false;
        ffticPlan->RecursiveBuildTree();

        auto fftmulPlan
            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_FFT_MUL, this, convolution, convolution);
        fftmulPlan->blue = blue;

        const auto fftr
            = convolution.with_element_type(rocfft_array_type_unset, precision, {2 * p, 0});
        NodeMetaData fftrPlanData(fftr, fftr, this);
        fftrPlanData.direction    = -direction;
        auto fftrPlan             = NodeFactory::CreateExplicitNode(fftrPlanData, this);
        fftrPlan->allowOutofplace = false;
        fftrPlan->RecursiveBuildTree();

        auto resmulInput = input;
        resmulInput.set_strides_and_distances(packed_strides);
        auto resmulPlan
            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_RES_MUL, this, resmulInput, output);
        resmulPlan->blue = blue;

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
    const auto p = blue->get_padded_length();
    switch(GetBluesteinType())
    {
    case BT_SINGLE_KERNEL:
    {
        auto& chirpPlan    = childNodes[0];
        auto& chirpFFTPlan = childNodes[1];
        auto& singlePlan   = childNodes[2];

        const std::array<size_t, 2> chirp_strides{1, p};
        set_stage_strides(*chirpPlan, chirp_strides, chirp_strides);
        set_stage_strides(*chirpFFTPlan, chirp_strides, chirp_strides);
        chirpFFTPlan->AssignParams();

        set_stage_strides(*singlePlan,
                          input_layout().strides_and_distances(),
                          output_layout().strides_and_distances());
        singlePlan->AssignParams();

        break;
    }
    case BT_MULTI_KERNEL_FUSED:
    {
        auto& fftFwdChirpPadPlan    = childNodes[0];
        auto& fftFwdChirpMulPadPlan = childNodes[1];
        auto& fftInvMulChirpMulPlan = childNodes[2];

        const std::array<size_t, 2> chirp_strides{1, blue->get_transform_length()};
        set_stage_strides(*fftFwdChirpPadPlan, chirp_strides, chirp_strides);
        fftFwdChirpPadPlan->AssignParams();

        const auto input_strides  = input_layout().strides_and_distances();
        const auto output_strides = output_layout().strides_and_distances();
        set_stage_strides(*fftFwdChirpMulPadPlan, input_strides, output_strides);
        fftFwdChirpMulPadPlan->AssignParams();

        set_stage_strides(*fftInvMulChirpMulPlan, input_strides, output_strides);
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

        const std::array<size_t, 2> chirp_strides{1, p};
        set_stage_strides(*chirpPlan, chirp_strides, chirp_strides);

        const auto packed_strides = packed_bluestein_strides(input_layout(), p);
        set_stage_strides(*padmulPlan, input_layout().strides_and_distances(), packed_strides);

        set_stage_strides(*ffticPlan, chirp_strides, chirp_strides);
        ffticPlan->AssignParams();

        set_stage_strides(*fftmulPlan, packed_strides, packed_strides);
        set_stage_strides(*fftrPlan, packed_strides, packed_strides);
        fftrPlan->AssignParams();

        set_stage_strides(*resmulPlan,
                          fftrPlan->output_layout().strides_and_distances(),
                          output_layout().strides_and_distances());

        break;
    }
    case BT_NONE:
        throw std::runtime_error("Invalid Bluestein type");
    }
}

BluesteinSingleNode::BluesteinSingleNode(TreeNode* p, ComputeScheme s, NodeLayouts&& layouts)
    : LeafNode(p, s, std::move(layouts))
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
            = pool.get_kernel(FMKey(blue->get_padded_length(), get_precision(), CS_KERNEL_STOCKHAM))
                  .factors;
}
