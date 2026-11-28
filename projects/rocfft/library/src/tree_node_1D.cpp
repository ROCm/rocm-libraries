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

#include "tree_node_1D.h"
#include "../../shared/arithmetic.h"
#include "../../shared/precision_type.h"
#include "function_pool.h"
#include "fuse_shim.h"
#include "node_factory.h"
#include "repo.h"
#include "tuning_helper.h"
#include <array>
#include <numeric>

namespace
{
    constexpr auto INPUT  = io_data_label::INPUT;
    constexpr auto OUTPUT = io_data_label::OUTPUT;

    void swap_factor_axes(data_set_t& layout)
    {
        layout.reorder_axes(std::array<size_t, 2>{1, 0}, axis_scope::length_only);
    }

    data_set_t split_factors(const data_set_t& layout, size_t inner_length)
    {
        // The child geometry comes from the stage; buffer assignment selects its type.
        return layout.split_length_axis(0, inner_length)
            .with_element_type(rocfft_array_type_unset, layout.get_precision());
    }

    template <typename SetStride>
    void visit_packed_factor_strides(const data_set_t& layout, bool swapped, SetStride set_stride)
    {
        const size_t inner = swapped ? 1 : 0;
        const size_t outer = 1 - inner;
        set_stride(inner, 1);
        size_t stride = layout.axis_span(inner);
        set_stride(outer, stride);
        stride *= layout.axis_span(outer);
        const size_t last = layout.get_full_rank() - 1;
        for(size_t axis = 2; axis < last; ++axis)
        {
            set_stride(axis, stride);
            stride *= layout.axis_span(axis);
        }
        set_stride(last, stride);
    }

    data_set_t packed_factors(data_set_t layout, bool swapped = false)
    {
        visit_packed_factor_strides(layout, swapped, [&](size_t axis, size_t stride) {
            layout.set_axis_stride(axis, stride);
        });
        return layout;
    }

    void set_packed_factor_strides(TreeNode& node, io_data_label io, bool swapped)
    {
        visit_packed_factor_strides(node.layout_for(io), swapped, [&](size_t axis, size_t stride) {
            node.set_axis_stride(io, axis, stride);
        });
    }

    void set_split_factor_strides(TreeNode&         node,
                                  io_data_label     io,
                                  const data_set_t& source,
                                  bool              swapped)
    {
        const auto&  target = node.layout_for(io);
        const size_t inner  = swapped ? 1 : 0;
        const size_t outer  = 1 - inner;
        const size_t rank   = source.get_full_rank();
        const size_t extra  = target.get_full_rank() - rank;
        assert(extra == 1 || extra == 2);
        const size_t inner_stride = source.axis_stride(0);
        const size_t outer_stride = inner_stride * target.axis_span(inner);
        node.set_axis_stride(io, inner, inner_stride);
        node.set_axis_stride(io, outer, outer_stride);
        // A nested CC/CRT under TRTRT splits the parent's first axis into three digits.
        if(extra == 2)
            node.set_axis_stride(io, 2, outer_stride * target.axis_span(outer));
        for(size_t axis = 1; axis < rank; ++axis)
            node.set_axis_stride(io, axis + extra, source.axis_stride(axis));
    }

    void set_swapped_factor_strides(TreeNode& node, io_data_label io, const data_set_t& source)
    {
        const size_t rank = source.get_full_rank();
        assert(node.layout_for(io).get_full_rank() == rank);
        node.set_axis_stride(io, 0, source.axis_stride(1));
        node.set_axis_stride(io, 1, source.axis_stride(0));
        for(size_t axis = 2; axis < rank; ++axis)
            node.set_axis_stride(io, axis, source.axis_stride(axis));
    }

    data_set_t bluestein_workspace(const data_set_t& packed_stage, bool swap_factors = false)
    {
        // The Bluestein workspace ABI treats every axis except the outer batch as a
        // length axis, including the demoted FFT factor and any inherited batch axes.
        auto spans   = packed_stage.lengths_and_batches();
        auto strides = packed_stage.strides_and_distances();
        if(swap_factors)
        {
            std::swap(spans[0], spans[1]);
            strides[1] = spans[0];
        }
        return data_set_t(std::vector<size_t>(spans.size(), 0), spans, strides);
    }
}

/*****************************************************
 * L1D_TRTRT  *
 *****************************************************/
void TRTRT1DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    ComputeScheme determined_scheme_node1 = CS_NONE;
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 5)
           || (child_scheme_trees[0]->curScheme != CS_KERNEL_TRANSPOSE)
           || (child_scheme_trees[2]->curScheme != CS_KERNEL_TRANSPOSE)
           || (child_scheme_trees[3]->curScheme != CS_KERNEL_STOCKHAM)
           || (child_scheme_trees[4]->curScheme != CS_KERNEL_TRANSPOSE))
        {
            throw std::runtime_error("TRTRT1DNode: Unexpected child scheme from solution map");
        }
        determined_scheme_node1 = child_scheme_trees[1]->curScheme;
    }

    if(length_factor == 0 || input_length(0) % length_factor != 0)
        throw std::runtime_error("L1D_TRTRT wrong factorization");

    auto adInput               = split_factors(input_layout(), input_length(0) / length_factor);
    auto adAfterFirstTranspose = packed_factors(adInput, true);
    auto daStage               = adAfterFirstTranspose;
    swap_factor_axes(daStage);

    auto adStage                = packed_factors(adInput);
    auto daAfterSecondTranspose = adStage;
    swap_factor_axes(daAfterSecondTranspose);

    auto adOutput = split_factors(output_layout(), length_factor);
    swap_factor_axes(adOutput);

    // [A,D] transpose -> [D,A] row -> [D,A] transpose -> [A,D] row and transpose.
    auto trans1Plan = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_TRANSPOSE, this, std::move(adInput), std::move(adAfterFirstTranspose));

    const auto   row1Stage = daStage.get_data_set_for_len_axes({0});
    NodeMetaData row1PlanData(row1Stage, row1Stage, this);

    auto row1Plan = NodeFactory::CreateExplicitNode(row1PlanData, this, determined_scheme_node1);
    row1Plan->large1D = 0;
    row1Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());

    auto trans2Plan = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_TRANSPOSE, this, std::move(daStage), std::move(daAfterSecondTranspose));
    trans2Plan->large1D = input_length(0);

    const auto row2Stage = adStage.get_data_set_for_len_axes({0});
    auto       row2Plan
        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM, this, row2Stage, row2Stage);

    auto trans3Plan = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_TRANSPOSE, this, std::move(adStage), std::move(adOutput));

    // --------------------------------
    // Fuse Shims
    // --------------------------------
    auto TR
        = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM, {trans1Plan.get(), row1Plan.get()});
    if(TR->IsSchemeFusable())
        fuseShims.emplace_back(std::move(TR));

    auto RT
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {row2Plan.get(), trans3Plan.get()});
    if(RT->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT));

    // --------------------------------
    // Push to child nodes : TRTRT
    // --------------------------------
    childNodes.emplace_back(std::move(trans1Plan));
    childNodes.emplace_back(std::move(row1Plan));
    childNodes.emplace_back(std::move(trans2Plan));
    childNodes.emplace_back(std::move(row2Plan));
    childNodes.emplace_back(std::move(trans3Plan));
}

void TRTRT1DNode::AssignParams_internal()
{
    auto& trans1Plan = childNodes[0];
    auto& row1Plan   = childNodes[1];
    auto& trans2Plan = childNodes[2];
    auto& row2Plan   = childNodes[3];
    auto& trans3Plan = childNodes[4];

    set_split_factor_strides(*trans1Plan, INPUT, input_layout(), false);

    // Top-level TRTRT and its nested TRTRT rows can reuse the parent's output strides.
    // Inside other multidimensional plans, non-temp intermediate buffers assume packing.
    const bool reuse_output_strides = parent == nullptr || parent->scheme == CS_L1D_TRTRT;
    if(trans1Plan->obOut == OB_TEMP || !reuse_output_strides)
        set_packed_factor_strides(*trans1Plan, OUTPUT, true);
    else
        set_split_factor_strides(*trans1Plan, OUTPUT, output_layout(), true);

    set_swapped_factor_strides(*row1Plan, INPUT, trans1Plan->output_layout());
    if(row1Plan->placement == rocfft_placement_inplace)
        row1Plan->set_strides(OUTPUT, row1Plan->input_layout());
    else
        set_split_factor_strides(*row1Plan, OUTPUT, output_layout(), false);
    row1Plan->AssignParams();

    trans2Plan->set_strides(INPUT, row1Plan->output_layout());
    if(trans2Plan->obOut == OB_TEMP || !reuse_output_strides)
        set_packed_factor_strides(*trans2Plan, OUTPUT, true);
    else
        set_split_factor_strides(*trans2Plan, OUTPUT, output_layout(), true);

    set_swapped_factor_strides(*row2Plan, INPUT, trans2Plan->output_layout());
    if(row2Plan->obIn == row2Plan->obOut)
        row2Plan->set_strides(OUTPUT, row2Plan->input_layout());
    else if(row2Plan->obOut == OB_TEMP || !reuse_output_strides)
        set_packed_factor_strides(*row2Plan, OUTPUT, false);
    else
        set_split_factor_strides(*row2Plan, OUTPUT, output_layout(), false);

    trans3Plan->set_strides(INPUT, row2Plan->output_layout());
    set_split_factor_strides(*trans3Plan, OUTPUT, output_layout(), true);
}

/*****************************************************
 * L1D_CC  *
 *****************************************************/
void CC1DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 2)
           || (child_scheme_trees[0]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CC)
           || (child_scheme_trees[1]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_RC))
        {
            throw std::runtime_error("CC1DNode: Unexpected child scheme from solution map");
        }
    }

    if(length_factor == 0 || input_length(0) % length_factor != 0)
        throw std::runtime_error("L1D_CC wrong factorization");

    auto       daInput = split_factors(input_layout(), input_length(0) / length_factor);
    const auto adStage = packed_factors(daInput);
    swap_factor_axes(daInput);
    auto daStage = adStage;
    swap_factor_axes(daStage);
    auto adOutput = split_factors(output_layout(), length_factor);
    swap_factor_axes(adOutput);

    // Column-to-column transforms D, then row-to-column transforms A.
    auto col2colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC,
                                                         this,
                                                         daInput.get_data_set_for_len_axes({0}),
                                                         daStage.get_data_set_for_len_axes({0}));

    if(GetBluesteinFuseType() != BFT_NONE)
        col2colPlan->blue = blue;

    // large1D flag to confirm we need multiply twiddle factor
    col2colPlan->large1D           = input_length(0);
    col2colPlan->legacyOutputOrder = OutputAxisOrder::SWAP_01;

    // second plan, row-to-column
    auto row2colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_RC,
                                                         this,
                                                         adStage.get_data_set_for_len_axes({0}),
                                                         adOutput.get_data_set_for_len_axes({0}));

    if(GetBluesteinFuseType() != BFT_NONE)
        row2colPlan->blue = blue;

    // CC , RC
    childNodes.emplace_back(std::move(col2colPlan));
    childNodes.emplace_back(std::move(row2colPlan));
}

void CC1DNode::AssignParams_internal()
{
    auto& col2colPlan = childNodes[0];
    auto& row2colPlan = childNodes[1];

    const bool output_to_boundary
        = obOut == OB_USER_OUT || obOut == OB_TEMP_CMPLX_FOR_REAL || obOut == OB_TEMP_BLUESTEIN;
    if(!output_to_boundary)
    {
        if(isRootNode())
            throw std::runtime_error("error: out-buffer mangled for root node (L1D_CC)");

        // Bluestein only takes this branch during its pre-pass, before buffers are assigned.
        assert((obOut == OB_UNINIT || !blue)
               && "fused Bluestein stages must take the first branch once buffers are assigned");
    }

    set_split_factor_strides(*col2colPlan, INPUT, input_layout(), true);

    // Nested TRTRT predicts the intermediate layout using its parent's output strides.
    // For other parents, or when writing from temp to the boundary, it is packed.
    if(!output_to_boundary && parent->scheme == CS_L1D_TRTRT)
        set_split_factor_strides(*col2colPlan, OUTPUT, parent->output_layout(), true);
    else
        set_packed_factor_strides(*col2colPlan, OUTPUT, true);

    set_swapped_factor_strides(*row2colPlan, INPUT, col2colPlan->output_layout());
    set_split_factor_strides(*row2colPlan, OUTPUT, output_layout(), true);

    if(output_to_boundary)
    {
        if(col2colPlan->GetBluesteinFuseType() == BFT_FWD_CHIRP_MUL
           || col2colPlan->GetBluesteinFuseType() == BFT_INV_CHIRP_MUL)
        {
            // The CC stage uses the same packed layout on either side of its workspace.
            auto workspace = bluestein_workspace(col2colPlan->output_layout());
            col2colPlan->blue->set_conv_buffer_layout(INPUT, workspace);
            col2colPlan->blue->set_conv_buffer_layout(OUTPUT, std::move(workspace));
        }

        if(row2colPlan->GetBluesteinFuseType() == BFT_FWD_CHIRP_MUL
           || row2colPlan->GetBluesteinFuseType() == BFT_INV_CHIRP_MUL)
        {
            // The RC stage's workspace output is packed in the transposed order.
            row2colPlan->blue->set_conv_buffer_layout(
                INPUT, bluestein_workspace(row2colPlan->input_layout()));
            row2colPlan->blue->set_conv_buffer_layout(
                OUTPUT, bluestein_workspace(row2colPlan->input_layout(), true));
        }
    }

    // special case for strided large 1D FFT with dist 1
    //
    // L1D_CC assumes consecutive sub-dimensional column-FFTs are
    // adjacent in memory, which makes column accesses efficient.
    // Strided L1D transforms break that assumption and have bad
    // performance.  But if dist is 1 then we can reorganize the
    // dimensions so the kernels use the batch dimension as the
    // adjacent one.
    //
    // Fused Bluestein is excluded: its stages' convolution buffer layouts
    // are set against the original geometry, which the reshape would break.
    const bool interleave = input_distance() == 1 && output_distance() == 1
                            && col2colPlan->obOut == OB_TEMP
                            && GetBluesteinType() != BT_MULTI_KERNEL_FUSED;
    col2colPlan->inner_batch_layout            = interleave;
    row2colPlan->inner_batch_layout            = interleave;
    col2colPlan->largeTwdBatchIsTransformCount = interleave;
    if(interleave)
    {
        // This traversal has only two non-batch strides. The old layout could
        // not pass the footprint check with additional inherited axes.
        if(data_rank() != 1)
            throw std::runtime_error(
                "Inconsistent length/stride dimensions given to compute_ptrdiff");

        // Store strides in logical [transform, factor, B] order. The kernel
        // sees [transform, B, factor], so both endpoints already have the
        // right strides; only the two views of the temp buffer change.
        col2colPlan->set_strides(OUTPUT, col2colPlan->input_layout());
        const auto first_stride = input_stride(0);
        row2colPlan->set_strides(
            INPUT,
            std::array<size_t, 3>{first_stride, row2colPlan->input_length(0) * first_stride, 1});
    }
}

/*****************************************************
 * L1D_CRT  *
 *****************************************************/
void CRT1DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 3)
           || (child_scheme_trees[0]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CC)
           || (child_scheme_trees[1]->curScheme != CS_KERNEL_STOCKHAM)
           || (child_scheme_trees[2]->curScheme != CS_KERNEL_TRANSPOSE))
        {
            throw std::runtime_error("CRT1DNode: Unexpected child scheme from solution map");
        }
    }

    if(length_factor == 0 || input_length(0) % length_factor != 0)
        throw std::runtime_error("L1D_CRT wrong factorization");

    auto daInput = split_factors(input_layout(), input_length(0) / length_factor);
    auto adStage = packed_factors(daInput);
    swap_factor_axes(daInput);
    auto daStage = adStage;
    swap_factor_axes(daStage);
    auto adOutput = split_factors(output_layout(), length_factor);
    swap_factor_axes(adOutput);

    auto col2colPlan               = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC,
                                                         this,
                                                         daInput.get_data_set_for_len_axes({0}),
                                                         daStage.get_data_set_for_len_axes({0}));
    col2colPlan->large1D           = input_length(0);
    col2colPlan->legacyOutputOrder = OutputAxisOrder::SWAP_01;

    const auto rowStage = adStage.get_data_set_for_len_axes({0});
    auto       row2rowPlan
        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM, this, rowStage, rowStage);

    auto transPlan = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_TRANSPOSE, this, std::move(adStage), std::move(adOutput));

    // --------------------------------
    // Fuse Shims
    // --------------------------------
    auto RT
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {row2rowPlan.get(), transPlan.get()});
    if(RT->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT));

    // --------------------------------
    // CRT
    // --------------------------------
    childNodes.emplace_back(std::move(col2colPlan));
    childNodes.emplace_back(std::move(row2rowPlan));
    childNodes.emplace_back(std::move(transPlan));
}

void CRT1DNode::AssignParams_internal()
{
    auto& col2colPlan = childNodes[0];
    auto& row2rowPlan = childNodes[1];
    auto& transPlan   = childNodes[2];

    const bool output_to_boundary = obOut == OB_USER_OUT || obOut == OB_TEMP_CMPLX_FOR_REAL;
    if(!output_to_boundary && isRootNode())
        throw std::runtime_error("error: out-buffer mangled for root node (L1D_CRT)");

    const bool reuse_parent_strides = !output_to_boundary && parent->scheme == CS_L1D_TRTRT;
    if(!output_to_boundary && !reuse_parent_strides)
    {
        // From a 2D/3D parent the intermediate layout assumes packing.
        assert(parent->output_stride(0) == 1);
        for(size_t axis = 1; axis < parent->data_rank(); ++axis)
            assert(parent->output_stride(axis)
                   == parent->output_stride(axis - 1) * parent->input_length(axis - 1));
    }

    set_split_factor_strides(*col2colPlan, INPUT, input_layout(), true);
    if(reuse_parent_strides)
        set_split_factor_strides(*col2colPlan, OUTPUT, parent->output_layout(), true);
    else
        set_packed_factor_strides(*col2colPlan, OUTPUT, true);

    set_swapped_factor_strides(*row2rowPlan, INPUT, col2colPlan->output_layout());
    row2rowPlan->set_strides(OUTPUT, row2rowPlan->input_layout());
    transPlan->set_strides(INPUT, row2rowPlan->output_layout());
    set_split_factor_strides(*transPlan, OUTPUT, output_layout(), true);
}

// Leaf Node..
/*****************************************************
 * CS_KERNEL_STOCKHAM  *
 *****************************************************/
void Stockham1DNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view        = GetKernelView();
    const auto  lens        = view.lengths();
    const auto  batch_accum = product(lens.begin() + 1, lens.end()) * view.batch;

    auto kernel = GetKernel();

    bwd      = kernel.transforms_per_block;
    wgs      = kernel.workgroup_size;
    gp.b_x   = (batch_accum + bwd - 1) / bwd;
    gp.wgs_x = wgs;

    // we don't even need lds (kernel_1,2,3,4,5,6,7,10,11,13,17) since we don't use them at all
    // TODO: we can even use swizzle to do the butterfly shuffle if threads_per_transform[0] <= warpSize
    //       such as kernel_8 = [4, 2] can probably gain some perf.
    if(kernel.threads_per_transform[0] <= deviceProp.warpSize && ebtype == EmbeddedType::NONE
       && kernel.factors.size() == 1)
        lds = 0;
    else
    {
        const auto lds_padding = ebtype != EmbeddedType::NONE ? 1 : 0;

        lds = (lens[0] + lds_padding) * bwd;
    }
}

bool Stockham1DNode::CreateDeviceResources()
{
    twd_attach_halfN = (ebtype != EmbeddedType::NONE);
    return LeafNode::CreateDeviceResources();
}

/*****************************************************
 * CS_KERNEL_STOCKHAM_PP  *
 *****************************************************/
void StockhamPP1DNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view        = GetKernelView();
    const auto  lens        = view.lengths();
    const auto  batch_accum = product(lens.begin() + 1, lens.end()) * view.batch;

    auto kernel = GetKernel();

    bwd      = kernel.transforms_per_block;
    wgs      = kernel.workgroup_size;
    gp.b_x   = (batch_accum + bwd - 1) / bwd;
    gp.wgs_x = wgs;

    const auto lds_padding = ebtype != EmbeddedType::NONE ? 1 : 0;

    lds = (lens[0] + lds_padding) * bwd;
}

bool StockhamPP1DNode::CreateDeviceResources()
{
    return CreatePartialPassDeviceResources(input_length(ppOffDim));
}

/*****************************************************
 * SBCC  *
 *****************************************************/
bool SBCCNode::KernelCheck(std::vector<FMKey>& kernel_keys)
{
    bool res = LeafNode::KernelCheck(kernel_keys);
    if(!res)
        return false;

    if(large1D > 0)
    {
        auto kernel    = GetKernel();
        largeTwd3Steps = kernel.use_3steps_large_twd;
        get_large_twd_base_steps(large1D, largeTwd3Steps, largeTwdBase, ltwdSteps);
    }

    // if we are doing tuning or running with the tuned solution, we have the specified_key.
    // we must directly run the kernel with the exact setting as the config
    // without the hardcoded tuning
    if(specified_key != nullptr)
    {
        InitIntrinsicMode();
        return true;
    }

    // hardocded-tuning according to benchmark
    TuneDirectRegType();

    // check if we can use buffer instr
    InitIntrinsicMode();
    // hardocded-tuning according to benchmark
    TuneIntrinsicMode();

    return true;
}

void SBCCNode::TuneDirectRegType()
{
    // half precision has not been tested yet, disable it for now.
    if(get_precision() == rocfft_precision_half)
    {
        dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
        return;
    }

    // for Navi, Haven't tested all.
    if(is_device_gcn_arch(deviceProp, "gfx1030"))
    {
        const auto len = input_length(0);
        if(len != 64 && len != 81 && len != 200 && len != 100 && len != 168)
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    else if(is_device_gcn_arch(deviceProp, "gfx908"))
    {
        // bad results from benchmark:
        // {125,sp}, {192,sp}, {216,sp}, {224,sp/dp}, {240,sp}, {243,sp}, {343,dp}
        // 100 and 168 can be better if enable half-lds
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {125, 192, 216, 224, 240, 243}},
               {rocfft_precision_double, {224, 343}}};
        if(length_excepted(exceptions, get_precision(), input_length(0)))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    else if(is_device_gcn_arch(deviceProp, "gfx90a"))
    {
        // bad results from benchmark:
        // {125,sp/dp}, {192,sp}, {200,sp}, {216,sp}, {224,sp/dp}, {240,sp}, {243,dp}
        // 100 and 168 can be better if enable half-lds
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {125, 192, 216, 200, 224, 240}},
               {rocfft_precision_double, {125, 224, 243}}};
        if(length_excepted(exceptions, get_precision(), input_length(0)))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
}

void SBCCNode::InitIntrinsicMode()
{
    const auto& view = GetKernelView();
    // 1. General rejections: (Guard) cases we definitely can't use buffer instruction
    // 2. half precision has not been tested yet, disable it for now.
    const auto precision    = get_precision();
    const auto element_size = complex_type_size(precision);
    if(((uint64_t)view.iDist * view.batch * element_size >= 0xFFFFFFFF)
       || ((uint64_t)view.oDist * view.batch * element_size >= 0xFFFFFFFF)
       || (precision == rocfft_precision_half))
    {
        intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;
        return;
    }

    // case 1: is running tuning or a tuned solution, then use the setting in the config
    if(specified_key != nullptr)
    {
        auto& config  = (*specified_key.get()).kernel_config;
        intrinsicMode = (config.intrinsic_buffer_inst) ? IntrinsicAccessType::ENABLE_BOTH
                                                       : IntrinsicAccessType::DISABLE_BOTH;
        return;
    }

    // case 2: un-tuned: auto decision: try to use buffer instruction as possible
    if(dir2regMode == TRY_ENABLE_IF_SUPPORT)
    {
        if(placement == rocfft_placement_inplace)
            intrinsicMode = IntrinsicAccessType::ENABLE_LOAD_ONLY;
        else
            intrinsicMode = IntrinsicAccessType::ENABLE_BOTH;
    }
}

// NB: remember set this value at this point instead of SetupGridParam_internal()
//     since we might need to pass this value to RTC generator
void SBCCNode::TuneIntrinsicMode()
{
    // already disabled
    if(intrinsicMode == IntrinsicAccessType::DISABLE_BOTH)
        return;

    // hardcoded turn-off in some exception cases
    // 1. currently we only enable this on 906, 908, 1030. TODO- test on 90a
    if((is_device_gcn_arch(deviceProp, "gfx906") == false)
       && (is_device_gcn_arch(deviceProp, "gfx908") == false)
       && (is_device_gcn_arch(deviceProp, "gfx1030") == false))
    {
        intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;
    }
    // 2. exception cases on 906. Based on benchmark results
    else if(is_device_gcn_arch(deviceProp, "gfx906"))
    {
        // bad results from benchmark:
        // {96,sp}, {125,sp}, {192,sp/dp}, {240,dp}, {256,sp/dp}, {343,sp/dp}
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {96, 125, 192, 256, 343}},
               {rocfft_precision_double, {192, 240, 256, 343}}};
        if(length_excepted(exceptions, get_precision(), input_length(0)))
            intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;
    }
    // 3. exception cases on 908. Based on benchmark results
    else if(is_device_gcn_arch(deviceProp, "gfx908"))
    {
        // bad results from benchmark:
        // {104,sp/dp}, {192,dp}, {240,dp}, {289,sp}
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {104, 289}}, {rocfft_precision_double, {104, 192, 240}}};
        if(length_excepted(exceptions, get_precision(), input_length(0)))
            intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;
    }
}

void SBCCNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view   = GetKernelView();
    const auto  lens   = view.lengths();
    auto        kernel = GetKernel();
    bwd                = kernel.transforms_per_block;
    wgs                = kernel.workgroup_size;

    lds = lens[0] * bwd;

    gp.b_x = (lens[1] - 1) / bwd + 1;
    gp.b_x *= product(lens.begin() + 2, lens.end()) * view.batch;
    gp.wgs_x = wgs;
}

/*****************************************************
 * SBCC Partial-Pass *
 *****************************************************/

void SBCCPPNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view   = GetKernelView();
    const auto  lens   = view.lengths();
    auto        kernel = GetKernel();

    bwd = kernel.transforms_per_block;
    wgs = kernel.workgroup_size;

    lds = lens[0] * bwd;

    gp.b_x = (lens[1] - 1) / bwd + 1;
    gp.b_x *= product(lens.begin() + 2, lens.end()) * view.batch;
    gp.wgs_x = wgs;

    // Grid arrangement is different than regular SBCC
    // for improved global memory access patterns.  A block covers every
    // off-dimension point that this kernel's partial pass transforms.
    auto factor = product(kernelFactorsPP.begin(), kernelFactorsPP.end());

    gp.b_x /= factor;
    gp.wgs_x *= factor;
    lds *= factor;
}

bool SBCCPPNode::CreateDeviceResources()
{
    // this node's length is rotated so that its own transform dimension comes
    // first, which shifts ppOffDim (a plan dimension) one slot to the right
    return CreatePartialPassDeviceResources(input_length((ppOffDim + 1) % data_rank()));
}

/*****************************************************
 * SBRC  *
 *****************************************************/
FMKey SBRCNode::GetKernelKey() const
{
    if(specified_key)
        return *specified_key.get();

    const auto len       = input_length(0);
    const auto precision = get_precision();

    // NB: Need to make sure that sbrcTranstype has the correct value
    if(sbrcTranstype == SBRC_TRANSPOSE_TYPE::NONE)
    {
        // find the base kernel at first
        FMKey baseKey(len, precision, scheme, TILE_ALIGNED);
        // if we have the base kernel, then we set the exact sbrc_trans_type and return the real key
        // if we don't, then we simply return a key with NONE sbrc_trans_type
        // which will make KernelCheck() trigger an exception
        if(pool.has_function(baseKey))
        {
            auto bwd      = pool.get_kernel(baseKey).transforms_per_block;
            sbrcTranstype = sbrc_transpose_type(bwd);
        }
    }

    return FMKey(len, precision, scheme, sbrcTranstype);
}

bool SBRCNode::KernelCheck(std::vector<FMKey>& kernel_keys)
{
    bool res = LeafNode::KernelCheck(kernel_keys);
    if(!res)
        return false;

    // if we are doing tuning or running with the tuned solution, we have the specified_key.
    // we must directly run the kernel with the exact setting as the config
    // without the hardcoded tuning
    if(specified_key != nullptr)
        return true;

    // hardocded-tuning according to benchmark
    TuneDirectRegType();

    return true;
}

void SBRCNode::TuneDirectRegType()
{
    // half precision has not been tested yet, disable it for now.
    if(get_precision() == rocfft_precision_half)
    {
        dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
        return;
    }

    if(is_device_gcn_arch(deviceProp, "gfx906"))
    {
        // bad results from benchmark:
        // {49,sp}, {128,sp}, {64,dp}, {81,dp}, {100,dp}
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {49, 128}}, {rocfft_precision_double, {64, 81, 100}}};
        if(length_excepted(exceptions, get_precision(), input_length(0)))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    else if(is_device_gcn_arch(deviceProp, "gfx908"))
    {
        // bad results from benchmark:
        // {81,sp}, {100,sp}, {128,sp/dp}, {192,sp}, {200,sp}, {512,sp}, {125,dp}
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {81, 100, 128, 192, 200, 512}},
               {rocfft_precision_double, {125, 128}}};
        if(length_excepted(exceptions, get_precision(), input_length(0)))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    else if(is_device_gcn_arch(deviceProp, "gfx90a"))
    {
        // bad results from benchmark:
        // {49,sp}, {81,sp/dp}, {100,sp/dp}, {125,sp/dp}, {200,sp}, {512,sp}, {64,dp}
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {49, 81, 100, 125, 200, 512}},
               {rocfft_precision_double, {64, 81, 100, 125}}};
        if(length_excepted(exceptions, get_precision(), input_length(0)))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    // we don't enable the features for others
    else
    {
        dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
}

void SBRCNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view = GetKernelView();
    const auto  lens = view.lengths();
    // sbrcTransType has already been assigned in KernelCheck();
    auto kernel = pool.get_kernel(GetKernelKey());
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;
    lds         = lens[0] * bwd;
    gp.b_x      = (lens[1] - 1) / bwd + 1;
    gp.b_x *= product(lens.begin() + 2, lens.end()) * view.batch;
    gp.wgs_x = wgs;
}

SBRC_TRANSPOSE_TYPE SBRCNode::sbrc_transpose_type(unsigned int blockWidth) const
{
    const auto alignment_dimension = inner_batch_layout ? batch_count() : input_length(1);
    return (alignment_dimension % blockWidth == 0) ? TILE_ALIGNED : TILE_UNALIGNED;
}

/*****************************************************
 * SBCR  *
 *****************************************************/
bool SBCRNode::KernelCheck(std::vector<FMKey>& kernel_keys)
{
    bool res = LeafNode::KernelCheck(kernel_keys);
    if(!res)
        return false;

    // if we are doing tuning or running with the tuned solution, we have the specified_key.
    // we must directly run the kernel with the exact setting as the config
    // without the hardcoded tuning
    if(specified_key != nullptr)
    {
        InitIntrinsicMode();
        return true;
    }

    // hardocded-tuning according to benchmark
    TuneDirectRegType();

    // check if we can use buffer instr
    InitIntrinsicMode();
    // hardocded-tuning according to benchmark
    TuneIntrinsicMode();

    return true;
}

void SBCRNode::TuneDirectRegType()
{
    // half precision has not been tested yet, disable it for now.
    if(get_precision() == rocfft_precision_half)
    {
        dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
        return;
    }

    // switch on/off according to the arch
    // tweaking the setting based on the benchmark results.

    // so far, we've only tested on gfx908, 90a. Navi and MI50 even don't use SBCR
    if(!is_device_gcn_arch(deviceProp, "gfx908") && !is_device_gcn_arch(deviceProp, "gfx90a"))
    {
        dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    // 90a has a exception, 908 seems good
    if(is_device_gcn_arch(deviceProp, "gfx90a"))
    {
        // {200,sp} is bad
        if(input_length(0) == 200 && get_precision() == rocfft_precision_single)
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
}

void SBCRNode::InitIntrinsicMode()
{
    const auto& view = GetKernelView();
    // 1. General rejections: (Guard) cases we definitely can't use buffer instruction
    // 2. half precision has not been tested yet, disable it for now.
    const auto precision    = get_precision();
    const auto element_size = complex_type_size(precision);
    if(((uint64_t)view.iDist * view.batch * element_size >= 0xFFFFFFFF)
       || ((uint64_t)view.oDist * view.batch * element_size >= 0xFFFFFFFF)
       || (precision == rocfft_precision_half))
    {
        intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;
        return;
    }

    // case 1: is running tuning or a tuned solution, then use the setting in the config
    if(specified_key != nullptr)
    {
        auto& config  = (*specified_key.get()).kernel_config;
        intrinsicMode = (config.intrinsic_buffer_inst) ? IntrinsicAccessType::ENABLE_BOTH
                                                       : IntrinsicAccessType::DISABLE_BOTH;
        return;
    }

    // case 2: un-tuned: auto decision: try to use buffer instruction as possible
    if(dir2regMode == TRY_ENABLE_IF_SUPPORT)
    {
        intrinsicMode = IntrinsicAccessType::ENABLE_BOTH;
    }
}

// NB: remember set this value at this point instead of SetupGridParam_internal()
//     since we might need to pass this value to RTC generator
void SBCRNode::TuneIntrinsicMode()
{
    // already disabled
    if(intrinsicMode == IntrinsicAccessType::DISABLE_BOTH)
        return;

    // hardcoded turn-off in some exception cases
    // 1. currently we only enable this on 908. TODO- test on 90a
    if(is_device_gcn_arch(deviceProp, "gfx908") == false)
    {
        intrinsicMode = IntrinsicAccessType::DISABLE_BOTH;
    }
}

void SBCRNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view   = GetKernelView();
    const auto  lens   = view.lengths();
    auto        kernel = pool.get_kernel(GetKernelKey());
    wgs                = kernel.workgroup_size;
    bwd                = kernel.transforms_per_block;

    const auto lds_padding = ebtype != EmbeddedType::NONE ? 1 : 0;

    lds    = (lens[0] + lds_padding) * bwd;
    gp.b_x = (lens[1] - 1) / bwd + 1;
    gp.b_x *= product(lens.begin() + 2, lens.end()) * view.batch;
    gp.wgs_x = wgs;

    return;
}

bool SBCRNode::CreateDeviceResources()
{
    twd_attach_halfN = (ebtype != EmbeddedType::NONE);
    return LeafNode::CreateDeviceResources();
}
