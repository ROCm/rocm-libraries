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

#include "tree_node_2D.h"
#include "../../shared/arithmetic.h"
#include "function_pool.h"
#include "fuse_shim.h"
#include "node_factory.h"
#include "repo.h"

#include <array>

namespace
{
    constexpr std::array<size_t, 2> yx{1, 0};

    // Pack Y first while describing the buffer in the [X, Y] frame.
    auto packed_y_strides(const data_set_t& xy)
    {
        std::vector<size_t> strides(xy.get_full_rank());
        strides[0] = xy.axis_span(1);
        strides[1] = 1;
        strides[2] = xy.axis_span(0) * strides[0];
        for(size_t axis = 3; axis < strides.size(); ++axis)
            strides[axis] = xy.axis_span(axis - 1) * strides[axis - 1];
        return strides;
    }
}

/*****************************************************
 * 2D_RTRT  *
 *****************************************************/
void RTRT2DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    ComputeScheme determined_scheme_node0 = CS_NONE;
    ComputeScheme determined_scheme_node2 = CS_NONE;
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 4)
           || (child_scheme_trees[1]->curScheme != CS_KERNEL_TRANSPOSE)
           || (child_scheme_trees[3]->curScheme != CS_KERNEL_TRANSPOSE))
        {
            throw std::runtime_error("RTRT2DNode: Unexpected child scheme from solution map");
        }
        determined_scheme_node0 = child_scheme_trees[0]->curScheme;
        determined_scheme_node2 = child_scheme_trees[2]->curScheme;
    }

    // First row FFT: transform X without changing the stage's axis order.
    NodeMetaData row1PlanData(input_layout().get_data_set_for_len_axes({0}),
                              output_layout().get_data_set_for_len_axes({0}),
                              this);
    auto row1Plan = NodeFactory::CreateExplicitNode(row1PlanData, this, determined_scheme_node0);
    row1Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    auto temp
        = output_layout().with_element_type(rocfft_array_type_complex_interleaved, get_precision());
    temp.set_strides_and_distances(packed_y_strides(temp));
    auto trans1Plan
        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this, output_layout(), temp);

    // Second row FFT: selecting Y puts it first and moves X into batch.
    const auto   tempY = temp.get_data_set_for_len_axes({1});
    NodeMetaData row2PlanData(tempY, tempY, this);
    auto row2Plan = NodeFactory::CreateExplicitNode(row2PlanData, this, determined_scheme_node2);
    row2Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

    // Second transpose: both sides describe [Y, X].
    auto outputYX = output_layout();
    outputYX.reorder_axes(yx, axis_scope::length_only);
    temp.reorder_axes(yx, axis_scope::length_only);
    auto trans2Plan = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_TRANSPOSE, this, std::move(temp), std::move(outputYX));

    // --------------------------------
    // Fuse Shims
    // --------------------------------
    auto RT1
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {row1Plan.get(), trans1Plan.get()});
    if(RT1->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT1));

    auto RT2
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {row2Plan.get(), trans2Plan.get()});
    if(RT2->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT2));

    // --------------------------------
    // RTRT
    // --------------------------------
    childNodes.emplace_back(std::move(row1Plan));
    childNodes.emplace_back(std::move(trans1Plan));
    childNodes.emplace_back(std::move(row2Plan));
    childNodes.emplace_back(std::move(trans2Plan));
}

void RTRT2DNode::AssignParams_internal()
{
    // blue is only ever set on Bluestein nodes and their descendants
    // (such nodes and their descendants are all 1D only)
    assert(!blue && "RTRT2DNode does not propagate Bluestein parameters to its children");

    auto& row1Plan = childNodes[0];
    row1Plan->set_strides(io_data_label::INPUT, input_layout());
    row1Plan->set_strides(io_data_label::OUTPUT, output_layout());

    row1Plan->AssignParams();

    auto& trans1Plan = childNodes[1];
    trans1Plan->set_strides(io_data_label::INPUT, row1Plan->output_layout());
    auto tempStrides = packed_y_strides(trans1Plan->output_layout());
    trans1Plan->set_strides(io_data_label::OUTPUT, tempStrides);

    auto& row2Plan = childNodes[2];
    std::swap(tempStrides[0], tempStrides[1]);
    row2Plan->set_strides(io_data_label::INPUT, tempStrides);
    row2Plan->set_strides(io_data_label::OUTPUT, tempStrides);

    row2Plan->AssignParams();

    auto& trans2Plan = childNodes[3];
    trans2Plan->set_strides(io_data_label::INPUT, row2Plan->output_layout());
    auto outputStrides = output_layout().strides_and_distances();
    std::swap(outputStrides[0], outputStrides[1]);
    trans2Plan->set_strides(io_data_label::OUTPUT, outputStrides);
}

/*****************************************************
 * 2D_RC  *
 *****************************************************/
void RC2DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    ComputeScheme determined_scheme_node0 = CS_NONE;
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 2)
           || (child_scheme_trees[1]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CC))
        {
            throw std::runtime_error("RC2DNode: Unexpected child scheme from solution map");
        }
        determined_scheme_node0 = child_scheme_trees[0]->curScheme;
    }

    // Row FFT along X.
    NodeMetaData rowPlanData(input_layout().get_data_set_for_len_axes({0}),
                             output_layout().get_data_set_for_len_axes({0}),
                             this);
    auto rowPlan = NodeFactory::CreateExplicitNode(rowPlanData, this, determined_scheme_node0);
    rowPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    // Column FFT along Y, using the [Y, X] view of the row result.
    const auto columns = output_layout().get_data_set_for_len_axes({1});
    auto       colPlan
        = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this, columns, columns);
    colPlan->large1D           = 0; // No twiddle factor in sbcc kernel
    colPlan->legacyOutputOrder = OutputAxisOrder::SWAP_01;

    // RC
    childNodes.emplace_back(std::move(rowPlan));
    childNodes.emplace_back(std::move(colPlan));
}

void RC2DNode::AssignParams_internal()
{
    auto& rowPlan = childNodes[0];
    auto& colPlan = childNodes[1];

    rowPlan->set_strides(io_data_label::INPUT, input_layout());
    auto outputStrides = output_layout().strides_and_distances();
    rowPlan->set_strides(io_data_label::OUTPUT, outputStrides);

    rowPlan->AssignParams();

    auto colInputStrides = rowPlan->output_layout().strides_and_distances();
    std::swap(colInputStrides[0], colInputStrides[1]);
    colPlan->set_strides(io_data_label::INPUT, colInputStrides);
    std::swap(outputStrides[0], outputStrides[1]);
    colPlan->set_strides(io_data_label::OUTPUT, outputStrides);
}

// Leaf Node..
/*****************************************************
 * CS_KERNEL_2D_SINGLE  *
 *****************************************************/
bool Single2DNode::CreateDeviceResources()
{
    // we need to put radices1/2 into key for twd_table, since when offline tuning,
    // we are generating kernels with different radices, so for different radices,
    // twd_table can't be shared
    std::vector<size_t> radices1, radices2;

    int    count               = 0;
    size_t cummulative_product = 1;
    while(cummulative_product != input_length(0))
    {
        cummulative_product *= kernelFactors[count++];
    }
    radices1.insert(radices1.cbegin(), kernelFactors.cbegin(), kernelFactors.cbegin() + count);
    radices2.insert(radices2.cbegin(), kernelFactors.cbegin() + count, kernelFactors.cend());

    twd_attach_halfN = (ebtype != EmbeddedType::NONE);
    // create one set of twiddles for each dimension
    std::tie(twiddles, twiddles_size) = Repo::GetTwiddles2D(input_length(0),
                                                            input_length(1),
                                                            get_precision(),
                                                            deviceProp,
                                                            twd_attach_halfN,
                                                            radices1,
                                                            radices2);

    return CreateLargeTwdTable();
}

void Single2DNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view   = GetKernelView();
    const auto  lens   = view.lengths();
    auto        kernel = GetKernel();
    bwd                = kernel.transforms_per_block;
    wgs                = kernel.workgroup_size;

    gp.b_x   = (view.batch + bwd - 1) / bwd;
    gp.wgs_x = wgs;

    size_t padded_len0 = lens[0];
    size_t padded_len1 = lens[1];

    if(ebtype != EmbeddedType::NONE)
        padded_len0 += 1;

    // if fastest length is power of 2, pad it to avoid LDS bank conflicts
    padded_len0 = IsPo2(padded_len0) ? padded_len0 + 1 : padded_len0;

    lds = padded_len0 * padded_len1 * bwd;

    // if we're doing 3D transform, we need to repeat the 2D
    // transform in the 3rd dimension
    if(lens.size() > 2)
        gp.b_x *= lens[2];
}
