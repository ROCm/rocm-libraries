// Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

    // first row fft
    NodeMetaData row1PlanData(this);
    row1PlanData.length.push_back(length[0]);
    row1PlanData.dimension = 1;
    row1PlanData.length.push_back(length[1]);
    for(size_t index = 2; index < length.size(); index++)
    {
        row1PlanData.length.push_back(length[index]);
    }
    auto row1Plan = NodeFactory::CreateExplicitNode(row1PlanData, this, determined_scheme_node0);
    row1Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    // first transpose
    auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    trans1Plan->length.push_back(length[0]);
    trans1Plan->length.push_back(length[1]);
    trans1Plan->dimension = 2;
    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->length.push_back(length[index]);
    }
    trans1Plan->SetTransposeOutputLength();

    // second row fft
    NodeMetaData row2PlanData(this);
    row2PlanData.length.push_back(length[1]);
    row2PlanData.dimension = 1;
    row2PlanData.length.push_back(length[0]);
    for(size_t index = 2; index < length.size(); index++)
    {
        row2PlanData.length.push_back(length[index]);
    }
    auto row2Plan = NodeFactory::CreateExplicitNode(row2PlanData, this, determined_scheme_node2);
    row2Plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

    // second transpose
    auto trans2Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE, this);
    trans2Plan->length.push_back(length[1]);
    trans2Plan->length.push_back(length[0]);
    trans2Plan->dimension = 2;
    for(size_t index = 2; index < length.size(); index++)
    {
        trans2Plan->length.push_back(length[index]);
    }
    trans2Plan->SetTransposeOutputLength();

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
    assert(inStrideBlue.size() == outStrideBlue.size());
    bool setBlueData = inStrideBlue.size();

    auto& row1Plan      = childNodes[0];
    row1Plan->inStride  = inStride;
    row1Plan->iDist     = iDist;
    row1Plan->outStride = outStride;
    row1Plan->oDist     = oDist;

    if(setBlueData)
    {
        row1Plan->inStrideBlue  = inStrideBlue;
        row1Plan->iDistBlue     = iDistBlue;
        row1Plan->outStrideBlue = outStrideBlue;
        row1Plan->oDistBlue     = oDistBlue;
    }

    row1Plan->AssignParams();

    auto& trans1Plan     = childNodes[1];
    trans1Plan->inStride = row1Plan->outStride;
    trans1Plan->iDist    = row1Plan->oDist;
    trans1Plan->outStride.push_back(trans1Plan->length[1]);
    trans1Plan->outStride.push_back(1);
    trans1Plan->oDist = trans1Plan->length[0] * trans1Plan->outStride[0];

    if(setBlueData)
    {
        trans1Plan->inStrideBlue = row1Plan->outStrideBlue;
        trans1Plan->iDistBlue    = row1Plan->oDistBlue;
        trans1Plan->outStrideBlue.push_back(trans1Plan->length[1]);
        trans1Plan->outStrideBlue.push_back(1);
        trans1Plan->oDistBlue = trans1Plan->length[0] * trans1Plan->outStrideBlue[0];
    }

    for(size_t index = 2; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];

        if(setBlueData)
        {
            trans1Plan->outStrideBlue.push_back(trans1Plan->oDistBlue);
            trans1Plan->oDistBlue *= length[index];
        }
    }

    auto& row2Plan     = childNodes[2];
    row2Plan->inStride = trans1Plan->outStride;
    std::swap(row2Plan->inStride[0], row2Plan->inStride[1]);
    row2Plan->iDist     = trans1Plan->oDist;
    row2Plan->outStride = row2Plan->inStride;
    row2Plan->oDist     = row2Plan->iDist;
    if(setBlueData)
    {
        row2Plan->inStrideBlue = trans1Plan->outStrideBlue;
        std::swap(row2Plan->inStrideBlue[0], row2Plan->inStrideBlue[1]);
        row2Plan->iDistBlue     = trans1Plan->oDistBlue;
        row2Plan->outStrideBlue = row2Plan->inStrideBlue;
        row2Plan->oDistBlue     = row2Plan->iDistBlue;
    }

    row2Plan->AssignParams();

    auto& trans2Plan      = childNodes[3];
    trans2Plan->inStride  = row2Plan->outStride;
    trans2Plan->iDist     = row2Plan->oDist;
    trans2Plan->outStride = outStride;
    std::swap(trans2Plan->outStride[0], trans2Plan->outStride[1]);
    trans2Plan->oDist = oDist;

    if(setBlueData)
    {
        trans2Plan->inStrideBlue  = row2Plan->outStrideBlue;
        trans2Plan->iDistBlue     = row2Plan->oDistBlue;
        trans2Plan->outStrideBlue = outStrideBlue;
        std::swap(trans2Plan->outStrideBlue[0], trans2Plan->outStrideBlue[1]);
        trans2Plan->oDistBlue = oDistBlue;
    }
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

    // row fft
    NodeMetaData rowPlanData(this);
    rowPlanData.length.push_back(length[0]);
    rowPlanData.dimension = 1;
    rowPlanData.length.push_back(length[1]);
    for(size_t index = 2; index < length.size(); index++)
    {
        rowPlanData.length.push_back(length[index]);
    }
    auto rowPlan = NodeFactory::CreateExplicitNode(rowPlanData, this, determined_scheme_node0);
    rowPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    // column fft
    auto colPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
    colPlan->length.push_back(length[1]);
    colPlan->dimension = 1;
    colPlan->length.push_back(length[0]);
    colPlan->large1D = 0; // No twiddle factor in sbcc kernel
    for(size_t index = 2; index < length.size(); index++)
    {
        colPlan->length.push_back(length[index]);
    }
    colPlan->outputLength = colPlan->length;
    std::swap(colPlan->outputLength[0], colPlan->outputLength[1]);

    // RC
    childNodes.emplace_back(std::move(rowPlan));
    childNodes.emplace_back(std::move(colPlan));
}

void RC2DNode::AssignParams_internal()
{
    auto& rowPlan = childNodes[0];
    auto& colPlan = childNodes[1];

    rowPlan->inStride = inStride;
    rowPlan->iDist    = iDist;

    // make output of row plan contiguous if the 2D node is at the root of the plan tree
    if(parent == nullptr)
    {
        rowPlan->outStride = {1, length[0]};
        rowPlan->oDist     = length[0] * length[1];
    }
    else
    {
        rowPlan->outStride = inStride;
        rowPlan->oDist     = iDist;
    }

    rowPlan->AssignParams();

    colPlan->inStride = rowPlan->outStride;
    std::swap(colPlan->inStride[0], colPlan->inStride[1]);

    colPlan->iDist = rowPlan->oDist;

    colPlan->outStride = outStride;
    std::swap(colPlan->outStride[0], colPlan->outStride[1]);
    colPlan->oDist = oDist;
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
    while(cummulative_product != length[0])
    {
        cummulative_product *= kernelFactors[count++];
    }
    radices1.insert(radices1.cbegin(), kernelFactors.cbegin(), kernelFactors.cbegin() + count);
    radices2.insert(radices2.cbegin(), kernelFactors.cbegin() + count, kernelFactors.cend());

    twd_attach_halfN  = (ebtype == EmbeddedType::Real2C_POST);
    twd_attach_halfN2 = (ebtype == EmbeddedType::C2Real_PRE);
    // create one set of twiddles for each dimension
    std::tie(twiddles, twiddles_size) = Repo::GetTwiddles2D(length[0],
                                                            length[1],
                                                            precision,
                                                            deviceProp,
                                                            twd_attach_halfN,
                                                            twd_attach_halfN2,
                                                            radices1,
                                                            radices2);

    return CreateLargeTwdTable();
}

void Single2DNode::SetupGridParam_internal(GridParam& gp)
{
    auto kernel = GetKernel();
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;

    gp.b_x   = (batch + bwd - 1) / bwd;
    gp.wgs_x = wgs;

    size_t padded_len0 = length[0];
    size_t padded_len1 = length[1];

    if(ebtype == EmbeddedType::Real2C_POST)
        padded_len0 += 1;
    if(ebtype == EmbeddedType::C2Real_PRE)
        padded_len1 += 1;

    // if fastest length is power of 2, pad it to avoid LDS bank conflicts
    if(ebtype != EmbeddedType::C2Real_PRE)
    {
        // for EBTYPE NONE/POST, the fastest length is length0
        padded_len0 = IsPo2(padded_len0) ? padded_len0 + 1 : padded_len0;
    }
    else
    {
        // for EBTYPE PRE, the fastest length is length1
        padded_len1 = IsPo2(padded_len1) ? padded_len1 + 1 : padded_len1;
    }

    lds = padded_len0 * padded_len1 * bwd;

    // if we're doing 3D transform, we need to repeat the 2D
    // transform in the 3rd dimension
    if(length.size() > 2)
        gp.b_x *= length[2];

    return;
}
