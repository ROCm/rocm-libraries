// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node_3D.h"
#include "../../shared/arithmetic.h"
#include "function_pool.h"
#include "logging.h"
#include "node_factory.h"
#include "tuning_helper.h"
#include <numeric>

/*****************************************************
 * 3D_RTRT  *
 *****************************************************/
void RTRT3DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    ComputeScheme determined_scheme_node0 = CS_NONE;
    ComputeScheme determined_scheme_node2 = CS_NONE;
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 4)
           || (child_scheme_trees[1]->curScheme != CS_KERNEL_TRANSPOSE_XY_Z)
           || (child_scheme_trees[3]->curScheme != CS_KERNEL_TRANSPOSE_Z_XY))
        {
            throw std::runtime_error("RTRT3DNode: Unexpected child scheme from solution map");
        }
        determined_scheme_node0 = child_scheme_trees[0]->curScheme;
        determined_scheme_node2 = child_scheme_trees[2]->curScheme;
    }

    // 2d fft
    NodeMetaData xyPlanData(this);
    xyPlanData.length    = length;
    xyPlanData.dimension = 2;
    auto xyPlan = NodeFactory::CreateExplicitNode(xyPlanData, this, determined_scheme_node0);
    xyPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    // first transpose
    auto trans1Plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z, this);
    trans1Plan->length = length;
    trans1Plan->SetTransposeOutputLength();
    std::swap(trans1Plan->length[1], trans1Plan->length[2]);
    trans1Plan->dimension = 2;

    // z fft
    NodeMetaData zPlanData(this);
    zPlanData.dimension = 1;
    zPlanData.length.push_back(length[2]);
    zPlanData.length.push_back(length[0]);
    zPlanData.length.push_back(length[1]);
    auto zPlan = NodeFactory::CreateExplicitNode(zPlanData, this, determined_scheme_node2);
    zPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

    // second transpose
    auto trans2Plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
    trans2Plan->length = zPlan->length;
    trans2Plan->SetTransposeOutputLength();
    trans2Plan->dimension = 2;

    // --------------------------------
    // Fuse Shims
    // --------------------------------
    auto RT1
        = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {xyPlan.get(), trans1Plan.get()});
    if(RT1->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT1));

    auto RT2 = NodeFactory::CreateFuseShim(FT_STOCKHAM_WITH_TRANS, {zPlan.get(), trans2Plan.get()});
    if(RT2->IsSchemeFusable())
        fuseShims.emplace_back(std::move(RT2));

    // --------------------------------
    // Push to child nodes : 3D_RTRT
    // --------------------------------
    childNodes.emplace_back(std::move(xyPlan));
    childNodes.emplace_back(std::move(trans1Plan));
    childNodes.emplace_back(std::move(zPlan)); // notice that move() will set zPlan to nullptr
    childNodes.emplace_back(std::move(trans2Plan));

    // NB:
    //    Don't fuse zPlan, trans2Plan to FT_STOCKHAM_WITH_TRANS_Z_XY
    //    because the preceding trans1 is XY_Z:
    //    xyPlan handles the first 2 dimensions without a transpose.
    //    So the XY_Z tranpose arranges the Z dimension to the fastest dim, and
    //    Z_XY puts it back to the expected arrangement for output.
}

void RTRT3DNode::AssignParams_internal()
{
    assert(childNodes.size() == 4);

    // B -> B
    auto& xyPlan     = childNodes[0];
    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->AssignParams();

    // B -> T
    auto& trans1Plan     = childNodes[1];
    trans1Plan->inStride = xyPlan->outStride;
    std::swap(trans1Plan->inStride[1], trans1Plan->inStride[2]);
    trans1Plan->iDist = xyPlan->oDist;

    trans1Plan->outStride.push_back(trans1Plan->length[1]);
    trans1Plan->outStride.push_back(1);
    trans1Plan->outStride.push_back(trans1Plan->length[0] * trans1Plan->outStride[0]);
    trans1Plan->oDist = trans1Plan->length[2] * trans1Plan->outStride[2];

    for(size_t index = 3; index < length.size(); index++)
    {
        trans1Plan->outStride.push_back(trans1Plan->oDist);
        trans1Plan->oDist *= length[index];
    }

    // T -> T
    auto& zPlan     = childNodes[2];
    zPlan->inStride = trans1Plan->outStride;
    std::swap(zPlan->inStride[0], zPlan->inStride[1]);
    zPlan->iDist = trans1Plan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;

    zPlan->AssignParams();

    // T -> B
    auto& trans2Plan     = childNodes[3];
    trans2Plan->inStride = zPlan->outStride;
    trans2Plan->iDist    = zPlan->oDist;

    trans2Plan->outStride = outStride;
    std::swap(trans2Plan->outStride[1], trans2Plan->outStride[2]);
    std::swap(trans2Plan->outStride[0], trans2Plan->outStride[1]);
    trans2Plan->oDist = oDist;
}

/*****************************************************
 * 3D_TRTRTR  *
 *****************************************************/
void TRTRTR3DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 6)
           || (child_scheme_trees[0]->curScheme != CS_KERNEL_TRANSPOSE_Z_XY)
           || (child_scheme_trees[2]->curScheme != CS_KERNEL_TRANSPOSE_Z_XY)
           || (child_scheme_trees[4]->curScheme != CS_KERNEL_TRANSPOSE_Z_XY))
        {
            throw std::runtime_error("TRTRTR3DNode: Unexpected child scheme from solution map");
        }
    }

    std::vector<size_t> cur_length = length;

    for(int i = 0; i < 6; i += 2)
    {
        // transpose Z_XY
        auto trans_plan    = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY, this);
        trans_plan->length = cur_length;
        trans_plan->SetTransposeOutputLength();
        trans_plan->dimension = 2;

        std::swap(cur_length[0], cur_length[1]);
        std::swap(cur_length[1], cur_length[2]);

        // row ffts
        NodeMetaData row_plan_data(this);
        row_plan_data.length    = cur_length;
        row_plan_data.dimension = 1;
        // skip the decide scheme part in node factory
        ComputeScheme determined_scheme
            = (noSolution) ? CS_NONE : child_scheme_trees[i + 1]->curScheme;
        auto row_plan = NodeFactory::CreateExplicitNode(row_plan_data, this, determined_scheme);
        row_plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[i + 1].get());

        // TR
        childNodes.emplace_back(std::move(trans_plan));
        childNodes.emplace_back(std::move(row_plan));
    }

    // --------------------------------
    // Fuse Shims
    // T-[RT]-RTR and TRT-[RT]-R
    // --------------------------------
    auto RT1 = NodeFactory::CreateFuseShim(
        FT_STOCKHAM_WITH_TRANS_Z_XY,
        {childNodes[0].get(), childNodes[1].get(), childNodes[2].get()});
    bool RT1Fusable = RT1->IsSchemeFusable();
    if(RT1Fusable)
        fuseShims.emplace_back(std::move(RT1));

    auto RT2 = NodeFactory::CreateFuseShim(
        FT_STOCKHAM_WITH_TRANS_Z_XY,
        {childNodes[2].get(), childNodes[3].get(), childNodes[4].get()});
    bool RT2Fusable = RT2->IsSchemeFusable();
    if(RT2Fusable)
        fuseShims.emplace_back(std::move(RT2));

    // --------------------------------
    // If only partial fusions work, try if we could fuse alternatively
    // [TR][TR][TR]
    // --------------------------------
    if(!RT1Fusable)
    {
        auto TR1 = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                               {childNodes[0].get(), childNodes[1].get()});
        if(TR1->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR1));
    }
    if(!RT1Fusable && !RT2Fusable)
    {
        auto TR2 = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                               {childNodes[2].get(), childNodes[3].get()});
        if(TR2->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR2));
    }
    if(!RT2Fusable)
    {
        auto TR3 = NodeFactory::CreateFuseShim(FT_TRANS_WITH_STOCKHAM,
                                               {childNodes[4].get(), childNodes[5].get()});
        if(TR3->IsSchemeFusable())
            fuseShims.emplace_back(std::move(TR3));
    }
}

void TRTRTR3DNode::AssignParams_internal()
{
    assert(childNodes.size() == 6);

    for(int i = 0; i < 6; i += 2)
    {
        auto& trans_plan = childNodes[i];
        if(i == 0)
        {
            trans_plan->inStride = inStride;
            trans_plan->iDist    = iDist;
        }
        else
        {
            trans_plan->inStride = childNodes[i - 1]->outStride;
            trans_plan->iDist    = childNodes[i - 1]->oDist;
        }

        trans_plan->outStride.push_back(1);
        trans_plan->outStride.push_back(trans_plan->outStride[0] * trans_plan->length[1]);
        trans_plan->outStride.push_back(trans_plan->outStride[1] * trans_plan->length[2]);
        trans_plan->oDist = trans_plan->outStride[2] * trans_plan->length[0];

        auto& row_plan     = childNodes[i + 1];
        row_plan->inStride = trans_plan->outStride;
        row_plan->iDist    = trans_plan->oDist;

        std::swap(trans_plan->outStride[1], trans_plan->outStride[2]);
        std::swap(trans_plan->outStride[0], trans_plan->outStride[1]);

        if(i == 4)
        {
            row_plan->outStride = outStride;
            row_plan->oDist     = oDist;
        }
        else
        {
            row_plan->outStride.push_back(1);
            row_plan->outStride.push_back(row_plan->outStride[0] * row_plan->length[0]);
            row_plan->outStride.push_back(row_plan->outStride[1] * row_plan->length[1]);
            row_plan->oDist = row_plan->outStride[2] * row_plan->length[2];
        }
        row_plan->AssignParams();
    }
}

/*****************************************************
 * CS_3D_BLOCK_RC  *
 *****************************************************/
void BLOCKRC3DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    std::vector<size_t> cur_length = length;

    // NB:
    //   The idea is to change the SBRC from doing XZ-plane to doing XY-plane
    //   Some problems are still faster with 3D_RC so they never go here
    bool is906    = is_device_gcn_arch(deviceProp, "gfx906");
    bool is908    = is_device_gcn_arch(deviceProp, "gfx908");
    bool is90a    = is_device_gcn_arch(deviceProp, "gfx90a");
    bool isDouble = (precision == rocfft_precision_double);

    // specific flags for lengths
    bool has50           = (length[0] == 50 || length[1] == 50 || length[2] == 50);
    bool has64           = (length[0] == 64 || length[1] == 64 || length[2] == 64);
    bool has100          = (length[0] == 100 || length[1] == 100 || length[2] == 100);
    bool has200          = (length[0] == 200 || length[1] == 200 || length[2] == 200);
    bool hasPow2         = (IsPo2(length[0]) || IsPo2(length[1]) || IsPo2(length[2]));
    bool isDiagonalTrans = is_diagonal_sbrc_3D_length(length[0]) && is_cube_size(length);

    //   none of diagonal is better by Z_XY (every arch)
    //   both 50, 100 are worse by Z_XY (every arch)
    bool use_ZXY_sbrc = (is906 || is908 || is90a) && (!isDiagonalTrans) && (!has50) && (!has100);

    if(use_ZXY_sbrc)
    {
        if(is906)
        {
            // sbrc 64 performs worse by Z_XY on 906
            use_ZXY_sbrc = (has64 == false);
        }
        else // 908, 90a
        {
            // pow-of-2 performs worse by z_xy on 908, 90a
            if(hasPow2)
                use_ZXY_sbrc = false;

            // sbrc_200 (dp) performs worse by z_xy on 90a
            if(is90a && has200 && isDouble)
                use_ZXY_sbrc = false;
        }
    }

    size_t total_sbrc = 0;
    for(int i = 0; i < 3; ++i)
    {
        // If we have an sbrc kernel for this length, use it,
        // otherwise, fall back to row FFT+transpose
        bool have_sbrc = pool.has_SBRC_kernel(cur_length.front(), precision);
        // ensure the kernel would be tile-aligned
        if(have_sbrc)
        {
            auto kernel = pool.get_kernel(
                FMKey(cur_length[0], precision, CS_KERNEL_STOCKHAM_BLOCK_RC, TILE_ALIGNED));

            size_t otherDim = use_ZXY_sbrc ? cur_length[1] : cur_length[2];
            if(otherDim % kernel.transforms_per_block != 0)
                have_sbrc = false;
        }
        if(have_sbrc)
        {
            ++total_sbrc;
        }

        // We are unable to do more than 2 sbrc kernels.  We'd ideally want 3 sbrc, but each needs
        // to be out-of-place:
        //
        // - kernel 1: IN/OUT -> TEMP
        // - kernel 2: TEMP -> OUT
        // - kernel 3: OUT -> ???
        //
        // So we have no way to put the results back into IN.  So
        // limit ourselves to 2 sbrc kernels in that case.
        if(total_sbrc >= 3 && placement == rocfft_placement_inplace)
        {
            have_sbrc = false;
        }

        if(have_sbrc)
        {
            auto sbrcScheme = (use_ZXY_sbrc) ? CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
                                             : CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z;
            if(!noSolution && (child_scheme_trees[childNodes.size()]->curScheme != sbrcScheme))
                throw std::runtime_error(
                    "BLOCKRC3DNode: Unexpected child scheme from solution map");
            auto sbrc_node    = NodeFactory::CreateNodeFromScheme(sbrcScheme, this);
            sbrc_node->length = cur_length;
            sbrc_node->SetTransposeOutputLength();
            childNodes.emplace_back(std::move(sbrc_node));
        }
        else
        {
            // row ffts
            NodeMetaData row_plan_data(this);
            row_plan_data.length    = cur_length;
            row_plan_data.dimension = 1;
            ComputeScheme determined_scheme
                = (noSolution) ? CS_NONE : child_scheme_trees[childNodes.size()]->curScheme;
            auto row_plan = NodeFactory::CreateExplicitNode(row_plan_data, this, determined_scheme);
            row_plan->RecursiveBuildTree(
                (noSolution) ? nullptr : child_scheme_trees[childNodes.size()].get());

            // transpose XY_Z
            auto transScheme = (use_ZXY_sbrc) ? CS_KERNEL_TRANSPOSE_Z_XY : CS_KERNEL_TRANSPOSE_XY_Z;
            if(!noSolution && (child_scheme_trees[childNodes.size() + 1]->curScheme != transScheme))
                throw std::runtime_error(
                    "BLOCKRC3DNode: Unexpected child scheme from solution map");
            auto trans_plan    = NodeFactory::CreateNodeFromScheme(transScheme, this);
            trans_plan->length = cur_length;
            trans_plan->SetTransposeOutputLength();
            if(!use_ZXY_sbrc)
                std::swap(trans_plan->length[1], trans_plan->length[2]);
            trans_plan->dimension = 2;

            // RT
            childNodes.emplace_back(std::move(row_plan));
            childNodes.emplace_back(std::move(trans_plan));
        }

        if(use_ZXY_sbrc)
        {
            std::swap(cur_length[1], cur_length[0]);
            std::swap(cur_length[2], cur_length[1]);
        }
        else
        {
            std::swap(cur_length[2], cur_length[1]);
            std::swap(cur_length[1], cur_length[0]);
        }
    }
}

void BLOCKRC3DNode::AssignParams_internal()
{
    childNodes.front()->inStride = inStride;
    childNodes.front()->iDist    = iDist;

    std::vector<size_t> prev_outStride;
    size_t              prev_oDist = 0;
    for(auto& node : childNodes)
    {
        // set initial inStride + iDist, or connect it to previous node
        if(prev_outStride.empty())
        {
            node->inStride = inStride;
            node->iDist    = iDist;
        }
        else
        {
            node->inStride = prev_outStride;
            node->iDist    = prev_oDist;
        }

        // each node is either:
        // - a fused sbrc+transpose node (for dimensions we have SBRC kernels for)
        // - a transpose node or a row FFT node (otherwise)
        switch(node->scheme)
        {
        case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->length[2]);
            node->outStride.push_back(node->outStride[1] * node->length[0]);
            node->oDist = node->outStride[2] * node->length[1];
            break;
        }
        case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
        {
            node->outStride.push_back(1);
            node->outStride.push_back(node->length[1]);
            node->outStride.push_back(node->outStride[1] * node->length[2]);
            node->oDist = node->outStride[2] * node->length[0];
            break;
        }
        case CS_KERNEL_TRANSPOSE_XY_Z:
        {
            std::swap(node->inStride[1], node->inStride[2]);
            node->outStride.push_back(node->length[1]);
            node->outStride.push_back(1);
            node->outStride.push_back(node->outStride[0] * node->length[0]);
            node->oDist = node->iDist;
            break;
        }
        case CS_KERNEL_TRANSPOSE_Z_XY:
        {
            node->outStride.push_back(node->length[1] * node->length[2]);
            node->outStride.push_back(1);
            node->outStride.push_back(node->length[1]);
            node->oDist = node->iDist;
            break;
        }
        default:
        {
            node->outStride = node->inStride;
            node->oDist     = node->iDist;
            node->AssignParams();
            break;
        }
        }
        prev_outStride = node->outStride;
        prev_oDist     = node->oDist;
        if(node->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
            std::swap(prev_outStride[0], prev_outStride[1]);
        else if(node->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
        {
            std::swap(prev_outStride[0], prev_outStride[1]);
            std::swap(prev_outStride[1], prev_outStride[2]);
        }
    }
}

/*****************************************************
 * CS_3D_BLOCK_CR  *
 *****************************************************/
void BLOCKCR3DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 3)
           || (child_scheme_trees[0]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CR)
           || (child_scheme_trees[1]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CR)
           || (child_scheme_trees[2]->curScheme != CS_KERNEL_STOCKHAM_BLOCK_CR))
        {
            throw std::runtime_error("BLOCKCR3DNode: Unexpected child scheme from solution map");
        }
    }

    // TODO: It works only for 3 SBCR children nodes for now.
    //       The final logic will be similar to what SBRC has.

    std::vector<size_t> cur_length = length;
    for(int i = 0; i < 3; ++i)
    {
        auto node = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR, this);
        node->length.push_back(cur_length[2]);
        node->length.push_back(cur_length[0] * cur_length[1]);
        childNodes.emplace_back(std::move(node));
        std::swap(cur_length[1], cur_length[2]);
        std::swap(cur_length[1], cur_length[0]);
    }
}

void BLOCKCR3DNode::AssignParams_internal()
{
    // TODO: It works only for 3 SBCR children nodes for now.
    //       The final logic will be similar to what SBRC has.

    assert(scheme == CS_3D_BLOCK_CR);

    childNodes[0]->inStride.push_back(inStride[2]);
    childNodes[0]->inStride.push_back(inStride[0]);
    childNodes[0]->iDist = iDist;

    childNodes[0]->outStride.push_back(1);
    childNodes[0]->outStride.push_back(childNodes[0]->length[0]);
    childNodes[0]->oDist = childNodes[0]->outStride[1] * childNodes[0]->length[1];

    childNodes[1]->inStride.push_back(childNodes[1]->length[1]);
    childNodes[1]->inStride.push_back(1);
    childNodes[1]->iDist = childNodes[0]->oDist;

    childNodes[1]->outStride.push_back(1);
    childNodes[1]->outStride.push_back(childNodes[1]->length[0]);
    childNodes[1]->oDist = childNodes[1]->outStride[1] * childNodes[1]->length[1];

    childNodes[2]->inStride.push_back(childNodes[2]->length[1]);
    childNodes[2]->inStride.push_back(1);
    childNodes[2]->iDist = childNodes[1]->oDist;

    childNodes[2]->outStride.push_back(outStride[0]);
    childNodes[2]->outStride.push_back(outStride[1]);
    childNodes[2]->oDist = oDist;
}

/*****************************************************
 * CS_3D_RC  *
 *****************************************************/
void RC3DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // check schemes from solution map
    ComputeScheme determined_scheme_node0 = CS_NONE;
    ComputeScheme determined_scheme_node1 = CS_NONE;
    if(!noSolution)
    {
        if((child_scheme_trees.size() != 2))
            throw std::runtime_error("RC3DNode: Unexpected child scheme from solution map");
        determined_scheme_node0 = child_scheme_trees[0]->curScheme;
        determined_scheme_node1 = child_scheme_trees[1]->curScheme;
    }

    // 2d fft
    NodeMetaData xyPlanData(this);
    xyPlanData.length.push_back(length[0]);
    xyPlanData.length.push_back(length[1]);
    xyPlanData.dimension = 2;
    xyPlanData.length.push_back(length[2]);
    for(size_t index = 3; index < length.size(); index++)
    {
        xyPlanData.length.push_back(length[index]);
    }
    auto xyPlan = NodeFactory::CreateExplicitNode(xyPlanData, this, determined_scheme_node0);
    xyPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    // z col fft
    NodeMetaData zPlanData(this);
    zPlanData.length.push_back(length[2]);
    zPlanData.dimension = 1;
    zPlanData.length.push_back(length[0]);
    zPlanData.length.push_back(length[1]);
    for(size_t index = 3; index < length.size(); index++)
    {
        zPlanData.length.push_back(length[index]);
    }
    zPlanData.outputLength = length;

    // use explicit SBCC kernel if available
    std::unique_ptr<TreeNode> zPlan;

    if(determined_scheme_node1 != CS_NONE)
    {
        zPlan = NodeFactory::CreateExplicitNode(zPlanData, this, determined_scheme_node1);
        zPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[1].get());
    }
    else
    {
        if(pool.has_SBCC_kernel(length[2], precision))
        {
            zPlan            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CC, this);
            zPlan->length    = zPlanData.length;
            zPlan->dimension = 1;
        }
        else
        {
            zPlan = NodeFactory::CreateExplicitNode(zPlanData, this);
            zPlan->RecursiveBuildTree(nullptr);
        }
    }

    // RC
    childNodes.emplace_back(std::move(xyPlan));
    childNodes.emplace_back(std::move(zPlan));
}

void RC3DNode::AssignParams_internal()
{
    auto& xyPlan = childNodes[0];
    auto& zPlan  = childNodes[1];

    xyPlan->inStride = inStride;
    xyPlan->iDist    = iDist;

    xyPlan->outStride = outStride;
    xyPlan->oDist     = oDist;

    xyPlan->AssignParams();

    zPlan->inStride.push_back(outStride[2]);
    zPlan->inStride.push_back(outStride[0]);
    zPlan->inStride.push_back(outStride[1]);
    for(size_t index = 3; index < length.size(); index++)
        zPlan->inStride.push_back(outStride[index]);

    zPlan->iDist = xyPlan->oDist;

    zPlan->outStride = zPlan->inStride;
    zPlan->oDist     = zPlan->iDist;

    zPlan->AssignParams();
}

/*****************************************************
 * CS_3D_PP  *
 *****************************************************/
size_t PP3DNode::GetPPOffDim() const
{
    auto key = PPFMKey(length[0], length[1], length[2], precision, scheme);
    if(!pool.has_function(key))
        throw std::runtime_error("GetPPOffDim failed to find a valid kernel");

    // CS_3D_PP will have two corresponding kernels in
    // the function pool, both will have the same off-dim
    // value, and at least of one them must be an SBCC PP
    auto child_scheme = CS_KERNEL_STOCKHAM_PP_BLOCK_CC;
    auto kernel       = pool.get_kernel(key, child_scheme);

    return kernel.pp_params.off_dim;
}

void PP3DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    ppOffDim = GetPPOffDim();

    switch(ppOffDim)
    {
    case 0: // work along x will be split between y and z
    {
        // y col fft + partial pass along x
        // partial pass along x + z col fft
        throw std::runtime_error(
            "PP3DNode::BuildTree_internal: partial-passes along x not currently supported");
        break;
    }
    case 1: // work along y will be split between x and z
    {
        // x row fft + partial pass along y
        // partial pass along y + z col fft

        // Create node for x row fft + partial pass(es) along y
        NodeMetaData xPartialPassPlanData(this);
        xPartialPassPlanData.length.push_back(length[0]);
        xPartialPassPlanData.length.push_back(length[1]);
        // technically 1 < dimension < 2 for x node.
        xPartialPassPlanData.dimension = 1;
        xPartialPassPlanData.length.push_back(length[2]);
        for(size_t index = 3; index < length.size(); index++)
        {
            xPartialPassPlanData.length.push_back(length[index]);
        }

        // use explicit SBRR partial-pass kernel
        std::unique_ptr<TreeNode> xPartialPassPlan;

        xPartialPassPlan         = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_PP, this);
        xPartialPassPlan->length = xPartialPassPlanData.length;
        xPartialPassPlan->dimension    = 1;
        xPartialPassPlan->ppOffDim     = ppOffDim;
        xPartialPassPlan->allowInplace = true;
        xPartialPassPlan->comments.push_back("partial-pass enabled for second dimension.");

        // Create node for partial pass(es) along y + z col fft
        NodeMetaData zPartialPassPlanData(this);
        zPartialPassPlanData.length.push_back(length[2]);
        // technically 1 < dimension < 2 for z node.
        zPartialPassPlanData.dimension = 1;
        zPartialPassPlanData.length.push_back(length[0]);
        zPartialPassPlanData.length.push_back(length[1]);
        for(size_t index = 3; index < length.size(); index++)
        {
            zPartialPassPlanData.length.push_back(length[index]);
        }
        zPartialPassPlanData.outputLength = length;

        // use explicit SBCC partial-pass kernel
        std::unique_ptr<TreeNode> zPartialPassPlan;

        zPartialPassPlan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_PP_BLOCK_CC, this);
        zPartialPassPlan->length       = zPartialPassPlanData.length;
        zPartialPassPlan->dimension    = 1;
        zPartialPassPlan->ppOffDim     = ppOffDim;
        zPartialPassPlan->allowInplace = false;
        zPartialPassPlan->comments.push_back("partial-pass enabled for second dimension.");

        childNodes.emplace_back(std::move(xPartialPassPlan));
        childNodes.emplace_back(std::move(zPartialPassPlan));

        break;
    }
    case 2: // work along z will be split between x and y
    {
        // x row fft + partial pass along z
        // partial pass along z + y col fft
        throw std::runtime_error(
            "PP3DNode::BuildTree_internal: partial-passes along z not currently supported");
        break;
    }
    default:
        throw std::runtime_error("PP3DNode::BuildTree_internal:: Unexpected ppOffDim");
    }
}

void PP3DNode::AssignParams_internal()
{
    switch(ppOffDim)
    {
    case 0: // work along x will be split between y and z
    {
        // y col fft + partial pass along x
        // partial pass along x + z col fft
        throw std::runtime_error(
            "PP3DNode::AssignParams_internal: partial-passes along x not currently supported");
        break;
    }
    case 1: // work along y will be split between x and z
    {
        // xy plan is a x row 1D-FFT + plus partial pass(es) along y
        // z plan is partial pass(es) along y + z col 1D-FFT.
        auto& xyPlan = childNodes[0];
        auto& zPlan  = childNodes[1];

        xyPlan->inStride = inStride;
        xyPlan->iDist    = iDist;

        xyPlan->outStride = outStride;
        xyPlan->oDist     = oDist;

        xyPlan->AssignParams();

        zPlan->inStride.push_back(outStride[2]);
        zPlan->inStride.push_back(outStride[0]);
        zPlan->inStride.push_back(outStride[1]);
        for(size_t index = 3; index < length.size(); index++)
            zPlan->inStride.push_back(outStride[index]);

        zPlan->iDist = xyPlan->oDist;

        zPlan->outStride = zPlan->inStride;
        zPlan->oDist     = zPlan->iDist;

        zPlan->AssignParams();
        break;
    }
    case 2: // work along z will be split between x and y
    {
        // x row fft + partial pass along z
        // partial pass along z + y col fft
        throw std::runtime_error(
            "PP3DNode::AssignParams_internal: partial-passes along z not currently supported");
        break;
    }
    default:
        throw std::runtime_error("PP3DNode::AssignParams_internal: Unexpected ppOffDim");
    }
}

// Leaf Node
/*****************************************************
 * Base Class of fused SBRC and Transpose
 *****************************************************/
FMKey SBRCTranspose3DNode::GetKernelKey() const
{
    if(specified_key)
        return *specified_key.get();

    // NB: Need to make sure that sbrcTranstype has the correct value
    if(sbrcTranstype == SBRC_TRANSPOSE_TYPE::NONE)
    {
        // find the base kernel at first
        FMKey baseKey(length[0], precision, scheme, TILE_ALIGNED);
        // if we have the base kernel, then we set the exact sbrc_trans_type and return the real key
        // if we don't, then we simply return a key with NONE sbrc_trans_type
        // which will make KernelCheck() trigger an exception
        if(pool.has_function(baseKey))
        {
            auto bwd      = pool.get_kernel(baseKey).transforms_per_block;
            sbrcTranstype = sbrc_transpose_type(bwd);
        }
    }

    return FMKey(length[0], precision, scheme, sbrcTranstype);
}

bool SBRCTranspose3DNode::KernelCheck(std::vector<FMKey>& kernel_keys)
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

void SBRCTranspose3DNode::TuneDirectRegType()
{
    // half precision has not been tested yet, disable it for now.
    if(precision == rocfft_precision_half)
    {
        dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
        return;
    }

    if(is_device_gcn_arch(deviceProp, "gfx906"))
    {
        // bad results from benchmark:
        //     {49,sp}, {128,sp},
        //     {64,dp}, {81,dp}, {100,dp} are bad
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {49, 128}}, {rocfft_precision_double, {64, 81, 100}}};
        if(length_excepted(exceptions, precision, length[0]))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    else if(is_device_gcn_arch(deviceProp, "gfx908"))
    {
        // bad results from benchmark:
        //     {81,sp}, {100,sp}, {128,sp}, {192,sp}, {200,sp}, {512,sp},
        //     {81,dp}, {512,dp} are bad
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {81, 100, 128, 192, 200, 512}},
               {rocfft_precision_double, {81, 512}}};
        if(length_excepted(exceptions, precision, length[0]))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    else if(is_device_gcn_arch(deviceProp, "gfx90a"))
    {
        // bad results from benchmark:
        //     {49,sp}, {64,sp}, {81,sp}, {125,sp}, {128,sp}, {192,sp}, {200,sp}, {512,sp},
        //     {64,dp}, {81,dp}, {100,dp}, {125,dp} are bad
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {49, 64, 81, 125, 128, 192, 200, 512}},
               {rocfft_precision_double, {64, 81, 100, 125}}};
        if(length_excepted(exceptions, precision, length[0]))
            dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
    // we don't enable the features for others
    else
    {
        dir2regMode = FORCE_OFF_OR_NOT_SUPPORT;
    }
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
 *****************************************************/
void SBRCTransXY_ZNode::SetupGridParam_internal(GridParam& gp)
{
    // sbrcTransType has already been assigned in KernelCheck();
    auto kernel = GetKernel();
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;
    lds         = length[0] * bwd;
    gp.b_x      = DivRoundingUp(length[2], bwd) * length[1] * batch;
    gp.wgs_x    = wgs;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
 *****************************************************/
void SBRCTransZ_XYNode::SetupGridParam_internal(GridParam& gp)
{
    // sbrcTransType has already been assigned in KernelCheck();
    auto kernel = GetKernel();
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;
    lds         = length[0] * bwd;
    gp.b_x      = DivRoundingUp(length[1], bwd) * length[2] * batch;
    gp.wgs_x    = wgs;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY
 *****************************************************/
void RealCmplxTransZ_XYNode::SetupGridParam_internal(GridParam& gp)
{
    // sbrcTransType has already been assigned in KernelCheck();
    auto kernel = pool.get_kernel(GetKernelKey());
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;
    // this kernel always does real-complex processing and needs one
    // extra element per row
    lds      = (length[0] + 1) * bwd;
    gp.b_x   = DivRoundingUp(length[1], bwd) * length[2] * batch;
    gp.wgs_x = wgs;
}

bool RealCmplxTransZ_XYNode::CreateDevKernelArgs()
{
    // We have a case where this 3D kernel is shoehorned into a 2D plan.
    // If so, add a third dimension when creating kernel args.
    if(length.size() == 2)
    {
        length.push_back(1);
        inStride.push_back(inStride.back());
        outStride.push_back(outStride.back());
    }
    return SBRCTranspose3DNode::CreateDevKernelArgs();
}
