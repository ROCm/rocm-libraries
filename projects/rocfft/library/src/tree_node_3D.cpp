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

#include <algorithm>
#include <array>

namespace
{
    constexpr std::array<size_t, 3> xyz{0, 1, 2};
    constexpr std::array<size_t, 3> xzy{0, 2, 1};
    constexpr std::array<size_t, 3> yxz{1, 0, 2};
    constexpr std::array<size_t, 3> yzx{1, 2, 0};
    constexpr std::array<size_t, 3> zxy{2, 0, 1};

    data_set_t in_frame(data_set_t layout, const std::array<size_t, 3>& order)
    {
        layout.reorder_axes(order, axis_scope::length_only);
        return layout;
    }

    std::array<size_t, 4> strides_3d(const data_set_t& layout)
    {
        return {layout.axis_stride(0),
                layout.axis_stride(1),
                layout.axis_stride(2),
                layout.axis_stride(3)};
    }

    // Pack in the requested physical order without changing the logical frame.
    std::array<size_t, 4> packed_3d_strides(const data_set_t&            layout,
                                            const std::array<size_t, 3>& order)
    {
        std::array<size_t, 4> strides;
        strides[order[0]] = 1;
        strides[order[1]] = layout.axis_span(order[0]);
        strides[order[2]] = strides[order[1]] * layout.axis_span(order[1]);
        strides[3]        = strides[order[2]] * layout.axis_span(order[2]);
        return strides;
    }

    auto rtrt_temp_strides(const data_set_t& layout)
    {
        // The first transpose is in the [X, Z, Y] frame, with Z packed first.
        const auto          base = packed_3d_strides(layout, yxz);
        std::vector<size_t> strides(layout.get_full_rank());
        std::copy(base.begin(), base.end(), strides.begin());
        for(size_t axis = 4; axis < strides.size(); ++axis)
            strides[axis] = strides[axis - 1] * layout.axis_span(axis - 1);
        return strides;
    }

    data_set_t fold_sbcr_plane(const data_set_t& stage, bool solutionEndpoint)
    {
        // Fold only the two transverse axes. The original outer batch stays separate.
        const auto inner = stage.axis_span(1);
        const auto outer = stage.axis_span(2);
        // A minimal solution-map key may force SBCR on a strided endpoint. Keep its
        // legacy projection through the first plane stride; packed stages must agree.
        if(!solutionEndpoint)
            assert(outer == 1 || stage.axis_stride(2) == inner * stage.axis_stride(1));

        const auto lower      = stage.lower();
        const auto upper      = stage.upper();
        const auto planeLower = lower[1] + inner * lower[2];
        const auto folded
            = data_set_t({lower[0], planeLower, lower[3]},
                         {upper[0], planeLower + inner * outer, upper[3]},
                         {stage.axis_stride(0), stage.axis_stride(1), stage.axis_stride(3)},
                         2);
        const auto offsets = stage.num_planes() == 1
                                 ? std::array<size_t, 2>{stage.get_offset(), 0}
                                 : std::array<size_t, 2>{stage.get_offset(data_part::REAL),
                                                         stage.get_offset(data_part::IMAGINARY)};
        return folded.with_element_type(stage.has_array_type() ? stage.get_array_type()
                                                               : rocfft_array_type_unset,
                                        stage.get_precision(),
                                        offsets);
    }

    void assign_rows_then_z(const TreeNode& parent, TreeNode& rows, TreeNode& z)
    {
        rows.set_strides(io_data_label::INPUT, parent.input_layout());
        rows.set_strides(io_data_label::OUTPUT, parent.output_layout());
        rows.AssignParams();

        auto strides = parent.output_layout().strides_and_distances();
        std::rotate(strides.begin(), strides.begin() + 2, strides.begin() + 3);
        strides.back() = rows.output_distance();
        z.set_strides(io_data_label::INPUT, strides);
        z.set_strides(io_data_label::OUTPUT, strides);
        z.AssignParams();
    }
}

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

    // 2D FFT: select XY and keep Z as a batch axis.
    NodeMetaData xyPlanData(input_layout().get_data_set_for_len_axes({0, 1}),
                            output_layout().get_data_set_for_len_axes({0, 1}),
                            this);
    auto xyPlan = NodeFactory::CreateExplicitNode(xyPlanData, this, determined_scheme_node0);
    xyPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    // XY_Z uses the [X, Z, Y] frame and writes a Z-first temporary.
    const auto outputXZY = in_frame(output_layout(), xzy);
    auto temp = outputXZY.with_element_type(rocfft_array_type_complex_interleaved, get_precision());
    temp.set_strides_and_distances(rtrt_temp_strides(temp));
    auto trans1Plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_XY_Z,
                                                        this,
                                                        outputXZY.get_data_set_for_len_axes({0, 1}),
                                                        temp.get_data_set_for_len_axes({0, 1}));

    // Z FFT: selecting Z gives [Z, X, Y].
    const auto   tempZ = temp.get_data_set_for_len_axes({1});
    NodeMetaData zPlanData(tempZ, tempZ, this);
    auto         zPlan = NodeFactory::CreateExplicitNode(zPlanData, this, determined_scheme_node2);
    zPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[2].get());

    // Z_XY uses [Z, X, Y], with Y remaining a batch axis.
    temp.reorder_axes(yxz, axis_scope::length_only);
    auto trans2Plan = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_TRANSPOSE_Z_XY,
        this,
        temp.get_data_set_for_len_axes({0, 1}),
        in_frame(output_layout(), zxy).get_data_set_for_len_axes({0, 1}));

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
    //    So the XY_Z transpose arranges the Z dimension to the fastest dim, and
    //    Z_XY puts it back to the expected arrangement for output.
}

void RTRT3DNode::AssignParams_internal()
{
    assert(childNodes.size() == 4);

    // B -> B
    auto& xyPlan = childNodes[0];
    xyPlan->set_strides(io_data_label::INPUT, input_layout());
    auto outputStrides = output_layout().strides_and_distances();
    xyPlan->set_strides(io_data_label::OUTPUT, outputStrides);

    xyPlan->AssignParams();

    // B -> T
    auto& trans1Plan     = childNodes[1];
    auto  transposeInput = xyPlan->output_layout().strides_and_distances();
    std::swap(transposeInput[1], transposeInput[2]);
    trans1Plan->set_strides(io_data_label::INPUT, transposeInput);
    auto tempStrides = rtrt_temp_strides(trans1Plan->output_layout());
    trans1Plan->set_strides(io_data_label::OUTPUT, tempStrides);

    // T -> T
    auto& zPlan = childNodes[2];
    std::swap(tempStrides[0], tempStrides[1]);
    zPlan->set_strides(io_data_label::INPUT, tempStrides);
    zPlan->set_strides(io_data_label::OUTPUT, tempStrides);

    zPlan->AssignParams();

    // T -> B
    auto& trans2Plan = childNodes[3];
    trans2Plan->set_strides(io_data_label::INPUT, zPlan->output_layout());
    std::rotate(outputStrides.begin(), outputStrides.begin() + 2, outputStrides.begin() + 3);
    trans2Plan->set_strides(io_data_label::OUTPUT, outputStrides);
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

    auto stage = input_layout();

    for(int i = 0; i < 6; i += 2)
    {
        // Transpose in the current frame and rebase the packed temporary for the row FFT.
        auto temp = stage.with_element_type(rocfft_array_type_complex_interleaved, get_precision());
        temp.set_strides_and_distances(packed_3d_strides(temp, yzx));
        auto trans_plan = NodeFactory::CreateNodeFromScheme(CS_KERNEL_TRANSPOSE_Z_XY,
                                                            this,
                                                            stage.get_data_set_for_len_axes({0, 1}),
                                                            temp.get_data_set_for_len_axes({0, 1}));

        temp.reorder_axes(yzx, axis_scope::length_only);
        const auto&  rowOutput = i == 4 ? output_layout() : temp;
        NodeMetaData row_plan_data(
            temp.get_data_set_for_len_axes({0}), rowOutput.get_data_set_for_len_axes({0}), this);
        ComputeScheme determined_scheme
            = (noSolution) ? CS_NONE : child_scheme_trees[i + 1]->curScheme;
        auto row_plan = NodeFactory::CreateExplicitNode(row_plan_data, this, determined_scheme);
        row_plan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[i + 1].get());

        // TR
        childNodes.emplace_back(std::move(trans_plan));
        childNodes.emplace_back(std::move(row_plan));
        stage = std::move(temp);
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

    const auto* stage = &input_layout();
    for(int i = 0; i < 6; i += 2)
    {
        auto& trans_plan = childNodes[i];
        trans_plan->set_strides(io_data_label::INPUT, *stage);
        const auto temp = packed_3d_strides(trans_plan->output_layout(), yzx);
        trans_plan->set_strides(io_data_label::OUTPUT, temp);

        auto&                       row_plan = childNodes[i + 1];
        const std::array<size_t, 4> rowStrides{temp[1], temp[2], temp[0], temp[3]};
        row_plan->set_strides(io_data_label::INPUT, rowStrides);
        row_plan->set_strides(io_data_label::OUTPUT,
                              i == 4 ? strides_3d(output_layout()) : rowStrides);
        row_plan->AssignParams();
        stage = &row_plan->output_layout();
    }
}

/*****************************************************
 * CS_3D_BLOCK_RC  *
 *****************************************************/
void BLOCKRC3DNode::BuildTree_internal(SchemeTreeVec& child_scheme_trees)
{
    bool noSolution = child_scheme_trees.empty();

    // NB:
    //   The idea is to change the SBRC from doing XZ-plane to doing XY-plane
    //   Some problems are still faster with 3D_RC so they never go here
    bool       is906         = is_device_gcn_arch(deviceProp, "gfx906");
    bool       is908         = is_device_gcn_arch(deviceProp, "gfx908");
    bool       is90a         = is_device_gcn_arch(deviceProp, "gfx90a");
    const auto nodePrecision = get_precision();
    bool       isDouble      = (nodePrecision == rocfft_precision_double);

    // specific flags for lengths
    const auto x         = input_length(0);
    const auto y         = input_length(1);
    const auto z         = input_length(2);
    bool       has50     = (x == 50 || y == 50 || z == 50);
    bool       has64     = (x == 64 || y == 64 || z == 64);
    bool       has100    = (x == 100 || y == 100 || z == 100);
    bool       has200    = (x == 200 || y == 200 || z == 200);
    bool       hasPow2   = (IsPo2(x) || IsPo2(y) || IsPo2(z));
    bool isDiagonalTrans = is_diagonal_sbrc_3D_length(x) && data_rank() == 3 && x == y && y == z;

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

    auto        stage      = input_layout();
    const auto& nextFrame  = use_ZXY_sbrc ? yzx : zxy;
    const auto& lastFrame  = use_ZXY_sbrc ? zxy : yzx;
    size_t      total_sbrc = 0;
    for(int i = 0; i < 3; ++i)
    {
        // If we have an sbrc kernel for this length, use it,
        // otherwise, fall back to row FFT+transpose
        const auto len       = stage.axis_span(0);
        bool       have_sbrc = pool.has_SBRC_kernel(len, nodePrecision);
        // ensure the kernel would be tile-aligned
        if(have_sbrc)
        {
            auto kernel = pool.get_kernel(
                FMKey(len, nodePrecision, CS_KERNEL_STOCKHAM_BLOCK_RC, TILE_ALIGNED));

            size_t otherDim = stage.axis_span(use_ZXY_sbrc ? 1 : 2);
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

        // The endpoint has the parent's output range and type, but this scheme still
        // assigns packed strides in the current frame, even on the last transpose.
        auto next   = i == 2 ? in_frame(output_layout(), lastFrame)
                             : stage.with_element_type(rocfft_array_type_complex_interleaved,
                                                     nodePrecision);
        auto packed = packed_3d_strides(next, nextFrame);
        if(!have_sbrc)
            packed.back() = stage.axis_stride(3);
        next.set_strides_and_distances(packed);

        if(have_sbrc)
        {
            auto sbrcScheme = (use_ZXY_sbrc) ? CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
                                             : CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z;
            if(!noSolution && (child_scheme_trees[childNodes.size()]->curScheme != sbrcScheme))
                throw std::runtime_error(
                    "BLOCKRC3DNode: Unexpected child scheme from solution map");
            auto sbrc_node = NodeFactory::CreateNodeFromScheme(sbrcScheme,
                                                               this,
                                                               stage.get_data_set_for_len_axes({0}),
                                                               next.get_data_set_for_len_axes({0}));
            childNodes.emplace_back(std::move(sbrc_node));
        }
        else
        {
            // The fallback row stays in the same frame and layout.
            const auto    rows = stage.get_data_set_for_len_axes({0});
            NodeMetaData  row_plan_data(rows, rows, this);
            ComputeScheme determined_scheme
                = (noSolution) ? CS_NONE : child_scheme_trees[childNodes.size()]->curScheme;
            auto row_plan = NodeFactory::CreateExplicitNode(row_plan_data, this, determined_scheme);
            row_plan->RecursiveBuildTree(
                (noSolution) ? nullptr : child_scheme_trees[childNodes.size()].get());

            // The standalone XY_Z uses [X, Z, Y] and has transform rank two.
            auto transScheme = (use_ZXY_sbrc) ? CS_KERNEL_TRANSPOSE_Z_XY : CS_KERNEL_TRANSPOSE_XY_Z;
            if(!noSolution && (child_scheme_trees[childNodes.size() + 1]->curScheme != transScheme))
                throw std::runtime_error(
                    "BLOCKRC3DNode: Unexpected child scheme from solution map");
            auto trans_plan = NodeFactory::CreateNodeFromScheme(
                transScheme,
                this,
                use_ZXY_sbrc ? stage.get_data_set_for_len_axes({0, 1})
                             : in_frame(stage, xzy).get_data_set_for_len_axes({0, 1}),
                use_ZXY_sbrc ? next.get_data_set_for_len_axes({0, 1})
                             : in_frame(next, xzy).get_data_set_for_len_axes({0, 1}));

            // RT
            childNodes.emplace_back(std::move(row_plan));
            childNodes.emplace_back(std::move(trans_plan));
        }

        // Rebase for the next partition without changing the producer's descriptors.
        if(i != 2)
        {
            next.reorder_axes(nextFrame, axis_scope::length_only);
            stage = std::move(next);
        }
    }
}

void BLOCKRC3DNode::AssignParams_internal()
{
    auto stage = strides_3d(input_layout());
    for(auto& node : childNodes)
    {
        const std::array<size_t, 3>* order;
        bool                         keepDistance = false;
        switch(node->scheme)
        {
        case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
            order = &zxy;
            break;
        case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
            order = &yzx;
            break;
        case CS_KERNEL_TRANSPOSE_XY_Z:
            std::swap(stage[1], stage[2]);
            order        = &yxz;
            keepDistance = true;
            break;
        case CS_KERNEL_TRANSPOSE_Z_XY:
            order        = &yzx;
            keepDistance = true;
            break;
        default:
            node->set_strides(io_data_label::INPUT, stage);
            node->set_strides(io_data_label::OUTPUT, stage);
            node->AssignParams();
            stage = strides_3d(node->output_layout());
            continue;
        }

        node->set_strides(io_data_label::INPUT, stage);
        auto packed = packed_3d_strides(node->output_layout(), *order);
        if(keepDistance)
            packed.back() = stage.back();
        node->set_strides(io_data_label::OUTPUT, packed);
        stage = {packed[(*order)[0]], packed[(*order)[1]], packed[(*order)[2]], packed[3]};
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

    // In order, transform Z, Y, then X, packing the transformed axis first.
    auto stage = in_frame(input_layout(), zxy);
    for(int i = 0; i < 3; ++i)
    {
        auto next = i == 2 ? output_layout()
                           : stage.with_element_type(rocfft_array_type_complex_interleaved,
                                                     get_precision());
        if(i != 2)
            next.set_strides_and_distances(packed_3d_strides(next, xyz));
        childNodes.emplace_back(
            NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_BLOCK_CR,
                                              this,
                                              fold_sbcr_plane(stage, !noSolution && i == 0),
                                              fold_sbcr_plane(next, !noSolution && i == 2)));
        if(i != 2)
        {
            next.reorder_axes(zxy, axis_scope::length_only);
            stage = std::move(next);
        }
    }
}

void BLOCKCR3DNode::AssignParams_internal()
{
    assert(scheme == CS_3D_BLOCK_CR);

    std::array<size_t, 3> inputStrides{input_stride(2), input_stride(0), input_distance()};
    for(size_t i = 0; i < childNodes.size(); ++i)
    {
        auto& node = childNodes[i];
        node->set_strides(io_data_label::INPUT, inputStrides);
        if(i == 2)
        {
            node->set_strides(io_data_label::OUTPUT,
                              std::array{output_stride(0), output_stride(1), output_distance()});
        }
        else
        {
            const auto s = node->output_length(0);
            const auto d = s * node->output_length(1);
            node->set_strides(io_data_label::OUTPUT, std::array<size_t, 3>{1, s, d});
            inputStrides = {childNodes[i + 1]->input_length(1), 1, d};
        }
    }
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

    // 2D FFT: select XY, leaving Z as a leading batch axis.
    NodeMetaData xyPlanData(input_layout().get_data_set_for_len_axes({0, 1}),
                            output_layout().get_data_set_for_len_axes({0, 1}),
                            this);
    auto xyPlan = NodeFactory::CreateExplicitNode(xyPlanData, this, determined_scheme_node0);
    xyPlan->RecursiveBuildTree((noSolution) ? nullptr : child_scheme_trees[0].get());

    // Z columns view the row result in [Z, X, Y] order.
    const auto                columns = output_layout().get_data_set_for_len_axes({2});
    std::unique_ptr<TreeNode> zPlan;
    if(determined_scheme_node1 == CS_NONE && pool.has_SBCC_kernel(input_length(2), get_precision()))
    {
        zPlan = NodeFactory::CreateNodeFromScheme(
            CS_KERNEL_STOCKHAM_BLOCK_CC, this, columns, columns);
    }
    else
    {
        NodeMetaData zPlanData(columns, columns, this);
        zPlan = NodeFactory::CreateExplicitNode(zPlanData, this, determined_scheme_node1);
        zPlan->legacyOutputOrder = OutputAxisOrder::ROTATE_LEFT_3;
        zPlan->RecursiveBuildTree(determined_scheme_node1 == CS_NONE ? nullptr
                                                                     : child_scheme_trees[1].get());
    }

    // RC
    childNodes.emplace_back(std::move(xyPlan));
    childNodes.emplace_back(std::move(zPlan));
}

void RC3DNode::AssignParams_internal()
{
    assign_rows_then_z(*this, *childNodes[0], *childNodes[1]);
}

/*****************************************************
 * CS_3D_PP  *
 *****************************************************/
size_t PP3DNode::GetPPOffDim() const
{
    auto key = PPFMKey(input_length(0),
                       input_length(1),
                       input_length(2),
                       get_precision(),
                       GetRootPlanTransformType(),
                       scheme);
    if(!pool.has_function(key, batch_count()))
        throw std::runtime_error("GetPPOffDim failed to find a valid kernel");

    // CS_3D_PP will have two corresponding kernels in
    // the function pool, both will have the same off-dim
    // value, and at least of one them must be an SBCC PP
    auto child_scheme = CS_KERNEL_STOCKHAM_PP_BLOCK_CC;
    auto kernel       = pool.get_kernel(key, child_scheme, batch_count());

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

        // Rank-one FFT along X, with Y and Z kept separate for the partial passes.
        auto xPartialPassPlan
            = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_PP,
                                                this,
                                                input_layout().get_data_set_for_len_axes({0}),
                                                output_layout().get_data_set_for_len_axes({0}));
        xPartialPassPlan->ppOffDim     = ppOffDim;
        xPartialPassPlan->allowInplace = true;
        xPartialPassPlan->comments.push_back("partial-pass enabled for second dimension.");

        // Rank-one FFT along Z takes the [Z, X, Y] view of the row result.
        const auto columns          = output_layout().get_data_set_for_len_axes({2});
        auto       zPartialPassPlan = NodeFactory::CreateNodeFromScheme(
            CS_KERNEL_STOCKHAM_PP_BLOCK_CC, this, columns, columns);
        // ppOffDim names the root axis (Y), not its index in the child's view.
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
        assign_rows_then_z(*this, *childNodes[0], *childNodes[1]);
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
        FMKey baseKey(input_length(0), get_precision(), scheme, TILE_ALIGNED);
        // if we have the base kernel, then we set the exact sbrc_trans_type and return the real key
        // if we don't, then we simply return a key with NONE sbrc_trans_type
        // which will make KernelCheck() trigger an exception
        if(pool.has_function(baseKey))
        {
            auto bwd      = pool.get_kernel(baseKey).transforms_per_block;
            sbrcTranstype = sbrc_transpose_type(bwd);
        }
    }

    return FMKey(input_length(0), get_precision(), scheme, sbrcTranstype);
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
    const auto nodePrecision = get_precision();
    if(nodePrecision == rocfft_precision_half)
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
        if(length_excepted(exceptions, nodePrecision, input_length(0)))
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
        if(length_excepted(exceptions, nodePrecision, input_length(0)))
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
        if(length_excepted(exceptions, nodePrecision, input_length(0)))
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
    const auto& view = GetKernelView();
    const auto  lens = view.lengths();
    // sbrcTransType has already been assigned in KernelCheck();
    auto kernel = GetKernel();
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;
    lds         = lens[0] * bwd;
    gp.b_x      = DivRoundingUp(lens[2], bwd) * lens[1] * view.batch;
    gp.wgs_x    = wgs;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
 *****************************************************/
void SBRCTransZ_XYNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view = GetKernelView();
    const auto  lens = view.lengths();
    // sbrcTransType has already been assigned in KernelCheck();
    auto kernel = GetKernel();
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;
    lds         = lens[0] * bwd;
    gp.b_x      = DivRoundingUp(lens[1], bwd) * lens[2] * view.batch;
    gp.wgs_x    = wgs;
}

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY
 *****************************************************/
void RealCmplxTransZ_XYNode::SetupGridParam_internal(GridParam& gp)
{
    const auto& view = GetKernelView();
    const auto  lens = view.lengths();
    // sbrcTransType has already been assigned in KernelCheck();
    auto kernel = pool.get_kernel(GetKernelKey());
    bwd         = kernel.transforms_per_block;
    wgs         = kernel.workgroup_size;
    // this kernel always does real-complex processing and needs one
    // extra element per row
    lds      = (lens[0] + 1) * bwd;
    gp.b_x   = DivRoundingUp(lens[1], bwd) * lens[2] * view.batch;
    gp.wgs_x = wgs;
}
