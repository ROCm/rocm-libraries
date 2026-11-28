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

#include "fuse_shim.h"
#include "../../shared/arithmetic.h"
#include "../../shared/array_predicate.h"
#include "function_pool.h"
#include "node_factory.h"

#include <array>
#include <numeric>

namespace
{
    constexpr std::array<size_t, 2> swap_01{1, 0};
    constexpr std::array<size_t, 3> swap_01_3d{1, 0, 2};
    constexpr std::array<size_t, 3> swap_12_3d{0, 2, 1};
    constexpr std::array<size_t, 3> rotate_left_3{1, 2, 0};
    constexpr std::array<size_t, 3> rotate_right_3{2, 0, 1};

    // Fusion takes the buffer's type but retains the offsets of the copied node.
    data_set_t
        with_edge_type(const data_set_t& shape, rocfft_array_type type, const data_set_t& original)
    {
        const bool planar = original.num_planes() == 2;
        const auto first  = original.get_offset(planar ? data_part::REAL : data_part::ALL);
        const auto second = array_type_is_planar(type)
                                ? (planar ? original.get_offset(data_part::IMAGINARY) : first)
                                : 0;
        return shape.with_element_type(type, original.get_precision(), {first, second});
    }

    // Map the leading stride slots explicitly where the fused kernel uses a
    // different order. Some variants retain the copied node's higher strides.
    void set_edge_strides(data_set_t&             layout,
                          const data_set_t&       edge,
                          std::span<const size_t> order               = {},
                          bool                    keep_higher_strides = false)
    {
        const auto distance_axis = layout.get_full_rank() - 1;
        const auto count         = keep_higher_strides ? order.size() : distance_axis;
        for(size_t axis = 0; axis < count; ++axis)
            layout.set_axis_stride(axis,
                                   edge.axis_stride(axis < order.size() ? order[axis] : axis));
        layout.set_axis_stride(distance_axis, edge.axis_stride(edge.get_full_rank() - 1));
    }

    void reset_real_even_output(TreeNode& parent, const TreeNode& fused)
    {
        // Keep the parent's rank, including when the fused SBRC uses a 3D
        // kernel for a 2D plan.
        for(size_t axis = 0; axis < parent.data_rank(); ++axis)
            parent.set_axis_stride(io_data_label::OUTPUT,
                                   axis,
                                   axis < fused.data_rank() ? fused.output_stride(axis) : 0);
        parent.set_axis_stride(io_data_label::OUTPUT, parent.data_rank(), fused.output_distance());
        parent.set_array_type(io_data_label::OUTPUT, fused.output_array_type());
    }
}

bool canOptimizeWithStride(TreeNode* stockham)
{
    // for 3D pow2 sizes, manipulating strides looks like it loses to
    // diagonal transpose
    if(IsPo2(stockham->input_length(0)) && stockham->data_rank() >= 3)
        return false;
    size_t numTrans = stockham->pool.get_kernel(stockham->GetKernelKey()).transforms_per_block;

    // ensure we are doing enough rows to coalesce properly. 4
    // seems to be enough for double-precision, whereas some
    // sizes that do 7 rows seem to be slower for single.
    // TODO: the threshold may be set dependent one what kind of transport is the fused kernel
    //   eg. different value for TRANSPOSE, Z_XY, and XY_Z...
    //   for example, 21504 -t 1 --precision double works quite good with minRows==2
    size_t minRows = (stockham->get_precision() == rocfft_precision_single
                      || stockham->get_precision() == rocfft_precision_half)
                         ? 8
                         : 4;
    return numTrans >= minRows;
}

// if the in/out buffer meets the placement requirement
// firstOBuf is used in R2CTrans only
bool FuseShim::PlacementFusable(OperatingBuffer iBuf,
                                OperatingBuffer firstOBuf,
                                OperatingBuffer lastOBuf)
{
    // when allowInplace, both in-place and out-of-place are allowed.
    // otherwise, only out-of-place is allowed.
    return (allowInplace) ? true : iBuf != lastOBuf;
}

// return the check result of if these schemes can be fused
bool FuseShim::IsSchemeFusable() const
{
    return schemeFusable;
}

void FuseShim::OverwriteFusableFlag(bool fusable)
{
    schemeFusable = fusable;
}

TreeNode* FuseShim::FirstFuseNode() const
{
    if(nodes.size() <= firstFusedNode)
        throw std::runtime_error("firstFusedNode exceeds vector size");
    return nodes[firstFusedNode];
}

TreeNode* FuseShim::LastFuseNode() const
{
    if(nodes.size() <= lastFusedNode)
        throw std::runtime_error("lastFusedNode exceeds vector size");
    return nodes[lastFusedNode];
}

void FuseShim::ForEachNode(std::function<void(TreeNode*)> func)
{
    for(size_t i = firstFusedNode; i <= lastFusedNode; ++i)
    {
        func(nodes[i]);
    }
}

/*****************************************************
 * TR= transpose + FFT
 *
 * if we have a transpose followed by a stockham
 * fft that does multiple rows in one kernel, adjust input
 * strides to replace the transpose.  Multiple rows will ensure
 * that the transposed column reads are coalesced.
 *****************************************************/
bool TRFuseShim::CheckSchemeFusable()
{
    auto transpose = nodes[0];
    auto stockham  = nodes[1];

    // NB: can't get rid of a transpose that also does large twiddle multiplication
    if((transpose->scheme != CS_KERNEL_TRANSPOSE && transpose->scheme != CS_KERNEL_TRANSPOSE_Z_XY
        && transpose->scheme != CS_KERNEL_TRANSPOSE_XY_Z)
       || transpose->large1D)
        return false;
    if(stockham->scheme != CS_KERNEL_STOCKHAM)
        return false;

    if(!canOptimizeWithStride(stockham))
        return false;

    // verify that the transpose output lengths match the FFT input lengths
    const auto rank = transpose->data_rank();
    if(stockham->data_rank() != rank)
        return false;
    const std::span<const size_t> order
        = transpose->scheme == CS_KERNEL_TRANSPOSE        ? std::span<const size_t>(swap_01)
          : transpose->scheme == CS_KERNEL_TRANSPOSE_Z_XY ? std::span<const size_t>(rotate_left_3)
                                                          : std::span<const size_t>(rotate_right_3);
    for(size_t axis = 0; axis < rank; ++axis)
        if(stockham->input_length(axis)
           != transpose->input_length(axis < order.size() ? order[axis] : axis))
            return false;

    firstFusedNode = 0;
    lastFusedNode  = 1;

    return true;
}

std::unique_ptr<TreeNode> TRFuseShim::FuseKernels()
{
    auto transpose = nodes[0];
    auto stockham  = nodes[1];

    if(!PlacementFusable(transpose->obIn, transpose->obOut, stockham->obOut))
        return nullptr;

    auto input = with_edge_type(
        stockham->input_layout(), transpose->input_array_type(), stockham->input_layout());
    auto        output = stockham->output_layout();
    const auto& edge   = transpose->input_layout();
    if(transpose->scheme == CS_KERNEL_TRANSPOSE)
        set_edge_strides(input, edge, swap_01);
    else if(transpose->scheme == CS_KERNEL_TRANSPOSE_Z_XY)
    {
        // The fused Stockham kernel traverses the two inner batch axes in
        // the opposite order, on both sides.
        std::array<size_t, KERN_ARGS_ARRAY_WIDTH> axes;
        auto order = std::span(axes).first(input.get_batch_rank());
        std::iota(order.begin(), order.end(), 0);
        std::swap(order[0], order[1]);
        input.reorder_axes(order, axis_scope::batch_only);
        output.reorder_axes(order, axis_scope::batch_only);
        set_edge_strides(input, edge, swap_01_3d, true);
    }
    else
        set_edge_strides(input, edge, rotate_right_3, true);

    auto fused = NodeFactory::CreateNodeFromScheme(
        stockham->scheme, stockham->parent, std::move(input), std::move(output));
    fused->CopyNodeAttributes(*stockham);
    // actually no need to check kernel exists, we already have the kernel with the same length/scheme
    std::vector<FMKey> kernelkey = {stockham->GetKernelKey()};
    if(!fused->KernelCheck(kernelkey))
        return nullptr;

    fused->placement = rocfft_placement_notinplace;
    fused->obIn      = transpose->obIn;
    fused->comments.push_back("TRFuseShim: fused " + PrintScheme(transpose->scheme)
                              + " and following " + PrintScheme(stockham->scheme));

    return fused;
}

/*****************************************************
 * RT= FFT + transpose
 *
 * If this is a stockham fft that does multiple rows in one
 * kernel, followed by a transpose, adjust output strides to
 * replace the transpose.  Multiple rows will ensure that the
 * transposed column writes are coalesced.
 *****************************************************/
bool RTFuseShim::CheckSchemeFusable()
{
    auto stockham  = nodes[0];
    auto transpose = nodes[1];

    if(stockham->scheme != CS_KERNEL_STOCKHAM)
        return false;

    if((transpose->scheme != CS_KERNEL_TRANSPOSE && transpose->scheme != CS_KERNEL_TRANSPOSE_Z_XY
        && transpose->scheme != CS_KERNEL_TRANSPOSE_XY_Z)
       || transpose->data_rank() != stockham->data_rank())
        return false;

    // Both legacy stride vectors were empty here, before AssignParams.
    // Placeholder strides from descriptor construction are not a fusion test.
    for(size_t axis = 0; axis < stockham->data_rank(); ++axis)
        if(transpose->input_length(axis) != stockham->input_length(axis))
            return false;

    // NB: Same as TR, can't get rid of a transpose that also does large twiddle multiplication
    if(transpose->large1D)
        return false;

    if(!canOptimizeWithStride(stockham))
        return false;

    firstFusedNode = 0;
    lastFusedNode  = 1;

    return true;
}

std::unique_ptr<TreeNode> RTFuseShim::FuseKernels()
{
    auto stockham  = nodes[0];
    auto transpose = nodes[1];

    if(!PlacementFusable(stockham->obIn, stockham->obOut, transpose->obOut))
        return nullptr;

    // The transpose has the same spans as the Stockham node, but the
    // fused Stockham still has only its original transform axes.
    auto output = with_edge_type(
        stockham->output_layout(), transpose->output_array_type(), stockham->output_layout());
    if(transpose->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
        set_edge_strides(output, transpose->output_layout(), rotate_left_3, true);
    else
        set_edge_strides(output, transpose->output_layout());

    auto fused = NodeFactory::CreateNodeFromScheme(
        stockham->scheme, stockham->parent, stockham->input_layout(), std::move(output));
    fused->CopyNodeAttributes(*stockham);
    // actually no need to check kernel exists, we already have the kernel with the same length/scheme
    std::vector<FMKey> kernelkey = {stockham->GetKernelKey()};
    if(!fused->KernelCheck(kernelkey))
        return nullptr;

    fused->placement = rocfft_placement_notinplace;
    fused->obOut     = transpose->obOut;
    fused->comments.push_back("RTFuseShim: fused " + PrintScheme(stockham->scheme)
                              + " and following " + PrintScheme(transpose->scheme));

    fused->legacyOutputOrder = transpose->legacyOutputOrder;

    return fused;
}

/*****************************************************
 * RT_ZXY FFT + transpose_Z_XY
 *****************************************************/
bool RT_ZXY_FuseShim::CheckSchemeFusable()
{
    auto previous = nodes[0]; // possible nullptr
    auto stockham = nodes[1];
    auto transZXY = nodes[2];

    if(stockham->scheme != CS_KERNEL_STOCKHAM)
        return false;
    if(transZXY->scheme != CS_KERNEL_TRANSPOSE_Z_XY)
        return false;

    if(previous && previous->scheme == CS_KERNEL_TRANSPOSE_XY_Z)
        return false;

    if(!transZXY->fuse_CS_KERNEL_TRANSPOSE_Z_XY())
        return false;

    firstFusedNode = 1; // not nodes[0] (not nodes.front())
    lastFusedNode  = 2;

    return true;
}

std::unique_ptr<TreeNode> RT_ZXY_FuseShim::FuseKernels()
{
    auto stockham  = nodes[1];
    auto transpose = nodes[2];

    if(!PlacementFusable(stockham->obIn, stockham->obOut, transpose->obOut))
        return nullptr;

    auto output = with_edge_type(
        stockham->input_layout(), transpose->output_array_type(), stockham->output_layout());
    set_edge_strides(output, transpose->output_layout());
    auto fused = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY,
                                                   stockham->parent,
                                                   stockham->input_layout(),
                                                   std::move(output));
    fused->CopyNodeAttributes(*stockham);
    // check if kernel exists, since the fused kernel uses different scheme other than stockham
    if(!fused->KernelCheck())
        return nullptr;

    fused->placement = rocfft_placement_notinplace;
    fused->obOut     = transpose->obOut;
    fused->comments.push_back("RT_ZXY_FuseShim: fused " + PrintScheme(CS_KERNEL_STOCKHAM)
                              + " and following " + PrintScheme(CS_KERNEL_TRANSPOSE_Z_XY));

    return fused;
}

/*****************************************************
 * RT_XYZ FFT + transpose_XY_Z
 *
 * combine one CS_KERNEL_STOCKHAM and following CS_KERNEL_TRANSPOSE_XY_Z in 3D complex to real
 *
 * NB: this should be replaced by combining
 * CS_KERNEL_TRANSPOSE_XY_Z and the following CS_KERNEL_STOCKHAM eventually,
 * in which we might fuse 2 pairs of TR.
 *****************************************************/
bool RT_XYZ_FuseShim::CheckSchemeFusable()
{
    auto stockham = nodes[0];
    auto transXYZ = nodes[1];
    auto last     = nodes[2]; // fusable only if a stockham, but this won't be fused

    // fusable when [stockham -> Trans_XY_Z] (fusion) -> stockham
    if(stockham->scheme != CS_KERNEL_STOCKHAM)
        return false;
    if(transXYZ->scheme != CS_KERNEL_TRANSPOSE_XY_Z)
        return false;
    if(!last || last->scheme != CS_KERNEL_STOCKHAM)
        return false;

    if(!transXYZ->fuse_CS_KERNEL_TRANSPOSE_XY_Z())
        return false;

    firstFusedNode = 0;
    lastFusedNode  = 1; // not nodes[2] (not nodes.back())

    return true;
}

std::unique_ptr<TreeNode> RT_XYZ_FuseShim::FuseKernels()
{
    auto stockham = nodes[0];
    auto transXYZ = nodes[1];

    if(!PlacementFusable(stockham->obIn, stockham->obOut, transXYZ->obOut))
        return nullptr;

    auto output = with_edge_type(
        stockham->input_layout(), transXYZ->output_array_type(), stockham->output_layout());
    // Map the standalone transpose's output strides back to the Stockham axes.
    set_edge_strides(output, transXYZ->output_layout(), rotate_left_3);
    auto fused = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
                                                   stockham->parent,
                                                   stockham->input_layout(),
                                                   std::move(output));
    fused->CopyNodeAttributes(*stockham);
    // check if kernel exists, since the fused kernel uses different scheme other than stockham
    if(!fused->KernelCheck())
        return nullptr;

    fused->placement = rocfft_placement_notinplace;
    fused->obOut     = transXYZ->obOut;
    fused->comments.push_back("RT_XYZ_FuseShim: fused " + PrintScheme(CS_KERNEL_STOCKHAM)
                              + " and following " + PrintScheme(CS_KERNEL_TRANSPOSE_XY_Z));
    return fused;
}

/*****************************************************
 * R2C_Trans = REAL_2_CMPLX + TRANSPOSE*
 *****************************************************/
bool R2CTrans_FuseShim::CheckSchemeFusable()
{
    auto realEven  = nodes[0];
    auto transpose = nodes[1];
    auto following = nodes[2];

    // check if the second is transpose
    if(transpose->scheme != CS_KERNEL_TRANSPOSE && transpose->scheme != CS_KERNEL_TRANSPOSE_Z_XY)
        return false;

    // fusable when first is realEven, and the last child of it is r2c
    if(realEven->scheme != CS_REAL_TRANSFORM_EVEN)
        return false;
    if(realEven->childNodes.back()->scheme != CS_KERNEL_R_TO_CMPLX)
        return false;

    // update the node to the final effective node
    nodes[0] = realEven->childNodes.back().get(); // the last child of real-trans-even
    nodes[2] = following->GetFirstLeaf();

    firstFusedNode = 0;
    lastFusedNode  = 1;

    // if the nextLeafNode is stockham or SBCC,
    //   we allow the EffectivePlacement of (r2c-in, trans-out) to be inplace,
    //   then we force it to be OP, but change the nextLeafNode's input..
    // So if the nextLeafNode isn't one of these (ex, a transpose for TRTRT)
    //   then we couldn't change transpose's input buffer
    ComputeScheme nextFFTScheme = nodes[2]->scheme;
    if(nextFFTScheme == CS_KERNEL_STOCKHAM || nextFFTScheme == CS_KERNEL_STOCKHAM_BLOCK_CC)
        allowInplace = true;
    else
        allowInplace = false;

    return true;
}

bool R2CTrans_FuseShim::PlacementFusable(OperatingBuffer iBuf,
                                         OperatingBuffer firstOBuf,
                                         OperatingBuffer lastOBuf)
{
    // allow inplace (out-of-place is allowed as well)
    if(allowInplace)
    {
        // if inBuf and outBuf is already op, then it is safe.
        return true;
    }
    // only allow out-of-place
    else
    {
        return iBuf != lastOBuf;
    }
}

std::unique_ptr<TreeNode> R2CTrans_FuseShim::FuseKernels()
{
    auto r2c       = nodes[0];
    auto transpose = nodes[1];
    auto nextLeaf  = nodes[2];

    if(!PlacementFusable(r2c->obIn, r2c->obOut, transpose->obOut))
        return nullptr;

    // Preserve the transpose rank, with the real post-kernel's complex input span.
    auto input = with_edge_type(transpose->input_layout().with_axis_span(0, r2c->input_length(0)),
                                r2c->input_array_type(),
                                r2c->input_layout());
    set_edge_strides(input, r2c->input_layout());
    auto output = with_edge_type(
        transpose->output_layout(), transpose->output_array_type(), r2c->output_layout());
    auto fused = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_R_TO_CMPLX_TRANSPOSE, r2c->parent, std::move(input), std::move(output));
    fused->CopyNodeAttributes(*r2c);
    // no need to check kernel exists, this scheme uses a built-in kernel
    fused->placement = rocfft_placement_notinplace;
    fused->comments.push_back("R2CTrans_FuseShim: fused " + PrintScheme(CS_KERNEL_R_TO_CMPLX)
                              + " and following " + PrintScheme(transpose->scheme));

    // if the effective placement is:
    //   1. out-of-place, then we simply combine them to fused kernel, result in OP
    //   2. inplace, we keep the fused-out as trans-out, also result in OP
    //      in this case, we need to change the nextLeaf's input from c2r-out to trans-out
    if(r2c->obIn != transpose->obOut)
    {
        fused->obOut = transpose->obOut;
    }
    else
    {
        nextLeaf->obIn = fused->obOut; // so we still don't break buffer flow of [r2c-trans]-next
        nextLeaf->placement = (nextLeaf->obIn == nextLeaf->obOut) ? rocfft_placement_inplace
                                                                  : rocfft_placement_notinplace;
    }

    // The fusion crosses subtrees. The real-even parent and the fused
    // kernel share the same axis order, but write with the transpose strides.
    reset_real_even_output(*r2c->parent, *fused);

    return fused;
}

/*****************************************************
 * Trans_C2R = TRANSPOSE + CMPLX_2_REAL
 *****************************************************/
bool TransC2R_FuseShim::CheckSchemeFusable()
{
    auto transpose = nodes[0];
    auto realEven  = nodes[1];

    // check if the first is transpose
    if(transpose->scheme != CS_KERNEL_TRANSPOSE && transpose->scheme != CS_KERNEL_TRANSPOSE_XY_Z)
        return false;

    // fusable when second is realEven, and the first child of it is c2r
    if(realEven->scheme != CS_REAL_TRANSFORM_EVEN)
        return false;
    if(realEven->childNodes.front()->scheme != CS_KERNEL_CMPLX_TO_R)
        return false;

    // update the node to the final effective node: the first child of real-trans-even
    // and node[2] keeps the next leaf node after C2R kernel,
    // since it's possible we'll change its input buffer
    nodes.resize(3);
    nodes[1] = realEven->childNodes.front().get();
    nodes[2] = std::find_if(realEven->childNodes.begin() + 1,
                            realEven->childNodes.end(),
                            [](const auto& n) { return !n->IsBluesteinChirpSetup(); })
                   ->get();

    firstFusedNode = 0;
    lastFusedNode  = 1;

    // if the nextLeafNode is stockham, SBCC or PAD-MUL
    //   we allow the EffectivePlacement of (trans-in, c2r-out) to be inplace,
    //   then we force it to be OP, but change the nextLeafNode's input..
    // So if the nextLeafNode isn't one of these (ex, a transpose for TRTRT)
    //   then we couldn't change transpose's input buffer
    ComputeScheme nextFFTScheme = nodes[2]->scheme;
    if(nextFFTScheme == CS_KERNEL_STOCKHAM || nextFFTScheme == CS_KERNEL_STOCKHAM_BLOCK_CC
       || nextFFTScheme == CS_KERNEL_PAD_MUL)
        allowInplace = true;
    else
        allowInplace = false;

    return true;
}

std::unique_ptr<TreeNode> TransC2R_FuseShim::FuseKernels()
{
    auto transpose = nodes[0];
    auto c2r       = nodes[1];
    auto nextLeaf  = nodes[2];

    if(!PlacementFusable(transpose->obIn, transpose->obOut, c2r->obOut))
        return nullptr;

    auto       input = transpose->input_layout();
    const bool is_3d = transpose->scheme == CS_KERNEL_TRANSPOSE_XY_Z;
    if(is_3d)
        input.reorder_axes(swap_12_3d, axis_scope::length_only);

    // In the fused frame the Hermitian axis is last among the transform
    // axes, and the output has the standalone C2R's complex span there.
    const auto& c2r_output = c2r->output_layout();
    auto        output
        = with_edge_type(input.with_axis_span(input.get_len_rank() - 1, c2r->output_length(0)),
                         c2r->output_array_type(),
                         transpose->output_layout());
    const auto order
        = is_3d ? std::span<const size_t>(rotate_left_3) : std::span<const size_t>(swap_01);
    set_edge_strides(output, c2r_output, order);
    auto fused = NodeFactory::CreateNodeFromScheme(
        CS_KERNEL_TRANSPOSE_CMPLX_TO_R, transpose->parent, std::move(input), std::move(output));
    fused->CopyNodeAttributes(*transpose);
    // no need to check kernel exists, this scheme uses a built-in kernel
    fused->placement = rocfft_placement_notinplace;
    fused->comments.push_back("TransC2R_FuseShim: fused " + PrintScheme(transpose->scheme)
                              + " and following " + PrintScheme(CS_KERNEL_CMPLX_TO_R));

    // if the effective placement is:
    //   1. out-of-place, then we simply combine them to fused kernel, result in OP
    //   2. inplace, we keep the fused-out as trans-out, also result in OP
    //      in this case, we need to change the nextLeaf's input from c2r-out to trans-out
    if(transpose->obIn != c2r->obOut)
    {
        fused->obOut = c2r->obOut;
    }
    else
    {
        nextLeaf->obIn = fused->obOut; // so we still don't break buffer flow of [trans-c2r]-next
        nextLeaf->placement = (nextLeaf->obIn == nextLeaf->obOut) ? rocfft_placement_inplace
                                                                  : rocfft_placement_notinplace;
    }

    // The parent keeps its Hermitian layout. Its incoming edge now holds
    // the preprocessed data in the standalone C2R's (unpermuted) frame.
    std::array<size_t, KERN_ARGS_ARRAY_WIDTH> strides;
    auto incoming = std::span(strides).first(c2r_output.get_full_rank());
    for(size_t axis = 0; axis < incoming.size(); ++axis)
        incoming[axis] = c2r_output.axis_stride(axis);
    c2r->parent->input_preprocessed = true;
    c2r->parent->set_strides(io_data_label::INPUT, incoming);

    return fused;
}

/*****************************************************
 * STK_R2C_Trans = STOCKHAM + REAL_2_CMPLX + TRANSPOSE*
 *****************************************************/
bool STK_R2CTrans_FuseShim::CheckSchemeFusable()
{
    auto realEven  = nodes[0];
    auto transpose = nodes[1];

    // check if the second node is transpose
    if(transpose->scheme != CS_KERNEL_TRANSPOSE && transpose->scheme != CS_KERNEL_TRANSPOSE_Z_XY)
        return false;

    // fusable when first node is realEven r2c, we will fuse the 2nd(STK) and the 3rd(r2c) children
    if(realEven->scheme != CS_REAL_TRANSFORM_EVEN || realEven->childNodes.size() != 3)
        return false;
    // we will fuse the 2nd(STK) and the 3rd(r2c) children, and with the following transpose
    if(realEven->childNodes[1]->scheme != CS_KERNEL_STOCKHAM
       || realEven->childNodes[2]->scheme != CS_KERNEL_R_TO_CMPLX)
        return false;

    // find if have sbrc and length fit requirement
    if(!realEven->childNodes[1]->fuse_CS_KERNEL_STK_R2C_TRANSPOSE())
        return false;

    // NB:
    //    for inplace cases: this fusion can't get correct results, so disable them now
    //    such as 128x128, 100x100, 256x256...etc (-t 2), Need more investigation
    if(realEven->GetPlanRoot()->placement == rocfft_placement_inplace)
        return false;

    // update the node to the final effective nodes:
    nodes.resize(3);
    nodes[0] = realEven->childNodes[1].get(); // stockham the cfft child of real-trans-even
    nodes[1] = realEven->childNodes[2].get(); // r2c
    nodes[2] = transpose; // transpose

    firstFusedNode = 0;
    lastFusedNode  = 2;

    return true;
}

std::unique_ptr<TreeNode> STK_R2CTrans_FuseShim::FuseKernels()
{
    auto stockham  = nodes[0];
    auto r2c       = nodes[1];
    auto transpose = nodes[2];

    if(!PlacementFusable(stockham->obIn, stockham->obOut, transpose->obOut))
        return nullptr;

    auto output = with_edge_type(
        r2c->output_layout(), transpose->output_array_type(), stockham->output_layout());
    set_edge_strides(output, transpose->output_layout());
    auto fused = NodeFactory::CreateNodeFromScheme(CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY,
                                                   stockham->parent,
                                                   stockham->input_layout(),
                                                   std::move(output));
    fused->CopyNodeAttributes(*stockham);
    // check if kernel exists, since the fused kernel uses different scheme other than stockham
    if(!fused->KernelCheck())
        return nullptr;

    fused->placement = rocfft_placement_notinplace;
    fused->obOut     = transpose->obOut;
    fused->comments.push_back("STK_R2CTrans_FuseShim: fused " + PrintScheme(CS_KERNEL_STOCKHAM)
                              + ", " + PrintScheme(CS_KERNEL_R_TO_CMPLX) + " and following "
                              + PrintScheme(transpose->scheme));

    // The fusion crosses subtrees; the real-even parent keeps the same axes
    // but now writes with the transpose strides.
    reset_real_even_output(*r2c->parent, *fused);

    return fused;
}
