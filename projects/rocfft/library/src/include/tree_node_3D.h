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

#ifndef TREE_NODE_3D_H
#define TREE_NODE_3D_H

#include "tree_node_2D.h"

/*****************************************************
 * 3D_RTRT  *
 * 3D 4 node builder,
 * R: 2D FFTs, T: transpose XY_Z,
 * R: row FFTs, T: transpose Z_XY
 *****************************************************/
class RTRT3DNode : public RTRT2DNode
{
    friend class NodeFactory;

protected:
    explicit RTRT3DNode(TreeNode* p)
        : RTRT2DNode(p)
    {
        scheme = CS_3D_RTRT;
    }

protected:
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * 3D_TRTRTR  *
 * 3D 6 node builder,
 * T: transpose Z_XY, R: row FFTs,
 * T: transpose Z_XY, R: row FFTs,
 * T: transpose Z_XY, R: row FFTs
 *****************************************************/
class TRTRTR3DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit TRTRTR3DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_3D_TRTRTR;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * CS_3D_BLOCK_RC  *
 * 3D 3-5 node builder.
 * Uses 3D SBRC kernels fused with transpose
 * for each dimension when possible to do
 * row FFTs + transpose,
 * falls back to separate kernels for
 * row FFTs + transpose XY_Z when not possible.
 *****************************************************/
class BLOCKRC3DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit BLOCKRC3DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_3D_BLOCK_RC;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * CS_3D_BLOCK_CR  *
 * 3D 3-5 node builder.
 * Uses 3D SBCR kernels fused with transpose
 * for each dimension when possible to do
 * row FFTs + transpose.
 *****************************************************/
class BLOCKCR3DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit BLOCKCR3DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_3D_BLOCK_CR;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * CS_3D_RC  *
 * 3D 2 node builder, R: 2D FFTs, C: SBCC
 * 2D FFTs could be:
 *   2D_SINGLE: results in one 2DFFT + SBCC,
 *   2D_RC: result in (row FFT+SBCC) + SBCC,
 *   2D_RTRT: rejected
 *****************************************************/
class RC3DNode : public RC2DNode
{
    friend class NodeFactory;

protected:
    explicit RC3DNode(TreeNode* p)
        : RC2DNode(p)
    {
        scheme = CS_3D_RC;
    }

    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * CS_3D_PP  *
 *****************************************************/
class PP3DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit PP3DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_3D_PP;
    }

    void   AssignParams_internal() override;
    void   BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
    size_t GetPPOffDim() const;
};

/*****************************************************
 * Base Class of fused SBRC and Transpose
 *****************************************************/
class SBRCTranspose3DNode : public LeafNode
{
    friend class NodeFactory;

protected:
    SBRCTranspose3DNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
        allowInplace   = false;
    }

    // should be overriden by the derived class
    virtual size_t sbrc_3D_alignment_dimension() const
    {
        return 0;
    }

    void TuneDirectRegType();

public:
    bool KernelCheck(std::vector<FMKey>& kernel_keys = EmptyFMKeyVec) override;

    // used when RTC kernels, and output to solution map for pre-building to AOT-cache
    unsigned int GetStaticDim() const override
    {
        // This is 3D kernels, but are sometimes shoehorned
        // into 2D plans.  Make sure they get at least 3 dims.
        auto ret = TreeNode::GetStaticDim();
        ret      = (ret == 2) ? 3 : ret;
        return ret;
    }

    SBRC_TRANSPOSE_TYPE sbrc_transpose_type(unsigned int blockWidth) const override
    {
        auto alignment_dimension = sbrc_3D_alignment_dimension();
        if(alignment_dimension == 0)
            return NONE;
        // NB: from the benchmark results, diagonal transpose
        // benefits only some architectures
        if(is_diagonal_sbrc_3D_length(length.front()) && is_cube_size(length)
           && (is_device_gcn_arch(deviceProp, "gfx906")
               || is_device_gcn_arch(deviceProp, "gfx1030")))
            return DIAGONAL;
        if(alignment_dimension % blockWidth == 0)
            return TILE_ALIGNED;
        return TILE_UNALIGNED;
    }

    // override for sbrcTransType
    FMKey GetKernelKey() const override;
};

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z
 *****************************************************/
class SBRCTransXY_ZNode : public SBRCTranspose3DNode
{
    friend class NodeFactory;

protected:
    SBRCTransXY_ZNode(TreeNode* p, ComputeScheme s)
        : SBRCTranspose3DNode(p, s)
    {
    }

    void   SetupGridParam_internal(GridParam& gp) override;
    size_t sbrc_3D_alignment_dimension() const override
    {
        return length[2];
    }

public:
    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY
 *****************************************************/
class SBRCTransZ_XYNode : public SBRCTranspose3DNode
{
    friend class NodeFactory;

protected:
    SBRCTransZ_XYNode(TreeNode* p, ComputeScheme s)
        : SBRCTranspose3DNode(p, s)
    {
    }

    void   SetupGridParam_internal(GridParam& gp) override;
    size_t sbrc_3D_alignment_dimension() const override
    {
        // return length[1] * length[2];
        return length[1];
    }

public:
    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

/*****************************************************
 * Derived Class of fused SBRC and Transpose
 * CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY
 *****************************************************/
class RealCmplxTransZ_XYNode : public SBRCTranspose3DNode
{
    friend class NodeFactory;

protected:
    RealCmplxTransZ_XYNode(TreeNode* p, ComputeScheme s)
        : SBRCTranspose3DNode(p, s)
    {
        twd_attach_halfN = true;
    }

    void   SetupGridParam_internal(GridParam& gp) override;
    size_t sbrc_3D_alignment_dimension() const override
    {
        // return length[1] * length[2];
        return length[1];
    }

public:
    bool CreateDevKernelArgs() override;
    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

#endif // TREE_NODE_3D_H
