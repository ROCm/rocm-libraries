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

#ifndef TREE_NODE_REAL_H
#define TREE_NODE_REAL_H

#include "tree_node.h"

/*****************************************************
 * CS_REAL_TRANSFORM_USING_CMPLX
 *****************************************************/
class RealTransCmplxNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit RealTransCmplxNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_TRANSFORM_USING_CMPLX;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;

public:
    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

/*****************************************************
 * CS_REAL_TRANSFORM_EVEN
 *****************************************************/
class RealTransEvenNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit RealTransEvenNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_TRANSFORM_EVEN;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;

public:
    // 3D Even can possibly set this
    bool try_fuse_pre_post_processing = false;

    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

/*****************************************************
 * CS_REAL_2D_EVEN
 *****************************************************/
class Real2DEvenNode : public InternalNode
{
    friend class NodeFactory;

    enum Solution
    {
        INPLACE_SBCC, // SBCC + SBRR with pre/post processing
        TR_PAIR, // RTRT Real2C, or TRTR for C2Real
        REAL_2D_SINGLE, // 2D_SINGLE with pre/post processing
    };

    bool UseOutputLengthForPadding() override
    {
        return true;
    }

protected:
    explicit Real2DEvenNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_2D_EVEN;
    }
    void     AssignParams_internal() override;
    void     BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
    Solution solution = TR_PAIR;

    void BuildTree_internal_2D_SINGLE();
    void BuildTree_internal_SBCC(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec);
    void BuildTree_internal_TR_pair(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec);

    void AssignParams_internal_2D_SINGLE();
    void AssignParams_internal_SBCC();
    void AssignParams_internal_TR_pair();
};

/*****************************************************
 * CS_REAL_3D_EVEN
 *****************************************************/
class Real3DEvenNode : public InternalNode
{
    friend class NodeFactory;

    enum Solution
    {
        REAL_2D_SINGLE_SBCC, // FWD: (2D_SINGLE with post-processing) + CC / BWD: CC + (2D_SINGLE with pre-processing)
        INPLACE_SBCC, // FWD: (RR with post-processing) + CC + CC / BWD: CC + CC + (RR with pre-processing)
        SBCR, // SBCR + SBCR + SBCR with pre-processing for C2Real only
        SBRC, // SBRC + SBRC + SBRC with post-processing for Real2C only // NOT IMPLEMENTED
        TR_PAIRS // RTRTRT Real2C, or TRTRTR for C2Real
    };

    bool UseOutputLengthForPadding() override
    {
        return true;
    }

protected:
    explicit Real3DEvenNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_REAL_3D_EVEN;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;

    Solution solution = TR_PAIRS;

    void Build_solution();

    void BuildTree_internal_2D_SINGLE_CC();
    void BuildTree_internal_SBCC(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec);
    void BuildTree_internal_SBCR(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec);
    void BuildTree_internal_TR_pairs(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec);

    void AssignParams_internal_2D_SINGLE_CC();
    void AssignParams_internal_SBCC();
    void AssignParams_internal_SBCR();
    void AssignParams_internal_TR_pairs();
};

/*****************************************************
 * CS_KERNEL_COPY_R_TO_CMPLX
 * CS_KERNEL_COPY_HERM_TO_CMPLX
 * CS_KERNEL_COPY_CMPLX_TO_HERM
 * CS_KERNEL_COPY_CMPLX_TO_R
 *****************************************************/
class RealTransDataCopyNode : public LeafNode
{
    friend class NodeFactory;

protected:
    RealTransDataCopyNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        /************
        * Placement
        *************/
        allowInplace = false;

        /********************
        * Buffer and ArrayType
        *********************/
        // the two r2c copy-head kernels MUST output to TEMP CMPLX buffer
        if(scheme == CS_KERNEL_COPY_R_TO_CMPLX || scheme == CS_KERNEL_COPY_HERM_TO_CMPLX)
        {
            allowedOutBuf        = OB_TEMP_CMPLX_FOR_REAL | OB_TEMP;
            allowedOutArrayTypes = {rocfft_array_type_complex_interleaved};
        }
        // should be real, but could be treated as CI (the alias type)
        else if(scheme == CS_KERNEL_COPY_CMPLX_TO_R)
        {
            allowedOutArrayTypes = {rocfft_array_type_real, rocfft_array_type_complex_interleaved};
        }
        // should be HI(or HP), but could be treated as CI(or HI) (the alias type)
        else if(scheme == CS_KERNEL_COPY_CMPLX_TO_HERM)
        {
            allowedOutArrayTypes = {rocfft_array_type_hermitian_interleaved,
                                    rocfft_array_type_complex_interleaved,
                                    rocfft_array_type_hermitian_planar,
                                    rocfft_array_type_complex_planar};
        }
    }

    void SetupGridParam_internal(GridParam& gp) override{};

public:
    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

/*****************************************************
 * CS_KERNEL_R_TO_CMPLX
 * CS_KERNEL_R_TO_CMPLX_TRANSPOSE
 * CS_KERNEL_CMPLX_TO_R
 * CS_KERNEL_TRANSPOSE_CMPLX_TO_R
 *****************************************************/
class PrePostKernelNode : public LeafNode
{
    friend class NodeFactory;

protected:
    PrePostKernelNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        need_twd_table = true;
        twd_no_radices = true;

        /************
        * Placement
        *************/
        // transpose-typed needs to be out-of-place
        if(scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE || scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        {
            allowInplace = false;
        }
        else
        {
            // CS_KERNEL_R_TO_CMPLX and CS_KERNEL_CMPLX_TO_R, inherit from parent (1D Even)
            // 1D Even = pre + fft(IP) + callback(IP) / callback(IP) + fft(IP) + post
            allowInplace    = parent->allowInplace;
            allowOutofplace = parent->allowOutofplace;
        }

        /********************
        * ArrayType
        *********************/
        if(scheme == CS_KERNEL_CMPLX_TO_R || scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R)
        {
            allowedOutArrayTypes = {rocfft_array_type_real, rocfft_array_type_complex_interleaved};
        }
    }

    size_t GetTwiddleTableLength() override;
    size_t GetTwiddleTableLengthLimit() override;
    void   SetupGridParam_internal(GridParam& gp) override{};

public:
    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

#endif // TREE_NODE_REAL_H
