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

#ifndef TREE_NODE_1D_H
#define TREE_NODE_1D_H

#include "tree_node.h"

/*****************************************************
 * L1D_TRTRT  *
 *****************************************************/
class TRTRT1DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit TRTRT1DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_L1D_TRTRT;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * L1D_CC  *
 *****************************************************/
class CC1DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit CC1DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_L1D_CC;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * L1D_CRT  *
 *****************************************************/
class CRT1DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit CRT1DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_L1D_CRT;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * CS_KERNEL_STOCKHAM  *
 *****************************************************/
class Stockham1DNode : public LeafNode
{
    friend class NodeFactory;

protected:
    Stockham1DNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
    }

    void SetupGridParam_internal(GridParam& gp) override;

public:
    bool                CreateDeviceResources() override;
    std::vector<size_t> CollapsibleDims() override;
    bool                UseOutputLengthForPadding() override
    {
        // with embedded r2c, stockham nodes will change length, so the
        // output length is different from the input length.
        return ebtype != EmbeddedType::NONE;
    }
};

/*****************************************************
 * CS_KERNEL_STOCKHAM_PP  *
 *****************************************************/
class StockhamPP1DNode : public Stockham1DNode
{
    friend class NodeFactory;

protected:
    StockhamPP1DNode(TreeNode* p, ComputeScheme s)
        : Stockham1DNode(p, s)
    {
    }

    void SetupGridParam_internal(GridParam& gp) override;

public:
    bool                CreateDeviceResources() override;
    std::vector<size_t> CollapsibleDims() override;
};

/*****************************************************
 * SBCC  *
 *****************************************************/
class SBCCNode : public LeafNode
{
    friend class NodeFactory;

protected:
    SBCCNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
    }

    void SetupGridParam_internal(GridParam& gp) override;

    // InitIntrinsicMode is the first step to check if eligible for buffer load/store
    void InitIntrinsicMode();

    // manually disable the functionality from benchmark result
    //     the settings are results of an observation, kinda we tune the param in hardcode
    // NB: When during tuning, we should not do the manual setting things.
    //     We should just use the exact setting from the specified config
    void TuneDirectRegType();
    void TuneIntrinsicMode();

public:
    // we can put codes here to switch-on/off some features at arch-wise
    bool KernelCheck(std::vector<FMKey>& kernel_keys = EmptyFMKeyVec) override;

    // reads + writes are along columns so both may benefit from padding
    bool PaddingBenefitsInput() override
    {
        return true;
    }
    bool PaddingBenefitsOutput() override
    {
        return true;
    }
    std::vector<size_t> CollapsibleDims() override;
};

/*****************************************************
 * SBCC Partial-Pass *
 *****************************************************/
class SBCCPPNode : public SBCCNode
{
    friend class NodeFactory;

protected:
    SBCCPPNode(TreeNode* p, ComputeScheme s)
        : SBCCNode(p, s)
    {
    }

    void SetupGridParam_internal(GridParam& gp) override;

public:
    std::vector<size_t> CollapsibleDims() override;
};

/*****************************************************
 * SBRC  *
 *****************************************************/
class SBRCNode : public LeafNode
{
    friend class NodeFactory;

protected:
    SBRCNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
        allowInplace   = false;
    }

    void SetupGridParam_internal(GridParam& gp) override;

    void TuneDirectRegType();

public:
    SBRC_TRANSPOSE_TYPE sbrc_transpose_type(unsigned int blockWidth) const override;

    // we can put codes here to switch-on/off some features at arch-wise
    bool KernelCheck(std::vector<FMKey>& kernel_keys = EmptyFMKeyVec) override;

    // override for sbrcTransType
    FMKey GetKernelKey() const override;

    // writes are along columns so they may benefit from padding
    bool PaddingBenefitsOutput() override
    {
        return true;
    }
    bool UseOutputLengthForPadding() override
    {
        return true;
    }
};

/*****************************************************
 * SBCR  *
 *****************************************************/
class SBCRNode : public LeafNode
{
    friend class NodeFactory;

protected:
    SBCRNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
        allowInplace   = false;
    }

    void SetupGridParam_internal(GridParam& gp) override;

    // InitIntrinsicMode is the first step to check if eligible for buffer load/store
    void InitIntrinsicMode();

    void TuneDirectRegType();
    void TuneIntrinsicMode();

public:
    // we can put codes here to switch-on/off some features at arch-wise
    bool KernelCheck(std::vector<FMKey>& kernel_keys = EmptyFMKeyVec) override;

    bool CreateDeviceResources() override;

    // reads are along columns so they may benefit from padding
    bool PaddingBenefitsInput() override
    {
        return true;
    }
};

#endif // TREE_NODE_1D_H
