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

#ifndef TREE_NODE_2D_H
#define TREE_NODE_2D_H

#include "node_factory.h"
#include "tree_node.h"

/*****************************************************
 * 2D_RTRT  *
 *****************************************************/
class RTRT2DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit RTRT2DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_2D_RTRT;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * 2D_RC  *
 *****************************************************/
class RC2DNode : public InternalNode
{
    friend class NodeFactory;

protected:
    explicit RC2DNode(TreeNode* p)
        : InternalNode(p)
    {
        scheme = CS_2D_RC;
    }
    void AssignParams_internal() override;
    void BuildTree_internal(SchemeTreeVec& child_scheme_trees = EmptySchemeTreeVec) override;
};

/*****************************************************
 * CS_KERNEL_2D_SINGLE  *
 *****************************************************/
class Single2DNode : public LeafNode
{
    friend class NodeFactory;

private:
    bool twd_attach_halfN2 = false;

protected:
    Single2DNode(TreeNode* p, ComputeScheme s)
        : LeafNode(p, s)
    {
        externalKernel = true;
        need_twd_table = true;
    }

    void SetupGridParam_internal(GridParam& gp) override;

public:
    bool CreateDeviceResources() override;
    bool UseOutputLengthForPadding() override
    {
        return ebtype != EmbeddedType::NONE;
    }
};

#endif // TREE_NODE_2D_H
