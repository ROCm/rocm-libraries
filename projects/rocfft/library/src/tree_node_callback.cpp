// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tree_node.h"

// find first leaf node reading from a buffer
static const TreeNode* FindFirstLeaf(const TreeNode* node, OperatingBuffer buf)
{
    if(!node->childNodes.empty())
    {
        // look at each child
        for(const auto& c : node->childNodes)
        {
            auto reader = FindFirstLeaf(c.get(), buf);
            if(reader)
                return reader;
        }
        // no child read anything?
        return nullptr;
    }
    // this is a leaf node
    return node->obIn == buf ? node : nullptr;
}

// find last leaf node writing to a buffer
static const TreeNode* FindLastLeaf(const TreeNode* node, OperatingBuffer buf)
{
    if(!node->childNodes.empty())
    {
        // look at each child in reverse order
        for(auto c = node->childNodes.crbegin(); c != node->childNodes.crend(); ++c)
        {
            auto writer = FindLastLeaf(c->get(), buf);
            if(writer)
                return writer;
        }
        // no child wrote anything?
        return nullptr;
    }
    // this is a leaf node
    return node->obOut == buf ? node : nullptr;
}

CallbackType TreeNode::GetCallbackType(bool enable_callbacks) const
{
    if(!enable_callbacks)
        return CallbackType::NONE;

    // We only treat real data as complex for even-length real-complex.
    // That is, we must be:
    //
    // - a (possibly indirect) child of an even-length internal node
    //   (CS_REAL_TRANSFORM_EVEN, CS_REAL_2D_EVEN, CS_REAL_3D_EVEN),
    //   and:
    //
    //   - the first leaf node that reads complex interleaved data from
    //     CS_REAL_*_EVEN's obIn (for forward FFT), or
    //
    //   - the last leaf node that writes complex interleaved data to
    //     CS_REAL_*_EVEN's obOut (for inverse FFT)

    // not a leaf node, no callback
    if(!childNodes.empty())
        return CallbackType::NONE;

    for(auto real_even_node = parent; real_even_node != nullptr;
        real_even_node      = real_even_node->parent)
    {
        if(real_even_node->scheme != CS_REAL_TRANSFORM_EVEN
           && real_even_node->scheme != CS_REAL_2D_EVEN
           && real_even_node->scheme != CS_REAL_3D_EVEN)
            continue;

        // if we're here, we must be under CS_REAL_*_EVEN

        // forward
        if(real_even_node->direction == -1)
        {
            return this == FindFirstLeaf(real_even_node, real_even_node->obIn)
                       ? CallbackType::USER_LOAD_STORE_R2C
                       : CallbackType::USER_LOAD_STORE;
        }
        // inverse
        else
        {
            return this == FindLastLeaf(real_even_node, real_even_node->obOut)
                       ? CallbackType::USER_LOAD_STORE_C2R
                       : CallbackType::USER_LOAD_STORE;
        }
    }
    // if we're here, we must not be under CS_REAL_*_EVEN, so
    // we'd use a normal complex callback
    return CallbackType::USER_LOAD_STORE;
}
