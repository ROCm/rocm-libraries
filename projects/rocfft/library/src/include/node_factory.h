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

#ifndef TREE_NODE_FACTORY_H
#define TREE_NODE_FACTORY_H

#include "tree_node.h"

class NodeFactory
{
private:
    // Maps from length[0] to divLength1 for 1D transforms in
    // single and double precision using block computing.
    typedef std::map<size_t, size_t> Map1DLength;
    static const Map1DLength         map1DLengthSingle;
    static const Map1DLength         map1DLengthDouble;

    // Maps from length[0] to divLength1 for TRTRT decomposition.
    // Normally we would choose the largest kernel that helps decompose
    // a 1D length, but this map holds exceptions that are known to be
    // better
    static const Map1DLength map1DLengthTRTRT;

    static bool Large1DLengthsValid(const function_pool& pool,
                                    const Map1DLength&   map1DLength,
                                    rocfft_precision     precision);
    static bool CheckLarge1DMaps(const function_pool& pool);

public:
    // Create node (user level) using this function
    static std::unique_ptr<TreeNode> CreateNodeFromScheme(ComputeScheme s,
                                                          TreeNode*     parent = nullptr);
    static std::unique_ptr<TreeNode> CreateExplicitNode(NodeMetaData& nodeData,
                                                        TreeNode*     parent,
                                                        ComputeScheme determined_scheme = CS_NONE);

    // Checks if there exists native (radix) support for a given length. If
    // no support exists for the given length, Bluestein algorithm is needed.
    static bool SupportedLength(const function_pool& pool, rocfft_precision precision, size_t len);

    // Checks if  the non-pow2 length input is supported for a Bluestein compute scheme
    static bool
        NonPow2LengthSupported(const function_pool& pool, rocfft_precision precision, size_t len);

    // Gets a (potentially non-pow2) length to run Bluestein
    static size_t
        GetBluesteinLength(const function_pool& pool, rocfft_precision precision, size_t len);

    // Decide scheme from the node meta node
    static ComputeScheme
        DecideNodeScheme(const function_pool& pool, NodeMetaData& nodeData, TreeNode* parent);
    static ComputeScheme DecideRealScheme(const function_pool& pool, NodeMetaData& nodeData);
    static ComputeScheme
        Decide1DScheme(const function_pool& pool, NodeMetaData& nodeData, TreeNode* parent);
    static ComputeScheme Decide2DScheme(const function_pool& pool, NodeMetaData& nodeData);
    static ComputeScheme Decide3DScheme(const function_pool& pool, NodeMetaData& nodeData);

    // determine function:
    static bool use_CS_2D_SINGLE(const function_pool& pool,
                                 NodeMetaData&        nodeData,
                                 rocfft_array_type    inArrayType,
                                 rocfft_array_type    outArrayType);
    static bool use_CS_2D_RC(const function_pool& pool,
                             NodeMetaData&        nodeData); // using scheme CS_2D_RC or not
    static bool use_CS_3D_BLOCK_RC(const function_pool& pool, NodeMetaData& nodeData);
    static bool use_CS_3D_RC(const function_pool& pool, NodeMetaData& nodeData);
    static bool use_CS_3D_PP(const function_pool& pool, NodeMetaData& nodeData);
    // how many SBRC kernels can we put into a 3D transform?
    static size_t count_3D_SBRC_nodes(const function_pool& pool, NodeMetaData& nodeData);

    // FuseShim Creator
    static std::unique_ptr<FuseShim> CreateFuseShim(FuseType                      type,
                                                    const std::vector<TreeNode*>& components);
};
#endif // TREE_NODE_FACTORY_H
