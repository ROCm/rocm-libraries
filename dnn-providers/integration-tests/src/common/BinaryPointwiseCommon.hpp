// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>

namespace test_binary_pointwise_common
{

struct BinaryPointwiseShapeCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> xStrides;
    std::vector<int64_t> yDims;
    std::vector<int64_t> yStrides;
    std::vector<int64_t> outDims;
    std::vector<int64_t> outStrides;
    std::string note;

    BinaryPointwiseShapeCase(std::vector<int64_t>&& dims, std::string&& caseNote)
        : xDims(dims)
        , xStrides(hipdnn_data_sdk::utilities::generateStrides(xDims))
        , yDims(xDims)
        , yStrides(xStrides)
        , outDims(std::move(dims))
        , outStrides(xStrides)
        , note(std::move(caseNote))
    {
    }

    BinaryPointwiseShapeCase(std::vector<int64_t>&& xShapeDims,
                             std::vector<int64_t>&& yShapeDims,
                             std::string&& caseNote)
        : xDims(std::move(xShapeDims))
        , xStrides(hipdnn_data_sdk::utilities::generateStrides(xDims))
        , yDims(std::move(yShapeDims))
        , yStrides(hipdnn_data_sdk::utilities::generateStrides(yDims))
        , outDims(xDims)
        , outStrides(xStrides)
        , note(std::move(caseNote))
    {
    }

    friend std::ostream& operator<<(std::ostream& ss, const BinaryPointwiseShapeCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(xDims:";
        vecToStream(ss, tc.xDims);
        ss << " xStrides:";
        vecToStream(ss, tc.xStrides);
        ss << " yDims:";
        vecToStream(ss, tc.yDims);
        ss << " yStrides:";
        vecToStream(ss, tc.yStrides);
        ss << " outDims:";
        vecToStream(ss, tc.outDims);
        ss << " outStrides:";
        vecToStream(ss, tc.outStrides);
        if(!tc.note.empty())
        {
            ss << " note:" << tc.note;
        }
        ss << ")";

        return ss;
    }
};

inline std::vector<BinaryPointwiseShapeCase> createBinaryPointwiseShapeCases()
{
    std::vector<BinaryPointwiseShapeCase> cases;

    cases.push_back(BinaryPointwiseShapeCase{{2, 4, 8, 8}, "4d"});
    cases.push_back(BinaryPointwiseShapeCase{{2, 4, 8, 8}, {1, 4, 1, 1}, "Broadcast"});
    cases.push_back(BinaryPointwiseShapeCase{{2, 4, 8, 8, 2}, "5d"});
    cases.push_back(BinaryPointwiseShapeCase{{2, 8, 64}, "3d"});

    return cases;
}

inline std::vector<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode> createBinaryPointwiseModes()
{
    using PointwiseMode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    return {PointwiseMode::ADD,
            PointwiseMode::SUB,
            PointwiseMode::MUL,
            PointwiseMode::MAX_OP,
            PointwiseMode::MIN_OP};
}

} // namespace test_binary_pointwise_common
