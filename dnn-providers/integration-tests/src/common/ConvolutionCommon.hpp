// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace test_conv_common
{

struct ConvTestCase
{
    std::vector<int64_t> xDims;
    std::vector<int64_t> wDims;
    std::vector<int64_t> yDims;
    std::vector<int64_t> convPrePadding;
    std::vector<int64_t> convPostPadding;
    std::vector<int64_t> convStride;
    std::vector<int64_t> convDilation;
    unsigned seed;
    std::string note;

    ConvTestCase(std::vector<int64_t>&& xDimsLocal,
                 std::vector<int64_t>&& wDimsLocal,
                 std::vector<int64_t>&& convPrePaddingLocal,
                 std::vector<int64_t>&& convPostPaddingLocal,
                 std::vector<int64_t>&& convStrideLocal,
                 std::vector<int64_t>&& convDilationLocal,
                 unsigned seedLocal)
        : xDims(std::move(xDimsLocal))
        , wDims(std::move(wDimsLocal))
        , convPrePadding(std::move(convPrePaddingLocal))
        , convPostPadding(std::move(convPostPaddingLocal))
        , convStride(std::move(convStrideLocal))
        , convDilation(std::move(convDilationLocal))
        , seed(seedLocal)
    {
        // Indices for dimensions
        // N - Batch size, always at index 0
        // C - Channels, always at index 1
        // D - Depth (for 5D tensors), always at index 2 if present
        // H - Height, always at index 2 for 4D tensors and index 3 for 5D tensors
        // W - Width, always at index 3 for 4D tensors and index 4 for 5D tensors
        constexpr int N = 0; // Batch size index

        if(xDims.size() != wDims.size())
        {
            throw std::invalid_argument("xDims and wDims must have the same number of dimensions.");
        }

        // Ensure xDims has at least 3 dimensions (N, C, and at least 1 spatial dimension)
        if(xDims.size() < 3)
        {
            throw std::invalid_argument(
                "xDims must have at least 3 dimensions (N, C, and at least 1 spatial dimension).");
        }

        // Determine the number of spatial dimensions
        auto spatialDims = xDims.size() - 2; // Exclude N and C

        // Validate that the convolution parameter vectors match the number of spatial dimensions
        if(convPrePadding.size() != spatialDims || convPostPadding.size() != spatialDims
           || convDilation.size() != spatialDims || convStride.size() != spatialDims)
        {
            throw std::invalid_argument(
                "Convolution parameter vectors must match the number of spatial dimensions.");
        }

        // Calculate output dimensions based on input dimensions and convolution parameters
        auto n = xDims[N];
        auto cOut = wDims[N];
        std::vector<int64_t> outputDims = {n, cOut};

        // Output dims are inferred by the frontend, duplicated here for logging.
        for(size_t i = 0; i < spatialDims; ++i)
        {
            auto paddedInputSize = xDims[2 + i] + convPrePadding[i] + convPostPadding[i];
            auto effectiveKernelSize = (convDilation[i] * (wDims[2 + i] - 1)) + 1;
            auto dimOut = ((paddedInputSize - effectiveKernelSize) / convStride[i]) + 1;
            outputDims.push_back(dimOut);
        }

        yDims = outputDims;
        note = generateNote();
    }

    std::string generateNote() const
    {
        std::vector<std::string> tags;

        const bool is1x1
            = std::all_of(wDims.begin() + 2, wDims.end(), [](int64_t d) { return d == 1; });
        if(is1x1)
        {
            tags.emplace_back("1x1 Filter");
        }

        if(wDims.size() >= 2 && wDims[1] > 0 && xDims[1] / wDims[1] > 1)
        {
            tags.emplace_back("Grouped");
        }

        if(xDims[0] > 1)
        {
            tags.emplace_back("Batched");
        }

        bool nonSquare = false;
        for(size_t i = 3; i < xDims.size(); ++i)
        {
            if(xDims[i] != xDims[2])
            {
                nonSquare = true;
                break;
            }
        }
        if(nonSquare)
        {
            tags.emplace_back("Non-square");
        }

        const bool hasPadding
            = std::any_of(
                  convPrePadding.begin(), convPrePadding.end(), [](int64_t v) { return v != 0; })
              || std::any_of(
                  convPostPadding.begin(), convPostPadding.end(), [](int64_t v) { return v != 0; });
        if(hasPadding)
        {
            tags.emplace_back("Padding");
        }

        const bool hasStride
            = std::any_of(convStride.begin(), convStride.end(), [](int64_t v) { return v != 1; });
        if(hasStride)
        {
            tags.emplace_back("Stride");
        }

        const bool hasDilation = std::any_of(
            convDilation.begin(), convDilation.end(), [](int64_t v) { return v != 1; });
        if(hasDilation)
        {
            tags.emplace_back("Dilation");
        }

        if(tags.empty())
        {
            return {};
        }

        std::string result;
        for(size_t i = 0; i < tags.size(); ++i)
        {
            if(i > 0)
            {
                result += '+';
            }
            result += tags[i];
        }
        return result;
    }

    friend std::ostream& operator<<(std::ostream& ss, const ConvTestCase& tc)
    {
        using namespace hipdnn_data_sdk::utilities;

        ss << "(x:";
        vecToStream(ss, tc.xDims);
        ss << " w:";
        vecToStream(ss, tc.wDims);
        ss << " y:";
        vecToStream(ss, tc.yDims);
        ss << " prePad:";
        vecToStream(ss, tc.convPrePadding);
        ss << " postPad:";
        vecToStream(ss, tc.convPostPadding);
        ss << " stride:";
        vecToStream(ss, tc.convStride);
        ss << " dilation:";
        vecToStream(ss, tc.convDilation);
        ss << " seed:" << tc.seed;
        if(!tc.note.empty())
        {
            ss << " note:" << tc.note;
        }
        ss << ")";

        return ss;
    }
};

inline std::vector<ConvTestCase> getConvTestCases4D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1
        {{1, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Filter 3x3, No Padding
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Padding
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Stride
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {2, 2}, {1, 1}, seed},
        // Dilation
        {{1, 16, 16, 16}, {1, 16, 3, 3}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, seed},
        // Batched convolution
        {{8, 16, 16, 16}, {1, 16, 1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, seed},
        // Non-square
        {{1, 16, 16, 8}, {1, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Grouped convolution - 2 groups
        {{1, 16, 16, 16}, {2, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, seed},
        // Grouped convolution - 2 batches, 4 groups, stride, padding, dilation
        {{2, 32, 16, 16}, {4, 8, 3, 3}, {1, 1}, {1, 1}, {2, 2}, {2, 2}, seed},
    };
}

inline std::vector<ConvTestCase> getConvTestCases5D()
{
    unsigned seed = hipdnn_test_sdk::utilities::getGlobalTestSeed();

    return {
        // Filter 1x1
        {{1, 16, 16, 16, 16}, {1, 16, 1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Filter 3x3, No Padding
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Padding
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
        // Stride
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, seed},
        // Dilation
        {{1, 16, 16, 16, 16}, {1, 16, 3, 3, 3}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, seed},
        // Batched convolution
        {{8, 16, 16, 16, 16}, {1, 16, 1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, seed},
        // Non-square
        {{1, 16, 16, 8, 4}, {1, 16, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
        // Grouped convolution - 2 groups
        {{1, 16, 16, 16, 16}, {2, 8, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, seed},
        // Grouped convolution - 2 batches, 4 groups, stride, padding, dilation
        {{2, 32, 16, 16, 16}, {4, 8, 3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, seed},
    };
}

} // namespace test_conv_common
