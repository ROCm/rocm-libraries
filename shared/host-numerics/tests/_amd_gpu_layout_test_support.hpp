// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <roc/host_numerics/amd_gpu_layout/mx.hpp>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#error "AMDGPULayout consumers must not inherit OpenMP compile flags."
#endif

using namespace roc::host_numerics::amd_gpu_layout;

#pragma once

namespace amd_gpu_layout_test {
namespace {
[[maybe_unused]] size_t runtimeSize(size_t value) {
    volatile size_t runtimeValue = value;
    return runtimeValue;
}

[[maybe_unused]] std::vector<std::byte> copyScaleStorage(const std::vector<uint8_t>& input,
                                                         std::array<size_t, 2> dimensions,
                                                         size_t blockSize,
                                                         MxScaleStorageLayout layout) {
    const MxScaleStoragePlan plan = planMxScaleStorage(dimensions, blockSize, layout);
    return copyMxScaleStorageToPhysicalLayout(reinterpret_cast<const std::byte*>(input.data()),
                                              input.size(), plan);
}

[[maybe_unused]] std::vector<std::byte> encodedBytes(const std::vector<uint8_t>& input) {
    std::vector<std::byte> result(input.size());
    std::transform(input.begin(), input.end(), result.begin(),
                   [](uint8_t value) { return static_cast<std::byte>(value); });
    return result;
}

std::vector<size_t> testStrides(const std::vector<size_t>& sizes,
                                const std::vector<size_t>& dimensionOrder = {}) {
    std::vector<size_t> strides(sizes.size());
    size_t stride = 1;
    if (dimensionOrder.empty()) {
        for (size_t dimension = 0; dimension < sizes.size(); ++dimension) {
            strides[dimension] = stride;
            stride *= sizes[dimension];
        }
    } else {
        for (const size_t dimension : dimensionOrder) {
            strides[dimension] = stride;
            stride *= sizes[dimension];
        }
    }
    return strides;
}
}  // namespace

class MultiIndex {
   public:
    std::vector<size_t> sizes;
    std::vector<size_t> indexes;
    std::vector<size_t> strides;

    explicit MultiIndex(const std::vector<size_t>& sizes) : sizes(sizes) {
        indexes.resize(sizes.size(), 0);
        strides = testStrides(this->sizes);
    }

    MultiIndex(const std::vector<size_t>& sizes, const std::vector<size_t>& dimensionOrder)
        : sizes(sizes) {
        indexes.resize(sizes.size(), 0);
        strides = testStrides(sizes, dimensionOrder);
    }

    size_t index() const {
        size_t result = 0;
        for (size_t dimension = 0; dimension < indexes.size(); ++dimension)
            result += indexes[dimension] * strides[dimension];
        return result;
    }

    size_t operator*() const {
        return index();
    }

    MultiIndex& operator++() {
        for (size_t dimension = 0; dimension < indexes.size(); ++dimension) {
            ++indexes[dimension];
            if (indexes[dimension] < sizes[dimension]) break;
            if (dimension == indexes.size() - 1) break;
            indexes[dimension] = 0;
        }
        return *this;
    }

    bool isEnd() const {
        return !indexes.empty() && indexes.back() >= sizes.back();
    }
};

inline void FillSwizzle(std::vector<float>& scales, size_t k, size_t mn,
                        std::vector<size_t> const& preSwizzleSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto numTilesK = k / tileK;
    auto numTilesMN = mn / tileMN;

    auto sizes = {nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      numTilesK,
                  nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, numTilesMN};

    auto mi = MultiIndex(sizes);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

inline void FillPreSwizzle(std::vector<float>& scales, size_t k, size_t mn,
                           std::vector<size_t> const& preSwizzleSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto numTilesK = k / tileK;
    auto numTilesMN = mn / tileMN;

    std::vector<size_t> sizes = {nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      numTilesK,
                                 nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, numTilesMN};

    std::vector<size_t> dimOrder;
    if (tileMN == 64) {
        // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
        dimOrder = {6, 1, 2, 3, 4, 5, 0, 7};
    } else if (tileMN == 32 && subTileK == 4) {
        // Pre swizzle: swap nSIMDIndexBlock (6) and nVGPRIndex (0)
        //              swap nSIMDBlock (2) and nVGPRBlock (1)
        dimOrder = {6, 2, 1, 3, 4, 5, 0, 7};
    } else if (tileMN == 32 && subTileK == 2) {
        // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
        dimOrder = {1, 2, 0, 3, 4, 5, 6, 7};
    }

    auto mi = MultiIndex(sizes, dimOrder);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

inline void FillSwizzleAndTile(std::vector<float>& scales, size_t k, size_t mn,
                               std::vector<size_t> const& preSwizzleSize,
                               std::vector<size_t> const& preTileSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t ptTileSizeK = preTileSize[0];
    size_t ptTileSizeMN = preTileSize[1];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    auto sizes = {
        nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      ptTileSizeK / tileK,   k / ptTileSizeK,
        nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, ptTileSizeMN / tileMN, mn / ptTileSizeMN};

    auto mi = MultiIndex(sizes);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

inline void FillPreSwizzleAndTile(std::vector<float>& scales, size_t k, size_t mn,
                                  std::vector<size_t> const& preSwizzleSize,
                                  std::vector<size_t> const& preTileSize) {
    auto tileMN = preSwizzleSize[0];
    auto tileK = preSwizzleSize[1];
    auto subTileK = preSwizzleSize[2];

    size_t ptTileSizeK = preTileSize[0];
    size_t ptTileSizeMN = preTileSize[1];

    size_t nLanesPerSIMD = 16;
    size_t nSIMDsPerWave = 4;
    size_t nSIMDIndex = tileMN / nLanesPerSIMD;
    size_t nSIMDBlock = nSIMDsPerWave / nSIMDIndex;
    size_t nVGPRIndex = std::min(nSIMDIndex, subTileK);
    size_t nVGPRBlock = tileK / nSIMDBlock / nVGPRIndex;
    size_t nSIMDIndexBlock = nVGPRIndex;
    size_t nSIMDIndexIndex = nSIMDIndex / nSIMDIndexBlock;

    std::vector<size_t> sizes = {
        nVGPRIndex,    nVGPRBlock,      nSIMDBlock,      ptTileSizeK / tileK,   k / ptTileSizeK,
        nLanesPerSIMD, nSIMDIndexIndex, nSIMDIndexBlock, ptTileSizeMN / tileMN, mn / ptTileSizeMN};

    std::vector<size_t> dimOrder;
    if (tileMN == 64) {
        // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {7, 1, 2, 3, 5, 6, 0, 8, 4, 9};
    } else if (tileMN == 32 && subTileK == 4) {
        // Pre swizzle: swap nSIMDIndexBlock (7) and nVGPRIndex (0)
        //              swap nSIMDBlock (2) and nVGPRBlock (1)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {7, 2, 1, 3, 5, 6, 0, 8, 4, 9};
    } else if (tileMN == 32 && subTileK == 2) {
        // Pre swizzle: rotate nVGPRIndex (0), nVGPRBlock (1), nSIMDBlock (2)
        // Pre tile: push workgroup tiles (4 and 9) to the end
        dimOrder = {1, 2, 0, 3, 5, 6, 7, 8, 4, 9};
    }

    auto mi = MultiIndex(sizes, dimOrder);

    for (; !mi.isEnd(); ++mi) {
        // Create a unique value based on multiple dimensions
        float value = 0.0f;
        for (size_t i = 0; i < mi.indexes.size(); ++i)
            value += static_cast<float>(mi.indexes[i]) * std::pow(10.0f, static_cast<float>(i));
        scales[*mi] = value;
    }
}

}  // namespace amd_gpu_layout_test
