// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

struct BlockScaleQuantizeArgs
{
    const void* x;
    void* y;
    void* scale;
    long long scaleElements;
    int nDim;
    int axis;
    int blockSize;
    double outputLowest;
    double outputMax;
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long xDims[5];
    long long xStrides[5];
    long long yStrides[5];
    long long scaleDims[5];
    long long scaleStrides[5];
    // NOLINTEND(modernize-avoid-c-arrays)
};

struct BlockScaleDequantizeArgs
{
    const void* x;
    const void* scale;
    void* y;
    long long elements;
    int nDim;
    int isNegativeScale;
    // NOLINTBEGIN(modernize-avoid-c-arrays)
    long long xDims[5];
    long long xStrides[5];
    long long yStrides[5];
    long long scaleStrides[5];
    long long effectiveBlockSize[5];
    // NOLINTEND(modernize-avoid-c-arrays)
};
