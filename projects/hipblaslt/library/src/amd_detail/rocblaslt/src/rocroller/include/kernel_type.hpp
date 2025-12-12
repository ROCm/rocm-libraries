/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/Command.hpp>

struct ScaleType
{
    rocRoller::Operations::ScaleMode mode;
    size_t blockRowSize = 32u;
    size_t blockColSize = 1u;
    rocRoller::DataType type = rocRoller::DataType::E8M0;

    auto operator<=>(const ScaleType& other) const = default;
};

template<>
struct std::hash<ScaleType>
{
    size_t operator()(const ScaleType& s) const noexcept
    {
        size_t modeHash = std::hash<rocRoller::Operations::ScaleMode>{}(s.mode);
        size_t blockRowSizeHash = std::hash<size_t>{}(s.blockRowSize);
        size_t blockColSizeHash = std::hash<size_t>{}(s.blockColSize);
        size_t typeHash = std::hash<rocRoller::DataType>{}(s.type);

        return modeHash ^ (blockRowSizeHash << 1) ^ (blockColSizeHash << 2) ^ (typeHash << 3);
    }
};

/**
 * @brief KernelType
 *
 * All of the values required for different types of kernels.
 * This should not include any optimization flags.
 *
 */
struct KernelType
{
    rocRoller::DataType typeA;
    rocRoller::DataType typeB;
    rocRoller::DataType typeC;
    rocRoller::DataType typeD;
    rocRoller::DataType typeAcc = rocRoller::DataType::Float;

    bool transA;
    bool transB;

    ScaleType scaleTypeA;
    ScaleType scaleTypeB;

    auto operator<=>(const KernelType& other) const = default;
};

template<>
struct std::hash<KernelType>
{
    size_t operator()(const KernelType& k) const noexcept
    {
        size_t typeAHash = std::hash<rocRoller::DataType>{}(k.typeA);
        size_t typeBHash = std::hash<rocRoller::DataType>{}(k.typeB);
        size_t typeCHash = std::hash<rocRoller::DataType>{}(k.typeC);
        size_t typeDHash = std::hash<rocRoller::DataType>{}(k.typeD);
        size_t typeAccHash = std::hash<rocRoller::DataType>{}(k.typeAcc);
        size_t scaleTypeAHash = std::hash<ScaleType>{}(k.scaleTypeA);
        size_t scaleTypeBHash = std::hash<ScaleType>{}(k.scaleTypeB);
        size_t transAHash = std::hash<bool>{}(k.transA);
        size_t transBHash = std::hash<bool>{}(k.transB);

        return typeAHash ^ (typeBHash << 1) ^ (typeCHash << 2) ^
                 (typeDHash << 3) ^ (typeAccHash << 4) ^ (scaleTypeAHash << 5) ^
                 (scaleTypeBHash << 6) ^ (transAHash << 7) ^ (transBHash << 8);
    }
};
