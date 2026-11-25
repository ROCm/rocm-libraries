// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only unit tests for generateMXInput (mxDataGen.cpp) with a padded leading dimension. A
// matrix of rows x cols stored with its columns ld elements apart occupies ld * (cols - 1) + rows
// elements, and its scales are one per (column, block) whatever ld is. Every MX data type and
// several init methods are checked in both orientations against the same matrix generated dense.
// See README.md in this directory for the HostUnit/smoke_ naming.

#if HIPBLASLT_ENABLE_MXDATAGENERATOR

#include "mxDataGen.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace
{
    constexpr uint8_t kUnwritten = 0xA5;
    constexpr size_t  kSlack     = 64;
    constexpr int     kBlock     = 32;

    size_t elementBits(hipDataType type)
    {
        if(type == HIP_R_6F_E2M3 || type == HIP_R_6F_E3M2)
            return 6;
        if(type == static_cast<hipDataType>(HIP_R_4F_E2M1))
            return 4;
        return 8;
    }

    size_t packedBytes(size_t elements, size_t bits)
    {
        return (elements * bits + 7) / 8;
    }

    uint8_t codeAt(std::vector<uint8_t> const& packed, size_t idx, size_t bits)
    {
        size_t const bit  = idx * bits;
        uint16_t     word = packed[bit / 8];
        if(bit / 8 + 1 < packed.size())
            word |= uint16_t(packed[bit / 8 + 1]) << 8;
        return uint8_t((word >> (bit % 8)) & ((1u << bits) - 1));
    }

    struct Generated
    {
        std::vector<uint8_t> data;
        std::vector<uint8_t> scale;
        std::vector<float>   ref;
    };

    // Matrix A, K along the stored rows when transposed, along the columns otherwise. Both
    // buffers get kSlack bytes of kUnwritten past what the call may write.
    Generated generate(hipDataType type,
                       char const* method,
                       uint64_t    rows,
                       uint64_t    cols,
                       uint64_t    ld,
                       bool        isTranspose)
    {
        size_t const bits = elementBits(type);
        Generated    g;
        g.data.assign(packedBytes(ld * (cols - 1) + rows, bits) + kSlack, kUnwritten);
        g.scale.assign(cols * ((rows + kBlock - 1) / kBlock) + kSlack, kUnwritten);
        g.ref = generateMXInput(type,
                                HIP_R_8U,
                                g.data.data(),
                                g.scale.data(),
                                rows,
                                cols,
                                ld,
                                isTranspose,
                                kBlock,
                                1,
                                /*isMatrixA=*/true,
                                MXScaleLayout::None,
                                method,
                                -1.0f,
                                1.0f,
                                "",
                                /*seedIndex=*/0);
        return g;
    }

    size_t unwrittenFrom(std::vector<uint8_t> const& bytes, size_t from)
    {
        size_t written = 0;
        for(size_t i = from; i < bytes.size(); i++)
            written += bytes[i] != kUnwritten;
        return written;
    }
}

// The client allocates exactly ld * (cols - 1) + rows elements for a padded operand; a generator
// buffer of ld * cols spilled past it. Each column must hold the dense matrix's column, the
// padding zeros, the scales the dense scales, and the reference the dense reference at the padded
// offsets.
TEST(HostUnitMXDataGenPaddedLd, smoke_PaddedColumnsMatchTheDenseMatrix)
{
    struct Shape
    {
        uint64_t rows, cols, ld;
        bool     isTranspose;
    };
    // K along the rows (transposed A) may end in a partial block; along the columns K stays whole.
    Shape const       shapes[] = {{32, 2, 64, true},
                                  {64, 3, 96, true},
                                  {33, 4, 40, true},
                                  {64, 32, 96, false},
                                  {33, 64, 40, false}};
    hipDataType const types[]  = {HIP_R_8F_E4M3,
                                  HIP_R_8F_E5M2,
                                  HIP_R_6F_E2M3,
                                  HIP_R_6F_E3M2,
                                  static_cast<hipDataType>(HIP_R_4F_E2M1)};
    for(hipDataType type : types)
        for(char const* method : {"hpl", "maf1_0", "maf2_0", "maf2_0_unit_scale"})
            for(Shape s : shapes)
            {
                size_t const bits = elementBits(type);
                if(bits == 6 && std::string(method) == "maf2_0_unit_scale")
                    continue; // unit scale has no FP6 row
                SCOPED_TRACE(std::string(method) + " type " + std::to_string(int(type)) + " rows "
                             + std::to_string(s.rows) + " cols " + std::to_string(s.cols) + " ld "
                             + std::to_string(s.ld) + (s.isTranspose ? " K rows" : " K cols"));
                Generated const dense
                    = generate(type, method, s.rows, s.cols, s.rows, s.isTranspose);
                Generated const padded
                    = generate(type, method, s.rows, s.cols, s.ld, s.isTranspose);
                size_t const extent = s.ld * (s.cols - 1) + s.rows;
                size_t const scales = s.cols * ((s.rows + kBlock - 1) / kBlock);

                EXPECT_EQ(unwrittenFrom(padded.data, packedBytes(extent, bits)), 0u)
                    << "data written past the " << extent << "-element matrix";
                EXPECT_EQ(unwrittenFrom(padded.scale, scales), 0u)
                    << "scales written past the " << scales << " the matrix has";
                EXPECT_EQ(unwrittenFrom(dense.scale, scales), 0u);
                for(size_t i = 0; i < scales; i++)
                    ASSERT_EQ(padded.scale[i], dense.scale[i]) << "scale " << i;

                ASSERT_EQ(dense.ref.size(), s.rows * s.cols);
                ASSERT_EQ(padded.ref.size(), s.ld * s.cols);
                for(size_t j = 0; j < s.cols; j++)
                    for(size_t i = 0; i < s.ld && j * s.ld + i < extent; i++)
                    {
                        size_t const at = j * s.ld + i;
                        if(i < s.rows)
                        {
                            ASSERT_EQ(codeAt(padded.data, at, bits),
                                      codeAt(dense.data, j * s.rows + i, bits))
                                << "element (" << i << ", " << j << ")";
                            ASSERT_EQ(padded.ref[at], dense.ref[j * s.rows + i])
                                << "reference (" << i << ", " << j << ")";
                        }
                        else
                        {
                            ASSERT_EQ(codeAt(padded.data, at, bits), 0u)
                                << "padding (" << i << ", " << j << ")";
                            ASSERT_EQ(padded.ref[at], 0.0f) << "padding (" << i << ", " << j << ")";
                        }
                    }
            }
}

// A leading dimension below the row count overlaps the columns; it throws before writing.
TEST(HostUnitMXDataGenPaddedLd, smoke_LeadingDimensionBelowTheRowsIsRefused)
{
    std::vector<uint8_t> data(64 * 2, kUnwritten), scale(64, kUnwritten);
    EXPECT_THROW(generateMXInput(HIP_R_8F_E4M3,
                                 HIP_R_8U,
                                 data.data(),
                                 scale.data(),
                                 64,
                                 2,
                                 32,
                                 /*isTranspose=*/true,
                                 kBlock,
                                 1,
                                 /*isMatrixA=*/true,
                                 MXScaleLayout::None,
                                 "hpl"),
                 std::invalid_argument);
    EXPECT_EQ(unwrittenFrom(data, 0), 0u);
    EXPECT_EQ(unwrittenFrom(scale, 0), 0u);
}

// With K along the columns (non-transposed A, transposed B), a partial trailing K block needs
// rows * ceil(K / block) scales, but the generator blocks along the rows and emits
// ceil(rows / block) * K. Realigning would read past the generated data and scales (64 rows,
// K = 33, ld 128 reads data index 2112 of 2112), so every method that realigns refuses the shape
// before writing either buffer. maf1_0 keeps the generator's layout and stays in bounds.
TEST(HostUnitMXDataGenPaddedLd, smoke_PartialKBlockAlongTheColumnsIsRefused)
{
    uint64_t const    rows = 64, k = 33;
    hipDataType const types[] = {HIP_R_8F_E4M3,
                                 HIP_R_8F_E5M2,
                                 HIP_R_6F_E2M3,
                                 HIP_R_6F_E3M2,
                                 static_cast<hipDataType>(HIP_R_4F_E2M1)};
    for(hipDataType type : types)
        for(char const* method : {"hpl",
                                  "norm_dist",
                                  "rand_int",
                                  "uniform_low_precision",
                                  "maf2_0",
                                  "maf2_0_unit_scale"})
            for(uint64_t ld : {rows, uint64_t(128)})
                for(bool isMatrixA : {true, false})
                {
                    size_t const bits = elementBits(type);
                    if(bits == 6 && std::string(method) == "maf2_0_unit_scale")
                        continue; // unit scale has no FP6 row
                    SCOPED_TRACE(std::string(method) + " type " + std::to_string(int(type)) + " ld "
                                 + std::to_string(ld)
                                 + (isMatrixA ? " non-transposed A" : " transposed B"));
                    std::vector<uint8_t> data(packedBytes(ld * (k - 1) + rows, bits) + kSlack,
                                              kUnwritten);
                    std::vector<uint8_t> scale(rows * ((k + kBlock - 1) / kBlock) + kSlack,
                                               kUnwritten);
                    EXPECT_THROW(generateMXInput(type,
                                                 HIP_R_8U,
                                                 data.data(),
                                                 scale.data(),
                                                 rows,
                                                 k,
                                                 ld,
                                                 /*isTranspose=*/!isMatrixA,
                                                 isMatrixA ? 1 : kBlock,
                                                 isMatrixA ? kBlock : 1,
                                                 isMatrixA,
                                                 MXScaleLayout::None,
                                                 method,
                                                 -1.0f,
                                                 1.0f,
                                                 "",
                                                 /*seedIndex=*/0),
                                 std::runtime_error);
                    EXPECT_EQ(unwrittenFrom(data, 0), 0u);
                    EXPECT_EQ(unwrittenFrom(scale, 0), 0u);
                }

    for(hipDataType type : types)
    {
        SCOPED_TRACE("maf1_0 type " + std::to_string(int(type)));
        Generated const g = generate(type, "maf1_0", rows, k, 128, /*isTranspose=*/false);
        EXPECT_EQ(g.ref.size(), 128 * k);
        EXPECT_EQ(unwrittenFrom(g.data, packedBytes(128 * (k - 1) + rows, elementBits(type))), 0u);
    }
}

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
