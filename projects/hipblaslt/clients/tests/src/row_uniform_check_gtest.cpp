// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the device helpers in clients/common/include/row_uniform.hpp, which
// hipblaslt-bench --row_uniform_check uses to make the inputs uniform along M and to check that
// every row of D equals row 0. They need a GPU.
//
// The suite name carries the "pre_checkin" token so the test_categories.yaml patterns select it.

#include <gtest/gtest.h>

#include "hipBuffer.hpp"
#include "row_uniform.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    // A column-major batched matrix of byte-sized elements, kept on the host and the device.
    struct Matrix
    {
        size_t                     elem;
        int64_t                    rows, cols, ld, stride, batches;
        std::vector<unsigned char> host;
        HipDeviceBuffer            dev;

        Matrix(size_t e, int64_t r, int64_t c, int64_t l, int64_t s, int64_t b)
            : elem(e)
            , rows(r)
            , cols(c)
            , ld(l)
            , stride(s)
            , batches(b)
            , host(size_t(s * (b - 1) + l * c) * e)
            , dev(HIP_R_8I, host.size())
        {
            for(size_t i = 0; i < host.size(); i++)
                host[i] = static_cast<unsigned char>((i * 2654435761u) >> 13);
            upload();
        }

        unsigned char* at(int64_t b, int64_t r, int64_t c)
        {
            return host.data() + size_t(b * stride + c * ld + r) * elem;
        }
        void upload()
        {
            EXPECT_EQ(hipMemcpy(dev.buf(), host.data(), host.size(), hipMemcpyHostToDevice),
                      hipSuccess);
        }
        void download()
        {
            EXPECT_EQ(hipMemcpy(host.data(), dev.buf(), host.size(), hipMemcpyDeviceToHost),
                      hipSuccess);
        }
        RowUniformDeviceResult check()
        {
            HipDeviceBuffer result(HIP_R_8I, sizeof(RowUniformDeviceResult));
            auto*           d = result.as<RowUniformDeviceResult>();
            EXPECT_EQ(row_uniform_reset_device(d, nullptr), hipSuccess);
            EXPECT_EQ(
                row_uniform_check_device(dev.buf(), elem, rows, cols, ld, stride, batches, d, nullptr),
                hipSuccess);
            RowUniformDeviceResult h{};
            EXPECT_EQ(hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost), hipSuccess);
            return h;
        }
    };

    constexpr size_t kSizes[] = {1, 2, 4, 8, 16};

    TEST(RowUniformCheck_pre_checkin, FillRowsMakesEveryRowEqualRowZero)
    {
        for(size_t elem : kSizes)
        {
            Matrix m(elem, 37, 5, 41, 41 * 5 + 3, 3);
            const std::vector<unsigned char> before = m.host;
            ASSERT_EQ(row_uniform_fill_rows_device(
                          m.dev.buf(), elem, m.rows, m.cols, m.ld, m.stride, m.batches, nullptr),
                      hipSuccess);
            m.download();
            for(int64_t b = 0; b < m.batches; b++)
                for(int64_t c = 0; c < m.cols; c++)
                {
                    for(int64_t r = 0; r < m.rows; r++)
                        ASSERT_EQ(std::memcmp(m.at(b, r, c), m.at(b, 0, c), elem), 0)
                            << "elem " << elem << " b " << b << " r " << r << " c " << c;
                    // The ld padding is left alone.
                    for(int64_t r = m.rows; r < m.ld; r++)
                    {
                        const size_t off = m.at(b, r, c) - m.host.data();
                        ASSERT_EQ(std::memcmp(m.at(b, r, c), before.data() + off, elem), 0);
                    }
                }
            EXPECT_EQ(m.check().bad_cols, 0u) << "elem " << elem;
        }
    }

    TEST(RowUniformCheck_pre_checkin, FillColsMakesEveryColumnEqualColumnZero)
    {
        // A transposed op(A) stores logical row m as column m.
        for(size_t elem : kSizes)
        {
            Matrix m(elem, 19, 7, 23, 23 * 7, 2);
            ASSERT_EQ(row_uniform_fill_cols_device(
                          m.dev.buf(), elem, m.rows, m.cols, m.ld, m.stride, m.batches, nullptr),
                      hipSuccess);
            m.download();
            for(int64_t b = 0; b < m.batches; b++)
                for(int64_t c = 0; c < m.cols; c++)
                    for(int64_t r = 0; r < m.rows; r++)
                        ASSERT_EQ(std::memcmp(m.at(b, r, c), m.at(b, r, 0), elem), 0)
                            << "elem " << elem << " b " << b << " r " << r << " c " << c;
        }
    }

    TEST(RowUniformCheck_pre_checkin, CheckFindsOneFlippedBit)
    {
        for(size_t elem : kSizes)
        {
            Matrix m(elem, 300, 9, 305, 305 * 9, 4);
            ASSERT_EQ(row_uniform_fill_rows_device(
                          m.dev.buf(), elem, m.rows, m.cols, m.ld, m.stride, m.batches, nullptr),
                      hipSuccess);
            m.download();

            // Padding below the last row is not part of D.
            *m.at(3, m.rows, 8) ^= 0x01;
            m.upload();
            EXPECT_EQ(m.check().bad_cols, 0u) << "elem " << elem;

            // The last row of the last batch, in the last column, in the top byte.
            *(m.at(3, m.rows - 1, 8) + elem - 1) ^= 0x80;
            m.upload();
            RowUniformDeviceResult r = m.check();
            EXPECT_EQ(r.bad_cols, 1u) << "elem " << elem;
            EXPECT_EQ(r.first_row(), m.rows - 1);
            EXPECT_EQ(r.first_col(), 8);

            // An earlier row in another column becomes the first difference.
            *m.at(1, 5, 2) ^= 0x01;
            m.upload();
            r = m.check();
            EXPECT_EQ(r.bad_cols, 2u) << "elem " << elem;
            EXPECT_EQ(r.first_row(), 5);
            EXPECT_EQ(r.first_col(), 2);
        }
    }

    TEST(RowUniformCheck_pre_checkin, CheckCoversBatchesAboveTheGridLimit)
    {
        // 65535 batches per launch; the last batch is in the second launch.
        Matrix m(2, 3, 1, 3, 3, 65537);
        ASSERT_EQ(row_uniform_fill_rows_device(
                      m.dev.buf(), 2, m.rows, m.cols, m.ld, m.stride, m.batches, nullptr),
                  hipSuccess);
        m.download();
        EXPECT_EQ(m.check().bad_cols, 0u);
        *m.at(m.batches - 1, 2, 0) ^= 0x01;
        m.upload();
        RowUniformDeviceResult r = m.check();
        EXPECT_EQ(r.bad_cols, 1u);
        EXPECT_EQ(r.first_row(), 2);
    }

    TEST(RowUniformCheck_pre_checkin, UniformMatrixWithOneRowPasses)
    {
        Matrix m(4, 1, 6, 1, 6, 1);
        EXPECT_EQ(m.check().bad_cols, 0u);
        EXPECT_EQ(m.check().first_row(), -1);
    }
}
