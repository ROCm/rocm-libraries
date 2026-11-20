// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the device helpers in clients/common/include/uniform_check.hpp, which
// hipblaslt-bench --uniform_check uses to make the inputs uniform along M and to check that every
// row of D equals row 0. They need a GPU.
//
// The suite name carries the "pre_checkin" token so the test_categories.yaml patterns select it.

#include <gtest/gtest.h>

#include "hipBuffer.hpp"
#include "uniform_check.hpp"

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
        void fillRows()
        {
            ASSERT_EQ(uniform_check_fill_rows_device(
                          dev.buf(), elem, rows, cols, ld, stride, batches, nullptr),
                      hipSuccess);
            download();
        }
        // flags is the scratch to use, or a fresh buffer when null.
        UniformCheckDeviceResult check(HipDeviceBuffer* flags = nullptr)
        {
            HipDeviceBuffer result(HIP_R_8I, sizeof(UniformCheckDeviceResult));
            HipDeviceBuffer own(HIP_R_8I, uniform_check_flags_bytes(cols, batches));
            auto*           d = result.as<UniformCheckDeviceResult>();
            EXPECT_EQ(uniform_check_reset_device(d, nullptr), hipSuccess);
            EXPECT_EQ(uniform_check_rows_device(dev.buf(),
                                                elem,
                                                rows,
                                                cols,
                                                ld,
                                                stride,
                                                batches,
                                                (flags ? flags : &own)->buf(),
                                                d,
                                                nullptr),
                      hipSuccess);
            UniformCheckDeviceResult h{};
            EXPECT_EQ(hipMemcpy(&h, d, sizeof(h), hipMemcpyDeviceToHost), hipSuccess);
            return h;
        }
    };

    constexpr size_t kSizes[] = {1, 2, 4, 8, 16};

    TEST(UniformCheck_pre_checkin, FillRowsMakesEveryRowEqualRowZero)
    {
        for(size_t elem : kSizes)
        {
            Matrix                           m(elem, 37, 5, 41, 41 * 5 + 3, 3);
            const std::vector<unsigned char> before = m.host;
            m.fillRows();
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
            EXPECT_EQ(m.check().bad, 0u) << "elem " << elem;
        }
    }

    TEST(UniformCheck_pre_checkin, FillColsMakesEveryColumnEqualColumnZero)
    {
        // A transposed op(A) stores logical row m as column m.
        for(size_t elem : kSizes)
        {
            Matrix m(elem, 19, 7, 23, 23 * 7, 2);
            ASSERT_EQ(uniform_check_fill_cols_device(
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

    TEST(UniformCheck_pre_checkin, CheckFindsOneFlippedBit)
    {
        for(size_t elem : kSizes)
        {
            Matrix m(elem, 300, 9, 305, 305 * 9, 4);
            m.fillRows();

            // Padding below the last row is not part of D.
            *m.at(3, m.rows, 8) ^= 0x01;
            m.upload();
            EXPECT_EQ(m.check().bad, 0u) << "elem " << elem;

            // The last row of the last batch, in the last column, in the top byte.
            *(m.at(3, m.rows - 1, 8) + elem - 1) ^= 0x80;
            m.upload();
            UniformCheckDeviceResult r = m.check();
            EXPECT_EQ(r.bad, 1u) << "elem " << elem;
            EXPECT_EQ(r.first_row(), m.rows - 1);
            EXPECT_EQ(r.first_col(), 8);

            // An earlier row in another column becomes the first difference.
            *m.at(1, 5, 2) ^= 0x01;
            m.upload();
            r = m.check();
            EXPECT_EQ(r.bad, 2u) << "elem " << elem;
            EXPECT_EQ(r.first, (5ull << 32) | 2);
            EXPECT_EQ(r.first_row(), 5);
            EXPECT_EQ(r.first_col(), 2);
        }
    }

    TEST(UniformCheck_pre_checkin, CheckCountsEachColumnOfEachBatchOnce)
    {
        Matrix m(4, 200, 6, 200, 200 * 6, 3);
        m.fillRows();
        // Many rows of column 4 of batch 1, and one row of column 4 of batch 2.
        for(int64_t r = 1; r < m.rows; r += 3)
            *m.at(1, r, 4) ^= 0x10;
        *m.at(2, 7, 4) ^= 0x10;
        m.upload();
        UniformCheckDeviceResult r = m.check();
        EXPECT_EQ(r.bad, 2u);
        EXPECT_EQ(r.first_row(), 1);
        EXPECT_EQ(r.first_col(), 4);
    }

    TEST(UniformCheck_pre_checkin, CheckClearsReusedFlags)
    {
        Matrix          m(2, 64, 5, 64, 64 * 5, 2);
        HipDeviceBuffer flags(HIP_R_8I, uniform_check_flags_bytes(m.cols, m.batches));
        m.fillRows();
        *m.at(1, 9, 3) ^= 0x01;
        m.upload();
        EXPECT_EQ(m.check(&flags).bad, 1u);
        // The difference moves within the same column and to another one; the flag the first
        // check set must not hide the column.
        *m.at(1, 9, 3) ^= 0x01;
        *m.at(1, 20, 3) ^= 0x01;
        *m.at(0, 2, 4) ^= 0x01;
        m.upload();
        UniformCheckDeviceResult r = m.check(&flags);
        EXPECT_EQ(r.bad, 2u);
        EXPECT_EQ(r.first_row(), 2);
        EXPECT_EQ(r.first_col(), 4);
    }

    TEST(UniformCheck_pre_checkin, CheckCoversBatchesBeyondOneGridPass)
    {
        // More elements than one pass of the capped grid covers; the last batch is reached by
        // the grid-stride loop.
        constexpr int64_t batches = (int64_t(1) << 22) + 5;
        Matrix            m(1, 5, 1, 5, 5, batches);
        m.fillRows();
        EXPECT_EQ(m.check().bad, 0u);
        *m.at(m.batches - 1, 4, 0) ^= 0x01;
        m.upload();
        UniformCheckDeviceResult r = m.check();
        EXPECT_EQ(r.bad, 1u);
        EXPECT_EQ(r.first_row(), 4);
    }

    TEST(UniformCheck_pre_checkin, CheckPassesOneRow)
    {
        Matrix m(4, 1, 6, 1, 6, 1);
        EXPECT_EQ(m.check().bad, 0u);
        EXPECT_EQ(m.check().first_row(), -1);
    }
}
