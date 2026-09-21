/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "lstm.hpp"

namespace {

struct TestCase
{
    TestCase(int dir_mode_,
             int no_hx_,
             int no_dhy_,
             int no_cx_,
             int no_dcy_,
             int no_hy_,
             int no_dhx_,
             int no_cy_,
             int no_dcx_)
        : dir_mode(dir_mode_),
          no_hx(no_hx_),
          no_dhy(no_dhy_),
          no_cx(no_cx_),
          no_dcy(no_dcy_),
          no_hy(no_hy_),
          no_dhx(no_dhx_),
          no_cy(no_cy_),
          no_dcx(no_dcx_)
    {
    }

    int dir_mode;
    int no_hx;
    int no_dhy;
    int no_cx;
    int no_dcy;
    int no_hy;
    int no_dhx;
    int no_cy;
    int no_dcx;

    friend std::ostream& operator<<(std::ostream& os, const TestCase& tc)
    {
        return os << "dir-mode:" << tc.dir_mode << " no-hx:" << tc.no_hx << " no-dhy:" << tc.no_dhy
                  << " no-cx:" << tc.no_cx << " no-dcy:" << tc.no_dcy << " no-hy:" << tc.no_hy
                  << " no-dhx:" << tc.no_dhx << " no-cy:" << tc.no_cy << " no-dcx:" << tc.no_dcx
                  << " no-dcx:" << tc.no_dcx;
    }
};

std::vector<TestCase> GetTestCases()
{
    return std::vector{
        // clang-format off
        //   dir-mode no-hx no-dhy no-cx no-dcy no-hy no-dhx no-cy no-dcx
        TestCase(0,     1,    0,     0,    0,     0,    0,     0,    0),
        TestCase(0,     0,    1,     0,    0,     0,    0,     0,    0),
        TestCase(0,     1,    1,     0,    0,     0,    0,     0,    0),
        TestCase(0,     0,    0,     1,    0,     0,    0,     0,    0),
        TestCase(0,     1,    0,     1,    0,     0,    0,     0,    0),
        TestCase(0,     0,    0,     0,    1,     0,    0,     0,    0),
        TestCase(0,     0,    0,     1,    1,     0,    0,     0,    0),
        TestCase(1,     1,    0,     0,    0,     0,    0,     0,    0),
        TestCase(1,     0,    1,     0,    0,     0,    0,     0,    0),
        TestCase(1,     1,    1,     0,    0,     0,    0,     0,    0),
        TestCase(1,     0,    0,     1,    0,     0,    0,     0,    0),
        TestCase(1,     1,    0,     1,    0,     0,    0,     0,    0),
        TestCase(1,     0,    0,     0,    1,     0,    0,     0,    0),
        TestCase(1,     0,    0,     1,    1,     0,    0,     0,    0),
        TestCase(0,     0,    0,     0,    0,     1,    0,     0,    0),
        TestCase(0,     0,    0,     0,    0,     0,    1,     0,    0),
        TestCase(0,     0,    0,     0,    0,     1,    1,     0,    0),
        TestCase(0,     0,    0,     0,    0,     0,    0,     1,    0),
        TestCase(0,     0,    0,     0,    0,     1,    0,     1,    0),
        TestCase(0,     0,    0,     0,    0,     0,    0,     0,    1),
        TestCase(0,     0,    0,     0,    0,     0,    0,     1,    1),
        TestCase(1,     0,    0,     0,    0,     1,    0,     0,    0),
        TestCase(1,     0,    0,     0,    0,     0,    1,     0,    0),
        TestCase(1,     0,    0,     0,    0,     1,    1,     0,    1),
        TestCase(1,     0,    0,     0,    0,     0,    0,     1,    0),
        TestCase(1,     0,    0,     0,    0,     1,    0,     1,    0),
        TestCase(1,     0,    0,     0,    0,     0,    0,     0,    1),
        TestCase(1,     0,    0,     0,    0,     0,    0,     1,    1),
        TestCase(0,     1,    1,     1,    1,     1,    1,     1,    1),
        TestCase(1,     1,    1,     1,    1,     1,    1,     1,    1)
        // clang-format on
    };
}

} // namespace

template <typename T>
struct GPU_LSTM_extra_test : LSTM_test<T>, testing::TestWithParam<TestCase>
{
protected:
    void SetUp() override
    {
        this->dataType = miopen_type<T>{};

        auto [dirMode, nohx, nodhy, nocx, nodcy, nohy, nodhx, nocy, nodcx] = GetParam();

        this->batchSize  = 32;
        this->seqLength  = 3;
        this->batchSeq   = {32, 32, 32};
        this->inVecLen   = 128;
        this->hiddenSize = 128;
        this->numLayers  = 1;
        this->inputMode  = 0;
        this->biasMode   = 0;
        this->dirMode    = dirMode;
        this->nohx       = bool(nohx);
        this->nodhy      = bool(nodhy);
        this->nocx       = bool(nocx);
        this->nodcy      = bool(nodcy);
        this->nohy       = bool(nohy);
        this->nodhx      = bool(nodhx);
        this->nocy       = bool(nocy);
        this->nodcx      = bool(nodcx);
    }
};

using GPU_LSTM_extra_FP16 = GPU_LSTM_extra_test<half_float::half>;

TEST_P(GPU_LSTM_extra_FP16, HalfTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_LSTM_extra_FP16, testing::ValuesIn(GetTestCases()));

using GPU_LSTM_extra_FP32 = GPU_LSTM_extra_test<float>;

TEST_P(GPU_LSTM_extra_FP32, FloatTest) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Full, GPU_LSTM_extra_FP32, testing::ValuesIn(GetTestCases()));
