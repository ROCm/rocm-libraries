/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "testing_getri.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int> getri_tuple;

// each tuple is {n, lda}
// case when n = -1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<int> n_size_range = {
    // invalid
    -1,
    // normal (valid) samples
    1,
    20,
    64,
    100,
};

const vector<int> lda_size_range = {
    // invalid (lda < n)
    1,
    // normal (valid) samples
    20,
    64,
    100,
    150,
};

Arguments getri_setup_arguments(getri_tuple tup)
{
    int n_size   = std::get<0>(tup);
    int lda_size = std::get<1>(tup);

    Arguments arg;

    arg.set<int>("n", n_size);
    arg.set<int>("lda", lda_size);

    // only testing standard use case/defaults for strides
    arg.timing = 0;

    return arg;
}

template <testAPI_t API, bool NPVT, typename I, typename SIZE>
class GETRI_BASE : public ::TestWithParam<getri_tuple>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getri_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1)
            testing_getri_bad_arg<API, BATCHED, STRIDED, T, I, SIZE>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_getri<API, BATCHED, STRIDED, NPVT, T, I, SIZE>(arg);
    }
};

class GETRI : public GETRI_BASE<API_NORMAL, false, int, int>
{
};

class GETRI_NPVT : public GETRI_BASE<API_NORMAL, true, int, int>
{
};

// batched tests
TEST_P(GETRI, batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(GETRI, batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(GETRI, batched__float_complex)
{
    run_tests<true, false, hipsolverComplex>();
}

TEST_P(GETRI, batched__double_complex)
{
    run_tests<true, false, hipsolverDoubleComplex>();
}

TEST_P(GETRI_NPVT, batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(GETRI_NPVT, batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(GETRI_NPVT, batched__float_complex)
{
    run_tests<true, false, hipsolverComplex>();
}

TEST_P(GETRI_NPVT, batched__double_complex)
{
    run_tests<true, false, hipsolverDoubleComplex>();
}

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRI,
                         Combine(ValuesIn(n_size_range), ValuesIn(lda_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRI_NPVT,
                         Combine(ValuesIn(n_size_range), ValuesIn(lda_size_range)));
