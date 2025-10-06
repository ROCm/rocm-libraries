/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_spgemm_csr.hpp"

#include <hipsparse.h>

typedef std::tuple<int,
                   int,
                   double,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseSpGEMMAlg_t>
    spgemm_csr_tuple;
typedef std::tuple<double,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseSpGEMMAlg_t,
                   std::string>
    spgemm_csr_bin_tuple;

int spgemm_csr_M_range[] = {567, 1149};
int spgemm_csr_K_range[] = {649, 2148};

std::vector<double> spgemm_csr_alpha_range = {2.0};

#if(!defined(CUDART_VERSION))
hipsparseIndexBase_t spgemm_csr_idxbaseA_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseIndexBase_t spgemm_csr_idxbaseB_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseIndexBase_t spgemm_csr_idxbaseC_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
#else // All matrices must use the same base index in cusparse
hipsparseIndexBase_t spgemm_csr_idxbaseA_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
hipsparseIndexBase_t spgemm_csr_idxbaseB_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
hipsparseIndexBase_t spgemm_csr_idxbaseC_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
#endif

hipsparseSpGEMMAlg_t spgemm_csr_alg_range[] = {HIPSPARSE_SPGEMM_DEFAULT};

std::string spgemm_csr_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_spgemm_csr : public testing::TestWithParam<spgemm_csr_tuple>
{
protected:
    parameterized_spgemm_csr() {}
    virtual ~parameterized_spgemm_csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spgemm_csr_bin : public testing::TestWithParam<spgemm_csr_bin_tuple>
{
protected:
    parameterized_spgemm_csr_bin() {}
    virtual ~parameterized_spgemm_csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spgemm_csr_arguments(spgemm_csr_tuple tup)
{
    Arguments arg;
    arg.M          = std::get<0>(tup);
    arg.K          = std::get<1>(tup);
    arg.alpha      = std::get<2>(tup);
    arg.baseA      = std::get<3>(tup);
    arg.baseB      = std::get<4>(tup);
    arg.baseC      = std::get<5>(tup);
    arg.spgemm_alg = std::get<6>(tup);
    arg.timing     = 0;
    return arg;
}

Arguments setup_spgemm_csr_arguments(spgemm_csr_bin_tuple tup)
{
    Arguments arg;
    arg.M          = -99;
    arg.K          = -99;
    arg.alpha      = std::get<0>(tup);
    arg.baseA      = std::get<1>(tup);
    arg.baseB      = std::get<2>(tup);
    arg.baseC      = std::get<3>(tup);
    arg.spgemm_alg = std::get<4>(tup);
    arg.timing     = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<5>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
TEST(spgemm_csr_bad_arg, spgemm_csr_float)
{
    testing_spgemm_csr_bad_arg();
}

TEST_P(parameterized_spgemm_csr, spgemm_csr_i32_float)
{
    Arguments arg = setup_spgemm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemm_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemm_csr, spgemm_csr_i32_float_complex)
{
    Arguments arg = setup_spgemm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemm_csr<int32_t, int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemm_csr_bin, spgemm_csr_bin_i32_float)
{
    Arguments arg = setup_spgemm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemm_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

// 64 bit indices not supported in cusparse
#if(!defined(CUDART_VERSION))
TEST_P(parameterized_spgemm_csr, spgemm_csr_i64_double)
{
    Arguments arg = setup_spgemm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemm_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemm_csr, spgemm_csr_i64_double_complex)
{
    Arguments arg = setup_spgemm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemm_csr<int64_t, int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemm_csr_bin, spgemm_csr_bin_i64_double)
{
    Arguments arg = setup_spgemm_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemm_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

INSTANTIATE_TEST_SUITE_P(spgemm_csr,
                         parameterized_spgemm_csr,
                         testing::Combine(testing::ValuesIn(spgemm_csr_M_range),
                                          testing::ValuesIn(spgemm_csr_K_range),
                                          testing::ValuesIn(spgemm_csr_alpha_range),
                                          testing::ValuesIn(spgemm_csr_idxbaseA_range),
                                          testing::ValuesIn(spgemm_csr_idxbaseB_range),
                                          testing::ValuesIn(spgemm_csr_idxbaseC_range),
                                          testing::ValuesIn(spgemm_csr_alg_range)));

INSTANTIATE_TEST_SUITE_P(spgemm_csr_bin,
                         parameterized_spgemm_csr_bin,
                         testing::Combine(testing::ValuesIn(spgemm_csr_alpha_range),
                                          testing::ValuesIn(spgemm_csr_idxbaseA_range),
                                          testing::ValuesIn(spgemm_csr_idxbaseB_range),
                                          testing::ValuesIn(spgemm_csr_idxbaseC_range),
                                          testing::ValuesIn(spgemm_csr_alg_range),
                                          testing::ValuesIn(spgemm_csr_bin)));
#endif
