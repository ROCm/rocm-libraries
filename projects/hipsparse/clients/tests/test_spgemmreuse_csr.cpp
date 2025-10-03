/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_spgemmreuse_csr.hpp"

#include <hipsparse.h>

typedef std::tuple<int,
                   int,
                   double,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseSpGEMMAlg_t>
    spgemmreuse_csr_tuple;
typedef std::tuple<double,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseIndexBase_t,
                   hipsparseSpGEMMAlg_t,
                   std::string>
    spgemmreuse_csr_bin_tuple;

int spgemmreuse_csr_M_range[] = {77, 981};
int spgemmreuse_csr_K_range[] = {64, 1723};

std::vector<double> spgemmreuse_csr_alpha_range = {2.0};

#if(!defined(CUDART_VERSION))
hipsparseIndexBase_t spgemmreuse_csr_idxbaseA_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseIndexBase_t spgemmreuse_csr_idxbaseB_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseIndexBase_t spgemmreuse_csr_idxbaseC_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
#else // All matrices must use the same base index in cusparse
hipsparseIndexBase_t spgemmreuse_csr_idxbaseA_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
hipsparseIndexBase_t spgemmreuse_csr_idxbaseB_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
hipsparseIndexBase_t spgemmreuse_csr_idxbaseC_range[] = {HIPSPARSE_INDEX_BASE_ZERO};
#endif

hipsparseSpGEMMAlg_t spgemmreuse_csr_alg_range[] = {HIPSPARSE_SPGEMM_DEFAULT};

std::string spgemmreuse_csr_bin[]
    = {"nos1.bin", "nos2.bin", "nos3.bin", "nos4.bin", "nos5.bin", "nos6.bin", "nos7.bin"};

class parameterized_spgemmreuse_csr : public testing::TestWithParam<spgemmreuse_csr_tuple>
{
protected:
    parameterized_spgemmreuse_csr() {}
    virtual ~parameterized_spgemmreuse_csr() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_spgemmreuse_csr_bin : public testing::TestWithParam<spgemmreuse_csr_bin_tuple>
{
protected:
    parameterized_spgemmreuse_csr_bin() {}
    virtual ~parameterized_spgemmreuse_csr_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_spgemmreuse_csr_arguments(spgemmreuse_csr_tuple tup)
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

Arguments setup_spgemmreuse_csr_arguments(spgemmreuse_csr_bin_tuple tup)
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

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
TEST(spgemmreuse_csr_bad_arg, spgemmreuse_csr_float)
{
    testing_spgemmreuse_csr_bad_arg();
}

TEST_P(parameterized_spgemmreuse_csr, spgemmreuse_csr_i32_float)
{
    Arguments arg = setup_spgemmreuse_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemmreuse_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemmreuse_csr, spgemmreuse_csr_i32_float_complex)
{
    Arguments arg = setup_spgemmreuse_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemmreuse_csr<int32_t, int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemmreuse_csr_bin, spgemmreuse_csr_bin_i32_float)
{
    Arguments arg = setup_spgemmreuse_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemmreuse_csr<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

// 64 bit indices not supported in cusparse
#if(!defined(CUDART_VERSION))
TEST_P(parameterized_spgemmreuse_csr, spgemmreuse_csr_i64_double)
{
    Arguments arg = setup_spgemmreuse_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemmreuse_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemmreuse_csr, spgemmreuse_csr_i64_double_complex)
{
    Arguments arg = setup_spgemmreuse_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemmreuse_csr<int64_t, int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_spgemmreuse_csr_bin, spgemmreuse_csr_bin_i64_double)
{
    Arguments arg = setup_spgemmreuse_csr_arguments(GetParam());

    hipsparseStatus_t status = testing_spgemmreuse_csr<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}
#endif

INSTANTIATE_TEST_SUITE_P(spgemmreuse_csr,
                         parameterized_spgemmreuse_csr,
                         testing::Combine(testing::ValuesIn(spgemmreuse_csr_M_range),
                                          testing::ValuesIn(spgemmreuse_csr_K_range),
                                          testing::ValuesIn(spgemmreuse_csr_alpha_range),
                                          testing::ValuesIn(spgemmreuse_csr_idxbaseA_range),
                                          testing::ValuesIn(spgemmreuse_csr_idxbaseB_range),
                                          testing::ValuesIn(spgemmreuse_csr_idxbaseC_range),
                                          testing::ValuesIn(spgemmreuse_csr_alg_range)));

INSTANTIATE_TEST_SUITE_P(spgemmreuse_csr_bin,
                         parameterized_spgemmreuse_csr_bin,
                         testing::Combine(testing::ValuesIn(spgemmreuse_csr_alpha_range),
                                          testing::ValuesIn(spgemmreuse_csr_idxbaseA_range),
                                          testing::ValuesIn(spgemmreuse_csr_idxbaseB_range),
                                          testing::ValuesIn(spgemmreuse_csr_idxbaseC_range),
                                          testing::ValuesIn(spgemmreuse_csr_alg_range),
                                          testing::ValuesIn(spgemmreuse_csr_bin)));
#endif
