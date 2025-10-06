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

#include "testing_sparse_to_dense_csc.hpp"

#include <hipsparse.h>

typedef std::tuple<int, int, hipsparseOrder_t, hipsparseIndexBase_t, hipsparseSparseToDenseAlg_t>
    sparse_to_dense_csc_tuple;
typedef std::tuple<hipsparseOrder_t, hipsparseIndexBase_t, hipsparseSparseToDenseAlg_t, std::string>
    sparse_to_dense_csc_bin_tuple;

int sparse_to_dense_csc_M_range[] = {50};
int sparse_to_dense_csc_N_range[] = {5};

hipsparseOrder_t     sparse_to_dense_csc_order_range[] = {HIPSPARSE_ORDER_COL, HIPSPARSE_ORDER_ROW};
hipsparseIndexBase_t sparse_to_dense_csc_idxbase_range[]
    = {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE};
hipsparseSparseToDenseAlg_t sparse_to_dense_csc_alg_range[] = {HIPSPARSE_SPARSETODENSE_ALG_DEFAULT};

std::string sparse_to_dense_csc_bin[] = {"nos1.bin",
                                         "nos2.bin",
                                         "nos3.bin",
                                         "nos4.bin",
                                         "nos5.bin",
                                         "nos6.bin",
                                         "nos7.bin",
                                         "bibd_22_8.bin"};

class parameterized_sparse_to_dense_csc : public testing::TestWithParam<sparse_to_dense_csc_tuple>
{
protected:
    parameterized_sparse_to_dense_csc() {}
    virtual ~parameterized_sparse_to_dense_csc() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

class parameterized_sparse_to_dense_csc_bin
    : public testing::TestWithParam<sparse_to_dense_csc_bin_tuple>
{
protected:
    parameterized_sparse_to_dense_csc_bin() {}
    virtual ~parameterized_sparse_to_dense_csc_bin() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

Arguments setup_sparse_to_dense_csc_arguments(sparse_to_dense_csc_tuple tup)
{
    Arguments arg;
    arg.M                = std::get<0>(tup);
    arg.N                = std::get<1>(tup);
    arg.orderA           = std::get<2>(tup);
    arg.baseA            = std::get<3>(tup);
    arg.sparse2dense_alg = std::get<4>(tup);
    arg.timing           = 0;
    return arg;
}

Arguments setup_sparse_to_dense_csc_arguments(sparse_to_dense_csc_bin_tuple tup)
{
    Arguments arg;
    arg.orderA           = std::get<0>(tup);
    arg.baseA            = std::get<1>(tup);
    arg.sparse2dense_alg = std::get<2>(tup);
    arg.timing           = 0;

    // Determine absolute path of test matrix
    std::string bin_file = std::get<3>(tup);

    // Matrices are stored at the same path in matrices directory
    arg.set_filename(get_filename(bin_file));

    return arg;
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
TEST(sparse_to_dense_csc_bad_arg, sparse_to_dense_csc_float)
{
    testing_sparse_to_dense_csc_bad_arg();
}

TEST_P(parameterized_sparse_to_dense_csc, sparse_to_dense_csc_i32_float)
{
    Arguments arg = setup_sparse_to_dense_csc_arguments(GetParam());

    hipsparseStatus_t status = testing_sparse_to_dense_csc<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sparse_to_dense_csc, sparse_to_dense_csc_i64_double)
{
    Arguments arg = setup_sparse_to_dense_csc_arguments(GetParam());

    hipsparseStatus_t status = testing_sparse_to_dense_csc<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sparse_to_dense_csc, sparse_to_dense_csc_i32_float_complex)
{
    Arguments arg = setup_sparse_to_dense_csc_arguments(GetParam());

    hipsparseStatus_t status = testing_sparse_to_dense_csc<int32_t, int32_t, hipComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sparse_to_dense_csc, sparse_to_dense_csc_i64_double_complex)
{
    Arguments arg = setup_sparse_to_dense_csc_arguments(GetParam());

    hipsparseStatus_t status = testing_sparse_to_dense_csc<int64_t, int64_t, hipDoubleComplex>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sparse_to_dense_csc_bin, sparse_to_dense_csc_bin_i32_float)
{
    Arguments arg = setup_sparse_to_dense_csc_arguments(GetParam());

    hipsparseStatus_t status = testing_sparse_to_dense_csc<int32_t, int32_t, float>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(parameterized_sparse_to_dense_csc_bin, sparse_to_dense_csc_bin_i64_double)
{
    Arguments arg = setup_sparse_to_dense_csc_arguments(GetParam());

    hipsparseStatus_t status = testing_sparse_to_dense_csc<int64_t, int64_t, double>(arg);
    EXPECT_EQ(status, HIPSPARSE_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(sparse_to_dense_csc,
                         parameterized_sparse_to_dense_csc,
                         testing::Combine(testing::ValuesIn(sparse_to_dense_csc_M_range),
                                          testing::ValuesIn(sparse_to_dense_csc_N_range),
                                          testing::ValuesIn(sparse_to_dense_csc_order_range),
                                          testing::ValuesIn(sparse_to_dense_csc_idxbase_range),
                                          testing::ValuesIn(sparse_to_dense_csc_alg_range)));

INSTANTIATE_TEST_SUITE_P(sparse_to_dense_csc_bin,
                         parameterized_sparse_to_dense_csc_bin,
                         testing::Combine(testing::ValuesIn(sparse_to_dense_csc_order_range),
                                          testing::ValuesIn(sparse_to_dense_csc_idxbase_range),
                                          testing::ValuesIn(sparse_to_dense_csc_alg_range),
                                          testing::ValuesIn(sparse_to_dense_csc_bin)));
#endif
