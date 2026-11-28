/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "utility.hpp"

#include <gtest/gtest.h>
#include <string>

// Host-only checks for status codes that used to escape the C API as exceptions.
// The pre_checkin prefix is the name CI selects. TheRock matches pre_checkin/*
// and Jenkins matches *checkin*. A plain TEST() has no such prefix, so it never
// runs in either lane.
class status_handling : public testing::TestWithParam<int>
{
};

TEST_P(status_handling, error_name_and_string)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10000)
    const char* success_name = hipsparseGetErrorName(HIPSPARSE_STATUS_SUCCESS);
    ASSERT_NE(success_name, nullptr);
#if defined(CUDART_VERSION)
    EXPECT_NE(std::string(success_name).find("CUSPARSE_STATUS_SUCCESS"), std::string::npos)
        << success_name;
#else
    EXPECT_NE(std::string(success_name).find("rocsparse_status_success"), std::string::npos)
        << success_name;
#endif

    const char* success_string = hipsparseGetErrorString(HIPSPARSE_STATUS_SUCCESS);
    ASSERT_NE(success_string, nullptr);
    EXPECT_STRNE(success_string, "Unrecognized status code");

    const char* invalid_name = hipsparseGetErrorName(HIPSPARSE_STATUS_INVALID_VALUE);
    ASSERT_NE(invalid_name, nullptr);
    EXPECT_STRNE(invalid_name, "Unrecognized status code");

    const hipsparseStatus_t unknown = static_cast<hipsparseStatus_t>(-1);
    EXPECT_STREQ(hipsparseGetErrorName(unknown), "Unrecognized status code");
    EXPECT_STREQ(hipsparseGetErrorString(unknown), "Unrecognized status code");
#endif
}

TEST_P(status_handling, mat_descr_setters_reject_invalid_enum)
{
    hipsparseMatDescr_t descr = nullptr;
    ASSERT_EQ(hipsparseCreateMatDescr(&descr), HIPSPARSE_STATUS_SUCCESS);

    EXPECT_EQ(hipsparseSetMatType(descr, static_cast<hipsparseMatrixType_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseSetMatFillMode(descr, static_cast<hipsparseFillMode_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseSetMatDiagType(descr, static_cast<hipsparseDiagType_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseSetMatIndexBase(descr, static_cast<hipsparseIndexBase_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_EQ(hipsparseGetMatType(descr), HIPSPARSE_MATRIX_TYPE_GENERAL);
    EXPECT_EQ(hipsparseGetMatFillMode(descr), HIPSPARSE_FILL_MODE_LOWER);
    EXPECT_EQ(hipsparseGetMatDiagType(descr), HIPSPARSE_DIAG_TYPE_NON_UNIT);
    EXPECT_EQ(hipsparseGetMatIndexBase(descr), HIPSPARSE_INDEX_BASE_ZERO);

    EXPECT_EQ(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_TRIANGULAR),
              HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseGetMatType(descr), HIPSPARSE_MATRIX_TYPE_TRIANGULAR);
    EXPECT_EQ(hipsparseSetMatFillMode(descr, HIPSPARSE_FILL_MODE_UPPER), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseGetMatFillMode(descr), HIPSPARSE_FILL_MODE_UPPER);
    EXPECT_EQ(hipsparseSetMatDiagType(descr, HIPSPARSE_DIAG_TYPE_UNIT), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseGetMatDiagType(descr), HIPSPARSE_DIAG_TYPE_UNIT);
    EXPECT_EQ(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ONE), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseGetMatIndexBase(descr), HIPSPARSE_INDEX_BASE_ONE);

    EXPECT_EQ(hipsparseDestroyMatDescr(descr), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(status_handling, dnvec_rejects_unsupported_datatype)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
    || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
    float                 value      = 1.0f;
    hipsparseDnVecDescr_t descr      = nullptr;
    const hipDataType     not_a_type = static_cast<hipDataType>(-1);

    EXPECT_EQ(hipsparseCreateDnVec(&descr, 1, &value, HIP_R_8U), HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr);
    EXPECT_EQ(hipsparseCreateDnVec(&descr, 1, &value, HIP_R_32U), HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr);
    EXPECT_EQ(hipsparseCreateDnVec(&descr, 1, &value, HIP_C_16F), HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(descr, nullptr);
    EXPECT_EQ(hipsparseCreateDnVec(&descr, 1, &value, not_a_type), HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(descr, nullptr);

    ASSERT_EQ(hipsparseCreateDnVec(&descr, 1, &value, HIP_R_32F), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseDestroyDnVec(descr), HIPSPARSE_STATUS_SUCCESS);

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)
    hipsparseConstDnVecDescr_t const_descr = nullptr;
    EXPECT_EQ(hipsparseCreateConstDnVec(&const_descr, 1, &value, HIP_R_8U),
              HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(const_descr, nullptr);
    EXPECT_EQ(hipsparseCreateConstDnVec(&const_descr, 1, &value, HIP_R_32U),
              HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(const_descr, nullptr);
    EXPECT_EQ(hipsparseCreateConstDnVec(&const_descr, 1, &value, HIP_C_16F),
              HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(const_descr, nullptr);
    EXPECT_EQ(hipsparseCreateConstDnVec(&const_descr, 1, &value, not_a_type),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(const_descr, nullptr);

    ASSERT_EQ(hipsparseCreateConstDnVec(&const_descr, 1, &value, HIP_R_32F),
              HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseDestroyDnVec(const_descr), HIPSPARSE_STATUS_SUCCESS);
#endif
#endif
}

TEST_P(status_handling, spmat_rejects_unsupported_datatype)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
    const hipDataType not_a_type = static_cast<hipDataType>(-1);

    hipsparseSpMatDescr_t coo = nullptr;
    EXPECT_EQ(hipsparseCreateCoo(&coo,
                                 1,
                                 1,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_BASE_ZERO,
                                 HIP_R_8U),
              HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(coo, nullptr);
    EXPECT_EQ(hipsparseCreateCoo(&coo,
                                 1,
                                 1,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_BASE_ZERO,
                                 not_a_type),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(coo, nullptr);

    hipsparseSpMatDescr_t csr = nullptr;
    EXPECT_EQ(hipsparseCreateCsr(&csr,
                                 1,
                                 1,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_BASE_ZERO,
                                 HIP_R_8U),
              HIPSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(csr, nullptr);
    EXPECT_EQ(hipsparseCreateCsr(&csr,
                                 1,
                                 1,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_BASE_ZERO,
                                 not_a_type),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(csr, nullptr);
#endif
}

TEST_P(status_handling, csr2csc_ex2_rejects_unsupported_datatype)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
    // The valType check runs before the handle and the arrays are used, so this
    // test passes none of them.
    const hipDataType not_a_type  = static_cast<hipDataType>(-1);
    size_t            buffer_size = 0;

    for(hipDataType val_type : {HIP_R_8U, HIP_C_16F})
    {
        EXPECT_EQ(hipsparseCsr2cscEx2_bufferSize(nullptr,
                                                 1,
                                                 1,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 val_type,
                                                 HIPSPARSE_ACTION_NUMERIC,
                                                 HIPSPARSE_INDEX_BASE_ZERO,
                                                 HIPSPARSE_CSR2CSC_ALG_DEFAULT,
                                                 &buffer_size),
                  HIPSPARSE_STATUS_NOT_SUPPORTED);
        EXPECT_EQ(hipsparseCsr2cscEx2(nullptr,
                                      1,
                                      1,
                                      0,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      val_type,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      HIPSPARSE_CSR2CSC_ALG_DEFAULT,
                                      nullptr),
                  HIPSPARSE_STATUS_NOT_SUPPORTED);
    }

    EXPECT_EQ(hipsparseCsr2cscEx2_bufferSize(nullptr,
                                             1,
                                             1,
                                             0,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             not_a_type,
                                             HIPSPARSE_ACTION_NUMERIC,
                                             HIPSPARSE_INDEX_BASE_ZERO,
                                             HIPSPARSE_CSR2CSC_ALG_DEFAULT,
                                             &buffer_size),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseCsr2cscEx2(nullptr,
                                  1,
                                  1,
                                  0,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  not_a_type,
                                  HIPSPARSE_ACTION_NUMERIC,
                                  HIPSPARSE_INDEX_BASE_ZERO,
                                  HIPSPARSE_CSR2CSC_ALG_DEFAULT,
                                  nullptr),
              HIPSPARSE_STATUS_INVALID_VALUE);
#endif
}

INSTANTIATE_TEST_SUITE_P(pre_checkin, status_handling, testing::Values(0));
