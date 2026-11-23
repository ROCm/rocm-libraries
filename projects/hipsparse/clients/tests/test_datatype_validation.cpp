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

#include "hipsparse_local_descr.hpp"
#include "hipsparse_local_dnmat.hpp"
#include "hipsparse_local_dnvec.hpp"
#include "hipsparse_local_handle.hpp"
#include "hipsparse_local_spmat.hpp"
#include "hipsparse_local_spvec.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "utility.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

// Every public hipSPARSE routine that takes a hipDataType by value has one test
// here. Each test checks three kinds of value:
//
//   - a supported value (HIP_R_32F): the call succeeds,
//   - a valid hipDataType that hipSPARSE does not support (HIP_R_8U, HIP_C_16F):
//     the call returns HIPSPARSE_STATUS_NOT_SUPPORTED,
//   - a value outside the hipDataType enumeration (-1): the call returns
//     HIPSPARSE_STATUS_INVALID_VALUE.
//
// The compute routines receive otherwise valid arguments (a real handle, real
// descriptors and a real buffer), so the datatype is the only reason for the
// error. Rejected calls run before the supported call at every stage of a
// multi-stage routine. A rejected call that modified a descriptor would break
// the supported call that follows it.
//
// The pre_checkin prefix is the name CI selects. TheRock matches pre_checkin/*
// and Jenkins matches *checkin*. A plain TEST() has no such prefix, so it never
// runs in either lane.
//
// Each API is guarded with the CUDART_VERSION range of its declaration in the
// public headers.

// Version-gated guards below compare against this value. It is larger than any
// real CUDA version for the AMD backend, which never defines CUDART_VERSION.
#if defined(CUDART_VERSION)
#define DTV_CUDART CUDART_VERSION
#else
#define DTV_CUDART 2147483647
#endif

using namespace hipsparse_test;

#if(DTV_CUDART > 10010)

namespace
{
    constexpr hipDataType supported_type      = HIP_R_32F;
    const hipDataType     unsupported_types[] = {HIP_R_8U, HIP_C_16F};
    const hipDataType     not_a_type          = static_cast<hipDataType>(-1);

    constexpr hipsparseOperation_t op_n  = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    constexpr hipsparseIndexType_t idx32 = HIPSPARSE_INDEX_32I;
    constexpr hipsparseIndexBase_t base0 = HIPSPARSE_INDEX_BASE_ZERO;

    // Calls call(type) for the unsupported and invalid values and checks the
    // status of each.
    template <typename Call>
    void expect_rejected(Call&& call)
    {
        for(hipDataType type : unsupported_types)
        {
            SCOPED_TRACE(testing::Message() << "hipDataType " << static_cast<int>(type));
            EXPECT_EQ(call(type), HIPSPARSE_STATUS_NOT_SUPPORTED);
        }
        {
            SCOPED_TRACE("hipDataType -1");
            EXPECT_EQ(call(not_a_type), HIPSPARSE_STATUS_INVALID_VALUE);
        }
    }

    // Checks a descriptor creation routine. create(&descr, type) creates the
    // descriptor and destroy(descr) releases it. A rejected call must leave the
    // output descriptor untouched.
    template <typename Descr, typename Create, typename Destroy>
    void expect_create(Create&& create, Destroy&& destroy)
    {
        Descr descr = nullptr;
        EXPECT_EQ(create(&descr, supported_type), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_NE(descr, nullptr);
        if(descr != nullptr)
        {
            EXPECT_EQ(destroy(descr), HIPSPARSE_STATUS_SUCCESS);
        }

        expect_rejected([&](hipDataType type) {
            Descr rejected = nullptr;
            auto  status   = create(&rejected, type);
            EXPECT_EQ(rejected, nullptr);
            return status;
        });
    }

    template <typename T>
    hipsparse_unique_ptr make_device(const std::vector<T>& host)
    {
        const size_t         bytes = std::max<size_t>(sizeof(T) * host.size(), 1);
        hipsparse_unique_ptr ptr{device_malloc(bytes), device_free};
        if(!host.empty())
        {
            CHECK_HIP_ERROR(
                hipMemcpy(ptr.get(), host.data(), sizeof(T) * host.size(), hipMemcpyHostToDevice));
        }
        return ptr;
    }

    // A routine that needs no workspace expects a null buffer. device_malloc(0)
    // returns one.
    hipsparse_unique_ptr make_buffer(size_t bytes)
    {
        hipsparse_unique_ptr ptr{device_malloc(bytes), device_free};
        if(bytes > 0)
        {
            CHECK_HIP_ERROR(hipMemset(ptr.get(), 0, bytes));
        }
        return ptr;
    }

    // A 2x2 problem that every routine below can run on. A is diag(2, 3) in CSR
    // form, which is also a valid triangular matrix for the solvers.
    struct problem
    {
        hipsparseLocalHandle_t handle;

        hipsparse_unique_ptr ptr3          = make_device(std::vector<int>{0, 1, 2});
        hipsparse_unique_ptr idx2          = make_device(std::vector<int>{0, 1});
        hipsparse_unique_ptr idx4          = make_device(std::vector<int>{0, 1, 0, 1});
        hipsparse_unique_ptr idx1          = make_device(std::vector<int>{0});
        hipsparse_unique_ptr slice_offsets = make_device(std::vector<int>{0, 2});
        hipsparse_unique_ptr val1          = make_device(std::vector<float>{1});
        hipsparse_unique_ptr val2          = make_device(std::vector<float>{2, 3});
        hipsparse_unique_ptr val2_out      = make_device(std::vector<float>{0, 0});
        hipsparse_unique_ptr val4          = make_device(std::vector<float>{2, 0, 0, 3});
        hipsparse_unique_ptr vec_x         = make_device(std::vector<float>{1, 1});
        hipsparse_unique_ptr vec_y         = make_device(std::vector<float>{0, 0});
        hipsparse_unique_ptr mat_x         = make_device(std::vector<float>{1, 0, 0, 1});
        hipsparse_unique_ptr mat_y         = make_device(std::vector<float>{0, 0, 0, 0});

        hipsparseLocalCsr   A;
        hipsparseLocalCsr   C_pattern; // A's pattern with writable values, for SDDMM
        hipsparseLocalDnVec x;
        hipsparseLocalDnVec y;
        hipsparseLocalSpVec sx;
        hipsparseLocalDnMat X;
        hipsparseLocalDnMat Y;

        float alpha = 1.0f;
        float beta  = 0.0f;

        problem()
            : A(2, 2, 2, ptr3.get(), idx2.get(), val2.get(), idx32, idx32, base0, supported_type)
            , C_pattern(2,
                        2,
                        2,
                        ptr3.get(),
                        idx2.get(),
                        val2_out.get(),
                        idx32,
                        idx32,
                        base0,
                        supported_type)
            , x(2, vec_x.get(), supported_type)
            , y(2, vec_y.get(), supported_type)
            , sx(2, 1, idx1.get(), val1.get(), idx32, base0, supported_type)
            , X(2, 2, 2, mat_x.get(), supported_type, HIPSPARSE_ORDER_COL)
            , Y(2, 2, 2, mat_y.get(), supported_type, HIPSPARSE_ORDER_COL)
        {
        }
    };
}

class datatype_validation : public testing::TestWithParam<int>
{
};

//
// Descriptor creation
//

#if(DTV_CUDART > 10010)
TEST_P(datatype_validation, hipsparseCreateSpVec)
{
    problem p;
    expect_create<hipsparseSpVecDescr_t>(
        [&](hipsparseSpVecDescr_t* d, hipDataType t) {
            return hipsparseCreateSpVec(d, 2, 1, p.idx1.get(), p.val1.get(), idx32, base0, t);
        },
        [](hipsparseSpVecDescr_t d) { return hipsparseDestroySpVec(d); });
}

TEST_P(datatype_validation, hipsparseCreateDnVec)
{
    problem p;
    expect_create<hipsparseDnVecDescr_t>(
        [&](hipsparseDnVecDescr_t* d, hipDataType t) {
            return hipsparseCreateDnVec(d, 2, p.vec_x.get(), t);
        },
        [](hipsparseDnVecDescr_t d) { return hipsparseDestroyDnVec(d); });
}
#endif

#if(DTV_CUDART >= 12000)
TEST_P(datatype_validation, hipsparseCreateConstSpVec)
{
    problem p;
    expect_create<hipsparseConstSpVecDescr_t>(
        [&](hipsparseConstSpVecDescr_t* d, hipDataType t) {
            return hipsparseCreateConstSpVec(d, 2, 1, p.idx1.get(), p.val1.get(), idx32, base0, t);
        },
        [](hipsparseConstSpVecDescr_t d) { return hipsparseDestroySpVec(d); });
}

TEST_P(datatype_validation, hipsparseCreateConstDnVec)
{
    problem p;
    expect_create<hipsparseConstDnVecDescr_t>(
        [&](hipsparseConstDnVecDescr_t* d, hipDataType t) {
            return hipsparseCreateConstDnVec(d, 2, p.vec_x.get(), t);
        },
        [](hipsparseConstDnVecDescr_t d) { return hipsparseDestroyDnVec(d); });
}

TEST_P(datatype_validation, hipsparseCreateConstDnMat)
{
    problem p;
    expect_create<hipsparseConstDnMatDescr_t>(
        [&](hipsparseConstDnMatDescr_t* d, hipDataType t) {
            return hipsparseCreateConstDnMat(d, 2, 2, 2, p.mat_x.get(), t, HIPSPARSE_ORDER_COL);
        },
        [](hipsparseConstDnMatDescr_t d) { return hipsparseDestroyDnMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateConstCoo)
{
    problem p;
    expect_create<hipsparseConstSpMatDescr_t>(
        [&](hipsparseConstSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateConstCoo(
                d, 2, 2, 2, p.idx2.get(), p.idx2.get(), p.val2.get(), idx32, base0, t);
        },
        [](hipsparseConstSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateConstCsc)
{
    problem p;
    expect_create<hipsparseConstSpMatDescr_t>(
        [&](hipsparseConstSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateConstCsc(
                d, 2, 2, 2, p.ptr3.get(), p.idx2.get(), p.val2.get(), idx32, idx32, base0, t);
        },
        [](hipsparseConstSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateConstBlockedEll)
{
    problem p;
    expect_create<hipsparseConstSpMatDescr_t>(
        [&](hipsparseConstSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateConstBlockedEll(
                d, 2, 2, 1, 2, p.idx4.get(), p.val4.get(), idx32, base0, t);
        },
        [](hipsparseConstSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}
#endif

#if(DTV_CUDART >= 12001)
TEST_P(datatype_validation, hipsparseCreateConstCsr)
{
    problem p;
    expect_create<hipsparseConstSpMatDescr_t>(
        [&](hipsparseConstSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateConstCsr(
                d, 2, 2, 2, p.ptr3.get(), p.idx2.get(), p.val2.get(), idx32, idx32, base0, t);
        },
        [](hipsparseConstSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}
#endif

#if(DTV_CUDART >= 10010)
TEST_P(datatype_validation, hipsparseCreateDnMat)
{
    problem p;
    expect_create<hipsparseDnMatDescr_t>(
        [&](hipsparseDnMatDescr_t* d, hipDataType t) {
            return hipsparseCreateDnMat(d, 2, 2, 2, p.mat_x.get(), t, HIPSPARSE_ORDER_COL);
        },
        [](hipsparseDnMatDescr_t d) { return hipsparseDestroyDnMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateCoo)
{
    problem p;
    expect_create<hipsparseSpMatDescr_t>(
        [&](hipsparseSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateCoo(
                d, 2, 2, 2, p.idx2.get(), p.idx2.get(), p.val2.get(), idx32, base0, t);
        },
        [](hipsparseSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateCsr)
{
    problem p;
    expect_create<hipsparseSpMatDescr_t>(
        [&](hipsparseSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateCsr(
                d, 2, 2, 2, p.ptr3.get(), p.idx2.get(), p.val2.get(), idx32, idx32, base0, t);
        },
        [](hipsparseSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}
#endif

// The AMD backend never defines CUDART_VERSION and always has this routine.
#if(!defined(CUDART_VERSION) || (CUDART_VERSION >= 10010 && CUDART_VERSION < 12000))
TEST_P(datatype_validation, hipsparseCreateCooAoS)
{
    problem p;
    expect_create<hipsparseSpMatDescr_t>(
        [&](hipsparseSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateCooAoS(d, 2, 2, 2, p.idx4.get(), p.val2.get(), idx32, base0, t);
        },
        [](hipsparseSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}
#endif

#if(DTV_CUDART >= 11020)
TEST_P(datatype_validation, hipsparseCreateCsc)
{
    problem p;
    expect_create<hipsparseSpMatDescr_t>(
        [&](hipsparseSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateCsc(
                d, 2, 2, 2, p.ptr3.get(), p.idx2.get(), p.val2.get(), idx32, idx32, base0, t);
        },
        [](hipsparseSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}
#endif

#if(DTV_CUDART >= 11021)
TEST_P(datatype_validation, hipsparseCreateBlockedEll)
{
    problem p;
    expect_create<hipsparseSpMatDescr_t>(
        [&](hipsparseSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateBlockedEll(
                d, 2, 2, 1, 2, p.idx4.get(), p.val4.get(), idx32, base0, t);
        },
        [](hipsparseSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}
#endif

#if(DTV_CUDART >= 12011)
TEST_P(datatype_validation, hipsparseCreateSlicedEll)
{
    problem p;
    expect_create<hipsparseSpMatDescr_t>(
        [&](hipsparseSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateSlicedEll(d,
                                            2,
                                            2,
                                            2,
                                            2,
                                            2,
                                            p.slice_offsets.get(),
                                            p.idx2.get(),
                                            p.val2.get(),
                                            idx32,
                                            idx32,
                                            base0,
                                            t);
        },
        [](hipsparseSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateConstSlicedEll)
{
    problem p;
    expect_create<hipsparseConstSpMatDescr_t>(
        [&](hipsparseConstSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateConstSlicedEll(d,
                                                 2,
                                                 2,
                                                 2,
                                                 2,
                                                 2,
                                                 p.slice_offsets.get(),
                                                 p.idx2.get(),
                                                 p.val2.get(),
                                                 idx32,
                                                 idx32,
                                                 base0,
                                                 t);
        },
        [](hipsparseConstSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateBsr)
{
    problem p;
    expect_create<hipsparseSpMatDescr_t>(
        [&](hipsparseSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateBsr(d,
                                      1,
                                      1,
                                      1,
                                      2,
                                      2,
                                      p.ptr3.get(),
                                      p.idx2.get(),
                                      p.val4.get(),
                                      idx32,
                                      idx32,
                                      base0,
                                      t,
                                      HIPSPARSE_ORDER_ROW);
        },
        [](hipsparseSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}

TEST_P(datatype_validation, hipsparseCreateConstBsr)
{
    problem p;
    expect_create<hipsparseConstSpMatDescr_t>(
        [&](hipsparseConstSpMatDescr_t* d, hipDataType t) {
            return hipsparseCreateConstBsr(d,
                                           1,
                                           1,
                                           1,
                                           2,
                                           2,
                                           p.ptr3.get(),
                                           p.idx2.get(),
                                           p.val4.get(),
                                           idx32,
                                           idx32,
                                           base0,
                                           t,
                                           HIPSPARSE_ORDER_ROW);
        },
        [](hipsparseConstSpMatDescr_t d) { return hipsparseDestroySpMat(d); });
}
#endif

//
// Conversion
//

#if(DTV_CUDART >= 10010)
TEST_P(datatype_validation, hipsparseCsr2cscEx2_bufferSize)
{
    problem p;
    auto    csc_val = make_device(std::vector<float>{0, 0});
    auto    csc_ptr = make_device(std::vector<int>{0, 0, 0});
    auto    csc_ind = make_device(std::vector<int>{0, 0});
    size_t  size    = 0;

    auto call = [&](hipDataType t) {
        return hipsparseCsr2cscEx2_bufferSize(p.handle,
                                              2,
                                              2,
                                              2,
                                              p.val2.get(),
                                              static_cast<const int*>(p.ptr3.get()),
                                              static_cast<const int*>(p.idx2.get()),
                                              csc_val.get(),
                                              static_cast<int*>(csc_ptr.get()),
                                              static_cast<int*>(csc_ind.get()),
                                              t,
                                              HIPSPARSE_ACTION_NUMERIC,
                                              base0,
                                              HIPSPARSE_CSR2CSC_ALG_DEFAULT,
                                              &size);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseCsr2cscEx2)
{
    problem p;
    auto    csc_val = make_device(std::vector<float>{0, 0});
    auto    csc_ptr = make_device(std::vector<int>{0, 0, 0});
    auto    csc_ind = make_device(std::vector<int>{0, 0});
    size_t  size    = 0;

    ASSERT_EQ(hipsparseCsr2cscEx2_bufferSize(p.handle,
                                             2,
                                             2,
                                             2,
                                             p.val2.get(),
                                             static_cast<const int*>(p.ptr3.get()),
                                             static_cast<const int*>(p.idx2.get()),
                                             csc_val.get(),
                                             static_cast<int*>(csc_ptr.get()),
                                             static_cast<int*>(csc_ind.get()),
                                             supported_type,
                                             HIPSPARSE_ACTION_NUMERIC,
                                             base0,
                                             HIPSPARSE_CSR2CSC_ALG_DEFAULT,
                                             &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseCsr2cscEx2(p.handle,
                                   2,
                                   2,
                                   2,
                                   p.val2.get(),
                                   static_cast<const int*>(p.ptr3.get()),
                                   static_cast<const int*>(p.idx2.get()),
                                   csc_val.get(),
                                   static_cast<int*>(csc_ptr.get()),
                                   static_cast<int*>(csc_ind.get()),
                                   t,
                                   HIPSPARSE_ACTION_NUMERIC,
                                   base0,
                                   HIPSPARSE_CSR2CSC_ALG_DEFAULT,
                                   buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif

//
// SpVV
//

TEST_P(datatype_validation, hipsparseSpVV_bufferSize)
{
    problem p;
    float   result = 0;
    size_t  size   = 0;

    auto call = [&](hipDataType t) {
        return hipsparseSpVV_bufferSize(p.handle, op_n, p.sx, p.x, &result, t, &size);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpVV)
{
    problem p;
    float   result = 0;
    size_t  size   = 0;

    ASSERT_EQ(hipsparseSpVV_bufferSize(p.handle, op_n, p.sx, p.x, &result, supported_type, &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSpVV(p.handle, op_n, p.sx, p.x, &result, t, buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

//
// SpMV
//

TEST_P(datatype_validation, hipsparseSpMV_bufferSize)
{
    problem p;
    size_t  size = 0;

    auto call = [&](hipDataType t) {
        return hipsparseSpMV_bufferSize(
            p.handle, op_n, &p.alpha, p.A, p.x, &p.beta, p.y, t, HIPSPARSE_SPMV_ALG_DEFAULT, &size);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpMV_preprocess)
{
    problem p;
    size_t  size = 0;

    ASSERT_EQ(hipsparseSpMV_bufferSize(p.handle,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       p.x,
                                       &p.beta,
                                       p.y,
                                       supported_type,
                                       HIPSPARSE_SPMV_ALG_DEFAULT,
                                       &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSpMV_preprocess(p.handle,
                                        op_n,
                                        &p.alpha,
                                        p.A,
                                        p.x,
                                        &p.beta,
                                        p.y,
                                        t,
                                        HIPSPARSE_SPMV_ALG_DEFAULT,
                                        buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpMV)
{
    problem p;
    size_t  size = 0;

    ASSERT_EQ(hipsparseSpMV_bufferSize(p.handle,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       p.x,
                                       &p.beta,
                                       p.y,
                                       supported_type,
                                       HIPSPARSE_SPMV_ALG_DEFAULT,
                                       &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSpMV(p.handle,
                             op_n,
                             &p.alpha,
                             p.A,
                             p.x,
                             &p.beta,
                             p.y,
                             t,
                             HIPSPARSE_SPMV_ALG_DEFAULT,
                             buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

//
// SpMM
//

TEST_P(datatype_validation, hipsparseSpMM_bufferSize)
{
    problem p;
    size_t  size = 0;

    auto call = [&](hipDataType t) {
        return hipsparseSpMM_bufferSize(p.handle,
                                        op_n,
                                        op_n,
                                        &p.alpha,
                                        p.A,
                                        p.X,
                                        &p.beta,
                                        p.Y,
                                        t,
                                        HIPSPARSE_SPMM_ALG_DEFAULT,
                                        &size);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

#if(DTV_CUDART >= 11021)
TEST_P(datatype_validation, hipsparseSpMM_preprocess)
{
    problem p;
    size_t  size = 0;

    ASSERT_EQ(hipsparseSpMM_bufferSize(p.handle,
                                       op_n,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       p.X,
                                       &p.beta,
                                       p.Y,
                                       supported_type,
                                       HIPSPARSE_SPMM_ALG_DEFAULT,
                                       &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSpMM_preprocess(p.handle,
                                        op_n,
                                        op_n,
                                        &p.alpha,
                                        p.A,
                                        p.X,
                                        &p.beta,
                                        p.Y,
                                        t,
                                        HIPSPARSE_SPMM_ALG_DEFAULT,
                                        buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif

TEST_P(datatype_validation, hipsparseSpMM)
{
    problem p;
    size_t  size = 0;

    ASSERT_EQ(hipsparseSpMM_bufferSize(p.handle,
                                       op_n,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       p.X,
                                       &p.beta,
                                       p.Y,
                                       supported_type,
                                       HIPSPARSE_SPMM_ALG_DEFAULT,
                                       &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSpMM(p.handle,
                             op_n,
                             op_n,
                             &p.alpha,
                             p.A,
                             p.X,
                             &p.beta,
                             p.Y,
                             t,
                             HIPSPARSE_SPMM_ALG_DEFAULT,
                             buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

//
// SDDMM
//

#if(DTV_CUDART >= 11022)
TEST_P(datatype_validation, hipsparseSDDMM_bufferSize)
{
    problem p;
    size_t  size = 0;

    auto call = [&](hipDataType t) {
        return hipsparseSDDMM_bufferSize(p.handle,
                                         op_n,
                                         op_n,
                                         &p.alpha,
                                         p.X,
                                         p.Y,
                                         &p.beta,
                                         p.C_pattern,
                                         t,
                                         HIPSPARSE_SDDMM_ALG_DEFAULT,
                                         &size);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSDDMM_preprocess)
{
    problem p;
    size_t  size = 0;

    ASSERT_EQ(hipsparseSDDMM_bufferSize(p.handle,
                                        op_n,
                                        op_n,
                                        &p.alpha,
                                        p.X,
                                        p.Y,
                                        &p.beta,
                                        p.C_pattern,
                                        supported_type,
                                        HIPSPARSE_SDDMM_ALG_DEFAULT,
                                        &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSDDMM_preprocess(p.handle,
                                         op_n,
                                         op_n,
                                         &p.alpha,
                                         p.X,
                                         p.Y,
                                         &p.beta,
                                         p.C_pattern,
                                         t,
                                         HIPSPARSE_SDDMM_ALG_DEFAULT,
                                         buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSDDMM)
{
    problem p;
    size_t  size = 0;

    ASSERT_EQ(hipsparseSDDMM_bufferSize(p.handle,
                                        op_n,
                                        op_n,
                                        &p.alpha,
                                        p.X,
                                        p.Y,
                                        &p.beta,
                                        p.C_pattern,
                                        supported_type,
                                        HIPSPARSE_SDDMM_ALG_DEFAULT,
                                        &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSDDMM(p.handle,
                              op_n,
                              op_n,
                              &p.alpha,
                              p.X,
                              p.Y,
                              &p.beta,
                              p.C_pattern,
                              t,
                              HIPSPARSE_SDDMM_ALG_DEFAULT,
                              buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif

//
// SpSV
//

#if(DTV_CUDART >= 11030)
TEST_P(datatype_validation, hipsparseSpSV_bufferSize)
{
    problem                 p;
    hipsparseLocalSpSVDescr descr;
    size_t                  size = 0;

    auto call = [&](hipDataType t) {
        return hipsparseSpSV_bufferSize(
            p.handle, op_n, &p.alpha, p.A, p.x, p.y, t, HIPSPARSE_SPSV_ALG_DEFAULT, descr, &size);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpSV_analysis)
{
    problem                 p;
    hipsparseLocalSpSVDescr descr;
    size_t                  size = 0;

    ASSERT_EQ(hipsparseSpSV_bufferSize(p.handle,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       p.x,
                                       p.y,
                                       supported_type,
                                       HIPSPARSE_SPSV_ALG_DEFAULT,
                                       descr,
                                       &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);

    auto call = [&](hipDataType t) {
        return hipsparseSpSV_analysis(p.handle,
                                      op_n,
                                      &p.alpha,
                                      p.A,
                                      p.x,
                                      p.y,
                                      t,
                                      HIPSPARSE_SPSV_ALG_DEFAULT,
                                      descr,
                                      buffer.get());
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpSV_solve)
{
    problem                 p;
    hipsparseLocalSpSVDescr descr;
    size_t                  size = 0;

    ASSERT_EQ(hipsparseSpSV_bufferSize(p.handle,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       p.x,
                                       p.y,
                                       supported_type,
                                       HIPSPARSE_SPSV_ALG_DEFAULT,
                                       descr,
                                       &size),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer = make_buffer(size);
    ASSERT_EQ(hipsparseSpSV_analysis(p.handle,
                                     op_n,
                                     &p.alpha,
                                     p.A,
                                     p.x,
                                     p.y,
                                     supported_type,
                                     HIPSPARSE_SPSV_ALG_DEFAULT,
                                     descr,
                                     buffer.get()),
              HIPSPARSE_STATUS_SUCCESS);

    auto call = [&](hipDataType t) {
        return hipsparseSpSV_solve(
            p.handle, op_n, &p.alpha, p.A, p.x, p.y, t, HIPSPARSE_SPSV_ALG_DEFAULT, descr);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif

//
// SpSM
//

#if(DTV_CUDART >= 11031)
namespace
{
    // Runs the stages of SpSM in order. Every member that takes a type returns
    // the status of that stage.
    struct spsm_problem
    {
        problem                 p;
        hipsparseLocalSpSMDescr descr;
        size_t                  size = 0;
        hipsparse_unique_ptr    buffer{nullptr, device_free};

        hipsparseStatus_t buffer_size(hipDataType t)
        {
            return hipsparseSpSM_bufferSize(p.handle,
                                            op_n,
                                            op_n,
                                            &p.alpha,
                                            p.A,
                                            p.X,
                                            p.Y,
                                            t,
                                            HIPSPARSE_SPSM_ALG_DEFAULT,
                                            descr,
                                            &size);
        }

        hipsparseStatus_t analysis(hipDataType t)
        {
            return hipsparseSpSM_analysis(p.handle,
                                          op_n,
                                          op_n,
                                          &p.alpha,
                                          p.A,
                                          p.X,
                                          p.Y,
                                          t,
                                          HIPSPARSE_SPSM_ALG_DEFAULT,
                                          descr,
                                          buffer.get());
        }

        hipsparseStatus_t solve(hipDataType t)
        {
            return hipsparseSpSM_solve(p.handle,
                                       op_n,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       p.X,
                                       p.Y,
                                       t,
                                       HIPSPARSE_SPSM_ALG_DEFAULT,
                                       descr,
                                       buffer.get());
        }

        hipsparseStatus_t solve_ex(hipDataType t)
        {
            return hipsparseSpSM_solve_ex(p.handle,
                                          op_n,
                                          op_n,
                                          &p.alpha,
                                          p.A,
                                          p.X,
                                          p.Y,
                                          t,
                                          HIPSPARSE_SPSM_ALG_DEFAULT,
                                          descr);
        }

        void prepare_buffer()
        {
            ASSERT_EQ(buffer_size(supported_type), HIPSPARSE_STATUS_SUCCESS);
            buffer = make_buffer(size);
        }

        void prepare_analysis()
        {
            ASSERT_NO_FATAL_FAILURE(prepare_buffer());
            ASSERT_EQ(analysis(supported_type), HIPSPARSE_STATUS_SUCCESS);
        }
    };
}

TEST_P(datatype_validation, hipsparseSpSM_bufferSize)
{
    spsm_problem s;
    expect_rejected([&](hipDataType t) { return s.buffer_size(t); });
    EXPECT_EQ(s.buffer_size(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpSM_analysis)
{
    spsm_problem s;
    ASSERT_NO_FATAL_FAILURE(s.prepare_buffer());
    expect_rejected([&](hipDataType t) { return s.analysis(t); });
    EXPECT_EQ(s.analysis(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpSM_solve)
{
    spsm_problem s;
    ASSERT_NO_FATAL_FAILURE(s.prepare_analysis());
    expect_rejected([&](hipDataType t) { return s.solve(t); });
    EXPECT_EQ(s.solve(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

#if(DTV_CUDART >= 12000)
// The CUDA 11.x definition of hipsparseSpSM_solve_ex is not exercised here. See
// the version ranges in hipsparse_spsm.h and nvidia_detail/generic/hipsparse_spsm.cpp.
TEST_P(datatype_validation, hipsparseSpSM_solve_ex)
{
    spsm_problem s;
    ASSERT_NO_FATAL_FAILURE(s.prepare_analysis());
    expect_rejected([&](hipDataType t) { return s.solve_ex(t); });
    EXPECT_EQ(s.solve_ex(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif
#endif

//
// SpGEMM
//

#if(DTV_CUDART >= 11000)
namespace
{
    // Runs the stages of SpGEMM in order. Every member that takes a type returns
    // the status of that stage.
    struct spgemm_problem
    {
        problem                   p;
        hipsparseLocalCsr         B;
        hipsparse_unique_ptr      c_ptr = make_device(std::vector<int>{0, 0, 0});
        hipsparseLocalCsr         C;
        hipsparseLocalSpGEMMDescr descr;

        size_t               size1 = 0;
        size_t               size2 = 0;
        hipsparse_unique_ptr buffer1{nullptr, device_free};
        hipsparse_unique_ptr buffer2{nullptr, device_free};
        hipsparse_unique_ptr c_col{nullptr, device_free};
        hipsparse_unique_ptr c_val{nullptr, device_free};

        spgemm_problem()
            : B(2,
                2,
                2,
                p.ptr3.get(),
                p.idx2.get(),
                p.val2.get(),
                idx32,
                idx32,
                base0,
                supported_type)
            , C(2, 2, 0, c_ptr.get(), nullptr, nullptr, idx32, idx32, base0, supported_type)
        {
        }

        hipsparseStatus_t work_estimation(hipDataType t, void* buffer)
        {
            return hipsparseSpGEMM_workEstimation(p.handle,
                                                  op_n,
                                                  op_n,
                                                  &p.alpha,
                                                  p.A,
                                                  B,
                                                  &p.beta,
                                                  C,
                                                  t,
                                                  HIPSPARSE_SPGEMM_DEFAULT,
                                                  descr,
                                                  &size1,
                                                  buffer);
        }

        hipsparseStatus_t compute(hipDataType t, void* buffer)
        {
            return hipsparseSpGEMM_compute(p.handle,
                                           op_n,
                                           op_n,
                                           &p.alpha,
                                           p.A,
                                           B,
                                           &p.beta,
                                           C,
                                           t,
                                           HIPSPARSE_SPGEMM_DEFAULT,
                                           descr,
                                           &size2,
                                           buffer);
        }

        hipsparseStatus_t copy(hipDataType t)
        {
            return hipsparseSpGEMM_copy(p.handle,
                                        op_n,
                                        op_n,
                                        &p.alpha,
                                        p.A,
                                        B,
                                        &p.beta,
                                        C,
                                        t,
                                        HIPSPARSE_SPGEMM_DEFAULT,
                                        descr);
        }

        void query_work_estimation()
        {
            ASSERT_EQ(work_estimation(supported_type, nullptr), HIPSPARSE_STATUS_SUCCESS);
            buffer1 = make_buffer(size1);
        }

        void run_work_estimation()
        {
            ASSERT_NO_FATAL_FAILURE(query_work_estimation());
            ASSERT_EQ(work_estimation(supported_type, buffer1.get()), HIPSPARSE_STATUS_SUCCESS);
        }

        void query_compute()
        {
            ASSERT_NO_FATAL_FAILURE(run_work_estimation());
            ASSERT_EQ(compute(supported_type, nullptr), HIPSPARSE_STATUS_SUCCESS);
            buffer2 = make_buffer(size2);
        }

        void run_compute()
        {
            ASSERT_NO_FATAL_FAILURE(query_compute());
            ASSERT_EQ(compute(supported_type, buffer2.get()), HIPSPARSE_STATUS_SUCCESS);
        }

        void set_c_pointers()
        {
            int64_t rows = 0, cols = 0, nnz = 0;
            ASSERT_EQ(hipsparseSpMatGetSize(C, &rows, &cols, &nnz), HIPSPARSE_STATUS_SUCCESS);
            c_col = make_buffer(sizeof(int) * nnz);
            c_val = make_buffer(sizeof(float) * nnz);
            ASSERT_EQ(hipsparseCsrSetPointers(C, c_ptr.get(), c_col.get(), c_val.get()),
                      HIPSPARSE_STATUS_SUCCESS);
        }

        void prepare_copy()
        {
            ASSERT_NO_FATAL_FAILURE(run_compute());
            ASSERT_NO_FATAL_FAILURE(set_c_pointers());
        }
    };
}

TEST_P(datatype_validation, hipsparseSpGEMM_workEstimation)
{
    spgemm_problem s;

    // Size query, then the call that runs with the buffer.
    expect_rejected([&](hipDataType t) { return s.work_estimation(t, nullptr); });
    ASSERT_NO_FATAL_FAILURE(s.query_work_estimation());
    expect_rejected([&](hipDataType t) { return s.work_estimation(t, s.buffer1.get()); });
    EXPECT_EQ(s.work_estimation(supported_type, s.buffer1.get()), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpGEMM_compute)
{
    spgemm_problem s;
    ASSERT_NO_FATAL_FAILURE(s.run_work_estimation());

    expect_rejected([&](hipDataType t) { return s.compute(t, nullptr); });
    ASSERT_EQ(s.compute(supported_type, nullptr), HIPSPARSE_STATUS_SUCCESS);
    s.buffer2 = make_buffer(s.size2);
    expect_rejected([&](hipDataType t) { return s.compute(t, s.buffer2.get()); });
    EXPECT_EQ(s.compute(supported_type, s.buffer2.get()), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpGEMM_copy)
{
    spgemm_problem s;
    ASSERT_NO_FATAL_FAILURE(s.prepare_copy());

    expect_rejected([&](hipDataType t) { return s.copy(t); });
    EXPECT_EQ(s.copy(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif

//
// SpGEMM with reuse
//

#if(DTV_CUDART >= 11031)
TEST_P(datatype_validation, hipsparseSpGEMMreuse_compute)
{
    problem           p;
    hipsparseLocalCsr B(
        2, 2, 2, p.ptr3.get(), p.idx2.get(), p.val2.get(), idx32, idx32, base0, supported_type);
    auto              c_ptr = make_device(std::vector<int>{0, 0, 0});
    hipsparseLocalCsr C(
        2, 2, 0, c_ptr.get(), nullptr, nullptr, idx32, idx32, base0, supported_type);
    hipsparseLocalSpGEMMDescr descr;
    const auto                alg = HIPSPARSE_SPGEMM_DEFAULT;

    // Work estimation
    size_t size1 = 0;
    ASSERT_EQ(hipsparseSpGEMMreuse_workEstimation(
                  p.handle, op_n, op_n, p.A, B, C, alg, descr, &size1, nullptr),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer1 = make_buffer(size1);
    ASSERT_EQ(hipsparseSpGEMMreuse_workEstimation(
                  p.handle, op_n, op_n, p.A, B, C, alg, descr, &size1, buffer1.get()),
              HIPSPARSE_STATUS_SUCCESS);

    // Number of non-zeros
    size_t size2 = 0, size3 = 0, size4 = 0;
    ASSERT_EQ(hipsparseSpGEMMreuse_nnz(p.handle,
                                       op_n,
                                       op_n,
                                       p.A,
                                       B,
                                       C,
                                       alg,
                                       descr,
                                       &size2,
                                       nullptr,
                                       &size3,
                                       nullptr,
                                       &size4,
                                       nullptr),
              HIPSPARSE_STATUS_SUCCESS);
    auto buffer2 = make_buffer(size2);
    auto buffer3 = make_buffer(size3);
    auto buffer4 = make_buffer(size4);
    ASSERT_EQ(hipsparseSpGEMMreuse_nnz(p.handle,
                                       op_n,
                                       op_n,
                                       p.A,
                                       B,
                                       C,
                                       alg,
                                       descr,
                                       &size2,
                                       buffer2.get(),
                                       &size3,
                                       buffer3.get(),
                                       &size4,
                                       buffer4.get()),
              HIPSPARSE_STATUS_SUCCESS);

    // C pointers
    int64_t rows = 0, cols = 0, nnz = 0;
    ASSERT_EQ(hipsparseSpMatGetSize(C, &rows, &cols, &nnz), HIPSPARSE_STATUS_SUCCESS);
    auto c_col = make_buffer(sizeof(int) * nnz);
    auto c_val = make_buffer(sizeof(float) * nnz);
    ASSERT_EQ(hipsparseCsrSetPointers(C, c_ptr.get(), c_col.get(), c_val.get()),
              HIPSPARSE_STATUS_SUCCESS);

    // Copy
    size_t size5 = 0;
    ASSERT_EQ(
        hipsparseSpGEMMreuse_copy(p.handle, op_n, op_n, p.A, B, C, alg, descr, &size5, nullptr),
        HIPSPARSE_STATUS_SUCCESS);
    auto buffer5 = make_buffer(size5);
    ASSERT_EQ(hipsparseSpGEMMreuse_copy(
                  p.handle, op_n, op_n, p.A, B, C, alg, descr, &size5, buffer5.get()),
              HIPSPARSE_STATUS_SUCCESS);

    auto call = [&](hipDataType t) {
        return hipsparseSpGEMMreuse_compute(
            p.handle, op_n, op_n, &p.alpha, p.A, B, &p.beta, C, t, alg, descr);
    };

    expect_rejected(call);
    EXPECT_EQ(call(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif

//
// SpGEAM
//

#if(DTV_CUDART >= 13030)
namespace
{
    // Runs the stages of SpGEAM in order. Every member that takes a type returns
    // the status of that stage.
    struct spgeam_problem
    {
        problem                   p;
        hipsparseLocalCsr         B;
        hipsparse_unique_ptr      c_ptr = make_device(std::vector<int>{0, 0, 0});
        hipsparseLocalCsr         C;
        hipsparseLocalSpGEAMDescr descr;

        float                beta_geam = 1.0f;
        size_t               size      = 0;
        hipsparse_unique_ptr buffer{nullptr, device_free};
        hipsparse_unique_ptr c_col{nullptr, device_free};
        hipsparse_unique_ptr c_val{nullptr, device_free};

        spgeam_problem()
            : B(2,
                2,
                2,
                p.ptr3.get(),
                p.idx2.get(),
                p.val2.get(),
                idx32,
                idx32,
                base0,
                supported_type)
            , C(2, 2, 0, c_ptr.get(), nullptr, nullptr, idx32, idx32, base0, supported_type)
        {
        }

        hipsparseStatus_t buffer_size(hipDataType t)
        {
            return hipsparseSpGEAM_bufferSize(p.handle,
                                              op_n,
                                              op_n,
                                              &p.alpha,
                                              p.A,
                                              &beta_geam,
                                              B,
                                              C,
                                              t,
                                              HIPSPARSE_SPGEAM_ALG1,
                                              descr,
                                              &size);
        }

        hipsparseStatus_t nnz(hipDataType t)
        {
            return hipsparseSpGEAM_nnz(p.handle,
                                       op_n,
                                       op_n,
                                       &p.alpha,
                                       p.A,
                                       &beta_geam,
                                       B,
                                       C,
                                       t,
                                       HIPSPARSE_SPGEAM_ALG1,
                                       descr,
                                       buffer.get());
        }

        hipsparseStatus_t compute(hipDataType t)
        {
            return hipsparseSpGEAM(p.handle,
                                   op_n,
                                   op_n,
                                   &p.alpha,
                                   p.A,
                                   &beta_geam,
                                   B,
                                   C,
                                   t,
                                   HIPSPARSE_SPGEAM_ALG1,
                                   descr,
                                   buffer.get());
        }

        void prepare_buffer()
        {
            ASSERT_EQ(buffer_size(supported_type), HIPSPARSE_STATUS_SUCCESS);
            buffer = make_buffer(size);
        }

        void prepare_compute()
        {
            ASSERT_NO_FATAL_FAILURE(prepare_buffer());
            ASSERT_EQ(nnz(supported_type), HIPSPARSE_STATUS_SUCCESS);

            int64_t rows = 0, cols = 0, nnz_c = 0;
            ASSERT_EQ(hipsparseSpMatGetSize(C, &rows, &cols, &nnz_c), HIPSPARSE_STATUS_SUCCESS);
            c_col = make_buffer(sizeof(int) * nnz_c);
            c_val = make_buffer(sizeof(float) * nnz_c);
            ASSERT_EQ(hipsparseCsrSetPointers(C, c_ptr.get(), c_col.get(), c_val.get()),
                      HIPSPARSE_STATUS_SUCCESS);
        }
    };
}

TEST_P(datatype_validation, hipsparseSpGEAM_bufferSize)
{
    spgeam_problem s;
    expect_rejected([&](hipDataType t) { return s.buffer_size(t); });
    EXPECT_EQ(s.buffer_size(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpGEAM_nnz)
{
    spgeam_problem s;
    ASSERT_NO_FATAL_FAILURE(s.prepare_buffer());
    expect_rejected([&](hipDataType t) { return s.nnz(t); });
    EXPECT_EQ(s.nnz(supported_type), HIPSPARSE_STATUS_SUCCESS);
}

TEST_P(datatype_validation, hipsparseSpGEAM)
{
    spgeam_problem s;
    ASSERT_NO_FATAL_FAILURE(s.prepare_compute());
    expect_rejected([&](hipDataType t) { return s.compute(t); });
    EXPECT_EQ(s.compute(supported_type), HIPSPARSE_STATUS_SUCCESS);
}
#endif

INSTANTIATE_TEST_SUITE_P(pre_checkin, datatype_validation, testing::Values(0));

#endif // DTV_CUDART > 10010
