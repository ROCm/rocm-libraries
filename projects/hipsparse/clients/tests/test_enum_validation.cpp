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

#include "hipsparse_local_handle.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>
#include <vector>

// Every public hipSPARSE routine that takes an enumeration by value (hipDataType,
// hipsparseOperation_t, hipsparseFillMode_t, hipsparseOrder_t, the algorithm
// enumerations, and so on) has one test here. Each test passes
//
//   - valid enumerators that the backend does not support (HIP_R_8U and HIP_C_16F
//     for hipDataType, HIPSPARSE_INDEX_16U on the rocSPARSE backend): the call
//     must return HIPSPARSE_STATUS_NOT_SUPPORTED,
//   - a value outside the enumeration (-1): the call must return
//     HIPSPARSE_STATUS_INVALID_VALUE.
//
// Neither call may throw. The enumerations are validated in parameter order. When
// a routine has two enumeration parameters, the test passes a bad value in each
// and checks that the status belongs to the first one in the parameter list.
//
// These calls pass a real handle and non-null placeholder pointers. A routine
// validates its enumerations before it reads any other argument, so the
// placeholders are never dereferenced.
//
// Supported enumerators run through real arguments elsewhere. The functional tests
// call the routines with real data, test_datatype_validation.cpp runs the
// hipDataType routines with real descriptors, and the round trip tests at the end
// of this file set and read back every supported value of the setters.
//
// The pre_checkin prefix is the name CI selects. TheRock matches pre_checkin/*
// and Jenkins matches *checkin*. A plain TEST() has no such prefix, so it never
// runs in either lane.
//
// Each API is guarded with the CUDART_VERSION range of its declaration in the
// public headers.

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace
{
    // Placeholder for every pointer argument. The routines under test return
    // before they read through it.
    alignas(64) unsigned char placeholder[4096];

    hipsparseHandle_t current_handle = nullptr;

    // Builds a default argument for a parameter of type T: the current handle for
    // a handle, a placeholder for any other pointer, and zero for the rest.
    template <typename T>
    T default_arg()
    {
        if constexpr(std::is_same_v<T, hipsparseHandle_t>)
        {
            return current_handle;
        }
        else if constexpr(std::is_pointer_v<T>)
        {
            return reinterpret_cast<T>(placeholder);
        }
        else
        {
            return T{};
        }
    }

    // Describes the enumeration parameter at position I: a supported value, which
    // fills the other parameters, and the valid values that the backend rejects as
    // unsupported.
    template <size_t I, typename T>
    struct enum_spec
    {
        static constexpr size_t index = I;
        using type                    = T;
        T              valid;
        std::vector<T> unsupported;
    };

    template <size_t I, typename T>
    enum_spec<I, T> spec(T valid, std::vector<T> unsupported = {})
    {
        return {valid, std::move(unsupported)};
    }

    std::vector<hipDataType> unsupported_data_types()
    {
        return {HIP_R_8U, HIP_C_16F};
    }

    // The rocSPARSE backend has no 16-bit unsigned index type. cuSPARSE does.
    template <typename T>
    std::vector<T> unsupported_index_types()
    {
#if !defined(CUDART_VERSION)
        return {static_cast<T>(HIPSPARSE_INDEX_16U)};
#else
        return {};
#endif
    }

    template <typename Ret, typename... P, typename... S>
    void check_enum_api(Ret (*fn)(P...), S... specs)
    {
        using args_t = std::tuple<P...>;

        hipsparseLocalHandle_t local_handle;

        auto make_args = [&]() {
            args_t args{default_arg<P>()...};
            ((std::get<S::index>(args) = specs.valid), ...);
            return args;
        };

        auto run = [&](args_t& args) {
            hipsparseStatus_t status = HIPSPARSE_STATUS_SUCCESS;
            EXPECT_NO_THROW(status = std::apply(fn, args));
            return status;
        };

        auto for_each_spec = [&](auto&& f) { (f(specs), ...); };

        current_handle = local_handle;

        // A value outside the enumeration, one parameter at a time.
        for_each_spec([&](auto s) {
            constexpr size_t I = decltype(s)::index;
            using T            = typename decltype(s)::type;
            SCOPED_TRACE(testing::Message() << "invalid value for parameter " << I);
            args_t args       = make_args();
            std::get<I>(args) = static_cast<T>(-1);
            EXPECT_EQ(run(args), HIPSPARSE_STATUS_INVALID_VALUE);
        });

        // A valid value that the backend does not support, one parameter at a time.
        for_each_spec([&](auto s) {
            constexpr size_t I = decltype(s)::index;
            for(auto value : s.unsupported)
            {
                SCOPED_TRACE(testing::Message() << "unsupported value " << static_cast<int>(value)
                                                << " for parameter " << I);
                args_t args       = make_args();
                std::get<I>(args) = value;
                EXPECT_EQ(run(args), HIPSPARSE_STATUS_NOT_SUPPORTED);
            }
        });

        // Two bad parameters: the status belongs to the one that comes first.
        for_each_spec([&](auto a) {
            for_each_spec([&](auto b) {
                constexpr size_t IA = decltype(a)::index;
                constexpr size_t IB = decltype(b)::index;
                using TB            = typename decltype(b)::type;
                if constexpr(IA != IB)
                {
                    // a is unsupported and b is invalid.
                    if(!a.unsupported.empty())
                    {
                        SCOPED_TRACE(testing::Message()
                                     << "parameter " << IA << " unsupported, parameter " << IB
                                     << " invalid");
                        args_t args        = make_args();
                        std::get<IA>(args) = a.unsupported.front();
                        std::get<IB>(args) = static_cast<TB>(-1);
                        EXPECT_EQ(run(args),
                                  IA < IB ? HIPSPARSE_STATUS_NOT_SUPPORTED
                                          : HIPSPARSE_STATUS_INVALID_VALUE);
                    }
                }
            });
        });

        current_handle = nullptr;
    }
}

class enum_validation : public testing::TestWithParam<int>
{
};

TEST_P(enum_validation, hipsparseSetPointerMode)
{
    check_enum_api(hipsparseSetPointerMode, spec<1>(HIPSPARSE_POINTER_MODE_HOST));
}

TEST_P(enum_validation, hipsparseSetMatType)
{
    check_enum_api(hipsparseSetMatType, spec<1>(HIPSPARSE_MATRIX_TYPE_GENERAL));
}

TEST_P(enum_validation, hipsparseSetMatFillMode)
{
    check_enum_api(hipsparseSetMatFillMode, spec<1>(HIPSPARSE_FILL_MODE_LOWER));
}

TEST_P(enum_validation, hipsparseSetMatDiagType)
{
    check_enum_api(hipsparseSetMatDiagType, spec<1>(HIPSPARSE_DIAG_TYPE_NON_UNIT));
}

TEST_P(enum_validation, hipsparseSetMatIndexBase)
{
    check_enum_api(hipsparseSetMatIndexBase, spec<1>(HIPSPARSE_INDEX_BASE_ZERO));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
      || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))))
TEST_P(enum_validation, hipsparseCreateSpVec)
{
    check_enum_api(hipsparseCreateSpVec,
                   spec<5>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<6>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<7>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseCreateConstSpVec)
{
    check_enum_api(hipsparseCreateConstSpVec,
                   spec<5>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<6>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<7>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)))
TEST_P(enum_validation, hipsparseCreateCoo)
{
    check_enum_api(hipsparseCreateCoo,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<9>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseCreateConstCoo)
{
    check_enum_api(hipsparseCreateConstCoo,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<9>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || (CUDART_VERSION >= 10010 && CUDART_VERSION < 12000))))
TEST_P(enum_validation, hipsparseCreateCooAoS)
{
    check_enum_api(hipsparseCreateCooAoS,
                   spec<6>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<7>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<8>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)))
TEST_P(enum_validation, hipsparseCreateCsr)
{
    check_enum_api(hipsparseCreateCsr,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<9>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<10>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12001)))
TEST_P(enum_validation, hipsparseCreateConstCsr)
{
    check_enum_api(hipsparseCreateConstCsr,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<9>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<10>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)))
TEST_P(enum_validation, hipsparseCreateCsc)
{
    check_enum_api(hipsparseCreateCsc,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<9>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<10>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseCreateConstCsc)
{
    check_enum_api(hipsparseCreateConstCsc,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<9>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<10>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)))
TEST_P(enum_validation, hipsparseCreateBlockedEll)
{
    check_enum_api(hipsparseCreateBlockedEll,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<9>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseCreateConstBlockedEll)
{
    check_enum_api(hipsparseCreateConstBlockedEll,
                   spec<7>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<8>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<9>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12011)))
TEST_P(enum_validation, hipsparseCreateSlicedEll)
{
    check_enum_api(hipsparseCreateSlicedEll,
                   spec<9>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<10>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<11>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<12>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12011)))
TEST_P(enum_validation, hipsparseCreateConstSlicedEll)
{
    check_enum_api(hipsparseCreateConstSlicedEll,
                   spec<9>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<10>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<11>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<12>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12011)))
TEST_P(enum_validation, hipsparseCreateBsr)
{
    check_enum_api(hipsparseCreateBsr,
                   spec<9>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<10>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<11>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<12>(HIP_R_32F, unsupported_data_types()),
                   spec<13>(HIPSPARSE_ORDER_COL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12011)))
TEST_P(enum_validation, hipsparseCreateConstBsr)
{
    check_enum_api(hipsparseCreateConstBsr,
                   spec<9>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<10>(HIPSPARSE_INDEX_32I, unsupported_index_types<hipsparseIndexType_t>()),
                   spec<11>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<12>(HIP_R_32F, unsupported_data_types()),
                   spec<13>(HIPSPARSE_ORDER_COL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpMatGetAttribute)
{
    check_enum_api(hipsparseSpMatGetAttribute, spec<1>(HIPSPARSE_SPMAT_FILL_MODE));
}
#endif

#if((((CUDART_VERSION >= 11030)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpMatGetAttribute)
{
    check_enum_api(hipsparseSpMatGetAttribute, spec<1>(HIPSPARSE_SPMAT_FILL_MODE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)))
TEST_P(enum_validation, hipsparseSpMatSetAttribute)
{
    check_enum_api(hipsparseSpMatSetAttribute, spec<1>(HIPSPARSE_SPMAT_FILL_MODE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION > 10010 \
      || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))))
TEST_P(enum_validation, hipsparseCreateDnVec)
{
    check_enum_api(hipsparseCreateDnVec, spec<3>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseCreateConstDnVec)
{
    check_enum_api(hipsparseCreateConstDnVec, spec<3>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)))
TEST_P(enum_validation, hipsparseCreateDnMat)
{
    check_enum_api(hipsparseCreateDnMat,
                   spec<5>(HIP_R_32F, unsupported_data_types()),
                   spec<6>(HIPSPARSE_ORDER_COL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseCreateConstDnMat)
{
    check_enum_api(hipsparseCreateConstDnMat,
                   spec<5>(HIP_R_32F, unsupported_data_types()),
                   spec<6>(HIPSPARSE_ORDER_COL));
}
#endif

TEST_P(enum_validation, hipsparseSbsr2csr)
{
    check_enum_api(hipsparseSbsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDbsr2csr)
{
    check_enum_api(hipsparseDbsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCbsr2csr)
{
    check_enum_api(hipsparseCbsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZbsr2csr)
{
    check_enum_api(hipsparseZbsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseXcoo2csr)
{
    check_enum_api(hipsparseXcoo2csr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}

TEST_P(enum_validation, hipsparseXcsr2bsrNnz)
{
    check_enum_api(hipsparseXcsr2bsrNnz, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseScsr2bsr)
{
    check_enum_api(hipsparseScsr2bsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDcsr2bsr)
{
    check_enum_api(hipsparseDcsr2bsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCcsr2bsr)
{
    check_enum_api(hipsparseCcsr2bsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZcsr2bsr)
{
    check_enum_api(hipsparseZcsr2bsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseXcsr2coo)
{
    check_enum_api(hipsparseXcsr2coo, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseScsr2csc)
{
    check_enum_api(
        hipsparseScsr2csc, spec<10>(HIPSPARSE_ACTION_NUMERIC), spec<11>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDcsr2csc)
{
    check_enum_api(
        hipsparseDcsr2csc, spec<10>(HIPSPARSE_ACTION_NUMERIC), spec<11>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCcsr2csc)
{
    check_enum_api(
        hipsparseCcsr2csc, spec<10>(HIPSPARSE_ACTION_NUMERIC), spec<11>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZcsr2csc)
{
    check_enum_api(
        hipsparseZcsr2csc, spec<10>(HIPSPARSE_ACTION_NUMERIC), spec<11>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)))
TEST_P(enum_validation, hipsparseCsr2cscEx2_bufferSize)
{
    check_enum_api(hipsparseCsr2cscEx2_bufferSize,
                   spec<10>(HIP_R_32F, unsupported_data_types()),
                   spec<11>(HIPSPARSE_ACTION_NUMERIC),
                   spec<12>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<13>(HIPSPARSE_CSR2CSC_ALG1));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)))
TEST_P(enum_validation, hipsparseCsr2cscEx2)
{
    check_enum_api(hipsparseCsr2cscEx2,
                   spec<10>(HIP_R_32F, unsupported_data_types()),
                   spec<11>(HIPSPARSE_ACTION_NUMERIC),
                   spec<12>(HIPSPARSE_INDEX_BASE_ZERO),
                   spec<13>(HIPSPARSE_CSR2CSC_ALG1));
}
#endif

TEST_P(enum_validation, hipsparseScsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseScsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDcsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseDcsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCcsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseCcsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZcsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseZcsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseXcsr2gebsrNnz)
{
    check_enum_api(hipsparseXcsr2gebsrNnz, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseScsr2gebsr)
{
    check_enum_api(hipsparseScsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDcsr2gebsr)
{
    check_enum_api(hipsparseDcsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCcsr2gebsr)
{
    check_enum_api(hipsparseCcsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZcsr2gebsr)
{
    check_enum_api(hipsparseZcsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseScsr2hyb)
{
    check_enum_api(hipsparseScsr2hyb, spec<9>(HIPSPARSE_HYB_PARTITION_AUTO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDcsr2hyb)
{
    check_enum_api(hipsparseDcsr2hyb, spec<9>(HIPSPARSE_HYB_PARTITION_AUTO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCcsr2hyb)
{
    check_enum_api(hipsparseCcsr2hyb, spec<9>(HIPSPARSE_HYB_PARTITION_AUTO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZcsr2hyb)
{
    check_enum_api(hipsparseZcsr2hyb, spec<9>(HIPSPARSE_HYB_PARTITION_AUTO));
}
#endif

TEST_P(enum_validation, hipsparseSgebsr2csr)
{
    check_enum_api(hipsparseSgebsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDgebsr2csr)
{
    check_enum_api(hipsparseDgebsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCgebsr2csr)
{
    check_enum_api(hipsparseCgebsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZgebsr2csr)
{
    check_enum_api(hipsparseZgebsr2csr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseSgebsr2gebsc)
{
    check_enum_api(hipsparseSgebsr2gebsc,
                   spec<12>(HIPSPARSE_ACTION_NUMERIC),
                   spec<13>(HIPSPARSE_INDEX_BASE_ZERO));
}

TEST_P(enum_validation, hipsparseDgebsr2gebsc)
{
    check_enum_api(hipsparseDgebsr2gebsc,
                   spec<12>(HIPSPARSE_ACTION_NUMERIC),
                   spec<13>(HIPSPARSE_INDEX_BASE_ZERO));
}

TEST_P(enum_validation, hipsparseCgebsr2gebsc)
{
    check_enum_api(hipsparseCgebsr2gebsc,
                   spec<12>(HIPSPARSE_ACTION_NUMERIC),
                   spec<13>(HIPSPARSE_INDEX_BASE_ZERO));
}

TEST_P(enum_validation, hipsparseZgebsr2gebsc)
{
    check_enum_api(hipsparseZgebsr2gebsc,
                   spec<12>(HIPSPARSE_ACTION_NUMERIC),
                   spec<13>(HIPSPARSE_INDEX_BASE_ZERO));
}

TEST_P(enum_validation, hipsparseSgebsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseSgebsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDgebsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseDgebsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCgebsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseCgebsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZgebsr2gebsr_bufferSize)
{
    check_enum_api(hipsparseZgebsr2gebsr_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseXgebsr2gebsrNnz)
{
    check_enum_api(hipsparseXgebsr2gebsrNnz, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseSgebsr2gebsr)
{
    check_enum_api(hipsparseSgebsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDgebsr2gebsr)
{
    check_enum_api(hipsparseDgebsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCgebsr2gebsr)
{
    check_enum_api(hipsparseCgebsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZgebsr2gebsr)
{
    check_enum_api(hipsparseZgebsr2gebsr, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseSnnz)
{
    check_enum_api(hipsparseSnnz, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseDnnz)
{
    check_enum_api(hipsparseDnnz, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseCnnz)
{
    check_enum_api(hipsparseCnnz, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

TEST_P(enum_validation, hipsparseZnnz)
{
    check_enum_api(hipsparseZnnz, spec<1>(HIPSPARSE_DIRECTION_ROW));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseXcsrgemmNnz)
{
    check_enum_api(hipsparseXcsrgemmNnz,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseScsrgemm)
{
    check_enum_api(hipsparseScsrgemm,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDcsrgemm)
{
    check_enum_api(hipsparseDcsrgemm,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCcsrgemm)
{
    check_enum_api(hipsparseCcsrgemm,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZcsrgemm)
{
    check_enum_api(hipsparseZcsrgemm,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseDenseToSparse_bufferSize)
{
    check_enum_api(hipsparseDenseToSparse_bufferSize,
                   spec<3>(static_cast<hipsparseDenseToSparseAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11020)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseDenseToSparse_bufferSize)
{
    check_enum_api(hipsparseDenseToSparse_bufferSize,
                   spec<3>(static_cast<hipsparseDenseToSparseAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseDenseToSparse_analysis)
{
    check_enum_api(hipsparseDenseToSparse_analysis,
                   spec<3>(static_cast<hipsparseDenseToSparseAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11020)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseDenseToSparse_analysis)
{
    check_enum_api(hipsparseDenseToSparse_analysis,
                   spec<3>(static_cast<hipsparseDenseToSparseAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseDenseToSparse_convert)
{
    check_enum_api(hipsparseDenseToSparse_convert,
                   spec<3>(static_cast<hipsparseDenseToSparseAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11020)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseDenseToSparse_convert)
{
    check_enum_api(hipsparseDenseToSparse_convert,
                   spec<3>(static_cast<hipsparseDenseToSparseAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSDDMM_bufferSize)
{
    check_enum_api(hipsparseSDDMM_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSDDMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11022)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSDDMM_bufferSize)
{
    check_enum_api(hipsparseSDDMM_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSDDMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSDDMM_preprocess)
{
    check_enum_api(hipsparseSDDMM_preprocess,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSDDMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11022)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSDDMM_preprocess)
{
    check_enum_api(hipsparseSDDMM_preprocess,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSDDMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSDDMM)
{
    check_enum_api(hipsparseSDDMM,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSDDMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11022)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSDDMM)
{
    check_enum_api(hipsparseSDDMM,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSDDMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSparseToDense_bufferSize)
{
    check_enum_api(hipsparseSparseToDense_bufferSize,
                   spec<3>(static_cast<hipsparseSparseToDenseAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11020)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSparseToDense_bufferSize)
{
    check_enum_api(hipsparseSparseToDense_bufferSize,
                   spec<3>(static_cast<hipsparseSparseToDenseAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSparseToDense)
{
    check_enum_api(hipsparseSparseToDense, spec<3>(static_cast<hipsparseSparseToDenseAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11020)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSparseToDense)
{
    check_enum_api(hipsparseSparseToDense, spec<3>(static_cast<hipsparseSparseToDenseAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 13030)))
TEST_P(enum_validation, hipsparseSpGEAM_bufferSize)
{
    check_enum_api(hipsparseSpGEAM_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEAMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 13030)))
TEST_P(enum_validation, hipsparseSpGEAM_nnz)
{
    check_enum_api(hipsparseSpGEAM_nnz,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEAMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 13030)))
TEST_P(enum_validation, hipsparseSpGEAM)
{
    check_enum_api(hipsparseSpGEAM,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEAMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpGEMM_workEstimation)
{
    check_enum_api(hipsparseSpGEMM_workEstimation,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11000)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpGEMM_workEstimation)
{
    check_enum_api(hipsparseSpGEMM_workEstimation,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpGEMM_compute)
{
    check_enum_api(hipsparseSpGEMM_compute,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11000)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpGEMM_compute)
{
    check_enum_api(hipsparseSpGEMM_compute,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpGEMM_copy)
{
    check_enum_api(hipsparseSpGEMM_copy,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11000)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpGEMM_copy)
{
    check_enum_api(hipsparseSpGEMM_copy,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpGEMMreuse_workEstimation)
{
    check_enum_api(hipsparseSpGEMMreuse_workEstimation,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpGEMMreuse_workEstimation)
{
    check_enum_api(hipsparseSpGEMMreuse_workEstimation,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpGEMMreuse_nnz)
{
    check_enum_api(hipsparseSpGEMMreuse_nnz,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpGEMMreuse_nnz)
{
    check_enum_api(hipsparseSpGEMMreuse_nnz,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpGEMMreuse_copy)
{
    check_enum_api(hipsparseSpGEMMreuse_copy,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpGEMMreuse_copy)
{
    check_enum_api(hipsparseSpGEMMreuse_copy,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpGEMMreuse_compute)
{
    check_enum_api(hipsparseSpGEMMreuse_compute,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpGEMMreuse_compute)
{
    check_enum_api(hipsparseSpGEMMreuse_compute,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpGEMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpMM_bufferSize)
{
    check_enum_api(hipsparseSpMM_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 10010)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpMM_bufferSize)
{
    check_enum_api(hipsparseSpMM_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpMM_preprocess)
{
    check_enum_api(hipsparseSpMM_preprocess,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11021)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpMM_preprocess)
{
    check_enum_api(hipsparseSpMM_preprocess,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpMM)
{
    check_enum_api(hipsparseSpMM,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpMMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 10010)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpMM)
{
    check_enum_api(hipsparseSpMM,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<8>(HIP_R_32F, unsupported_data_types()),
                   spec<9>(static_cast<hipsparseSpMMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpMV_bufferSize)
{
    check_enum_api(hipsparseSpMV_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpMVAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))) \
     && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpMV_bufferSize)
{
    check_enum_api(hipsparseSpMV_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpMVAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpMV_preprocess)
{
    check_enum_api(hipsparseSpMV_preprocess,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpMVAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))) \
     && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpMV_preprocess)
{
    check_enum_api(hipsparseSpMV_preprocess,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpMVAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpMV)
{
    check_enum_api(hipsparseSpMV,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpMVAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))) \
     && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpMV)
{
    check_enum_api(hipsparseSpMV,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpMVAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpSM_bufferSize)
{
    check_enum_api(hipsparseSpSM_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpSM_bufferSize)
{
    check_enum_api(hipsparseSpSM_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpSM_analysis)
{
    check_enum_api(hipsparseSpSM_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpSM_analysis)
{
    check_enum_api(hipsparseSpSM_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpSM_solve)
{
    check_enum_api(hipsparseSpSM_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpSM_solve)
{
    check_enum_api(hipsparseSpSM_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpSM_solve_ex)
{
    check_enum_api(hipsparseSpSM_solve_ex,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11031)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpSM_solve_ex)
{
    check_enum_api(hipsparseSpSM_solve_ex,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<7>(HIP_R_32F, unsupported_data_types()),
                   spec<8>(static_cast<hipsparseSpSMAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpSV_bufferSize)
{
    check_enum_api(hipsparseSpSV_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(HIP_R_32F, unsupported_data_types()),
                   spec<7>(static_cast<hipsparseSpSVAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11030)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpSV_bufferSize)
{
    check_enum_api(hipsparseSpSV_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(HIP_R_32F, unsupported_data_types()),
                   spec<7>(static_cast<hipsparseSpSVAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpSV_analysis)
{
    check_enum_api(hipsparseSpSV_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(HIP_R_32F, unsupported_data_types()),
                   spec<7>(static_cast<hipsparseSpSVAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11030)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpSV_analysis)
{
    check_enum_api(hipsparseSpSV_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(HIP_R_32F, unsupported_data_types()),
                   spec<7>(static_cast<hipsparseSpSVAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpSV_solve)
{
    check_enum_api(hipsparseSpSV_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(HIP_R_32F, unsupported_data_types()),
                   spec<7>(static_cast<hipsparseSpSVAlg_t>(0)));
}
#endif

#if((((CUDART_VERSION >= 11030)) && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpSV_solve)
{
    check_enum_api(hipsparseSpSV_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<6>(HIP_R_32F, unsupported_data_types()),
                   spec<7>(static_cast<hipsparseSpSVAlg_t>(0)));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpVV_bufferSize)
{
    check_enum_api(hipsparseSpVV_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<5>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if((((CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))) \
     && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpVV_bufferSize)
{
    check_enum_api(hipsparseSpVV_bufferSize,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<5>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000)))
TEST_P(enum_validation, hipsparseSpVV)
{
    check_enum_api(hipsparseSpVV,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<5>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if((((CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))) \
     && !((!defined(CUDART_VERSION) || CUDART_VERSION >= 12000))))
TEST_P(enum_validation, hipsparseSpVV)
{
    check_enum_api(hipsparseSpVV,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<5>(HIP_R_32F, unsupported_data_types()));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseSaxpyi)
{
    check_enum_api(hipsparseSaxpyi, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDaxpyi)
{
    check_enum_api(hipsparseDaxpyi, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCaxpyi)
{
    check_enum_api(hipsparseCaxpyi, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZaxpyi)
{
    check_enum_api(hipsparseZaxpyi, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCdotci)
{
    check_enum_api(hipsparseCdotci, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZdotci)
{
    check_enum_api(hipsparseZdotci, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseSdoti)
{
    check_enum_api(hipsparseSdoti, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDdoti)
{
    check_enum_api(hipsparseDdoti, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCdoti)
{
    check_enum_api(hipsparseCdoti, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZdoti)
{
    check_enum_api(hipsparseZdoti, spec<6>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseSgthr)
{
    check_enum_api(hipsparseSgthr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDgthr)
{
    check_enum_api(hipsparseDgthr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCgthr)
{
    check_enum_api(hipsparseCgthr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZgthr)
{
    check_enum_api(hipsparseZgthr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseSgthrz)
{
    check_enum_api(hipsparseSgthrz, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDgthrz)
{
    check_enum_api(hipsparseDgthrz, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCgthrz)
{
    check_enum_api(hipsparseCgthrz, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZgthrz)
{
    check_enum_api(hipsparseZgthrz, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseSroti)
{
    check_enum_api(hipsparseSroti, spec<7>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDroti)
{
    check_enum_api(hipsparseDroti, spec<7>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseSsctr)
{
    check_enum_api(hipsparseSsctr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDsctr)
{
    check_enum_api(hipsparseDsctr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCsctr)
{
    check_enum_api(hipsparseCsctr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZsctr)
{
    check_enum_api(hipsparseZsctr, spec<5>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

TEST_P(enum_validation, hipsparseSbsrmv)
{
    check_enum_api(hipsparseSbsrmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseDbsrmv)
{
    check_enum_api(hipsparseDbsrmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseCbsrmv)
{
    check_enum_api(hipsparseCbsrmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseZbsrmv)
{
    check_enum_api(hipsparseZbsrmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrsv2_bufferSize)
{
    check_enum_api(hipsparseSbsrsv2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrsv2_bufferSize)
{
    check_enum_api(hipsparseDbsrsv2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrsv2_bufferSize)
{
    check_enum_api(hipsparseCbsrsv2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrsv2_bufferSize)
{
    check_enum_api(hipsparseZbsrsv2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

TEST_P(enum_validation, hipsparseSbsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseSbsrsv2_bufferSizeExt,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseDbsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseDbsrsv2_bufferSizeExt,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseCbsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseCbsrsv2_bufferSizeExt,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseZbsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseZbsrsv2_bufferSizeExt,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrsv2_analysis)
{
    check_enum_api(hipsparseSbsrsv2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<11>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrsv2_analysis)
{
    check_enum_api(hipsparseDbsrsv2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<11>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrsv2_analysis)
{
    check_enum_api(hipsparseCbsrsv2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<11>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrsv2_analysis)
{
    check_enum_api(hipsparseZbsrsv2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<11>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrsv2_solve)
{
    check_enum_api(hipsparseSbsrsv2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<14>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrsv2_solve)
{
    check_enum_api(hipsparseDbsrsv2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<14>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrsv2_solve)
{
    check_enum_api(hipsparseCbsrsv2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<14>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrsv2_solve)
{
    check_enum_api(hipsparseZbsrsv2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<14>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrxmv)
{
    check_enum_api(hipsparseSbsrxmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrxmv)
{
    check_enum_api(hipsparseDbsrxmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrxmv)
{
    check_enum_api(hipsparseCbsrxmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrxmv)
{
    check_enum_api(hipsparseZbsrxmv,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseScsrmv)
{
    check_enum_api(hipsparseScsrmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDcsrmv)
{
    check_enum_api(hipsparseDcsrmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCcsrmv)
{
    check_enum_api(hipsparseCcsrmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZcsrmv)
{
    check_enum_api(hipsparseZcsrmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseScsrsv2_bufferSize)
{
    check_enum_api(hipsparseScsrsv2_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDcsrsv2_bufferSize)
{
    check_enum_api(hipsparseDcsrsv2_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCcsrsv2_bufferSize)
{
    check_enum_api(hipsparseCcsrsv2_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZcsrsv2_bufferSize)
{
    check_enum_api(hipsparseZcsrsv2_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

TEST_P(enum_validation, hipsparseScsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseScsrsv2_bufferSizeExt, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseDcsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseDcsrsv2_bufferSizeExt, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseCcsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseCcsrsv2_bufferSizeExt, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseZcsrsv2_bufferSizeExt)
{
    check_enum_api(hipsparseZcsrsv2_bufferSizeExt, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseScsrsv2_analysis)
{
    check_enum_api(hipsparseScsrsv2_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<9>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDcsrsv2_analysis)
{
    check_enum_api(hipsparseDcsrsv2_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<9>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCcsrsv2_analysis)
{
    check_enum_api(hipsparseCcsrsv2_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<9>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZcsrsv2_analysis)
{
    check_enum_api(hipsparseZcsrsv2_analysis,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<9>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseScsrsv2_solve)
{
    check_enum_api(hipsparseScsrsv2_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDcsrsv2_solve)
{
    check_enum_api(hipsparseDcsrsv2_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCcsrsv2_solve)
{
    check_enum_api(hipsparseCcsrsv2_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZcsrsv2_solve)
{
    check_enum_api(hipsparseZcsrsv2_solve,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseSgemvi_bufferSize)
{
    check_enum_api(hipsparseSgemvi_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDgemvi_bufferSize)
{
    check_enum_api(hipsparseDgemvi_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCgemvi_bufferSize)
{
    check_enum_api(hipsparseCgemvi_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZgemvi_bufferSize)
{
    check_enum_api(hipsparseZgemvi_bufferSize, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseSgemvi)
{
    check_enum_api(hipsparseSgemvi,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDgemvi)
{
    check_enum_api(hipsparseDgemvi,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCgemvi)
{
    check_enum_api(hipsparseCgemvi,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZgemvi)
{
    check_enum_api(hipsparseZgemvi,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<12>(HIPSPARSE_INDEX_BASE_ZERO));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseShybmv)
{
    check_enum_api(hipsparseShybmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDhybmv)
{
    check_enum_api(hipsparseDhybmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseChybmv)
{
    check_enum_api(hipsparseChybmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZhybmv)
{
    check_enum_api(hipsparseZhybmv, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

TEST_P(enum_validation, hipsparseSbsrmm)
{
    check_enum_api(hipsparseSbsrmm,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseDbsrmm)
{
    check_enum_api(hipsparseDbsrmm,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseCbsrmm)
{
    check_enum_api(hipsparseCbsrmm,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

TEST_P(enum_validation, hipsparseZbsrmm)
{
    check_enum_api(hipsparseZbsrmm,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrsm2_bufferSize)
{
    check_enum_api(hipsparseSbsrsm2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrsm2_bufferSize)
{
    check_enum_api(hipsparseDbsrsm2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrsm2_bufferSize)
{
    check_enum_api(hipsparseCbsrsm2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrsm2_bufferSize)
{
    check_enum_api(hipsparseZbsrsm2_bufferSize,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrsm2_analysis)
{
    check_enum_api(hipsparseSbsrsm2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<13>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrsm2_analysis)
{
    check_enum_api(hipsparseDbsrsm2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<13>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrsm2_analysis)
{
    check_enum_api(hipsparseCbsrsm2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<13>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrsm2_analysis)
{
    check_enum_api(hipsparseZbsrsm2_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<13>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrsm2_solve)
{
    check_enum_api(hipsparseSbsrsm2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<18>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrsm2_solve)
{
    check_enum_api(hipsparseDbsrsm2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<18>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrsm2_solve)
{
    check_enum_api(hipsparseCbsrsm2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<18>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrsm2_solve)
{
    check_enum_api(hipsparseZbsrsm2_solve,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<18>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseScsrmm)
{
    check_enum_api(hipsparseScsrmm, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDcsrmm)
{
    check_enum_api(hipsparseDcsrmm, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCcsrmm)
{
    check_enum_api(hipsparseCcsrmm, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZcsrmm)
{
    check_enum_api(hipsparseZcsrmm, spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseScsrmm2)
{
    check_enum_api(hipsparseScsrmm2,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseDcsrmm2)
{
    check_enum_api(hipsparseDcsrmm2,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseCcsrmm2)
{
    check_enum_api(hipsparseCcsrmm2,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 11000)))
TEST_P(enum_validation, hipsparseZcsrmm2)
{
    check_enum_api(hipsparseZcsrmm2,
                   spec<1>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseScsrsm2_bufferSizeExt)
{
    check_enum_api(hipsparseScsrsm2_bufferSizeExt,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDcsrsm2_bufferSizeExt)
{
    check_enum_api(hipsparseDcsrsm2_bufferSizeExt,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCcsrsm2_bufferSizeExt)
{
    check_enum_api(hipsparseCcsrsm2_bufferSizeExt,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZcsrsm2_bufferSizeExt)
{
    check_enum_api(hipsparseZcsrsm2_bufferSizeExt,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseScsrsm2_analysis)
{
    check_enum_api(hipsparseScsrsm2_analysis,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDcsrsm2_analysis)
{
    check_enum_api(hipsparseDcsrsm2_analysis,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCcsrsm2_analysis)
{
    check_enum_api(hipsparseCcsrsm2_analysis,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZcsrsm2_analysis)
{
    check_enum_api(hipsparseZcsrsm2_analysis,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseScsrsm2_solve)
{
    check_enum_api(hipsparseScsrsm2_solve,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseDcsrsm2_solve)
{
    check_enum_api(hipsparseDcsrsm2_solve,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseCcsrsm2_solve)
{
    check_enum_api(hipsparseCcsrsm2_solve,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 12000)))
TEST_P(enum_validation, hipsparseZcsrsm2_solve)
{
    check_enum_api(hipsparseZcsrsm2_solve,
                   spec<2>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<3>(HIPSPARSE_OPERATION_NON_TRANSPOSE),
                   spec<15>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsric02_bufferSize)
{
    check_enum_api(hipsparseSbsric02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsric02_bufferSize)
{
    check_enum_api(hipsparseDbsric02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsric02_bufferSize)
{
    check_enum_api(hipsparseCbsric02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsric02_bufferSize)
{
    check_enum_api(hipsparseZbsric02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsric02_analysis)
{
    check_enum_api(hipsparseSbsric02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsric02_analysis)
{
    check_enum_api(hipsparseDbsric02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsric02_analysis)
{
    check_enum_api(hipsparseCbsric02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsric02_analysis)
{
    check_enum_api(hipsparseZbsric02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsric02)
{
    check_enum_api(hipsparseSbsric02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsric02)
{
    check_enum_api(hipsparseDbsric02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsric02)
{
    check_enum_api(hipsparseCbsric02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsric02)
{
    check_enum_api(hipsparseZbsric02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrilu02_bufferSize)
{
    check_enum_api(hipsparseSbsrilu02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrilu02_bufferSize)
{
    check_enum_api(hipsparseDbsrilu02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrilu02_bufferSize)
{
    check_enum_api(hipsparseCbsrilu02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrilu02_bufferSize)
{
    check_enum_api(hipsparseZbsrilu02_bufferSize, spec<1>(HIPSPARSE_DIRECTION_ROW));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrilu02_analysis)
{
    check_enum_api(hipsparseSbsrilu02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrilu02_analysis)
{
    check_enum_api(hipsparseDbsrilu02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrilu02_analysis)
{
    check_enum_api(hipsparseCbsrilu02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrilu02_analysis)
{
    check_enum_api(hipsparseZbsrilu02_analysis,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseSbsrilu02)
{
    check_enum_api(hipsparseSbsrilu02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDbsrilu02)
{
    check_enum_api(hipsparseDbsrilu02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCbsrilu02)
{
    check_enum_api(hipsparseCbsrilu02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZbsrilu02)
{
    check_enum_api(hipsparseZbsrilu02,
                   spec<1>(HIPSPARSE_DIRECTION_ROW),
                   spec<10>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseScsric02_analysis)
{
    check_enum_api(hipsparseScsric02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDcsric02_analysis)
{
    check_enum_api(hipsparseDcsric02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCcsric02_analysis)
{
    check_enum_api(hipsparseCcsric02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZcsric02_analysis)
{
    check_enum_api(hipsparseZcsric02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseScsric02)
{
    check_enum_api(hipsparseScsric02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDcsric02)
{
    check_enum_api(hipsparseDcsric02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCcsric02)
{
    check_enum_api(hipsparseCcsric02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZcsric02)
{
    check_enum_api(hipsparseZcsric02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseScsrilu02_analysis)
{
    check_enum_api(hipsparseScsrilu02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDcsrilu02_analysis)
{
    check_enum_api(hipsparseDcsrilu02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCcsrilu02_analysis)
{
    check_enum_api(hipsparseCcsrilu02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZcsrilu02_analysis)
{
    check_enum_api(hipsparseZcsrilu02_analysis, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseScsrilu02)
{
    check_enum_api(hipsparseScsrilu02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseDcsrilu02)
{
    check_enum_api(hipsparseDcsrilu02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseCcsrilu02)
{
    check_enum_api(hipsparseCcsrilu02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

#if(((!defined(CUDART_VERSION) || CUDART_VERSION < 13000)))
TEST_P(enum_validation, hipsparseZcsrilu02)
{
    check_enum_api(hipsparseZcsrilu02, spec<8>(HIPSPARSE_SOLVE_POLICY_NO_LEVEL));
}
#endif

// The setters take an enumeration and have an observable effect. Every supported
// value must succeed and read back unchanged, and an invalid value must leave the
// stored value alone.
TEST_P(enum_validation, supported_values_round_trip)
{
    hipsparseLocalHandle_t handle;

    for(hipsparsePointerMode_t mode : {HIPSPARSE_POINTER_MODE_HOST, HIPSPARSE_POINTER_MODE_DEVICE})
    {
        hipsparsePointerMode_t got = HIPSPARSE_POINTER_MODE_HOST;
        EXPECT_EQ(hipsparseSetPointerMode(handle, mode), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseGetPointerMode(handle, &got), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(got, mode);
    }
    EXPECT_EQ(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST),
              HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipsparseSetPointerMode(handle, static_cast<hipsparsePointerMode_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    {
        hipsparsePointerMode_t got = HIPSPARSE_POINTER_MODE_DEVICE;
        EXPECT_EQ(hipsparseGetPointerMode(handle, &got), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(got, HIPSPARSE_POINTER_MODE_HOST);
    }

    hipsparseMatDescr_t descr = nullptr;
    ASSERT_EQ(hipsparseCreateMatDescr(&descr), HIPSPARSE_STATUS_SUCCESS);

    for(hipsparseMatrixType_t type : {HIPSPARSE_MATRIX_TYPE_GENERAL,
                                      HIPSPARSE_MATRIX_TYPE_SYMMETRIC,
                                      HIPSPARSE_MATRIX_TYPE_HERMITIAN,
                                      HIPSPARSE_MATRIX_TYPE_TRIANGULAR})
    {
        EXPECT_EQ(hipsparseSetMatType(descr, type), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseGetMatType(descr), type);
    }
    EXPECT_EQ(hipsparseSetMatType(descr, static_cast<hipsparseMatrixType_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseGetMatType(descr), HIPSPARSE_MATRIX_TYPE_TRIANGULAR);

    for(hipsparseFillMode_t mode : {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER})
    {
        EXPECT_EQ(hipsparseSetMatFillMode(descr, mode), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseGetMatFillMode(descr), mode);
    }
    EXPECT_EQ(hipsparseSetMatFillMode(descr, static_cast<hipsparseFillMode_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseGetMatFillMode(descr), HIPSPARSE_FILL_MODE_UPPER);

    for(hipsparseDiagType_t diag : {HIPSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_NON_UNIT})
    {
        EXPECT_EQ(hipsparseSetMatDiagType(descr, diag), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseGetMatDiagType(descr), diag);
    }
    EXPECT_EQ(hipsparseSetMatDiagType(descr, static_cast<hipsparseDiagType_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseGetMatDiagType(descr), HIPSPARSE_DIAG_TYPE_NON_UNIT);

    for(hipsparseIndexBase_t base : {HIPSPARSE_INDEX_BASE_ZERO, HIPSPARSE_INDEX_BASE_ONE})
    {
        EXPECT_EQ(hipsparseSetMatIndexBase(descr, base), HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseGetMatIndexBase(descr), base);
    }
    EXPECT_EQ(hipsparseSetMatIndexBase(descr, static_cast<hipsparseIndexBase_t>(-1)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseGetMatIndexBase(descr), HIPSPARSE_INDEX_BASE_ONE);

    EXPECT_EQ(hipsparseDestroyMatDescr(descr), HIPSPARSE_STATUS_SUCCESS);
}

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
TEST_P(enum_validation, supported_values_round_trip_spmat_attribute)
{
    void* buffer = nullptr;
    ASSERT_EQ(hipMalloc(&buffer, 64), hipSuccess);

    hipsparseSpMatDescr_t mat = nullptr;
    ASSERT_EQ(hipsparseCreateCsr(&mat,
                                 2,
                                 2,
                                 2,
                                 buffer,
                                 buffer,
                                 buffer,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_32I,
                                 HIPSPARSE_INDEX_BASE_ZERO,
                                 HIP_R_32F),
              HIPSPARSE_STATUS_SUCCESS);

    for(hipsparseFillMode_t fill : {HIPSPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_UPPER})
    {
        hipsparseFillMode_t got = HIPSPARSE_FILL_MODE_LOWER;
        EXPECT_EQ(hipsparseSpMatSetAttribute(mat, HIPSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)),
                  HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseSpMatGetAttribute(mat, HIPSPARSE_SPMAT_FILL_MODE, &got, sizeof(got)),
                  HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(got, fill);
    }

    for(hipsparseDiagType_t diag : {HIPSPARSE_DIAG_TYPE_UNIT, HIPSPARSE_DIAG_TYPE_NON_UNIT})
    {
        hipsparseDiagType_t got = HIPSPARSE_DIAG_TYPE_UNIT;
        EXPECT_EQ(hipsparseSpMatSetAttribute(mat, HIPSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)),
                  HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(hipsparseSpMatGetAttribute(mat, HIPSPARSE_SPMAT_DIAG_TYPE, &got, sizeof(got)),
                  HIPSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(got, diag);
    }

    const hipsparseSpMatAttribute_t invalid = static_cast<hipsparseSpMatAttribute_t>(-1);
    int                             value   = 0;
    EXPECT_EQ(hipsparseSpMatSetAttribute(mat, invalid, &value, sizeof(value)),
              HIPSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(hipsparseSpMatGetAttribute(mat, invalid, &value, sizeof(value)),
              HIPSPARSE_STATUS_INVALID_VALUE);

    EXPECT_EQ(hipsparseDestroySpMat(mat), HIPSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(hipFree(buffer), hipSuccess);
}
#endif

INSTANTIATE_TEST_SUITE_P(pre_checkin, enum_validation, testing::Values(0));
