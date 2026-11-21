// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/hostnumerics/Types.hpp>
#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/gemm.hpp>

#include <gtest/gtest.h>

#include <span>
#include <vector>

// The FP64 CPU reference must compute in double: a float accumulator gives about 1e-8 relative
// error with non-integer data and alpha, which FP64 results checked at double tolerances fail.
// This follows the reference's composition for alpha != 1: the product in the accumulator type,
// then an epilogue that scales it by alpha. K = 1 and beta = 0, so D = alpha * a * b up to double
// rounding.
TEST(HostUnitMatmulReference, smoke_Fp64GemmKeepsDoublePrecision)
{
    using namespace roc::hostnumerics;

    const ScalarType accumulator = hipblaslt::hostnumerics::referenceAccumulatorType(HIP_R_64F);
    ASSERT_EQ(accumulator, ScalarType::Float64);

    constexpr size_t    m = 4, n = 3, k = 1;
    std::vector<double> aValues(m * k), bValues(k * n);
    for(size_t i = 0; i < m; i++)
        aValues[i] = 0.1 + 1e-3 * double(i);
    for(size_t j = 0; j < n; j++)
        bValues[j] = 0.3 - 1e-3 * double(j);
    const double alphaValue = 0.1;

    const Tensor a = Tensor::copyNativeStorage(Layout::contiguousFirstDimensionFastest(Shape{m, k}),
                                               std::span<const double>(aValues));
    const Tensor b = Tensor::copyNativeStorage(Layout::contiguousFirstDimensionFastest(Shape{k, n}),
                                               std::span<const double>(bValues));

    for(GemmBackend backend : {GemmBackend::Blocked, GemmBackend::Automatic})
    {
        SCOPED_TRACE(static_cast<int>(backend));
        const Layout layout = Layout::contiguousFirstDimensionFastest(Shape{m, n});
        Tensor       product(accumulator, layout);
        matmulInto(a,
                   b,
                   product,
                   MatmulOptions(accumulator),
                   IndexRange::all(product.elementCount()),
                   LastDimensionFastest{},
                   backend);
        EpilogueOptions conversion(accumulator);
        conversion.inputScale = Tensor::scalar(accumulator, alphaValue);
        Tensor d(ScalarType::Float64, layout);
        referenceEpilogueInto(product, {.output = d}, conversion);

        for(size_t j = 0; j < n; j++)
            for(size_t i = 0; i < m; i++)
                EXPECT_DOUBLE_EQ(d.loadAs<double>({i, j}), alphaValue * aValues[i] * bValues[j])
                    << "i=" << i << " j=" << j;
    }
}
