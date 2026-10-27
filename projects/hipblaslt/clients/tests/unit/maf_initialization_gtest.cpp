// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <TensileLite/Client/hostnumerics/MafInitialization.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <hipblaslt/hostnumerics/HipblasltDataInitialization.hpp>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace hn     = roc::hostnumerics;
namespace client = hipblaslt::hostnumerics;

TEST(HostUnitMafInitialization, LogicalBatchIndexAndBroadcast)
{
    hn::Tensor tensor(hn::ScalarType::Float64, hn::Layout(hn::Shape{3, 4, 2}, {1, 6, 30}, 0));
    std::ranges::fill(tensor.rawEncodedBackingStorage(), std::byte{0xa5});
    client::initializeMatrix(tensor,
                             client::MatrixRole::A,
                             hipblaslt_initialization::maf1_0,
                             0,
                             false,
                             std::nullopt,
                             false,
                             {});
    for(size_t b = 0; b < 2; ++b)
        for(size_t j = 0; j < 4; ++j)
            for(size_t i = 0; i < 3; ++i)
                EXPECT_EQ(tensor.loadAs<double>({i, j, b}), std::sin(double(i + 3 * j + 12 * b)));
    EXPECT_EQ(tensor.rawEncodedBackingStorage()[3 * sizeof(double)], std::byte{0xa5});
    hn::Tensor broadcast(hn::ScalarType::Float64, hn::Layout(hn::Shape{3, 4, 2}, {1, 3, 0}, 0));
    client::initializeMatrix(broadcast,
                             client::MatrixRole::B,
                             hipblaslt_initialization::maf1_0,
                             0,
                             false,
                             std::nullopt,
                             false,
                             {});
    EXPECT_EQ(broadcast.loadAs<double>({1, 0, 1}), std::cos(1.0));
    hn::Tensor batch(hn::ScalarType::Float64, hn::Shape{3, 4});
    client::initializeMatrix(batch,
                             client::MatrixRole::A,
                             hipblaslt_initialization::maf1_0,
                             0,
                             false,
                             std::nullopt,
                             false,
                             {},
                             1);
    EXPECT_EQ(batch.loadAs<double>({1, 0}), tensor.loadAs<double>({1, 0, 1}));
}

TEST(HostUnitMafInitialization, IndependentCodesScalesAndTailBlocks)
{
    auto tensor = client::generateMxData(HIP_R_4F_E2M1,
                                         HIP_R_8F_UE8M0,
                                         hn::Shape{40, 4},
                                         40,
                                         0,
                                         32,
                                         hipblaslt_initialization::maf1_0,
                                         0,
                                         client::MatrixRole::A);
    EXPECT_EQ(tensor.scales.elementCount(), 8);
    EXPECT_EQ(tensor.data.loadAs<double>({1, 0}), 1.0);
    EXPECT_EQ(tensor.reference.loadAs<double>({1, 0}), 0.5);
    auto unit = client::generateMxData(HIP_R_4F_E2M1,
                                       HIP_R_8F_E4M3,
                                       hn::Shape{32, 3},
                                       32,
                                       0,
                                       16,
                                       hipblaslt_initialization::maf2_0_unit_scale,
                                       0,
                                       client::MatrixRole::A);
    for(size_t i = 0; i < unit.scales.elementCount(); ++i)
        EXPECT_EQ(
            unit.scales.loadAs<double>(unit.scales.shape().coordinatesLastDimensionFastest(i)),
            1.0);
    EXPECT_THROW(client::generateMxData(HIP_R_4F_E2M1,
                                        HIP_R_8F_E4M3,
                                        hn::Shape{32, 3},
                                        32,
                                        0,
                                        32,
                                        hipblaslt_initialization::maf2_0,
                                        0,
                                        client::MatrixRole::A),
                 std::runtime_error);
}

TEST(HostUnitMafInitialization, NonScaledFp8UsesNormalAndRejectsIntegers)
{
    hn::Tensor data(hn::ScalarType::Float8E4M3, hn::Shape{32768, 1});
    client::initializeMatrix(data,
                             client::MatrixRole::A,
                             hipblaslt_initialization::maf2_0,
                             0,
                             false,
                             std::nullopt,
                             false,
                             {});
    double squares = 0;
    for(size_t i = 0; i < 32768; ++i)
        squares += std::pow(data.loadAs<double>({i, 0}), 2);
    EXPECT_NEAR(squares / 32768, 1.0, 0.04);
    hn::Tensor integer(hn::ScalarType::Int32, hn::Shape{1, 1});
    EXPECT_THROW(client::initializeMatrix(integer,
                                          client::MatrixRole::A,
                                          hipblaslt_initialization::maf1_0,
                                          0,
                                          false,
                                          std::nullopt,
                                          false,
                                          {}),
                 std::runtime_error);
    EXPECT_THROW(
        client::groupedGemmInitializationRecipe(hn::ScalarType::Float32,
                                                hipblaslt_initialization::maf2_0,
                                                client::initialization::OperandSequence::MatrixA,
                                                0),
        std::runtime_error);
}

TEST(HostUnitMafInitialization, SeedBatchTraversalAndThreadIndependence)
{
    hn::Tensor batches(hn::ScalarType::Float64,
                       hn::Layout(hn::Shape{8192, 2, 2}, {1, 8192, 16384}, 0));
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    client::maf::fill(batches, client::maf::Mode::Maf2, 0);
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif
    hn::Tensor separate(hn::ScalarType::Float64, hn::Layout(hn::Shape{8192, 2}, {2, 1}, 0));
    client::initializeMatrix(separate,
                             client::MatrixRole::A,
                             hipblaslt_initialization::maf2_0,
                             1234,
                             false,
                             std::nullopt,
                             false,
                             {},
                             1);
#ifdef _OPENMP
    omp_set_num_threads(previous);
#endif
    for(size_t j = 0; j < 2; ++j)
        for(size_t i = 0; i < 8192; ++i)
            EXPECT_EQ(separate.loadAs<double>({i, j}), batches.loadAs<double>({i, j, 1}));
    EXPECT_NE(batches.loadAs<double>({0, 0, 0}), batches.loadAs<double>({0, 0, 1}));
    EXPECT_NE(client::maf::seed(0, 0), client::maf::seed(1, 0));
    EXPECT_NE(client::maf::seed(0, 0), client::maf::seed(0, 0, true));
    hn::Tensor empty(hn::ScalarType::Float64, hn::Shape{0, 4, 2});
    EXPECT_NO_THROW(client::maf::fill(empty, client::maf::Mode::Maf1, 0));
    hn::Tensor sentinel(hn::ScalarType::Float64, hn::Shape{1, 1, 2});
    sentinel.storeFrom({0, 0, 0}, 17.0);
    EXPECT_THROW(
        client::maf::fill(sentinel, client::maf::Mode::Maf1, 0, std::numeric_limits<size_t>::max()),
        std::runtime_error);
    EXPECT_EQ(sentinel.loadAs<double>({0, 0, 0}), 17.0);
}

TEST(HostUnitMafInitialization, EveryMxFormatHasIndependentScales)
{
    for(auto data : {hn::ScalarType::Float4E2M1,
                     hn::ScalarType::Float6E2M3,
                     hn::ScalarType::Float6E3M2,
                     hn::ScalarType::Float8E4M3,
                     hn::ScalarType::Float8E5M2})
        for(auto scale : {hn::ScalarType::E8M0, hn::ScalarType::E4M3, hn::ScalarType::E5M3})
        {
            if(data != hn::ScalarType::Float4E2M1 && scale != hn::ScalarType::E8M0)
                continue;
            for(size_t axis : {0u, 1u})
            {
                hn::MxGenerationOptions options;
                options.dataType  = data;
                options.scaleType = scale;
                options.blockSize = scale == hn::ScalarType::E8M0 ? 32 : 16;
                options.blockAxis = axis;
                auto mx           = client::maf::generateMx(
                    hn::Shape{40, 48}, options, client::maf::Mode::Maf2, 1, 2);
                for(size_t j = 0; j < 48; ++j)
                    for(size_t i = 0; i < 40; ++i)
                    {
                        const auto scaleIndex = axis == 0
                                                    ? std::vector<size_t>{j, i / options.blockSize}
                                                    : std::vector<size_t>{j / options.blockSize, i};
                        EXPECT_EQ(mx.reference.loadAs<double>({i, j}),
                                  mx.data.loadAs<double>({i, j})
                                      * mx.scales.loadAs<double>(scaleIndex));
                    }
            }
        }
}
