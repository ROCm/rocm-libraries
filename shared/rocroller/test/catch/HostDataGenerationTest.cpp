// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <map>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include <omp.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <rocRoller/HostNumerics/HostDataGeneration.hpp>

#include "CustomSections.hpp"
#include "SimpleTest.hpp"

using namespace rocRoller;

namespace HostDataGenerationTest
{
    namespace
    {
        template <typename T>
        constexpr bool isBlockScaledType
            = std::is_same_v<
                  T,
                  FP4> || std::is_same_v<T, FP6> || std::is_same_v<T, BF6> || std::is_same_v<T, FP8> || std::is_same_v<T, BF8>;

        std::vector<uint8_t> bytes(roc::host_numerics::Tensor const& tensor)
        {
            std::vector<uint8_t> result(tensor.rawEncodedBackingStorage().size());
            std::transform(tensor.rawEncodedBackingStorage().begin(),
                           tensor.rawEncodedBackingStorage().end(),
                           result.begin(),
                           [](std::byte value) { return std::to_integer<uint8_t>(value); });
            return result;
        }

        template <typename T>
        std::optional<HostNumerics::BlockScaleGeneration> blockScaleGeneration(size_t blockSize)
        {
            if constexpr(isBlockScaledType<T>)
                return HostNumerics::BlockScaleGeneration{DataType::E8M0, 1, blockSize};
            return std::nullopt;
        }

        template <typename T>
        constexpr HostNumerics::DataTypeInterpretation dataTypeInterpretation()
        {
            if constexpr(isBlockScaledType<T>)
                return HostNumerics::DataTypeInterpretation::BlockScaled;
            return HostNumerics::DataTypeInterpretation::Unscaled;
        }
    }

    class HostDataGenerationTest : public SimpleTest
    {
    public:
        HostDataGenerationTest() = default;

        template <typename T>
        void executeDataGeneratorTest(unsigned    dimension0,
                                      unsigned    dimension1,
                                      const float minimum      = -1.0f,
                                      const float maximum      = 1.0f,
                                      const int   blockScaling = 32)
        {
            auto             dataType = TypeInfo<T>::Var.dataType;
            TensorDescriptor descriptor(dataType, {dimension0, dimension1}, "T");

            auto generated = HostNumerics::generateHostTensor(
                descriptor,
                {},
                blockScaleGeneration<T>(static_cast<size_t>(blockScaling)),
                minimum,
                maximum,
                9861u,
                true);

            using PackedType = typename PackedTypeOf<T>::type;
            auto packed      = HostNumerics::copyTensorStorage<PackedType>(generated.data);
            auto packedView
                = HostNumerics::hostTensor(descriptor, packed, dataTypeInterpretation<T>());

            for(size_t row = 0; row < dimension0; ++row)
            {
                for(size_t column = 0; column < dimension1; ++column)
                {
                    auto const value = generated.data.template loadAs<float>({row, column});
                    CHECK(packedView.template loadAs<float>({row, column}) == value);

                    REQUIRE(generated.reference);
                    auto const reference
                        = generated.reference->template loadAs<float>({row, column});
                    if constexpr(isBlockScaledType<T>)
                    {
                        REQUIRE(generated.scales);
                        auto const scale = generated.scales->template loadAs<float>(
                            {row, column / static_cast<size_t>(blockScaling)});
                        CHECK(value * scale == reference);
                    }
                    else
                    {
                        CHECK(value == reference);
                    }
                }
            }
        }
    };

    TEMPLATE_TEST_CASE("Generate host tensor data",
                       "[host-data-generation]",
                       FP4,
                       FP6,
                       BF6,
                       FP8,
                       BF8,
                       Half,
                       BFloat16,
                       float)
    {
        HostDataGenerationTest test;

        SUPPORTED_ARCH_SECTION(arch)
        {
            test.executeDataGeneratorTest<TestType>(32, 32);
        }
    }

    TEMPLATE_TEST_CASE("Host tensor generation is deterministic across calls and thread counts",
                       "[host-data-generation]",
                       FP4,
                       FP6,
                       BF6,
                       FP8,
                       BF8,
                       Half,
                       BFloat16,
                       float)
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            constexpr size_t dimension0        = 1024;
            constexpr size_t dimension1        = 1024;
            constexpr float  minimum           = -1.0f;
            constexpr float  maximum           = 1.0f;
            constexpr size_t blockScaling      = 32;
            using DataType                     = TestType;
            auto const       rocRollerDataType = TypeInfo<DataType>::Var.dataType;
            TensorDescriptor descriptor(rocRollerDataType, {dimension0, dimension1}, "T");

            std::vector<uint32_t> shuffledSeeds = {9861u, 12345u};
            std::shuffle(shuffledSeeds.begin(), shuffledSeeds.end(), std::default_random_engine{});

            const int        originalThreads = omp_get_max_threads();
            std::vector<int> threadCounts    = {originalThreads, 1, 2, 4, 8};

            for(int threadCount : threadCounts)
            {
                omp_set_num_threads(threadCount);

                std::map<uint32_t, std::vector<uint8_t>> firstGenerationData;
                std::map<uint32_t, std::vector<uint8_t>> firstGenerationScales;
                std::map<uint32_t, std::vector<uint8_t>> firstGenerationReference;

                for(uint32_t seed : shuffledSeeds)
                {
                    auto generated = HostNumerics::generateHostTensor(
                        descriptor,
                        {},
                        blockScaleGeneration<DataType>(blockScaling),
                        minimum,
                        maximum,
                        seed,
                        true);
                    firstGenerationData[seed] = bytes(generated.data);
                    firstGenerationScales[seed]
                        = generated.scales ? bytes(*generated.scales) : std::vector<uint8_t>{};
                    REQUIRE(generated.reference);
                    firstGenerationReference[seed] = bytes(*generated.reference);
                }

                std::shuffle(
                    shuffledSeeds.begin(), shuffledSeeds.end(), std::default_random_engine{});

                for(uint32_t seed : shuffledSeeds)
                {
                    auto generated = HostNumerics::generateHostTensor(
                        descriptor,
                        {},
                        blockScaleGeneration<DataType>(blockScaling),
                        minimum,
                        maximum,
                        seed,
                        true);

                    CHECK(firstGenerationData[seed] == bytes(generated.data));
                    CHECK(
                        firstGenerationScales[seed]
                        == (generated.scales ? bytes(*generated.scales) : std::vector<uint8_t>{}));
                    REQUIRE(generated.reference);
                    CHECK(firstGenerationReference[seed] == bytes(*generated.reference));
                }
            }

            omp_set_num_threads(originalThreads);
        }
    }
} // namespace HostDataGenerationTest
