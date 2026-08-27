// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/HostNumerics/HostReference.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace rocRoller;
    using namespace rocRoller::HostNumerics;
    using roc::host_validation::Layout;
    using roc::host_validation::ScalarType;
    using roc::host_validation::Shape;
    using roc::host_validation::Tensor;

    void require(bool condition, std::string const& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    template <typename T, size_t Size>
    Tensor nativeTensor(ScalarType type, Layout layout, std::array<T, Size> const& values)
    {
        auto const bytes = std::as_bytes(std::span(values));
        return Tensor::takeOwnershipOfEncodedBackingStorage(
            type, std::move(layout), std::vector<std::byte>(bytes.begin(), bytes.end()));
    }

    Tensor scaleTensor(Shape shape, std::initializer_list<uint8_t> values)
    {
        std::vector<std::byte> storage;
        storage.reserve(values.size());
        for(uint8_t value : values)
            storage.push_back(static_cast<std::byte>(value));
        return Tensor::takeOwnershipOfEncodedBackingStorage(ScalarType::E8M0, Layout::contiguousLastDimensionFastest(shape), std::move(storage));
    }

    void testUnscaledReference()
    {
        const std::array<float, 4> a{1, 3, 2, 4};
        const std::array<float, 4> b{5, 7, 6, 8};
        const std::array<float, 4> c{1, 1, 1, 1};
        GeneratedGEMMInputs        inputs{
            nativeTensor(ScalarType::Float32, Layout(Shape{2, 2}, {1, 2}), a),
            nativeTensor(ScalarType::Float32, Layout(Shape{2, 2}, {1, 2}), b),
            nativeTensor(ScalarType::Float32, Layout(Shape{2, 2}, {1, 2}), c),
            std::nullopt,
            std::nullopt,
        };

        const HostReferenceProblem problem
            = makeHostReferenceProblem(inputs, std::nullopt, std::nullopt, 0, 2.0f, 3.0f);
        const Tensor reference = computeHostReference(problem);
        require(convertHostReference<float>(reference) == std::vector<float>({41, 89, 47, 103}),
                "Unscaled rocroller-gemm host reference mismatch.");
    }

    void testZeroExtentReference()
    {
        const std::array<float, 0> empty{};

        {
            const std::array<float, 6> b{1, 2, 3, 4, 5, 6};
            GeneratedGEMMInputs        inputs{
                nativeTensor(ScalarType::Float32, Layout(Shape{0, 2}, {1, 0}), empty),
                nativeTensor(ScalarType::Float32, Layout(Shape{2, 3}, {1, 2}), b),
                nativeTensor(ScalarType::Float32, Layout(Shape{0, 3}, {1, 0}), empty),
                std::nullopt,
                std::nullopt,
            };

            const Tensor reference = computeHostReference(
                makeHostReferenceProblem(inputs, std::nullopt, std::nullopt, 0, 1.0f, 0.0f));
            require(reference.shape() == Shape{0, 3}
                        && convertHostReference<float>(reference).empty(),
                    "M-zero rocroller-gemm host reference mismatch.");
        }

        {
            const std::array<float, 6> a{1, 2, 3, 4, 5, 6};
            GeneratedGEMMInputs        inputs{
                nativeTensor(ScalarType::Float32, Layout(Shape{2, 3}, {1, 2}), a),
                nativeTensor(ScalarType::Float32, Layout(Shape{3, 0}, {1, 3}), empty),
                nativeTensor(ScalarType::Float32, Layout(Shape{2, 0}, {1, 2}), empty),
                std::nullopt,
                std::nullopt,
            };

            const Tensor reference = computeHostReference(
                makeHostReferenceProblem(inputs, std::nullopt, std::nullopt, 0, 1.0f, 0.0f));
            require(reference.shape() == Shape{2, 0}
                        && convertHostReference<float>(reference).empty(),
                    "N-zero rocroller-gemm host reference mismatch.");
        }

        {
            const std::array<float, 4> c{1, 2, 3, 4};
            GeneratedGEMMInputs        inputs{
                nativeTensor(ScalarType::Float32, Layout(Shape{2, 0}, {1, 2}), empty),
                nativeTensor(ScalarType::Float32, Layout(Shape{0, 2}, {1, 0}), empty),
                nativeTensor(ScalarType::Float32, Layout(Shape{2, 2}, {1, 2}), c),
                std::nullopt,
                std::nullopt,
            };

            const Tensor reference = computeHostReference(
                makeHostReferenceProblem(inputs, std::nullopt, std::nullopt, 0, 7.0f, -2.0f));
            require(convertHostReference<float>(reference) == std::vector<float>({-2, -4, -6, -8}),
                    "K-zero rocroller-gemm host reference did not apply beta to C.");
        }
    }

    void testScaledReference()
    {
        const std::array<float, 4> a{1, 1, 1, 1};
        const std::array<float, 4> b{1, 1, 1, 1};
        const std::array<float, 1> c{0};
        GeneratedGEMMInputs        inputs{
            nativeTensor(ScalarType::Float32, Layout(Shape{1, 4}, {1, 1}), a),
            nativeTensor(ScalarType::Float32, Layout(Shape{4, 1}, {1, 4}), b),
            nativeTensor(ScalarType::Float32, Layout(Shape{1, 1}, {1, 1}), c),
            scaleTensor(Shape{1, 2}, {128, 129}),
            scaleTensor(Shape{1, 2}, {130, 131}),
        };

        const HostReferenceProblem problem
            = makeHostReferenceProblem(inputs, std::nullopt, std::nullopt, 2, 1.0f, 0.0f);
        const Tensor reference = computeHostReference(problem);
        require(convertHostReference<float>(reference) == std::vector<float>({160}),
                "Block-scaled rocroller-gemm host reference mismatch.");

        const std::array<uint8_t, 1> singleScaleA{128};
        const std::array<uint8_t, 1> singleScaleB{130};
        HostReferenceProblem         singleScaleProblem
            = makeHostReferenceProblem(inputs,
                                       hostScaleTensor(DataType::E8M0, singleScaleA, 1, 4, 4),
                                       hostScaleTensor(DataType::E8M0, singleScaleB, 1, 4, 4),
                                       4,
                                       1.0f,
                                       0.0f);
        const Tensor singleScaleReference = computeHostReference(singleScaleProblem);
        require(convertHostReference<float>(singleScaleReference) == std::vector<float>({64}),
                "Single-scale rocroller-gemm host reference mismatch.");

        GeneratedGEMMInputs onlyA = inputs;
        onlyA.scaleB.reset();
        const Tensor onlyAReference = computeHostReference(
            makeHostReferenceProblem(onlyA, std::nullopt, std::nullopt, 2, 1.0f, 0.0f));
        require(convertHostReference<float>(onlyAReference) == std::vector<float>({12}),
                "One-sided A scaling did not use a unity B scale.");

        GeneratedGEMMInputs onlyB = inputs;
        onlyB.scaleA.reset();
        const Tensor onlyBReference = computeHostReference(
            makeHostReferenceProblem(onlyB, std::nullopt, std::nullopt, 2, 1.0f, 0.0f));
        require(convertHostReference<float>(onlyBReference) == std::vector<float>({48}),
                "One-sided B scaling did not use a unity A scale.");
    }

    void testGeneratedLogicalKScales()
    {
        const TensorDescriptor    descriptorA(DataType::FP4, {2, 4}, "N");
        const TensorDescriptor    descriptorB(DataType::FP4, {4, 3}, "T");
        const TensorDescriptor    descriptorC(DataType::Float, {2, 3}, "N");
        const DataInitialization  ones{DataInitializationMode::Ones};
        const GeneratedGEMMInputs inputs = generateGEMMInputs(descriptorA,
                                                              descriptorB,
                                                              descriptorC,
                                                              ones,
                                                              ones,
                                                              ones,
                                                              DataType::E8M0,
                                                              DataType::E8M0,
                                                              2,
                                                              -1.0f,
                                                              1.0f,
                                                              31415u);

        require(inputs.scaleA && inputs.scaleB,
                "Logical-K generation did not return both scale tensors.");
        require(inputs.scaleA->layout() == Layout(Shape{2, 2}, {1, 2})
                    && inputs.scaleB->layout() == Layout(Shape{3, 2}, {1, 3}),
                "Logical-K generation returned the wrong canonical scale layouts.");

        const Tensor reference = computeHostReference(
            makeHostReferenceProblem(inputs, std::nullopt, std::nullopt, 2, 1.0f, 0.0f));
        require(convertHostReference<float>(reference) == std::vector<float>({4, 4, 4, 4, 4, 4}),
                "Generated logical-K scales produced the wrong host reference.");
    }

    void testOutputConversion()
    {
        constexpr float            bfloat16RoundingBoundary = 1.005859375f;
        const std::array<float, 3> values{1.25f, -2.5f, bfloat16RoundingBoundary};
        const Tensor input = nativeTensor(ScalarType::Float32, Layout(Shape{3, 1}, {1, 3}), values);

        require(convertHostReference<float>(input)
                    == std::vector<float>(values.begin(), values.end()),
                "F32 host-reference output conversion mismatch.");

        const auto half = convertHostReference<Half>(input);
        require(static_cast<float>(half[0]) == static_cast<float>(Half(values[0]))
                    && static_cast<float>(half[1]) == static_cast<float>(Half(values[1])),
                "F16 host-reference output conversion mismatch.");

        const auto bfloat16 = convertHostReference<BFloat16>(input);
        require(bfloat16[0].data == BFloat16(values[0]).data
                    && bfloat16[1].data == BFloat16(values[1]).data,
                "BF16 host-reference output conversion mismatch.");
        require(bfloat16[2].data == 0x3f80,
                "BF16 host-reference output did not preserve rocRoller truncation.");
    }

    void testStrictComparison()
    {
        const std::array<float, 1> expected{1.0f};
        const std::array<float, 1> observed{2.0f};
        const auto                 expectedView = hostOutputTensor<float>(expected, 1, 1);
        const auto                 observedView = hostOutputTensor<float>(observed, 1, 1);

        const HostComparisonResult boundary = compareHostReference(
            observedView, expectedView, AcceptableGEMMError{1.0, "strict boundary"});
        require(boundary.relativeNormL2 == 1.0 && !boundary.ok
                    && !boundary.statistics.frobeniusPassed,
                "rocroller-gemm comparison did not preserve strict less-than acceptance.");

        const HostComparisonResult aboveBoundary = compareHostReference(
            observedView, expectedView, AcceptableGEMMError{1.01, "above boundary"});
        require(aboveBoundary.ok && aboveBoundary.statistics.passed(),
                "rocroller-gemm comparison rejected a relative error below tolerance.");

        const std::array<float, 1> zero{};
        const auto zeroResult = compareHostReference(hostOutputTensor<float>(zero, 1, 1),
                                                     hostOutputTensor<float>(zero, 1, 1),
                                                     AcceptableGEMMError{1.0, "zero reference"});
        require(std::isnan(zeroResult.relativeNormL2) && !zeroResult.ok,
                "rocroller-gemm comparison changed zero-reference acceptance semantics.");

        const std::array<float, 1> infinity{std::numeric_limits<float>::infinity()};
        const auto                 infinityResult
            = compareHostReference(hostOutputTensor<float>(infinity, 1, 1),
                                   hostOutputTensor<float>(infinity, 1, 1),
                                   AcceptableGEMMError{1.0, "matching infinity"});
        require(std::isnan(infinityResult.relativeNormL2) && !infinityResult.ok,
                "rocroller-gemm comparison changed matching-infinity acceptance semantics.");
    }
}

int main()
{
    testUnscaledReference();
    testZeroExtentReference();
    testScaledReference();
    testGeneratedLogicalKScales();
    testOutputConversion();
    testStrictComparison();
    return 0;
}
