// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <roc/host_validation/mx.hpp>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace roc::host_validation;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool sameStorage(const Tensor& first, const Tensor& second) {
    return first.type() == second.type() && first.layout() == second.layout() &&
           first.storage().size() == second.storage().size() &&
           std::equal(first.storage().begin(), first.storage().end(), second.storage().begin());
}

bool sameResult(const MxGenerationResult& first, const MxGenerationResult& second) {
    return sameStorage(first.data, second.data) && sameStorage(first.scales, second.scales) &&
           sameStorage(first.scaleIndices, second.scaleIndices) &&
           sameStorage(first.reference, second.reference);
}

size_t expectedScaleCount(const MxGenerationProblem& problem) {
    const size_t blockedExtent = problem.shape[problem.blockAxis];
    const size_t freeExtent = problem.shape[1 - problem.blockAxis];
    return ((blockedExtent + problem.blockSize - 1) / problem.blockSize) * freeExtent;
}

size_t expectedScaleIndex(const MxGenerationProblem& problem, size_t row, size_t column) {
    if (problem.blockAxis == 0) {
        const size_t blocks = (problem.shape[0] + problem.blockSize - 1) / problem.blockSize;
        return row / problem.blockSize + column * blocks;
    }
    return row + (column / problem.blockSize) * problem.shape[0];
}

void checkReference(const MxGenerationProblem& problem, const MxGenerationResult& result) {
    const Tensor data = result.data;
    const Tensor scales = result.scales;
    const Tensor scaleIndices = result.scaleIndices;
    const Tensor reference = result.reference;
    const size_t rows = problem.shape[0];
    const size_t columns = problem.shape[1];
    const size_t scaleCount = expectedScaleCount(problem);
    require(scales.shape() == Shape{scaleCount}, "MX scale shape mismatch.");
    require(scaleIndices.shape() == problem.shape, "MX scale-index shape mismatch.");
    require(reference.shape() == problem.shape, "MX reference shape mismatch.");

    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            const size_t scaleIndex = scaleIndices.loadAs<uint32_t>({row, column});
            require(scaleIndex == expectedScaleIndex(problem, row, column),
                    "MX scale-index mapping mismatch.");
            require(scaleIndex < scaleCount, "MX scale index exceeds scale storage.");
            const float expected =
                data.loadAs<float>({row, column}) * scales.loadAs<float>({scaleIndex});
            const float observed = reference.loadAs<float>({row, column});
            if (std::isnan(expected))
                require(std::isnan(observed), "MX NaN reference mismatch.");
            else
                require(expected == observed, "MX reference value mismatch.");
        }
    }
}

MxGenerationProblem stochasticProblem(MxGenerationMode mode) {
    MxGenerationProblem problem;
    problem.dataType = ScalarType::Float4E2M1;
    problem.scaleType = ScalarType::E8M0;
    problem.shape = Shape{257, 67};
    problem.leadingDimension = 263;
    problem.blockAxis = 0;
    problem.blockSize = 16;
    problem.seed = 12345;
    problem.data.mode = mode;
    switch (mode) {
        case MxGenerationMode::Bounded:
        case MxGenerationMode::BoundedAlternatingSign:
            problem.data.parameter0 = -1.0;
            problem.data.parameter1 = 1.0;
            break;
        case MxGenerationMode::Normal:
            problem.data.parameter0 = 0.0;
            problem.data.parameter1 = 1.25;
            break;
        case MxGenerationMode::UniformInteger:
            problem.data.parameter0 = -4.0;
            problem.data.parameter1 = 4.0;
            break;
        default:
            break;
    }
    return problem;
}
}  // namespace

int main() {
    const std::array typePairs{
        std::pair{ScalarType::Float8E5M2, ScalarType::E8M0},
        std::pair{ScalarType::Float8E4M3, ScalarType::E8M0},
        std::pair{ScalarType::Float6E2M3, ScalarType::E8M0},
        std::pair{ScalarType::Float6E3M2, ScalarType::E8M0},
        std::pair{ScalarType::Float4E2M1, ScalarType::E8M0},
        std::pair{ScalarType::Float4E2M1, ScalarType::E4M3},
        std::pair{ScalarType::Float4E2M1, ScalarType::E5M3},
    };
    for (const auto& [dataType, scaleType] : typePairs) {
        for (const size_t blockSize : {size_t{16}, size_t{32}}) {
            MxGenerationProblem problem;
            problem.dataType = dataType;
            problem.scaleType = scaleType;
            problem.shape = Shape{67, 5};
            problem.leadingDimension = 73;
            problem.blockAxis = 0;
            problem.blockSize = blockSize;
            problem.data.mode = MxGenerationMode::Bounded;
            problem.data.parameter0 = -1;
            problem.data.parameter1 = 1;
            const MxGenerationResult result = generateMx(problem);
            checkReference(problem, result);
            const size_t expectedBytes =
                (static_cast<size_t>(problem.leadingDimension) * problem.shape[1] *
                     scalarTypeInfo(dataType).storageBits +
                 7) /
                8;
            require(result.data.storage().size() == expectedBytes,
                    "MX padded data-storage size mismatch.");
        }
    }

    MxGenerationProblem blockAxisOne;
    blockAxisOne.dataType = ScalarType::Float6E2M3;
    blockAxisOne.scaleType = ScalarType::E8M0;
    blockAxisOne.shape = Shape{5, 37};
    blockAxisOne.leadingDimension = 8;
    blockAxisOne.blockAxis = 1;
    blockAxisOne.blockSize = 16;
    checkReference(blockAxisOne, generateMx(blockAxisOne));

    MxGenerationProblem paddedRegression;
    paddedRegression.dataType = ScalarType::Float4E2M1;
    paddedRegression.scaleType = ScalarType::E8M0;
    paddedRegression.shape = Shape{64, 2};
    paddedRegression.leadingDimension = 80;
    paddedRegression.blockAxis = 0;
    paddedRegression.blockSize = 32;
    checkReference(paddedRegression, generateMx(paddedRegression));

    for (const ptrdiff_t leadingDimension : {ptrdiff_t{5}, ptrdiff_t{6}, ptrdiff_t{7}}) {
        MxGenerationProblem fp6PackingTail;
        fp6PackingTail.dataType = ScalarType::Float6E3M2;
        fp6PackingTail.scaleType = ScalarType::E8M0;
        fp6PackingTail.shape = Shape{5, 1};
        fp6PackingTail.leadingDimension = leadingDimension;
        fp6PackingTail.blockAxis = 0;
        fp6PackingTail.blockSize = 4;
        const MxGenerationResult result = generateMx(fp6PackingTail);
        checkReference(fp6PackingTail, result);
        const size_t physicalElements = static_cast<size_t>(leadingDimension);
        require(result.data.storage().size() == (physicalElements * 6 + 7) / 8,
                "FP6 packing-tail storage size mismatch.");
    }

    const std::array deterministicModes{
        MxGenerationMode::Identity,
        MxGenerationMode::Ones,
        MxGenerationMode::Zeros,
        MxGenerationMode::Sequential,
        MxGenerationMode::RowIndex,
        MxGenerationMode::ColumnIndex,
        MxGenerationMode::Checkerboard,
        MxGenerationMode::ScaledDiagonal,
        MxGenerationMode::Twos,
        MxGenerationMode::NegativeOnes,
        MxGenerationMode::Maximum,
        MxGenerationMode::DenormalMinimum,
        MxGenerationMode::DenormalMaximum,
        MxGenerationMode::NaN,
        MxGenerationMode::UniformInteger,
    };
    for (const MxGenerationMode mode : deterministicModes) {
        MxGenerationProblem problem = stochasticProblem(mode);
        const MxGenerationResult result = generateMx(problem);
        checkReference(problem, result);
        if (mode == MxGenerationMode::NaN) {
            for (size_t column = 0; column < problem.shape[1]; ++column)
                for (size_t row = 0; row < problem.shape[0]; ++row)
                    require(std::isnan(result.reference.loadAs<float>({row, column})),
                            "MX NaN mode produced a finite reference value.");
        }
    }

    MxGenerationProblem infinity;
    infinity.dataType = ScalarType::Float8E5M2;
    infinity.scaleType = ScalarType::E8M0;
    infinity.shape = Shape{16, 3};
    infinity.blockSize = 16;
    infinity.data.mode = MxGenerationMode::Infinity;
    checkReference(infinity, generateMx(infinity));

    MxGenerationProblem explicitScale = stochasticProblem(MxGenerationMode::Bounded);
    explicitScale.scale = MxScaleGenerationMode::One;
    const MxGenerationResult explicitlyScaled = generateMx(explicitScale);
    for (size_t scaleIndex = 0; scaleIndex < explicitlyScaled.scales.shape()[0]; ++scaleIndex)
        require(explicitlyScaled.scales.loadAs<float>({scaleIndex}) == 1.0f,
                "MX explicit unity-scale generation mismatch.");
    checkReference(explicitScale, explicitlyScaled);

    const std::array constantScaleModes{
        std::pair{MxScaleGenerationMode::Minimum, uint8_t{0}},
        std::pair{MxScaleGenerationMode::One, uint8_t{127}},
        std::pair{MxScaleGenerationMode::Two, uint8_t{128}},
    };
    for (const auto& [mode, expectedRaw] : constantScaleModes) {
        MxGenerationProblem constantScale = stochasticProblem(MxGenerationMode::Sequential);
        constantScale.scale = mode;
        const MxGenerationResult result = generateMx(constantScale);
        for (size_t scaleIndex = 0; scaleIndex < result.scales.shape()[0]; ++scaleIndex)
            require(std::to_integer<uint8_t>(result.scales.storage()[scaleIndex]) == expectedRaw,
                    "MX explicit constant-scale generation mismatch.");
        checkReference(constantScale, result);
    }

    MxGenerationProblem maximumScale = stochasticProblem(MxGenerationMode::Sequential);
    maximumScale.scale = MxScaleGenerationMode::Maximum;
    const MxGenerationResult maximumScaled = generateMx(maximumScale);
    for (size_t scaleIndex = 0; scaleIndex < maximumScaled.scales.shape()[0]; ++scaleIndex)
        require(std::to_integer<uint8_t>(maximumScaled.scales.storage()[scaleIndex]) == 0xfeU,
                "MX explicit maximum-scale generation mismatch.");
    checkReference(maximumScale, maximumScaled);

    MxGenerationProblem nanScale = stochasticProblem(MxGenerationMode::Sequential);
    nanScale.scale = MxScaleGenerationMode::NaN;
    const MxGenerationResult nanScaled = generateMx(nanScale);
    for (size_t scaleIndex = 0; scaleIndex < nanScaled.scales.shape()[0]; ++scaleIndex)
        require(std::to_integer<uint8_t>(nanScaled.scales.storage()[scaleIndex]) == 0xffU,
                "MX explicit NaN-scale generation mismatch.");
    checkReference(nanScale, nanScaled);

    MxGenerationProblem impossibleInterval = explicitScale;
    impossibleInterval.shape = Shape{8192, 1};
    impossibleInterval.leadingDimension = 8192;
    impossibleInterval.data.parameter0 = 0.1;
    impossibleInterval.data.parameter1 = 0.2;
    bool rejectedImpossibleInterval = false;
    try {
        (void)generateMx(impossibleInterval);
    } catch (const std::invalid_argument&) {
        rejectedImpossibleInterval = true;
    }
    require(rejectedImpossibleInterval,
            "MX parallel generation did not reject an unrepresentable bounded interval.");

    const std::array stochasticModes{
        MxGenerationMode::Bounded,   MxGenerationMode::BoundedAlternatingSign,
        MxGenerationMode::Unbounded, MxGenerationMode::Trigonometric,
        MxGenerationMode::Normal,    MxGenerationMode::UniformInteger,
    };
    for (const MxGenerationMode mode : stochasticModes) {
        MxGenerationProblem problem = stochasticProblem(mode);
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
        const MxGenerationResult oneThread = generateMx(problem);
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        const MxGenerationResult fourThreads = generateMx(problem);
        require(sameResult(oneThread, fourThreads),
                "MX generation changed with OpenMP thread count.");
        checkReference(problem, oneThread);

        ++problem.seed;
        const MxGenerationResult differentSeed = generateMx(problem);
        require(!sameStorage(oneThread.data, differentSeed.data) ||
                    !sameStorage(oneThread.scales, differentSeed.scales),
                "MX stochastic generation ignored the seed.");
    }

    return 0;
}
