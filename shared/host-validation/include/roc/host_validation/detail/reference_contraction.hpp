// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <complex>
#include <cstdint>
#include <optional>
#include <roc/host_validation/detail/reference_common.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace roc::host_validation {
using ContractionDimension = uint32_t;

struct TensorContractionOperand {
    TensorContractionOperand(TensorView tensor, std::vector<ContractionDimension> tensorDimensions)
        : values(std::move(tensor)), dimensions(std::move(tensorDimensions)) {}

    TensorView values;
    std::vector<ContractionDimension> dimensions;
    std::optional<ScalarType> computeType;
    bool conjugate = false;
};

struct TensorContractionProblem {
    TensorContractionProblem(TensorContractionOperand aOperand, TensorContractionOperand bOperand,
                             TensorView cTensor,
                             std::vector<ContractionDimension> cTensorDimensions,
                             MutableTensorView dTensor,
                             std::vector<ContractionDimension> dTensorDimensions,
                             std::vector<ContractionDimension> reducedDimensions,
                             ScalarType accumulator)
        : a(std::move(aOperand)),
          b(std::move(bOperand)),
          c(std::move(cTensor)),
          cDimensions(std::move(cTensorDimensions)),
          d(std::move(dTensor)),
          dDimensions(std::move(dTensorDimensions)),
          reductionDimensions(std::move(reducedDimensions)),
          accumulatorType(accumulator) {}

    TensorContractionOperand a;
    TensorContractionOperand b;
    TensorView c;
    std::vector<ContractionDimension> cDimensions;
    MutableTensorView d;
    std::vector<ContractionDimension> dDimensions;
    std::vector<ContractionDimension> reductionDimensions;
    ScalarType accumulatorType;
    MathMode mathMode = MathMode::Default;
    std::complex<double> alpha = {1.0, 0.0};
    std::complex<double> beta = {0.0, 0.0};
    OutputSelection outputSelection = OutputSelection::all();
};

struct TensorContractionRunInfo {
    size_t outputElementsComputed = 0;
    size_t multiplyAddsComputed = 0;
};

namespace detail {
struct TensorContractionPlan {
    std::vector<ContractionDimension> dimensions;
    std::vector<size_t> extents;
    std::vector<size_t> aSlots;
    std::vector<size_t> bSlots;
    std::vector<size_t> cSlots;
    std::vector<size_t> dSlots;
    std::vector<size_t> reductionSlots;
    Shape reductionShape;
};

inline void requireUniqueDimensions(std::span<const ContractionDimension> dimensions,
                                    const char* name) {
    std::vector<ContractionDimension> sorted(dimensions.begin(), dimensions.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::invalid_argument(std::string("Reference contraction ") + name +
                                    " dimensions must be unique.");
}

inline TensorContractionPlan validateTensorContraction(const TensorContractionProblem& problem) {
    if (problem.a.dimensions.size() != problem.a.values.shape().rank() ||
        problem.b.dimensions.size() != problem.b.values.shape().rank() ||
        problem.cDimensions.size() != problem.c.shape().rank() ||
        problem.dDimensions.size() != problem.d.shape().rank())
        throw std::invalid_argument(
            "Reference contraction dimension-label count does not match tensor rank.");
    requireUniqueDimensions(problem.a.dimensions, "A");
    requireUniqueDimensions(problem.b.dimensions, "B");
    requireUniqueDimensions(problem.cDimensions, "C");
    requireUniqueDimensions(problem.dDimensions, "D");
    requireUniqueDimensions(problem.reductionDimensions, "reduction");

    TensorContractionPlan plan;
    plan.dimensions = problem.dDimensions;
    for (const ContractionDimension dimension : problem.reductionDimensions) {
        if (std::find(plan.dimensions.begin(), plan.dimensions.end(), dimension) !=
            plan.dimensions.end())
            throw std::invalid_argument(
                "Reference contraction output and reduction dimensions overlap.");
        plan.dimensions.push_back(dimension);
    }
    plan.extents.assign(plan.dimensions.size(), 0);

    auto slotsFor = [&](std::span<const ContractionDimension> dimensions, const char* name) {
        std::vector<size_t> slots;
        slots.reserve(dimensions.size());
        for (const ContractionDimension dimension : dimensions) {
            const auto position =
                std::find(plan.dimensions.begin(), plan.dimensions.end(), dimension);
            if (position == plan.dimensions.end())
                throw std::invalid_argument(std::string("Reference contraction ") + name +
                                            " uses an unknown dimension.");
            slots.push_back(static_cast<size_t>(position - plan.dimensions.begin()));
        }
        return slots;
    };
    plan.aSlots = slotsFor(problem.a.dimensions, "A");
    plan.bSlots = slotsFor(problem.b.dimensions, "B");
    plan.cSlots = slotsFor(problem.cDimensions, "C");
    plan.dSlots = slotsFor(problem.dDimensions, "D");
    plan.reductionSlots = slotsFor(problem.reductionDimensions, "reduction");

    auto recordExtents = [&](const Shape& shape, std::span<const size_t> slots, const char* name) {
        for (size_t axis = 0; axis < shape.rank(); ++axis) {
            size_t& extent = plan.extents[slots[axis]];
            if (extent == 0)
                extent = shape[axis];
            else if (extent != shape[axis])
                throw std::invalid_argument(std::string("Reference contraction ") + name +
                                            " dimension extent mismatch.");
        }
    };
    recordExtents(problem.a.values.shape(), plan.aSlots, "A");
    recordExtents(problem.b.values.shape(), plan.bSlots, "B");
    recordExtents(problem.c.shape(), plan.cSlots, "C");
    recordExtents(problem.d.shape(), plan.dSlots, "D");
    for (const size_t extent : plan.extents)
        if (extent == 0)
            throw std::invalid_argument("Reference contraction has an unbound dimension.");

    std::vector<size_t> reductionExtents;
    reductionExtents.reserve(plan.reductionSlots.size());
    for (const size_t slot : plan.reductionSlots) reductionExtents.push_back(plan.extents[slot]);
    plan.reductionShape = Shape(std::move(reductionExtents));

    auto validateType = [&](ScalarType type, const char* name) {
        if (type == ScalarType::Count || type == ScalarType::Boolean || isScaleScalarType(type))
            throw std::invalid_argument(std::string("Reference contraction ") + name +
                                        " has an unsupported scalar type.");
    };
    validateType(problem.a.values.type(), "A");
    validateType(problem.b.values.type(), "B");
    validateType(problem.c.type(), "C");
    validateType(problem.d.type(), "D");
    if (problem.a.computeType) validateType(*problem.a.computeType, "A compute input");
    if (problem.b.computeType) validateType(*problem.b.computeType, "B compute input");

    const bool complexAccumulator = problem.accumulatorType == ScalarType::ComplexFloat32 ||
                                    problem.accumulatorType == ScalarType::ComplexFloat64;
    if (!complexAccumulator &&
        (isComplexScalarType(problem.a.values.type()) ||
         isComplexScalarType(problem.b.values.type()) || isComplexScalarType(problem.c.type()) ||
         isComplexScalarType(problem.d.type())))
        throw std::invalid_argument("Real reference contraction cannot consume complex tensors.");
    if (complexAccumulator != isComplexScalarType(problem.d.type()))
        throw std::invalid_argument(
            "Reference contraction accumulator/output complexity mismatch.");

    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::Int32:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            break;
        default:
            throw std::invalid_argument("Reference contraction accumulator type is unsupported.");
    }
    if (problem.mathMode == MathMode::XFloat32 && problem.accumulatorType != ScalarType::Float32)
        throw std::invalid_argument("XFloat32 contraction math requires a Float32 accumulator.");
    if (!complexAccumulator && (problem.alpha.imag() != 0.0 || problem.beta.imag() != 0.0))
        throw std::invalid_argument("Real reference contraction has complex alpha or beta.");
    (void)problem.outputSelection.selectedCount(problem.d.shape().elementCount());
    return plan;
}

inline void contractionCoordinatesFromLinear(size_t linear, const Shape& shape,
                                             std::vector<size_t>& coordinates) {
    coordinates.resize(shape.rank());
    for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
        const size_t axis = dimension - 1;
        coordinates[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
}

template <typename Accumulator>
TensorContractionRunInfo referenceTensorContractionTyped(const TensorContractionProblem& problem,
                                                         const TensorContractionPlan& plan) {
    const RuntimeTensorReader<Accumulator> a(problem.a.values);
    const RuntimeTensorReader<Accumulator> b(problem.b.values);
    const RuntimeTensorReader<Accumulator> c(problem.c);
    const RuntimeTensorWriter<Accumulator> d(problem.d);
    const RuntimeQuantizer<Accumulator> quantizeA(problem.a.computeType);
    const RuntimeQuantizer<Accumulator> quantizeB(problem.b.computeType);
    const RuntimeQuantizer<Accumulator> quantizeAccumulator(
        problem.accumulatorType == ScalarType::Float16 ||
                problem.accumulatorType == ScalarType::BFloat16
            ? std::optional<ScalarType>(problem.accumulatorType)
            : std::nullopt);
    const RuntimeMathFunction<Accumulator> operandMath =
        runtimeMathFunction<Accumulator>(problem.mathMode);
    const Accumulator alpha =
        quantizeAccumulator(runtimeScalar<Accumulator>(problem.alpha, "alpha"));
    const Accumulator beta = quantizeAccumulator(runtimeScalar<Accumulator>(problem.beta, "beta"));
    auto multiply = [&](Accumulator left, Accumulator right) {
        return quantizeAccumulator(left * right);
    };
    auto add = [&](Accumulator left, Accumulator right) {
        return quantizeAccumulator(left + right);
    };

    std::vector<size_t> globalCoordinates(plan.dimensions.size(), 0);
    std::vector<size_t> outputCoordinates;
    std::vector<size_t> reductionCoordinates;
    std::vector<size_t> aCoordinates(problem.a.values.shape().rank());
    std::vector<size_t> bCoordinates(problem.b.values.shape().rank());
    std::vector<size_t> cCoordinates(problem.c.shape().rank());
    auto coordinatesFor = [&](std::span<const size_t> slots, std::vector<size_t>& coordinates) {
        for (size_t axis = 0; axis < slots.size(); ++axis)
            coordinates[axis] = globalCoordinates[slots[axis]];
    };

    const size_t outputElements = problem.d.shape().elementCount();
    const size_t reductionElements = plan.reductionShape.elementCount();
    const auto selected = problem.outputSelection.indices(outputElements);
    for (const size_t outputLinear : selected) {
        contractionCoordinatesFromLinear(outputLinear, problem.d.shape(), outputCoordinates);
        for (size_t axis = 0; axis < plan.dSlots.size(); ++axis)
            globalCoordinates[plan.dSlots[axis]] = outputCoordinates[axis];

        Accumulator sum{};
        for (size_t reductionLinear = 0; reductionLinear < reductionElements; ++reductionLinear) {
            contractionCoordinatesFromLinear(reductionLinear, plan.reductionShape,
                                             reductionCoordinates);
            for (size_t axis = 0; axis < plan.reductionSlots.size(); ++axis)
                globalCoordinates[plan.reductionSlots[axis]] = reductionCoordinates[axis];
            coordinatesFor(plan.aSlots, aCoordinates);
            coordinatesFor(plan.bSlots, bCoordinates);
            Accumulator aValue =
                conjugateIfNeeded(a(std::span<const size_t>(aCoordinates)), problem.a.conjugate);
            Accumulator bValue =
                conjugateIfNeeded(b(std::span<const size_t>(bCoordinates)), problem.b.conjugate);
            aValue = operandMath(quantizeA(aValue));
            bValue = operandMath(quantizeB(bValue));
            sum = add(sum, multiply(aValue, bValue));
        }

        coordinatesFor(plan.cSlots, cCoordinates);
        const Accumulator result =
            add(multiply(alpha, sum), multiply(beta, c(std::span<const size_t>(cCoordinates))));
        d.store(std::span<const size_t>(outputCoordinates), result);
    }

    return {
        .outputElementsComputed = selected.size(),
        .multiplyAddsComputed = selected.size() * reductionElements,
    };
}
}  // namespace detail

inline TensorContractionRunInfo referenceTensorContraction(
    const TensorContractionProblem& problem) {
    const detail::TensorContractionPlan plan = detail::validateTensorContraction(problem);
    switch (problem.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            return detail::referenceTensorContractionTyped<float>(problem, plan);
        case ScalarType::Float64:
            return detail::referenceTensorContractionTyped<double>(problem, plan);
        case ScalarType::Int32:
            return detail::referenceTensorContractionTyped<int32_t>(problem, plan);
        case ScalarType::ComplexFloat32:
            return detail::referenceTensorContractionTyped<std::complex<float>>(problem, plan);
        case ScalarType::ComplexFloat64:
            return detail::referenceTensorContractionTyped<std::complex<double>>(problem, plan);
        default:
            throw std::invalid_argument(
                "Unsupported reference tensor contraction accumulator type.");
    }
}
}  // namespace roc::host_validation
