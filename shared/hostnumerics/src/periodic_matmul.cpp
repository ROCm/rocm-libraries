// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <roc/hostnumerics/periodic_matmul.hpp>
#include <utility>
#include <vector>

#include "detail/checked_arithmetic.hpp"
#include "detail/runtime_tensor_io.hpp"

namespace roc::hostnumerics {
namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double exactLimit = 0x1p24;
constexpr int noQuantum = 10000;

size_t plus(size_t a, size_t b) {
    return detail::checkedAdd(a, b, "Periodic verification size overflow.");
}
size_t times(size_t a, size_t b) {
    return detail::checkedMultiply(a, b, "Periodic verification size overflow.");
}
double upward(double value) {
    return value == 0 ? 0 : std::nextafter(value, infinity);
}
double addBound(double a, double b) {
    return upward(a + b);
}
double multiplyBound(double a, double b) {
    return upward(a * b);
}
double gamma(size_t operations, double unitRoundoff) {
    const double x = multiplyBound(upward(static_cast<double>(operations)), unitRoundoff);
    if (x >= 1) return infinity;
    return upward(x / std::nextafter(1 - x, 0.0));
}
int quantum(double value) {
    return value == 0 ? noQuantum : std::ilogb(std::abs(value)) - 23;
}
bool finiteF32(double value) {
    return std::isfinite(value) && static_cast<double>(static_cast<float>(value)) == value &&
           (value == 0 || std::abs(value) >= std::numeric_limits<float>::min());
}
bool integral(double value) {
    return std::isfinite(value) && std::trunc(value) == value;
}
void validateRegion(const Shape& full, std::span<const size_t> origin, const Shape& extent) {
    if (origin.size() != 2 || extent.rank() != 2)
        throw std::runtime_error("Periodic matmul regions must have rank two.");
    for (size_t d = 0; d < 2; ++d)
        if (origin[d] > full[d] || extent[d] > full[d] - origin[d])
            throw std::runtime_error("Periodic matmul region exceeds the output shape.");
}
VerificationFailure unsupported(std::string reason) {
    return {VerificationStatus::Unsupported, std::move(reason)};
}
VerificationFailure incomplete(std::string reason) {
    return {VerificationStatus::Incomplete, std::move(reason)};
}
struct Magnitudes {
    bool valid = true;
    double maximum = 0;
    int minimumQuantum = noQuantum;
};
Magnitudes magnitudes(const Tensor& tensor, bool exact) {
    Magnitudes result;
    if (scalarTypeInfo(tensor.type()).category == ScalarCategory::Complex) {
        result.valid = false;
        return result;
    }
    const detail::RuntimeTensorReader<double> reader(tensor);
    tensor.shape().forEachCoordinateLastDimensionFastest(
        IndexRange::all(tensor.shape().elementCount()), [&](size_t, std::span<const size_t> index) {
            const double v = reader(index);
            result.valid &= finiteF32(v) && (!exact || integral(v));
            result.maximum = std::max(result.maximum, std::abs(v));
            result.minimumQuantum = std::min(result.minimumQuantum, quantum(v));
        });
    return result;
}
}  // namespace

struct PeriodicMatmulPlan::Data {
    Shape shape;
    size_t k = 0;
    PeriodicMatmulMode mode;
    PeriodicMatmulLimits limits;
    PeriodicMatmulStatistics statistics;
    int minimumProductQuantum = noQuantum;
    std::vector<PeriodicMatmulSegment> segments;
    std::vector<Tensor> products;
};

PeriodicMatmulPlan::PeriodicMatmulPlan(std::shared_ptr<const Data> data)
    : m_data(std::move(data)) {}
const Shape& PeriodicMatmulPlan::shape() const {
    return m_data->shape;
}
size_t PeriodicMatmulPlan::reductionLength() const {
    return m_data->k;
}
const PeriodicMatmulStatistics& PeriodicMatmulPlan::statistics() const {
    return m_data->statistics;
}

PeriodicMatmulPreparation preparePeriodicMatmul(Shape outputShape, size_t reductionLength,
                                                std::span<const PeriodicMatmulSegment> segments,
                                                const MatmulOptions& operation,
                                                PeriodicMatmulMode mode,
                                                const PeriodicMatmulLimits& limits) {
    if (outputShape.rank() != 2)
        throw std::runtime_error("Periodic matmul output must have rank two.");
    if (mode != PeriodicMatmulMode::ExactInteger && mode != PeriodicMatmulMode::FloatingPoint)
        throw std::runtime_error("Invalid periodic matmul numerical mode.");
    if (operation.accumulatorType != ScalarType::Float32 ||
        operation.mathMode != MathMode::Default || operation.conjugateA || operation.conjugateB ||
        (operation.computeTypeA && *operation.computeTypeA != ScalarType::Float32) ||
        (operation.computeTypeB && *operation.computeTypeB != ScalarType::Float32) ||
        !operation.preQuantizationScalesA.empty() || !operation.preQuantizationScalesB.empty() ||
        operation.blockScaleA || operation.blockScaleB ||
        (operation.accumulationRounding != AccumulationRounding::TypeDefault &&
         operation.accumulationRounding != AccumulationRounding::FullPrecision))
        return unsupported(
            "Periodic verification currently requires ordinary real F32 accumulation.");

    auto data = std::make_shared<PeriodicMatmulPlan::Data>();
    data->shape = std::move(outputShape);
    data->k = reductionLength;
    data->mode = mode;
    data->limits = limits;
    auto& stats = data->statistics;
    const bool empty = data->shape.elementCount() == 0;
    size_t nextK = 0, maxTemporaryBytes = 0;
    for (const auto& segment : segments) {
        const auto& a = segment.a;
        const auto& b = segment.b;
        if (a.shape().rank() != 2 || b.shape().rank() != 2 || a.shape()[0] != data->shape[0] ||
            b.shape()[1] != data->shape[1] || a.shape()[1] != b.shape()[0] ||
            segment.reductionBegin != nextK)
            throw std::runtime_error("Periodic matmul segment shapes or K partition are invalid.");
        const size_t ks = a.shape()[1];
        nextK = plus(nextK, ks);
        if (nextK > reductionLength) throw std::runtime_error("Periodic matmul segments exceed K.");
        if (empty || ks == 0) continue;
        if (a.periods()[1] != ks || b.periods()[0] != ks ||
            scalarTypeInfo(a.type()).category == ScalarCategory::Complex ||
            scalarTypeInfo(b.type()).category == ScalarCategory::Complex)
            return unsupported("Periodic matmul supports real A row and B column repetition only.");
        const size_t p = std::min(a.periods()[0], data->shape[0]);
        const size_t q = std::min(b.periods()[1], data->shape[1]);
        stats.referenceMultiplyAdds = plus(stats.referenceMultiplyAdds, times(times(p, q), ks));
        stats.productBytes = plus(stats.productBytes, times(times(p, q), sizeof(double)));
        stats.seedBytes = plus(stats.seedBytes, plus(a.seedStorageBytes(), b.seedStorageBytes()));
        // Encoded materialization, F32 conversion, F64 operands, and row bounds.
        const size_t operands = times(plus(p, q), ks);
        maxTemporaryBytes = std::max(maxTemporaryBytes,
                                     plus(times(operands, 32), times(plus(p, ks), sizeof(double))));
        stats.combinedRowPeriod =
            times(stats.combinedRowPeriod / std::gcd(stats.combinedRowPeriod, p), p);
        stats.combinedColumnPeriod =
            times(stats.combinedColumnPeriod / std::gcd(stats.combinedColumnPeriod, q), q);
    }
    if (nextK != reductionLength)
        throw std::runtime_error("Periodic matmul segments do not cover K.");
    stats.preparationBytesBound =
        plus(plus(stats.seedBytes, stats.productBytes), maxTemporaryBytes);
    if (stats.referenceMultiplyAdds > limits.maxReferenceMultiplyAdds ||
        stats.preparationBytesBound > limits.maxPreparationBytes)
        return incomplete("Periodic reference preparation exceeds its work or storage budget.");
    // The F32 bound must exist before any expensive preparation.
    if (!std::isfinite(gamma(plus(times(reductionLength, 2), 4), 0x1p-24)))
        return unsupported("The periodic F32 forward-error bound is not finite for this K.");

    for (const auto& segment : segments) {
        const size_t ks = segment.a.shape()[1];
        if (empty || ks == 0) continue;
        const size_t p = std::min(segment.a.periods()[0], data->shape[0]);
        const size_t q = std::min(segment.b.periods()[1], data->shape[1]);
        Tensor encodedA(segment.a.type(), Shape{p, ks});
        Tensor encodedB(segment.b.type(), Shape{ks, q});
        segment.a.fillInto(encodedA);
        segment.b.fillInto(encodedB);
        Tensor a =
            encodedA.copyConvertedTo(ScalarType::Float32).copyConvertedTo(ScalarType::Float64);
        Tensor b =
            encodedB.copyConvertedTo(ScalarType::Float32).copyConvertedTo(ScalarType::Float64);
        const auto am = magnitudes(a, mode == PeriodicMatmulMode::ExactInteger);
        const auto bm = magnitudes(b, mode == PeriodicMatmulMode::ExactInteger);
        if (!am.valid || !bm.valid)
            return unsupported(
                "Periodic operands must be finite F32 values, and integral in exact mode.");
        const int productQuantum = am.minimumQuantum + bm.minimumQuantum;
        if (am.maximum != 0 && bm.maximum != 0 && productQuantum < -126)
            return unsupported(
                "Periodic operand products cannot exclude F32 underflow or flush-to-zero.");
        if (am.maximum != 0 && bm.maximum != 0)
            data->minimumProductQuantum = std::min(data->minimumProductQuantum, productQuantum);

        const detail::RuntimeMatrixReader<double> aReader(a), bReader(b);
        std::vector<double> bMax(ks, 0);
        for (size_t k = 0; k < ks; ++k)
            for (size_t j = 0; j < q; ++j) bMax[k] = std::max(bMax[k], std::abs(bReader(k, j)));
        double segmentBound = 0;
        for (size_t i = 0; i < p; ++i) {
            double rowBound = 0;
            for (size_t k = 0; k < ks; ++k)
                rowBound = addBound(rowBound, multiplyBound(std::abs(aReader(i, k)), bMax[k]));
            segmentBound = std::max(segmentBound, rowBound);
        }
        stats.absoluteProductSumBound = addBound(stats.absoluteProductSumBound, segmentBound);
        if (!std::isfinite(stats.absoluteProductSumBound) ||
            stats.absoluteProductSumBound >= std::numeric_limits<float>::max() / 2 ||
            (mode == PeriodicMatmulMode::ExactInteger &&
             stats.absoluteProductSumBound >= exactLimit))
            return unsupported("Periodic partial sums exceed the certified F32 arithmetic range.");
        data->products.push_back(matmul(a, b, ScalarType::Float64,
                                        MatmulOptions(ScalarType::Float64), std::nullopt,
                                        GemmBackend::Automatic));
        data->segments.push_back(segment);
    }
    return PeriodicMatmulPlan(std::move(data));
}

Tensor PeriodicMatmulPlan::productRegion(std::span<const size_t> origin, Shape extent) const {
    validateRegion(shape(), origin, extent);
    if (times(extent.elementCount(), sizeof(double)) > m_data->limits.maxCheckBytes)
        throw std::runtime_error("Periodic product region exceeds its storage budget.");
    Tensor result(ScalarType::Float64, extent);
    struct ProductReader {
        detail::RuntimeMatrixReader<double> reader;
        size_t rows, columns;
    };
    std::vector<ProductReader> products;
    for (const auto& product : m_data->products)
        products.push_back({detail::RuntimeMatrixReader<double>(product),
                            product.shape().dimensions()[0], product.shape().dimensions()[1]});
    const detail::RuntimeMatrixWriter<double> writer(result);
    const auto dimensions = extent.dimensions();
    for (size_t i = 0; i < dimensions[0]; ++i)
        for (size_t j = 0; j < dimensions[1]; ++j) {
            double sum = 0;
            for (const auto& product : products)
                sum += product.reader((origin[0] + i) % product.rows,
                                      (origin[1] + j) % product.columns);
            writer.store(i, j, sum);
        }
    return result;
}

PeriodicMatmulCheck PeriodicMatmulPlan::checkRegion(const Tensor& observed,
                                                    std::span<const size_t> origin,
                                                    const EpilogueOptions& epilogue) const {
    validateRegion(shape(), origin, observed.shape());
    auto fail = [](VerificationStatus status, std::string reason) {
        PeriodicMatmulCheck result;
        result.status = status;
        result.reason = std::move(reason);
        return result;
    };
    if (observed.elementCount() == 0) {
        PeriodicMatmulCheck result;
        result.status = VerificationStatus::Passed;
        return result;
    }
    if (times(observed.elementCount(), 2 * sizeof(double)) > m_data->limits.maxCheckBytes)
        return fail(VerificationStatus::Incomplete,
                    "Periodic comparison exceeds its block budget.");
    if (observed.type() != ScalarType::Float32 || epilogue.computeType != ScalarType::Float32 ||
        !epilogue.inputScaleFactors.empty() || epilogue.auxiliaryInput || epilogue.gateResidual ||
        epilogue.outputConversion != OutputConversion::Default ||
        !std::holds_alternative<IdentityActivation>(epilogue.activation) ||
        epilogue.activationApplication != ActivationApplication::Forward ||
        epilogue.accumulateAmax || epilogue.outputScale.shape().rank() != 0 ||
        epilogue.outputScale.item<double>() != 1 || epilogue.auxiliaryScale.shape().rank() != 0 ||
        epilogue.auxiliaryScale.item<double>() != 1)
        return fail(VerificationStatus::Unsupported,
                    "Periodic comparison requires a linear F32 epilogue and output.");
    const double alpha = epilogue.inputScale ? epilogue.inputScale->item<double>() : 1;
    const double beta = epilogue.addendScale ? epilogue.addendScale->item<double>() : 1;
    const bool exact = m_data->mode == PeriodicMatmulMode::ExactInteger;
    if (!finiteF32(alpha) || !finiteF32(beta) || (exact && (!integral(alpha) || !integral(beta))))
        return fail(VerificationStatus::Unsupported,
                    "Periodic epilogue coefficients are outside its arithmetic contract.");

    Magnitudes cm, bm;
    for (const auto* tensor : {&epilogue.addend, &epilogue.bias})
        if (*tensor && broadcastShapes((*tensor)->shape(), observed.shape()) != observed.shape())
            throw std::runtime_error("Periodic epilogue tensor does not broadcast to the region.");
    if (epilogue.addend && beta != 0) cm = magnitudes(*epilogue.addend, exact);
    if (epilogue.bias) bm = magnitudes(*epilogue.bias, exact);
    if (!cm.valid || !bm.valid)
        return fail(VerificationStatus::Unsupported,
                    "Periodic C/bias must be finite F32 values, and integral in exact mode.");
    if ((alpha != 0 && m_data->minimumProductQuantum != noQuantum &&
         m_data->minimumProductQuantum + quantum(alpha) < -126) ||
        (beta != 0 && cm.maximum != 0 && cm.minimumQuantum + quantum(beta) < -126) ||
        (bm.maximum != 0 && bm.minimumQuantum < -126))
        return fail(VerificationStatus::Unsupported,
                    "Periodic epilogue cannot exclude F32 underflow or flush-to-zero.");
    const double bound =
        addBound(addBound(multiplyBound(std::abs(alpha), statistics().absoluteProductSumBound),
                          multiplyBound(std::abs(beta), cm.maximum)),
                 bm.maximum);
    if (!std::isfinite(bound) || bound >= std::numeric_limits<float>::max() / 2 ||
        (exact && bound >= exactLimit))
        return fail(VerificationStatus::Unsupported,
                    "Periodic epilogue exceeds the certified F32 range.");

    ComparisonOptions comparison;
    if (!exact) {
        const size_t operations = plus(times(m_data->k, 2), 4);
        const double rho =
            addBound(gamma(operations, 0x1p-24),
                     gamma(plus(operations, times(m_data->products.size(), 2)), 0x1p-53));
        comparison.absoluteTolerance = multiplyBound(rho, bound);
    }
    const Tensor product = productRegion(origin, observed.shape());
    EpilogueOptions reference = epilogue;
    reference.computeType = ScalarType::Float64;
    // beta=0 does not read C, including a nonfinite C.
    if (beta == 0) reference.addend.reset();
    const auto expected = referenceEpilogue(product, {.output = ScalarType::Float64}, reference);
    PeriodicMatmulCheck result;
    result.comparison = compare(observed, expected.output, comparison);
    result.status =
        result.comparison.passed() ? VerificationStatus::Passed : VerificationStatus::Mismatch;
    result.absoluteTolerance = comparison.absoluteTolerance;
    for (auto& mismatch : result.comparison.reportedMismatches) {
        for (size_t d = 0; d < 2; ++d) mismatch.coordinates[d] += origin[d];
        mismatch.index = shape().linearIndexLastDimensionFastest(mismatch.coordinates);
    }
    return result;
}
}  // namespace roc::hostnumerics
