// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <roc/hostnumerics/freivalds.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>
#include <utility>

#include "detail/checked_arithmetic.hpp"
#include "detail/exact_integer.hpp"
#include "detail/runtime_tensor_io.hpp"

namespace roc::hostnumerics {
namespace {
constexpr uint64_t p = freivaldsModulus;
constexpr uint64_t boundCap = uint64_t{1} << 61;
std::atomic<uint64_t> nextIdentity{1};

uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
uint64_t reduce64(uint64_t x) {
    const uint64_t y = (x & p) + (x >> 61);
    return y >= p ? y - p : y;
}
uint64_t add(uint64_t a, uint64_t b) {
    const uint64_t sum = a + b;
    return sum >= p ? sum - p : sum;
}
uint64_t multiplyResidues(uint64_t a, uint64_t b) {
    // Portable 31-bit limbs. Since 2^61 == 1, the high product at 2^62
    // contributes twice its value. Every intermediate fits in uint64_t.
    constexpr uint64_t mask31 = (uint64_t{1} << 31) - 1;
    constexpr uint64_t mask30 = (uint64_t{1} << 30) - 1;
    const uint64_t al = a & mask31, ah = a >> 31, bl = b & mask31, bh = b >> 31;
    const uint64_t cross = al * bh + ah * bl;
    return add(add(reduce64(al * bl), add((cross & mask30) << 31, cross >> 30)), 2 * ah * bh);
}
struct ModularAccumulator {
#if defined(__SIZEOF_INT128__)
    unsigned __int128 sum = 0;
    unsigned terms = 0;
    static uint64_t reduce(unsigned __int128 x) {
        x = (x & p) + (x >> 61);
        x = (x & p) + (x >> 61);
        return reduce64(static_cast<uint64_t>(x));
    }
    void append(uint64_t a, uint64_t b) {
        sum += static_cast<unsigned __int128>(a) * b;
        if (++terms == 32) {
            sum = reduce(sum);
            terms = 0;
        }
    }
    uint64_t value() const {
        return reduce(sum);
    }
#else
    uint64_t sum = 0;
    void append(uint64_t a, uint64_t b) {
        sum = add(sum, multiplyResidues(a, b));
    }
    uint64_t value() const {
        return sum;
    }
#endif
};
size_t plus(size_t a, size_t b) {
    return detail::checkedAdd(a, b, "Freivalds size overflow.");
}
size_t times(size_t a, size_t b) {
    return detail::checkedMultiply(a, b, "Freivalds size overflow.");
}
// Conservative payload bound: two projection copies, coordinates for sorting,
// and worst-case reported exceptions. Complete Tensor backing storage and
// exception vector capacity are charged. Allocator bookkeeping is excluded.
size_t observationBytes(size_t projectionBytes, size_t projectionSize, size_t exceptionCapacity,
                        size_t reportedAxes) {
    return plus(plus(plus(projectionBytes, times(projectionSize, sizeof(uint64_t))),
                     times(exceptionCapacity, 2 * sizeof(FreivaldsException) + sizeof(size_t))),
                times(reportedAxes, sizeof(size_t)));
}
uint64_t absolute(int64_t v) {
    return v < 0 ? static_cast<uint64_t>(-v) : static_cast<uint64_t>(v);
}
uint64_t boundAdd(uint64_t a, uint64_t b) {
    return b >= boundCap - a ? boundCap : a + b;
}
uint64_t boundMultiply(uint64_t a, uint64_t b) {
    return a != 0 && b >= boundCap / a ? boundCap : a * b;
}
bool factor(int64_t a, int64_t b, int64_t& output) {
    if (boundMultiply(absolute(a), absolute(b)) >= boundCap) return false;
    output = a * b;
    return true;
}
bool real(const Tensor& t) {
    return scalarTypeInfo(t.type()).category != ScalarCategory::Complex;
}
Tensor snapshot(const Tensor& input) {
    Tensor result(input.type(), input.shape());
    result.copyLogicalElementsFrom(input);
    return result;
}
bool validateIntegers(const Tensor& t, ScalarType computeType) {
    if (!real(t)) return false;
    const detail::RuntimeTensorReader<double> reader(t);
    const detail::PreparedExactOutput convert(computeType);
    bool valid = true;
    t.shape().forEachCoordinateLastDimensionFastest(
        IndexRange::all(t.shape().elementCount()), [&](size_t, std::span<const size_t> at) {
            const double v = reader(at);
            valid &= detail::isExactInteger(v) && convert(v) == v;
        });
    return valid;
}
// The same ordered operand scales and compute-input conversion as matmul.
// Certification requires every multiplication and conversion to be exact;
// rounded operands never silently acquire a different Freivalds contract.
std::optional<Tensor> exactOperand(const Tensor& input, const std::vector<Tensor>& scales,
                                   std::optional<ScalarType> computeType,
                                   ScalarType accumulatorType) {
    if (!validateIntegers(input, accumulatorType)) return std::nullopt;
    Tensor value = snapshot(input);
    for (const auto& scale : scales) {
        if (broadcastShapes(scale.shape(), input.shape()) != input.shape())
            throw std::runtime_error("Freivalds operand scale does not broadcast to its input.");
        if (!validateIntegers(scale, accumulatorType)) return std::nullopt;
        value = multiply(value, scale, ScalarType::Float64, ScalarType::Float64, std::nullopt);
        const detail::RuntimeTensorReader<double> reader(value);
        bool bounded = true;
        value.shape().forEachCoordinateLastDimensionFastest(
            IndexRange::all(value.shape().elementCount()),
            [&](size_t, std::span<const size_t> at) { bounded &= std::abs(reader(at)) < 0x1p53; });
        if (!bounded || !validateIntegers(value, accumulatorType)) return std::nullopt;
    }
    if (computeType &&
        (detail::exactIntegerLimit(*computeType) == 0 || !validateIntegers(value, *computeType)))
        return std::nullopt;
    if (scales.empty()) return value;
    return value.copyConvertedTo(accumulatorType);
}
class BroadcastIntegerReader {
   public:
    explicit BroadcastIntegerReader(const Tensor& tensor)
        : m_reader(tensor),
          m_rank(tensor.shape().rank()),
          m_row(m_rank == 2 && tensor.shape().dimensions()[0] != 1),
          m_column(m_rank != 0 && tensor.shape().dimensions()[m_rank - 1] != 1) {}
    int64_t operator()(size_t row, size_t column) const {
        if (m_rank == 0) return m_reader(std::span<const size_t>{});
        if (m_rank == 1) return m_reader({m_column ? column : 0});
        return m_reader({m_row ? row : 0, m_column ? column : 0});
    }

   private:
    detail::RuntimeTensorReader<int64_t> m_reader;
    size_t m_rank;
    bool m_row, m_column;
};
std::vector<uint64_t> project(const Tensor& tensor, std::span<const uint64_t> probe,
                              bool transpose) {
    const size_t rows = tensor.shape()[transpose ? 1 : 0];
    const size_t columns = tensor.shape()[transpose ? 0 : 1];
    const detail::RuntimeMatrixReader<int64_t> reader(tensor);
    std::vector<uint64_t> result(rows);
#pragma omp parallel for if (rows * columns >= 16384)
    for (ptrdiff_t r = 0; r < static_cast<ptrdiff_t>(rows); ++r) {
        ModularAccumulator sum;
        for (size_t c = 0; c < columns; ++c)
            sum.append(freivaldsResidue(transpose ? reader(c, r) : reader(r, c)), probe[c]);
        result[r] = sum.value();
    }
    return result;
}
FreivaldsPreparationFailure unsupported(std::string reason) {
    return {VerificationStatus::Unsupported, std::move(reason)};
}
FreivaldsPreparationFailure incomplete(std::string reason) {
    return {VerificationStatus::Incomplete, std::move(reason)};
}
}  // namespace

uint64_t freivaldsAdd(uint64_t a, uint64_t b) {
    if (a >= p || b >= p) throw std::runtime_error("Freivalds operands must be residues.");
    return add(a, b);
}
uint64_t freivaldsMultiply(uint64_t a, uint64_t b) {
    if (a >= p || b >= p) throw std::runtime_error("Freivalds operands must be residues.");
    return multiplyResidues(a, b);
}
uint64_t freivaldsResidue(int64_t value) {
    const int64_t residue = value % static_cast<int64_t>(p);
    return residue < 0 ? static_cast<uint64_t>(residue + static_cast<int64_t>(p))
                       : static_cast<uint64_t>(residue);
}
namespace {
template <FreivaldsProbeAlgorithm Algorithm>
uint64_t preparedProbe(uint64_t seed, uint64_t index) {
    uint64_t word = splitmix(splitmix(seed) ^ index);
    if constexpr (Algorithm == FreivaldsProbeAlgorithm::SplitMix64Legacy) return 1 + word % (p - 1);
    // Mapping uniform words to 61 bits and rejecting endpoints avoids modulo
    // bias. The seeded PRNG is reproducible, not a proof of independent probes.
    for (uint64_t attempt = 1;; ++attempt) {
        const uint64_t residue = word & p;
        if (residue != 0 && residue != p) return residue;
        word = splitmix(word + attempt);
    }
}
using ProbeFunction = uint64_t (*)(uint64_t, uint64_t);
ProbeFunction probeFunction(FreivaldsProbeAlgorithm algorithm) {
    switch (algorithm) {
        case FreivaldsProbeAlgorithm::SplitMix64Legacy:
            return preparedProbe<FreivaldsProbeAlgorithm::SplitMix64Legacy>;
        case FreivaldsProbeAlgorithm::SplitMix64Rejection:
            return preparedProbe<FreivaldsProbeAlgorithm::SplitMix64Rejection>;
        default:
            throw std::runtime_error("Invalid Freivalds probe algorithm.");
    }
}
}  // namespace
uint64_t freivaldsProbe(uint64_t seed, uint64_t index, FreivaldsProbeAlgorithm algorithm) {
    return probeFunction(algorithm)(seed, index);
}

Tensor reduceFreivaldsProjections(const Tensor& partials) {
    if (partials.type() != ScalarType::UInt64 || partials.shape().rank() != 2)
        throw std::runtime_error("Freivalds partials require a rank-two UInt64 Tensor.");
    Tensor result(ScalarType::UInt64, Shape{partials.shape()[1]});
    const detail::RuntimeMatrixReader<uint64_t> reader(partials);
    const detail::RuntimeTensorWriter<uint64_t> writer(result);
    const auto dimensions = partials.shape().dimensions();
    for (size_t coordinate = 0; coordinate < dimensions[1]; ++coordinate) {
        uint64_t sum = 0;
        for (size_t chunk = 0; chunk < dimensions[0]; ++chunk) {
            const auto residue = reader(chunk, coordinate);
            if (residue >= p) throw std::runtime_error("Freivalds operands must be residues.");
            sum = add(sum, residue);
        }
        writer.store({coordinate}, sum);
    }
    return result;
}

struct FreivaldsPlan::Data {
    // Never expose mutable Tensor handles into private plan storage. The small
    // integer work arrays stay private; public numerical payloads are Tensors.
    struct Request {
        uint64_t identity = 0;
        Shape shape;
        ScalarType outputType = ScalarType::Float32;
        double fastMagnitudeLimit = 0;
        size_t maximumExceptions = 0;
        std::vector<uint64_t> rightProbe, leftProbe;
    } request;
    FreivaldsStatistics statistics;
    FreivaldsLimits limits;
    ScalarType computeType = ScalarType::Float32;
    size_t k = 0;
    int64_t alpha = 1, beta = 0, outputScale = 1;
    std::vector<int64_t> rowScale, columnScale;
    std::optional<Tensor> a, b, c, bias;
    std::vector<uint64_t> rows, columns;
};
FreivaldsPlan::FreivaldsPlan(std::shared_ptr<const Data> data) : m_data(std::move(data)) {}
FreivaldsRequest FreivaldsPlan::copyRequest() const {
    const auto& r = m_data->request;
    return {r.identity,
            r.shape,
            r.outputType,
            r.fastMagnitudeLimit,
            r.maximumExceptions,
            Tensor::copyNativeStorage(std::span(r.rightProbe)),
            Tensor::copyNativeStorage(std::span(r.leftProbe))};
}
uint64_t FreivaldsPlan::requestIdentity() const {
    return m_data->request.identity;
}
const FreivaldsStatistics& FreivaldsPlan::statistics() const {
    return m_data->statistics;
}
Tensor FreivaldsPlan::copyExpectedRowSums() const {
    return Tensor::copyNativeStorage(std::span(m_data->rows));
}
Tensor FreivaldsPlan::copyExpectedColumnSums() const {
    return Tensor::copyNativeStorage(std::span(m_data->columns));
}

FreivaldsPreparation prepareFreivalds(const Tensor& a, const Tensor& b,
                                      const MatmulOptions& operation,
                                      const EpilogueOptions& epilogue, ScalarType outputType,
                                      const FreivaldsProbeOptions& probes,
                                      const FreivaldsLimits& limits) {
    if (a.shape().rank() != 2 || b.shape().rank() != 2 || a.shape()[1] != b.shape()[0])
        throw std::runtime_error("Freivalds requires compatible rank-two operands.");
    const size_t m = a.shape()[0], n = b.shape()[1], k = a.shape()[1];
    const Shape shape{m, n};
    const auto count = shape.elementCount();
    // Supported limits are zero or positive powers of two no greater than 2^53.
    const auto accLimit =
        static_cast<uint64_t>(detail::exactIntegerLimit(operation.accumulatorType));
    if (accLimit == 0 || detail::exactIntegerLimit(outputType) == 0 ||
        epilogue.computeType != operation.accumulatorType ||
        operation.mathMode != MathMode::Default || operation.blockScaleA || operation.blockScaleB ||
        operation.blockSizeA || operation.blockSizeB ||
        (operation.accumulationRounding != AccumulationRounding::TypeDefault &&
         operation.accumulationRounding != AccumulationRounding::FullPrecision) ||
        epilogue.auxiliaryInput || epilogue.gateResidual || epilogue.accumulateAmax ||
        !std::holds_alternative<IdentityActivation>(epilogue.activation) ||
        epilogue.activationApplication != ActivationApplication::Forward ||
        (epilogue.outputConversion != OutputConversion::Default &&
         !(epilogue.outputConversion == OutputConversion::SaturatingInt8 &&
           outputType == ScalarType::Int8)))
        return unsupported(
            "Freivalds requires supported exact-integer matmul and linear epilogue semantics.");
    if (probes.algorithm != FreivaldsProbeAlgorithm::SplitMix64Legacy &&
        probes.algorithm != FreivaldsProbeAlgorithm::SplitMix64Rejection)
        throw std::runtime_error("Invalid Freivalds probe algorithm.");
    auto data = std::make_shared<FreivaldsPlan::Data>();
    data->limits = limits;
    data->computeType = operation.accumulatorType;
    data->k = k;
    auto& request = data->request;
    request.identity = nextIdentity.fetch_add(1, std::memory_order_relaxed);
    request.shape = shape;
    request.outputType = outputType;
    request.fastMagnitudeLimit = detail::exactObservationLimit(outputType);
    if (count == 0) return FreivaldsPlan(std::move(data));

    const double alpha = epilogue.inputScale ? epilogue.inputScale->item<double>() : 1;
    const double beta =
        epilogue.addend && epilogue.addendScale ? epilogue.addendScale->item<double>() : 1;
    const double scaleD = epilogue.outputScale.item<double>();
    if (!detail::isExactInteger(alpha) || !detail::isExactInteger(beta) ||
        !detail::isExactInteger(scaleD) ||
        detail::roundExactOutput(alpha, operation.accumulatorType) != alpha ||
        detail::roundExactOutput(beta, operation.accumulatorType) != beta ||
        detail::roundExactOutput(scaleD, operation.accumulatorType) != scaleD)
        return unsupported("Freivalds requires exactly representable integer coefficients.");
    data->outputScale = static_cast<int64_t>(scaleD);
    if (!factor(static_cast<int64_t>(alpha), data->outputScale, data->alpha) ||
        !factor(epilogue.addend ? static_cast<int64_t>(beta) : 0, data->outputScale, data->beta))
        return unsupported("Freivalds combined coefficients exceed the integer range.");
    const bool useC = epilogue.addend && beta != 0;
    for (const auto* t : {&epilogue.addend, &epilogue.bias})
        if (*t && (broadcastShapes((*t)->shape(), shape) != shape || (*t)->shape().rank() > 2))
            throw std::runtime_error("Freivalds epilogue input does not broadcast to the output.");
    const bool transformA = !operation.preQuantizationScalesA.empty();
    const bool transformB = !operation.preQuantizationScalesB.empty();
    size_t bytes = plus(storageBytesForLayout(transformA ? operation.accumulatorType : a.type(),
                                              Layout::contiguousLastDimensionFastest(a.shape())),
                        storageBytesForLayout(transformB ? operation.accumulatorType : b.type(),
                                              Layout::contiguousLastDimensionFastest(b.shape())));
    const size_t transformBytes =
        times(3 * sizeof(double),
              plus(transformA ? a.elementCount() : 0, transformB ? b.elementCount() : 0));
    if (useC)
        bytes = plus(bytes, storageBytesForLayout(
                                epilogue.addend->type(),
                                Layout::contiguousLastDimensionFastest(epilogue.addend->shape())));
    if (epilogue.bias)
        bytes = plus(bytes, storageBytesForLayout(
                                epilogue.bias->type(),
                                Layout::contiguousLastDimensionFastest(epilogue.bias->shape())));
    data->statistics.retainedInputBytes = bytes;
    data->statistics.preparationMultiplyAdds =
        times(2, plus(plus(times(m, k), times(k, n)),
                      times(count, size_t(useC) + size_t(bool(epilogue.bias)))));
    data->statistics.preparationMultiplyAdds =
        plus(data->statistics.preparationMultiplyAdds,
             plus(times(a.elementCount(), operation.preQuantizationScalesA.size()),
                  times(b.elementCount(), operation.preQuantizationScalesB.size())));
    const size_t summaryBytes = times(plus(plus(m, n), k), 12 * sizeof(uint64_t));
    if (plus(plus(bytes, summaryBytes), transformBytes) > limits.maxPreparationBytes ||
        data->statistics.preparationMultiplyAdds > limits.maxPreparationMultiplyAdds)
        return incomplete("Freivalds preparation exceeds its work or storage budget.");
    const size_t axes = plus(m, n);
    const size_t reportedAxes =
        plus(std::min(m, limits.maxReportedIndices), std::min(n, limits.maxReportedIndices));
    const size_t observationBase =
        observationBytes(times(axes, sizeof(uint64_t)), axes, 0, reportedAxes);
    if (observationBase > limits.maxObservationBytes)
        return incomplete("Freivalds projections exceed the observation storage budget.");
    request.maximumExceptions =
        std::min({count, size_t(limits.maxRecomputeMultiplyAdds / std::max(size_t(1), k)),
                  (limits.maxObservationBytes - observationBase) /
                      (2 * sizeof(FreivaldsException) + sizeof(size_t))});
    data->rowScale.assign(m, 1);
    data->columnScale.assign(n, 1);
    for (size_t factorIndex = 0; factorIndex < epilogue.inputScaleFactors.size(); ++factorIndex) {
        const auto& f = epilogue.inputScaleFactors[factorIndex];
        if (broadcastShapes(f.shape(), shape) != shape || f.shape().rank() > 2)
            throw std::runtime_error("Freivalds input factor does not broadcast to the output.");
        const bool rowFactor = f.shape().rank() == 2 && f.shape()[1] == 1;
        const bool columnFactor = f.shape().rank() < 2 || f.shape()[0] == 1;
        if (!rowFactor && !columnFactor)
            return unsupported(
                "Freivalds input factors must be integral scalar, row or column factors.");
        if (!validateIntegers(f, operation.accumulatorType))
            return unsupported(
                "Freivalds input factors must be integers represented exactly by the computation.");
        const BroadcastIntegerReader reader(f);
        auto& values = rowFactor ? data->rowScale : data->columnScale;
        for (size_t i = 0; i < values.size(); ++i)
            if (!factor(values[i], reader(rowFactor ? i : 0, rowFactor ? 0 : i), values[i]))
                return unsupported("Freivalds combined input factors exceed the integer range.");
        // Factors are combined in their declared order, before multiplying the
        // matmul result. A later zero must not hide an inexact/overflowing prefix.
        uint64_t rowMax = 0, columnMax = 0;
        for (int64_t value : data->rowScale) rowMax = std::max(rowMax, absolute(value));
        for (int64_t value : data->columnScale) columnMax = std::max(columnMax, absolute(value));
        if (boundMultiply(rowMax, columnMax) >= accLimit)
            return unsupported(
                "Freivalds input factor prefix exceeds the compute type's exact "
                "integer range.");
    }
    data->a = exactOperand(a, operation.preQuantizationScalesA, operation.computeTypeA,
                           operation.accumulatorType);
    data->b = exactOperand(b, operation.preQuantizationScalesB, operation.computeTypeB,
                           operation.accumulatorType);
    if (!data->a || !data->b ||
        (useC && !validateIntegers(*epilogue.addend, operation.accumulatorType)))
        return unsupported(
            "Freivalds found a non-integer value in A, B or C, or an input not "
            "represented exactly by the computation.");
    if (epilogue.bias && !validateIntegers(*epilogue.bias, operation.accumulatorType))
        return unsupported(
            "Freivalds requires integer bias entries represented exactly by the computation.");
    if (useC) data->c = snapshot(*epilogue.addend);
    if (epilogue.bias) data->bias = snapshot(*epilogue.bias);

    const detail::RuntimeMatrixReader<int64_t> aReader(*data->a), bReader(*data->b);
    const auto cReader = data->c ? std::optional(BroadcastIntegerReader(*data->c)) : std::nullopt;
    const auto biasReader =
        data->bias ? std::optional(BroadcastIntegerReader(*data->bias)) : std::nullopt;
    std::vector<uint64_t> bMax(k, 0);
    for (size_t z = 0; z < k; ++z)
        for (size_t j = 0; j < n; ++j) bMax[z] = std::max(bMax[z], absolute(bReader(z, j)));
    uint64_t columnMax = 0;
    for (int64_t v : data->columnScale) columnMax = std::max(columnMax, absolute(v));
    uint64_t partialMax = 0, resultMax = 0, epilogueMax = 0;
    for (size_t i = 0; i < m; ++i) {
        uint64_t raw = 0;
        for (size_t z = 0; z < k; ++z)
            raw = boundAdd(raw, boundMultiply(absolute(aReader(i, z)), bMax[z]));
        const uint64_t inputScaled =
            boundMultiply(boundMultiply(raw, absolute(data->rowScale[i])), columnMax);
        const uint64_t scaled = boundMultiply(inputScaled, absolute(data->alpha));
        // Include the pre-output-scale expression, which the GPU must form too.
        const uint64_t beforeOutputScale =
            boundMultiply(boundMultiply(boundMultiply(raw, absolute(data->rowScale[i])), columnMax),
                          absolute(static_cast<int64_t>(alpha)));
        partialMax = std::max({partialMax, raw, inputScaled, scaled, beforeOutputScale});
        uint64_t cMax = 0, biasMax = 0;
        for (size_t j = 0; j < n; ++j) {
            if (data->c) cMax = std::max(cMax, absolute((*cReader)(i, j)));
            if (data->bias) biasMax = std::max(biasMax, absolute((*biasReader)(i, j)));
        }
        resultMax = std::max(resultMax,
                             boundAdd(boundAdd(scaled, boundMultiply(absolute(data->beta), cMax)),
                                      boundMultiply(absolute(data->outputScale), biasMax)));
        epilogueMax =
            std::max(epilogueMax,
                     boundAdd(boundAdd(beforeOutputScale,
                                       boundMultiply(absolute(static_cast<int64_t>(beta)), cMax)),
                              biasMax));
    }
    data->statistics.maximumPartialMagnitude = static_cast<double>(partialMax);
    data->statistics.maximumResultMagnitude = static_cast<double>(resultMax);
    if (partialMax >= accLimit || resultMax >= accLimit || epilogueMax >= accLimit)
        return FreivaldsPreparationFailure{VerificationStatus::Unsupported,
                                           "Freivalds refuses this configuration: partial sums or "
                                           "results exceed the range the compute type holds "
                                           "exactly.",
                                           data->statistics};
    request.rightProbe.resize(n);
    request.leftProbe.resize(m);
    const auto probe = probeFunction(probes.algorithm);
    std::vector<uint64_t> right(n), left(m);
    for (size_t j = 0; j < n; ++j) {
        request.rightProbe[j] = probe(2 * probes.seed, j);
        right[j] = multiplyResidues(request.rightProbe[j], freivaldsResidue(data->columnScale[j]));
    }
    for (size_t i = 0; i < m; ++i) {
        request.leftProbe[i] = probe(2 * probes.seed + 1, i);
        left[i] = multiplyResidues(request.leftProbe[i], freivaldsResidue(data->rowScale[i]));
    }
    auto br = project(*data->b, right, false);
    data->rows = project(*data->a, br, false);
    auto ta = project(*data->a, left, true);
    data->columns = project(*data->b, ta, true);
    for (size_t i = 0; i < m; ++i)
        data->rows[i] = multiplyResidues(
            multiplyResidues(freivaldsResidue(data->alpha), freivaldsResidue(data->rowScale[i])),
            data->rows[i]);
    for (size_t j = 0; j < n; ++j)
        data->columns[j] = multiplyResidues(
            multiplyResidues(freivaldsResidue(data->alpha), freivaldsResidue(data->columnScale[j])),
            data->columns[j]);
    for (const auto& [t, coefficient] :
         {std::pair{&data->c, data->beta}, std::pair{&data->bias, data->outputScale}}) {
        if (!*t) continue;
        const auto broadcast = (*t)->broadcastTo(shape);
        const auto r = project(broadcast, request.rightProbe, false);
        const auto c = project(broadcast, request.leftProbe, true);
        for (size_t i = 0; i < m; ++i)
            data->rows[i] =
                add(data->rows[i], multiplyResidues(freivaldsResidue(coefficient), r[i]));
        for (size_t j = 0; j < n; ++j)
            data->columns[j] =
                add(data->columns[j], multiplyResidues(freivaldsResidue(coefficient), c[j]));
    }
    return FreivaldsPlan(std::move(data));
}

int64_t FreivaldsPlan::exactValue(size_t row, size_t column) const {
    if (row >= m_data->request.shape[0] || column >= m_data->request.shape[1])
        throw std::runtime_error("Freivalds coordinate exceeds the output shape.");
    // The certified bound makes each product and arbitrary partial sum exact
    // in double (including F64's integer range) and representable in int64_t.
    const detail::RuntimeMatrixReader<int64_t> aReader(*m_data->a), bReader(*m_data->b);
    double sum = 0;
    for (size_t k = 0; k < m_data->k; ++k)
        sum += static_cast<double>(aReader(row, k)) * static_cast<double>(bReader(k, column));
    const double inputFactor = static_cast<double>(m_data->rowScale[row]) *
                               static_cast<double>(m_data->columnScale[column]);
    const double scaled = sum * inputFactor * static_cast<double>(m_data->alpha);
    const double c = m_data->c
                         ? static_cast<double>(BroadcastIntegerReader(*m_data->c)(row, column)) *
                               static_cast<double>(m_data->beta)
                         : 0;
    const double bias =
        m_data->bias ? static_cast<double>(BroadcastIntegerReader(*m_data->bias)(row, column)) *
                           static_cast<double>(m_data->outputScale)
                     : 0;
    return static_cast<int64_t>(scaled + c + bias);
}
double FreivaldsPlan::roundedValue(size_t row, size_t column) const {
    return detail::roundExactOutput(static_cast<double>(exactValue(row, column)),
                                    m_data->request.outputType);
}

FreivaldsCheck FreivaldsPlan::check(const Tensor& observed, size_t exceptionLimit,
                                    const FreivaldsCheckOptions& options) const {
    if (observed.shape() != m_data->request.shape || observed.type() != m_data->request.outputType)
        throw std::runtime_error("Freivalds observed tensor shape/type does not match the plan.");
    FreivaldsObservation observation;
    observation.requestIdentity = m_data->request.identity;
    observation.checkedElements = observed.elementCount();
    observation.rowSums = Tensor(ScalarType::UInt64, Shape{m_data->rows.size()});
    observation.columnSums = Tensor(ScalarType::UInt64, Shape{m_data->columns.size()});
    if (observation.checkedElements == 0) return check(observation, exceptionLimit, options);
    exceptionLimit = std::min(exceptionLimit, m_data->request.maximumExceptions);
    const detail::RuntimeMatrixReader<double> reader(observed);
    const detail::RuntimeVectorReader<uint64_t> rowsReader(observation.rowSums),
        columnsReader(observation.columnSums);
    const detail::RuntimeTensorWriter<uint64_t> rowsWriter(observation.rowSums),
        columnsWriter(observation.columnSums);
    const auto dimensions = observed.shape().dimensions();
    for (size_t i = 0; i < dimensions[0]; ++i)
        for (size_t j = 0; j < dimensions[1]; ++j) {
            const double value = reader(i, j);
            if (!detail::isExactInteger(value) ||
                std::abs(value) >= m_data->request.fastMagnitudeLimit) {
                if (observation.exceptions.size() >= exceptionLimit) {
                    observation.complete = false;
                    return {VerificationStatus::Incomplete,
                            "Freivalds exceptions exceed the recomputation or storage budget."};
                }
                if (observation.exceptions.size() == observation.exceptions.capacity())
                    observation.exceptions.reserve(
                        std::min(exceptionLimit,
                                 std::max(size_t(1), 2 * observation.exceptions.capacity())));
                observation.exceptions.push_back({i, j, value});
            } else {
                const auto residue = freivaldsResidue(static_cast<int64_t>(value));
                rowsWriter.store(
                    {i},
                    add(rowsReader[i], multiplyResidues(residue, m_data->request.rightProbe[j])));
                columnsWriter.store(
                    {j},
                    add(columnsReader[j], multiplyResidues(residue, m_data->request.leftProbe[i])));
            }
        }
    return check(observation, exceptionLimit, options);
}

FreivaldsCheck FreivaldsPlan::check(const FreivaldsObservation& observation, size_t exceptionLimit,
                                    const FreivaldsCheckOptions& options) const {
    FreivaldsCheck result;
    const double error = options.maximumAbsoluteComputationError;
    if (!std::isfinite(error) || error < 0 || error >= 0.5) {
        result.status = VerificationStatus::Unsupported;
        result.reason = "Freivalds computation error must be finite, nonnegative and below 0.5.";
        return result;
    }
    if (error != 0 && m_data->computeType != ScalarType::Float32) {
        result.status = VerificationStatus::Unsupported;
        result.reason = "Freivalds computation error bounds currently require F32 computation.";
        return result;
    }
    auto incompleteResult = [&](std::string reason) {
        result.status = VerificationStatus::Incomplete;
        result.reason = std::move(reason);
        return result;
    };
    if (observation.requestIdentity != m_data->request.identity || !observation.complete ||
        observation.checkedElements != m_data->request.shape.elementCount() ||
        observation.rowSums.type() != ScalarType::UInt64 ||
        observation.columnSums.type() != ScalarType::UInt64 ||
        observation.rowSums.shape() != Shape{m_data->rows.size()} ||
        observation.columnSums.shape() != Shape{m_data->columns.size()})
        return incompleteResult(
            "Freivalds observation is stale, incomplete, or has incorrect dimensions.");
    const size_t reportedAxes =
        plus(std::min(m_data->rows.size(), m_data->limits.maxReportedIndices),
             std::min(m_data->columns.size(), m_data->limits.maxReportedIndices));
    if (observationBytes(
            plus(observation.rowSums.rawEncodedBackingStorage().size(),
                 observation.columnSums.rawEncodedBackingStorage().size()),
            plus(observation.rowSums.elementCount(), observation.columnSums.elementCount()),
            observation.exceptions.capacity(), reportedAxes) > m_data->limits.maxObservationBytes)
        return incompleteResult("Freivalds observation exceeds its storage budget.");
    const detail::RuntimeVectorReader<uint64_t> rowsReader(observation.rowSums),
        columnsReader(observation.columnSums);
    if (observation.exceptions.size() > std::min(exceptionLimit, m_data->request.maximumExceptions))
        return incompleteResult("Freivalds exceptions exceed the recomputation or storage budget.");
    std::vector<size_t> coordinates;
    coordinates.reserve(observation.exceptions.size());
    for (const auto& exception : observation.exceptions) {
        if (exception.row >= m_data->request.shape[0] ||
            exception.column >= m_data->request.shape[1])
            return incompleteResult("Freivalds exception coordinates are invalid.");
        coordinates.push_back(exception.row * m_data->request.shape[1] + exception.column);
    }
    std::sort(coordinates.begin(), coordinates.end());
    if (std::adjacent_find(coordinates.begin(), coordinates.end()) != coordinates.end())
        return incompleteResult("Freivalds exception coordinates are duplicated.");
    // Copy only logical residues, after validating the complete incoming storage
    // budget. Do not deep-copy padding or mutate the caller's observation.
    std::vector<uint64_t> rows(m_data->rows.size()), columns(m_data->columns.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = rowsReader[i];
        if (rows[i] >= p)
            return incompleteResult("Freivalds observation contains a noncanonical residue.");
    }
    for (size_t j = 0; j < columns.size(); ++j) {
        columns[j] = columnsReader[j];
        if (columns[j] >= p)
            return incompleteResult("Freivalds observation contains a noncanonical residue.");
    }
    result.badRows.reserve(std::min(rows.size(), m_data->limits.maxReportedIndices));
    result.badColumns.reserve(std::min(columns.size(), m_data->limits.maxReportedIndices));
    result.badElements.reserve(
        std::min(observation.exceptions.size(), m_data->limits.maxReportedIndices));
    const detail::PreparedExactOutput convertOutput(m_data->request.outputType);
    const auto computationEndpoint = [error](double exact, bool upper) {
        const double delta = upper ? error : -error;
        double sum = exact + delta;
        const double part = sum - exact;
        // TwoSum retains the rounding error without relying on a wider long double.
        const double remainder = (exact - (sum - part)) + (delta - part);
        // The computed value is representable in F32. Round endpoints inward
        // so adding a small bound to a large integer cannot admit an extra ULP.
        if ((upper && remainder < 0) || (!upper && remainder > 0))
            sum = std::nextafter(sum, upper ? -std::numeric_limits<double>::infinity()
                                            : std::numeric_limits<double>::infinity());
        float value = static_cast<float>(sum);
        if ((upper && static_cast<double>(value) > sum) ||
            (!upper && static_cast<double>(value) < sum))
            value = std::nextafter(value, upper ? -std::numeric_limits<float>::infinity()
                                                : std::numeric_limits<float>::infinity());
        return static_cast<double>(value);
    };
    for (const auto& exception : observation.exceptions) {
        const int64_t exact = exactValue(exception.row, exception.column);
        const double rounded = convertOutput(static_cast<double>(exact));
        bool matches = rounded == exception.storedValue;
        if (!matches && error != 0) {
            // Compare encoded endpoints: a small computation error can cross
            // an output-rounding tie even when it is far below one output ULP.
            const double lower =
                convertOutput(computationEndpoint(static_cast<double>(exact), false));
            const double upper =
                convertOutput(computationEndpoint(static_cast<double>(exact), true));
            matches = exception.storedValue >= lower && exception.storedValue <= upper;
        }
        if (!matches) {
            ++result.badElementCount;
            if (result.badElements.size() < m_data->limits.maxReportedIndices)
                result.badElements.push_back(exception);
        }
        const uint64_t value = freivaldsResidue(exact);
        rows[exception.row] =
            add(rows[exception.row],
                multiplyResidues(value, m_data->request.rightProbe[exception.column]));
        columns[exception.column] =
            add(columns[exception.column],
                multiplyResidues(value, m_data->request.leftProbe[exception.row]));
        ++result.recomputedElements;
    }
    for (size_t i = 0; i < rows.size(); ++i)
        if (rows[i] != m_data->rows[i]) {
            ++result.badRowCount;
            if (result.badRows.size() < m_data->limits.maxReportedIndices)
                result.badRows.push_back(i);
        }
    for (size_t j = 0; j < columns.size(); ++j)
        if (columns[j] != m_data->columns[j]) {
            ++result.badColumnCount;
            if (result.badColumns.size() < m_data->limits.maxReportedIndices)
                result.badColumns.push_back(j);
        }
    result.checkedElements = observation.checkedElements;
    result.status = result.badElementCount || result.badRowCount || result.badColumnCount
                        ? VerificationStatus::Mismatch
                        : VerificationStatus::Passed;
    return result;
}
}  // namespace roc::hostnumerics
