// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <roc/hostnumerics/comparison.hpp>
#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/gemm.hpp>
#include <roc/hostnumerics/periodic_initialization.hpp>
#include <roc/hostnumerics/verification.hpp>
#include <variant>

namespace roc::hostnumerics {
struct PeriodicMatmulSegment {
    size_t reductionBegin;
    PeriodicInitialization a;  // Logical [M, segment K], repeats only along M.
    PeriodicInitialization b;  // Logical [segment K, N], repeats only along N.
};

enum class PeriodicMatmulMode { ExactInteger, FloatingPoint, Count };

struct PeriodicMatmulLimits {
    size_t maxReferenceMultiplyAdds = 1'000'000'000;
    size_t maxPreparationBytes = size_t{64} * 1024 * 1024;
    size_t maxCheckBytes = size_t{8} * 1024 * 1024;
};

struct PeriodicMatmulStatistics {
    size_t referenceMultiplyAdds = 0;
    size_t seedBytes = 0;
    size_t productBytes = 0;
    size_t preparationBytesBound = 0;
    size_t combinedRowPeriod = 1;
    size_t combinedColumnPeriod = 1;
    double absoluteProductSumBound = 0;
};

struct PeriodicMatmulCheck {
    VerificationStatus status = VerificationStatus::Incomplete;
    std::string reason;
    ComparisonReport comparison;
    double absoluteTolerance = 0;
};

class PeriodicMatmulPlan;
using PeriodicMatmulPreparation = std::variant<PeriodicMatmulPlan, VerificationFailure>;

// Reference for real F32 accumulation, using effective F32 operands decoded
// from the initializations' stored encodings and F64 small products. Unsupported
// transforms/rounding modes are reported before computing those products.
// Plans own immutable descriptions and products; preparation is reusable.
class PeriodicMatmulPlan {
   public:
    const Shape& shape() const;
    size_t reductionLength() const;
    const PeriodicMatmulStatistics& statistics() const;

    // Produces raw F64 A*B for a bounded region in global logical coordinates.
    Tensor productRegion(std::span<const size_t> origin, Shape extent) const;

    // Compares every F32 observed element. Epilogue tensors describe this region,
    // not the full output. Initially accepts alpha*A*B + beta*C + bias, with
    // F32 epilogue arithmetic, default output conversion, and no activation.
    // The exact mode certifies integral effective inputs/intermediates and uses
    // zero tolerance. FloatingPoint uses the documented forward-error bound.
    // Mismatch coordinates/indices are rebased to the full output shape.
    PeriodicMatmulCheck checkRegion(const Tensor& observed, std::span<const size_t> origin,
                                    const EpilogueOptions& epilogue = EpilogueOptions{}) const;

   private:
    struct Data;
    explicit PeriodicMatmulPlan(std::shared_ptr<const Data> data);
    std::shared_ptr<const Data> m_data;
    friend PeriodicMatmulPreparation preparePeriodicMatmul(Shape, size_t,
                                                           std::span<const PeriodicMatmulSegment>,
                                                           const MatmulOptions&, PeriodicMatmulMode,
                                                           const PeriodicMatmulLimits&);
};

// Segments must partition [0, reductionLength) in order. Empty K may use an
// empty segment list. Each segment's full M/N must match outputShape. The
// verifier uses the same definitions that the caller used to initialize A/B;
// it does not infer periodicity from a few samples of arbitrary arrays.
PeriodicMatmulPreparation preparePeriodicMatmul(
    Shape outputShape, size_t reductionLength, std::span<const PeriodicMatmulSegment> segments,
    const MatmulOptions& operation = MatmulOptions{},
    PeriodicMatmulMode mode = PeriodicMatmulMode::FloatingPoint,
    const PeriodicMatmulLimits& limits = {});
}  // namespace roc::hostnumerics
