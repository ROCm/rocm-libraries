// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <roc/hostnumerics/epilogue.hpp>
#include <roc/hostnumerics/gemm.hpp>
#include <roc/hostnumerics/verification.hpp>
#include <string>
#include <variant>
#include <vector>

namespace roc::hostnumerics {
inline constexpr uint64_t freivaldsModulus = (uint64_t{1} << 61) - 1;

enum class FreivaldsProbeAlgorithm { SplitMix64Rejection, SplitMix64Legacy, Count };
struct FreivaldsProbeOptions {
    uint64_t seed = 0;
    FreivaldsProbeAlgorithm algorithm = FreivaldsProbeAlgorithm::SplitMix64Rejection;
};

// Integer-only arithmetic and reproducible protocol vectors. These functions
// have no HIP dependency. Multiply/add require residues less than the modulus.
uint64_t freivaldsProbe(uint64_t seed, uint64_t index, FreivaldsProbeAlgorithm algorithm);
uint64_t freivaldsResidue(int64_t value);
uint64_t freivaldsAdd(uint64_t a, uint64_t b);
uint64_t freivaldsMultiply(uint64_t a, uint64_t b);

// Sum canonical modular partials [chunk, coordinate] over chunks. Input may
// have any valid layout; the result is a contiguous rank-one UInt64 Tensor.
// An empty chunk dimension produces zeros. Rejects noncanonical residues.
Tensor reduceFreivaldsProjections(const Tensor& partials);

struct FreivaldsLimits {
    size_t maxPreparationBytes = size_t{8} * 1024 * 1024 * 1024;
    // Observation backing storage, logical working copies, exception coordinates and
    // bounded diagnostics. Does not include retained plan inputs or device work.
    size_t maxObservationBytes = size_t{512} * 1024 * 1024;
    uint64_t maxPreparationMultiplyAdds = uint64_t{1} << 36;
    uint64_t maxRecomputeMultiplyAdds = uint64_t{1} << 34;
    size_t maxReportedIndices = 64;
};

struct FreivaldsStatistics {
    double maximumPartialMagnitude = 0;
    double maximumResultMagnitude = 0;
    size_t retainedInputBytes = 0;
    uint64_t preparationMultiplyAdds = 0;
};

struct FreivaldsPreparationFailure {
    VerificationStatus status;
    std::string reason;
    FreivaldsStatistics statistics;
};

// A product supplies layout, device pointers and execution ordering separately.
// Projections omit exceptional elements from BOTH directions. Each omitted
// coordinate appears exactly once in exceptions. All other logical elements
// contribute once to each applicable sum. Probes and residues are integers.
struct FreivaldsRequest {
    uint64_t identity = 0;
    Shape shape;
    ScalarType outputType = ScalarType::Float32;
    double fastMagnitudeLimit = 0;
    size_t maximumExceptions = 0;
    Tensor rightProbe{ScalarType::UInt64, Shape{0}};  // [N], for D*r
    Tensor leftProbe{ScalarType::UInt64, Shape{0}};   // [M], for t^T*D
};
struct FreivaldsException {
    size_t row;
    size_t column;
    double storedValue;
};
struct FreivaldsObservation {
    uint64_t requestIdentity = 0;
    size_t checkedElements = 0;
    bool complete = true;
    // Rank-one UInt64 tensors, one canonical residue per row/column. Layouts
    // may be strided. Plans with no output elements use empty projections.
    Tensor rowSums{ScalarType::UInt64, Shape{0}};
    Tensor columnSums{ScalarType::UInt64, Shape{0}};
    std::vector<FreivaldsException> exceptions;
};
struct FreivaldsCheck {
    VerificationStatus status = VerificationStatus::Incomplete;
    std::string reason;
    size_t checkedElements = 0;
    size_t recomputedElements = 0;
    size_t badRowCount = 0;
    size_t badColumnCount = 0;
    size_t badElementCount = 0;
    std::vector<size_t> badRows;
    std::vector<size_t> badColumns;
    std::vector<FreivaldsException> badElements;
};

struct FreivaldsCheckOptions {
    // Absolute error in the computed result before its output conversion.
    // Zero retains exact equality. A bound must be finite, nonnegative and
    // strictly below 0.5 so an integer fast-path error cannot be hidden.
    // Nonzero bounds currently require F32 computation.
    double maximumAbsoluteComputationError = 0;
};

class FreivaldsPlan;
using FreivaldsPreparation = std::variant<FreivaldsPlan, FreivaldsPreparationFailure>;

class FreivaldsPlan {
   public:
    // Explicit detached exports: mutating these shallow-const Tensor handles
    // cannot change the plan or any other export. Probes are contiguous UInt64.
    FreivaldsRequest copyRequest() const;
    uint64_t requestIdentity() const;
    const FreivaldsStatistics& statistics() const;
    Tensor copyExpectedRowSums() const;
    Tensor copyExpectedColumnSums() const;

    // CPU oracle and device-observation path use the same numerical check.
    // A caller checking several batches may further restrict exception work
    // to the remainder of a shared budget. This never raises the plan's limit.
    FreivaldsCheck check(const Tensor& observed,
                         size_t exceptionLimit = std::numeric_limits<size_t>::max(),
                         const FreivaldsCheckOptions& options = {}) const;
    FreivaldsCheck check(const FreivaldsObservation& observation,
                         size_t exceptionLimit = std::numeric_limits<size_t>::max(),
                         const FreivaldsCheckOptions& options = {}) const;

    // Independently useful for bounded diagnostics/exception lists. Coordinates
    // are checked. The plan owns compact immutable inputs used by these calls.
    int64_t exactValue(size_t row, size_t column) const;
    double roundedValue(size_t row, size_t column) const;

   private:
    struct Data;
    explicit FreivaldsPlan(std::shared_ptr<const Data> data);
    std::shared_ptr<const Data> m_data;
    friend FreivaldsPreparation prepareFreivalds(const Tensor&, const Tensor&, const MatmulOptions&,
                                                 const EpilogueOptions&, ScalarType,
                                                 const FreivaldsProbeOptions&,
                                                 const FreivaldsLimits&);
};

// One logical rank-two matmul. Requires exact integral effective inputs and
// intermediates, including each ordered pre-quantization operand scale and the
// compute-input conversions in MatmulOptions. Scales may broadcast along any
// operand axis; conversions must preserve the effective values exactly.
// Supports linear epilogues with scalar/row/column input factors,
// scalar output scale, C and broadcast bias; output conversion may round/saturate.
// Nonlinear/auxiliary checks are separate operations. No special input pattern
// or generation recipe is required. Copies logical inputs into private storage.
FreivaldsPreparation prepareFreivalds(const Tensor& a, const Tensor& b,
                                      const MatmulOptions& operation = MatmulOptions{},
                                      const EpilogueOptions& epilogue = EpilogueOptions{},
                                      ScalarType outputType = ScalarType::Float32,
                                      const FreivaldsProbeOptions& probes = {},
                                      const FreivaldsLimits& limits = {});
}  // namespace roc::hostnumerics
