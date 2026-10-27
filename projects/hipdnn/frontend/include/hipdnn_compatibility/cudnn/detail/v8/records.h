// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_utils.h, include/cudnn_frontend_Operation.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// Immutable validated state behind one cuDNN descriptor, shared as
// std::shared_ptr<const T>. The kind tag replaces RTTI: consumers may build
// with RTTI disabled, so downcasts go through recordAs().
// The concrete constructors are public on purpose: core tests build records
// no factory would accept to reach lowering's rejection paths. Production code
// must create records through the validate.h factories; lowering relies on
// their invariants (required slots set, target matching kind, BN groups).
class Record
{
public:
    cudnnBackendDescriptorType_t kind() const noexcept
    {
        return _kind;
    }

protected:
    explicit Record(cudnnBackendDescriptorType_t kind) noexcept
        : _kind(kind)
    {
    }

    // Never deleted through a Record pointer: shared_ptr keeps the deleter of
    // the concrete type.
    ~Record() = default;
    Record(const Record&) = default;
    Record(Record&&) = default;
    Record& operator=(const Record&) = default;
    Record& operator=(Record&&) = default;

private:
    cudnnBackendDescriptorType_t _kind;
};

template <cudnnBackendDescriptorType_t K>
class RecordOf : public Record
{
public:
    static constexpr cudnnBackendDescriptorType_t KIND = K;

    static constexpr bool isKind(cudnnBackendDescriptorType_t kind) noexcept
    {
        return kind == K;
    }

protected:
    RecordOf() noexcept
        : Record(K)
    {
    }
};

template <typename T>
const T& recordAs(const Record& record)
{
    static_assert(std::is_base_of_v<Record, T>);
    if(!T::isKind(record.kind()))
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        "record of descriptor type "
                            + std::to_string(static_cast<int>(record.kind()))
                            + " does not have the expected type");
    }
    return static_cast<const T&>(record);
}

template <typename T>
std::shared_ptr<const T> recordAs(const std::shared_ptr<const Record>& record)
{
    if(!record)
    {
        return nullptr;
    }
    static_cast<void>(recordAs<T>(*record));
    return std::static_pointer_cast<const T>(record);
}

class TensorRecord final : public RecordOf<CUDNN_BACKEND_TENSOR_DESCRIPTOR>
{
public:
    struct Fields
    {
        DataType_t dataType = DataType_t::NOT_SET;
        std::vector<std::int64_t> dimensions;
        std::vector<std::int64_t> strides;
        std::int64_t uniqueId = -1;
        std::int64_t alignment = 0;
        bool isVirtual = false;
        bool isByValue = false;
        std::optional<std::vector<std::byte>> constantValue;
        std::shared_ptr<const TensorRecord> raggedOffset;
    };

    explicit TensorRecord(Fields fields)
        : _fields(std::move(fields))
    {
    }

    DataType_t dataType() const noexcept
    {
        return _fields.dataType;
    }

    const std::vector<std::int64_t>& dimensions() const noexcept
    {
        return _fields.dimensions;
    }

    const std::vector<std::int64_t>& strides() const noexcept
    {
        return _fields.strides;
    }

    std::int64_t uniqueId() const noexcept
    {
        return _fields.uniqueId;
    }

    std::int64_t alignment() const noexcept
    {
        return _fields.alignment;
    }

    bool isVirtual() const noexcept
    {
        return _fields.isVirtual;
    }

    bool isByValue() const noexcept
    {
        return _fields.isByValue;
    }

    // cuDNN's alignment is a lower bound and 16 is hipDNN's default, so only a
    // smaller value is forwarded.
    std::optional<std::int64_t> forwardedAlignment() const noexcept
    {
        return _fields.alignment < HIPDNN_DEFAULT_ALIGNMENT
                   ? std::optional<std::int64_t>(_fields.alignment)
                   : std::nullopt;
    }

    // IS_BY_VALUE without a constant value; a constant lowers to the tensor
    // value only.
    bool isRuntimePassByValue() const noexcept
    {
        return _fields.isByValue && !_fields.constantValue;
    }

    const std::optional<std::vector<std::byte>>& constantValue() const noexcept
    {
        return _fields.constantValue;
    }

    const std::shared_ptr<const TensorRecord>& raggedOffset() const noexcept
    {
        return _fields.raggedOffset;
    }

private:
    static constexpr std::int64_t HIPDNN_DEFAULT_ALIGNMENT = 16;

    Fields _fields;
};

using TensorPtr = std::shared_ptr<const TensorRecord>;

class ConvolutionRecord final : public RecordOf<CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR>
{
public:
    struct Fields
    {
        DataType_t computeType = DataType_t::NOT_SET;
        ConvolutionMode_t mode = ConvolutionMode_t::NOT_SET;
        std::int64_t spatialDims = 0;
        std::vector<std::int64_t> prePaddings;
        std::vector<std::int64_t> postPaddings;
        std::vector<std::int64_t> dilations;
        std::vector<std::int64_t> filterStrides;
    };

    explicit ConvolutionRecord(Fields fields)
        : _fields(std::move(fields))
    {
    }

    DataType_t computeType() const noexcept
    {
        return _fields.computeType;
    }

    ConvolutionMode_t mode() const noexcept
    {
        return _fields.mode;
    }

    std::int64_t spatialDims() const noexcept
    {
        return _fields.spatialDims;
    }

    const std::vector<std::int64_t>& prePaddings() const noexcept
    {
        return _fields.prePaddings;
    }

    const std::vector<std::int64_t>& postPaddings() const noexcept
    {
        return _fields.postPaddings;
    }

    const std::vector<std::int64_t>& dilations() const noexcept
    {
        return _fields.dilations;
    }

    const std::vector<std::int64_t>& filterStrides() const noexcept
    {
        return _fields.filterStrides;
    }

private:
    Fields _fields;
};

// The optional mode parameters hold only what lowering forwards: the values
// that differ from upstream's defaults (see makePointwiseRecord).
class PointwiseRecord final : public RecordOf<CUDNN_BACKEND_POINTWISE_DESCRIPTOR>
{
public:
    struct Fields
    {
        PointwiseMode_t mode = PointwiseMode_t::NOT_SET;
        DataType_t computeType = DataType_t::NOT_SET;
        std::optional<double> reluLowerClip;
        std::optional<double> reluUpperClip;
        std::optional<double> reluLowerClipSlope;
        std::optional<double> eluAlpha;
        std::optional<double> softplusBeta;
        std::optional<double> swishBeta;
        std::optional<std::int64_t> axis;
    };

    explicit PointwiseRecord(const Fields& fields) noexcept
        : _fields(fields)
    {
    }

    PointwiseMode_t mode() const noexcept
    {
        return _fields.mode;
    }

    DataType_t computeType() const noexcept
    {
        return _fields.computeType;
    }

    std::optional<double> reluLowerClip() const noexcept
    {
        return _fields.reluLowerClip;
    }

    std::optional<double> reluUpperClip() const noexcept
    {
        return _fields.reluUpperClip;
    }

    std::optional<double> reluLowerClipSlope() const noexcept
    {
        return _fields.reluLowerClipSlope;
    }

    std::optional<double> eluAlpha() const noexcept
    {
        return _fields.eluAlpha;
    }

    std::optional<double> softplusBeta() const noexcept
    {
        return _fields.softplusBeta;
    }

    std::optional<double> swishBeta() const noexcept
    {
        return _fields.swishBeta;
    }

    std::optional<std::int64_t> axis() const noexcept
    {
        return _fields.axis;
    }

private:
    Fields _fields;
};

class MatmulRecord final : public RecordOf<CUDNN_BACKEND_MATMUL_DESCRIPTOR>
{
public:
    explicit MatmulRecord(DataType_t computeType) noexcept
        : _computeType(computeType)
    {
    }

    DataType_t computeType() const noexcept
    {
        return _computeType;
    }

private:
    DataType_t _computeType;
};

class ReductionRecord final : public RecordOf<CUDNN_BACKEND_REDUCTION_DESCRIPTOR>
{
public:
    struct Fields
    {
        ReductionMode_t mode = ReductionMode_t::NOT_SET;
        DataType_t computeType = DataType_t::NOT_SET;
        bool isDeterministic = false;
    };

    explicit ReductionRecord(Fields fields) noexcept
        : _fields(fields)
    {
    }

    ReductionMode_t mode() const noexcept
    {
        return _fields.mode;
    }

    DataType_t computeType() const noexcept
    {
        return _fields.computeType;
    }

    bool isDeterministic() const noexcept
    {
        return _fields.isDeterministic;
    }

private:
    Fields _fields;
};

// The cuDNN fraction arrays are integral here (the factory rejects anything
// else).
class ResampleRecord final : public RecordOf<CUDNN_BACKEND_RESAMPLE_DESCRIPTOR>
{
public:
    struct Fields
    {
        ResampleMode_t mode = ResampleMode_t::NOT_SET;
        DataType_t computeType = DataType_t::NOT_SET;
        PaddingMode_t paddingMode = PaddingMode_t::NOT_SET;
        std::vector<std::int64_t> windowDims;
        std::vector<std::int64_t> prePaddings;
        std::vector<std::int64_t> postPaddings;
        std::vector<std::int64_t> strides;
    };

    explicit ResampleRecord(Fields fields)
        : _fields(std::move(fields))
    {
    }

    ResampleMode_t mode() const noexcept
    {
        return _fields.mode;
    }

    DataType_t computeType() const noexcept
    {
        return _fields.computeType;
    }

    PaddingMode_t paddingMode() const noexcept
    {
        return _fields.paddingMode;
    }

    const std::vector<std::int64_t>& windowDims() const noexcept
    {
        return _fields.windowDims;
    }

    const std::vector<std::int64_t>& prePaddings() const noexcept
    {
        return _fields.prePaddings;
    }

    const std::vector<std::int64_t>& postPaddings() const noexcept
    {
        return _fields.postPaddings;
    }

    const std::vector<std::int64_t>& strides() const noexcept
    {
        return _fields.strides;
    }

private:
    Fields _fields;
};

// The RNG operation is not lowered, so only the distribution is validated.
class RngRecord final : public RecordOf<CUDNN_BACKEND_RNG_DESCRIPTOR>
{
public:
    explicit RngRecord(RngDistribution_t distribution) noexcept
        : _distribution(distribution)
    {
    }

    RngDistribution_t distribution() const noexcept
    {
        return _distribution;
    }

private:
    RngDistribution_t _distribution;
};

// The hipDNN graph node an operation lowers to; named after the
// hipdnn_frontend attribute class it fills.
enum class OperationTarget : std::uint8_t
{
    NOT_SET,
    CONVOLUTION_FPROP,
    CONVOLUTION_DGRAD,
    CONVOLUTION_WGRAD,
    POINTWISE,
    MATMUL,
    REDUCTION,
    RESAMPLE_FWD,
    RESAMPLE_BWD,
    BATCHNORM,
    BATCHNORM_INFERENCE,
    BATCHNORM_BACKWARD,
    LAYERNORM,
    LAYERNORM_BACKWARD,
    RMSNORM,
    RMSNORM_BACKWARD
};

// Pointwise mode classes from upstream Operation_v8 (is_pointwise_*_op); the
// port count alone does not decide the class (EXP is activation-forward, ERF is
// math).
enum class PointwiseModeClass : std::uint8_t
{
    MATH,
    IDENTITY,
    ACTIVATION_FORWARD,
    ACTIVATION_BACKWARD
};

struct PointwiseModeTraits
{
    PointwiseModeClass modeClass;
    std::int64_t portCount; // upstream get_pointwise_mode_port_count
};

inline const char* pointwiseModeClassName(PointwiseModeClass modeClass) noexcept
{
    switch(modeClass)
    {
    case PointwiseModeClass::MATH:
        return "math";
    case PointwiseModeClass::IDENTITY:
        return "identity";
    case PointwiseModeClass::ACTIVATION_FORWARD:
        return "activation-forward";
    case PointwiseModeClass::ACTIVATION_BACKWARD:
        return "activation-backward";
    default:
        return "unknown";
    }
}

// Base of every lowerable operation record.
class OperationRecord : public Record
{
public:
    // Defined after the operation records.
    static constexpr bool isKind(cudnnBackendDescriptorType_t kind) noexcept;

    OperationTarget target() const noexcept
    {
        return _target;
    }

protected:
    OperationRecord(cudnnBackendDescriptorType_t kind, OperationTarget target) noexcept
        : Record(kind)
        , _target(target)
    {
    }

private:
    OperationTarget _target;
};

template <cudnnBackendDescriptorType_t K>
class OperationRecordOf : public OperationRecord
{
public:
    static constexpr cudnnBackendDescriptorType_t KIND = K;

    static constexpr bool isKind(cudnnBackendDescriptorType_t kind) noexcept
    {
        return kind == K;
    }

protected:
    explicit OperationRecordOf(OperationTarget target) noexcept
        : OperationRecord(K, target)
    {
    }
};

// One record for the three convolution operations. `tensors` holds inputs 0
// and 1, then the output: X, W -> Y (forward); DY, W -> DX (backward data);
// DY, X -> DW (backward filter).
template <cudnnBackendDescriptorType_t K, OperationTarget T>
class ConvolutionOperationRecordOf final : public OperationRecordOf<K>
{
public:
    struct Fields
    {
        std::shared_ptr<const ConvolutionRecord> convolution;
        std::array<TensorPtr, 3> tensors;
    };

    explicit ConvolutionOperationRecordOf(Fields fields)
        : OperationRecordOf<K>(T)
        , _fields(std::move(fields))
    {
    }

    const std::shared_ptr<const ConvolutionRecord>& convolution() const noexcept
    {
        return _fields.convolution;
    }

    const std::array<TensorPtr, 3>& tensors() const noexcept
    {
        return _fields.tensors;
    }

private:
    Fields _fields;
};

using ConvolutionForwardOperationRecord
    = ConvolutionOperationRecordOf<CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                   OperationTarget::CONVOLUTION_FPROP>;
using ConvolutionBackwardDataOperationRecord
    = ConvolutionOperationRecordOf<CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                   OperationTarget::CONVOLUTION_DGRAD>;
using ConvolutionBackwardFilterOperationRecord
    = ConvolutionOperationRecordOf<CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                   OperationTarget::CONVOLUTION_WGRAD>;

// Slot map: forward classes X -> IN_0, B -> IN_1, T -> IN_2, Y -> OUT_0;
// activation-backward DY -> IN_0, X -> IN_1, DX -> OUT_0.
class PointwiseOperationRecord final
    : public OperationRecordOf<CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR>
{
public:
    struct Fields
    {
        std::shared_ptr<const PointwiseRecord> pointwise;
        PointwiseModeTraits traits{PointwiseModeClass::MATH, 0};
        TensorPtr in0;
        TensorPtr in1;
        TensorPtr in2;
        TensorPtr out0;
    };

    explicit PointwiseOperationRecord(Fields fields)
        : OperationRecordOf(OperationTarget::POINTWISE)
        , _fields(std::move(fields))
    {
    }

    const std::shared_ptr<const PointwiseRecord>& pointwise() const noexcept
    {
        return _fields.pointwise;
    }

    PointwiseModeClass modeClass() const noexcept
    {
        return _fields.traits.modeClass;
    }

    std::int64_t portCount() const noexcept
    {
        return _fields.traits.portCount;
    }

    const TensorPtr& in0() const noexcept
    {
        return _fields.in0;
    }

    // Null for unary modes.
    const TensorPtr& in1() const noexcept
    {
        return _fields.in1;
    }

    // Null unless the mode takes three inputs (BINARY_SELECT).
    const TensorPtr& in2() const noexcept
    {
        return _fields.in2;
    }

    const TensorPtr& out0() const noexcept
    {
        return _fields.out0;
    }

private:
    Fields _fields;
};

// A, B -> C.
class MatmulOperationRecord final
    : public OperationRecordOf<CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR>
{
public:
    struct Fields
    {
        std::shared_ptr<const MatmulRecord> matmul;
        TensorPtr a;
        TensorPtr b;
        TensorPtr c;
    };

    explicit MatmulOperationRecord(Fields fields)
        : OperationRecordOf(OperationTarget::MATMUL)
        , _fields(std::move(fields))
    {
    }

    const std::shared_ptr<const MatmulRecord>& matmul() const noexcept
    {
        return _fields.matmul;
    }

    const TensorPtr& a() const noexcept
    {
        return _fields.a;
    }

    const TensorPtr& b() const noexcept
    {
        return _fields.b;
    }

    const TensorPtr& c() const noexcept
    {
        return _fields.c;
    }

private:
    Fields _fields;
};

// X -> Y.
class ReductionOperationRecord final
    : public OperationRecordOf<CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR>
{
public:
    struct Fields
    {
        std::shared_ptr<const ReductionRecord> reduction;
        TensorPtr x;
        TensorPtr y;
    };

    explicit ReductionOperationRecord(Fields fields)
        : OperationRecordOf(OperationTarget::REDUCTION)
        , _fields(std::move(fields))
    {
    }

    const std::shared_ptr<const ReductionRecord>& reduction() const noexcept
    {
        return _fields.reduction;
    }

    const TensorPtr& x() const noexcept
    {
        return _fields.x;
    }

    const TensorPtr& y() const noexcept
    {
        return _fields.y;
    }

private:
    Fields _fields;
};

// X -> Y, plus the optional INDEX output (generate_index follows its presence).
class ResampleForwardOperationRecord final
    : public OperationRecordOf<CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR>
{
public:
    struct Fields
    {
        std::shared_ptr<const ResampleRecord> resample;
        TensorPtr x;
        TensorPtr y;
        TensorPtr index;
    };

    explicit ResampleForwardOperationRecord(Fields fields)
        : OperationRecordOf(OperationTarget::RESAMPLE_FWD)
        , _fields(std::move(fields))
    {
    }

    const std::shared_ptr<const ResampleRecord>& resample() const noexcept
    {
        return _fields.resample;
    }

    const TensorPtr& x() const noexcept
    {
        return _fields.x;
    }

    const TensorPtr& y() const noexcept
    {
        return _fields.y;
    }

    // Null when unset.
    const TensorPtr& index() const noexcept
    {
        return _fields.index;
    }

private:
    Fields _fields;
};

// DY, INDEX -> DX.
class ResampleBackwardOperationRecord final
    : public OperationRecordOf<CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR>
{
public:
    struct Fields
    {
        std::shared_ptr<const ResampleRecord> resample;
        TensorPtr dx;
        TensorPtr dy;
        TensorPtr index;
    };

    explicit ResampleBackwardOperationRecord(Fields fields)
        : OperationRecordOf(OperationTarget::RESAMPLE_BWD)
        , _fields(std::move(fields))
    {
    }

    const std::shared_ptr<const ResampleRecord>& resample() const noexcept
    {
        return _fields.resample;
    }

    const TensorPtr& dx() const noexcept
    {
        return _fields.dx;
    }

    const TensorPtr& dy() const noexcept
    {
        return _fields.dy;
    }

    // Null when unset; always set for MAXPOOL.
    const TensorPtr& index() const noexcept
    {
        return _fields.index;
    }

private:
    Fields _fields;
};

// Lowers, by target(), to BatchnormAttributes (BATCH + TRAINING),
// BatchnormInferenceAttributes (BATCH + INFERENCE), LayernormAttributes or
// RMSNormAttributes. Optional tensors are null when unset or not lowered.
class NormForwardOperationRecord final
    : public OperationRecordOf<CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR>
{
public:
    struct Fields
    {
        NormFwdPhase_t phase = NormFwdPhase_t::NOT_SET;
        std::int64_t normalizedDimCount = 0;
        TensorPtr x;
        TensorPtr scale;
        TensorPtr bias;
        TensorPtr epsilon;
        TensorPtr y;
        TensorPtr mean;
        TensorPtr invVariance;
        TensorPtr inputRunningMean;
        TensorPtr inputRunningVariance;
        TensorPtr expAvgFactor;
        TensorPtr outputRunningMean;
        TensorPtr outputRunningVariance;
    };

    NormForwardOperationRecord(OperationTarget target, Fields fields)
        : OperationRecordOf(target)
        , _fields(std::move(fields))
    {
    }

    NormFwdPhase_t phase() const noexcept
    {
        return _fields.phase;
    }

    // cuDNN norm operations carry no compute type; every hipDNN norm node needs
    // one.
    static constexpr DataType_t computeType() noexcept
    {
        return DataType_t::FLOAT;
    }

    // LAYERNORM only (0 otherwise); hipDNN has no cuDNN counterpart attribute.
    std::int64_t normalizedDimCount() const noexcept
    {
        return _fields.normalizedDimCount;
    }

    const TensorPtr& x() const noexcept
    {
        return _fields.x;
    }

    const TensorPtr& scale() const noexcept
    {
        return _fields.scale;
    }

    // Optional for RMSNORM.
    const TensorPtr& bias() const noexcept
    {
        return _fields.bias;
    }

    // Null for BATCHNORM_INFERENCE: hipDNN's batchnorm inference has no epsilon.
    const TensorPtr& epsilon() const noexcept
    {
        return _fields.epsilon;
    }

    const TensorPtr& y() const noexcept
    {
        return _fields.y;
    }

    // Outputs except for BATCHNORM_INFERENCE, where they are inputs. RMSNORM
    // never has a mean; its inverse variance lowers to INV_RMS.
    const TensorPtr& mean() const noexcept
    {
        return _fields.mean;
    }

    const TensorPtr& invVariance() const noexcept
    {
        return _fields.invVariance;
    }

    // BATCHNORM only, all or none. The exponential average factor lowers to
    // MOMENTUM.
    const TensorPtr& inputRunningMean() const noexcept
    {
        return _fields.inputRunningMean;
    }

    const TensorPtr& inputRunningVariance() const noexcept
    {
        return _fields.inputRunningVariance;
    }

    const TensorPtr& expAvgFactor() const noexcept
    {
        return _fields.expAvgFactor;
    }

    const TensorPtr& outputRunningMean() const noexcept
    {
        return _fields.outputRunningMean;
    }

    const TensorPtr& outputRunningVariance() const noexcept
    {
        return _fields.outputRunningVariance;
    }

private:
    Fields _fields;
};

// Lowers, by target(), to BatchnormBackwardAttributes,
// LayernormBackwardAttributes or RMSNormBackwardAttributes. Optional tensors
// are null when unset or not lowered.
class NormBackwardOperationRecord final
    : public OperationRecordOf<CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR>
{
public:
    struct Fields
    {
        std::int64_t normalizedDimCount = 0;
        TensorPtr x;
        TensorPtr dy;
        TensorPtr scale;
        TensorPtr mean;
        TensorPtr invVariance;
        TensorPtr epsilon;
        TensorPtr dx;
        TensorPtr dscale;
        TensorPtr dbias;
    };

    NormBackwardOperationRecord(OperationTarget target, Fields fields)
        : OperationRecordOf(target)
        , _fields(std::move(fields))
    {
    }

    static constexpr DataType_t computeType() noexcept
    {
        return DataType_t::FLOAT;
    }

    // LAYERNORM_BACKWARD only (0 otherwise).
    std::int64_t normalizedDimCount() const noexcept
    {
        return _fields.normalizedDimCount;
    }

    const TensorPtr& x() const noexcept
    {
        return _fields.x;
    }

    const TensorPtr& dy() const noexcept
    {
        return _fields.dy;
    }

    const TensorPtr& scale() const noexcept
    {
        return _fields.scale;
    }

    // Never set for RMSNORM_BACKWARD.
    const TensorPtr& mean() const noexcept
    {
        return _fields.mean;
    }

    // Lowers to INV_RMS for RMSNORM_BACKWARD.
    const TensorPtr& invVariance() const noexcept
    {
        return _fields.invVariance;
    }

    // LAYERNORM_BACKWARD only: hipDNN's batchnorm backward has no epsilon.
    const TensorPtr& epsilon() const noexcept
    {
        return _fields.epsilon;
    }

    const TensorPtr& dx() const noexcept
    {
        return _fields.dx;
    }

    const TensorPtr& dscale() const noexcept
    {
        return _fields.dscale;
    }

    // Optional for RMSNORM_BACKWARD (compute_dbias follows its presence).
    const TensorPtr& dbias() const noexcept
    {
        return _fields.dbias;
    }

private:
    Fields _fields;
};

template <typename... Records>
constexpr bool isAnyKind(cudnnBackendDescriptorType_t kind) noexcept
{
    return ((kind == Records::KIND) || ...);
}

constexpr bool OperationRecord::isKind(cudnnBackendDescriptorType_t kind) noexcept
{
    return isAnyKind<ConvolutionForwardOperationRecord,
                     ConvolutionBackwardDataOperationRecord,
                     ConvolutionBackwardFilterOperationRecord,
                     PointwiseOperationRecord,
                     MatmulOperationRecord,
                     ReductionOperationRecord,
                     ResampleForwardOperationRecord,
                     ResampleBackwardOperationRecord,
                     NormForwardOperationRecord,
                     NormBackwardOperationRecord>(kind);
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
