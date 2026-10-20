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
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/parameter_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/tensor_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_compatibility/cudnn/detail/v8/validate.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

using v8::OperationTarget;
using v8::PointwiseModeClass;

// Common base of every CUDNN_BACKEND_OPERATION_* class. Finalize reads the
// stored attributes into the v8 factory inputs; the factory owns the rules.
// Tensor slot accessors return the referenced descriptor of the record's slot.
class OperationDescriptor : public Descriptor
{
public:
    OperationTarget target() const noexcept
    {
        return _record ? _record->target() : OperationTarget::NOT_SET;
    }

    // Null until finalized.
    const std::shared_ptr<const v8::OperationRecord>& operationRecord() const noexcept
    {
        return _record;
    }

protected:
    using Descriptor::Descriptor;

    template <typename T>
    std::shared_ptr<T> referent(cudnnBackendAttributeName_t name,
                                cudnnBackendDescriptorType_t referentType) const
    {
        std::shared_ptr<Descriptor> value = descriptorAttribute(name);
        if(!value)
        {
            return nullptr;
        }
        descriptorCast<T>(*value, referentType);
        return std::static_pointer_cast<T>(std::move(value));
    }

    std::shared_ptr<TensorDescriptor> tensor(cudnnBackendAttributeName_t name) const
    {
        return referent<TensorDescriptor>(name, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    }

    template <typename T>
    static auto recordOf(const std::shared_ptr<T>& descriptor)
        -> std::remove_cv_t<std::remove_reference_t<decltype(descriptor->record())>>
    {
        return descriptor ? descriptor->record() : nullptr;
    }

    // Set with at least one element; descriptor-valued scalars always hold one.
    bool isSet(cudnnBackendAttributeName_t name) const
    {
        const StoredAttribute* value = findAttribute(name);
        return value != nullptr && value->count > 0;
    }

    void commit(std::shared_ptr<const v8::OperationRecord> record,
                std::initializer_list<std::shared_ptr<TensorDescriptor>> tensors)
    {
        _tensors.clear();
        for(const std::shared_ptr<TensorDescriptor>& tensor : tensors)
        {
            if(tensor)
            {
                _tensors.push_back(tensor);
            }
        }
        _record = std::move(record);
    }

    template <typename RecordT>
    const RecordT& typedRecord() const
    {
        return v8::recordAs<RecordT>(*_record);
    }

    std::shared_ptr<TensorDescriptor> descriptorOf(const v8::TensorPtr& tensor) const
    {
        if(tensor)
        {
            for(const std::shared_ptr<TensorDescriptor>& candidate : _tensors)
            {
                if(candidate->record() == tensor)
                {
                    return candidate;
                }
            }
        }
        return nullptr;
    }

private:
    std::shared_ptr<const v8::OperationRecord> _record;
    std::vector<std::shared_ptr<TensorDescriptor>> _tensors;
};

// Tag-checked downcast to the operation base; every operation descriptor type
// maps to a subclass of OperationDescriptor.
inline OperationDescriptor& operationCast(Descriptor& descriptor)
{
    if(!isOperationDescriptorType(descriptor.type()))
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string("expected an operation descriptor, got ")
                            + descriptorTypeName(descriptor.type()));
    }
    return static_cast<OperationDescriptor&>(descriptor);
}

// Operation types the shim cannot lower. Finalize fails before looking at any
// attribute, so legacy samples skip instead of failing on a missing one.
class UnsupportedOperation final : public OperationDescriptor
{
public:
    using OperationDescriptor::OperationDescriptor;

protected:
    void finalizeImpl() override
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        std::string(descriptorTypeName(type()))
                            + " is not supported by the hipDNN cuDNN shim");
    }
};

// The three convolution operations differ only in their attribute names, the
// record they produce and which tensor is the output. `tensors` is in the
// record's slot order.
class ConvolutionOperationBase : public OperationDescriptor
{
public:
    const std::shared_ptr<ConvolutionDescriptor>& convolution() const noexcept
    {
        return _convolution;
    }

protected:
    struct AttributeNames
    {
        cudnnBackendAttributeName_t convolution;
        cudnnBackendAttributeName_t alpha;
        cudnnBackendAttributeName_t beta;
        std::array<cudnnBackendAttributeName_t, 3> tensors;
    };

    using MakeRecord
        = std::shared_ptr<const v8::OperationRecord> (*)(const v8::ConvolutionOperationInputs&,
                                                         const v8::ConvolutionOperationLabels&);

    ConvolutionOperationBase(cudnnBackendDescriptorType_t type,
                             const AttributeNames& names,
                             MakeRecord makeRecord)
        : OperationDescriptor(type)
        , _names(names)
        , _labels{attributeLabel(names.convolution),
                  attributeLabel(names.alpha),
                  attributeLabel(names.beta),
                  {attributeLabel(names.tensors[0]),
                   attributeLabel(names.tensors[1]),
                   attributeLabel(names.tensors[2])},
                  attributeLabel(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS)}
        , _makeRecord(makeRecord)
    {
    }

    void finalizeImpl() override
    {
        std::shared_ptr<ConvolutionDescriptor> convolution = referent<ConvolutionDescriptor>(
            _names.convolution, CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        const std::array<std::shared_ptr<TensorDescriptor>, 3> tensors{
            tensor(_names.tensors[0]), tensor(_names.tensors[1]), tensor(_names.tensors[2])};
        std::shared_ptr<const v8::OperationRecord> record
            = _makeRecord({recordOf(convolution),
                           {recordOf(tensors[0]), recordOf(tensors[1]), recordOf(tensors[2])},
                           realAttribute(_names.alpha),
                           realAttribute(_names.beta)},
                          _labels);
        _convolution = std::move(convolution);
        commit(std::move(record), {tensors[0], tensors[1], tensors[2]});
    }

private:
    AttributeNames _names;
    v8::ConvolutionOperationLabels _labels;
    MakeRecord _makeRecord;
    std::shared_ptr<ConvolutionDescriptor> _convolution;
};

template <typename RecordT>
class ConvolutionOperationOf : public ConvolutionOperationBase
{
protected:
    ConvolutionOperationOf(cudnnBackendDescriptorType_t type, const AttributeNames& names)
        : ConvolutionOperationBase(type, names, &makeRecord)
    {
    }

    const RecordT& record() const
    {
        return typedRecord<RecordT>();
    }

private:
    static std::shared_ptr<const v8::OperationRecord>
        makeRecord(const v8::ConvolutionOperationInputs& inputs,
                   const v8::ConvolutionOperationLabels& labels)
    {
        return v8::makeOperationRecord<RecordT>(inputs, labels);
    }
};

// Lowers to ConvFpropAttributes: X, W -> Y.
class ConvolutionForwardOperation final
    : public ConvolutionOperationOf<v8::ConvolutionForwardOperationRecord>
{
public:
    ConvolutionForwardOperation()
        : ConvolutionOperationOf(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                 {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                  CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                  CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                                  {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                   CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                   CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y}})
    {
    }

    std::shared_ptr<TensorDescriptor> x() const
    {
        return descriptorOf(record().x());
    }

    std::shared_ptr<TensorDescriptor> w() const
    {
        return descriptorOf(record().w());
    }

    std::shared_ptr<TensorDescriptor> y() const
    {
        return descriptorOf(record().y());
    }
};

// Lowers to ConvDgradAttributes: DY, W -> DX.
class ConvolutionBackwardDataOperation final
    : public ConvolutionOperationOf<v8::ConvolutionBackwardDataOperationRecord>
{
public:
    ConvolutionBackwardDataOperation()
        : ConvolutionOperationOf(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                 {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                                  {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                   CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                   CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX}})
    {
    }

    std::shared_ptr<TensorDescriptor> dy() const
    {
        return descriptorOf(record().dy());
    }

    std::shared_ptr<TensorDescriptor> w() const
    {
        return descriptorOf(record().w());
    }

    std::shared_ptr<TensorDescriptor> dx() const
    {
        return descriptorOf(record().dx());
    }
};

// Lowers to ConvWgradAttributes: DY, X -> DW.
class ConvolutionBackwardFilterOperation final
    : public ConvolutionOperationOf<v8::ConvolutionBackwardFilterOperationRecord>
{
public:
    ConvolutionBackwardFilterOperation()
        : ConvolutionOperationOf(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                 {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                                  CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                                  {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                   CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                   CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW}})
    {
    }

    std::shared_ptr<TensorDescriptor> dy() const
    {
        return descriptorOf(record().dy());
    }

    std::shared_ptr<TensorDescriptor> x() const
    {
        return descriptorOf(record().x());
    }

    std::shared_ptr<TensorDescriptor> dw() const
    {
        return descriptorOf(record().dw());
    }
};

// Lowers to PointwiseAttributes; the slot map lives in the record.
class PointwiseOperation final : public OperationDescriptor
{
public:
    PointwiseOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    {
    }

    const std::shared_ptr<PointwiseDescriptor>& pointwise() const noexcept
    {
        return _pointwise;
    }

    PointwiseModeClass modeClass() const
    {
        return record().modeClass();
    }

    std::int64_t portCount() const
    {
        return record().portCount();
    }

    std::shared_ptr<TensorDescriptor> in0() const
    {
        return descriptorOf(record().in0());
    }

    std::shared_ptr<TensorDescriptor> in1() const
    {
        return descriptorOf(record().in1());
    }

    std::shared_ptr<TensorDescriptor> in2() const
    {
        return descriptorOf(record().in2());
    }

    std::shared_ptr<TensorDescriptor> out0() const
    {
        return descriptorOf(record().out0());
    }

protected:
    void finalizeImpl() override
    {
        static const v8::PointwiseOperationLabels s_labels{
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_XDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_BDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_TDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_YDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1),
            attributeLabel(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2),
        };
        std::shared_ptr<PointwiseDescriptor> pointwise = referent<PointwiseDescriptor>(
            CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        const std::shared_ptr<TensorDescriptor> x = tensor(CUDNN_ATTR_OPERATION_POINTWISE_XDESC);
        const std::shared_ptr<TensorDescriptor> b = tensor(CUDNN_ATTR_OPERATION_POINTWISE_BDESC);
        const std::shared_ptr<TensorDescriptor> t = tensor(CUDNN_ATTR_OPERATION_POINTWISE_TDESC);
        const std::shared_ptr<TensorDescriptor> y = tensor(CUDNN_ATTR_OPERATION_POINTWISE_YDESC);
        const std::shared_ptr<TensorDescriptor> dy = tensor(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC);
        const std::shared_ptr<TensorDescriptor> dx = tensor(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC);
        std::shared_ptr<const v8::PointwiseOperationRecord> record = v8::makeOperationRecord(
            v8::PointwiseOperationInputs{recordOf(pointwise),
                                         recordOf(x),
                                         recordOf(b),
                                         recordOf(t),
                                         recordOf(y),
                                         recordOf(dy),
                                         recordOf(dx),
                                         realAttribute(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1),
                                         realAttribute(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2)},
            s_labels);
        _pointwise = std::move(pointwise);
        commit(std::move(record), {x, b, t, y, dy, dx});
    }

private:
    const v8::PointwiseOperationRecord& record() const
    {
        return typedRecord<v8::PointwiseOperationRecord>();
    }

    std::shared_ptr<PointwiseDescriptor> _pointwise;
};

// Lowers to MatmulAttributes: A, B -> C.
class MatmulOperation final : public OperationDescriptor
{
public:
    MatmulOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
    {
    }

    const std::shared_ptr<MatmulDescriptor>& matmul() const noexcept
    {
        return _matmul;
    }

    std::shared_ptr<TensorDescriptor> a() const
    {
        return descriptorOf(record().a());
    }

    std::shared_ptr<TensorDescriptor> b() const
    {
        return descriptorOf(record().b());
    }

    std::shared_ptr<TensorDescriptor> c() const
    {
        return descriptorOf(record().c());
    }

protected:
    void finalizeImpl() override
    {
        static const v8::MatmulOperationLabels s_labels{
            attributeLabel(CUDNN_ATTR_OPERATION_MATMUL_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_MATMUL_ADESC),
            attributeLabel(CUDNN_ATTR_OPERATION_MATMUL_BDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_MATMUL_CDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC),
        };
        std::shared_ptr<MatmulDescriptor> matmul = referent<MatmulDescriptor>(
            CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        const std::shared_ptr<TensorDescriptor> a = tensor(CUDNN_ATTR_OPERATION_MATMUL_ADESC);
        const std::shared_ptr<TensorDescriptor> b = tensor(CUDNN_ATTR_OPERATION_MATMUL_BDESC);
        const std::shared_ptr<TensorDescriptor> c = tensor(CUDNN_ATTR_OPERATION_MATMUL_CDESC);
        std::shared_ptr<const v8::MatmulOperationRecord> record = v8::makeOperationRecord(
            v8::MatmulOperationInputs{recordOf(matmul),
                                      recordOf(a),
                                      recordOf(b),
                                      recordOf(c),
                                      isSet(CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC),
                                      isSet(CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC),
                                      isSet(CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC)},
            s_labels);
        _matmul = std::move(matmul);
        commit(std::move(record), {a, b, c});
    }

private:
    const v8::MatmulOperationRecord& record() const
    {
        return typedRecord<v8::MatmulOperationRecord>();
    }

    std::shared_ptr<MatmulDescriptor> _matmul;
};

// Lowers to ReductionAttributes: X -> Y.
class ReductionOperation final : public OperationDescriptor
{
public:
    ReductionOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
    {
    }

    const std::shared_ptr<ReductionDescriptor>& reduction() const noexcept
    {
        return _reduction;
    }

    std::shared_ptr<TensorDescriptor> x() const
    {
        return descriptorOf(record().x());
    }

    std::shared_ptr<TensorDescriptor> y() const
    {
        return descriptorOf(record().y());
    }

protected:
    void finalizeImpl() override
    {
        static const v8::ReductionOperationLabels s_labels{
            attributeLabel(CUDNN_ATTR_OPERATION_REDUCTION_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_REDUCTION_XDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_REDUCTION_YDESC),
        };
        std::shared_ptr<ReductionDescriptor> reduction = referent<ReductionDescriptor>(
            CUDNN_ATTR_OPERATION_REDUCTION_DESC, CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        const std::shared_ptr<TensorDescriptor> x = tensor(CUDNN_ATTR_OPERATION_REDUCTION_XDESC);
        const std::shared_ptr<TensorDescriptor> y = tensor(CUDNN_ATTR_OPERATION_REDUCTION_YDESC);
        std::shared_ptr<const v8::ReductionOperationRecord> record = v8::makeOperationRecord(
            v8::ReductionOperationInputs{recordOf(reduction), recordOf(x), recordOf(y)}, s_labels);
        _reduction = std::move(reduction);
        commit(std::move(record), {x, y});
    }

private:
    const v8::ReductionOperationRecord& record() const
    {
        return typedRecord<v8::ReductionOperationRecord>();
    }

    std::shared_ptr<ReductionDescriptor> _reduction;
};

// Lowers to ResampleFwdAttributes: X -> Y, plus the optional INDEX output
// (generate_index follows its presence).
class ResampleForwardOperation final : public OperationDescriptor
{
public:
    ResampleForwardOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
    {
    }

    const std::shared_ptr<ResampleDescriptor>& resample() const noexcept
    {
        return _resample;
    }

    std::shared_ptr<TensorDescriptor> x() const
    {
        return descriptorOf(record().x());
    }

    std::shared_ptr<TensorDescriptor> y() const
    {
        return descriptorOf(record().y());
    }

    // Null when unset.
    std::shared_ptr<TensorDescriptor> index() const
    {
        return descriptorOf(record().index());
    }

protected:
    void finalizeImpl() override
    {
        static const v8::ResampleForwardOperationLabels s_labels{
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA),
        };
        std::shared_ptr<ResampleDescriptor> resample = referent<ResampleDescriptor>(
            CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        const std::shared_ptr<TensorDescriptor> x = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC);
        const std::shared_ptr<TensorDescriptor> y = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC);
        const std::shared_ptr<TensorDescriptor> index
            = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC);
        std::shared_ptr<const v8::ResampleForwardOperationRecord> record = v8::makeOperationRecord(
            v8::ResampleForwardOperationInputs{
                recordOf(resample),
                recordOf(x),
                recordOf(y),
                recordOf(index),
                realAttribute(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA),
                realAttribute(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA)},
            s_labels);
        _resample = std::move(resample);
        commit(std::move(record), {x, y, index});
    }

private:
    const v8::ResampleForwardOperationRecord& record() const
    {
        return typedRecord<v8::ResampleForwardOperationRecord>();
    }

    std::shared_ptr<ResampleDescriptor> _resample;
};

// Lowers to ResampleBwdAttributes: DY, INDEX -> DX.
class ResampleBackwardOperation final : public OperationDescriptor
{
public:
    ResampleBackwardOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
    {
    }

    const std::shared_ptr<ResampleDescriptor>& resample() const noexcept
    {
        return _resample;
    }

    std::shared_ptr<TensorDescriptor> dx() const
    {
        return descriptorOf(record().dx());
    }

    std::shared_ptr<TensorDescriptor> dy() const
    {
        return descriptorOf(record().dy());
    }

    // Null when unset; always set for MAXPOOL.
    std::shared_ptr<TensorDescriptor> index() const
    {
        return descriptorOf(record().index());
    }

protected:
    void finalizeImpl() override
    {
        static const v8::ResampleBackwardOperationLabels s_labels{
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC),
        };
        std::shared_ptr<ResampleDescriptor> resample = referent<ResampleDescriptor>(
            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC, CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        const std::shared_ptr<TensorDescriptor> dx
            = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC);
        const std::shared_ptr<TensorDescriptor> dy
            = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC);
        const std::shared_ptr<TensorDescriptor> index
            = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC);
        std::shared_ptr<const v8::ResampleBackwardOperationRecord> record = v8::makeOperationRecord(
            v8::ResampleBackwardOperationInputs{
                recordOf(resample),
                recordOf(dx),
                recordOf(dy),
                recordOf(index),
                realAttribute(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA),
                realAttribute(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA),
                isSet(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC),
                isSet(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC)},
            s_labels);
        _resample = std::move(resample);
        commit(std::move(record), {dx, dy, index});
    }

private:
    const v8::ResampleBackwardOperationRecord& record() const
    {
        return typedRecord<v8::ResampleBackwardOperationRecord>();
    }

    std::shared_ptr<ResampleDescriptor> _resample;
};

// Lowers, by target(), to BatchnormAttributes (BATCH + TRAINING),
// BatchnormInferenceAttributes (BATCH + INFERENCE), LayernormAttributes or
// RMSNormAttributes. Optional tensors read as null when unset or not lowered.
class NormForwardOperation final : public OperationDescriptor
{
public:
    NormForwardOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
    {
    }

    NormFwdPhase_t phase() const
    {
        return record().phase();
    }

    static constexpr DataType_t computeType() noexcept
    {
        return v8::NormForwardOperationRecord::computeType();
    }

    std::int64_t normalizedDimCount() const
    {
        return record().normalizedDimCount();
    }

    std::shared_ptr<TensorDescriptor> x() const
    {
        return descriptorOf(record().x());
    }

    std::shared_ptr<TensorDescriptor> scale() const
    {
        return descriptorOf(record().scale());
    }

    std::shared_ptr<TensorDescriptor> bias() const
    {
        return descriptorOf(record().bias());
    }

    std::shared_ptr<TensorDescriptor> epsilon() const
    {
        return descriptorOf(record().epsilon());
    }

    std::shared_ptr<TensorDescriptor> y() const
    {
        return descriptorOf(record().y());
    }

    std::shared_ptr<TensorDescriptor> mean() const
    {
        return descriptorOf(record().mean());
    }

    std::shared_ptr<TensorDescriptor> invVariance() const
    {
        return descriptorOf(record().invVariance());
    }

    std::shared_ptr<TensorDescriptor> inputRunningMean() const
    {
        return descriptorOf(record().inputRunningMean());
    }

    std::shared_ptr<TensorDescriptor> inputRunningVariance() const
    {
        return descriptorOf(record().inputRunningVariance());
    }

    std::shared_ptr<TensorDescriptor> expAvgFactor() const
    {
        return descriptorOf(record().expAvgFactor());
    }

    std::shared_ptr<TensorDescriptor> outputRunningMean() const
    {
        return descriptorOf(record().outputRunningMean());
    }

    std::shared_ptr<TensorDescriptor> outputRunningVariance() const
    {
        return descriptorOf(record().outputRunningVariance());
    }

protected:
    void finalizeImpl() override
    {
        static const v8::NormForwardOperationLabels s_labels{
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_MODE),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_PHASE),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_XDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_YDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS),
        };
        const std::shared_ptr<TensorDescriptor> x = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_XDESC);
        const std::shared_ptr<TensorDescriptor> scale
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC);
        const std::shared_ptr<TensorDescriptor> bias
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC);
        const std::shared_ptr<TensorDescriptor> epsilon
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC);
        const std::shared_ptr<TensorDescriptor> y = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_YDESC);
        const std::shared_ptr<TensorDescriptor> mean
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC);
        const std::shared_ptr<TensorDescriptor> invVariance
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC);
        const std::shared_ptr<TensorDescriptor> inputRunningMean
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC);
        const std::shared_ptr<TensorDescriptor> inputRunningVariance
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC);
        const std::shared_ptr<TensorDescriptor> expAvgFactor
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC);
        const std::shared_ptr<TensorDescriptor> outputRunningMean
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC);
        const std::shared_ptr<TensorDescriptor> outputRunningVariance
            = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC);

        v8::NormForwardOperationInputs inputs;
        inputs.mode = scalarAttribute<cudnnBackendNormMode_t>(CUDNN_ATTR_OPERATION_NORM_FWD_MODE);
        inputs.phase
            = scalarAttribute<cudnnBackendNormFwdPhase_t>(CUDNN_ATTR_OPERATION_NORM_FWD_PHASE);
        inputs.x = recordOf(x);
        inputs.scale = recordOf(scale);
        inputs.bias = recordOf(bias);
        inputs.epsilon = recordOf(epsilon);
        inputs.y = recordOf(y);
        inputs.mean = recordOf(mean);
        inputs.invVariance = recordOf(invVariance);
        inputs.inputRunningMean = recordOf(inputRunningMean);
        inputs.inputRunningVariance = recordOf(inputRunningVariance);
        inputs.expAvgFactor = recordOf(expAvgFactor);
        inputs.outputRunningMean = recordOf(outputRunningMean);
        inputs.outputRunningVariance = recordOf(outputRunningVariance);
        inputs.hasPeerStats = isSet(CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS);
        commit(v8::makeOperationRecord(inputs, s_labels),
               {x,
                scale,
                bias,
                epsilon,
                y,
                mean,
                invVariance,
                inputRunningMean,
                inputRunningVariance,
                expAvgFactor,
                outputRunningMean,
                outputRunningVariance});
    }

private:
    const v8::NormForwardOperationRecord& record() const
    {
        return typedRecord<v8::NormForwardOperationRecord>();
    }
};

// Lowers, by target(), to BatchnormBackwardAttributes,
// LayernormBackwardAttributes or RMSNormBackwardAttributes. Optional tensors
// read as null when unset or not lowered.
class NormBackwardOperation final : public OperationDescriptor
{
public:
    NormBackwardOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
    {
    }

    static constexpr DataType_t computeType() noexcept
    {
        return v8::NormBackwardOperationRecord::computeType();
    }

    std::int64_t normalizedDimCount() const
    {
        return record().normalizedDimCount();
    }

    std::shared_ptr<TensorDescriptor> x() const
    {
        return descriptorOf(record().x());
    }

    std::shared_ptr<TensorDescriptor> dy() const
    {
        return descriptorOf(record().dy());
    }

    std::shared_ptr<TensorDescriptor> scale() const
    {
        return descriptorOf(record().scale());
    }

    std::shared_ptr<TensorDescriptor> mean() const
    {
        return descriptorOf(record().mean());
    }

    std::shared_ptr<TensorDescriptor> invVariance() const
    {
        return descriptorOf(record().invVariance());
    }

    std::shared_ptr<TensorDescriptor> epsilon() const
    {
        return descriptorOf(record().epsilon());
    }

    std::shared_ptr<TensorDescriptor> dx() const
    {
        return descriptorOf(record().dx());
    }

    std::shared_ptr<TensorDescriptor> dscale() const
    {
        return descriptorOf(record().dscale());
    }

    std::shared_ptr<TensorDescriptor> dbias() const
    {
        return descriptorOf(record().dbias());
    }

protected:
    void finalizeImpl() override
    {
        static const v8::NormBackwardOperationLabels s_labels{
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_MODE),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_XDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC),
            attributeLabel(CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS),
        };
        const std::shared_ptr<TensorDescriptor> x = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_XDESC);
        const std::shared_ptr<TensorDescriptor> dy = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC);
        const std::shared_ptr<TensorDescriptor> scale
            = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC);
        const std::shared_ptr<TensorDescriptor> mean
            = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC);
        const std::shared_ptr<TensorDescriptor> invVariance
            = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC);
        const std::shared_ptr<TensorDescriptor> epsilon
            = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC);
        const std::shared_ptr<TensorDescriptor> dx = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC);
        const std::shared_ptr<TensorDescriptor> dscale
            = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC);
        const std::shared_ptr<TensorDescriptor> dbias
            = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC);

        v8::NormBackwardOperationInputs inputs;
        inputs.mode = scalarAttribute<cudnnBackendNormMode_t>(CUDNN_ATTR_OPERATION_NORM_BWD_MODE);
        inputs.x = recordOf(x);
        inputs.dy = recordOf(dy);
        inputs.scale = recordOf(scale);
        inputs.mean = recordOf(mean);
        inputs.invVariance = recordOf(invVariance);
        inputs.epsilon = recordOf(epsilon);
        inputs.dx = recordOf(dx);
        inputs.dscale = recordOf(dscale);
        inputs.dbias = recordOf(dbias);
        inputs.hasPeerStats = isSet(CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS);
        commit(v8::makeOperationRecord(inputs, s_labels),
               {x, dy, scale, mean, invVariance, epsilon, dx, dscale, dbias});
    }

private:
    const v8::NormBackwardOperationRecord& record() const
    {
        return typedRecord<v8::NormBackwardOperationRecord>();
    }
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
