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
#include <initializer_list>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>
#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/parameter_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/tensor_descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/type_mapping.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
{

// The hipDNN graph node a finalized operation lowers to; named after the
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

constexpr std::optional<PointwiseModeTraits> pointwiseModeTraits(PointwiseMode_t mode) noexcept
{
    switch(mode)
    {
    case PointwiseMode_t::ADD:
    case PointwiseMode_t::MUL:
    case PointwiseMode_t::DIV:
    case PointwiseMode_t::SUB:
    case PointwiseMode_t::ADD_SQUARE:
    case PointwiseMode_t::MIN:
    case PointwiseMode_t::MAX:
    case PointwiseMode_t::MOD:
    case PointwiseMode_t::POW:
    case PointwiseMode_t::CMP_EQ:
    case PointwiseMode_t::CMP_NEQ:
    case PointwiseMode_t::CMP_GT:
    case PointwiseMode_t::CMP_GE:
    case PointwiseMode_t::CMP_LT:
    case PointwiseMode_t::CMP_LE:
    case PointwiseMode_t::LOGICAL_AND:
    case PointwiseMode_t::LOGICAL_OR:
        return PointwiseModeTraits{PointwiseModeClass::MATH, 3};
    case PointwiseMode_t::SQRT:
    case PointwiseMode_t::RSQRT:
    case PointwiseMode_t::SIN:
    case PointwiseMode_t::COS:
    case PointwiseMode_t::TAN:
    case PointwiseMode_t::LOG:
    case PointwiseMode_t::NEG:
    case PointwiseMode_t::ABS:
    case PointwiseMode_t::CEIL:
    case PointwiseMode_t::FLOOR:
    case PointwiseMode_t::LOGICAL_NOT:
    case PointwiseMode_t::GEN_INDEX:
    case PointwiseMode_t::ERF:
    case PointwiseMode_t::RECIPROCAL:
        return PointwiseModeTraits{PointwiseModeClass::MATH, 2};
    case PointwiseMode_t::BINARY_SELECT:
        return PointwiseModeTraits{PointwiseModeClass::MATH, 4};
    case PointwiseMode_t::IDENTITY:
        return PointwiseModeTraits{PointwiseModeClass::IDENTITY, 2};
    case PointwiseMode_t::RELU_FWD:
    case PointwiseMode_t::TANH_FWD:
    case PointwiseMode_t::SIGMOID_FWD:
    case PointwiseMode_t::ELU_FWD:
    case PointwiseMode_t::GELU_FWD:
    case PointwiseMode_t::GELU_APPROX_TANH_FWD:
    case PointwiseMode_t::SOFTPLUS_FWD:
    case PointwiseMode_t::SWISH_FWD:
    case PointwiseMode_t::EXP:
        return PointwiseModeTraits{PointwiseModeClass::ACTIVATION_FORWARD, 2};
    case PointwiseMode_t::RELU_BWD:
    case PointwiseMode_t::TANH_BWD:
    case PointwiseMode_t::SIGMOID_BWD:
    case PointwiseMode_t::ELU_BWD:
    case PointwiseMode_t::GELU_BWD:
    case PointwiseMode_t::GELU_APPROX_TANH_BWD:
    case PointwiseMode_t::SOFTPLUS_BWD:
    case PointwiseMode_t::SWISH_BWD:
        return PointwiseModeTraits{PointwiseModeClass::ACTIVATION_BACKWARD, 3};
    case PointwiseMode_t::NOT_SET:
    case PointwiseMode_t::COUNT:
    default:
        return std::nullopt;
    }
}

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

enum class NormFamily : std::uint8_t
{
    BATCH,
    LAYER,
    RMS
};

inline NormFamily requireNormFamily(cudnnBackendNormMode_t mode, cudnnBackendAttributeName_t name)
{
    const char* unsupported = "an unknown mode";
    switch(mode)
    {
    case CUDNN_BATCH_NORM:
        return NormFamily::BATCH;
    case CUDNN_LAYER_NORM:
        return NormFamily::LAYER;
    case CUDNN_RMS_NORM:
        return NormFamily::RMS;
    case CUDNN_ADA_LAYER_NORM:
        unsupported = "CUDNN_ADA_LAYER_NORM";
        break;
    case CUDNN_GROUP_NORM:
        unsupported = "CUDNN_GROUP_NORM";
        break;
    case CUDNN_INSTANCE_NORM:
        unsupported = "CUDNN_INSTANCE_NORM";
        break;
    default:
        break;
    }
    throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                    attributeName(name) + " " + unsupported
                        + " is not supported by the hipDNN cuDNN shim");
}

// hipDNN's layernorm normalized-dim count, derived from the scale shape exactly
// as LayerNormNode::infer_properties_node does: the trailing non-1 scale dims of
// a full-rank scale (at least 1), else every scale dim.
inline std::int64_t layernormNormalizedDimCount(const TensorDescriptor& x,
                                                const TensorDescriptor& scale)
{
    const std::vector<std::int64_t>& scaleDims = scale.dimensions();
    const auto rank = static_cast<std::int64_t>(scaleDims.size());
    if(x.dimensions().size() != scaleDims.size())
    {
        return rank;
    }
    std::int64_t count = 1;
    for(std::int64_t i = rank - 1; i >= 0; --i)
    {
        if(scaleDims[static_cast<std::size_t>(i)] == 1)
        {
            break;
        }
        count = rank - i;
    }
    return count;
}

// Common base of every CUDNN_BACKEND_OPERATION_* class. Finalize follows one
// order in every subclass: unsupported type, mode classification, required
// attributes (BAD_PARAM), attributes the mode does not use (BAD_PARAM),
// alpha/beta (NOT_SUPPORTED), set-but-unlowerable attributes (NOT_SUPPORTED).
class OperationDescriptor : public Descriptor
{
public:
    // Valid once finalized.
    OperationTarget target() const noexcept
    {
        return _target;
    }

protected:
    using Descriptor::Descriptor;

    void setTarget(OperationTarget target) noexcept
    {
        _target = target;
    }

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

    template <typename T>
    std::shared_ptr<T> requireReferent(cudnnBackendAttributeName_t name,
                                       cudnnBackendDescriptorType_t referentType) const
    {
        std::shared_ptr<T> value = referent<T>(name, referentType);
        if(!value)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM, attributeName(name) + " is required");
        }
        return value;
    }

    std::shared_ptr<TensorDescriptor> tensor(cudnnBackendAttributeName_t name) const
    {
        return referent<TensorDescriptor>(name, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    }

    std::shared_ptr<TensorDescriptor> requireTensor(cudnnBackendAttributeName_t name) const
    {
        return requireReferent<TensorDescriptor>(name, CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    }

    // All of `names` or none of them; the message names the first missing one.
    void requireAllOrNone(std::initializer_list<cudnnBackendAttributeName_t> names) const
    {
        const cudnnBackendAttributeName_t* present = nullptr;
        const cudnnBackendAttributeName_t* missing = nullptr;
        for(const cudnnBackendAttributeName_t& name : names)
        {
            if(hasAttribute(name))
            {
                present = present != nullptr ? present : &name;
            }
            else
            {
                missing = missing != nullptr ? missing : &name;
            }
        }
        if(present != nullptr && missing != nullptr)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            attributeName(*missing) + " is required when " + attributeName(*present)
                                + " is set");
        }
    }

    // Alpha/beta: absent means the default; any other value cannot be lowered.
    void requireScaling(cudnnBackendAttributeName_t name, double expected) const
    {
        const std::optional<double> value = realAttribute(name);
        if(value && *value != expected)
        {
            // Default stream formatting matches %g: "0.5", "1", "nan".
            std::ostringstream message;
            message << attributeName(name) << " = " << *value
                    << " is not supported by the hipDNN cuDNN shim; only " << expected << " is";
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED, message.str());
        }
    }

    void rejectUnlowerable(cudnnBackendAttributeName_t name, const char* context) const
    {
        const StoredAttribute* value = findAttribute(name);
        if(value != nullptr && value->count > 0)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            attributeName(name) + " is not supported by the hipDNN cuDNN shim"
                                + context);
        }
    }

private:
    OperationTarget _target = OperationTarget::NOT_SET;
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

// The three convolution operations differ only in their attribute names and
// which tensor is the output.
class ConvolutionOperationBase : public OperationDescriptor
{
public:
    // Valid once finalized.
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

    ConvolutionOperationBase(cudnnBackendDescriptorType_t type,
                             OperationTarget target,
                             const AttributeNames& names)
        : OperationDescriptor(type)
        , _names(names)
        , _loweredTarget(target)
    {
    }

    void finalizeImpl() override
    {
        std::shared_ptr<ConvolutionDescriptor> convolution = requireReferent<ConvolutionDescriptor>(
            _names.convolution, CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        std::array<std::shared_ptr<TensorDescriptor>, 3> tensors;
        for(std::size_t i = 0; i < tensors.size(); ++i)
        {
            tensors[i] = requireTensor(_names.tensors[i]);
        }
        const std::size_t rank = static_cast<std::size_t>(convolution->spatialDims()) + 2;
        for(std::size_t i = 0; i < tensors.size(); ++i)
        {
            const std::size_t actual = tensors[i]->dimensions().size();
            if(actual != rank)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                attributeName(_names.tensors[i]) + " has rank "
                                    + std::to_string(actual)
                                    + ", expected CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS + 2 = "
                                    + std::to_string(rank));
            }
        }
        requireScaling(_names.alpha, 1.0);
        requireScaling(_names.beta, 0.0);

        _convolution = std::move(convolution);
        _tensors = std::move(tensors);
        setTarget(_loweredTarget);
    }

    const std::shared_ptr<TensorDescriptor>& tensorAt(std::size_t index) const noexcept
    {
        return _tensors[index];
    }

private:
    AttributeNames _names;
    OperationTarget _loweredTarget;
    std::shared_ptr<ConvolutionDescriptor> _convolution;
    std::array<std::shared_ptr<TensorDescriptor>, 3> _tensors;
};

// Lowers to ConvFpropAttributes: X, W -> Y.
class ConvolutionForwardOperation final : public ConvolutionOperationBase
{
public:
    ConvolutionForwardOperation()
        : ConvolutionOperationBase(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                   OperationTarget::CONVOLUTION_FPROP,
                                   {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                                    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                                    {CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y}})
    {
    }

    const std::shared_ptr<TensorDescriptor>& x() const noexcept
    {
        return tensorAt(0);
    }

    const std::shared_ptr<TensorDescriptor>& w() const noexcept
    {
        return tensorAt(1);
    }

    const std::shared_ptr<TensorDescriptor>& y() const noexcept
    {
        return tensorAt(2);
    }
};

// Lowers to ConvDgradAttributes: DY, W -> DX.
class ConvolutionBackwardDataOperation final : public ConvolutionOperationBase
{
public:
    ConvolutionBackwardDataOperation()
        : ConvolutionOperationBase(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
                                   OperationTarget::CONVOLUTION_DGRAD,
                                   {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                                    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                                    {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX}})
    {
    }

    const std::shared_ptr<TensorDescriptor>& dy() const noexcept
    {
        return tensorAt(0);
    }

    const std::shared_ptr<TensorDescriptor>& w() const noexcept
    {
        return tensorAt(1);
    }

    const std::shared_ptr<TensorDescriptor>& dx() const noexcept
    {
        return tensorAt(2);
    }
};

// Lowers to ConvWgradAttributes: DY, X -> DW.
class ConvolutionBackwardFilterOperation final : public ConvolutionOperationBase
{
public:
    ConvolutionBackwardFilterOperation()
        : ConvolutionOperationBase(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
                                   OperationTarget::CONVOLUTION_WGRAD,
                                   {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC,
                                    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                                    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                                    {CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                     CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW}})
    {
    }

    const std::shared_ptr<TensorDescriptor>& dy() const noexcept
    {
        return tensorAt(0);
    }

    const std::shared_ptr<TensorDescriptor>& x() const noexcept
    {
        return tensorAt(1);
    }

    const std::shared_ptr<TensorDescriptor>& dw() const noexcept
    {
        return tensorAt(2);
    }
};

// Lowers to PointwiseAttributes. Slot map: forward classes X -> IN_0,
// B -> IN_1, T -> IN_2, Y -> OUT_0; activation-backward DY -> IN_0, X -> IN_1,
// DX -> OUT_0.
class PointwiseOperation final : public OperationDescriptor
{
public:
    PointwiseOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    {
    }

    // Valid once finalized.
    const std::shared_ptr<PointwiseDescriptor>& pointwise() const noexcept
    {
        return _pointwise;
    }

    PointwiseModeClass modeClass() const noexcept
    {
        return _traits.modeClass;
    }

    std::int64_t portCount() const noexcept
    {
        return _traits.portCount;
    }

    const std::shared_ptr<TensorDescriptor>& in0() const noexcept
    {
        return _in0;
    }

    // Null for unary modes.
    const std::shared_ptr<TensorDescriptor>& in1() const noexcept
    {
        return _in1;
    }

    // Null unless the mode takes three inputs (BINARY_SELECT).
    const std::shared_ptr<TensorDescriptor>& in2() const noexcept
    {
        return _in2;
    }

    const std::shared_ptr<TensorDescriptor>& out0() const noexcept
    {
        return _out0;
    }

protected:
    void finalizeImpl() override
    {
        std::shared_ptr<PointwiseDescriptor> pointwise = requireReferent<PointwiseDescriptor>(
            CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        const std::optional<PointwiseModeTraits> traits = pointwiseModeTraits(pointwise->mode());
        if(!traits)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR has no pointwise mode");
        }
        const bool backward = traits->modeClass == PointwiseModeClass::ACTIVATION_BACKWARD;

        std::shared_ptr<TensorDescriptor> x = requireTensor(CUDNN_ATTR_OPERATION_POINTWISE_XDESC);
        std::shared_ptr<TensorDescriptor> in0;
        std::shared_ptr<TensorDescriptor> in1;
        std::shared_ptr<TensorDescriptor> in2;
        std::shared_ptr<TensorDescriptor> out0;
        if(backward)
        {
            in0 = requireTensor(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC);
            out0 = requireTensor(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC);
            in1 = std::move(x);
        }
        else
        {
            out0 = requireTensor(CUDNN_ATTR_OPERATION_POINTWISE_YDESC);
            if(traits->portCount >= 3)
            {
                in1 = requireTensor(CUDNN_ATTR_OPERATION_POINTWISE_BDESC);
            }
            if(traits->portCount >= 4)
            {
                in2 = requireTensor(CUDNN_ATTR_OPERATION_POINTWISE_TDESC);
            }
            in0 = std::move(x);
        }

        if(backward)
        {
            rejectUnusedSlot(CUDNN_ATTR_OPERATION_POINTWISE_YDESC, *traits);
            rejectUnusedSlot(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, *traits);
            rejectUnusedSlot(CUDNN_ATTR_OPERATION_POINTWISE_TDESC, *traits);
        }
        else
        {
            rejectUnusedSlot(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC, *traits);
            rejectUnusedSlot(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC, *traits);
            // Upstream sends BDESC only for port >= 3 and TDESC only for port 4.
            if(traits->portCount < 3)
            {
                rejectUnusedSlot(CUDNN_ATTR_OPERATION_POINTWISE_BDESC, *traits);
            }
            if(traits->portCount < 4)
            {
                rejectUnusedSlot(CUDNN_ATTR_OPERATION_POINTWISE_TDESC, *traits);
            }
        }

        requireScaling(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, 1.0);
        requireScaling(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, 1.0);

        _pointwise = std::move(pointwise);
        _traits = *traits;
        _in0 = std::move(in0);
        _in1 = std::move(in1);
        _in2 = std::move(in2);
        _out0 = std::move(out0);
        setTarget(OperationTarget::POINTWISE);
    }

private:
    void rejectUnusedSlot(cudnnBackendAttributeName_t name, const PointwiseModeTraits& traits) const
    {
        if(hasAttribute(name))
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            attributeName(name) + " is not used by a pointwise "
                                + pointwiseModeClassName(traits.modeClass) + " mode with "
                                + std::to_string(traits.portCount) + " ports");
        }
    }

    std::shared_ptr<PointwiseDescriptor> _pointwise;
    PointwiseModeTraits _traits{PointwiseModeClass::MATH, 0};
    std::shared_ptr<TensorDescriptor> _in0;
    std::shared_ptr<TensorDescriptor> _in1;
    std::shared_ptr<TensorDescriptor> _in2;
    std::shared_ptr<TensorDescriptor> _out0;
};

// Lowers to MatmulAttributes: A, B -> C.
class MatmulOperation final : public OperationDescriptor
{
public:
    MatmulOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
    {
    }

    // Valid once finalized.
    const std::shared_ptr<MatmulDescriptor>& matmul() const noexcept
    {
        return _matmul;
    }

    const std::shared_ptr<TensorDescriptor>& a() const noexcept
    {
        return _a;
    }

    const std::shared_ptr<TensorDescriptor>& b() const noexcept
    {
        return _b;
    }

    const std::shared_ptr<TensorDescriptor>& c() const noexcept
    {
        return _c;
    }

protected:
    void finalizeImpl() override
    {
        std::shared_ptr<MatmulDescriptor> matmul = requireReferent<MatmulDescriptor>(
            CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        std::shared_ptr<TensorDescriptor> a = requireTensor(CUDNN_ATTR_OPERATION_MATMUL_ADESC);
        std::shared_ptr<TensorDescriptor> b = requireTensor(CUDNN_ATTR_OPERATION_MATMUL_BDESC);
        std::shared_ptr<TensorDescriptor> c = requireTensor(CUDNN_ATTR_OPERATION_MATMUL_CDESC);
        for(const cudnnBackendAttributeName_t name :
            {CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC,
             CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC,
             CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC})
        {
            rejectUnlowerable(name, "");
        }

        _matmul = std::move(matmul);
        _a = std::move(a);
        _b = std::move(b);
        _c = std::move(c);
        setTarget(OperationTarget::MATMUL);
    }

private:
    std::shared_ptr<MatmulDescriptor> _matmul;
    std::shared_ptr<TensorDescriptor> _a;
    std::shared_ptr<TensorDescriptor> _b;
    std::shared_ptr<TensorDescriptor> _c;
};

// Lowers to ReductionAttributes: X -> Y.
class ReductionOperation final : public OperationDescriptor
{
public:
    ReductionOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
    {
    }

    // Valid once finalized.
    const std::shared_ptr<ReductionDescriptor>& reduction() const noexcept
    {
        return _reduction;
    }

    const std::shared_ptr<TensorDescriptor>& x() const noexcept
    {
        return _x;
    }

    const std::shared_ptr<TensorDescriptor>& y() const noexcept
    {
        return _y;
    }

protected:
    void finalizeImpl() override
    {
        std::shared_ptr<ReductionDescriptor> reduction = requireReferent<ReductionDescriptor>(
            CUDNN_ATTR_OPERATION_REDUCTION_DESC, CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        std::shared_ptr<TensorDescriptor> x = requireTensor(CUDNN_ATTR_OPERATION_REDUCTION_XDESC);
        std::shared_ptr<TensorDescriptor> y = requireTensor(CUDNN_ATTR_OPERATION_REDUCTION_YDESC);

        _reduction = std::move(reduction);
        _x = std::move(x);
        _y = std::move(y);
        setTarget(OperationTarget::REDUCTION);
    }

private:
    std::shared_ptr<ReductionDescriptor> _reduction;
    std::shared_ptr<TensorDescriptor> _x;
    std::shared_ptr<TensorDescriptor> _y;
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

    // Valid once finalized.
    const std::shared_ptr<ResampleDescriptor>& resample() const noexcept
    {
        return _resample;
    }

    const std::shared_ptr<TensorDescriptor>& x() const noexcept
    {
        return _x;
    }

    const std::shared_ptr<TensorDescriptor>& y() const noexcept
    {
        return _y;
    }

    // Null when unset.
    const std::shared_ptr<TensorDescriptor>& index() const noexcept
    {
        return _index;
    }

protected:
    void finalizeImpl() override
    {
        std::shared_ptr<ResampleDescriptor> resample = requireReferent<ResampleDescriptor>(
            CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        std::shared_ptr<TensorDescriptor> x
            = requireTensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC);
        std::shared_ptr<TensorDescriptor> y
            = requireTensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC);
        std::shared_ptr<TensorDescriptor> index = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC);
        requireScaling(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA, 1.0);
        requireScaling(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA, 0.0);

        _resample = std::move(resample);
        _x = std::move(x);
        _y = std::move(y);
        _index = std::move(index);
        setTarget(OperationTarget::RESAMPLE_FWD);
    }

private:
    std::shared_ptr<ResampleDescriptor> _resample;
    std::shared_ptr<TensorDescriptor> _x;
    std::shared_ptr<TensorDescriptor> _y;
    std::shared_ptr<TensorDescriptor> _index;
};

// Lowers to ResampleBwdAttributes: DY, INDEX -> DX.
class ResampleBackwardOperation final : public OperationDescriptor
{
public:
    ResampleBackwardOperation()
        : OperationDescriptor(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
    {
    }

    // Valid once finalized.
    const std::shared_ptr<ResampleDescriptor>& resample() const noexcept
    {
        return _resample;
    }

    const std::shared_ptr<TensorDescriptor>& dx() const noexcept
    {
        return _dx;
    }

    const std::shared_ptr<TensorDescriptor>& dy() const noexcept
    {
        return _dy;
    }

    // Null when unset; always set for MAXPOOL.
    const std::shared_ptr<TensorDescriptor>& index() const noexcept
    {
        return _index;
    }

protected:
    void finalizeImpl() override
    {
        std::shared_ptr<ResampleDescriptor> resample = requireReferent<ResampleDescriptor>(
            CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC, CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        std::shared_ptr<TensorDescriptor> dx
            = requireTensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC);
        std::shared_ptr<TensorDescriptor> dy
            = requireTensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC);
        std::shared_ptr<TensorDescriptor> index = tensor(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC);
        // hipDNN's max-pool backward needs the forward pass's argmax indices.
        if(resample->mode() == ResampleMode_t::MAXPOOL && !index)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC is required for "
                            "CUDNN_RESAMPLE_MAXPOOL");
        }
        requireScaling(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA, 1.0);
        requireScaling(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA, 0.0);
        // hipDNN's resample backward has no X or Y slot.
        rejectUnlowerable(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC, " for resample backward");
        rejectUnlowerable(CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC, " for resample backward");

        _resample = std::move(resample);
        _dx = std::move(dx);
        _dy = std::move(dy);
        _index = std::move(index);
        setTarget(OperationTarget::RESAMPLE_BWD);
    }

private:
    std::shared_ptr<ResampleDescriptor> _resample;
    std::shared_ptr<TensorDescriptor> _dx;
    std::shared_ptr<TensorDescriptor> _dy;
    std::shared_ptr<TensorDescriptor> _index;
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

    // Valid once finalized.
    NormFwdPhase_t phase() const noexcept
    {
        return _phase;
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
        return _normalizedDimCount;
    }

    const std::shared_ptr<TensorDescriptor>& x() const noexcept
    {
        return _tensors.x;
    }

    const std::shared_ptr<TensorDescriptor>& scale() const noexcept
    {
        return _tensors.scale;
    }

    // Optional for RMSNORM.
    const std::shared_ptr<TensorDescriptor>& bias() const noexcept
    {
        return _tensors.bias;
    }

    // Null for BATCHNORM_INFERENCE: hipDNN's batchnorm inference has no epsilon.
    const std::shared_ptr<TensorDescriptor>& epsilon() const noexcept
    {
        return _tensors.epsilon;
    }

    const std::shared_ptr<TensorDescriptor>& y() const noexcept
    {
        return _tensors.y;
    }

    // Outputs except for BATCHNORM_INFERENCE, where they are inputs. RMSNORM
    // never has a mean; its inverse variance lowers to INV_RMS.
    const std::shared_ptr<TensorDescriptor>& mean() const noexcept
    {
        return _tensors.mean;
    }

    const std::shared_ptr<TensorDescriptor>& invVariance() const noexcept
    {
        return _tensors.invVariance;
    }

    // BATCHNORM only, all or none. The exponential average factor lowers to
    // MOMENTUM.
    const std::shared_ptr<TensorDescriptor>& inputRunningMean() const noexcept
    {
        return _tensors.inputRunningMean;
    }

    const std::shared_ptr<TensorDescriptor>& inputRunningVariance() const noexcept
    {
        return _tensors.inputRunningVariance;
    }

    const std::shared_ptr<TensorDescriptor>& expAvgFactor() const noexcept
    {
        return _tensors.expAvgFactor;
    }

    const std::shared_ptr<TensorDescriptor>& outputRunningMean() const noexcept
    {
        return _tensors.outputRunningMean;
    }

    const std::shared_ptr<TensorDescriptor>& outputRunningVariance() const noexcept
    {
        return _tensors.outputRunningVariance;
    }

protected:
    void finalizeImpl() override
    {
        const NormFamily family
            = requireNormFamily(requireAttribute(scalarAttribute<cudnnBackendNormMode_t>(
                                                     CUDNN_ATTR_OPERATION_NORM_FWD_MODE),
                                                 CUDNN_ATTR_OPERATION_NORM_FWD_MODE),
                                CUDNN_ATTR_OPERATION_NORM_FWD_MODE);
        const cudnnBackendNormFwdPhase_t rawPhase = requireAttribute(
            scalarAttribute<cudnnBackendNormFwdPhase_t>(CUDNN_ATTR_OPERATION_NORM_FWD_PHASE),
            CUDNN_ATTR_OPERATION_NORM_FWD_PHASE);
        const NormFwdPhase_t phase = requireFe(rawPhase, CUDNN_ATTR_OPERATION_NORM_FWD_PHASE);
        OperationTarget target = OperationTarget::RMSNORM;
        if(family == NormFamily::BATCH)
        {
            target = rawPhase == CUDNN_NORM_FWD_TRAINING ? OperationTarget::BATCHNORM
                                                         : OperationTarget::BATCHNORM_INFERENCE;
        }
        else if(family == NormFamily::LAYER)
        {
            target = OperationTarget::LAYERNORM;
        }

        Tensors tensors;
        tensors.x = requireTensor(CUDNN_ATTR_OPERATION_NORM_FWD_XDESC);
        tensors.scale = requireTensor(CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC);
        if(target == OperationTarget::BATCHNORM_INFERENCE)
        {
            tensors.mean = requireTensor(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC);
            tensors.invVariance = requireTensor(CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC);
        }
        if(target != OperationTarget::RMSNORM)
        {
            tensors.bias = requireTensor(CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC);
        }
        if(target != OperationTarget::BATCHNORM_INFERENCE)
        {
            tensors.epsilon = requireTensor(CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC);
        }
        tensors.y = requireTensor(CUDNN_ATTR_OPERATION_NORM_FWD_YDESC);
        if(target == OperationTarget::BATCHNORM || target == OperationTarget::LAYERNORM)
        {
            requireAllOrNone({CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                              CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC});
        }
        if(target == OperationTarget::BATCHNORM)
        {
            requireAllOrNone({CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                              CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                              CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                              CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                              CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC});
        }

        rejectUnlowerable(CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS, "");
        if(target != OperationTarget::BATCHNORM)
        {
            for(const cudnnBackendAttributeName_t name :
                {CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                 CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                 CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                 CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                 CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC})
            {
                rejectUnlowerable(name, " for this normalization mode and phase");
            }
        }
        if(target == OperationTarget::RMSNORM)
        {
            rejectUnlowerable(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC, " for CUDNN_RMS_NORM");
            tensors.bias = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC);
        }
        if(target != OperationTarget::BATCHNORM_INFERENCE)
        {
            tensors.mean = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC);
            tensors.invVariance = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC);
        }
        if(target == OperationTarget::BATCHNORM)
        {
            tensors.inputRunningMean
                = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC);
            tensors.inputRunningVariance
                = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC);
            tensors.expAvgFactor = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC);
            tensors.outputRunningMean
                = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC);
            tensors.outputRunningVariance
                = tensor(CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC);
        }

        _normalizedDimCount = target == OperationTarget::LAYERNORM
                                  ? layernormNormalizedDimCount(*tensors.x, *tensors.scale)
                                  : 0;
        _phase = phase;
        _tensors = std::move(tensors);
        setTarget(target);
    }

private:
    struct Tensors
    {
        std::shared_ptr<TensorDescriptor> x;
        std::shared_ptr<TensorDescriptor> scale;
        std::shared_ptr<TensorDescriptor> bias;
        std::shared_ptr<TensorDescriptor> epsilon;
        std::shared_ptr<TensorDescriptor> y;
        std::shared_ptr<TensorDescriptor> mean;
        std::shared_ptr<TensorDescriptor> invVariance;
        std::shared_ptr<TensorDescriptor> inputRunningMean;
        std::shared_ptr<TensorDescriptor> inputRunningVariance;
        std::shared_ptr<TensorDescriptor> expAvgFactor;
        std::shared_ptr<TensorDescriptor> outputRunningMean;
        std::shared_ptr<TensorDescriptor> outputRunningVariance;
    };

    NormFwdPhase_t _phase = NormFwdPhase_t::NOT_SET;
    std::int64_t _normalizedDimCount = 0;
    Tensors _tensors;
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

    // Valid once finalized.
    static constexpr DataType_t computeType() noexcept
    {
        return DataType_t::FLOAT;
    }

    // LAYERNORM_BACKWARD only (0 otherwise).
    std::int64_t normalizedDimCount() const noexcept
    {
        return _normalizedDimCount;
    }

    const std::shared_ptr<TensorDescriptor>& x() const noexcept
    {
        return _tensors.x;
    }

    const std::shared_ptr<TensorDescriptor>& dy() const noexcept
    {
        return _tensors.dy;
    }

    const std::shared_ptr<TensorDescriptor>& scale() const noexcept
    {
        return _tensors.scale;
    }

    // Never set for RMSNORM_BACKWARD.
    const std::shared_ptr<TensorDescriptor>& mean() const noexcept
    {
        return _tensors.mean;
    }

    // Lowers to INV_RMS for RMSNORM_BACKWARD.
    const std::shared_ptr<TensorDescriptor>& invVariance() const noexcept
    {
        return _tensors.invVariance;
    }

    // LAYERNORM_BACKWARD only: hipDNN's batchnorm backward has no epsilon.
    const std::shared_ptr<TensorDescriptor>& epsilon() const noexcept
    {
        return _tensors.epsilon;
    }

    const std::shared_ptr<TensorDescriptor>& dx() const noexcept
    {
        return _tensors.dx;
    }

    const std::shared_ptr<TensorDescriptor>& dscale() const noexcept
    {
        return _tensors.dscale;
    }

    // Optional for RMSNORM_BACKWARD (compute_dbias follows its presence).
    const std::shared_ptr<TensorDescriptor>& dbias() const noexcept
    {
        return _tensors.dbias;
    }

protected:
    void finalizeImpl() override
    {
        const NormFamily family
            = requireNormFamily(requireAttribute(scalarAttribute<cudnnBackendNormMode_t>(
                                                     CUDNN_ATTR_OPERATION_NORM_BWD_MODE),
                                                 CUDNN_ATTR_OPERATION_NORM_BWD_MODE),
                                CUDNN_ATTR_OPERATION_NORM_BWD_MODE);
        OperationTarget target = OperationTarget::RMSNORM_BACKWARD;
        if(family == NormFamily::BATCH)
        {
            target = OperationTarget::BATCHNORM_BACKWARD;
        }
        else if(family == NormFamily::LAYER)
        {
            target = OperationTarget::LAYERNORM_BACKWARD;
        }

        Tensors tensors;
        tensors.x = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_XDESC);
        tensors.dy = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC);
        tensors.scale = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC);
        // Without saved statistics hipDNN's batchnorm backward would need the
        // epsilon, which it cannot take.
        if(target == OperationTarget::BATCHNORM_BACKWARD)
        {
            tensors.mean = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC);
        }
        if(target != OperationTarget::LAYERNORM_BACKWARD)
        {
            tensors.invVariance = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC);
        }
        tensors.dx = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC);
        tensors.dscale = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC);
        if(target != OperationTarget::RMSNORM_BACKWARD)
        {
            tensors.dbias = requireTensor(CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC);
        }
        if(target == OperationTarget::LAYERNORM_BACKWARD)
        {
            requireAllOrNone({CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                              CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC});
        }

        rejectUnlowerable(CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS, "");
        if(target == OperationTarget::RMSNORM_BACKWARD)
        {
            rejectUnlowerable(CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC, " for CUDNN_RMS_NORM");
            tensors.dbias = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC);
        }
        if(target == OperationTarget::LAYERNORM_BACKWARD)
        {
            tensors.mean = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC);
            tensors.invVariance = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC);
            tensors.epsilon = tensor(CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC);
        }

        _normalizedDimCount = target == OperationTarget::LAYERNORM_BACKWARD
                                  ? layernormNormalizedDimCount(*tensors.x, *tensors.scale)
                                  : 0;
        _tensors = std::move(tensors);
        setTarget(target);
    }

private:
    struct Tensors
    {
        std::shared_ptr<TensorDescriptor> x;
        std::shared_ptr<TensorDescriptor> dy;
        std::shared_ptr<TensorDescriptor> scale;
        std::shared_ptr<TensorDescriptor> mean;
        std::shared_ptr<TensorDescriptor> invVariance;
        std::shared_ptr<TensorDescriptor> epsilon;
        std::shared_ptr<TensorDescriptor> dx;
        std::shared_ptr<TensorDescriptor> dscale;
        std::shared_ptr<TensorDescriptor> dbias;
    };

    std::int64_t _normalizedDimCount = 0;
    Tensors _tensors;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
