// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file ConvNative.cpp
 * @brief The conv-forward engine's native half: matching, scoring, dispatch, and the
 *        one function that registers them.
 *
 * A second engine beside Pointwise, split on graph node type (`ConvolutionFwdAttributes`
 * vs `PointwiseAttributes`) rather than operation, so one pack needs no operation
 * matcher of its own. Deliberately narrow -- stride 1, dilation 1, no padding,
 * cross-correlation, packed NCHW/KCRS/NKPQ, FLOAT/HALF -- and the matcher must refuse
 * everything outside that shape rather than let the naive kernel compute wrong answers.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.conv_fwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.conv_fwd.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.conv_fwd.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.conv_fwd.dispatch";

// KMD fields this pack varies along, and the tokens matching binds for dispatch.
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view X_TOKEN = "conv_fwd.x.uid";
constexpr std::string_view W_TOKEN = "conv_fwd.w.uid";
constexpr std::string_view Y_TOKEN = "conv_fwd.y.uid";

// The problem itself, which a UHD actually ranks on: the fields are RFC 0020 §6.1's
// (`dims[i]` positionally, the derived `dtype`), and without them the only bindable
// feature is a tensor uid -- a high-cardinality handle actively harmful as a model input.
//
// The ROOT spelling deliberately diverges from that grammar and must be reconciled when
// the pattern-driven path lands. §6.1 has `tensor-ref = tvar "." tensor-field`: two
// levels, where `tvar` is an operand name a UED `nodes` pattern binds (`$q.dims[2]`).
// These roots are three levels, op-qualified. A native matcher has no UED pattern to name
// its operands, so the matcher author picks the names, and op-qualifying them is what
// keeps two ops binding in one graph unambiguous; it also extends the `conv_fwd.x.uid`
// spelling already shipped above rather than standing a second convention beside it.
// Resolution is a flat lookup into BoundTokens, so depth costs nothing -- the divergence
// is a naming question for RFC 0020's owner, not a structural one.
constexpr std::string_view X_ROOT = "conv_fwd.x";
constexpr std::string_view W_ROOT = "conv_fwd.w";
constexpr std::string_view Y_ROOT = "conv_fwd.y";

// RFC 0019 §13.6's precomputed op-intrinsic cost fields, `$<op>.flops` and `$<op>.bytes`,
// which a features_signature divides into arithmetic intensity. They are facts about the
// op, identical for every engine implementing it, so they belong to the binding layer
// rather than to any one UHD.
constexpr std::string_view FLOPS_TOKEN = "conv_fwd.flops";
constexpr std::string_view BYTES_TOKEN = "conv_fwd.bytes";

/// x and y are 4-D NCHW/NKPQ; w is 4-D KCRS. All three are rank 4, which is the only
/// fact this constant states -- the per-tensor role is fixed by which uid is read.
constexpr uint32_t SUPPORTED_RANK = 4;
/// x/w/y are 2-D-spatial (H,W / R,S), so stride/dilation/padding are each length 2.
constexpr size_t SUPPORTED_SPATIAL_RANK = 2;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids a matched conv graph binds, in kernel argument order.
struct ConvFwdBinding
{
    int64_t x = 0;
    int64_t w = 0;
    int64_t y = 0;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// True when @p values has exactly @p expectedSize elements, all equal to @p expected.
/// A null or wrongly-sized vector is a refusal, not a vacuous pass: a 0-length vector
/// would otherwise satisfy "every element equals" for a conv this pack cannot serve.
bool allEqual(const flatbuffers::Vector<int64_t>* values, size_t expectedSize, int64_t expected)
{
    if(values == nullptr || values->size() != expectedSize)
    {
        return false;
    }
    for(const auto value : *values)
    {
        if(value != expected)
        {
            return false;
        }
    }
    return true;
}

/// True when @p tensor is a supported rank, holds packed row-major strides for its own
/// dims, is real device data, and is a dtype this pack's kernel is compiled for. The
/// packed-strides check matters because the kernel takes no stride arguments at all --
/// a tensor merely ordered NCHW but not contiguous would be read at the wrong offset
/// without ever failing to match.
bool isSupportedOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || strides->size() != dims->size()
       || dims->size() != SUPPORTED_RANK)
    {
        return false;
    }

    int64_t expectedStride = 1;
    for(size_t i = dims->size(); i-- > 0;)
    {
        const auto axis = static_cast<flatbuffers::uoffset_t>(i);
        if(strides->Get(axis) != expectedStride)
        {
            return false;
        }
        expectedStride *= dims->Get(axis);
    }

    if(tensor.virtual_())
    {
        return false;
    }

    // A rank-4 tensor is also the shape a pass-by-value scalar can take; that variant-
    // pack slot holds a host pointer, not a device one.
    if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return false;
    }

    const auto dataType = tensor.data_type();
    return dataType == data_objects::DataType::FLOAT || dataType == data_objects::DataType::HALF;
}

std::string dataTypeName(data_objects::DataType dataType)
{
    return data_objects::EnumNameDataType(dataType);
}

/// The two runtime facts about a dtype the binding publishes.
///
/// `spelling` is what `to_string(DataType)` in hipdnn_frontend/Types.hpp answers -- the
/// only vocabulary a `$q.dtype` binding may hold, and the one CategoricalEncoding.hpp
/// encodes. It is restated here rather than called because this provider does not link
/// the frontend, and `EnumNameDataType` answers a different vocabulary ("FLOAT", "HALF")
/// that no model-side encoding knows.
///
/// An empty `spelling` is `to_string`'s "unknown" fallthrough, and `bytes == 0` is a
/// width this pack will not state: both make the dependent token absent instead of
/// silently wrong, which is what RFC 0019 §13.6 requires of a cost field with no exact
/// form. The sub-byte types are the second kind -- their footprint is a property of the
/// packing, not of the element, so no per-element byte count exists to publish.
struct DataTypeFacts
{
    std::string_view spelling;
    int64_t bytes;
};

DataTypeFacts dataTypeFacts(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return {"fp32", 4};
    case data_objects::DataType::HALF:
        return {"fp16", 2};
    case data_objects::DataType::BFLOAT16:
        return {"bf16", 2};
    case data_objects::DataType::DOUBLE:
        return {"fp64", 8};
    case data_objects::DataType::UINT8:
        return {"uint8", 1};
    case data_objects::DataType::INT32:
        return {"int32", 4};
    case data_objects::DataType::INT8:
        return {"int8", 1};
    case data_objects::DataType::FP8_E4M3:
        return {"fp8_e4m3", 1};
    case data_objects::DataType::FP8_E5M2:
        return {"fp8_e5m2", 1};
    case data_objects::DataType::FP8_E8M0:
        return {"fp8_e8m0", 1};
    case data_objects::DataType::FP8_E4M3_FNUZ:
        return {"fp8_e4m3_fnuz", 1};
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return {"fp8_e5m2_fnuz", 1};
    case data_objects::DataType::INT64:
        return {"int64", 8};
    case data_objects::DataType::BOOLEAN:
        return {"boolean", 1};
    case data_objects::DataType::FP4_E2M1:
        return {"fp4_e2m1", 0};
    case data_objects::DataType::FP6_E2M3:
        return {"fp6_e2m3", 0};
    case data_objects::DataType::FP6_E3M2:
        return {"fp6_e3m2", 0};
    case data_objects::DataType::INT4:
        return {"int4", 0};
    case data_objects::DataType::UNSET:
    default:
        return {{}, 0};
    }
}

/// @p left * @p right, or nullopt if the product would overflow or either factor is not
/// positive. Dims arrive from the caller and nothing upstream bounds their product, so an
/// unchecked multiply would publish a wrapped, negative "cost" as a model feature -- the
/// silently-wrong value RFC 0019 §13.6 rules out.
std::optional<int64_t> checkedMultiply(int64_t left, int64_t right)
{
    if(left <= 0 || right <= 0 || left > std::numeric_limits<int64_t>::max() / right)
    {
        return std::nullopt;
    }
    return left * right;
}

std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t result = 1;
    for(const auto factor : factors)
    {
        const auto next = checkedMultiply(result, factor);
        if(!next.has_value())
        {
            return std::nullopt;
        }
        result = *next;
    }
    return result;
}

std::optional<int64_t> elementCount(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    int64_t count = 1;
    for(const auto dim : *dims)
    {
        const auto next = checkedMultiply(count, dim);
        if(!next.has_value())
        {
            return std::nullopt;
        }
        count = *next;
    }
    return count;
}

/// Bytes moved by the op: the sum over its operands of element_count x **that tensor's
/// own** dtype size. Reading one operand's dtype and applying it to all of them is the
/// bug RFC 0019 §13.6's caveat names -- it is wrong the moment a graph is mixed-precision
/// (fp8 in, fp16 out). This pack refuses mixed dtypes today, so the per-tensor sum is the
/// form rather than yet a difference; a pack that admits them inherits the right answer.
///
/// nullopt when any operand has no statable element width, so the token is absent rather
/// than short by one tensor.
std::optional<int64_t>
    movedBytes(std::initializer_list<const data_objects::TensorAttributes*> operands)
{
    int64_t total = 0;
    for(const auto* operand : operands)
    {
        const auto count = elementCount(*operand);
        const auto width = dataTypeFacts(operand->data_type()).bytes;
        if(!count.has_value() || width == 0)
        {
            return std::nullopt;
        }
        const auto operandBytes = checkedMultiply(*count, width);
        if(!operandBytes.has_value()
           || total > std::numeric_limits<int64_t>::max() - *operandBytes)
        {
            return std::nullopt;
        }
        total += *operandBytes;
    }
    return total;
}

/// Publishes @p tensor's RFC 0020 §6.1 shape fields under @p root: every dim positionally
/// as `dims[i]`, and the derived `dtype`.
///
/// dtype binds as the runtime spelling **string**, never a pre-encoded number: the
/// integer code space is CategoricalEncoding.hpp's and is applied downstream by the
/// feature extractor, so a number here would freeze that code space inside the matcher
/// and drift from it silently.
void bindTensorFields(BoundTokens& bound,
                      std::string_view root,
                      const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const std::string prefix(root);
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        bound[prefix + ".dims[" + std::to_string(axis) + "]"] = dims->Get(axis);
    }

    const auto spelling = dataTypeFacts(tensor.data_type()).spelling;
    if(!spelling.empty())
    {
        bound[prefix + ".dtype"] = std::string(spelling);
    }
}

/// The node this engine's matchers read, or nullptr if the graph is not a single
/// conv-forward node.
const data_objects::ConvolutionFwdAttributes* convFwdNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::ConvolutionFwdAttributes>();
}

/// The graph's element type, from the x operand; the matcher below requires every
/// operand to agree, so any of them would answer the same.
std::optional<data_objects::DataType> graphDataType(const MatchContext& context)
{
    const auto* attributes = convFwdNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes->x_tensor_uid());
    return x == nullptr ? std::nullopt : std::make_optional(x->data_type());
}

/**
 * @brief Graph-scoped applicability: is this the one conv shape this engine's kernel
 *        can launch? One pack, one operation, so this matcher (unlike Pointwise's)
 *        both admits the node type and validates it in one pass.
 */
std::optional<BoundTokens> convFwdGraphMatches(const MatchContext& context)
{

    const auto* attributesPtr = convFwdNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    if(attributes.conv_mode() != data_objects::ConvMode::CROSS_CORRELATION)
    {
        return std::nullopt;
    }

    // Deliberately narrow: stride 1, dilation 1, no padding is the only shape the
    // in-kernel p = h - r + 1 / q = width - s + 1 formula is correct for.
    if(!allEqual(attributes.stride(), SUPPORTED_SPATIAL_RANK, 1)
       || !allEqual(attributes.dilation(), SUPPORTED_SPATIAL_RANK, 1)
       || !allEqual(attributes.pre_padding(), SUPPORTED_SPATIAL_RANK, 0)
       || !allEqual(attributes.post_padding(), SUPPORTED_SPATIAL_RANK, 0))
    {
        return std::nullopt;
    }

    const auto* x = findTensor(context, attributes.x_tensor_uid());
    const auto* w = findTensor(context, attributes.w_tensor_uid());
    const auto* y = findTensor(context, attributes.y_tensor_uid());
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        return std::nullopt;
    }

    if(!isSupportedOperand(*x) || !isSupportedOperand(*w) || !isSupportedOperand(*y))
    {
        return std::nullopt;
    }

    // Uniform dtype across operands; mixed precision is a different kernel.
    if(x->data_type() != w->data_type() || x->data_type() != y->data_type())
    {
        return std::nullopt;
    }

    const auto* xDims = x->dims();
    const auto* wDims = w->dims();
    const auto* yDims = y->dims();
    const auto xC = xDims->Get(1);
    const auto xH = xDims->Get(2);
    const auto xW = xDims->Get(3);
    const auto wK = wDims->Get(0);
    const auto wR = wDims->Get(2);
    const auto wS = wDims->Get(3);

    // Filter channels vs. input channels -- this also refuses grouped conv, which is
    // encoded purely as a smaller w channel count; the kernel has no notion of groups
    // and indexes w using c from x alone, so a filter with fewer channels would read
    // past the end.
    if(wDims->Get(1) != xC)
    {
        return std::nullopt;
    }

    // r <= h and s <= width, or p = h - r + 1 / q = width - s + 1 go non-positive,
    // which the kernel's flat-index unravel (ConvFwd.cpp) never expects.
    if(wR > xH || wS > xW)
    {
        return std::nullopt;
    }

    // y must be exactly the shape the kernel computes: total = n*k*p*q comes from x
    // and w alone, and the kernel writes every index < total into y, so a smaller y
    // overflows.
    if(yDims->Get(0) != xDims->Get(0) || yDims->Get(1) != wK || yDims->Get(2) != xH - wR + 1
       || yDims->Get(3) != xW - wS + 1)
    {
        return std::nullopt;
    }

    // Binds operand uids for the dispatch handler to read back rather than re-deriving
    // them from the graph.
    BoundTokens bound;
    bound[std::string(X_TOKEN)] = attributes.x_tensor_uid();
    bound[std::string(W_TOKEN)] = attributes.w_tensor_uid();
    bound[std::string(Y_TOKEN)] = attributes.y_tensor_uid();

    // The dims and dtypes validated just above, published instead of discarded: they are
    // the only features a UHD can rank a conv on (RFC 0020 §6.1).
    bindTensorFields(bound, X_ROOT, *x);
    bindTensorFields(bound, W_ROOT, *w);
    bindTensorFields(bound, Y_ROOT, *y);

    // 2 x N x K x P x Q x C x R x S: two flops per multiply-accumulate, over every output
    // element and every filter tap.
    //
    // Convention: the settled project convention is rocKE's causal-EFFECTIVE count
    // (`attention_flops` in rocke/.../stage1_benchmark/_ua_shape_utils.py), not the dense
    // one RFC 0019 §887 and RFC 0019.13 §4.3.3 state -- those differ by ~2x for square
    // causal attention. Conv forward has no masking, so both conventions give this same
    // number; the choice is recorded because the next op bound here will not be so lucky.
    const auto flops
        = checkedProduct({2, xDims->Get(0), wK, yDims->Get(2), yDims->Get(3), xC, wR, wS});
    if(flops.has_value())
    {
        bound[std::string(FLOPS_TOKEN)] = *flops;
    }

    const auto bytes = movedBytes({x, w, y});
    if(bytes.has_value())
    {
        bound[std::string(BYTES_TOKEN)] = *bytes;
    }

    return bound;
}

/**
 * @brief Kernel-scoped applicability: does this kernel's dtype match the graph's?
 *        Evaluated once per candidate kernel; without it an f32 graph could reach an
 *        f16 binary and return wrong numbers rather than failing.
 */
bool convFwdKernelMatches(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel)
{
    const auto dataType = graphDataType(context);
    if(!dataType.has_value())
    {
        return false;
    }

    return kernel.getStringMetadata(std::string(DTYPE_FIELD)) == dataTypeName(*dataType);
}

double convFwdScore(const MatchContext& /*context*/,
                    const BoundTokens& /*bound*/,
                    const KernelDefinition& kernel)
{
    // A stand-in for a trained model: prefers the larger block size.
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));
}

/**
 * @brief Re-reads the operand bindings a match established.
 *
 * @throws HipdnnPluginException if the graph is not one this matcher accepts.
 */
ConvFwdBinding convFwdBinding(const BoundTokens& bound)
{
    // Every token was written by the engine's graph match, which admitted this graph; a
    // missing one means the catalog was built by an engine other than ours.
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "conv_fwd dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold a tensor uid");
        }
        return *value;
    };

    return {read(X_TOKEN), read(W_TOKEN), read(Y_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything its launch geometry needs: the operand uids and
/// the seven dims read from the graph at prepare() time, owning nothing that points
/// back into it.
class PreparedConvFwd : public PreparedDispatch
{
public:
    PreparedConvFwd(std::unique_ptr<compilation::ICompiledProgram> program,
                    std::unique_ptr<compilation::IRunnableKernel> kernel,
                    ConvFwdBinding binding,
                    int n,
                    int c,
                    int h,
                    int width,
                    int k,
                    int r,
                    int s)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
        , _n(n)
        , _c(c)
        , _h(h)
        , _width(width)
        , _k(k)
        , _r(r)
        , _s(s)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const ConvFwdBinding& binding() const
    {
        return _binding;
    }

    int n() const
    {
        return _n;
    }
    int c() const
    {
        return _c;
    }
    int h() const
    {
        return _h;
    }
    int width() const
    {
        return _width;
    }
    int k() const
    {
        return _k;
    }
    int r() const
    {
        return _r;
    }
    int s() const
    {
        return _s;
    }

private:
    // The runnable kernel is a view into its program's module, so the program must
    // outlive it; both are held here for the plan's lifetime.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    ConvFwdBinding _binding;
    int _n;
    int _c;
    int _h;
    int _width;
    int _k;
    int _r;
    int _s;
};

/// The C++ type the kernel is compiled for, from the kernel's dtype metadata.
std::string elementTypeFor(const KernelDefinition& kernel)
{
    const auto& dtype = kernel.getStringMetadata(std::string(DTYPE_FIELD));
    if(dtype == "FLOAT")
    {
        return "float";
    }
    if(dtype == "HALF")
    {
        return "_Float16";
    }

    // Unreachable via matching, which admits only dtypes this pack declares.
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "kernel '" + toString(kernel.kernelId) + "' declares unsupported dtype '" + dtype + "'");
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    if(it == tensors.end() || it->second == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched conv_fwd graph has no tensor for uid " + std::to_string(uid));
    }
    return *it->second;
}

/**
 * @brief The native dispatch behind this pack's UDD: sizes and launches the conv
 *        kernel. Splits per RFC 0017 §8.5: everything graph/kernel-derived resolves
 *        once at prepare(); execute() only resolves buffers and launches, so nothing
 *        mutates once prepared and concurrent execution is safe.
 */
class ConvFwdDispatchHandler : public hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime.
    explicit ConvFwdDispatchHandler(const compilation::IKernelCompiler& kernelCompiler)
        : _kernelCompiler(kernelCompiler)
    {
    }

    /// This reference kernel needs no scratch: every output element is accumulated in
    /// a register and written once.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        // Reads the operand uids the graph match bound rather than re-deriving them.
        const auto binding = convFwdBinding(bound);

        const auto& xTensor = requireTensor(context, binding.x);
        const auto& wTensor = requireTensor(context, binding.w);

        // Dims were validated NCHW/KCRS-shaped, packed, rank 4 by the graph matcher.
        const auto* xDims = xTensor.dims();
        const auto* wDims = wTensor.dims();
        const auto n = static_cast<int>(xDims->Get(0));
        const auto c = static_cast<int>(xDims->Get(1));
        const auto h = static_cast<int>(xDims->Get(2));
        const auto width = static_cast<int>(xDims->Get(3));
        const auto k = static_cast<int>(wDims->Get(0));
        const auto r = static_cast<int>(wDims->Get(2));
        const auto s = static_cast<int>(wDims->Get(3));

        const auto blockSize
            = static_cast<unsigned int>(kernel.getIntMetadata(std::string(BLOCK_SIZE_FIELD)));

        compilation::KernelCompileOptions options(&xTensor, context.deviceProperties.gcnArchName);
        options.add("HIP_PLUGIN_CONV_TYPE", elementTypeFor(kernel));
        options.add("HIP_PLUGIN_CONV_BLOCK_SIZE", blockSize);

        // The only KernelSourceKind this dispatch handler knows how to load.
        auto program = _kernelCompiler.compile(kernel.source.sourceFile, options);
        auto runnableKernel = program->getKernel(kernel.source.entryPoint);

        const auto p = h - r + 1;
        const auto q = width - s + 1;
        // int64_t: n*k*p*q can exceed 2^31 for shapes this matcher admits. A 32-bit
        // product here previously wrapped silently, corrupting both the grid size and
        // the kernel's own bounds guard (ConvFwd.cpp).
        const int64_t total = static_cast<int64_t>(n) * k * p * q;
        const auto gridSize = static_cast<unsigned int>(
            (total + static_cast<int64_t>(blockSize) - 1) / static_cast<int64_t>(blockSize));

        runnableKernel->setBlockSize(blockSize, 1, 1);
        runnableKernel->setGridSize(gridSize, 1, 1);

        return std::make_unique<PreparedConvFwd>(
            std::move(program), std::move(runnableKernel), binding, n, c, h, width, k, r, s);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedConvFwd = dynamic_cast<const PreparedConvFwd&>(prepared);
        const auto& binding = preparedConvFwd.binding();

        const auto x
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.x, deviceBuffers, numDeviceBuffers);
        const auto w
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.w, deviceBuffers, numDeviceBuffers);
        const auto y
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.y, deviceBuffers, numDeviceBuffers);

        preparedConvFwd.kernel().launch(handle.getStream(),
                                        x.ptr,
                                        w.ptr,
                                        y.ptr,
                                        preparedConvFwd.n(),
                                        preparedConvFwd.c(),
                                        preparedConvFwd.h(),
                                        preparedConvFwd.width(),
                                        preparedConvFwd.k(),
                                        preparedConvFwd.r(),
                                        preparedConvFwd.s());
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
};

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so
/// it (and the compiler it holds) must outlive every Container.
const ConvFwdDispatchHandler& convFwdDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const ConvFwdDispatchHandler s_dispatchHandler(s_kernelCompiler);
    return s_dispatchHandler;
}

} // namespace

void registerConvFwdSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &convFwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &convFwdKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &convFwdScore);
    scope.add(std::string(DISPATCH_SYMBOL), &convFwdDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
