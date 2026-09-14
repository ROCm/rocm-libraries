#include "graph_translate.hpp"

#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace studio
{
namespace
{

using json = nlohmann::json;
using hipdnn_frontend::ConvolutionMode;
using hipdnn_frontend::DataType;
using hipdnn_frontend::NormFwdPhase;
using hipdnn_frontend::PaddingMode;
using hipdnn_frontend::PointwiseMode;
using hipdnn_frontend::ReductionMode;
using hipdnn_frontend::ResampleMode;
using hipdnn_frontend::graph::BlockScaleDequantizeAttributes;
using hipdnn_frontend::graph::BlockScaleQuantizeAttributes;
using hipdnn_frontend::graph::ConvDgradAttributes;
using hipdnn_frontend::graph::ConvFpropAttributes;
using hipdnn_frontend::graph::ConvWgradAttributes;
using hipdnn_frontend::graph::LayernormAttributes;
using hipdnn_frontend::graph::LayernormBackwardAttributes;
using hipdnn_frontend::graph::MatmulAttributes;
using hipdnn_frontend::graph::PointwiseAttributes;
using hipdnn_frontend::graph::ResampleFwdAttributes;
using hipdnn_frontend::graph::RMSNormAttributes;
using hipdnn_frontend::graph::RMSNormBackwardAttributes;
using hipdnn_frontend::graph::TensorAttributes;

int64_t g_nextUid = 0;

// ── Studio enum strings → frontend enums ───────────────────────────────
// Tables are tiny (≤17 entries), so a linear scan beats a hash map and needs
// no dynamic initialization.
template <typename E>
struct EnumEntry
{
    std::string_view name;
    E value;
};

template <typename E, std::size_t N>
bool lookupEnum(const EnumEntry<E> (&table)[N], std::string_view key, E& out)
{
    for(const auto& entry : table)
    {
        if(entry.name == key)
        {
            out = entry.value;
            return true;
        }
    }
    return false;
}

template <typename E, std::size_t N>
E lookupEnumOr(const EnumEntry<E> (&table)[N], std::string_view key, E fallback)
{
    E value{};
    return lookupEnum(table, key, value) ? value : fallback;
}

constexpr EnumEntry<DataType> kDataTypes[] = {
    {"float32", DataType::FLOAT},
    {"float", DataType::FLOAT},
    {"float16", DataType::HALF},
    {"half", DataType::HALF},
    {"bfloat16", DataType::BFLOAT16},
    {"int8", DataType::INT8},
};

constexpr EnumEntry<ConvolutionMode> kConvModes[] = {
    {"CONVOLUTION", ConvolutionMode::CONVOLUTION},
    {"CROSS_CORRELATION", ConvolutionMode::CROSS_CORRELATION},
};

constexpr EnumEntry<NormFwdPhase> kNormPhases[] = {
    {"inference", NormFwdPhase::INFERENCE},
    {"training", NormFwdPhase::TRAINING},
};

constexpr EnumEntry<ResampleMode> kResampleModes[] = {
    {"MAXPOOL", ResampleMode::MAXPOOL},
    {"AVGPOOL_EXCLUDE_PADDING", ResampleMode::AVGPOOL_EXCLUDE_PADDING},
    {"AVGPOOL_INCLUDE_PADDING", ResampleMode::AVGPOOL_INCLUDE_PADDING},
    {"BILINEAR", ResampleMode::BILINEAR},
};

constexpr EnumEntry<PaddingMode> kPaddingModes[] = {
    {"NEG_INF_PAD", PaddingMode::NEG_INF_PAD},
    {"ZERO_PAD", PaddingMode::ZERO_PAD},
};

constexpr EnumEntry<PointwiseMode> kPointwiseModes[] = {
    {"relu_fwd", PointwiseMode::RELU_FWD},
    {"sigmoid_fwd", PointwiseMode::SIGMOID_FWD},
    {"tanh_fwd", PointwiseMode::TANH_FWD},
    {"gelu_fwd", PointwiseMode::GELU_FWD},
    {"exp", PointwiseMode::EXP},
    {"log", PointwiseMode::LOG},
    {"abs", PointwiseMode::ABS},
    {"neg", PointwiseMode::NEG},
    {"sqrt", PointwiseMode::SQRT},
    {"rsqrt", PointwiseMode::RSQRT},
    {"reciprocal", PointwiseMode::RECIPROCAL},
    {"add", PointwiseMode::ADD},
    {"mul", PointwiseMode::MUL},
    {"sub", PointwiseMode::SUB},
    {"div", PointwiseMode::DIV},
    {"max_op", PointwiseMode::MAX},
    {"min_op", PointwiseMode::MIN},
};

constexpr EnumEntry<ReductionMode> kReductionModes[] = {
    {"add", ReductionMode::ADD},
    {"mul", ReductionMode::MUL},
    {"min", ReductionMode::MIN},
    {"max", ReductionMode::MAX},
    {"amax", ReductionMode::AMAX},
    {"avg", ReductionMode::AVG},
    {"norm1", ReductionMode::NORM1},
    {"norm2", ReductionMode::NORM2},
};

// ── Studio JSON accessors ──────────────────────────────────────────────

int64_t intParam(const json& params, const char* key, int64_t fallback)
{
    auto it = params.find(key);
    if(it == params.end() || !it->is_number())
        return fallback;
    return it->get<int64_t>();
}

std::string stringParam(const json& params, const char* key, const std::string& fallback)
{
    auto it = params.find(key);
    if(it == params.end() || !it->is_string())
        return fallback;
    return it->get<std::string>();
}

double floatParam(const json& params, const char* key, double fallback)
{
    auto it = params.find(key);
    if(it == params.end() || !it->is_number())
        return fallback;
    return it->get<double>();
}

bool boolParam(const json& params, const char* key, bool fallback)
{
    auto it = params.find(key);
    if(it == params.end() || !it->is_boolean())
        return fallback;
    return it->get<bool>();
}

std::vector<int64_t> rowMajorStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(int i = static_cast<int>(dims.size()) - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

std::vector<int64_t> parseShape(const std::string& csv)
{
    std::vector<int64_t> dims;
    std::size_t start = 0;
    while(start < csv.size())
    {
        std::size_t comma = csv.find(',', start);
        if(comma == std::string::npos)
            comma = csv.size();
        const std::string tok = csv.substr(start, comma - start);
        try
        {
            const long long v = std::stoll(tok);
            if(v > 0)
                dims.push_back(static_cast<int64_t>(v));
        }
        catch(...)
        {
            // Skip non-numeric fragments; hipDNN validation will flag bad shapes.
        }
        start = comma + 1;
    }
    return dims;
}

// Resolve the source (node, port) feeding a target (node, port). Returns the
// edge (or nullptr) and fills srcNode/srcPort.
const json* findEdgeSource(const json& edges,
                           const std::string& node,
                           const std::string& port,
                           std::string& srcNodeOut,
                           std::string& srcPortOut)
{
    for(const auto& e : edges)
    {
        if(e.at("to").at("node").get<std::string>() == node
           && e.at("to").at("port").get<std::string>() == port)
        {
            srcNodeOut = e.at("from").at("node").get<std::string>();
            srcPortOut = e.at("from").at("port").get<std::string>();
            return &e;
        }
    }
    return nullptr;
}

std::string portKey(const std::string& node, const std::string& port)
{
    return node + ":" + port;
}

} // namespace

std::size_t dtypeSize(DataType dt)
{
    switch(dt)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
        return 8;
    case DataType::FLOAT:
    case DataType::INT32:
        return 4;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    default:
        return 1;
    }
}

DataType pickIoDtype(const json& root)
{
    for(const auto& n : root.at("nodes"))
    {
        if(n.at("type").get<std::string>() == "Input")
        {
            const json params = n.value("params", json::object());
            return lookupEnumOr(
                kDataTypes, stringParam(params, "dtype", "float16"), DataType::HALF);
        }
    }
    return DataType::HALF;
}

void translateGraph(const json& root,
                    Graph& g,
                    DataType ioDtype,
                    std::unordered_map<std::string, TensorPtr>& outputByPort)
{
    const json& nodes = root.at("nodes");
    const json& edges = root.at("edges");

    // Resolve the tensor feeding (nodeId, inPort) by following its edge to the
    // producing (srcNode, srcPort). Throws if unconnected.
    auto resolveInput = [&](const std::string& nodeId, const std::string& inPort) -> TensorPtr {
        std::string srcNode, srcPort;
        if(!findEdgeSource(edges, nodeId, inPort, srcNode, srcPort))
        {
            throw BuildInputError{"INVALID_VALUE", "Input port '" + inPort + "' is not connected."};
        }
        auto it = outputByPort.find(portKey(srcNode, srcPort));
        if(it == outputByPort.end())
        {
            throw BuildInputError{"INVALID_VALUE", "Upstream port produced no tensor (cycle?)."};
        }
        return it->second;
    };

    // Like resolveInput but returns nullptr when the port is unconnected — for
    // optional inputs (e.g. saved mean/inv_variance, optional bias).
    auto resolveInputOpt = [&](const std::string& nodeId, const std::string& inPort) -> TensorPtr {
        std::string srcNode, srcPort;
        if(!findEdgeSource(edges, nodeId, inPort, srcNode, srcPort))
            return nullptr;
        auto it = outputByPort.find(portKey(srcNode, srcPort));
        return it == outputByPort.end() ? nullptr : it->second;
    };

    // A pass-by-value scalar tensor (dim {1}); embedded in the plan, never in the
    // variant pack. Used for epsilon on the norm ops.
    auto makeScalar = [&](double v, const std::string& name) -> TensorPtr {
        return Graph::tensor(TensorAttributes()
                                 .set_value(static_cast<float>(v))
                                 .set_name(name)
                                 .set_uid(++g_nextUid)
                                 .set_is_virtual(false));
    };

    std::unordered_map<std::string, bool> done;
    std::size_t remaining = 0;
    for(const auto& n : nodes)
    {
        if(n.at("type").get<std::string>() != "Output")
            ++remaining;
    }

    auto inputsReady = [&](const json& n) -> bool {
        if(n.at("type").get<std::string>() == "Input")
            return true;
        const std::string id = n.at("id").get<std::string>();
        for(const auto& e : edges)
        {
            if(e.at("to").at("node").get<std::string>() == id
               && !done.count(e.at("from").at("node").get<std::string>()))
            {
                return false;
            }
        }
        return true;
    };

    bool progress = true;
    while(remaining > 0 && progress)
    {
        progress = false;
        for(const auto& n : nodes)
        {
            const std::string id = n.at("id").get<std::string>();
            const std::string type = n.at("type").get<std::string>();
            if(type == "Output" || done.count(id) || !inputsReady(n))
                continue;

            const json params = n.value("params", json::object());
            const std::string title = n.value("title", type);
            auto reg = [&](const std::string& port, const TensorPtr& t) {
                if(t)
                    outputByPort[portKey(id, port)] = t;
            };

            if(type == "Input")
            {
                const std::vector<int64_t> dims = parseShape(params.value("shape", std::string()));
                if(dims.empty())
                {
                    throw BuildInputError{"INVALID_VALUE",
                                          "Input '" + title + "' has no valid shape."};
                }
                reg("out",
                    Graph::tensor(TensorAttributes()
                                      .set_dim(dims)
                                      .set_stride(rowMajorStrides(dims))
                                      .set_data_type(ioDtype)
                                      .set_name(title)
                                      .set_uid(++g_nextUid)
                                      .set_is_virtual(false)));
            }
            else if(type == "Pointwise")
            {
                PointwiseMode mode;
                if(!lookupEnum(kPointwiseModes, stringParam(params, "mode", "relu_fwd"), mode))
                {
                    throw BuildInputError{"INVALID_VALUE", "Unknown Pointwise mode."};
                }
                reg("y", g.pointwise(resolveInput(id, "x"), PointwiseAttributes().set_mode(mode)));
            }
            else if(type == "PointwiseBinary")
            {
                PointwiseMode mode;
                if(!lookupEnum(kPointwiseModes, stringParam(params, "mode", "add"), mode))
                {
                    throw BuildInputError{"INVALID_VALUE", "Unknown Pointwise mode."};
                }
                reg("y",
                    g.pointwise(resolveInput(id, "a"),
                                resolveInput(id, "b"),
                                PointwiseAttributes().set_mode(mode)));
            }
            else if(type == "Reduction")
            {
                ReductionMode mode;
                if(!lookupEnum(kReductionModes, stringParam(params, "mode", "add"), mode))
                {
                    throw BuildInputError{"INVALID_VALUE", "Unknown Reduction mode."};
                }
                TensorPtr y
                    = g.reduction(resolveInput(id, "x"),
                                  hipdnn_frontend::graph::ReductionAttributes().set_mode(mode));
                const std::vector<int64_t> yDims = parseShape(stringParam(params, "out_shape", ""));
                if(yDims.empty())
                {
                    throw BuildInputError{
                        "INVALID_VALUE",
                        "Reduction needs a valid 'Output shape' (the reduced dims)."};
                }
                y->set_dim(yDims).set_stride(rowMajorStrides(yDims));
                reg("y", y);
            }
            else if(type == "MatMul")
            {
                reg("y",
                    g.matmul(resolveInput(id, "a"), resolveInput(id, "b"), MatmulAttributes()));
            }
            else if(type == "ResampleFwd")
            {
                const int64_t w = intParam(params, "window", 2);
                const int64_t s = intParam(params, "stride", 2);
                const int64_t p = intParam(params, "padding", 0);
                auto outs = g.resample(resolveInput(id, "x"),
                                       ResampleFwdAttributes()
                                           .set_pre_padding({p, p})
                                           .set_post_padding({p, p})
                                           .set_stride({s, s})
                                           .set_window({w, w})
                                           .set_resample_mode(lookupEnumOr(
                                               kResampleModes,
                                               stringParam(params, "resample_mode", "MAXPOOL"),
                                               ResampleMode::MAXPOOL))
                                           .set_padding_mode(lookupEnumOr(
                                               kPaddingModes,
                                               stringParam(params, "padding_mode", "NEG_INF_PAD"),
                                               PaddingMode::NEG_INF_PAD)));
                reg("y", outs[0]);
            }
            else if(type == "ConvolutionFprop")
            {
                const int64_t pad = intParam(params, "padding", 1);
                const int64_t stride = intParam(params, "stride", 1);
                const int64_t dil = intParam(params, "dilation", 1);
                reg("y",
                    g.conv_fprop(resolveInput(id, "x"),
                                 resolveInput(id, "w"),
                                 ConvFpropAttributes()
                                     .set_pre_padding({pad, pad})
                                     .set_post_padding({pad, pad})
                                     .set_stride({stride, stride})
                                     .set_dilation({dil, dil})
                                     .set_convolution_mode(lookupEnumOr(
                                         kConvModes,
                                         stringParam(params, "conv_mode", "CROSS_CORRELATION"),
                                         ConvolutionMode::CROSS_CORRELATION))));
            }
            else if(type == "ConvolutionDgrad")
            {
                const int64_t pad = intParam(params, "padding", 1);
                const int64_t stride = intParam(params, "stride", 1);
                const int64_t dil = intParam(params, "dilation", 1);
                const std::vector<int64_t> dxDims = parseShape(stringParam(params, "dx_shape", ""));
                if(dxDims.empty())
                {
                    throw BuildInputError{
                        "INVALID_VALUE",
                        "Convolution Dgrad needs a valid 'dx shape' (the forward input)."};
                }
                TensorPtr dx
                    = g.conv_dgrad(resolveInput(id, "dy"),
                                   resolveInput(id, "w"),
                                   ConvDgradAttributes()
                                       .set_pre_padding({pad, pad})
                                       .set_post_padding({pad, pad})
                                       .set_stride({stride, stride})
                                       .set_dilation({dil, dil})
                                       .set_convolution_mode(lookupEnumOr(
                                           kConvModes,
                                           stringParam(params, "conv_mode", "CROSS_CORRELATION"),
                                           ConvolutionMode::CROSS_CORRELATION)));
                dx->set_dim(dxDims).set_stride(rowMajorStrides(dxDims));
                reg("dx", dx);
            }
            else if(type == "ConvolutionWgrad")
            {
                const int64_t pad = intParam(params, "padding", 1);
                const int64_t stride = intParam(params, "stride", 1);
                const int64_t dil = intParam(params, "dilation", 1);
                const std::vector<int64_t> dwDims = parseShape(stringParam(params, "dw_shape", ""));
                if(dwDims.empty())
                {
                    throw BuildInputError{
                        "INVALID_VALUE",
                        "Convolution Wgrad needs a valid 'dw shape' (the filter dims)."};
                }
                TensorPtr dw
                    = g.conv_wgrad(resolveInput(id, "dy"),
                                   resolveInput(id, "x"),
                                   ConvWgradAttributes()
                                       .set_pre_padding({pad, pad})
                                       .set_post_padding({pad, pad})
                                       .set_stride({stride, stride})
                                       .set_dilation({dil, dil})
                                       .set_convolution_mode(lookupEnumOr(
                                           kConvModes,
                                           stringParam(params, "conv_mode", "CROSS_CORRELATION"),
                                           ConvolutionMode::CROSS_CORRELATION)));
                dw->set_dim(dwDims).set_stride(rowMajorStrides(dwDims));
                reg("dw", dw);
            }
            else if(type == "BatchNormInference")
            {
                reg("y",
                    g.batchnorm_inference(resolveInput(id, "x"),
                                          resolveInput(id, "mean"),
                                          resolveInput(id, "inv_variance"),
                                          resolveInput(id, "scale"),
                                          resolveInput(id, "bias"),
                                          hipdnn_frontend::graph::BatchnormInferenceAttributes()));
            }
            else if(type == "BatchNormBackward")
            {
                auto outs
                    = g.batchnorm_backward(resolveInput(id, "dy"),
                                           resolveInput(id, "x"),
                                           resolveInput(id, "scale"),
                                           hipdnn_frontend::graph::BatchnormBackwardAttributes());
                reg("dx", outs[0]);
                reg("dscale", outs[1]);
                reg("dbias", outs[2]);
            }
            else if(type == "Layernorm")
            {
                LayernormAttributes attr;
                attr.set_forward_phase(
                        lookupEnumOr(kNormPhases,
                                     stringParam(params, "forward_phase", "training"),
                                     NormFwdPhase::TRAINING))
                    .set_epsilon(makeScalar(floatParam(params, "epsilon", 1e-5), title + "_eps"));
                auto outs = g.layernorm(resolveInput(id, "x"),
                                        resolveInput(id, "scale"),
                                        resolveInput(id, "bias"),
                                        attr);
                reg("y", outs[0]);
                reg("mean", outs[1]);
                reg("inv_variance", outs[2]);
            }
            else if(type == "LayernormBackward")
            {
                LayernormBackwardAttributes attr;
                TensorPtr mean = resolveInputOpt(id, "mean");
                TensorPtr invVar = resolveInputOpt(id, "inv_variance");
                if(mean)
                    attr.set_mean(mean);
                if(invVar)
                    attr.set_inv_variance(invVar);
                auto outs = g.layernorm_backward(
                    resolveInput(id, "dy"), resolveInput(id, "x"), resolveInput(id, "scale"), attr);
                reg("dx", outs[0]);
                reg("dscale", outs[1]);
                reg("dbias", outs[2]);
            }
            else if(type == "RMSNorm")
            {
                RMSNormAttributes attr;
                attr.set_forward_phase(
                        lookupEnumOr(kNormPhases,
                                     stringParam(params, "forward_phase", "training"),
                                     NormFwdPhase::TRAINING))
                    .set_epsilon(makeScalar(floatParam(params, "epsilon", 1e-5), title + "_eps"));
                TensorPtr bias = resolveInputOpt(id, "bias");
                if(bias)
                    attr.set_bias(bias);
                auto outs = g.rmsnorm(resolveInput(id, "x"), resolveInput(id, "scale"), attr);
                reg("y", outs[0]);
                reg("inv_rms", outs[1]);
            }
            else if(type == "RMSNormBackward")
            {
                RMSNormBackwardAttributes attr;
                attr.set_compute_dbias(boolParam(params, "compute_dbias", true));
                auto outs = g.rmsnorm_backward(resolveInput(id, "dy"),
                                               resolveInput(id, "x"),
                                               resolveInput(id, "scale"),
                                               resolveInput(id, "inv_rms"),
                                               attr);
                reg("dx", outs[0]);
                reg("dscale", outs[1]);
                reg("dbias", outs[2]);
            }
            else if(type == "BlockScaleQuantize")
            {
                auto outs = g.block_scale_quantize(
                    resolveInput(id, "x"),
                    BlockScaleQuantizeAttributes()
                        .set_block_size(static_cast<int32_t>(intParam(params, "block_size", 32)))
                        .set_axis(intParam(params, "axis", 1))
                        .set_transpose(boolParam(params, "transpose", false)));
                reg("y", outs[0]);
                reg("scale", outs[1]);
            }
            else if(type == "BlockScaleDequantize")
            {
                reg("y",
                    g.block_scale_dequantize(
                        resolveInput(id, "x"),
                        resolveInput(id, "scale"),
                        BlockScaleDequantizeAttributes()
                            .set_block_size(
                                static_cast<int32_t>(intParam(params, "block_size", 32)))
                            .set_is_negative_scale(boolParam(params, "is_negative_scale", false))));
            }
            else
            {
                throw BuildInputError{"INVALID_VALUE",
                                      "Operator '" + type + "' has no hipDNN mapping yet."};
            }

            done[id] = true;
            --remaining;
            progress = true;
        }
    }

    if(remaining > 0)
    {
        throw BuildInputError{"INVALID_VALUE", "Graph has a cycle or an unreachable node."};
    }

    // A produced port that no edge consumes is a terminal result (e.g. an Output
    // node's tensor, or dscale/dbias from a backward op). Mark every such port as
    // a graph output so it gets a UID and a device buffer at execute time.
    std::unordered_set<std::string> consumed;
    for(const auto& e : edges)
    {
        consumed.insert(portKey(e.at("from").at("node").get<std::string>(),
                                e.at("from").at("port").get<std::string>()));
    }

    bool sawOutput = false;
    for(const auto& n : nodes)
    {
        if(n.at("type").get<std::string>() != "Output")
            continue;
        const std::string id = n.at("id").get<std::string>();
        const json params = n.value("params", json::object());
        TensorPtr t = resolveInput(id, "in");
        t->set_output(true).set_uid(++g_nextUid);
        if(!boolParam(params, "use_defaults", true))
        {
            const std::vector<int64_t> dims = parseShape(stringParam(params, "shape", ""));
            if(dims.empty())
            {
                throw BuildInputError{"INVALID_VALUE",
                                      "Output '" + n.value("title", std::string("Output"))
                                          + "' has 'Use defaults' off but no valid shape."};
            }
            t->set_dim(dims).set_stride(rowMajorStrides(dims));
        }
        sawOutput = true;
    }
    for(const auto& kv : outputByPort)
    {
        if(consumed.count(kv.first))
            continue;
        kv.second->set_output(true).set_uid(++g_nextUid);
    }
    if(!sawOutput)
    {
        throw BuildInputError{"INVALID_VALUE",
                              "Graph has no Output node; add one so hipDNN has a result tensor."};
    }
}

} // namespace studio
