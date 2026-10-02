// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/ArgumentResolver.hpp>

#include <hipdnn_corpus_gen/GraphBuilders.hpp>

#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

/// @file GraphBuilderRegistry.hpp
/// @brief Name-to-adapter table for the builder a metadata file names (RFC 0019.13 §4.3.6).
/// An adapter uses only the arguments it was given; a missing one is a refusal, never a guess.

namespace hipdnn_corpus_gen
{

/// A built graph, as the serialized bytes a problem file carries.
using GraphBytes = std::vector<uint8_t>;

struct BuildResult
{
    GraphBytes bytes;
    std::string error;

    bool ok() const
    {
        return error.empty() && !bytes.empty();
    }
};

namespace detail
{

/// The FlatBuffers dtype a declared name denotes, or nullopt for an unknown name (never a
/// silent FLOAT default).
inline std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType>
    dataTypeFor(const std::string& name)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    // Defers to the SDK's name table. Only runtime spellings (`to_string(DataType)`) and these
    // aliases are accepted: numpy spellings like `float16` would put names in the corpus that
    // the runtime's categorical encoding (RFC 0019 §6.5) never emits.
    static const std::map<std::string, std::string> s_aliases{
        {"fp32", "float"},
        {"fp64", "double"},
        {"fp16", "half"},
        {"bf16", "bfloat16"},
    };

    const auto alias = s_aliases.find(name);
    const std::string canonical = alias == s_aliases.end() ? name : alias->second;

    // NLOHMANN_JSON_SERIALIZE_ENUM maps an unknown name to its first entry (UNSET);
    // the round trip turns that into a refusal.
    const auto type = nlohmann::json(canonical).get<DataType>();
    if(type == DataType::UNSET || nlohmann::json(type).get<std::string>() != canonical)
    {
        return std::nullopt;
    }
    return type;
}

/// Reads a resolved argument as a dims list, by name. Never by position: metadata argument
/// order need not match the builder's (§4.2's example declares dims before strides).
inline const std::vector<int64_t>* dims(const ArgumentResolution& resolved, const char* name)
{
    const auto* argument = resolved.find(name);
    return argument == nullptr ? nullptr : std::get_if<std::vector<int64_t>>(&argument->value);
}

inline std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType>
    dtype(const ArgumentResolution& resolved, const char* name)
{
    const auto* argument = resolved.find(name);
    if(argument == nullptr)
    {
        return std::nullopt;
    }
    const auto* declared = std::get_if<std::string>(&argument->value);
    return declared == nullptr ? std::nullopt : dataTypeFor(*declared);
}

/// Reads a resolved argument as a boolean; absent reads as false, as the builders default.
inline bool flag(const ArgumentResolution& resolved, const char* name)
{
    const auto* argument = resolved.find(name);
    if(argument == nullptr)
    {
        return false;
    }
    if(const auto* held = std::get_if<bool>(&argument->value))
    {
        return *held;
    }
    // A bool parameter may arrive as its declared spelling.
    const auto* text = std::get_if<std::string>(&argument->value);
    return text != nullptr && (*text == "true" || *text == "1");
}

inline std::string enumName(const ArgumentResolution& resolved, const char* name)
{
    const auto* argument = resolved.find(name);
    if(argument == nullptr)
    {
        return {};
    }
    const auto* declared = std::get_if<std::string>(&argument->value);
    return declared == nullptr ? std::string{} : *declared;
}

/// Reads the causal diagonal's anchor. Absent is TOP_LEFT (the schema default); an unknown
/// spelling is nullopt, since the anchors mask different triangles when seqlen_q < seqlen_k.
inline std::optional<hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment>
    diagonalAlignment(const ArgumentResolution& resolved, const char* name)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;

    const auto declared = enumName(resolved, name);
    if(declared.empty())
    {
        return DiagonalAlignment::TOP_LEFT;
    }
    if(declared == "top_left")
    {
        return DiagonalAlignment::TOP_LEFT;
    }
    if(declared == "bottom_right")
    {
        return DiagonalAlignment::BOTTOM_RIGHT;
    }
    return std::nullopt;
}

/// @brief Assembles one tensor role from `<role>Dims`, `<role>Strides` and an element type.
/// Undeclared dims or strides are a refusal, not a packed default.
inline std::optional<builders::TensorSpec>
    tensorRole(const ArgumentResolution& resolved,
               int64_t uid,
               const std::string& role,
               hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    const auto* d = dims(resolved, (role + "Dims").c_str());
    const auto* st = dims(resolved, (role + "Strides").c_str());
    if(d == nullptr || st == nullptr)
    {
        return std::nullopt;
    }
    builders::TensorSpec spec;
    spec.uid = uid;
    spec.name = role;
    spec.dims = *d;
    spec.strides = *st;
    // `<role>DataType` overrides the dtype for one tensor, e.g. fp32 saved norm statistics.
    spec.dataType = dtype(resolved, (role + "DataType").c_str()).value_or(type);
    // A declared `rank` pads with trailing unit dims (see padToDeclaredRank), row-major.
    const auto declared = enumName(resolved, "rank");
    if(!declared.empty() && spec.dims.size() < static_cast<size_t>(std::stoll(declared)))
    {
        spec.dims.resize(static_cast<size_t>(std::stoll(declared)), 1);
        spec.strides = rowMajorStrides(spec.dims);
    }
    return spec;
}

/// @brief The graph's element types: io as given, compute and intermediate as declared.
/// fp16/bf16 declarations need a FLOAT compute type: MIOpen's conv builder refuses any other.
inline builders::GraphTypes graphTypesFrom(const ArgumentResolution& resolved,
                                           hipdnn_flatbuffers_sdk::data_objects::DataType io)
{
    builders::GraphTypes types = builders::GraphTypes::uniform(io);
    types.compute = dtype(resolved, "computeDataType").value_or(io);
    types.intermediate = dtype(resolved, "intermediateDataType").value_or(io);
    return types;
}

inline builders::TensorSpec tensorFrom(int64_t uid,
                                       const char* name,
                                       const std::vector<int64_t>& dims,
                                       const std::vector<int64_t>& strides,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    builders::TensorSpec spec;
    spec.uid = uid;
    spec.name = name;
    spec.dims = dims;
    spec.strides = strides;
    spec.dataType = type;
    return spec;
}

/// A norm's scale/bias shape for @p x: x's rank, 1 everywhere but the trailing (normalized)
/// dimension. The frontend reads which dimensions are normalized off where scale is not 1.
inline std::vector<int64_t> normAffineDims(const std::vector<int64_t>& x)
{
    std::vector<int64_t> dims(x.size(), 1);
    if(!dims.empty())
    {
        dims.back() = x.back();
    }
    return dims;
}

/// A norm's epsilon: a pass-by-value fp32 scalar, which the frontend requires it to be.
inline builders::TensorSpec normEpsilon(int64_t uid)
{
    auto spec = tensorFrom(
        uid, "epsilon", {1}, {1}, hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
    spec.scalarValue = 1e-5F;
    return spec;
}

/// One batchnorm tensor. Activations carry the declared dtype; stats and affine tensors carry
/// `statsDataType` if declared (MIOpen and HIP_MLOPS take them only in fp32).
inline std::optional<builders::TensorSpec>
    batchnormRole(const ArgumentResolution& resolved,
                  int64_t uid,
                  const std::string& role,
                  hipdnn_flatbuffers_sdk::data_objects::DataType io)
{
    if(role == "epsilon")
    {
        return normEpsilon(uid);
    }
    const bool activation = role == "x" || role == "y" || role == "dy" || role == "dx";
    return tensorRole(
        resolved, uid, role, activation ? io : dtype(resolved, "statsDataType").value_or(io));
}

/// @p dims padded with trailing unit dims to the declared `rank`, if any: the same problem,
/// spelled for engines (HIP_MLOPS) that take only 4-D or 5-D tensors.
inline std::vector<int64_t> padToDeclaredRank(const ArgumentResolution& resolved,
                                              std::vector<int64_t> dims)
{
    const auto declared = enumName(resolved, "rank");
    if(!declared.empty())
    {
        const auto rank = static_cast<size_t>(std::stoll(declared));
        while(dims.size() < rank)
        {
            dims.push_back(1);
        }
    }
    return dims;
}

/// Normalized trailing-dim count after padding to the declared rank: the padded unit dims fall
/// inside the normalized span, so they are added to @p declaredCount.
inline int64_t paddedNormalizedDimCount(const ArgumentResolution& resolved,
                                        size_t declaredRank,
                                        int64_t declaredCount)
{
    const auto padded = padToDeclaredRank(resolved, std::vector<int64_t>(declaredRank, 1));
    return declaredCount + static_cast<int64_t>(padded.size() - declaredRank);
}

/// Reads a resolved argument as a floating-point scalar; absent reads as @p fallback.
inline double scalar(const ArgumentResolution& resolved, const char* name, double fallback = 0.0)
{
    const auto* argument = resolved.find(name);
    if(argument == nullptr)
    {
        return fallback;
    }
    if(const auto* held = std::get_if<double>(&argument->value))
    {
        return *held;
    }
    if(const auto* held = std::get_if<int64_t>(&argument->value))
    {
        return static_cast<double>(*held);
    }
    return fallback;
}

/// Reads a resolved argument as an integer; absent reads as @p fallback.
inline int64_t integer(const ArgumentResolution& resolved, const char* name, int64_t fallback = 0)
{
    const auto* argument = resolved.find(name);
    if(argument == nullptr)
    {
        return fallback;
    }
    if(const auto* held = std::get_if<int64_t>(&argument->value))
    {
        return *held;
    }
    if(const auto* held = std::get_if<std::vector<int64_t>>(&argument->value))
    {
        return held->empty() ? fallback : held->front();
    }
    return fallback;
}

/// A pointwise node's mode (by FlatBuffers enumerator name) and its activation scalars.
inline bool pointwiseModeAndScalars(const ArgumentResolution& resolved,
                                    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode& mode,
                                    builders::PointwiseScalars& scalars,
                                    std::string& error)
{
    const auto declared = enumName(resolved, "mode");
    if(!declared.empty())
    {
        const auto* names = hipdnn_flatbuffers_sdk::data_objects::EnumNamesPointwiseMode();
        bool matched = false;
        for(size_t i = 0; names[i] != nullptr; ++i)
        {
            if(declared == names[i])
            {
                mode = static_cast<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode>(i);
                matched = true;
            }
        }
        if(!matched)
        {
            error = "unknown pointwise mode '" + declared + "'";
            return false;
        }
    }
    // An absent argument leaves its field null, as the schema intends.
    const auto optional = [&resolved](const char* name) -> flatbuffers::Optional<float> {
        if(resolved.find(name) == nullptr)
        {
            return flatbuffers::nullopt;
        }
        return static_cast<float>(scalar(resolved, name));
    };
    scalars.reluLowerClip = optional("reluLowerClip");
    scalars.reluUpperClip = optional("reluUpperClip");
    scalars.reluLowerClipSlope = optional("reluLowerClipSlope");
    scalars.swishBeta = optional("swishBeta");
    scalars.eluAlpha = optional("eluAlpha");
    scalars.softplusBeta = optional("softplusBeta");
    return true;
}

/// The activation a fused graph ends with, when `fusion` declares one: read from `mode` like any
/// pointwise node. nullopt when `fusion` is absent or "none".
inline std::optional<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode>
    fusedActivation(const ArgumentResolution& resolved, std::string& error)
{
    const auto fusion = enumName(resolved, "fusion");
    if(fusion.empty() || fusion == "none")
    {
        return std::nullopt;
    }
    auto mode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD;
    builders::PointwiseScalars unused;
    if(!pointwiseModeAndScalars(resolved, mode, unused, error))
    {
        return std::nullopt;
    }
    return mode;
}

/// Resolves a declared enumerator name against a FlatBuffers EnumNames table; false if
/// unknown. MIN and MAX are reserved in the schemas, so the names are MIN_OP and MAX_OP.
template <typename Enum>
bool resolveEnum(const ArgumentResolution& resolved,
                 const char* name,
                 const char* const* table,
                 Enum& out)
{
    const auto declared = enumName(resolved, name);
    if(declared.empty())
    {
        return true; // not declared; the caller's default stands
    }
    for(size_t i = 0; table[i] != nullptr; ++i)
    {
        if(declared == table[i])
        {
            out = static_cast<Enum>(i);
            return true;
        }
    }
    return false;
}

inline GraphBytes toBytes(const flatbuffers::FlatBufferBuilder& builder)
{
    const auto* data = builder.GetBufferPointer();
    return {data, data + builder.GetSize()};
}

} // namespace detail

/// One adapter: resolved arguments in, serialized graph out.
using BuilderAdapter = std::function<BuildResult(const ArgumentResolution&)>;

/// @brief The builders a metadata file may name (§4.4 check 5 requires the name to resolve).
inline const std::map<std::string, BuilderAdapter>& builderRegistry()
{
    static const std::map<std::string, BuilderAdapter> s_registry{
        {"convolutionForward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* x = detail::dims(resolved, "xDims");
             const auto* xs = detail::dims(resolved, "xStrides");
             const auto* w = detail::dims(resolved, "wDims");
             const auto* ws = detail::dims(resolved, "wStrides");
             const auto* y = detail::dims(resolved, "yDims");
             const auto* ys = detail::dims(resolved, "yStrides");
             const auto* pre = detail::dims(resolved, "prePadding");
             const auto* post = detail::dims(resolved, "postPadding");
             const auto* stride = detail::dims(resolved, "convStrides");
             const auto* dil = detail::dims(resolved, "convDilation");
             const auto type = detail::dtype(resolved, "dataType");
             if(x == nullptr || xs == nullptr || w == nullptr || ws == nullptr || y == nullptr
                || ys == nullptr || pre == nullptr || post == nullptr || stride == nullptr
                || dil == nullptr || !type.has_value())
             {
                 return {{},
                         "convolutionForward needs xDims, xStrides, wDims, wStrides, yDims, "
                         "yStrides, prePadding, postPadding, convStrides, convDilation, "
                         "dataType"};
             }
             return {builders::convolutionForward(
                         detail::tensorFrom(1, "x", *x, *xs, *type),
                         detail::tensorFrom(2, "w", *w, *ws, *type),
                         detail::tensorFrom(3, "y", *y, *ys, *type),
                         builders::ConvGeometry{
                             *pre,
                             *post,
                             *stride,
                             *dil,
                             hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION},
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"convolutionBiasActivation",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* x = detail::dims(resolved, "xDims");
             const auto* xs = detail::dims(resolved, "xStrides");
             const auto* w = detail::dims(resolved, "wDims");
             const auto* ws = detail::dims(resolved, "wStrides");
             const auto* y = detail::dims(resolved, "yDims");
             const auto* ys = detail::dims(resolved, "yStrides");
             const auto* pre = detail::dims(resolved, "prePadding");
             const auto* post = detail::dims(resolved, "postPadding");
             const auto* stride = detail::dims(resolved, "convStrides");
             const auto* dil = detail::dims(resolved, "convDilation");
             const auto type = detail::dtype(resolved, "dataType");
             if(x == nullptr || xs == nullptr || w == nullptr || ws == nullptr || y == nullptr
                || ys == nullptr || pre == nullptr || post == nullptr || stride == nullptr
                || dil == nullptr || !type.has_value())
             {
                 return {{}, "convolutionBiasActivation needs the convolutionForward arguments"};
             }
             auto activation = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD;
             builders::PointwiseScalars unused;
             std::string error;
             if(!detail::pointwiseModeAndScalars(resolved, activation, unused, error))
             {
                 return {{}, error};
             }
             // `fusion` names the graph's shape: conv -> bias -> activation, or conv ->
             // activation. Only the first reads the bias tensor.
             std::optional<builders::TensorSpec> bias;
             if(detail::enumName(resolved, "fusion") != "activation")
             {
                 bias = detail::tensorRole(resolved, 3, "bias", *type);
                 if(!bias.has_value())
                 {
                     return {{}, "convolutionBiasActivation with a bias needs biasDims/Strides"};
                 }
             }
             return {builders::convolutionBiasActivation(
                         detail::tensorFrom(1, "x", *x, *xs, *type),
                         detail::tensorFrom(2, "w", *w, *ws, *type),
                         bias,
                         detail::tensorFrom(4, "y", *y, *ys, *type),
                         builders::ConvGeometry{
                             *pre,
                             *post,
                             *stride,
                             *dil,
                             hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION},
                         activation,
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"sdpaForward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* q = detail::dims(resolved, "qDims");
             const auto* qs = detail::dims(resolved, "qStrides");
             const auto* k = detail::dims(resolved, "kDims");
             const auto* ks = detail::dims(resolved, "kStrides");
             const auto* v = detail::dims(resolved, "vDims");
             const auto* vs = detail::dims(resolved, "vStrides");
             const auto* o = detail::dims(resolved, "oDims");
             const auto* os = detail::dims(resolved, "oStrides");
             const auto type = detail::dtype(resolved, "dataType");
             if(q == nullptr || qs == nullptr || k == nullptr || ks == nullptr || v == nullptr
                || vs == nullptr || o == nullptr || os == nullptr || !type.has_value())
             {
                 return {{},
                         "sdpaForward needs qDims/qStrides, kDims/kStrides, vDims/vStrides, "
                         "oDims/oStrides, dataType"};
             }
             const auto alignment = detail::diagonalAlignment(resolved, "diagonalAlignment");
             if(!alignment.has_value())
             {
                 return {{}, "sdpaForward: diagonalAlignment must be top_left or bottom_right"};
             }
             builders::SdpaOptions options;
             options.causalMask = detail::flag(resolved, "causalMask");
             options.paddingMask = detail::flag(resolved, "paddingMask");
             options.alibiMask = detail::flag(resolved, "alibiMask");
             options.generateStats = detail::flag(resolved, "generateStats");
             options.attnScale = static_cast<float>(detail::scalar(resolved, "attnScale"));
             options.dropoutProbability
                 = static_cast<float>(detail::scalar(resolved, "dropoutProbability"));
             options.leftBound = detail::integer(resolved, "leftBound", -1);
             options.rightBound = detail::integer(resolved, "rightBound", -1);
             options.diagonalAlignment = *alignment;
             return {builders::sdpaForward(detail::tensorFrom(1, "q", *q, *qs, *type),
                                           detail::tensorFrom(2, "k", *k, *ks, *type),
                                           detail::tensorFrom(3, "v", *v, *vs, *type),
                                           detail::tensorFrom(4, "o", *o, *os, *type),
                                           options,
                                           detail::graphTypesFrom(resolved, *type),
                                           options.generateStats
                                               ? detail::tensorRole(resolved, 5, "stats", *type)
                                               : std::nullopt),
                     ""};
         }},

        {"convolutionBackwardData",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* dy = detail::dims(resolved, "dyDims");
             const auto* dys = detail::dims(resolved, "dyStrides");
             const auto* w = detail::dims(resolved, "wDims");
             const auto* ws = detail::dims(resolved, "wStrides");
             const auto* dx = detail::dims(resolved, "dxDims");
             const auto* dxs = detail::dims(resolved, "dxStrides");
             const auto* pre = detail::dims(resolved, "prePadding");
             const auto* post = detail::dims(resolved, "postPadding");
             const auto* stride = detail::dims(resolved, "convStrides");
             const auto* dil = detail::dims(resolved, "convDilation");
             const auto type = detail::dtype(resolved, "dataType");
             if(dy == nullptr || dys == nullptr || w == nullptr || ws == nullptr || dx == nullptr
                || dxs == nullptr || pre == nullptr || post == nullptr || stride == nullptr
                || dil == nullptr || !type.has_value())
             {
                 return {{},
                         "convolutionBackwardData needs dyDims/dyStrides, wDims/wStrides, "
                         "dxDims/dxStrides, prePadding, postPadding, convStrides, "
                         "convDilation, dataType"};
             }
             return {builders::convolutionBackwardData(
                         detail::tensorFrom(1, "dy", *dy, *dys, *type),
                         detail::tensorFrom(2, "w", *w, *ws, *type),
                         detail::tensorFrom(3, "dx", *dx, *dxs, *type),
                         builders::ConvGeometry{
                             *pre,
                             *post,
                             *stride,
                             *dil,
                             hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION},
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"convolutionBackwardWeights",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* x = detail::dims(resolved, "xDims");
             const auto* xs = detail::dims(resolved, "xStrides");
             const auto* dy = detail::dims(resolved, "dyDims");
             const auto* dys = detail::dims(resolved, "dyStrides");
             const auto* dw = detail::dims(resolved, "dwDims");
             const auto* dws = detail::dims(resolved, "dwStrides");
             const auto* pre = detail::dims(resolved, "prePadding");
             const auto* post = detail::dims(resolved, "postPadding");
             const auto* stride = detail::dims(resolved, "convStrides");
             const auto* dil = detail::dims(resolved, "convDilation");
             const auto type = detail::dtype(resolved, "dataType");
             if(x == nullptr || xs == nullptr || dy == nullptr || dys == nullptr || dw == nullptr
                || dws == nullptr || pre == nullptr || post == nullptr || stride == nullptr
                || dil == nullptr || !type.has_value())
             {
                 return {{},
                         "convolutionBackwardWeights needs xDims/xStrides, dyDims/dyStrides, "
                         "dwDims/dwStrides, prePadding, postPadding, convStrides, "
                         "convDilation, dataType"};
             }
             return {builders::convolutionBackwardWeights(
                         detail::tensorFrom(1, "x", *x, *xs, *type),
                         detail::tensorFrom(2, "dy", *dy, *dys, *type),
                         detail::tensorFrom(3, "dw", *dw, *dws, *type),
                         builders::ConvGeometry{
                             *pre,
                             *post,
                             *stride,
                             *dil,
                             hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION},
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"resampleForward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* x = detail::dims(resolved, "xDims");
             const auto* xs = detail::dims(resolved, "xStrides");
             const auto* y = detail::dims(resolved, "yDims");
             const auto* ys = detail::dims(resolved, "yStrides");
             const auto* window = detail::dims(resolved, "window");
             const auto* stride = detail::dims(resolved, "poolStrides");
             const auto* pre = detail::dims(resolved, "prePadding");
             const auto* post = detail::dims(resolved, "postPadding");
             const auto type = detail::dtype(resolved, "dataType");
             if(x == nullptr || xs == nullptr || y == nullptr || ys == nullptr || window == nullptr
                || stride == nullptr || pre == nullptr || post == nullptr || !type.has_value())
             {
                 return {{},
                         "resampleForward needs xDims/xStrides, yDims/yStrides, window, "
                         "poolStrides, prePadding, postPadding, dataType"};
             }
             builders::ResampleGeometry geometry{
                 *window,
                 *stride,
                 *pre,
                 *post,
                 hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
                 hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD};
             if(!detail::resolveEnum(resolved,
                                     "resampleMode",
                                     hipdnn_flatbuffers_sdk::data_objects::EnumNamesResampleMode(),
                                     geometry.mode))
             {
                 return {{}, "unknown resample mode"};
             }
             if(!detail::resolveEnum(resolved,
                                     "paddingMode",
                                     hipdnn_flatbuffers_sdk::data_objects::EnumNamesPaddingMode(),
                                     geometry.paddingMode))
             {
                 return {{}, "unknown padding mode"};
             }
             // `index` = "yes" adds max pooling's index output, int32 and shaped like y.
             std::optional<builders::TensorSpec> index;
             if(detail::enumName(resolved, "index") == "yes")
             {
                 index = detail::tensorRole(
                     resolved, 3, "index", hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
                 if(!index)
                 {
                     return {{}, "resampleForward with an index needs indexDims/Strides"};
                 }
             }
             return {builders::resampleForward(detail::tensorFrom(1, "x", *x, *xs, *type),
                                               detail::tensorFrom(2, "y", *y, *ys, *type),
                                               geometry,
                                               detail::graphTypesFrom(resolved, *type),
                                               index),
                     ""};
         }},

        {"sdpaBackward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "sdpaBackward needs dataType"};
             }
             const std::vector<std::string> roles
                 = {"q", "k", "v", "o", "dO", "stats", "dq", "dk", "dv"};
             std::vector<builders::TensorSpec> t;
             for(size_t i = 0; i < roles.size(); ++i)
             {
                 auto one
                     = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{}, std::string("sdpaBackward needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             builders::SdpaOptions options;
             options.causalMask = detail::flag(resolved, "causalMask");
             options.paddingMask = detail::flag(resolved, "paddingMask");
             options.alibiMask = detail::flag(resolved, "alibiMask");
             options.attnScale = static_cast<float>(detail::scalar(resolved, "attnScale"));
             options.dropoutProbability
                 = static_cast<float>(detail::scalar(resolved, "dropoutProbability"));
             options.leftBound = detail::integer(resolved, "leftBound", -1);
             options.rightBound = detail::integer(resolved, "rightBound", -1);
             const auto alignment = detail::diagonalAlignment(resolved, "diagonalAlignment");
             if(!alignment.has_value())
             {
                 return {{}, "sdpaBackward: diagonalAlignment must be top_left or bottom_right"};
             }
             options.diagonalAlignment = *alignment;
             return {builders::sdpaBackward(t[0],
                                            t[1],
                                            t[2],
                                            t[3],
                                            t[4],
                                            t[5],
                                            t[6],
                                            t[7],
                                            t[8],
                                            options,
                                            detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"layernormBackward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "layernormBackward needs dataType"};
             }
             const std::vector<std::string> roles = {"dy", "x", "scale", "dx", "dscale", "dbias"};
             std::vector<builders::TensorSpec> t;
             for(size_t i = 0; i < roles.size(); ++i)
             {
                 auto one
                     = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("layernormBackward needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {builders::layernormBackward(
                         t[0],
                         t[1],
                         t[2],
                         t[3],
                         t[4],
                         t[5],
                         detail::paddedNormalizedDimCount(
                             resolved,
                             detail::dims(resolved, "xDims")->size(),
                             detail::integer(resolved, "normalizedDimCount", 1)),
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"rmsNormBackward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "rmsNormBackward needs dataType"};
             }
             const std::vector<std::string> roles = {"dy", "x", "scale", "invRms", "dx", "dscale"};
             std::vector<builders::TensorSpec> t;
             for(size_t i = 0; i < roles.size(); ++i)
             {
                 auto one
                     = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{}, std::string("rmsNormBackward needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {
                 builders::rmsNormBackward(
                     t[0], t[1], t[2], t[3], t[4], t[5], detail::graphTypesFrom(resolved, *type)),
                 ""};
         }},

        {"batchnormForwardTraining",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "batchnormForwardTraining needs dataType"};
             }
             const std::vector<std::string> roles
                 = {"x", "scale", "bias", "epsilon", "y", "mean", "invVariance"};
             std::vector<builders::TensorSpec> t;
             for(size_t i = 0; i < roles.size(); ++i)
             {
                 auto one = detail::batchnormRole(
                     resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("batchnormForwardTraining needs ") + roles[i]
                                 + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             std::string error;
             const auto activation = detail::fusedActivation(resolved, error);
             if(!error.empty())
             {
                 return {{}, error};
             }
             // Running statistics, when declared: four per-channel stats tensors and a
             // pass-by-value momentum.
             std::optional<builders::BatchnormRunningStats> running;
             if(detail::enumName(resolved, "runningStats") == "yes")
             {
                 const auto stat = [&](int64_t uid, const char* role) {
                     return detail::batchnormRole(resolved, uid, role, *type);
                 };
                 const auto pm = stat(8, "prevRunningMean");
                 const auto pv = stat(9, "prevRunningVariance");
                 const auto nm = stat(11, "nextRunningMean");
                 const auto nv = stat(12, "nextRunningVariance");
                 if(!pm || !pv || !nm || !nv)
                 {
                     return {{},
                             "batchnormForwardTraining with running statistics needs "
                             "prev/nextRunningMean and prev/nextRunningVariance Dims/Strides"};
                 }
                 auto momentum = detail::normEpsilon(10);
                 momentum.name = "momentum";
                 momentum.scalarValue = 0.1F;
                 running = builders::BatchnormRunningStats{*pm, *pv, momentum, *nm, *nv};
             }
             return {builders::batchnormForwardTraining(t[0],
                                                        t[1],
                                                        t[2],
                                                        t[3],
                                                        t[4],
                                                        t[5],
                                                        t[6],
                                                        detail::graphTypesFrom(resolved, *type),
                                                        running,
                                                        activation),
                     ""};
         }},

        {"batchnormInference",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "batchnormInference needs dataType"};
             }
             const std::vector<std::string> roles
                 = {"x", "mean", "invVariance", "scale", "bias", "y"};
             std::vector<builders::TensorSpec> t;
             for(size_t i = 0; i < roles.size(); ++i)
             {
                 auto one = detail::batchnormRole(
                     resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("batchnormInference needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             std::string error;
             const auto activation = detail::fusedActivation(resolved, error);
             if(!error.empty())
             {
                 return {{}, error};
             }
             return {builders::batchnormInference(t[0],
                                                  t[1],
                                                  t[2],
                                                  t[3],
                                                  t[4],
                                                  t[5],
                                                  detail::graphTypesFrom(resolved, *type),
                                                  activation),
                     ""};
         }},

        {"batchnormInferenceActivationBackward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "batchnormInferenceActivationBackward needs dataType"};
             }
             const std::vector<std::string> roles
                 = {"x", "mean", "invVariance", "scale", "bias", "dy", "dx", "dscale", "dbias"};
             std::vector<builders::TensorSpec> t;
             for(size_t i = 0; i < roles.size(); ++i)
             {
                 auto one = detail::batchnormRole(
                     resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             "batchnormInferenceActivationBackward needs " + roles[i]
                                 + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             auto mode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD;
             builders::PointwiseScalars unused;
             std::string error;
             if(!detail::pointwiseModeAndScalars(resolved, mode, unused, error))
             {
                 return {{}, error};
             }
             return {builders::batchnormInferenceActivationBackward(
                         t[0],
                         t[1],
                         t[2],
                         t[3],
                         t[4],
                         t[5],
                         t[6],
                         t[7],
                         t[8],
                         mode,
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"batchnormBackward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "batchnormBackward needs dataType"};
             }
             const std::vector<std::string> roles = {"dy", "x", "scale", "dx", "dscale", "dbias"};
             std::vector<builders::TensorSpec> t;
             for(size_t i = 0; i < roles.size(); ++i)
             {
                 auto one = detail::batchnormRole(
                     resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("batchnormBackward needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {
                 builders::batchnormBackward(
                     t[0], t[1], t[2], t[3], t[4], t[5], detail::graphTypesFrom(resolved, *type)),
                 ""};
         }},

        {"resampleBackward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             const auto* window = detail::dims(resolved, "window");
             const auto* stride = detail::dims(resolved, "poolStrides");
             const auto* pre = detail::dims(resolved, "prePadding");
             const auto* post = detail::dims(resolved, "postPadding");
             if(!type.has_value() || window == nullptr || stride == nullptr || pre == nullptr
                || post == nullptr)
             {
                 return {{},
                         "resampleBackward needs window, poolStrides, prePadding, "
                         "postPadding, dataType"};
             }
             const auto dy = detail::tensorRole(resolved, 1, "dy", *type);
             const auto dx = detail::tensorRole(resolved, 2, "dx", *type);
             if(!dy.has_value() || !dx.has_value())
             {
                 return {{}, "resampleBackward needs dyDims/dyStrides and dxDims/dxStrides"};
             }
             builders::ResampleGeometry geometry{
                 *window,
                 *stride,
                 *pre,
                 *post,
                 hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
                 hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD};
             if(!detail::resolveEnum(resolved,
                                     "resampleMode",
                                     hipdnn_flatbuffers_sdk::data_objects::EnumNamesResampleMode(),
                                     geometry.mode)
                || !detail::resolveEnum(
                    resolved,
                    "paddingMode",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamesPaddingMode(),
                    geometry.paddingMode))
             {
                 return {{}, "unknown resample or padding mode"};
             }
             std::optional<builders::TensorSpec> index;
             if(detail::enumName(resolved, "index") == "yes")
             {
                 index = detail::tensorRole(
                     resolved, 3, "index", hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
                 if(!index)
                 {
                     return {{}, "resampleBackward with an index needs indexDims/Strides"};
                 }
             }
             return {builders::resampleBackward(
                         *dy, *dx, geometry, detail::graphTypesFrom(resolved, *type), index),
                     ""};
         }},

        {"blockScaleQuantize",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             const auto scaleType = detail::dtype(resolved, "scaleDataType");
             if(!type.has_value())
             {
                 return {{}, "blockScaleQuantize needs dataType"};
             }
             const auto x = detail::tensorRole(resolved, 1, "x", *type);
             const auto y = detail::tensorRole(resolved, 2, "y", *type);
             // Scale has its own dtype: quantization exists to make it differ from the data's.
             const auto scale = detail::tensorRole(resolved, 3, "scale", scaleType.value_or(*type));
             if(!x.has_value() || !y.has_value() || !scale.has_value())
             {
                 return {{},
                         "blockScaleQuantize needs xDims/xStrides, yDims/yStrides, "
                         "scaleDims/scaleStrides"};
             }
             return {builders::blockScaleQuantize(
                         *x,
                         *y,
                         *scale,
                         static_cast<int32_t>(detail::integer(resolved, "blockSize", 32)),
                         detail::flag(resolved, "transpose"),
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"blockScaleDequantize",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             const auto scaleType = detail::dtype(resolved, "scaleDataType");
             if(!type.has_value())
             {
                 return {{}, "blockScaleDequantize needs dataType"};
             }
             const auto x = detail::tensorRole(resolved, 1, "x", *type);
             const auto scale = detail::tensorRole(resolved, 2, "scale", scaleType.value_or(*type));
             const auto y = detail::tensorRole(resolved, 3, "y", *type);
             if(!x.has_value() || !scale.has_value() || !y.has_value())
             {
                 return {{},
                         "blockScaleDequantize needs xDims/xStrides, scaleDims/scaleStrides, "
                         "yDims/yStrides"};
             }
             const auto* block = detail::dims(resolved, "blockSize");
             if(block == nullptr)
             {
                 return {{}, "blockScaleDequantize needs blockSize"};
             }
             std::vector<int32_t> blockSize;
             blockSize.reserve(block->size());
             for(const auto one : *block)
             {
                 blockSize.push_back(static_cast<int32_t>(one));
             }
             return {builders::blockScaleDequantize(*x,
                                                    *scale,
                                                    *y,
                                                    blockSize,
                                                    detail::flag(resolved, "negativeScale"),
                                                    detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"moeGroupedMatmul",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             const auto offsetType = detail::dtype(resolved, "offsetDataType");
             if(!type.has_value())
             {
                 return {{}, "moeGroupedMatmul needs dataType"};
             }
             const auto token = detail::tensorRole(resolved, 1, "token", *type);
             const auto weight = detail::tensorRole(resolved, 2, "weight", *type);
             // Offsets are indices, typed independently of the GEMM's element type.
             const auto offset = detail::tensorRole(
                 resolved,
                 3,
                 "firstTokenOffset",
                 offsetType.value_or(hipdnn_flatbuffers_sdk::data_objects::DataType::INT32));
             const auto output = detail::tensorRole(resolved, 4, "output", *type);
             if(!token.has_value() || !weight.has_value() || !offset.has_value()
                || !output.has_value())
             {
                 return {{},
                         "moeGroupedMatmul needs token, weight, firstTokenOffset and output "
                         "Dims/Strides"};
             }
             auto mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode::NONE;
             if(!detail::resolveEnum(
                    resolved,
                    "moeMode",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamesMoeGroupedMatmulMode(),
                    mode))
             {
                 return {{}, "unknown MoE grouped matmul mode"};
             }
             return {builders::moeGroupedMatmul(
                         *token,
                         *weight,
                         *offset,
                         *output,
                         mode,
                         static_cast<int32_t>(detail::integer(resolved, "topK", 1)),
                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"moeGroupedMatmulBackward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             const auto offsetType = detail::dtype(resolved, "offsetDataType");
             if(!type.has_value())
             {
                 return {{}, "moeGroupedMatmulBackward needs dataType"};
             }
             const auto dOutput = detail::tensorRole(resolved, 1, "dOutput", *type);
             const auto token = detail::tensorRole(resolved, 2, "token", *type);
             const auto offset = detail::tensorRole(
                 resolved,
                 3,
                 "firstTokenOffset",
                 offsetType.value_or(hipdnn_flatbuffers_sdk::data_objects::DataType::INT32));
             const auto dWeight = detail::tensorRole(resolved, 4, "dWeight", *type);
             if(!dOutput.has_value() || !token.has_value() || !offset.has_value()
                || !dWeight.has_value())
             {
                 return {{},
                         "moeGroupedMatmulBackward needs dOutput, token, firstTokenOffset "
                         "and dWeight Dims/Strides"};
             }
             return {
                 builders::moeGroupedMatmulBackward(
                     *dOutput, *token, *offset, *dWeight, detail::graphTypesFrom(resolved, *type)),
                 ""};
         }},

        {"matmul",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* a = detail::dims(resolved, "aDims");
             const auto* as = detail::dims(resolved, "aStrides");
             const auto* b = detail::dims(resolved, "bDims");
             const auto* bs = detail::dims(resolved, "bStrides");
             const auto* c = detail::dims(resolved, "cDims");
             const auto* cs = detail::dims(resolved, "cStrides");
             const auto type = detail::dtype(resolved, "dataType");
             if(a == nullptr || as == nullptr || b == nullptr || bs == nullptr || c == nullptr
                || cs == nullptr || !type.has_value())
             {
                 return {{},
                         "matmul needs aDims, aStrides, bDims, bStrides, cDims, cStrides, "
                         "dataType"};
             }
             return {builders::matmul(detail::tensorFrom(1, "a", *a, *as, *type),
                                      detail::tensorFrom(2, "b", *b, *bs, *type),
                                      detail::tensorFrom(3, "c", *c, *cs, *type),
                                      detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"matmulEpilogue",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto type = detail::dtype(resolved, "dataType");
             if(!type.has_value())
             {
                 return {{}, "matmulEpilogue needs dataType"};
             }
             const auto a = detail::tensorRole(resolved, 1, "a", *type);
             const auto b = detail::tensorRole(resolved, 2, "b", *type);
             const auto c = detail::tensorRole(resolved, 3, "c", *type);
             if(!a || !b || !c)
             {
                 return {{}, "matmulEpilogue needs a, b, c Dims/Strides"};
             }
             // `epilogue` names what follows the matmul: bias, activation, or bias_activation.
             const auto epilogue = detail::enumName(resolved, "epilogue");
             std::optional<builders::TensorSpec> bias;
             std::optional<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode> activation;
             if(epilogue != "activation")
             {
                 bias = detail::tensorRole(resolved, 4, "bias", *type);
                 if(!bias)
                 {
                     return {{}, "matmulEpilogue with a bias needs biasDims/Strides"};
                 }
             }
             if(epilogue != "bias")
             {
                 auto mode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD;
                 builders::PointwiseScalars unused;
                 std::string error;
                 if(!detail::pointwiseModeAndScalars(resolved, mode, unused, error))
                 {
                     return {{}, error};
                 }
                 activation = mode;
             }
             return {builders::matmulEpilogue(
                         *a, *b, bias, activation, *c, detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"blockScaledMatmul",
         [](const ArgumentResolution& resolved) -> BuildResult {
             // The operands carry the declared MX dtype, the scales `scaleDataType`, and the
             // result `outputDataType`; each is a separate declared fact.
             const auto type = detail::dtype(resolved, "dataType");
             const auto scaleType = detail::dtype(resolved, "scaleDataType");
             const auto outType = detail::dtype(resolved, "outputDataType");
             const auto block = detail::integer(resolved, "blockSize", 0);
             if(!type.has_value() || !scaleType.has_value() || !outType.has_value() || block <= 0)
             {
                 return {{},
                         "blockScaledMatmul needs dataType, scaleDataType, outputDataType, "
                         "blockSize"};
             }
             const auto a = detail::tensorRole(resolved, 1, "a", *type);
             const auto aScale = detail::tensorRole(resolved, 2, "aScale", *scaleType);
             const auto b = detail::tensorRole(resolved, 3, "b", *type);
             const auto bScale = detail::tensorRole(resolved, 4, "bScale", *scaleType);
             const auto c = detail::tensorRole(resolved, 5, "c", *outType);
             if(!a || !aScale || !b || !bScale || !c)
             {
                 return {{}, "blockScaledMatmul needs a, aScale, b, bScale, c Dims/Strides"};
             }
             return {builders::blockScaledMatmul(*a,
                                                 *aScale,
                                                 *b,
                                                 *bScale,
                                                 *c,
                                                 {static_cast<int32_t>(block)},
                                                 detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"pointwiseBinary",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* d = detail::dims(resolved, "dims");
             const auto* st = detail::dims(resolved, "strides");
             const auto type = detail::dtype(resolved, "dataType");
             if(d == nullptr || st == nullptr || !type.has_value())
             {
                 return {{}, "pointwiseBinary needs dims, strides, mode, dataType"};
             }
             auto mode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD;
             builders::PointwiseScalars scalars;
             std::string error;
             if(!detail::pointwiseModeAndScalars(resolved, mode, scalars, error))
             {
                 return {{}, error};
             }
             return {builders::pointwiseBinary(detail::tensorFrom(1, "in_0", *d, *st, *type),
                                               detail::tensorFrom(2, "in_1", *d, *st, *type),
                                               detail::tensorFrom(3, "out_0", *d, *st, *type),
                                               mode,
                                               scalars,
                                               detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"pointwiseUnary",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* d = detail::dims(resolved, "dims");
             const auto* st = detail::dims(resolved, "strides");
             const auto type = detail::dtype(resolved, "dataType");
             if(d == nullptr || st == nullptr || !type.has_value())
             {
                 return {{}, "pointwiseUnary needs dims, strides, mode, dataType"};
             }
             auto mode = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY;
             builders::PointwiseScalars scalars;
             std::string error;
             if(!detail::pointwiseModeAndScalars(resolved, mode, scalars, error))
             {
                 return {{}, error};
             }
             return {builders::pointwiseUnary(detail::tensorFrom(1, "in_0", *d, *st, *type),
                                              detail::tensorFrom(2, "out_0", *d, *st, *type),
                                              mode,
                                              scalars,
                                              detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},

        {"layernormForward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* d = detail::dims(resolved, "dims");
             const auto* st = detail::dims(resolved, "strides");
             const auto type = detail::dtype(resolved, "dataType");
             if(d == nullptr || st == nullptr || !type.has_value())
             {
                 return {{}, "layernormForward needs dims, strides, dataType"};
             }
             // Scale and bias: x's rank, 1 except the normalized trailing dim (normAffineDims).
             const auto x = detail::padToDeclaredRank(resolved, *d);
             const auto xStrides = x.size() == d->size() ? *st : detail::rowMajorStrides(x);
             const auto affine = detail::padToDeclaredRank(resolved, detail::normAffineDims(*d));
             return {
                 builders::layernormForward(
                     detail::tensorFrom(1, "x", x, xStrides, *type),
                     detail::tensorFrom(2, "scale", affine, detail::rowMajorStrides(affine), *type),
                     detail::tensorFrom(3, "bias", affine, detail::rowMajorStrides(affine), *type),
                     detail::normEpsilon(4),
                     detail::tensorFrom(5, "y", x, xStrides, *type),
                     detail::paddedNormalizedDimCount(resolved, d->size(), 1),
                     hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase::INFERENCE,
                     detail::graphTypesFrom(resolved, *type)),
                 ""};
         }},

        {"rmsNormForward",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* d = detail::dims(resolved, "dims");
             const auto* st = detail::dims(resolved, "strides");
             const auto type = detail::dtype(resolved, "dataType");
             if(d == nullptr || st == nullptr || !type.has_value())
             {
                 return {{}, "rmsNormForward needs dims, strides, dataType"};
             }
             const auto x = detail::padToDeclaredRank(resolved, *d);
             const auto xStrides = x.size() == d->size() ? *st : detail::rowMajorStrides(x);
             const auto affine = detail::padToDeclaredRank(resolved, detail::normAffineDims(*d));
             return {
                 builders::rmsNormForward(
                     detail::tensorFrom(1, "x", x, xStrides, *type),
                     detail::tensorFrom(2, "scale", affine, detail::rowMajorStrides(affine), *type),
                     detail::normEpsilon(3),
                     detail::tensorFrom(4, "y", x, xStrides, *type),
                     hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase::INFERENCE,
                     detail::graphTypesFrom(resolved, *type)),
                 ""};
         }},

        {"reduction",
         [](const ArgumentResolution& resolved) -> BuildResult {
             const auto* in = detail::dims(resolved, "inDims");
             const auto* ins = detail::dims(resolved, "inStrides");
             const auto* out = detail::dims(resolved, "outDims");
             const auto* outs = detail::dims(resolved, "outStrides");
             const auto type = detail::dtype(resolved, "dataType");
             if(in == nullptr || ins == nullptr || out == nullptr || outs == nullptr
                || !type.has_value())
             {
                 return {{},
                         "reduction needs inDims, inStrides, outDims, outStrides, mode, "
                         "dataType"};
             }
             auto mode = hipdnn_flatbuffers_sdk::data_objects::ReductionMode::ADD;
             const auto declared = detail::enumName(resolved, "mode");
             if(!declared.empty())
             {
                 const auto* names = hipdnn_flatbuffers_sdk::data_objects::EnumNamesReductionMode();
                 bool matched = false;
                 for(size_t i = 0; names[i] != nullptr; ++i)
                 {
                     if(declared == names[i])
                     {
                         mode = static_cast<hipdnn_flatbuffers_sdk::data_objects::ReductionMode>(i);
                         matched = true;
                     }
                 }
                 if(!matched)
                 {
                     return {{}, "unknown reduction mode '" + declared + "'"};
                 }
             }
             return {builders::reduction(detail::tensorFrom(1, "in", *in, *ins, *type),
                                         detail::tensorFrom(2, "out", *out, *outs, *type),
                                         mode,
                                         /*deterministic=*/false,
                                         detail::graphTypesFrom(resolved, *type)),
                     ""};
         }},
    };
    return s_registry;
}

/// @brief Builds the graph @p metadata describes for @p point: resolves the declared
/// arguments, then calls the named builder. Either failure is reported in the error.
inline BuildResult buildGraphFor(const OperationMetadata& metadata, const ProblemPoint& point)
{
    const auto& registry = builderRegistry();
    const auto adapter = registry.find(metadata.graphBuilder.function);
    if(adapter == registry.end())
    {
        // §4.4 check 5: the metadata names a builder nobody registered.
        return {{}, "no builder registered for '" + metadata.graphBuilder.function + "'"};
    }

    const auto resolved = resolveArguments(metadata.graphBuilder, point);
    if(!resolved.ok())
    {
        return {{}, resolved.error};
    }
    return adapter->second(resolved);
}

/// Builder names this generator can call, for reporting what a metadata file may name.
inline std::vector<std::string> registeredBuilders()
{
    std::vector<std::string> names;
    for(const auto& entry : builderRegistry())
    {
        names.push_back(entry.first);
    }
    return names;
}

} // namespace hipdnn_corpus_gen
