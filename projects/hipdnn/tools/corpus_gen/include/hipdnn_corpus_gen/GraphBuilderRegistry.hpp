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
/// @brief Calling the builder a metadata file names (RFC 0019.13 §4.3.6).
///
/// Dispatch is by `graph_builder.function`, so an operation is added by writing a metadata
/// file rather than by changing the generator. What the generator needs is a way to call a
/// C++ function whose name arrives as a string, which C++ cannot do by itself: hence a table.
///
/// Each entry adapts resolved arguments to a builder in GraphBuilders.hpp, which this tool
/// owns. §4.3.6 points instead at the test SDK's fixtures; that section was written against
/// their inventory rather than their contract, and the difference is load-bearing -- see the
/// file comment on GraphBuilders.hpp. An adapter that guessed at a missing argument would be
/// the same failure the resolver refuses: a graph that builds, benchmarks, and describes
/// something other than what the row says. So an adapter takes what it was given or declines.

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

/// The FlatBuffers dtype a declared name denotes, or nullopt for a name nobody defined.
///
/// Declining is deliberate: a dtype silently defaulted to FLOAT would make every problem
/// single-precision while the corpus recorded whatever the metadata claimed -- the exact
/// failure that made an earlier version of this generator unable to reach an engine's
/// half-precision kernels.
inline std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType>
    dataTypeFor(const std::string& name)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;

    // Resolved against the SDK's own name table rather than a copy of it. An earlier copy here
    // listed ten of the eighteen types the backend accepts, so a declaration naming fp4_e2m1 or
    // any fnuz variant failed to resolve -- and failed the same way a typo does, which is not
    // how "hipDNN cannot express this" should read. Deferring to the shared table also means a
    // dtype added to the schema is nameable here without anyone remembering to add it twice.
    static const std::map<std::string, std::string> s_aliases{
        {"float32", "float"},
        {"fp32", "float"},
        {"float64", "double"},
        {"fp64", "double"},
        {"float16", "half"},
        {"fp16", "half"},
        {"bf16", "bfloat16"},
    };

    const auto alias = s_aliases.find(name);
    const std::string canonical = alias == s_aliases.end() ? name : alias->second;

    // NLOHMANN_JSON_SERIALIZE_ENUM answers with its first entry for a name it does not know,
    // so an unknown dtype would silently become UNSET and produce tensors with no type at all.
    // Converting back is what turns that into a refusal.
    const auto type = nlohmann::json(canonical).get<DataType>();
    if(type == DataType::UNSET || nlohmann::json(type).get<std::string>() != canonical)
    {
        return std::nullopt;
    }
    return type;
}

/// Reads a resolved argument as a dims list, by the name the metadata gave it.
///
/// By name rather than by position, and the difference is not cosmetic. RFC 0019.13 §4.2's own
/// worked example declares `dims` before `strides`, while the builder it names takes
/// `(strides, dims, ...)`. Positional dispatch would swap them and produce a tensor whose
/// extents and strides are exchanged -- which still builds, still benchmarks, and describes a
/// different problem. Names make the metadata's ordering its own business.
inline const std::vector<int64_t>* dims(const ArgumentResolution& resolved, const char* name)
{
    const auto* argument = resolved.find(name);
    return argument == nullptr ? nullptr
                               : std::get_if<std::vector<int64_t>>(&argument->value);
}

/// Reads a resolved argument as a dtype, by name.
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

/// Reads a resolved argument as a boolean, by name. Absent reads as false, which matches the
/// builders' own defaults for their optional flags.
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

/// Reads a resolved argument as a declared enum name, by name.
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

/// Builds a tensor spec from a resolved dims/strides pair.
/// @brief Assembles one tensor role from `<role>Dims`, `<role>Strides` and an element type.
///
/// Every builder below takes whole tensors, so the arguments come in threes. Naming the role
/// once, rather than spelling out three lookups per tensor, is what keeps a nine-tensor
/// operation like SDPA backward readable -- and it makes an undeclared stride a refusal instead
/// of a silently packed default, which would quietly change the layout the corpus claims.
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
    spec.dataType = type;
    return spec;
}

/// @brief The graph's three element types: io as given, compute and intermediate as declared.
///
/// Defaulting all three to the io type describes a graph that accumulates in its storage type,
/// and for fp16 or bf16 that is not what anyone runs -- mixed precision means fp16 operands with
/// fp32 accumulation. It is also not merely unrealistic: MIOpen's convolution builder declines
/// any node whose `compute_data_type` is not FLOAT, so a uniform-fp16 graph is refused outright
/// and the corpus reports an engine that serves no half precision at all.
inline builders::GraphTypes
    graphTypesFrom(const ArgumentResolution& resolved,
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

/// Resolves a declared enumerator name against a FlatBuffers EnumNames table.
///
/// Refused rather than defaulted when the name is unknown: a corpus row claiming MUL while the
/// graph adds is wrong in a way no column reveals. Note MIN and MAX are reserved words in the
/// schemas, so the enumerators are MIN_OP and MAX_OP.
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

/// @brief The builders a metadata file may name.
///
/// Every entry is positional over the declared `arguments`, so a metadata file's argument
/// order is part of its contract with the builder it names. That is checkable -- the adapter
/// refuses a resolution that does not supply what it reads -- and it is why §4.4 check 5
/// requires `graph_builder.function` to resolve.
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
                 return {{}, "convolutionForward needs xDims, xStrides, wDims, wStrides, yDims, "
                             "yStrides, prePadding, postPadding, convStrides, convDilation, "
                             "dataType"};
             }
             return {builders::convolutionForward(
                         detail::tensorFrom(1, "x", *x, *xs, *type),
                         detail::tensorFrom(2, "w", *w, *ws, *type),
                         detail::tensorFrom(3, "y", *y, *ys, *type),
                         builders::ConvGeometry{*pre, *post, *stride, *dil,
                             hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION},
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
                 return {{}, "sdpaForward needs qDims/qStrides, kDims/kStrides, vDims/vStrides, "
                             "oDims/oStrides, dataType"};
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
             return {builders::sdpaForward(detail::tensorFrom(1, "q", *q, *qs, *type),
                                           detail::tensorFrom(2, "k", *k, *ks, *type),
                                           detail::tensorFrom(3, "v", *v, *vs, *type),
                                           detail::tensorFrom(4, "o", *o, *os, *type),
                                           options,
                                           detail::graphTypesFrom(resolved, *type)),
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
                 return {{}, "convolutionBackwardData needs dyDims/dyStrides, wDims/wStrides, "
                             "dxDims/dxStrides, prePadding, postPadding, convStrides, "
                             "convDilation, dataType"};
             }
             return {builders::convolutionBackwardData(
                         detail::tensorFrom(1, "dy", *dy, *dys, *type),
                         detail::tensorFrom(2, "w", *w, *ws, *type),
                         detail::tensorFrom(3, "dx", *dx, *dxs, *type),
                         builders::ConvGeometry{*pre, *post, *stride, *dil,
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
                 return {{}, "convolutionBackwardWeights needs xDims/xStrides, dyDims/dyStrides, "
                             "dwDims/dwStrides, prePadding, postPadding, convStrides, "
                             "convDilation, dataType"};
             }
             return {builders::convolutionBackwardWeights(
                         detail::tensorFrom(1, "x", *x, *xs, *type),
                         detail::tensorFrom(2, "dy", *dy, *dys, *type),
                         detail::tensorFrom(3, "dw", *dw, *dws, *type),
                         builders::ConvGeometry{*pre, *post, *stride, *dil,
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
             if(x == nullptr || xs == nullptr || y == nullptr || ys == nullptr
                || window == nullptr || stride == nullptr || pre == nullptr || post == nullptr
                || !type.has_value())
             {
                 return {{}, "resampleForward needs xDims/xStrides, yDims/yStrides, window, "
                             "poolStrides, prePadding, postPadding, dataType"};
             }
             builders::ResampleGeometry geometry{*window, *stride, *pre, *post,
                 hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
                 hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD};
             if(!detail::resolveEnum(
                    resolved, "resampleMode",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamesResampleMode(), geometry.mode))
             {
                 return {{}, "unknown resample mode"};
             }
             if(!detail::resolveEnum(
                    resolved, "paddingMode",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamesPaddingMode(),
                    geometry.paddingMode))
             {
                 return {{}, "unknown padding mode"};
             }
             return {builders::resampleForward(detail::tensorFrom(1, "x", *x, *xs, *type),
                                               detail::tensorFrom(2, "y", *y, *ys, *type),
                                               geometry,
                                               detail::graphTypesFrom(resolved, *type)),
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
                 auto one = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
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
             return {builders::sdpaBackward(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8],
                                            options, detail::graphTypesFrom(resolved, *type)),
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
                 auto one = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("layernormBackward needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {builders::layernormBackward(t[0], t[1], t[2], t[3], t[4], t[5],
                                                 detail::integer(resolved, "normalizedDimCount", 1),
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
                 auto one = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("rmsNormBackward needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {builders::rmsNormBackward(t[0], t[1], t[2], t[3], t[4], t[5],
                                               detail::graphTypesFrom(resolved, *type)),
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
                 auto one = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{}, std::string("batchnormForwardTraining needs ") + roles[i]
                                     + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {builders::batchnormForwardTraining(t[0], t[1], t[2], t[3], t[4], t[5], t[6],
                                                        detail::graphTypesFrom(resolved, *type)),
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
                 auto one = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("batchnormInference needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {builders::batchnormInference(t[0], t[1], t[2], t[3], t[4], t[5],
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
                 auto one = detail::tensorRole(resolved, static_cast<int64_t>(i) + 1, roles[i], *type);
                 if(!one.has_value())
                 {
                     return {{},
                             std::string("batchnormBackward needs ") + roles[i] + "Dims/Strides"};
                 }
                 t.push_back(*one);
             }
             return {builders::batchnormBackward(t[0], t[1], t[2], t[3], t[4], t[5],
                                                 detail::graphTypesFrom(resolved, *type)),
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
                 return {{}, "resampleBackward needs window, poolStrides, prePadding, "
                             "postPadding, dataType"};
             }
             const auto dy = detail::tensorRole(resolved, 1, "dy", *type);
             const auto dx = detail::tensorRole(resolved, 2, "dx", *type);
             if(!dy.has_value() || !dx.has_value())
             {
                 return {{}, "resampleBackward needs dyDims/dyStrides and dxDims/dxStrides"};
             }
             builders::ResampleGeometry geometry{*window, *stride, *pre, *post,
                 hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL,
                 hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD};
             if(!detail::resolveEnum(
                    resolved, "resampleMode",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamesResampleMode(), geometry.mode)
                || !detail::resolveEnum(
                    resolved, "paddingMode",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamesPaddingMode(),
                    geometry.paddingMode))
             {
                 return {{}, "unknown resample or padding mode"};
             }
             return {builders::resampleBackward(*dy, *dx, geometry,
                                                detail::graphTypesFrom(resolved, *type)),
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
             // The scale tensor carries its own element type; quantization exists precisely to
             // make it differ from the data's, so defaulting it to dataType would describe a
             // problem nobody asked for.
             const auto scale
                 = detail::tensorRole(resolved, 3, "scale", scaleType.value_or(*type));
             if(!x.has_value() || !y.has_value() || !scale.has_value())
             {
                 return {{}, "blockScaleQuantize needs xDims/xStrides, yDims/yStrides, "
                             "scaleDims/scaleStrides"};
             }
             return {builders::blockScaleQuantize(
                         *x, *y, *scale,
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
             const auto scale
                 = detail::tensorRole(resolved, 2, "scale", scaleType.value_or(*type));
             const auto y = detail::tensorRole(resolved, 3, "y", *type);
             if(!x.has_value() || !scale.has_value() || !y.has_value())
             {
                 return {{}, "blockScaleDequantize needs xDims/xStrides, scaleDims/scaleStrides, "
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
             return {builders::blockScaleDequantize(*x, *scale, *y, blockSize,
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
             // Offsets are indices, not activations: they are integer-typed independently of
             // the GEMM's element type.
             const auto offset = detail::tensorRole(
                 resolved, 3, "firstTokenOffset",
                 offsetType.value_or(hipdnn_flatbuffers_sdk::data_objects::DataType::INT32));
             const auto output = detail::tensorRole(resolved, 4, "output", *type);
             if(!token.has_value() || !weight.has_value() || !offset.has_value()
                || !output.has_value())
             {
                 return {{}, "moeGroupedMatmul needs token, weight, firstTokenOffset and output "
                             "Dims/Strides"};
             }
             auto mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode::NONE;
             if(!detail::resolveEnum(
                    resolved, "moeMode",
                    hipdnn_flatbuffers_sdk::data_objects::EnumNamesMoeGroupedMatmulMode(), mode))
             {
                 return {{}, "unknown MoE grouped matmul mode"};
             }
             return {builders::moeGroupedMatmul(
                         *token, *weight, *offset, *output, mode,
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
                 resolved, 3, "firstTokenOffset",
                 offsetType.value_or(hipdnn_flatbuffers_sdk::data_objects::DataType::INT32));
             const auto dWeight = detail::tensorRole(resolved, 4, "dWeight", *type);
             if(!dOutput.has_value() || !token.has_value() || !offset.has_value()
                || !dWeight.has_value())
             {
                 return {{}, "moeGroupedMatmulBackward needs dOutput, token, firstTokenOffset "
                             "and dWeight Dims/Strides"};
             }
             return {builders::moeGroupedMatmulBackward(*dOutput, *token, *offset, *dWeight,
                                                        detail::graphTypesFrom(resolved, *type)),
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
                 return {{}, "matmul needs aDims, aStrides, bDims, bStrides, cDims, cStrides, "
                             "dataType"};
             }
             return {builders::matmul(detail::tensorFrom(1, "a", *a, *as, *type),
                                      detail::tensorFrom(2, "b", *b, *bs, *type),
                                      detail::tensorFrom(3, "c", *c, *cs, *type),
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
             const auto declared = detail::enumName(resolved, "mode");
             if(!declared.empty())
             {
                 const auto* names
                     = hipdnn_flatbuffers_sdk::data_objects::EnumNamesPointwiseMode();
                 bool matched = false;
                 for(size_t i = 0; names[i] != nullptr; ++i)
                 {
                     if(declared == names[i])
                     {
                         mode = static_cast<
                             hipdnn_flatbuffers_sdk::data_objects::PointwiseMode>(i);
                         matched = true;
                     }
                 }
                 if(!matched)
                 {
                     return {{}, "unknown pointwise mode '" + declared + "'"};
                 }
             }
             builders::PointwiseScalars scalars;
             scalars.reluLowerClip
                 = static_cast<float>(detail::scalar(resolved, "reluLowerClip"));
             scalars.reluUpperClip
                 = static_cast<float>(detail::scalar(resolved, "reluUpperClip"));
             scalars.reluLowerClipSlope
                 = static_cast<float>(detail::scalar(resolved, "reluLowerClipSlope"));
             scalars.swishBeta = static_cast<float>(detail::scalar(resolved, "swishBeta"));
             scalars.eluAlpha = static_cast<float>(detail::scalar(resolved, "eluAlpha"));
             scalars.softplusBeta
                 = static_cast<float>(detail::scalar(resolved, "softplusBeta"));
             return {builders::pointwiseBinary(detail::tensorFrom(1, "in_0", *d, *st, *type),
                                               detail::tensorFrom(2, "in_1", *d, *st, *type),
                                               detail::tensorFrom(3, "out_0", *d, *st, *type),
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
             // Scale and bias span the normalized trailing dimension; epsilon is a scalar.
             const std::vector<int64_t> perChannel{d->back()};
             const std::vector<int64_t> unitStride{1};
             const std::vector<int64_t> scalar{1};
             return {builders::layernormForward(
                         detail::tensorFrom(1, "x", *d, *st, *type),
                         detail::tensorFrom(2, "scale", perChannel, unitStride, *type),
                         detail::tensorFrom(3, "bias", perChannel, unitStride, *type),
                         detail::tensorFrom(4, "epsilon", scalar, unitStride,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT),
                         detail::tensorFrom(5, "y", *d, *st, *type),
                         /*normalizedDimCount=*/1,
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
             const std::vector<int64_t> perChannel{d->back()};
             const std::vector<int64_t> unitStride{1};
             const std::vector<int64_t> scalar{1};
             return {builders::rmsNormForward(
                         detail::tensorFrom(1, "x", *d, *st, *type),
                         detail::tensorFrom(2, "scale", perChannel, unitStride, *type),
                         detail::tensorFrom(3, "epsilon", scalar, unitStride,
                                            hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT),
                         detail::tensorFrom(4, "y", *d, *st, *type),
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
                 return {{}, "reduction needs inDims, inStrides, outDims, outStrides, mode, "
                             "dataType"};
             }
             auto mode = hipdnn_flatbuffers_sdk::data_objects::ReductionMode::ADD;
             const auto declared = detail::enumName(resolved, "mode");
             if(!declared.empty())
             {
                 const auto* names
                     = hipdnn_flatbuffers_sdk::data_objects::EnumNamesReductionMode();
                 bool matched = false;
                 for(size_t i = 0; names[i] != nullptr; ++i)
                 {
                     if(declared == names[i])
                     {
                         mode = static_cast<
                             hipdnn_flatbuffers_sdk::data_objects::ReductionMode>(i);
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

/// @brief Builds the graph @p metadata describes for @p point.
///
/// The whole path in one call: resolve the declared arguments against the problem point, then
/// hand them to the named builder. Both halves fail closed, and the message names which.
inline BuildResult buildGraphFor(const OperationMetadata& metadata, const ProblemPoint& point)
{
    const auto& registry = builderRegistry();
    const auto adapter = registry.find(metadata.graphBuilder.function);
    if(adapter == registry.end())
    {
        // §4.4 check 5. A metadata file naming a builder nobody registered is a file that
        // cannot produce a problem, and saying so names the missing adapter.
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
