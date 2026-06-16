/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************
 *
 * AI Candidate Selection Models for Kernel Tuning using a candidate selection approach.
 * Also known as a "Two Towers" model.
 * Contains: CandidateSelectionMetadata, CandidateSelectionModel, and helpers.
 *
 *******************************************************************************/

#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <nlohmann/json.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/stringutils.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cmath>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace ai {
namespace tuning {
namespace candidate_selection {

// --- CandidateSelectionMetadata ---------------------------------------------

MIOPEN_INTERNALS_EXPORT
CandidateSelectionMetadata::CandidateSelectionMetadata(const std::string& arch,
                                                       const std::string& solver)
{
    const auto path = GetSystemDbPath() / (arch + "_" + solver + "_metadata.tn.model");
    MIOPEN_LOG_I2("Loading metadata file: " + path.string());
    std::ifstream file(path);
    if(!file.is_open())
    {
        MIOPEN_THROW("Could not open metadata file: " + path.string());
    }
    nlohmann::json metadata;
    try
    {
        file >> metadata;
    }
    catch(const std::exception& ex)
    {
        MIOPEN_THROW("JSON parse error in metadata file: " + path.string() + ": " + ex.what());
    }

    input_params_  = metadata.value("input_params", std::vector<std::string>{});
    output_params_ = metadata.value("output_params", std::vector<std::string>{});

    for(size_t i = 0; i < input_params_.size(); ++i)
        input_param_indices_[input_params_[i]] = i;
    for(size_t i = 0; i < output_params_.size(); ++i)
        output_param_indices_[output_params_[i]] = i;

    if(metadata.contains("encodings"))
    {
        feature_encodings_ = metadata["encodings"].value("inputs", decltype(feature_encodings_){});
        sequence_encodings_ =
            metadata["encodings"].value("outputs", decltype(sequence_encodings_){});
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'encodings' section");
    }

    // Compute-unit count the model was trained with; normalizes hardware-aware derived features.
    if(metadata.contains("gpu") && metadata["gpu"].contains("num_cu"))
        num_cu_ = metadata["gpu"]["num_cu"].get<std::size_t>();
    else
        MIOPEN_LOG_W(
            "CandidateSelectionMetadata: no gpu.num_cu in metadata; hardware-aware derived "
            "features will use num_cu=0 (degraded 2D predictions)");

    if(metadata.contains("decodings") && metadata["decodings"].contains("outputs"))
    {
        // Decoding values may be serialized as numbers (e.g. NumGroupsToMerge: {"0": 1, ...}) or
        // strings. Coerce every value to string so a numeric JSON value does not fail the parse.
        for(const auto& [param_name, value_map] : metadata["decodings"]["outputs"].items())
        {
            std::map<std::string, std::string> decoded;
            for(const auto& [key, value] : value_map.items())
                decoded[key] = value.is_string() ? value.get<std::string>() : value.dump();
            sequence_decodings_[param_name] = std::move(decoded);
        }
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'decodings' section for outputs");
    }

    if(metadata.contains("constants"))
    {
        constants_features_ =
            metadata["constants"].value("inputs", decltype(constants_features_){});
        constants_sequence_ =
            metadata["constants"].value("outputs", decltype(constants_sequence_){});
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'constants' section");
    }

    if(metadata.contains("missing_value_token"))
    {
        missing_value_token_ = metadata["missing_value_token"].get<float>();
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'missing_value_token' section");
    }

    if(metadata.contains("kernel_str_mapping"))
    {
        kernel_str_mapping_ = metadata["kernel_str_mapping"]
                                  .get<std::map<std::string, std::map<std::string, std::string>>>();
    }
    else
    {
        MIOPEN_THROW("Metadata file does not contain 'kernel_str_mapping' section");
    }

    if(metadata.contains("split_k_values"))
    {
        split_k_values_ = metadata["split_k_values"].get<std::vector<int>>();
    }
    else
    {
        split_k_values_ = {1}; // Default to 1 if not specified
    }

    // Optional: declarative descriptors for kernels with variable type-string layouts;
    // absence => all kernels are static.
    if(metadata.contains("conditional_layouts"))
    {
        for(const auto& [kernel_name, layout_json] : metadata["conditional_layouts"].items())
        {
            ConditionalLayout layout;
            layout.base_param_count = layout_json.value("base_param_count", std::size_t{0});

            // Note: iterate the actual sub-object (not a .value(...) temporary -- items() over an
            // rvalue json throws invalid_iterator.214).
            if(layout_json.contains("conditional_params"))
            {
                for(const auto& [param_name, spec] : layout_json["conditional_params"].items())
                {
                    const std::string kind = spec.at("kind").get<std::string>();
                    ConditionalLayout::ConditionalParam cp;
                    if(kind == "present_if_gt_one")
                    {
                        cp.kind       = ConditionalLayout::ConditionalKind::present_if_gt_one;
                        cp.base_index = spec.value("base_index", std::size_t{0});
                    }
                    else if(kind == "appended_suffix")
                    {
                        cp.kind = ConditionalLayout::ConditionalKind::appended_suffix;
                    }
                    else if(kind == "inline_after_optionals")
                    {
                        cp.kind       = ConditionalLayout::ConditionalKind::inline_after_optionals;
                        cp.base_index = spec.value("base_index", std::size_t{0});
                    }
                    else
                    {
                        MIOPEN_THROW((std::ostringstream()
                                      << "Unknown conditional param kind '" << kind << "' for '"
                                      << param_name << "' in conditional_layouts[" << kernel_name
                                      << "]")
                                         .str());
                    }
                    layout.conditional_params.emplace(param_name, cp);
                }
            }

            if(layout_json.contains("packed_params"))
            {
                for(const auto& [param_name, spec] : layout_json["packed_params"].items())
                {
                    const std::string codec = spec.at("codec").get<std::string>();
                    ConditionalLayout::PackedParam pp;
                    pp.index   = spec.value("index", std::size_t{0});
                    pp.outputs = spec.value("outputs", std::vector<std::string>{});
                    if(codec == "tile_load_math_lm")
                    {
                        pp.codec = ConditionalLayout::PackedCodec::tile_load_math_lm;
                    }
                    else
                    {
                        MIOPEN_THROW((std::ostringstream()
                                      << "Unknown packed-param codec '" << codec << "' for '"
                                      << param_name << "' in conditional_layouts[" << kernel_name
                                      << "]")
                                         .str());
                    }
                    layout.packed_params.emplace(param_name, std::move(pp));
                }
            }

            // ResolveConditionalParamIndex infers optional-param presence from the candidate
            // length, which is unambiguous only for a single optional. Reject >1 here so a future
            // multi-optional kernel fails loudly at load instead of silently mis-decoding.
            const auto num_present_if_gt_one =
                std::ranges::count_if(layout.conditional_params, [](const auto& entry) {
                    return entry.second.kind ==
                           ConditionalLayout::ConditionalKind::present_if_gt_one;
                });
            if(num_present_if_gt_one > 1)
            {
                MIOPEN_THROW((std::ostringstream()
                              << "conditional_layouts[" << kernel_name
                              << "]: more than one 'present_if_gt_one' param is not supported "
                                 "(ResolveConditionalParamIndex assumes at most one optional).")
                                 .str());
            }

            conditional_layouts_.emplace(kernel_name, std::move(layout));
        }
    }
}

MIOPEN_INTERNALS_EXPORT
size_t CandidateSelectionMetadata::GetInputParamIndex(const std::string& name) const
{
    auto it = input_param_indices_.find(name);
    if(it == input_param_indices_.end())
        MIOPEN_THROW("Input parameter not found: " + name);
    return it->second;
}

MIOPEN_INTERNALS_EXPORT
size_t CandidateSelectionMetadata::GetOutputParamIndex(const std::string& name) const
{
    auto it = output_param_indices_.find(name);
    if(it == output_param_indices_.end())
        MIOPEN_THROW("Output parameter not found: " + name);
    return it->second;
}

MIOPEN_INTERNALS_EXPORT
std::optional<std::string>
CandidateSelectionMetadata::GetInputConstant(const std::string& name) const
{
    auto it = constants_features_.find(name);
    if(it != constants_features_.end())
        return it->second;
    return std::nullopt;
}

MIOPEN_INTERNALS_EXPORT
std::optional<std::string>
CandidateSelectionMetadata::GetOutputConstant(const std::string& name) const
{
    auto it = constants_sequence_.find(name);
    if(it != constants_sequence_.end())
        return it->second;
    return std::nullopt;
}

MIOPEN_INTERNALS_EXPORT
std::vector<size_t> CandidateSelectionMetadata::GetConstantInputIndices() const
{
    std::vector<size_t> indices;
    for(const auto& [name, value] : constants_features_)
    {
        auto it = input_param_indices_.find(name);
        if(it != input_param_indices_.end())
            indices.push_back(it->second);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

MIOPEN_INTERNALS_EXPORT
std::vector<size_t> CandidateSelectionMetadata::GetConstantOutputIndices() const
{
    std::vector<size_t> indices;
    for(const auto& [name, value] : constants_sequence_)
    {
        auto it = output_param_indices_.find(name);
        if(it != output_param_indices_.end())
            indices.push_back(it->second);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

MIOPEN_INTERNALS_EXPORT
std::map<std::string, std::string>
CandidateSelectionMetadata::GetKernelStrMapping(const std::string& kernel_name) const
{
    auto it = kernel_str_mapping_.find(kernel_name);
    if(it != kernel_str_mapping_.end())
    {
        return it->second;
    }
    else
    {
        MIOPEN_THROW("Kernel string mapping not found for kernel: " + kernel_name);
    }
}

MIOPEN_INTERNALS_EXPORT
const std::map<std::string, std::map<std::string, int>>&
CandidateSelectionMetadata::sequence_encodings() const
{
    return sequence_encodings_;
}
float CandidateSelectionMetadata::GetMissingValueToken() const { return missing_value_token_; }

MIOPEN_INTERNALS_EXPORT
const std::vector<int>& CandidateSelectionMetadata::GetSplitKValues() const
{
    return split_k_values_;
}

MIOPEN_INTERNALS_EXPORT
std::size_t CandidateSelectionMetadata::GetInputEncodingClassCount(const std::string& name) const
{
    const auto it = feature_encodings_.find(name);
    return it == feature_encodings_.end() ? 0 : it->second.size();
}

MIOPEN_INTERNALS_EXPORT
std::size_t CandidateSelectionMetadata::GetInputEncodingIndex(const std::string& feature,
                                                              const std::string& key) const
{
    const auto enc_it = feature_encodings_.find(feature);
    if(enc_it != feature_encodings_.end())
    {
        const auto key_it = enc_it->second.find(key);
        if(key_it != enc_it->second.end())
            return static_cast<std::size_t>(key_it->second);
    }
    // Absent value/feature: return an out-of-range index (== class count) so callers fall back
    // rather than collide with a valid class.
    return GetInputEncodingClassCount(feature);
}

MIOPEN_INTERNALS_EXPORT
std::size_t CandidateSelectionMetadata::GetNumCu() const { return num_cu_; }

MIOPEN_INTERNALS_EXPORT
const ConditionalLayout*
CandidateSelectionMetadata::GetConditionalLayout(const std::string& kernel_name) const
{
    const auto it = conditional_layouts_.find(kernel_name);
    return it == conditional_layouts_.end() ? nullptr : &it->second;
}

// --- Input feature engineering ----------------------------------------------

namespace {

float FeatureAt(const std::map<std::string, float>& features, const std::string& key)
{
    const auto it = features.find(key);
    if(it == features.end())
        MIOPEN_THROW("EngineerCandidateSelectionInputFeatures: missing feature '" + key + "'");
    return it->second;
}

// True when this model targets 2D convolutions, per the spatial_dim constant in its metadata.
// Feature engineering (input + kernel-config) is 2D-only; 3D models use raw features.
bool IsTwoDimensional(const CandidateSelectionMetadata& metadata)
{
    const auto spatial_dim = metadata.GetInputConstant("spatial_dim");
    return spatial_dim.has_value() && *spatial_dim == "2";
}

} // namespace

MIOPEN_INTERNALS_EXPORT
std::vector<float>
EngineerCandidateSelectionInputFeatures(const std::map<std::string, float>& features_by_name,
                                        const CandidateSelectionMetadata& metadata)
{
    // Callers gate this 2D-only path via IsTwoDimensional(metadata); no runtime spatial_dim check
    // here so a model whose feature map omits/differs on spatial_dim still engineers correctly.
    MIOPEN_LOG_I2("Using engineered 2d features for Candidate Selection");

    // Shares the derived-feature math with ExtractTunaNetND2dFeatures (ai_heuristics.cpp) via
    // common::EngineeredConvFeatures; only the input source and the omitted direction one-hot
    // differ.
    const float direction_code = FeatureAt(features_by_name, "direction");
    const bool is_fwd          = direction_code == 0.0f;

    const std::size_t N = static_cast<std::size_t>(FeatureAt(features_by_name, "batchsize"));
    const std::size_t C_in =
        static_cast<std::size_t>(is_fwd ? FeatureAt(features_by_name, "in_channels")
                                        : FeatureAt(features_by_name, "out_channels"));
    const std::size_t C_out =
        static_cast<std::size_t>(is_fwd ? FeatureAt(features_by_name, "out_channels")
                                        : FeatureAt(features_by_name, "in_channels"));
    const std::size_t H_in = static_cast<std::size_t>(
        is_fwd ? FeatureAt(features_by_name, "in_h") : FeatureAt(features_by_name, "out_h"));
    const std::size_t W_in = static_cast<std::size_t>(
        is_fwd ? FeatureAt(features_by_name, "in_w") : FeatureAt(features_by_name, "out_w"));
    const std::size_t H_out = static_cast<std::size_t>(
        is_fwd ? FeatureAt(features_by_name, "out_h") : FeatureAt(features_by_name, "in_h"));
    const std::size_t W_out = static_cast<std::size_t>(
        is_fwd ? FeatureAt(features_by_name, "out_w") : FeatureAt(features_by_name, "in_w"));
    const std::size_t K_h = static_cast<std::size_t>(FeatureAt(features_by_name, "fil_h"));
    const std::size_t K_w = static_cast<std::size_t>(FeatureAt(features_by_name, "fil_w"));
    std::size_t groups    = static_cast<std::size_t>(FeatureAt(features_by_name, "group_count"));
    // CU count the model was trained with, for the hardware-aware derived features.
    const std::size_t num_cu = metadata.GetNumCu();

    // Layout one-hot widths come from the metadata encodings (binary NCHW/NHWC fallback when the
    // model has no such encoding). The layout value in the feature map is already its class index.
    const auto layout_width = [&](const std::string& feature) {
        const auto width = metadata.GetInputEncodingClassCount(feature);
        return width == 0 ? std::size_t{2} : width;
    };
    const auto in_layout =
        common::OneHot(static_cast<long long>(FeatureAt(features_by_name, "in_layout")),
                       layout_width("in_layout"));
    const auto fil_layout =
        common::OneHot(static_cast<long long>(FeatureAt(features_by_name, "fil_layout")),
                       layout_width("fil_layout"));
    const auto out_layout =
        common::OneHot(static_cast<long long>(FeatureAt(features_by_name, "out_layout")),
                       layout_width("out_layout"));

    // Precision: the datatype->name mapping is a stable API fact; the name->index and one-hot width
    // come from the metadata so they track the trained model.
    const std::size_t precision_class_count = metadata.GetInputEncodingClassCount("precision");
    if(precision_class_count == 0)
    {
        // A 2D model with no precision encoding is malformed; throw so the caller falls back to the
        // non-AI heuristic rather than guessing a width/index.
        MIOPEN_THROW(
            "EngineerCandidateSelectionInputFeatures: metadata has no 'precision' encoding");
    }
    const auto data_type =
        static_cast<miopenDataType_t>(static_cast<int>(FeatureAt(features_by_name, "precision")));
    const char* precision_key = common::DataTypeToEncodingKey(data_type);
    const std::size_t precision_index =
        precision_key == nullptr ? precision_class_count
                                 : metadata.GetInputEncodingIndex("precision", precision_key);
    if(precision_index >= precision_class_count)
    {
        // The problem's precision isn't one this model was trained on (e.g. INT8 on a model with
        // only 3 precision classes). Throw so the caller falls back to the non-AI heuristic rather
        // than feeding an all-zero precision one-hot (a silently degraded prediction).
        MIOPEN_THROW("EngineerCandidateSelectionInputFeatures: precision '" +
                     std::string(precision_key == nullptr ? "unknown" : precision_key) +
                     "' not supported by this model (" + std::to_string(precision_class_count) +
                     " precision classes)");
    }
    const auto precision =
        common::OneHot(static_cast<long long>(precision_index), precision_class_count);
    // Direction one-hot is present in ExtractTunaNetND2dFeatures but omitted here because
    // CandidateSelection metadata holds direction as a constant input.

    std::vector<float> engineered;
    for(const auto* one_hot : {&in_layout, &fil_layout, &out_layout, &precision})
        for(const auto bit : *one_hot)
            engineered.push_back(static_cast<float>(bit));

    // Raw passthrough features.
    const std::vector<float> raw_tail = {
        static_cast<float>(C_in),
        static_cast<float>(H_in),
        static_cast<float>(W_in),
        static_cast<float>(C_out),
        static_cast<float>(H_out),
        static_cast<float>(W_out),
        static_cast<float>(K_h),
        static_cast<float>(K_w),
        FeatureAt(features_by_name, "pad_h"),
        FeatureAt(features_by_name, "pad_w"),
        FeatureAt(features_by_name, "conv_stride_h"),
        FeatureAt(features_by_name, "conv_stride_w"),
        FeatureAt(features_by_name, "dilation_h"),
        FeatureAt(features_by_name, "dilation_w"),
        FeatureAt(features_by_name, "batchsize"),
        FeatureAt(features_by_name, "group_count"),
    };
    engineered.insert(engineered.end(), raw_tail.begin(), raw_tail.end());

    // Derived feature block (shared with the TunaNet path). Dimensions above are normalized to the
    // forward (driver) convention; the GEMM assignment is selected by the actual direction. The
    // direction feature is encoded as 0=Forward, 1=BackwardData, 2=BackwardWeights.
    const auto gemm_dir = direction_code == 0.0f   ? common::ConvDirection::Forward
                          : direction_code == 1.0f ? common::ConvDirection::BackwardData
                                                   : common::ConvDirection::BackwardWeights;
    const auto derived  = common::EngineeredConvFeatures(
        N, C_in, C_out, H_in, W_in, H_out, W_out, K_h, K_w, groups, num_cu, gemm_dir);
    engineered.insert(engineered.end(), derived.begin(), derived.end());

    const std::size_t expected_size = in_layout.size() + fil_layout.size() + out_layout.size() +
                                      precision.size() + raw_tail.size() + derived.size();
    if(engineered.size() != expected_size)
    {
        MIOPEN_THROW("EngineerCandidateSelectionInputFeatures: expected " +
                     std::to_string(expected_size) + " features, got " +
                     std::to_string(engineered.size()));
    }

    return engineered;
}

namespace {

std::vector<std::string> ActiveOutputParams(const CandidateSelectionMetadata& metadata)
{
    std::vector<std::string> active;
    active.reserve(metadata.output_params().size());
    for(const auto& param_name : metadata.output_params())
    {
        if(!metadata.GetOutputConstant(param_name).has_value())
            active.push_back(param_name);
    }
    return active;
}

// Number of derived kernel-config features appended below (see the derived-feature block in
// EngineerCandidateSelectionKernelConfigFeatures).
constexpr std::size_t kKernelConfigDerivedFeatureCount = 10;

std::size_t ComputeKernelConfigPreprocessorOutputDim(const CandidateSelectionMetadata& metadata,
                                                     const std::vector<std::string>& active_params)
{
    const auto& sequence_encodings     = metadata.sequence_encodings();
    std::size_t onehot_features        = 0;
    std::size_t raw_numerical_features = 0;

    for(const auto& param_name : active_params)
    {
        const auto enc_it = sequence_encodings.find(param_name);
        if(enc_it != sequence_encodings.end())
            onehot_features += enc_it->second.size();
        else
            ++raw_numerical_features;
    }

    return onehot_features + raw_numerical_features + kKernelConfigDerivedFeatureCount;
}

float GetRawConfigParamBySuffix(const std::vector<float>& raw_config_features,
                                const std::vector<std::string>& active_params,
                                const std::string& suffix,
                                float missing_token)
{
    for(std::size_t i = 0; i < active_params.size(); ++i)
    {
        if(EndsWith(active_params[i], suffix))
            return raw_config_features[i];
    }
    return missing_token;
}

bool IsMissingConfigValue(float value, float missing_token)
{
    return value == missing_token || std::isnan(value);
}

float SafeConfigValueForDerived(float value, float missing_token)
{
    return value == missing_token ? 1.0f : value;
}

void AppendKernelConfigOneHot(std::vector<float>& engineered,
                              float encoded_value,
                              const std::map<std::string, int>& encoding_map,
                              float missing_token)
{
    const std::size_t num_categories = encoding_map.size();
    if(IsMissingConfigValue(encoded_value, missing_token))
    {
        engineered.insert(engineered.end(), num_categories, 0.0f);
        return;
    }

    const int index = static_cast<int>(encoded_value);
    if(index < 0 || static_cast<std::size_t>(index) >= num_categories)
    {
        // The encoded value indexes outside the metadata's class list (e.g. non-contiguous encoding
        // indices). Throw rather than emit a silent all-zero one-hot, so the caller falls back to
        // the non-AI heuristic instead of ranking on a degraded feature vector.
        MIOPEN_THROW("AppendKernelConfigOneHot: encoded index " + std::to_string(index) +
                     " out of range for " + std::to_string(num_categories) + " classes");
    }
    for(std::size_t c = 0; c < num_categories; ++c)
        engineered.push_back(c == static_cast<std::size_t>(index) ? 1.0f : 0.0f);
}

// Engineering implementation shared by the public entry point and the per-candidate loop in
// EncodeKernelConfigs. active_params and expected_output_dim depend only on the metadata, so a
// caller engineering many candidates derives them once and passes them in rather than re-deriving
// per candidate (ActiveOutputParams is otherwise recomputed for every candidate).
std::vector<float> EngineerKernelConfigFeaturesImpl(const std::vector<float>& raw_config_features,
                                                    const CandidateSelectionMetadata& metadata,
                                                    const std::vector<std::string>& active_params,
                                                    std::size_t expected_output_dim)
{
    const auto& sequence_encodings = metadata.sequence_encodings();
    const float missing_token      = metadata.GetMissingValueToken();

    if(raw_config_features.size() != active_params.size())
    {
        MIOPEN_THROW("EngineerCandidateSelectionKernelConfigFeatures: expected " +
                     std::to_string(active_params.size()) + " raw features, got " +
                     std::to_string(raw_config_features.size()));
    }

    std::vector<float> engineered;
    engineered.reserve(expected_output_dim);

    // 1. One-hot encoding for categorical output params (ConvKernConfigPreprocessor.forward).
    for(std::size_t i = 0; i < active_params.size(); ++i)
    {
        const auto& param_name = active_params[i];
        const auto enc_it      = sequence_encodings.find(param_name);
        if(enc_it == sequence_encodings.end())
            continue;

        AppendKernelConfigOneHot(engineered, raw_config_features[i], enc_it->second, missing_token);
    }

    // 2. Raw numerical features (non-categorical active params).
    for(std::size_t i = 0; i < active_params.size(); ++i)
    {
        if(sequence_encodings.find(active_params[i]) == sequence_encodings.end())
            engineered.push_back(raw_config_features[i]);
    }

    // 3. Derived features (ConvKernConfigPreprocessor.compute_derived_features).
    constexpr float kEps = 1e-8f;

    const auto get_param = [&](const std::string& suffix) {
        return GetRawConfigParamBySuffix(raw_config_features, active_params, suffix, missing_token);
    };
    const auto safe_param = [&](const std::string& suffix) {
        return SafeConfigValueForDerived(get_param(suffix), missing_token);
    };

    const float block_size  = safe_param("BlockSize");
    const float m_per_block = safe_param("MPerBlock");
    const float n_per_block = safe_param("NPerBlock");
    const float k_per_block = safe_param("KPerBlock");
    const float m_per_xdl   = safe_param("MPerXDL");
    const float n_per_xdl   = safe_param("NPerXDL");
    // MXdlPerWave / NXdlPerWave use the raw value (not the missing->1 clamp the others use), to
    // match the trained feature definitions: a missing value stays as the missing token here.
    const float m_xdl_wave  = get_param("MXdlPerWave");
    const float n_xdl_wave  = get_param("NXdlPerWave");
    const float a_block_vec = safe_param("ABlockTransferSrcScalarPerVector");
    const float b_block_vec = safe_param("BBlockTransferSrcScalarPerVector");

    // Block-level work distribution.
    engineered.push_back((m_per_block * n_per_block) / (block_size + kEps));
    engineered.push_back(m_per_block / (n_per_block + kEps));
    engineered.push_back(std::log1pf(block_size));

    // XDL utilization.
    engineered.push_back(m_xdl_wave * n_xdl_wave);
    engineered.push_back((m_per_xdl * m_xdl_wave) / (m_per_block + kEps));
    engineered.push_back((n_per_xdl * n_xdl_wave) / (n_per_block + kEps));
    engineered.push_back(block_size / 64.0f);

    // Memory transfer efficiency.
    engineered.push_back(a_block_vec / (b_block_vec + kEps));
    engineered.push_back(a_block_vec + b_block_vec);

    // K-dimension.
    engineered.push_back(std::log1pf(k_per_block));

    if(engineered.size() != expected_output_dim)
    {
        MIOPEN_THROW("EngineerCandidateSelectionKernelConfigFeatures: expected " +
                     std::to_string(expected_output_dim) + " features, got " +
                     std::to_string(engineered.size()));
    }

    return engineered;
}

} // namespace

MIOPEN_INTERNALS_EXPORT
std::vector<float>
EngineerCandidateSelectionKernelConfigFeatures(const std::vector<float>& raw_config_features,
                                               const CandidateSelectionMetadata& metadata)
{
    const auto active_params = ActiveOutputParams(metadata);
    const auto expected_output_dim =
        ComputeKernelConfigPreprocessorOutputDim(metadata, active_params);
    return EngineerKernelConfigFeaturesImpl(
        raw_config_features, metadata, active_params, expected_output_dim);
}

// --- CandidateSelectionModel ------------------------------------------------

MIOPEN_INTERNALS_EXPORT
CandidateSelectionModel::CandidateSelectionModel(const std::string& arch, const std::string& solver)
    : metadata_(arch, solver), arch_(arch), solver_(solver)
{
}

MIOPEN_INTERNALS_EXPORT
CandidateSelectionModel::~CandidateSelectionModel() = default;

MIOPEN_INTERNALS_EXPORT
std::vector<float>
CandidateSelectionModel::EncodeInputFeatures(const std::map<std::string, float>& features) const
{
    std::vector<float> filtered_features;
    const auto& input_params = metadata_.input_params();

    for(const auto& name : input_params)
    {
        // Skip constant features
        if(metadata_.GetInputConstant(name) != std::nullopt)
            continue;

        // Only add if present in the input map
        auto it = features.find(name);
        if(it != features.end())
        {
            filtered_features.push_back(it->second);
        }
        else
        {
            MIOPEN_THROW((std::ostringstream() << "Input parameter not found: " << name).str());
        }
    }

    // Feature engineering is a 2D-only path: the 2D models expect the engineered input vector,
    // while the 3D models still consume the raw (filtered) features as-is. The per-model
    // dimensionality is declared by the spatial_dim constant in the metadata.
    if(IsTwoDimensional(metadata_))
    {
        const auto engineered_features =
            EngineerCandidateSelectionInputFeatures(features, metadata_);
        return EncodeInputFeaturesWithFdeep(engineered_features, arch_, solver_);
    }
    return EncodeInputFeaturesWithFdeep(filtered_features, arch_, solver_);
}

MIOPEN_INTERNALS_EXPORT
std::vector<std::vector<float>> CandidateSelectionModel::EncodeKernelConfigs(
    const std::vector<std::vector<float>>& encoded_candidates) const
{
    // 2D-only feature engineering (see EncodeInputFeatures). 3D models consume the raw
    // metadata-ordered encoded candidates directly.
    if(!IsTwoDimensional(metadata_))
        return EncodeKernelConfigsWithFdeep(encoded_candidates, arch_, solver_);

    // active_params and the output dim are metadata-derived (candidate-independent), so derive them
    // once here rather than per candidate inside the loop.
    const auto active_params = ActiveOutputParams(metadata_);
    const auto expected_output_dim =
        ComputeKernelConfigPreprocessorOutputDim(metadata_, active_params);

    std::vector<std::vector<float>> engineered_candidates;
    engineered_candidates.reserve(encoded_candidates.size());
    for(const auto& candidate : encoded_candidates)
    {
        engineered_candidates.push_back(EngineerKernelConfigFeaturesImpl(
            candidate, metadata_, active_params, expected_output_dim));
    }
    return EncodeKernelConfigsWithFdeep(engineered_candidates, arch_, solver_);
}

MIOPEN_INTERNALS_EXPORT
std::vector<std::pair<int, float>> CandidateSelectionModel::SelectBestCandidateIndices(
    const std::vector<float>& encoded_features,
    const std::vector<std::vector<float>>& encoded_configs) const
{
    if(encoded_configs.empty() || encoded_features.empty())
    {
        MIOPEN_THROW(miopenStatusInternalError,
                     "Empty features or configs in SelectBestCandidateIndices");
    }

    size_t feature_dim    = encoded_features.size();
    size_t num_candidates = encoded_configs.size();

    std::vector<std::pair<int, float>> scored_candidates;
    scored_candidates.reserve(num_candidates);

    for(size_t i = 0; i < num_candidates; ++i)
    {
        if(encoded_configs[i].size() != feature_dim)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Config dimension mismatch in SelectBestCandidateIndices");

        float score = std::inner_product(
            encoded_configs[i].begin(), encoded_configs[i].end(), encoded_features.begin(), 0.0f);
        scored_candidates.emplace_back(static_cast<int>(i), score);
    }

    // Check if all scores are NaN (all candidates unsupported)
    bool all_nan = std::all_of(scored_candidates.begin(),
                               scored_candidates.end(),
                               [](const auto& candidate) { return std::isnan(candidate.second); });

    if(all_nan)
    {
        MIOPEN_LOG_W("All candidate kernels are unsupported by the AI model - cannot rank");
        MIOPEN_THROW(miopenStatusInternalError,
                     "AI model does not support any of the provided kernel candidates");
    }

    // NaN-aware comparator: ensures NaN scores (unsupported kernels) sort last
    auto score_comparator_nan_aware = [](const std::pair<int, float>& a,
                                         const std::pair<int, float>& b) {
        bool a_is_nan = std::isnan(a.second);
        bool b_is_nan = std::isnan(b.second);

        if(a_is_nan && b_is_nan)
            return false; // Both NaN, consider equal
        if(a_is_nan)
            return false; // a is NaN, b comes first
        if(b_is_nan)
            return true; // b is NaN, a comes first

        return a.second > b.second; // Normal descending order by score
    };

    // Sort by score in descending order (best to worst), with NaNs last
    std::sort(scored_candidates.begin(), scored_candidates.end(), score_comparator_nan_aware);

    return scored_candidates;
}
// --- Factory and Helper Functions -------------------------------------------

// Helper: Expand kernel params with split_k and keep mapping
MIOPEN_INTERNALS_EXPORT
std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<int, int>>>
ExpandKernelParamsWithSplitK(const std::vector<std::vector<std::string>>& kernels,
                             const std::vector<int>& indexes,
                             const std::vector<int>& split_ks,
                             ValidationFunc&& is_valid)
{
    std::vector<std::vector<std::string>> expanded;
    std::vector<std::pair<int, int>> mapping;

    for(size_t i = 0; i < kernels.size(); ++i)
    {
        for(int split_k : split_ks)
        {
            if(is_valid(indexes[i], split_k))
            {
                auto candidate = kernels[i];
                candidate.push_back(std::to_string(split_k));
                expanded.push_back(candidate);
                mapping.emplace_back(indexes[i], split_k);
            }
        }
    }
    return {expanded, mapping};
}

MIOPEN_INTERNALS_EXPORT
const CandidateSelectionModel& GetCandidateSelectionModel(const std::string& arch,
                                                          const std::string& solver)
{
    static std::map<std::string, std::unique_ptr<CandidateSelectionModel>> models;
    static std::mutex models_mutex;
    std::string key = arch + "_" + solver;

    std::lock_guard<std::mutex> lock(models_mutex);
    try
    {
        auto [it, inserted] =
            models.try_emplace(key, std::make_unique<CandidateSelectionModel>(arch, solver));
        MIOPEN_LOG_I2("CandidateSelectionModel created for arch: " << arch
                                                                   << ", solver: " << solver);
        return *(it->second);
    }
    catch(const std::exception& ex)
    {
        {
            std::ostringstream oss;
            oss << "Failed to construct CandidateSelectionModel for arch: " << arch
                << ", solver: " << solver << ". Exception: " << ex.what();
            MIOPEN_THROW(miopenStatusInternalError, oss.str());
        }
    }
}

// Conditional/packed kernel layouts are described declaratively in the metadata's
// "conditional_layouts" section (parsed into ConditionalLayout). The descriptor is exported into
// the model metadata, so adding/changing a conditional kernel needs no C++ change.
//
//  - "present_if_gt_one": CK emits the param only when its value > 1, at base_index; its presence
//  is
//    data-dependent (e.g. NumGroupsToMerge).
//  - "appended_suffix": MIOpen appends the param after the closing '>' as "+<int>". The C++
//  tokenizer
//    (GetKernelAsTokens) drops that suffix, and it is re-attached as the last candidate token by
//    ExpandKernelParamsWithSplitK when use_split_k is set. Thus within one EncodeKernelParams call,
//    "appended param present" == use_split_k (e.g. SplitK).
//  - "inline_after_optionals": always present at base_index, but shifted right by however many
//    optional (present_if_gt_one) params actually appear before it (e.g. BlkGemmPipelineScheduler/
//    -Version sit after the optional NumGroupsToMerge). Position-based, so it is independent of
//    whether the token keeps its "BlkGemmPipelineScheduler:" label or is a bare value -- the
//    encodings key it by whatever the metadata recorded.
//  - packed codec: a single type-string token unpacks into several features (e.g. Wavelet's
//    "{load}l+{math}m" -> TileLoad/TileMathThreadGroupSize via codec "tile_load_math_lm").

// Resolve the candidate-vector index for a conditional param given the layout. candidate_len
// includes kernel_name at [0] and the optional trailing appended (split_k) token. Returns
// std::nullopt when the param is absent for this candidate.
// num_inline_after_optionals is the count of always-present inline_after_optionals params in the
// layout. They each occupy one token, so they (and the appended suffix) are discounted from the
// candidate length to recover how many optional (present_if_gt_one) params actually appear.
//
// TODO: this infers presence from the candidate length, which is only unambiguous for a single
// optional param. Supporting >1 present_if_gt_one param would need per-optional presence detection
// (e.g. value-based, from the raw type string) rather than a length count. The metadata loader
// rejects >1 such param today (see the CandidateSelectionMetadata constructor) so this stays safe.
inline std::optional<size_t>
ResolveConditionalParamIndex(const ConditionalLayout::ConditionalParam& spec,
                             size_t base_param_count,
                             size_t candidate_len,
                             size_t num_inline_after_optionals,
                             bool use_split_k)
{
    // Base candidate length (kernel_name + base_param_count type-string params), no tail.
    const size_t base_len   = base_param_count + 1;
    const size_t fixed_tail = num_inline_after_optionals + (use_split_k ? 1 : 0);
    const size_t num_present_optionals =
        (candidate_len > base_len + fixed_tail) ? (candidate_len - base_len - fixed_tail) : 0;

    switch(spec.kind)
    {
    case ConditionalLayout::ConditionalKind::present_if_gt_one:
        // Present only when an optional slot exists (currently at most one such param per kernel).
        if(num_present_optionals >= 1)
            return spec.base_index + 1; // Python list index -> candidate index (+1 for kernel_name)
        return std::nullopt;
    case ConditionalLayout::ConditionalKind::appended_suffix:
        if(!use_split_k || candidate_len < base_len + 1)
            return std::nullopt;
        return candidate_len - 1; // always the appended last token when present
    case ConditionalLayout::ConditionalKind::inline_after_optionals:
        // Always present, but shifted right by however many optional params precede it. The value
        // is whatever token sits there (a labeled "BlkGemmPipelineScheduler:..." or a bare value).
        return spec.base_index + 1 + num_present_optionals;
    }
    return std::nullopt;
}

// Decode a packed token into its constituent feature values per `codec`, aligned with the
// descriptor's output order. Returns std::nullopt if the token does not match. The codecs must
// match those used to produce the model metadata.
inline std::optional<std::vector<std::string>>
DecodePackedParam(ConditionalLayout::PackedCodec codec, const std::string& token)
{
    switch(codec)
    {
    case ConditionalLayout::PackedCodec::tile_load_math_lm: {
        // "{load}l+{math}m" (e.g. "128l+256m") -> {load, math}.
        const auto l_pos = token.find("l+");
        if(l_pos == std::string::npos || l_pos == 0)
            return std::nullopt;
        if(token.empty() || token.back() != 'm')
            return std::nullopt;

        const std::string load = token.substr(0, l_pos);
        const std::string math =
            token.substr(l_pos + 2, token.size() - (l_pos + 2) - 1); // drop "l+","m"

        const auto all_digits = [](const std::string& s) {
            return !s.empty() &&
                   std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
        };
        if(!all_digits(load) || !all_digits(math))
            return std::nullopt;

        return std::vector<std::string>{load, math};
    }
    }
    return std::nullopt;
}

MIOPEN_INTERNALS_EXPORT
std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata,
                   bool use_split_k)
{
    std::vector<std::vector<float>> encoded_candidates;
    const auto& output_params          = metadata.output_params();
    const auto& sequence_encodings     = metadata.sequence_encodings();
    const float missing_value_encoding = metadata.GetMissingValueToken();

    for(const auto& candidate : valid_kernel_params)
    {
        std::ostringstream candidate_str;
        candidate_str << "[";
        for(size_t i = 0; i < candidate.size(); ++i)
        {
            if(i > 0)
                candidate_str << ", ";
            candidate_str << "\"" << candidate[i] << "\"";
        }
        candidate_str << "]";
        MIOPEN_LOG_I2("Kernel Parameter Candidate: " << candidate_str.str());
        // Get kernel_str_mapping for this candidate's kernel_name
        if(candidate.empty())
            MIOPEN_THROW("Candidate vector is empty, cannot extract kernel_name.");
        const std::string& kernel_name = candidate[0];

        // Try to get kernel string mapping - if not found, this is an unsupported kernel
        std::map<std::string, std::string> kernel_str_mapping;
        try
        {
            kernel_str_mapping = metadata.GetKernelStrMapping(kernel_name);
        }
        catch(const std::exception&)
        {
            // Kernel not in metadata - likely a new CK kernel not yet supported by the model
            // Log warning and create sentinel encoding to preserve index alignment
            MIOPEN_LOG_I2("Kernel not in metadata (new CK kernel?): "
                          << kernel_name
                          << ". AI model cannot predict for this kernel - it will be ranked last. "
                             "Consider updating the AI model to support this kernel type");

            // Create sentinel encoding (all NaN) to ensure this kernel ranks last
            // NaN propagates through dot product, resulting in NaN score which sorts last
            std::vector<float> sentinel_encoding(output_params.size() -
                                                     metadata.GetConstantOutputIndices().size(),
                                                 std::numeric_limits<float>::quiet_NaN());
            encoded_candidates.push_back(sentinel_encoding);
            continue; // Skip to next candidate
        }

        const ConditionalLayout* conditional_layout = metadata.GetConditionalLayout(kernel_name);

        // For conditional kernels, the static kernel_str_mapping index is only valid when the
        // candidate carries the full param tail; collect those param names to skip below.
        const auto is_conditional_param = [&](const std::string& mapped_name) {
            if(conditional_layout == nullptr)
                return false;
            return std::ranges::any_of(
                conditional_layout->conditional_params,
                [&](const auto& entry) { return EndsWith(mapped_name, entry.first); });
        };

        // Build a map from param_name to value for this candidate
        std::map<std::string, std::string> param_value_map;
        bool mapping_valid = true;
        for(const auto& kv : kernel_str_mapping)
        {
            try
            {
                // Skip conditional params here; resolve their data-dependent positions below.
                if(is_conditional_param(kv.second))
                    continue;

                // Use std::stoull for unsigned long long, then validate range
                unsigned long long ull_idx = std::stoull(kv.first);
                size_t idx                 = static_cast<size_t>(ull_idx);

                if(idx < candidate.size())
                    param_value_map[kv.second] = candidate[idx];
                else
                {
                    MIOPEN_LOG_W("Index " << idx << " out of bounds for candidate of size "
                                          << candidate.size() << " in kernel " << kernel_name);
                    mapping_valid = false;
                    break;
                }
            }
            catch(const std::exception& ex)
            {
                MIOPEN_LOG_W("Invalid index format in kernel_str_mapping: "
                             << kv.first << ", error: " << ex.what());
                mapping_valid = false;
                break;
            }
        }

        if(!mapping_valid)
        {
            // Skip this entire candidate rather than partial processing
            // also give a clear log message about the candidate being skipped
            std::ostringstream invalid_candidate_str;
            invalid_candidate_str << "[";
            for(size_t i = 0; i < candidate.size(); ++i)
            {
                if(i > 0)
                    invalid_candidate_str << ", ";
                invalid_candidate_str << "\"" << candidate[i] << "\"";
            }
            invalid_candidate_str << "]";

            MIOPEN_LOG_W("Skipping candidate due to invalid kernel string mapping. "
                         << "Kernel: " << kernel_name
                         << ", Candidate: " << invalid_candidate_str.str()
                         << ", Total mappings: " << kernel_str_mapping.size());
            continue; // Continue to the next candidate
        }

        // Conditional kernels: the static kernel_str_mapping cannot express data-dependent param
        // positions, nor packed tokens. Fill the affected output params here from raw candidate
        // indices per the metadata descriptor; the encode loop below then applies the normal
        // categorical/numeric encoding (or skips them when declared constant).
        if(conditional_layout != nullptr)
        {
            // Packed tokens: decode the token at descriptor index into its named output features.
            for(const auto& [packed_name, packed] : conditional_layout->packed_params)
            {
                const size_t token_idx = packed.index + 1; // +1 for candidate[0]=kernel_name
                if(token_idx >= candidate.size())
                    continue;
                const auto decoded = DecodePackedParam(packed.codec, candidate[token_idx]);
                if(!decoded)
                {
                    MIOPEN_LOG_W("Packed param '"
                                 << packed_name << "' token '" << candidate[token_idx]
                                 << "' did not match its codec for kernel " << kernel_name);
                    continue;
                }
                for(size_t out_pos = 0;
                    out_pos < packed.outputs.size() && out_pos < decoded->size();
                    ++out_pos)
                {
                    const std::string& out_name = packed.outputs[out_pos];
                    for(const auto& param_name : output_params)
                    {
                        if(EndsWith(param_name, out_name))
                            param_value_map[param_name] = (*decoded)[out_pos];
                    }
                }
            }

            // Conditional params: resolve position against the actual candidate length. Absent
            // params are left out of param_value_map, so the encode loop emits the missing token.
            const size_t num_inline_after_optionals = static_cast<size_t>(std::ranges::count_if(
                conditional_layout->conditional_params, [](const auto& entry) {
                    return entry.second.kind ==
                           ConditionalLayout::ConditionalKind::inline_after_optionals;
                }));
            for(const auto& [cond_name, spec] : conditional_layout->conditional_params)
            {
                const auto idx = ResolveConditionalParamIndex(spec,
                                                              conditional_layout->base_param_count,
                                                              candidate.size(),
                                                              num_inline_after_optionals,
                                                              use_split_k);
                if(!idx || *idx >= candidate.size())
                    continue;
                for(const auto& param_name : output_params)
                {
                    if(EndsWith(param_name, cond_name))
                        param_value_map[param_name] = candidate[*idx];
                }
            }
        }

        std::vector<float> encoded;
        bool encode_failed = false;
        for(const auto& param_name : output_params)
        {
            // Skip constant parameters
            if(metadata.GetOutputConstant(param_name).has_value())
                continue;

            float value = missing_value_encoding;

            auto val_it = param_value_map.find(param_name);
            if(val_it != param_value_map.end())
            {
                const std::string& param_value = val_it->second;

                // Handle "nan" token
                if(param_value == "nan")
                {
                    value = missing_value_encoding;
                }
                else
                {
                    // Encode using sequence_encodings
                    const auto enc_it = sequence_encodings.find(param_name);

                    if(enc_it == sequence_encodings.end())
                    {
                        // Try to cast param_value to float if no encoding is found
                        try
                        {
                            value = std::stof(param_value);
                        }
                        catch(const std::exception&)
                        {
                            // Localize the failure to this candidate: rank it last via the NaN
                            // sentinel (as the invalid-mapping path above does) instead of
                            // throwing, which would abort candidate selection for the whole
                            // problem.
                            MIOPEN_LOG_WE("Kernel: "
                                          << kernel_name << " - output parameter '" << param_name
                                          << "' value '" << param_value
                                          << "' has no encoding and is not a valid float; "
                                             "ranking this candidate last.");
                            encode_failed = true;
                            break;
                        }
                    }
                    else
                    {
                        const auto& value_map = enc_it->second;
                        const auto map_it     = value_map.find(param_value);

                        if(map_it == value_map.end())
                        {
                            // Secondary check: try matching param_value with all whitespace removed
                            std::string param_value_ws;
                            std::remove_copy_if(param_value.begin(),
                                                param_value.end(),
                                                std::back_inserter(param_value_ws),
                                                [](unsigned char c) { return std::isspace(c); });

                            bool found_ws = false;
                            for(const auto& kv : value_map)
                            {
                                std::string key_ws;
                                std::remove_copy_if(
                                    kv.first.begin(),
                                    kv.first.end(),
                                    std::back_inserter(key_ws),
                                    [](unsigned char c) { return std::isspace(c); });
                                if(param_value_ws == key_ws)
                                {
                                    value    = static_cast<float>(kv.second);
                                    found_ws = true;
                                    break;
                                }
                            }

                            if(!found_ws)
                            {
                                MIOPEN_LOG_WE(
                                    "Kernel: "
                                    << kernel_name << " - No encoding found in metadata for value '"
                                    << param_value << "' of output parameter: " << param_name);
                                MIOPEN_LOG_WE("setting it to the NaN value");
                                value = missing_value_encoding;
                            }
                        }
                        else
                        {
                            // Use the encoded value from the map
                            value = static_cast<float>(map_it->second);
                        }
                    }
                }
            }
            // If not present, value remains missing_value_encoding
            encoded.push_back(value);
        }
        if(encode_failed)
        {
            // All-NaN sentinel ensures this candidate sorts last; mirrors the unsupported-kernel
            // path above so one un-encodable candidate cannot disable selection for the problem.
            encoded_candidates.emplace_back(output_params.size() -
                                                metadata.GetConstantOutputIndices().size(),
                                            std::numeric_limits<float>::quiet_NaN());
            continue;
        }
        encoded_candidates.push_back(encoded);
    }

    return encoded_candidates;
}

MIOPEN_INTERNALS_EXPORT
CandidateSelectionResult
ModelSelectBestCandidate(const std::string& arch,
                         const std::string& solver,
                         const std::map<std::string, float>& features,
                         const std::vector<std::vector<std::string>>& valid_kernel_params,
                         const bool use_split_k,
                         ValidationFunc&& is_valid)
{
    try
    {
        const auto& model = GetCandidateSelectionModel(arch, solver);
        // debug: show that we successfully retrieved the model
        MIOPEN_LOG_I2("Retrieved CandidateSelectionModel for arch: " << arch
                                                                     << ", solver: " << solver);
        std::vector<std::vector<std::string>> expanded_params = valid_kernel_params;
        std::vector<std::pair<int, int>> mapping_pairs;
        std::vector<int> heuristic_indexes;
        heuristic_indexes.reserve(valid_kernel_params.size()); // Pre-allocate capacity
        for(size_t i = 0; i < valid_kernel_params.size(); ++i)
            heuristic_indexes.push_back(static_cast<int>(i));

        if(use_split_k)
        {
            // get split_k values from metadata
            const auto& split_ks = model.metadata().GetSplitKValues();

            // Expand kernel params with split_k and keep mapping
            std::tie(expanded_params, mapping_pairs) =
                ExpandKernelParamsWithSplitK(valid_kernel_params,
                                             heuristic_indexes,
                                             split_ks,
                                             std::forward<ValidationFunc>(is_valid));

            // check if any valid combinations were found
            if(expanded_params.empty())
            {
                MIOPEN_LOG_W("No valid kernel+split_k combinations found after filtering");
                return CandidateSelectionResult{{}, {}};
            }
        }
        else
        {

            // If split_k is 0, we do not expand, just use the original kernels
            for(int heuristic_index : heuristic_indexes)
            {
                mapping_pairs.emplace_back(heuristic_index, 1); // Default split_k of 1
            }
        }
        const auto& encoded_candidates =
            EncodeKernelParams(expanded_params, model.metadata(), use_split_k);

        if(encoded_candidates.empty())
        {
            MIOPEN_LOG_W("No valid encoded candidates available");
            return CandidateSelectionResult{{}, {}};
        }

        const auto& encoded_features = model.EncodeInputFeatures(features);
        {
            std::ostringstream encoded_features_log;
            miopen::LogRange(encoded_features_log << "Encoded features: [", encoded_features, ", ")
                << "]";
            MIOPEN_LOG_I2(encoded_features_log.str());
        }
        const auto& encoded_configs = model.EncodeKernelConfigs(encoded_candidates);
        {
            std::ostringstream encoded_configs_log;
            encoded_configs_log << "Encoded configs: [";
            bool first_config = true;
            for(const auto& cfg : encoded_configs)
            {
                if(!first_config)
                    encoded_configs_log << ", ";
                first_config = false;
                miopen::LogRange(encoded_configs_log << "[", cfg, ", ") << "]";
            }
            encoded_configs_log << "]";
            MIOPEN_LOG_I2(encoded_configs_log.str());
        }
        // Get all candidates sorted by score (best to worst)
        auto scored_candidates =
            model.SelectBestCandidateIndices(encoded_features, encoded_configs);
        ;

        CandidateSelectionResult result;
        result.kernel_indices.reserve(scored_candidates.size());
        result.split_k_values.reserve(scored_candidates.size());

        for(const auto& [candidate_idx, score] : scored_candidates)
        {
            if(candidate_idx >= 0 && candidate_idx < static_cast<int>(mapping_pairs.size()))
            {
                result.kernel_indices.push_back(mapping_pairs[candidate_idx].first);
                result.split_k_values.push_back(mapping_pairs[candidate_idx].second);
            }
        }

        return result;
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] Candidate selection model failed: " << ex.what());
        return CandidateSelectionResult{{}, {}};
    }
    catch(const std::exception& ex)
    {
        MIOPEN_LOG_I2(
            "[Warning] Candidate selection model failed with std exception: " << ex.what());
        return CandidateSelectionResult{{}, {}};
    }
}

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
