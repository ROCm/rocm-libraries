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
#include <miopen/timer.hpp>
#include <nlohmann/json.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/logger.hpp>
#include <miopen/stringutils.hpp>
#include <algorithm>
#include <functional>
#include <set>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cmath>
#include <sstream>

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

    // Kernel-config derived-feature schema (name-driven, mirrors the input tower's
    // derived_features). The ordered names select which config derived features the registry in
    // EngineerCandidateSelectionKernelConfigFeatures emits, so training can reorder / add features
    // (declared in metadata) without a positional C++ convention. kernel_config_raw_exclude lists
    // params handled as a derived feature instead of raw passthrough (e.g. 015_SplitK).
    kernel_config_derived_features_ =
        metadata.value("kernel_config_derived_features", std::vector<std::string>{});
    {
        const auto excl = metadata.value("kernel_config_raw_exclude", std::vector<std::string>{});
        kernel_config_raw_exclude_ = std::set<std::string>(excl.begin(), excl.end());
    }
    // Back-compat: legacy metadata (pre-schema) carries neither field. Fall back to the original
    // 10-feature positional list with no raw exclusion, reproducing prior behavior exactly.
    if(kernel_config_derived_features_.empty())
        kernel_config_derived_features_ = {"work_per_thread",
                                           "block_aspect_ratio",
                                           "log_block_size",
                                           "xdl_ops_per_block",
                                           "xdl_coverage_m",
                                           "xdl_coverage_n",
                                           "waves_per_block",
                                           "vector_load_ratio",
                                           "total_vector_transfer",
                                           "log_k_per_block"};

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
bool IsTwoDimensional(const CandidateSelectionMetadata& metadata)
{
    const auto spatial_dim = metadata.GetInputConstant("spatial_dim");
    return spatial_dim.has_value() && *spatial_dim == "2";
}

bool IsThreeDimensional(const CandidateSelectionMetadata& metadata)
{
    const auto spatial_dim = metadata.GetInputConstant("spatial_dim");
    return spatial_dim.has_value() && *spatial_dim == "3";
}

bool UsesEngineeredInputFeatures(const CandidateSelectionMetadata& metadata)
{
    return IsTwoDimensional(metadata) || IsThreeDimensional(metadata);
}

bool IsCandidateSelectionCategoricalOneHot(const std::string& name)
{
    return name == "in_layout" || name == "fil_layout" || name == "out_layout" ||
           name == "precision";
}

bool ShouldSkipCandidateSelectionRawInput(const CandidateSelectionMetadata& metadata,
                                          const std::string& name)
{
    if(metadata.GetInputConstant(name).has_value())
        return true;
    if(IsCandidateSelectionCategoricalOneHot(name))
        return true;
    // Direction drives derived GEMM assignment only; never part of the raw passthrough block.
    if(name == "direction")
        return true;
    return false;
}

float CandidateSelectionRawFeatureValue(const std::string& name,
                                        const std::map<std::string, float>& features_by_name,
                                        bool is_fwd)
{
    if(name == "in_channels")
        return is_fwd ? FeatureAt(features_by_name, "in_channels")
                      : FeatureAt(features_by_name, "out_channels");
    if(name == "in_d")
        return is_fwd ? FeatureAt(features_by_name, "in_d") : FeatureAt(features_by_name, "out_d");
    if(name == "in_h")
        return is_fwd ? FeatureAt(features_by_name, "in_h") : FeatureAt(features_by_name, "out_h");
    if(name == "in_w")
        return is_fwd ? FeatureAt(features_by_name, "in_w") : FeatureAt(features_by_name, "out_w");
    if(name == "out_channels")
        return is_fwd ? FeatureAt(features_by_name, "out_channels")
                      : FeatureAt(features_by_name, "in_channels");
    if(name == "out_d")
        return is_fwd ? FeatureAt(features_by_name, "out_d") : FeatureAt(features_by_name, "in_d");
    if(name == "out_h")
        return is_fwd ? FeatureAt(features_by_name, "out_h") : FeatureAt(features_by_name, "in_h");
    if(name == "out_w")
        return is_fwd ? FeatureAt(features_by_name, "out_w") : FeatureAt(features_by_name, "in_w");
    return FeatureAt(features_by_name, name);
}

std::vector<float>
ExtractCandidateSelectionRawInput(const std::map<std::string, float>& features_by_name,
                                  const CandidateSelectionMetadata& metadata)
{
    const float direction_code = FeatureAt(features_by_name, "direction");
    const bool is_fwd          = direction_code == 0.0f;

    std::vector<float> raw;
    for(const auto& feature_name : metadata.input_params())
    {
        if(ShouldSkipCandidateSelectionRawInput(metadata, feature_name))
            continue;
        raw.push_back(CandidateSelectionRawFeatureValue(feature_name, features_by_name, is_fwd));
    }
    return raw;
}

} // namespace

MIOPEN_INTERNALS_EXPORT
std::vector<float>
EngineerCandidateSelectionInputFeatures(const std::map<std::string, float>& features_by_name,
                                        const CandidateSelectionMetadata& metadata)
{
    const bool is_3d = IsThreeDimensional(metadata);
    MIOPEN_LOG_I2("Using engineered " << (is_3d ? "3" : "2")
                                      << "d features for Candidate Selection");

    // Shares the derived-feature math with ExtractTunaNetNDFeatures (ai_heuristics.cpp) via
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
    std::size_t D_in      = 1;
    std::size_t D_out     = 1;
    std::size_t K_d       = 1;
    if(is_3d)
    {
        D_in  = static_cast<std::size_t>(is_fwd ? FeatureAt(features_by_name, "in_d")
                                               : FeatureAt(features_by_name, "out_d"));
        D_out = static_cast<std::size_t>(is_fwd ? FeatureAt(features_by_name, "out_d")
                                                : FeatureAt(features_by_name, "in_d"));
        K_d   = static_cast<std::size_t>(FeatureAt(features_by_name, "fil_d"));
    }
    std::size_t groups = static_cast<std::size_t>(FeatureAt(features_by_name, "group_count"));
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
    // Direction one-hot is present in ExtractTunaNetNDFeatures but omitted here because
    // CandidateSelection metadata holds direction as a constant input.

    std::vector<float> engineered;
    for(const auto* one_hot : {&in_layout, &fil_layout, &out_layout, &precision})
        for(const auto bit : *one_hot)
            engineered.push_back(static_cast<float>(bit));

    // Raw numerics follow metadata input_params order; constants and categoricals are handled
    // separately. Parameters removed during training (e.g. dilation_* for MI355_3d) are absent
    // from input_params and are not appended here.
    std::size_t raw_count = 0;
    for(const auto& feature_name : metadata.input_params())
    {
        if(ShouldSkipCandidateSelectionRawInput(metadata, feature_name))
            continue;
        engineered.push_back(
            CandidateSelectionRawFeatureValue(feature_name, features_by_name, is_fwd));
        ++raw_count;
    }

    // Derived feature block (shared with the TunaNet path). Dimensions above are normalized to the
    // forward (driver) convention; the GEMM assignment is selected by the actual direction. The
    // direction feature is encoded as 0=Forward, 1=BackwardData, 2=BackwardWeights.
    const auto gemm_dir = direction_code == 0.0f   ? common::ConvDirection::Forward
                          : direction_code == 1.0f ? common::ConvDirection::BackwardData
                                                   : common::ConvDirection::BackwardWeights;
    const auto derived =
        is_3d ? common::EngineeredConvFeatures(N,
                                               C_in,
                                               C_out,
                                               H_in,
                                               W_in,
                                               H_out,
                                               W_out,
                                               K_h,
                                               K_w,
                                               groups,
                                               num_cu,
                                               gemm_dir,
                                               3,
                                               D_in,
                                               D_out,
                                               K_d)
              : common::EngineeredConvFeatures(
                    N, C_in, C_out, H_in, W_in, H_out, W_out, K_h, K_w, groups, num_cu, gemm_dir);
    engineered.insert(engineered.end(), derived.begin(), derived.end());

    // The refreshed two-tower CandidateSelection encoders drop the 2D depth feature: the training
    // preprocessor emits log1p(in_d) only for problems with real depth (3D), matching the shared
    // EngineeredConvFeatures (3D-only). Earlier encoders carried a constant log1p(1) for 2D; the
    // current models do not, so no extra depth feature is appended here for 2D.
    const std::size_t expected_size = in_layout.size() + fil_layout.size() + out_layout.size() +
                                      precision.size() + raw_count + derived.size();
    if(engineered.size() != expected_size)
    {
        MIOPEN_THROW("EngineerCandidateSelectionInputFeatures: expected " +
                     std::to_string(expected_size) + " features, got " +
                     std::to_string(engineered.size()));
    }

    return engineered;
}

namespace {

std::string JoinFloats(const std::vector<float>& values)
{
    std::ostringstream ss;
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        if(i > 0)
            ss << ", ";
        ss << values[i];
    }
    return ss.str();
}

void LogInputEncoderBlock(const std::vector<float>& raw_active,
                          const std::vector<float>& engineered)
{
    if(!miopen::IsLogging(miopen::LoggingLevel::Info2))
        return;

    MIOPEN_LOG_I2("InputEncoder RAW: [" << JoinFloats(raw_active) << "]");
    MIOPEN_LOG_I2("InputEncoder features: [" << JoinFloats(engineered) << "]");
}

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

std::size_t ComputeKernelConfigPreprocessorOutputDim(const CandidateSelectionMetadata& metadata,
                                                     const std::vector<std::string>& active_params)
{
    const auto& sequence_encodings     = metadata.sequence_encodings();
    const auto& raw_exclude            = metadata.kernel_config_raw_exclude();
    std::size_t onehot_features        = 0;
    std::size_t raw_numerical_features = 0;

    for(const auto& param_name : active_params)
    {
        // A raw-excluded param (e.g. 015_SplitK) is consumed by a derived feature, not passed
        // through as raw, and is not one-hot encoded -- skip it from both base counts.
        if(raw_exclude.count(param_name) != 0)
            continue;
        const auto enc_it = sequence_encodings.find(param_name);
        if(enc_it != sequence_encodings.end())
            onehot_features += enc_it->second.size();
        else
            ++raw_numerical_features;
    }

    // Derived-feature count is the length of the metadata-declared (or legacy) name list.
    return onehot_features + raw_numerical_features +
           metadata.kernel_config_derived_features().size();
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

    // 2. Raw numerical features (non-categorical active params, excluding those handled as a
    //    derived feature instead of raw passthrough -- e.g. 015_SplitK).
    const auto& raw_exclude = metadata.kernel_config_raw_exclude();
    for(std::size_t i = 0; i < active_params.size(); ++i)
    {
        if(sequence_encodings.find(active_params[i]) == sequence_encodings.end() &&
           raw_exclude.count(active_params[i]) == 0)
            engineered.push_back(raw_config_features[i]);
    }

    // 3. Derived features (ConvKernConfigPreprocessor.compute_derived_features).
    //
    // Each derived feature is a function of one or more raw config params. When any of a feature's
    // input params is missing (the suffix isn't in this solver's active_params, so get_param
    // returns missing_token) the whole feature is emitted as the missing sentinel instead of being
    // computed from a substituted value. This lets the model recognize "this derived quantity does
    // not apply to this candidate" rather than seeing a plausible-but-fabricated number. The
    // safe_param values below only keep the arithmetic finite for the present-input case; whenever
    // an input is missing the computed value is discarded in favor of the sentinel.
    //
    // The sentinel is missing_token itself (== -1 today), so missing raw params and missing derived
    // features share one representation. NOTE: -1 is a valid out-of-band marker only because every
    // derived feature here is naturally non-negative (products/ratios of non-negative params, log1p
    // of non-negatives); a future derived feature that can be negative would need a different
    // scheme.
    constexpr float kEps        = 1e-8f;
    const float kMissingDerived = missing_token;

    const auto get_param = [&](const std::string& suffix) {
        return GetRawConfigParamBySuffix(raw_config_features, active_params, suffix, missing_token);
    };
    // Mirror ConvKernConfigPreprocessor: a param is missing iff it equals the token (NaN is not
    // treated as missing here, matching both safe_param and the Python `== nantoken` check).
    const auto is_missing = [&](const std::string& suffix) {
        return get_param(suffix) == missing_token;
    };
    const auto safe_param = [&](const std::string& suffix) {
        return SafeConfigValueForDerived(get_param(suffix), missing_token);
    };
    // Push a derived feature: the missing sentinel if any input param is absent, else the value.
    const auto emit = [&](bool any_missing, float value) {
        engineered.push_back(any_missing ? kMissingDerived : value);
    };

    const float block_size  = safe_param("BlockSize");
    const float m_per_block = safe_param("MPerBlock");
    const float n_per_block = safe_param("NPerBlock");
    const float k_per_block = safe_param("KPerBlock");
    const float m_per_xdl   = safe_param("MPerXDL");
    const float n_per_xdl   = safe_param("NPerXDL");
    const float m_xdl_wave  = safe_param("MXdlPerWave");
    const float n_xdl_wave  = safe_param("NXdlPerWave");
    const float a_block_vec = safe_param("ABlockTransferSrcScalarPerVector");
    const float b_block_vec = safe_param("BBlockTransferSrcScalarPerVector");

    const bool block_size_missing  = is_missing("BlockSize");
    const bool m_per_block_missing = is_missing("MPerBlock");
    const bool n_per_block_missing = is_missing("NPerBlock");
    const bool k_per_block_missing = is_missing("KPerBlock");
    const bool m_per_xdl_missing   = is_missing("MPerXDL");
    const bool n_per_xdl_missing   = is_missing("NPerXDL");
    const bool m_xdl_wave_missing  = is_missing("MXdlPerWave");
    const bool n_xdl_wave_missing  = is_missing("NXdlPerWave");
    const bool a_block_vec_missing = is_missing("ABlockTransferSrcScalarPerVector");
    const bool b_block_vec_missing = is_missing("BBlockTransferSrcScalarPerVector");

    // Metadata-declared derived-feature registry. Each entry appends exactly one feature; the
    // emit order is driven by metadata.kernel_config_derived_features() (legacy metadata falls back
    // to the 10-feature list in the CandidateSelectionMetadata ctor). Training can reorder / extend
    // the set by declaring it in metadata; a new feature still needs a registry entry here.
    const std::unordered_map<std::string, std::function<void()>> derived_registry = {
        {"work_per_thread",
         [&] {
             emit(m_per_block_missing || n_per_block_missing || block_size_missing,
                  (m_per_block * n_per_block) / (block_size + kEps));
         }},
        {"block_aspect_ratio",
         [&] {
             emit(m_per_block_missing || n_per_block_missing, m_per_block / (n_per_block + kEps));
         }},
        {"log_block_size", [&] { emit(block_size_missing, std::log1pf(block_size)); }},
        {"xdl_ops_per_block",
         [&] { emit(m_xdl_wave_missing || n_xdl_wave_missing, m_xdl_wave * n_xdl_wave); }},
        {"xdl_coverage_m",
         [&] {
             emit(m_per_xdl_missing || m_xdl_wave_missing || m_per_block_missing,
                  (m_per_xdl * m_xdl_wave) / (m_per_block + kEps));
         }},
        {"xdl_coverage_n",
         [&] {
             emit(n_per_xdl_missing || n_xdl_wave_missing || n_per_block_missing,
                  (n_per_xdl * n_xdl_wave) / (n_per_block + kEps));
         }},
        {"waves_per_block", [&] { emit(block_size_missing, block_size / 64.0f); }},
        {"vector_load_ratio",
         [&] {
             emit(a_block_vec_missing || b_block_vec_missing, a_block_vec / (b_block_vec + kEps));
         }},
        {"total_vector_transfer",
         [&] { emit(a_block_vec_missing || b_block_vec_missing, a_block_vec + b_block_vec); }},
        {"log_k_per_block", [&] { emit(k_per_block_missing, std::log1pf(k_per_block)); }},
        // SplitK: NOT routed through emit() -- autodeduce (SplitK<1) / absent is a real state, not
        // "missing" (mirrors ConvKernConfigPreprocessor.compute_derived_features split encoding:
        // log2 for the real {1,2,...,128} ladder, a separate is_autodeduce flag for the -1 mode).
        {"log2_splitk",
         [&] {
             const float sk = get_param("SplitK");
             engineered.push_back(sk >= 1.0f ? std::log2(sk) : 0.0f);
         }},
        {"is_autodeduce",
         [&] {
             const float sk = get_param("SplitK");
             engineered.push_back(sk < 0.0f ? 1.0f : 0.0f);
         }},
    };

    for(const auto& name : metadata.kernel_config_derived_features())
    {
        const auto it = derived_registry.find(name);
        if(it == derived_registry.end())
            MIOPEN_THROW("EngineerCandidateSelectionKernelConfigFeatures: unknown kernel-config "
                         "derived feature '" +
                         name +
                         "' in metadata; add its computer to the registry to match this "
                         "model.");
        it->second();
    }

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

    // Engineered input path for 2D/3D models (spatial_dim constant in metadata). Other models
    // consume the raw filtered features as-is.
    if(UsesEngineeredInputFeatures(metadata_))
    {
        const auto raw_active = ExtractCandidateSelectionRawInput(features, metadata_);
        const auto engineered_features =
            EngineerCandidateSelectionInputFeatures(features, metadata_);
        LogInputEncoderBlock(raw_active, engineered_features);
        return EncodeInputFeaturesWithFdeep(engineered_features, arch_, solver_);
    }
    return EncodeInputFeaturesWithFdeep(filtered_features, arch_, solver_);
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
        MIOPEN_LOG_I2("ConfigEncoder score: candidate " << i << " = " << score);
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
MIOPEN_INTERNALS_EXPORT std::optional<std::pair<std::string, int>> SelectBestVocabCandidate(
    const std::string& arch,
    const std::string& solver,
    const std::map<std::string, float>& features,
    bool use_tf32,
    const std::function<bool(const std::string&, int, bool)>& is_args_supported)
{
    // MIOPEN_DEBUG_LGBM_ONLY turns the two-tower model off; the lazy path is that model, so bypass
    // it here (mirrors ModelSelectBestCandidate) -> caller falls to the solver's non-AI default.
    if(common::LgbmOnly())
    {
        MIOPEN_LOG_I2("Lazy-applicability selection bypassed via MIOPEN_DEBUG_LGBM_ONLY for "
                      << solver << " on " << arch);
        return std::nullopt;
    }

    // Parse the "+<split_k>" a split-k solver appends after the closing '>' of the GetTypeString.
    // The base kernel string itself can contain '+' (e.g. "<256l+256m, ...>"), so search only after
    // the final '>'. No suffix (non-split-k solvers) -> split_k 1.
    const auto split_k_of = [](const std::string& kid) -> int {
        const auto gt   = kid.rfind('>');
        const auto plus = kid.find('+', gt == std::string::npos ? 0 : gt);
        if(plus == std::string::npos)
            return 1;
        try
        {
            return std::stoi(kid.substr(plus + 1));
        }
        catch(...)
        {
            return 1;
        }
    };

    try
    {
        // Total time of the lazy-applicability selection, directly comparable to the enumeration
        // path's "GetDefaultPerformanceConfig(default path)" span.
        const miopen::ScopedTimeLogger total_timer(solver + " lazy_applicability.total");

        const auto& vocab = GetConfigEmbeddingVocab(arch, solver);
        if(vocab.empty())
            return std::nullopt; // no v3 table -> caller (RunAIHeuristics) fails closed to non-AI

        // Rank the whole vocabulary best-first using the stored config embeddings (no fdeep config
        // encoder, no FillValidKernels).
        std::vector<std::pair<int, float>> scored;
        {
            const miopen::ScopedTimeLogger rank_timer(solver + " lazy_applicability.rank (" +
                                                      std::to_string(vocab.size()) + " vocab)");
            const auto& model           = GetCandidateSelectionModel(arch, solver);
            const auto encoded_features = model.EncodeInputFeatures(features);
            // Score the cached embedding matrix in place (parallel to `vocab`) instead of copying
            // every embedding out of the vocab on each call.
            scored = model.SelectBestCandidateIndices(encoded_features,
                                                      GetConfigEmbeddingMatrix(arch, solver));
        }

        // Walk best-first, checking applicability lazily; this span + the check count show how much
        // of the old ~200 ms enumeration we replaced (ideally one check instead of ~thousands).
        const miopen::ScopedTimeLogger check_timer(solver +
                                                   " lazy_applicability.is_args_supported");
        // One top-down pass over the ranked vocabulary. fp32_pass selects the TF32 instance set
        // (false) or the FP32 fallback set (true); the ranking is computed once and reused.
        const auto walk = [&](bool fp32_pass) -> std::optional<std::pair<std::string, int>> {
            int checks = 0;
            for(const auto& [idx, score] : scored)
            {
                if(idx < 0 || static_cast<std::size_t>(idx) >= vocab.size())
                    continue;
                const std::string& kernel_id = vocab[idx].first;
                const int split_k            = split_k_of(kernel_id);
                // Hand the check the bare GetTypeString; the caller's ck_validator re-appends
                // "+<split_k>" for split-k solvers only. Return the same bare string.
                std::string base = kernel_id;
                {
                    const auto gt   = kernel_id.rfind('>');
                    const auto plus = kernel_id.find('+', gt == std::string::npos ? 0 : gt);
                    if(plus != std::string::npos)
                        base = kernel_id.substr(0, plus);
                }
                ++checks;
                if(is_args_supported(base, split_k, fp32_pass))
                {
                    MIOPEN_LOG_I2("Lazy-applicability selected: "
                                  << base << " (split_k=" << split_k << ", score=" << score
                                  << ", fp32_pass=" << fp32_pass << ") after " << checks
                                  << " applicability check(s) of " << vocab.size() << " vocab");
                    return std::make_pair(base, split_k);
                }
            }
            return std::nullopt;
        };

        // First walk against the TF32 instance set (or, for non-TF32 problems, the only set); a
        // TF32 request falls back to the FP32 set only if no TF32 kernel applies, mirroring the
        // enumeration path (FillValidKernels TF32-first, FP32 on empty).
        if(auto hit = walk(/*fp32_pass=*/false))
            return hit;
        if(use_tf32)
        {
            if(auto hit = walk(/*fp32_pass=*/true))
                return hit;
        }
        MIOPEN_LOG_I2("Lazy-applicability: none of " << vocab.size() << " vocab applicable for "
                                                     << solver << " on " << arch
                                                     << "; -> solver non-AI default");
        return std::nullopt;
    }
    catch(const std::exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] SelectBestVocabCandidate failed: " << ex.what());
        return std::nullopt;
    }
}

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
