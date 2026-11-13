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

} // namespace

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
    // it here -> caller falls to the solver's non-AI default.
    if(common::LgbmOnly())
    {
        MIOPEN_LOG_I2("Lazy-applicability selection bypassed via MIOPEN_DEBUG_LGBM_ONLY for "
                      << solver << " on " << arch);
        return std::nullopt;
    }

    // Split a vocab kernel id into its bare GetTypeString and the "+<split_k>" a split-k solver
    // appends after the closing '>'. The base string itself can contain '+' (e.g.
    // "<256l+256m, ...>"), so search only after the final '>'. No suffix (non-split-k solvers) ->
    // base unchanged, split_k 1.
    const auto parse_kernel_id = [](const std::string& kid) -> std::pair<std::string, int> {
        const auto gt   = kid.rfind('>');
        const auto plus = kid.find('+', gt == std::string::npos ? 0 : gt);
        if(plus == std::string::npos)
            return {kid, 1};
        int split_k = 1;
        try
        {
            split_k = std::stoi(kid.substr(plus + 1));
        }
        catch(...)
        {
            // non-numeric suffix: keep split_k = 1
        }
        return {kid.substr(0, plus), split_k};
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
                const std::string& kernel_id = vocab[idx];
                // Hand the check the bare GetTypeString; the caller's ck_validator re-appends
                // "+<split_k>" for split-k solvers only.
                const auto [base, split_k] = parse_kernel_id(kernel_id);
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
        // TF32 request falls back to the FP32 set only if no TF32 kernel applies, matching
        // FillValidKernels' order (TF32 set first, FP32 only on empty).
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
