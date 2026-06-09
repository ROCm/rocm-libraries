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

    if(metadata.contains("decodings") && metadata["decodings"].contains("outputs"))
    {
        sequence_decodings_ = metadata["decodings"]["outputs"]
                                  .get<std::map<std::string, std::map<std::string, std::string>>>();
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

// --- Input feature engineering ----------------------------------------------

namespace {

// Widths expected by fdeep submodels. Update when retraining changes them.
// ExtractTunaNetND2dFeatures emits 46 features; the input_encoder model expects 43 because
// direction one-hot is omitted (direction is a constant in CandidateSelection metadata).
constexpr std::size_t kCandidateSelectionEncoderInputSize = 43;

float FeatureAt(const std::map<std::string, float>& features, const std::string& key)
{
    const auto it = features.find(key);
    if(it == features.end())
        MIOPEN_THROW("EngineerCandidateSelectionInputFeatures: missing feature '" + key + "'");
    return it->second;
}

std::vector<int> OneHot(std::size_t label, std::size_t num_classes)
{
    std::vector<int> out(num_classes, 0);
    if(label < num_classes)
        out[label] = 1;
    else
        MIOPEN_LOG_W("EngineerCandidateSelectionInputFeatures: one_hot label "
                     << label << " out of range for " << num_classes << " classes");
    return out;
}

std::size_t EncodePrecisionLabel(float precision_feature)
{
    const auto data_type = static_cast<miopenDataType_t>(static_cast<int>(precision_feature));
    if(data_type == miopenBFloat16)
        return 0;
    if(data_type == miopenHalf)
        return 1;
    if(data_type == miopenFloat)
        return 2;
    MIOPEN_LOG_W("EngineerCandidateSelectionInputFeatures: unsupported precision, defaulting to 0");
    return 0;
}

} // namespace

MIOPEN_INTERNALS_EXPORT
std::vector<float>
EngineerCandidateSelectionInputFeatures(const std::vector<float>& raw_features,
                                        const std::map<std::string, float>& features_by_name)
{
    (void)raw_features;

    if(FeatureAt(features_by_name, "spatial_dim") != 2.0f)
    {
        MIOPEN_THROW("EngineerCandidateSelectionInputFeatures: only 2D problems are supported");
    }

    MIOPEN_LOG_I2("Using engineered 2d features for Candidate Selection");

    // Mirror ExtractTunaNetND2dFeatures (ai_heuristics.cpp); keep in sync manually.
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
    const std::size_t K_h    = static_cast<std::size_t>(FeatureAt(features_by_name, "fil_h"));
    const std::size_t K_w    = static_cast<std::size_t>(FeatureAt(features_by_name, "fil_w"));
    std::size_t groups       = static_cast<std::size_t>(FeatureAt(features_by_name, "group_count"));
    const std::size_t num_cu = 254;

    const auto in_layout =
        OneHot(static_cast<std::size_t>(FeatureAt(features_by_name, "in_layout")), 2);
    const auto fil_layout =
        OneHot(static_cast<std::size_t>(FeatureAt(features_by_name, "fil_layout")), 2);
    const auto out_layout =
        OneHot(static_cast<std::size_t>(FeatureAt(features_by_name, "out_layout")), 2);
    const auto precision =
        OneHot(EncodePrecisionLabel(FeatureAt(features_by_name, "precision")), 3);
    // Direction one-hot is present in ExtractTunaNetND2dFeatures but omitted here because
    // CandidateSelection metadata holds direction as a constant input.

    if(groups < 1)
        groups = 1;

    const auto safe_ratio = [](double numerator, double denominator) -> double {
        if(denominator == 0.0)
            return 0.0;
        const double value = numerator / denominator;
        return std::isfinite(value) ? value : 0.0;
    };

    const auto safe_log1p = [](double value) -> double {
        if(value <= -1.0 || !std::isfinite(value))
            return 0.0;
        const double logged = std::log1p(value);
        return std::isfinite(logged) ? logged : 0.0;
    };

    const double flops = safe_ratio(2.0 * static_cast<double>(N) * static_cast<double>(C_out) *
                                        static_cast<double>(C_in) * static_cast<double>(K_h) *
                                        static_cast<double>(K_w) * static_cast<double>(H_out) *
                                        static_cast<double>(W_out),
                                    static_cast<double>(groups));

    const double M =
        safe_ratio(static_cast<double>(N) * static_cast<double>(H_out) * static_cast<double>(W_out),
                   static_cast<double>(groups));
    const double N_gemm = safe_ratio(static_cast<double>(C_out), static_cast<double>(groups));
    const double K_gemm =
        static_cast<double>(C_in) * static_cast<double>(K_h) * static_cast<double>(K_w);
    const double gemm_size = M * N_gemm * K_gemm;
    const double work_per_cu =
        safe_ratio(static_cast<double>(N) * static_cast<double>(H_out) *
                       static_cast<double>(W_out) * static_cast<double>(C_out),
                   static_cast<double>(groups) * static_cast<double>(num_cu));
    const double spatial_reduction =
        safe_ratio(static_cast<double>(H_in) * static_cast<double>(W_in),
                   static_cast<double>(H_out) * static_cast<double>(W_out));
    const double filter_coverage =
        safe_ratio(static_cast<double>(K_h) * static_cast<double>(K_w),
                   static_cast<double>(H_in) * static_cast<double>(W_in));
    const double channel_ratio = safe_ratio(static_cast<double>(C_in), static_cast<double>(C_out));
    const double group_density = safe_ratio(static_cast<double>(groups), static_cast<double>(C_in));

    std::vector<float> engineered = {
        static_cast<float>(in_layout[0]),
        static_cast<float>(in_layout[1]),
        static_cast<float>(fil_layout[0]),
        static_cast<float>(fil_layout[1]),
        static_cast<float>(out_layout[0]),
        static_cast<float>(out_layout[1]),
        static_cast<float>(precision[0]),
        static_cast<float>(precision[1]),
        static_cast<float>(precision[2]),

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

        static_cast<float>(safe_log1p(flops)),
        static_cast<float>(safe_log1p(M)),
        static_cast<float>(safe_log1p(N_gemm)),
        static_cast<float>(safe_log1p(K_gemm)),
        static_cast<float>(safe_ratio(M, N_gemm)),
        static_cast<float>(safe_ratio(M, K_gemm)),
        static_cast<float>(safe_ratio(N_gemm, K_gemm)),
        static_cast<float>(safe_log1p(gemm_size)),
        static_cast<float>(safe_log1p(work_per_cu)),
        static_cast<float>(spatial_reduction),
        static_cast<float>(filter_coverage),
        static_cast<float>(channel_ratio),
        static_cast<float>(group_density),
        static_cast<float>(safe_log1p(static_cast<double>(H_in))),
        static_cast<float>(safe_log1p(static_cast<double>(W_in))),
        static_cast<float>(safe_log1p(static_cast<double>(C_in))),
        static_cast<float>(safe_log1p(static_cast<double>(C_out))),
        static_cast<float>(safe_log1p(static_cast<double>(N))),
    };

    if(engineered.size() != kCandidateSelectionEncoderInputSize)
    {
        MIOPEN_THROW("EngineerCandidateSelectionInputFeatures: expected " +
                     std::to_string(kCandidateSelectionEncoderInputSize) + " features, got " +
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

// Derived feature count from ConvKernConfigPreprocessor::_count_derived_features (models.py).
constexpr std::size_t kKernelConfigDerivedFeatureCount = 10;

std::size_t ComputeKernelConfigPreprocessorOutputDim(const CandidateSelectionMetadata& metadata)
{
    const auto active_params           = ActiveOutputParams(metadata);
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

bool ParamNameEndsWith(const std::string& param_name, const std::string& suffix)
{
    return param_name.size() >= suffix.size() &&
           param_name.compare(param_name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

float GetRawConfigParamBySuffix(const std::vector<float>& raw_config_features,
                                const std::vector<std::string>& active_params,
                                const std::string& suffix,
                                float missing_token)
{
    for(std::size_t i = 0; i < active_params.size(); ++i)
    {
        if(ParamNameEndsWith(active_params[i], suffix))
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
    for(std::size_t c = 0; c < num_categories; ++c)
        engineered.push_back(c == static_cast<std::size_t>(index) ? 1.0f : 0.0f);
}

} // namespace

MIOPEN_INTERNALS_EXPORT
std::vector<float>
EngineerCandidateSelectionKernelConfigFeatures(const std::vector<float>& raw_config_features,
                                               const CandidateSelectionMetadata& metadata)
{
    const auto active_params              = ActiveOutputParams(metadata);
    const auto& sequence_encodings        = metadata.sequence_encodings();
    const float missing_token             = metadata.GetMissingValueToken();
    const std::size_t expected_output_dim = ComputeKernelConfigPreprocessorOutputDim(metadata);

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
    const float m_xdl_wave  = safe_param("MXdlPerWave");
    const float n_xdl_wave  = safe_param("NXdlPerWave");
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

    const auto engineered_features =
        EngineerCandidateSelectionInputFeatures(filtered_features, features);
    return EncodeInputFeaturesWithFdeep(engineered_features, arch_, solver_);
}

MIOPEN_INTERNALS_EXPORT
std::vector<std::vector<float>> CandidateSelectionModel::EncodeKernelConfigs(
    const std::vector<std::vector<float>>& encoded_candidates) const
{
    std::vector<std::vector<float>> engineered_candidates;
    engineered_candidates.reserve(encoded_candidates.size());
    for(const auto& candidate : encoded_candidates)
    {
        engineered_candidates.push_back(
            EngineerCandidateSelectionKernelConfigFeatures(candidate, metadata_));
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

MIOPEN_INTERNALS_EXPORT
std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata)
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

        // Build a map from param_name to value for this candidate
        std::map<std::string, std::string> param_value_map;
        bool mapping_valid = true;
        for(const auto& kv : kernel_str_mapping)
        {
            try
            {
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

        std::vector<float> encoded;
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
                            std::ostringstream msg;
                            msg << "No sequence encoding found for output parameter: " << param_name
                                << " and value '" << param_value << "' is not a valid float.";
                            MIOPEN_THROW(msg.str());
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
        const auto& encoded_candidates = EncodeKernelParams(expanded_params, model.metadata());

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
