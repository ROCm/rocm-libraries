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
 *******************************************************************************/
#pragma once

#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <map>
#include <functional>
#include <miopen/config.hpp>

namespace miopen {
namespace ai {
namespace tuning {
namespace candidate_selection {

class CandidateSelectionMetadata;

// Declarative description of a CK kernel whose type-string layout is variable. Loaded from the
// "conditional_layouts" metadata section; absence means the kernel is fully static.
struct ConditionalLayout
{
    // Kind of a data-dependent (conditional) parameter.
    enum class ConditionalKind
    {
        present_if_gt_one,      // CK emits it only when value > 1, at base_index
        appended_suffix,        // appended after '>' as "+<int>"; last token when present
        inline_after_optionals, // always present at base_index, shifted right by the number of
                                // optional (present_if_gt_one) params that actually appear
    };
    struct ConditionalParam
    {
        ConditionalKind kind;
        std::size_t base_index = 0; // meaningful for present_if_gt_one and inline_after_optionals
    };
    // Codec for a packed token that unpacks into several features.
    enum class PackedCodec
    {
        tile_load_math_lm, // "{load}l+{math}m" -> two numeric features
    };
    struct PackedParam
    {
        std::size_t index = 0; // type-string index of the packed token
        PackedCodec codec;
        std::vector<std::string> outputs; // produced feature names, in order
    };

    std::size_t base_param_count = 0;
    std::map<std::string, ConditionalParam> conditional_params; // by param name (e.g. "SplitK")
    std::map<std::string, PackedParam> packed_params;           // by packed param name
};

// Forward declarations for the helpers implemented in ai_heuristics.cpp
std::vector<float> EncodeInputFeaturesWithFdeep(const std::vector<float>& features,
                                                const std::string& arch,
                                                const std::string& solver);

std::vector<std::vector<float>>
EncodeKernelConfigsWithFdeep(const std::vector<std::vector<float>>& encoded_candidates,
                             const std::string& arch,
                             const std::string& solver);

/// Expands problem features into the vector consumed by the input_encoder submodel.
/// Produces the same 2D engineered features as ExtractTunaNetND2dFeatures (ai_heuristics.cpp) --
/// they share the derived-feature math via common::EngineeredConvFeatures -- except the direction
/// one-hot is omitted here when direction is a CandidateSelection constant.
/// All metadata-driven quantities (precision/layout one-hot encodings and widths, and num_cu for
/// the derived block) are read from `metadata`.
MIOPEN_INTERNALS_EXPORT std::vector<float>
EngineerCandidateSelectionInputFeatures(const std::map<std::string, float>& features_by_name,
                                        const CandidateSelectionMetadata& metadata);

/// Expands metadata-ordered encoded kernel params into the vector consumed by the
/// kernel_config_encoder submodel (one-hot + raw numerical + derived features).
MIOPEN_INTERNALS_EXPORT std::vector<float>
EngineerCandidateSelectionKernelConfigFeatures(const std::vector<float>& raw_config_features,
                                               const CandidateSelectionMetadata& metadata);

using ValidationFunc = std::function<bool(int, int)>;

class CandidateSelectionMetadata
{
public:
    MIOPEN_INTERNALS_EXPORT CandidateSelectionMetadata(const std::string& arch,
                                                       const std::string& solver);
    MIOPEN_INTERNALS_EXPORT size_t GetInputParamIndex(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT size_t GetOutputParamIndex(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT std::optional<std::string>
    GetInputConstant(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT std::optional<std::string>
    GetOutputConstant(const std::string& name) const;
    MIOPEN_INTERNALS_EXPORT std::vector<size_t> GetConstantInputIndices() const;
    MIOPEN_INTERNALS_EXPORT std::vector<size_t> GetConstantOutputIndices() const;
    MIOPEN_INTERNALS_EXPORT std::map<std::string, std::string>
    GetKernelStrMapping(const std::string& kernel_name) const;
    // Getter functions for private members
    MIOPEN_INTERNALS_EXPORT const std::vector<std::string>& input_params() const
    {
        return input_params_;
    }
    MIOPEN_INTERNALS_EXPORT const std::vector<std::string>& output_params() const
    {
        return output_params_;
    }
    MIOPEN_INTERNALS_EXPORT const std::map<std::string, std::map<std::string, int>>&
    sequence_encodings() const;
    MIOPEN_INTERNALS_EXPORT float GetMissingValueToken() const;
    MIOPEN_INTERNALS_EXPORT const std::vector<int>& GetSplitKValues() const;
    // Number of one-hot classes for an input feature (e.g. "precision"), from the metadata's
    // input encodings. Returns 0 when the feature has no encoding (i.e. is not categorical).
    MIOPEN_INTERNALS_EXPORT std::size_t GetInputEncodingClassCount(const std::string& name) const;
    // One-hot index of a categorical input value (e.g. feature "precision", key "FP32"), from the
    // metadata's input encodings. Returns GetInputEncodingClassCount(feature) (an out-of-range
    // index) when the value is absent, so an unsupported value degrades cleanly rather than
    // colliding with a valid class.
    MIOPEN_INTERNALS_EXPORT std::size_t GetInputEncodingIndex(const std::string& feature,
                                                              const std::string& key) const;
    // Compute-unit count the model was trained with, from the metadata's "gpu.num_cu". Used to
    // normalize hardware-aware derived features. Returns 0 when absent.
    MIOPEN_INTERNALS_EXPORT std::size_t GetNumCu() const;
    // Conditional/packed layout for a kernel, or nullptr when the kernel is fully static (the
    // common case). Loaded from the metadata's "conditional_layouts" section.
    MIOPEN_INTERNALS_EXPORT const ConditionalLayout*
    GetConditionalLayout(const std::string& kernel_name) const;

private:
    // Internal mappings and encodings
    std::vector<std::string> input_params_;
    std::vector<std::string> output_params_;
    std::map<std::string, std::map<std::string, int>> sequence_encodings_;
    std::map<std::string, size_t> input_param_indices_;
    std::map<std::string, size_t> output_param_indices_;
    std::map<std::string, std::map<std::string, int>> feature_encodings_;
    std::map<std::string, std::map<std::string, std::string>> sequence_decodings_;
    std::map<std::string, std::string> constants_features_;
    std::map<std::string, std::string> constants_sequence_;
    std::map<std::string, std::map<std::string, std::string>> kernel_str_mapping_;
    float missing_value_token_;
    std::vector<int> split_k_values_;
    std::map<std::string, ConditionalLayout> conditional_layouts_;
    std::size_t num_cu_ = 0;
};

class CandidateSelectionModel
{
public:
    MIOPEN_INTERNALS_EXPORT CandidateSelectionModel(const std::string& arch,
                                                    const std::string& solver);
    MIOPEN_INTERNALS_EXPORT ~CandidateSelectionModel();

    MIOPEN_INTERNALS_EXPORT std::vector<float>
    EncodeInputFeatures(const std::map<std::string, float>& features) const;
    MIOPEN_INTERNALS_EXPORT std::vector<std::vector<float>>
    EncodeKernelConfigs(const std::vector<std::vector<float>>& encoded_candidates) const;
    MIOPEN_INTERNALS_EXPORT std::vector<std::pair<int, float>>
    SelectBestCandidateIndices(const std::vector<float>& encoded_features,
                               const std::vector<std::vector<float>>& encoded_configs) const;
    const CandidateSelectionMetadata& metadata() const { return metadata_; }

private:
    CandidateSelectionMetadata metadata_;
    std::string arch_;
    std::string solver_;
};

MIOPEN_INTERNALS_EXPORT const CandidateSelectionModel&
GetCandidateSelectionModel(const std::string& arch, const std::string& solver);

MIOPEN_INTERNALS_EXPORT std::vector<std::vector<float>>
EncodeKernelParams(const std::vector<std::vector<std::string>>& valid_kernel_params,
                   const CandidateSelectionMetadata& metadata,
                   bool use_split_k);

struct MIOPEN_INTERNALS_EXPORT CandidateSelectionResult
{
    std::vector<int> kernel_indices; // Sorted list of kernel indices (best to worst)
    std::vector<int> split_k_values; // Corresponding split_k values

    // Helper methods for backward compatibility and convenience
    int GetBestKernelIndex() const { return kernel_indices.empty() ? -1 : kernel_indices[0]; }
    int GetBestSplitK() const { return split_k_values.empty() ? 1 : split_k_values[0]; }

    int GetFallbackKernelIndex(size_t fallback_level = 1) const
    {
        return (fallback_level < kernel_indices.size()) ? kernel_indices[fallback_level] : -1;
    }

    int GetFallbackSplitK(size_t fallback_level = 1) const
    {
        return (fallback_level < split_k_values.size()) ? split_k_values[fallback_level] : 1;
    }

    size_t GetNumCandidates() const { return kernel_indices.size(); }
    bool IsEmpty() const { return kernel_indices.empty(); }
};

MIOPEN_INTERNALS_EXPORT
CandidateSelectionResult
ModelSelectBestCandidate(const std::string& arch,
                         const std::string& solver,
                         const std::map<std::string, float>& features,
                         const std::vector<std::vector<std::string>>& valid_kernel_params,
                         bool use_split_k,
                         ValidationFunc&& is_valid);

MIOPEN_INTERNALS_EXPORT
std::pair<std::vector<std::vector<std::string>>, std::vector<std::pair<int, int>>>
ExpandKernelParamsWithSplitK(const std::vector<std::vector<std::string>>& kernels,
                             const std::vector<int>& indexes,
                             const std::vector<int>& split_ks,
                             ValidationFunc&& is_valid);

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen
