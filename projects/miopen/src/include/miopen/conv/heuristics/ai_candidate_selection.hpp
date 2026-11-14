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

// Forward declarations for the helpers implemented in ai_heuristics.cpp
std::vector<float> EncodeInputFeaturesWithFdeep(const std::vector<float>& features,
                                                const std::string& arch,
                                                const std::string& solver);

/// Ordered kernel-id vocabulary from the installed v3 config-embedding table. Empty when no v3
/// table is present (callers then fail closed to the solver's non-AI default). Implemented in
/// ai_heuristics.cpp.
MIOPEN_INTERNALS_EXPORT const std::vector<std::string>&
GetConfigEmbeddingVocab(const std::string& arch, const std::string& solver);

/// Embeddings for GetConfigEmbeddingVocab as a contiguous matrix (same order), so the scorer reads
/// them in place; the ids live in GetConfigEmbeddingVocab, so each embedding is held once.
MIOPEN_INTERNALS_EXPORT const std::vector<std::vector<float>>&
GetConfigEmbeddingMatrix(const std::string& arch, const std::string& solver);

/// Test hook: parse a config-embedding .bin at bin_path (md5-checked against the input encoder at
/// input_encoder_path) and return its kernel-id vocabulary; empty on any rejection. Lets loader
/// tests craft temp files instead of writing into the installed system db. In ai_heuristics.cpp.
MIOPEN_INTERNALS_EXPORT std::vector<std::string>
LoadConfigEmbeddingVocabFromFile(const std::string& bin_path,
                                 const std::string& input_encoder_path);

/// Rank-then-lazy-applicability candidate selection. Ranks the whole precomputed config vocabulary
/// by the two-tower score (input encoder + stored config embeddings -- no FillValidKernels, no
/// fdeep config encoder), then walks the ranking best-first and returns the first candidate
/// accepted by is_args_supported. Returns {bare kernel id (the GetTypeString; the caller re-appends
/// the "+split_k" suffix for split-k solvers), split_k} or nullopt when no vocab is available or
/// none is applicable (caller fails closed to the solver's non-AI default). Behaviour-preserving:
/// first applicable in score order is the argmax over applicable candidates.
MIOPEN_INTERNALS_EXPORT std::optional<std::pair<std::string, int>> SelectBestVocabCandidate(
    const std::string& arch,
    const std::string& solver,
    const std::map<std::string, float>& features,
    bool use_tf32,
    const std::function<bool(const std::string& base, int split_k, bool fp32_pass)>&
        is_args_supported);

/// Expands problem features into the vector consumed by the input_encoder submodel.
/// Produces the engineered features consumed by the input_encoder submodel for 2D and 3D models.
/// Shares derived-feature math with ExtractTunaNetNDFeatures (ai_heuristics.cpp); the direction
/// one-hot is omitted here when direction is a CandidateSelection constant. Raw numerics follow
/// metadata input_params order (training-time constants such as dilation_* for 3D are excluded).
/// All metadata-driven quantities (precision/layout one-hot encodings and widths, and num_cu for
/// the derived block) are read from `metadata`.
MIOPEN_INTERNALS_EXPORT std::vector<float>
EngineerCandidateSelectionInputFeatures(const std::map<std::string, float>& features_by_name,
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

} // namespace candidate_selection
} // namespace tuning
} // namespace ai
} // namespace miopen
