// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "IUhdAdapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <flatbuffers/flatbuffers.h>
#include <fstream>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/mlp_model_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Accumulators held in registers with fused multiply-adds, where the build allows. MSVC defines
// __AVX2__ for /arch:AVX2 but never __FMA__; every AVX2 CPU it can target has FMA.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define HIPDNN_MLP_AVX2 1
#include <immintrin.h>
#endif

// `__restrict` is spelled the same by MSVC, GCC and Clang.
#define MLP_RESTRICT __restrict

namespace hipdnn_plugin_sdk::uhd
{

/// @brief Ensemble of small MLPs estimating one engine's metric (UHD adapter `mlp_data`).
///
/// One network ensemble per engine and architecture, scoring a feature row to the log of a
/// ranking metric in a few microseconds: an engine estimate (`predict_engine`) scores the
/// graph's row, a catalog ranker (`sort_kernel_catalog`) each candidate's; a tree_data model
/// walks hundreds of trees for the same row. The artifact (`mlp_model.fbs`,
/// identifier `HMLP`) names that metric; the UHD declares the same `score.metric` with
/// `transform: log`, and the loader refuses a mismatch.
///
/// A raw row x is mapped to t = sign(x) * log1p(|x|), with a missing (NaN) value at 0, and
/// standardized per network input; one 0/1 flag per `missing_flags` slot follows (1 when the
/// raw value is NaN). Every member is a stack of fully connected layers with SiLU between
/// them, ending in one output; the score is `target_mean + target_scale * mean over members`.
///
/// Members run in float32 with the ensemble mean taken in double. The kernels are plain C++
/// written for the compiler's auto-vectorizer (input-major weights, fixed-width accumulator
/// blocks, a branch-free polynomial exp). A build that enables AVX2 and FMA (`-mavx2 -mfma`,
/// `/arch:AVX2`) switches the layer product to register-resident intrinsics, about 1.6x faster;
/// results agree with the portable path to float rounding. Scoring allocates nothing: scratch
/// lives on the stack, so concurrent score() calls are safe.
class MlpDataAdapter : public IUhdAdapter
{
public:
    /// Load an MLP ensemble from a FlatBuffer artifact file.
    /// @param expectedFeaturesHash The UHD's `features_hash`; the artifact's must equal it.
    ///        Empty skips the check.
    /// @param expectedMetric The UHD's `score.metric`; the artifact's must equal it.
    /// @param expectedModelHash Artifact SHA-256 hex; empty skips the check.
    /// @returns nullptr if loading or validation fails, with the reason logged.
    static std::unique_ptr<MlpDataAdapter> load(const std::string& modelPath,
                                                const std::string& expectedFeaturesHash,
                                                const std::string& expectedMetric,
                                                const std::string& expectedModelHash = "");

    /// Load from an in-memory buffer; everything the model needs is copied out of it.
    /// Parameters as for load().
    /// @returns nullptr if validation fails, with the reason logged.
    static std::unique_ptr<MlpDataAdapter> loadFromBuffer(const uint8_t* buffer,
                                                          size_t size,
                                                          const std::string& expectedFeaturesHash,
                                                          const std::string& expectedMetric,
                                                          const std::string& expectedModelHash
                                                          = "");

    ~MlpDataAdapter() override = default;

    MlpDataAdapter(const MlpDataAdapter&) = delete;
    MlpDataAdapter& operator=(const MlpDataAdapter&) = delete;

    /// @returns The predicted log of the model's metric, or NaN (declined) if @p features does
    /// not have the model's width or holds an infinity, which no network input can encode.
    double score(const std::vector<double>& features) const override;

    BatchScores scoreBatch(const FeatureRows& rows) const override;

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

    bool isTrainedForArch(const std::string& arch) const override;

    /// Networks in the ensemble.
    size_t memberCount() const
    {
        return _memberEnd.size();
    }

private:
    /// Widest input or hidden layer, and deepest member, a model may have. They size the
    /// stack scratch, so an artifact cannot make scoring allocate.
    static constexpr size_t MAX_WIDTH = 512;
    static constexpr size_t MAX_LAYERS = 8;
    static constexpr size_t MAX_MEMBERS = 64;
    static constexpr size_t MAX_FEATURES = 4096;
    static constexpr size_t MAX_ARTIFACT_BYTES = size_t{64} * 1024 * 1024;

    /// A hidden layer's width is padded to a multiple of this, so the kernels work on whole
    /// blocks; the padding's weights and biases are zero and it contributes nothing.
    static constexpr size_t BLOCK = 32;

    /// One layer, its parameters at offsets into _parameters. A hidden layer's weights are
    /// input-major, `in` rows of `outPadded` floats (columns past `out` are zero), followed by
    /// `outPadded` biases. The final layer, one output, stores its `in` weights then its bias.
    struct Layer
    {
        size_t in;
        size_t outPadded;
        size_t weights;
        size_t bias;
    };

    MlpDataAdapter() = default;

    /// Checks the verified model and builds the scoring tables.
    /// @returns Why @p model is unusable, or empty with @p adapter filled.
    static std::string build(const hipdnn_flatbuffers_sdk::data_objects::MlpModel& model,
                             MlpDataAdapter& adapter);

    /// x[0, n) = SiLU(x[0, n)) in place, `x / (1 + exp(-x))`, exp accurate to about 1e-6
    /// relative. @p n is at most MAX_WIDTH.
    static void silu(float* x, size_t n);

    /// y[0, outPadded) = bias + x[0, in) * weights, @p outPadded a multiple of BLOCK.
    static void affine(const float* x, const Layer& layer, const float* parameters, float* y);

    /// bias + x[0, in) . weights for the final one-output layer.
    static float dot(const float* x, const Layer& layer, const float* parameters);

    double scoreRow(const double* row, size_t width) const;

    size_t _numFeatures = 0;
    std::string _featuresHash;
    std::string _metric;
    std::vector<std::string> _trainingArches;

    std::vector<uint32_t> _inputColumns;
    std::vector<double> _mean;
    std::vector<double> _scale;
    std::vector<uint32_t> _missingColumns;
    double _targetMean = 0.0;
    double _targetScale = 1.0;

    std::vector<Layer> _layers;
    std::vector<size_t> _memberEnd; // One past each member's last layer in _layers.
    std::vector<float> _parameters;
};

inline std::unique_ptr<MlpDataAdapter> MlpDataAdapter::load(const std::string& modelPath,
                                                            const std::string& expectedFeaturesHash,
                                                            const std::string& expectedMetric,
                                                            const std::string& expectedModelHash)
{
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if(!file)
    {
        HIPDNN_SDK_LOG_ERROR("MlpDataAdapter: cannot open '" << modelPath << "'");
        return nullptr;
    }

    const auto size = file.tellg();
    if(size <= 0 || static_cast<size_t>(size) > MAX_ARTIFACT_BYTES)
    {
        HIPDNN_SDK_LOG_ERROR("MlpDataAdapter: '" << modelPath << "' is empty or too large");
        return nullptr;
    }

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        return nullptr;
    }

    return loadFromBuffer(
        buffer.data(), buffer.size(), expectedFeaturesHash, expectedMetric, expectedModelHash);
}

inline std::unique_ptr<MlpDataAdapter>
    MlpDataAdapter::loadFromBuffer(const uint8_t* buffer,
                                   size_t size,
                                   const std::string& expectedFeaturesHash,
                                   const std::string& expectedMetric,
                                   const std::string& expectedModelHash)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4 || size > MAX_ARTIFACT_BYTES)
    {
        return nullptr;
    }

    // RFC 0019 §9.2: the digest identifies the model to persistent caches, so mismatched
    // bytes must not be scored under it.
    if(!expectedModelHash.empty())
    {
        const std::string actualHash = sha256(buffer, size);
        if(actualHash != expectedModelHash)
        {
            HIPDNN_SDK_LOG_ERROR(
                "MlpDataAdapter: model hash mismatch - expected='"
                << expectedModelHash << "' actual='" << actualHash
                << "'; the model is not used -- an engine estimate is reported as 0");
            return nullptr;
        }
    }

    std::string reason;
    const fb::MlpModel* model = nullptr;
    if(!flatbuffers::BufferHasIdentifier(buffer, fb::MlpModelIdentifier()))
    {
        reason = "not an MlpModel FlatBuffer (identifier HMLP)";
    }
    else
    {
        flatbuffers::Verifier verifier(buffer, size);
        if(!fb::VerifyMlpModelBuffer(verifier))
        {
            reason = "the MlpModel FlatBuffer does not verify";
        }
        else
        {
            model = fb::GetMlpModel(buffer);
        }
    }

    std::unique_ptr<MlpDataAdapter> adapter(new MlpDataAdapter());
    if(model != nullptr)
    {
        reason = build(*model, *adapter);
    }
    if(reason.empty() && !expectedMetric.empty() && adapter->_metric != expectedMetric)
    {
        reason = "the model predicts '" + adapter->_metric + "' but the UHD scores '"
                 + expectedMetric + "'";
    }
    if(!reason.empty())
    {
        HIPDNN_SDK_LOG_ERROR("MlpDataAdapter: " << reason
                                                << "; the model is not used -- an engine "
                                                   "estimate is reported as 0");
        return nullptr;
    }

    // RFC 0019 §6.3 check 3, worded like the other adapters so one log search finds them all.
    if(!expectedFeaturesHash.empty() && adapter->_featuresHash != expectedFeaturesHash)
    {
        HIPDNN_SDK_LOG_ERROR("MlpDataAdapter: features hash mismatch - expected='"
                             << expectedFeaturesHash << "' actual='" << adapter->_featuresHash
                             << "'; the model is not used -- an engine estimate is reported "
                                "as 0");
        return nullptr;
    }
    return adapter;
}

namespace mlp_detail
{

/// Copies @p values (at most @p limit, each finite) into @p out. @returns Why not, or empty.
inline std::string floats(const flatbuffers::Vector<float>* values,
                          const std::string& label,
                          size_t limit,
                          std::vector<float>& out)
{
    out.clear();
    if(values == nullptr || values->size() > limit)
    {
        return label + " must hold at most " + std::to_string(limit) + " values";
    }
    out.assign(values->begin(), values->end());
    if(!std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); }))
    {
        return label + " holds a value that is not finite";
    }
    return {};
}

/// A feature-row slot below @p width, else -1.
inline int64_t slot(int32_t value, size_t width)
{
    return value >= 0 && static_cast<size_t>(value) < width ? value : -1;
}

} // namespace mlp_detail

inline std::string
    MlpDataAdapter::build(const hipdnn_flatbuffers_sdk::data_objects::MlpModel& model,
                          MlpDataAdapter& adapter)
{
    if(model.num_features() <= 0 || static_cast<size_t>(model.num_features()) > MAX_FEATURES)
    {
        return "num_features must be between 1 and " + std::to_string(MAX_FEATURES);
    }
    const auto width = static_cast<size_t>(model.num_features());
    adapter._numFeatures = width;

    if(model.metric() == nullptr || model.metric()->empty())
    {
        return "the model names no metric";
    }
    adapter._metric = model.metric()->str();
    if(model.features_hash() != nullptr)
    {
        adapter._featuresHash = model.features_hash()->str();
    }
    if(model.training_arches() != nullptr)
    {
        for(const auto* arch : *model.training_arches())
        {
            if(arch != nullptr)
            {
                adapter._trainingArches.push_back(arch->str());
            }
        }
    }

    const auto* inputs = model.inputs();
    const auto* missing = model.missing_flags();
    const size_t inputCount = inputs == nullptr ? 0 : inputs->size();
    const size_t missingCount = missing == nullptr ? 0 : missing->size();
    const size_t networkWidth = inputCount + missingCount;
    if(networkWidth == 0 || networkWidth > MAX_WIDTH)
    {
        return "the network input must have between 1 and " + std::to_string(MAX_WIDTH) + " values";
    }
    for(size_t i = 0; i < inputCount; ++i)
    {
        const auto* input = inputs->Get(static_cast<flatbuffers::uoffset_t>(i));
        const auto column = input == nullptr ? -1 : mlp_detail::slot(input->feature(), width);
        if(column < 0)
        {
            return "input " + std::to_string(i) + " reads a slot outside the "
                   + std::to_string(width) + " features";
        }
        if(!std::isfinite(input->mean()) || !std::isfinite(input->scale())
           || !(input->scale() > 0.0))
        {
            return "input " + std::to_string(i) + " needs a finite mean and a positive scale";
        }
        adapter._inputColumns.push_back(static_cast<uint32_t>(column));
        adapter._mean.push_back(input->mean());
        adapter._scale.push_back(input->scale());
    }
    for(size_t i = 0; i < missingCount; ++i)
    {
        const auto column
            = mlp_detail::slot(missing->Get(static_cast<flatbuffers::uoffset_t>(i)), width);
        if(column < 0)
        {
            return "missing flag " + std::to_string(i) + " reads a slot outside the "
                   + std::to_string(width) + " features";
        }
        adapter._missingColumns.push_back(static_cast<uint32_t>(column));
    }

    adapter._targetMean = model.target_mean();
    adapter._targetScale = model.target_scale();
    if(!std::isfinite(adapter._targetMean) || !std::isfinite(adapter._targetScale)
       || !(adapter._targetScale > 0.0))
    {
        return "target_mean must be finite and target_scale positive";
    }

    const auto* networks = model.networks();
    if(networks == nullptr || networks->empty() || networks->size() > MAX_MEMBERS)
    {
        return "networks must hold between 1 and " + std::to_string(MAX_MEMBERS) + " networks";
    }

    std::vector<float> weights;
    std::vector<float> bias;
    for(flatbuffers::uoffset_t m = 0; m < networks->size(); ++m)
    {
        const auto* network = networks->Get(m);
        const std::string name = "network " + std::to_string(m);
        const auto* layers = network == nullptr ? nullptr : network->layers();
        if(layers == nullptr || layers->empty() || layers->size() > MAX_LAYERS)
        {
            return name + " must hold between 1 and " + std::to_string(MAX_LAYERS) + " layers";
        }
        size_t previous = networkWidth;
        for(flatbuffers::uoffset_t l = 0; l < layers->size(); ++l)
        {
            const auto* layer = layers->Get(l);
            const std::string label = name + " layer " + std::to_string(l);
            const bool last = l + 1 == layers->size();
            if(layer == nullptr || layer->inputs() <= 0 || layer->outputs() <= 0)
            {
                return label + " must declare positive inputs and outputs";
            }
            const auto in = static_cast<size_t>(layer->inputs());
            const auto out = static_cast<size_t>(layer->outputs());
            if(in != previous)
            {
                return label + " takes " + std::to_string(in) + " inputs, but "
                       + std::to_string(previous) + " are available";
            }
            if(out > MAX_WIDTH || (last && out != 1))
            {
                return label + " has a width outside [1, " + std::to_string(MAX_WIDTH)
                       + "], or the last layer does not have exactly one output";
            }
            auto reason
                = mlp_detail::floats(layer->weights(), label + " weights", in * out, weights);
            if(reason.empty())
            {
                reason = mlp_detail::floats(layer->biases(), label + " biases", out, bias);
            }
            if(!reason.empty())
            {
                return reason;
            }
            if(weights.size() != in * out || bias.size() != out)
            {
                return label + " must hold inputs*outputs weights and outputs biases";
            }

            Layer compiled{in, last ? size_t{1} : (out + BLOCK - 1) / BLOCK * BLOCK, 0, 0};
            compiled.weights = adapter._parameters.size();
            if(last)
            {
                adapter._parameters.insert(
                    adapter._parameters.end(), weights.begin(), weights.end());
            }
            else
            {
                // Transpose to input-major, padding each row with zeros.
                adapter._parameters.resize(compiled.weights + in * compiled.outPadded, 0.0F);
                for(size_t o = 0; o < out; ++o)
                {
                    for(size_t i = 0; i < in; ++i)
                    {
                        adapter._parameters[compiled.weights + i * compiled.outPadded + o]
                            = weights[o * in + i];
                    }
                }
            }
            compiled.bias = adapter._parameters.size();
            adapter._parameters.insert(adapter._parameters.end(), bias.begin(), bias.end());
            adapter._parameters.resize(compiled.bias + compiled.outPadded, 0.0F);
            adapter._layers.push_back(compiled);
            previous = out;
        }
        adapter._memberEnd.push_back(adapter._layers.size());
    }
    return {};
}

namespace mlp_detail
{

/// One block of outputs of an affine layer: y[j] = bias[j] + sum_i x[i] * w[i * stride + j] for
/// j in [0, WIDTH). The pointers do not alias; saying so is what lets compilers keep the loops
/// vectorized.
template <size_t WIDTH>
inline void affineBlock(const float* MLP_RESTRICT x,
                        size_t in,
                        const float* MLP_RESTRICT w,
                        size_t stride,
                        const float* MLP_RESTRICT bias,
                        float* MLP_RESTRICT y)
{
    for(size_t j = 0; j < WIDTH; ++j)
    {
        y[j] = bias[j];
    }
    for(size_t i = 0; i < in; ++i, w += stride)
    {
        const float xi = x[i];
        for(size_t j = 0; j < WIDTH; ++j)
        {
            y[j] += xi * w[j];
        }
    }
}

#if defined(HIPDNN_MLP_AVX2)
/// affineBlock<8 * VECTORS> with the accumulators in registers: the same sums in the same
/// order, but fused multiply-adds.
template <size_t VECTORS>
inline void affineVectors(
    const float* x, size_t in, const float* w, size_t stride, const float* bias, float* y)
{
    __m256 acc[VECTORS];
    for(size_t v = 0; v < VECTORS; ++v)
    {
        acc[v] = _mm256_loadu_ps(bias + 8 * v);
    }
    for(size_t i = 0; i < in; ++i, w += stride)
    {
        const __m256 xi = _mm256_set1_ps(x[i]);
        for(size_t v = 0; v < VECTORS; ++v)
        {
            acc[v] = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 8 * v), acc[v]);
        }
    }
    for(size_t v = 0; v < VECTORS; ++v)
    {
        _mm256_storeu_ps(y + 8 * v, acc[v]);
    }
}
#endif

} // namespace mlp_detail

inline void MlpDataAdapter::silu(float* x, size_t n)
{
    // exp(-x) = 2^k * exp(r) with |r| <= ln(2) / 2: Cody-Waite reduction and a degree-5
    // polynomial. Done in passes over short blocks so each loop is one the compiler can
    // vectorize: clamps are selects, and 2^k is built by writing the exponent field of an
    // integer array and reading it back as floats.
    constexpr size_t CHUNK = 64;
    for(size_t first = 0; first < n; first += CHUNK)
    {
        const size_t count = std::min(CHUNK, n - first);
        float* v = x + first;
        std::array<float, CHUNK> reduced;
        std::array<float, CHUNK> scale;
        std::array<uint32_t, CHUNK> exponent;
        for(size_t j = 0; j < count; ++j)
        {
            float e = -v[j];
            e = e < -87.0F ? -87.0F : e;
            e = e > 88.0F ? 88.0F : e;
            const float t = e * 1.44269504088896341F;
            const auto k = static_cast<int32_t>(t + (t >= 0.0F ? 0.5F : -0.5F));
            const auto kf = static_cast<float>(k);
            reduced[j] = (e - kf * 0.693359375F) - kf * -2.12194440e-4F;
            exponent[j] = static_cast<uint32_t>(k + 127) << 23;
        }
        std::memcpy(scale.data(), exponent.data(), count * sizeof(float));
        for(size_t j = 0; j < count; ++j)
        {
            const float r = reduced[j];
            float p = 1.0F / 120.0F;
            p = p * r + 1.0F / 24.0F;
            p = p * r + 1.0F / 6.0F;
            p = p * r + 0.5F;
            p = p * r + 1.0F;
            p = p * r + 1.0F;
            v[j] = v[j] / (1.0F + p * scale[j]);
        }
    }
}

inline void
    MlpDataAdapter::affine(const float* x, const Layer& layer, const float* parameters, float* y)
{
    const size_t stride = layer.outPadded;
    const float* weights = parameters + layer.weights;
    const float* bias = parameters + layer.bias;
    size_t first = 0;
#if defined(HIPDNN_MLP_AVX2)
    for(; first + 64 <= stride; first += 64)
    {
        mlp_detail::affineVectors<8>(x, layer.in, weights + first, stride, bias + first, y + first);
    }
    // BLOCK divides the stride, so what is left is one block.
    if(first < stride)
    {
        mlp_detail::affineVectors<4>(x, layer.in, weights + first, stride, bias + first, y + first);
    }
#else
    for(; first < stride; first += BLOCK)
    {
        mlp_detail::affineBlock<BLOCK>(
            x, layer.in, weights + first, stride, bias + first, y + first);
    }
#endif
}

inline float MlpDataAdapter::dot(const float* x, const Layer& layer, const float* parameters)
{
    constexpr size_t LANES = 8;
    const float* w = parameters + layer.weights;
    std::array<float, LANES> lanes{};
    size_t i = 0;
    for(; i + LANES <= layer.in; i += LANES)
    {
        for(size_t j = 0; j < LANES; ++j)
        {
            lanes[j] += x[i + j] * w[i + j];
        }
    }
    for(size_t j = 0; i + j < layer.in; ++j)
    {
        lanes[j] += x[i + j] * w[i + j];
    }
    return parameters[layer.bias]
           + (((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
              + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7])));
}

inline double MlpDataAdapter::scoreRow(const double* row, size_t width) const
{
    if(width != _numFeatures)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Left uninitialized: every value read is written first.
    alignas(32) std::array<float, MAX_WIDTH> input;
    alignas(32) std::array<float, MAX_WIDTH> ping;
    alignas(32) std::array<float, MAX_WIDTH> pong;

    // z = sign(x) * log1p(|x|), a missing value at 0, standardized; then the missing flags.
    size_t n = 0;
    bool finite = true;
    for(size_t k = 0; k < _inputColumns.size(); ++k)
    {
        const double x = row[_inputColumns[k]];
        const double z = std::isnan(x) ? 0.0 : std::copysign(std::log1p(std::fabs(x)), x);
        const double standardized = (z - _mean[k]) / _scale[k];
        finite = finite && std::isfinite(standardized);
        input[n++] = static_cast<float>(standardized);
    }
    for(const uint32_t column : _missingColumns)
    {
        input[n++] = std::isnan(row[column]) ? 1.0F : 0.0F;
    }
    if(!finite)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const float* parameters = _parameters.data();
    double sum = 0.0;
    size_t layer = 0;
    for(const size_t end : _memberEnd)
    {
        const float* x = input.data();
        float* y = ping.data();
        for(; layer + 1 < end; ++layer)
        {
            affine(x, _layers[layer], parameters, y);
            silu(y, _layers[layer].outPadded);
            x = y;
            y = y == ping.data() ? pong.data() : ping.data();
        }
        sum += static_cast<double>(dot(x, _layers[layer], parameters));
        ++layer;
    }
    return _targetMean + _targetScale * (sum / static_cast<double>(_memberEnd.size()));
}

inline double MlpDataAdapter::score(const std::vector<double>& features) const
{
    return scoreRow(features.data(), features.size());
}

inline BatchScores MlpDataAdapter::scoreBatch(const FeatureRows& rows) const
{
    BatchScores results;
    results.raw.reserve(rows.count);
    for(size_t i = 0; i < rows.count; ++i)
    {
        results.raw.push_back(scoreRow(rows.row(i), rows.width));
    }
    return results;
}

inline bool MlpDataAdapter::isTrainedForArch(const std::string& arch) const
{
    if(_trainingArches.empty())
    {
        return true;
    }

    const auto target = stripArchFeatures(arch);
    return std::find(_trainingArches.begin(), _trainingArches.end(), target)
           != _trainingArches.end();
}

} // namespace hipdnn_plugin_sdk::uhd

#undef MLP_RESTRICT
#undef HIPDNN_MLP_AVX2

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
