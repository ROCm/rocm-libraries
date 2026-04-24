/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

// This file implements AI-based heuristics for convolution solver and kernel tuning.
// Sections:
//   1. Includes
//   2. Common utilities
//   3. Immediate mode AI models
//   4. Kernel tuning AI models (sequential prediction of kernel parameters)

#include <miopen/conv/heuristics/ai_heuristics.hpp>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <fdeep/fdeep.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/env.hpp>

#include <any>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_AI_FDEEP_USE_SINGLE_THREAD_PREDICT)

// 3D AI heuristics - now declared properly in header
// No need for local forward declarations since we include the header

namespace miopen {
namespace ai {
// Common utilities now in ai_common.hpp

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
namespace immed_mode {
Metadata::Metadata(const std::string& arch)
    : json(common::LoadJSON(GetSystemDbPath() / (arch + "_metadata.tn.model"))),
      direction_encodings(json["encodings"]["Direction"]),
      precision_encodings(json["encodings"]["Precision"]),
      layout_encodings(json["encodings"]["Layout"]),
      features(json["conv_params_used_as_features"]),
      num_inputs(json["num_inputs"]),
      num_outputs(json["num_outputs"]),
      num_solvers(json["num_solvers"]),
      solver_map(common::ReverseMap<std::string, size_t>(json["encodings"]["solver"])),
      features_mean(common::LookupValues<std::string, float>(
          features, json["stats"]["overall"]["features"]["mean"])),
      features_std(common::LookupValues<std::string, float>(
          features, json["stats"]["overall"]["features"]["std"])),
      test_features_mean(common::LookupValues<std::string, float>(
          features, json["stats"]["test"]["features"]["mean"])),
      test_features_std(common::LookupValues<std::string, float>(
          features, json["stats"]["test"]["features"]["std"]))
{
}

size_t Metadata::EncodeDirection(miopen::conv::Direction dir) const
{
    if(dir == conv::Direction::BackwardWeights)
        return direction_encodings.at("W");
    else if(dir == conv::Direction::BackwardData)
        return direction_encodings.at("B");
    else
        return direction_encodings.at("F");
}

size_t Metadata::EncodePrecision(miopenDataType_t data_type) const
{
    if(data_type == miopenBFloat16)
        return precision_encodings.at("BF16");
    else if(data_type == miopenHalf)
        return precision_encodings.at("FP16");
    else if(data_type == miopenFloat)
        return precision_encodings.at("FP32");
    MIOPEN_THROW("Unsupported data type passed to TunaNet");
}

size_t Metadata::EncodeLayout(const std::string& layout) const
{
    return layout_encodings.at(layout);
}

/** `Model` encapuslates the machinery required to run inference on a TunaNet model
 *
 * The `Model` class encapuslates all the machinery needed to run inference on a
 * TunaNet model, including loading the TunaNet model, formatting a problem so that it
 * can be fed into TunaNet for inference, and getting TunaNet's predictions etc.
 *
 * @param arch Architecture
 */
class Model
{
public:
    Metadata metadata;
    Model(const std::string& arch)
        : metadata(Metadata(arch)),
          model(fdeep::load_model(ModelPath(arch), true, fdeep::dev_null_logger)),
          input_shape(fdeep::tensor_shape(metadata.num_inputs)),
          offset(metadata.num_outputs - metadata.num_solvers)
    {
    }
    virtual ~Model() = default;
    /** Is given problem supported by TunaNet?
     *
     * A TunaNet model can only work with problems "similar" to the problems it was trained on.
     * Since our training data has changed over time, a TunaNet model trained for an earlier
     * GPU might not support the same set of problems as a TunaNet model trained for a later
     * GPU might. Thus, each subclass of `Model`, specializing `Model` to a specific GPU, must
     * implement its own `IsProblemSupported` function.
     *
     * @param problem Problem
     * @param ctx Execution context
     */
    virtual bool IsProblemSupported(const conv::ProblemDescription& problem,
                                    const ExecutionContext& ctx) const = 0;
    /** Forward (i.e., run inference on) problem through TunaNet
     *
     * This function takes in a problem, converts it to a numeric vector and feeds it TunaNet
     * for inference. Its output is a numeric vector that represents a probability distribution.
     * Each index in this vector represents a solver (as given in metadata.solver_map) and the
     * value at each index represents the probability that that solver is the fastest for given
     * convolution problem.
     *
     * @param problem Problem
     */
    std::vector<float> Forward(const conv::ProblemDescription& problem) const
    {
        std::vector<float> features       = ToFeatures(problem);
        std::vector<fdeep::tensor> output = model.predict({fdeep::tensor(input_shape, features)});
        std::vector<float> output_vector  = output.front().to_vector();
        std::vector<float> res(output_vector.begin() + offset, output_vector.end());
        return res;
    }

protected:
    const fdeep::model model;              // TunaNet model
    const fdeep::tensor_shape input_shape; // Shape of input tensor required by TunaNet
    const size_t offset; // Some TunaNet models output some "fluff" before they output kernel
                         // probabilites. This offset tells how many indexes of fluff need to
                         // be skipped in order to get to kernel probabilities.
    /** Path to model file for given GPU
     *
     * The model files for each GPU are identified by the GPU architecture. This function takes
     * in a GPU architecture and returns the path to its TunaNet model.
     *
     * @param arch Architecture
     */
    static std::string ModelPath(const std::string& arch)
    {
        const auto file_path = GetSystemDbPath() / (arch + ".tn.model");
        if(!fs::exists(file_path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load AI model file:" + file_path);
        return file_path.string();
    }
    /** Convert given problem to a numeric vector
     *
     * TunaNet takes in a numeric vector representing the given problem. The exact details
     * of this vector vary from one TunaNet model to another, and thus this function, which
     * converts a problem into a numeric vector that can be fed to TunaNet, must be implemented
     * by each sub-class of `Model` on its own.
     *
     * @param problem Problem
     */
    virtual std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const = 0;
};

class Gfx908Model final : public Model
{
public:
    Gfx908Model() : Model("gfx908") {}
    bool IsProblemSupported(const conv::ProblemDescription& problem,
                            const ExecutionContext& ctx) const override
    {
        // check if problem is of the kind TunaNet was trained to handle
        if(!problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Problem not 2D");
            return false;
        }
        if(problem.GetGroupCount() != 1)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Group count not 1");
            return false;
        }
        if(problem.GetInLayout() != "NCHW" && problem.GetInLayout() != "NCDHW")
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Layout not supported");
            return false;
        }
        if(problem.GetWeightsHeight() != problem.GetWeightsWidth())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Filters must be square (fil_h == fil_w)");
            return false;
        }
        if(problem.GetPadH() != problem.GetPadW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Padding must be equal along all axes");
            return false;
        }
        if(problem.GetKernelStrideH() != problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Stride must be equal along all axes");
            return false;
        }
        if(problem.GetDilationH() != 1 || problem.GetDilationW() != 1)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Dilation must be 1");
            return false;
        }
        const auto data_type = problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Unsupported data type");
            return false;
        }

        // check if the context is s.t. no solver TunaNet may predict would be applicable
        size_t applicable_solvers = 0;
        for(const auto& solver_name : metadata.solver_map)
        {
            auto solver_id = solver::Id{solver_name.second};
            auto solver    = solver_id.GetSolver();
            if(solver.IsApplicable(ctx, problem))
            {
                applicable_solvers++;
                break;
            }
        }
        if(applicable_solvers == 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: No solver that TunaNet may predict applies");
            return false;
        }
        return true;
    }

protected:
    std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const override
    {
        const bool isFwd            = problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? problem.GetInChannels() : problem.GetOutChannels()),
            static_cast<float>(isFwd ? problem.GetInDepth() : problem.GetOutDepth()),
            static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()),
            static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),
            static_cast<float>(problem.GetWeightsDepth()),
            static_cast<float>(problem.GetWeightsHeight()),
            static_cast<float>(problem.GetWeightsWidth()),
            static_cast<float>(isFwd ? problem.GetOutChannels() : problem.GetInChannels()),
            static_cast<float>(isFwd ? problem.GetOutDepth() : problem.GetInDepth()),
            static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()),
            static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),
            static_cast<float>(problem.GetOutBatchSize()),
            static_cast<float>(1), // TunaNet was trained on a dataset of 2D
                                   // problems where PadD was incorrectly set to 1
            static_cast<float>(problem.GetPadH()),
            static_cast<float>(problem.GetPadW()),
            static_cast<float>(1), // TunaNet was trained on a dataset of 2D
                                   // problems where StrideD was incorrectly set to 1
            static_cast<float>(problem.GetKernelStrideH()),
            static_cast<float>(problem.GetKernelStrideW()),
            static_cast<float>(problem.GetDilationH()),
            static_cast<float>(problem.GetDilationW()),
            static_cast<float>(metadata.EncodeLayout(problem.GetInLayout())),
            static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())),
            static_cast<float>(metadata.EncodeDirection(problem.GetDirection())),
            static_cast<float>(problem.GetGroupCount())};

        // normalize
        for(size_t i = 0; i < features.size(); ++i)
            features[i] = (features[i] - metadata.features_mean[i]) / metadata.features_std[i];

        return features;
    }
};

class Gfx90aModel final : public Model
{
public:
    Gfx90aModel() : Model("gfx90a") {}
    bool IsProblemSupported(const conv::ProblemDescription& problem,
                            const ExecutionContext& ctx) const override
    {
        // check if problem is of the kind TunaNet was trained to handle
        if(!problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Problem not 2D");
            return false;
        }
        if(problem.GetInLayout() != "NCHW")
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Layout not supported");
            return false;
        }
        if(problem.GetKernelStrideH() != problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Stride must be equal along all axes");
            return false;
        }
        if(problem.GetDilationH() != problem.GetDilationW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Dilation must be equal along all axes");
            return false;
        }
        if(problem.GetBias() != 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Bias must be 0");
            return false;
        }
        const auto data_type = problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Unsupported data type");
            return false;
        }

        // check if the context is s.t. no solver TunaNet may predict would be applicable
        size_t applicable_solvers = 0;
        for(const auto& solver_name : metadata.solver_map)
        {
            auto solver_id = solver::Id{solver_name.second};
            auto solver    = solver_id.GetSolver();
            if(solver.IsApplicable(ctx, problem))
            {
                applicable_solvers++;
                break;
            }
        }
        if(applicable_solvers == 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: No solver that TunaNet may predict applies");
            return false;
        }
        MIOPEN_LOG_I2("TunaNet Applicable");
        return true;
    }

protected:
    std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const override
    {
        const bool isFwd            = problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? problem.GetInChannels() : problem.GetOutChannels()),
            static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()),
            static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),
            static_cast<float>(isFwd ? problem.GetOutChannels() : problem.GetInChannels()),
            static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()),
            static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),
            static_cast<float>(problem.GetWeightsHeight()),
            static_cast<float>(problem.GetWeightsWidth()),
            static_cast<float>(problem.GetPadH()),
            static_cast<float>(problem.GetPadW()),
            static_cast<float>(problem.GetKernelStrideH()),
            static_cast<float>(problem.GetKernelStrideW()),
            static_cast<float>(problem.GetDilationH()),
            static_cast<float>(problem.GetDilationW()),
            static_cast<float>(problem.GetOutBatchSize()),
            static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())),
            static_cast<float>(metadata.EncodeDirection(problem.GetDirection())),
            static_cast<float>(problem.GetGroupCount())};

        // normalize
        for(size_t i = 0; i < features.size(); ++i)
            features[i] = (features[i] - metadata.features_mean[i]) / metadata.features_std[i];

        return features;
    }
};

class Gfx942Model final : public Model
{
public:
    Gfx942Model() : Model("gfx942") {}
    bool IsProblemSupported(const conv::ProblemDescription& problem,
                            const ExecutionContext& ctx) const override
    {
        // check if problem is of the kind TunaNet was trained to handle
        if(!problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Problem not 2D");
            return false;
        }
        if(problem.GetInLayout() != "NCHW" && problem.GetInLayout() != "NHWC")
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Layout not supported");
            return false;
        }
        if(problem.GetKernelStrideH() != problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Stride must be equal along all axes");
            return false;
        }
        if(problem.GetDilationH() != problem.GetDilationW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Dilation must be equal along all axes");
            return false;
        }
        if(problem.GetBias() != 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Bias must be 0");
            return false;
        }
        const auto data_type = problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Unsupported data type");
            return false;
        }

        // check if the context is s.t. no solver TunaNet may predict would be applicable
        size_t applicable_solvers = 0;
        for(const auto& solver_name : metadata.solver_map)
        {
            auto solver_id = solver::Id{solver_name.second};
            auto solver    = solver_id.GetSolver();
            if(solver.IsApplicable(ctx, problem))
            {
                applicable_solvers++;
                break;
            }
        }
        if(applicable_solvers == 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: No solver that TunaNet may predict applies");
            return false;
        }
        MIOPEN_LOG_I2("TunaNet Applicable");
        return true;
    }

protected:
    std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const override
    {
        const bool isFwd            = problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? problem.GetInChannels() : problem.GetOutChannels()),
            static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()),
            static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),
            static_cast<float>(isFwd ? problem.GetOutChannels() : problem.GetInChannels()),
            static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()),
            static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),
            static_cast<float>(problem.GetWeightsHeight()),
            static_cast<float>(problem.GetWeightsWidth()),
            static_cast<float>(problem.GetPadH()),
            static_cast<float>(problem.GetPadW()),
            static_cast<float>(problem.GetKernelStrideH()),
            static_cast<float>(problem.GetKernelStrideW()),
            static_cast<float>(problem.GetDilationH()),
            static_cast<float>(problem.GetDilationW()),
            static_cast<float>(problem.GetOutBatchSize()),
            static_cast<float>(metadata.EncodeLayout(problem.GetInLayout())),
            static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())),
            static_cast<float>(metadata.EncodeDirection(problem.GetDirection())),
            static_cast<float>(problem.GetGroupCount())};

        // normalize
        for(size_t i = 0; i < features.size(); ++i)
            features[i] =
                (features[i] - metadata.test_features_mean[i]) / metadata.test_features_std[i];

        return features;
    }
};

/**
 * @brief Check if device has legacy TunaNet model support (2D only)
 * @param device GPU device name (e.g., "gfx908", "gfx90a")
 * @return true if legacy 2D TunaNet models exist for this device
 */
static bool HasLegacyTunaNetSupport(const std::string& device)
{
    // Legacy 2D TunaNet models exist for these architectures only
    return (device == "gfx908" || device == "gfx90a");
}

/**
 * @brief Check if device has ND TunaNet model support (2D and 3D)
 * @param device GPU device name (e.g., "gfx942", "gfx950")
 * @return true if ND TunaNet models exist for this device
 */
static bool HasNDTunaNetSupport(const std::string& device)
{
    // ND TunaNet models (supporting both 2D and 3D) exist for these architectures
    return (device == "gfx942" || device == "gfx950");
}

std::unique_ptr<Model> GetModel(const std::string& device)
{
    // Legacy 2D models only - caller should check HasLegacyTunaNetSupport() first
    // No default fallback to avoid using inappropriate models for unsupported architectures
    if(device == "gfx90a")
        return std::make_unique<Gfx90aModel>();
    if(device == "gfx908")
        return std::make_unique<Gfx908Model>();

    // No model for unsupported architectures - return nullptr to trigger WTI fallback
    MIOPEN_LOG_I2("No legacy TunaNet model for device: " << device);
    return nullptr;
}

/**
 * @brief Retrieve cached AI heuristics results if available
 * @param problem Convolution problem description
 * @param device GPU device name
 * @param is3d Whether this is a 3D convolution problem
 * @return Cached solver IDs, or empty vector if no cache hit
 */
std::vector<uint64_t>
GetCachedPrediction(const conv::ProblemDescription& problem, const std::string& device, bool is3d)
{
    std::string est_name = is3d ? (":memory:3d_" + device) : (":memory:" + device);
    auto& db             = AnyRamDb::GetCached(est_name);
    auto db_res          = db.FindRecord(problem);

    if(!db_res)
    {
        return {};
    }

    const std::string model_type = is3d ? "3D " : "";
    MIOPEN_LOG_I2("Cached " << model_type << "heuristic (TunaNet) result found");

    std::vector<uint64_t> db_sol(db_res->size());
    std::transform(db_res->begin(), db_res->end(), db_sol.begin(), [](std::any id) {
        return std::any_cast<uint64_t>(id);
    });

    if(miopen::IsLogging(LoggingLevel::Info2))
    {
        std::stringstream ss;
        for(auto& id : db_sol)
            ss << solver::Id{id}.ToString() << " ID:" << id << ", ";
        MIOPEN_LOG_I2("Cached " << model_type << "solvers: " << ss.str());
    }

    return db_sol;
}

/**
 * @brief Store AI heuristics results in cache for future use
 * @param problem Convolution problem description
 * @param device GPU device name
 * @param is3d Whether this is a 3D convolution problem
 * @param any_sol Vector of solver IDs to cache
 */
void StorePredictionCache(const conv::ProblemDescription& problem,
                          const std::string& device,
                          std::vector<std::any>& any_sol)
{
    const bool is3d      = problem.Is3d();
    std::string est_name = is3d ? (":memory:3d_" + device) : (":memory:" + device);
    auto& db             = AnyRamDb::GetCached(est_name);
    db.StoreRecord(problem, any_sol);
}

/**
 * @brief Result structure for processed AI heuristics predictions
 */
struct PredictionResult
{
    std::vector<uint64_t> solver_ids;     ///< Sorted solver IDs by probability
    std::vector<std::any> any_solver_ids; ///< Same IDs in std::any format for caching
};

/**
 * @brief Process raw model predictions into sorted solver recommendations
 * @param predictions Raw probability scores from TunaNet model
 * @param solver_map Mapping from solver indices to solver names
 * @param is3d Whether processing 3D or 2D predictions (for logging)
 * @return Sorted solver IDs with highest probability first
 */
PredictionResult ProcessPredictions(const std::vector<float>& predictions,
                                    const std::unordered_map<size_t, std::string>& solver_map,
                                    bool use_nd)
{
    // Debug: Print raw prediction probabilities
    const std::string model_type = use_nd ? "ND " : "";

    // Log individual solver predictions with scores
    if(miopen::IsLogging(LoggingLevel::Info2) && !solver_map.empty())
    {
        MIOPEN_LOG_I2("=== " << model_type << "Solver Predictions (Logits) ===");
        for(size_t idx = 0; idx < predictions.size() && idx < solver_map.size(); idx++)
        {
            if(solver_map.find(idx) != solver_map.end())
            {
                MIOPEN_LOG_I2("  [" << idx << "] " << solver_map.at(idx) << " = "
                                    << predictions[idx]);
            }
        }
    }

    // Sort solvers in order of their probabilities
    std::vector<std::pair<int, float>> sort_res(predictions.size());
    for(auto idx = 0ULL; idx < predictions.size(); idx++)
        sort_res[idx] = {idx, predictions[idx]};

    const auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) -> bool {
        return a.second > b.second;
    };
    std::sort(sort_res.begin(), sort_res.end(), cmp);

    // Log sorted results (top solvers)
    if(miopen::IsLogging(LoggingLevel::Info2) && !solver_map.empty())
    {
        MIOPEN_LOG_I2("=== " << model_type << "Top Ranked Solvers ===");
        for(size_t i = 0; i < std::min(size_t(3), sort_res.size()); i++)
        {
            const auto idx   = sort_res[i].first;
            const auto score = sort_res[i].second;
            if(solver_map.find(idx) != solver_map.end())
            {
                MIOPEN_LOG_I2("  Rank " << (i + 1) << ": " << solver_map.at(idx)
                                        << " (score: " << score << ")");
            }
        }
    }

    // Map solver idx to solver id and then to anysolver
    PredictionResult result;

    for(const auto& kinder : sort_res)
    {
        const auto id = kinder.first;
        // const auto prob = kinder.second; // Unused for now

        // Check if solver index exists in map
        if(solver_map.find(id) == solver_map.end())
        {
            MIOPEN_LOG_I2("Invalid solver index " << id << " not found in solver map");
            continue;
        }

        const auto& solver_name = solver_map.at(id);

        const auto sol_id = solver::Id{solver_name};
        if(!sol_id.IsValid())
        {
            MIOPEN_LOG_I2("Invalid " << model_type << "solver " << solver_name << " removed");
            continue;
        }

        result.solver_ids.push_back(sol_id.Value());
        result.any_solver_ids.push_back(sol_id.Value());
    }

    return result;
}

static void PromoteSolverToFront(PredictionResult& result, const char* solver_name)
{
    const auto solver_id = solver::Id{solver_name};
    if(!solver_id.IsValid())
        return;

    const auto target = solver_id.Value();
    const auto it     = std::find(result.solver_ids.begin(), result.solver_ids.end(), target);
    if(it == result.solver_ids.end() || it == result.solver_ids.begin())
        return;

    result.solver_ids.erase(it);
    result.solver_ids.insert(result.solver_ids.begin(), target);

    result.any_solver_ids.clear();
    result.any_solver_ids.reserve(result.solver_ids.size());
    for(const auto id : result.solver_ids)
        result.any_solver_ids.push_back(id);
}

/**
 * @brief Common logic for running TunaNet prediction and caching results
 * @param problem Convolution problem description
 * @param device GPU device name
 * @param is3d Whether this is a 3D or 2D problem
 * @param predictions Raw model predictions (solver probabilities)
 * @param solver_map Mapping from solver indices to solver names
 * @return Sorted solver IDs with highest probability first
 */
static std::vector<uint64_t>
ProcessAndCachePredictions(const conv::ProblemDescription& problem,
                           const std::string& device,
                           const bool& use_nd,
                           const std::vector<float>& predictions,
                           const std::unordered_map<size_t, std::string>& solver_map)
{
    const std::string model_type = use_nd ? "ND " : "";

    // Process predictions (sort by probability, filter invalid solvers)
    auto result = ProcessPredictions(predictions, solver_map, use_nd);

    // TunaNet override: promote GemmBwdRest for point-output backward-data problems
    if(problem.Is3d() && conv::IsBwdDataPointOutput3dStrideEqFilter(problem))
    {
        PromoteSolverToFront(result, "GemmBwdRest");
        MIOPEN_LOG_I2("TunaNet override: promoting GemmBwdRest for point-output backward-data");
    }

    // Cache results for future use
    StorePredictionCache(problem, device, result.any_solver_ids);

    // Log results if verbose logging enabled
    if(miopen::IsLogging(LoggingLevel::Info2))
    {
        std::stringstream ss;
        for(auto& id : result.solver_ids)
            ss << solver::Id{id}.ToString() << " ID:" << id << ", ";
        MIOPEN_LOG_I2(model_type << "TunaNet Result: " << ss.str());
    }

    return result.solver_ids;
}

std::vector<uint64_t> PredictSolver(const conv::ProblemDescription& problem,
                                    const ExecutionContext& ctx,
                                    const std::string& device)
{
    const bool is3d = problem.Is3d();
    const bool is2d = problem.Is2d();

    // Check cache FIRST - avoids expensive model creation if we have cached results
    {
        auto cached_result = GetCachedPrediction(problem, device, is3d);
        if(!cached_result.empty())
            return cached_result;
    }

    // Strategy:
    // 1. Try ND model first (for gfx942/gfx950, supports both 2D and 3D)
    // 2. Fall back to legacy model (for gfx908/gfx90a, 2D only)
    // 3. Return empty vector to trigger WTI fallback for unsupported architectures

    // Try ND model first (preferred for gfx942/gfx950)
    if((is2d || is3d) && HasNDTunaNetSupport(device))
    {
        int dim                        = is3d ? 3 : 2;
        std::unique_ptr<ModelND> model = GetNDModel(device, dim);

        if(model && model->IsProblemSupported(problem, ctx))
        {
            MIOPEN_LOG_I2("Evaluating ND TunaNet for " << device);
            std::vector<float> predictions = model->Forward(problem);
            return ProcessAndCachePredictions(
                problem, device, true, predictions, model->GetSolverMap());
        }
        // If ND model failed for this architecture, don't try legacy - go to WTI
        MIOPEN_LOG_I2("ND TunaNet not applicable for this problem on " << device);
        return {};
    }

    // Fall back to legacy 2D model (for gfx908/gfx90a only)
    if(is2d && HasLegacyTunaNetSupport(device))
    {
        std::unique_ptr<Model> model = GetModel(device);

        if(model && model->IsProblemSupported(problem, ctx))
        {
            MIOPEN_LOG_I2("Evaluating legacy TunaNet for " << device);
            std::vector<float> predictions = model->Forward(problem);
            return ProcessAndCachePredictions(
                problem, device, false, predictions, model->metadata.solver_map);
        }
        MIOPEN_LOG_I2("Legacy TunaNet not applicable for this problem on " << device);
        return {};
    }

    // No TunaNet model available for this device/problem combination
    // Return empty vector to trigger WTI fallback
    MIOPEN_LOG_I2("No TunaNet model available for " << device << ", falling back to WTI");
    return {};
}

// MetadataND implementation moved to metadata_nd.cpp

class TunaNetNDModel : public ModelND
{
private:
    const std::string device_name; // Device name (e.g., "gfx942", "gfx950")

public:
    MetadataND metadata;

    explicit TunaNetNDModel(const std::string& device, const int& dim)
        : device_name(device), metadata(MetadataND(device, dim))
    {
        MIOPEN_LOG_I2("TunaNetNDModel initialized for device: " << device_name << " dim: " << dim);
    }

    std::vector<float> Forward(const conv::ProblemDescription& problem) const override
    {
        std::vector<float> features = ToFeatures(problem);
        MIOPEN_LOG_I2("TunaNetNDModel: Extracted " << features.size() << " features");

        // Use fdeep to run TunaNetND inference
        const int dim                = problem.Is3d() ? 3 : 2;
        const std::string model_path = ModelNDPath(device_name, dim);
        const auto model             = fdeep::load_model(model_path, true, fdeep::dev_null_logger);
        MIOPEN_LOG_I2("TunaNetNDModel: Loaded fdeep model from " << model_path << ".");

        // Convert features to fdeep tensor
        const auto input_tensor = fdeep::tensor(fdeep::tensor_shape(features.size()), features);
        const auto result       = model.predict({input_tensor});

        // Extract predictions from result
        const auto predictions = result[0].to_vector();
        MIOPEN_LOG_I2("TunaNetNDModel: TunaNetND returned " << predictions.size()
                                                            << " predictions");
        return predictions;
    }

    const std::unordered_map<size_t, std::string>& GetSolverMap() const override
    {
        return metadata.GetSolverMap();
    }

    bool IsProblemSupported(const conv::ProblemDescription& problem,
                            const ExecutionContext& /*ctx*/) const override
    {
        // Get the model's spatial dimension
        const int model_dim = metadata.GetSpatialDim();

        // Check if problem dimension matches model dimension
        if(model_dim == 3 && !problem.Is3d())
        {
            MIOPEN_LOG_I2("TunaNetND 3D model cannot handle 2D problem");
            return false;
        }
        if(model_dim == 2 && !problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNetND 2D model cannot handle 3D problem");
            return false;
        }

        // Reject problems that are neither 2D nor 3D
        if(!problem.Is3d() && !problem.Is2d())
        {
            MIOPEN_LOG_I2("Problem is neither 2D nor 3D");
            return false;
        }

        MIOPEN_LOG_I2("Problem dimension matches TunaNetND model dimension");
        return true;
    }

protected:
    std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const override
    {
        const bool isFwd = problem.GetDirection() == conv::Direction::Forward;

        std::vector<float> features = {};
        if(problem.Is2d()) // 2d version as in the
        {
            features = {
                // Input dimensions
                static_cast<float>(isFwd ? problem.GetInChannels()
                                         : problem.GetOutChannels()), // in_channels
                static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()), // in_h
                static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),   // in_w

                // Output dimensions
                static_cast<float>(isFwd ? problem.GetOutChannels()
                                         : problem.GetInChannels()), // out_channels
                static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()), // out_h
                static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),   // out_w

                // Filter dimensions
                static_cast<float>(problem.GetWeightsHeight()), // fil_h
                static_cast<float>(problem.GetWeightsWidth()),  // fil_w

                // Padding
                static_cast<float>(problem.GetPadH()), // pad_h
                static_cast<float>(problem.GetPadW()), // pad_w

                // Stride
                static_cast<float>(problem.GetKernelStrideH()), // stride_h
                static_cast<float>(problem.GetKernelStrideW()), // stride_w

                // Dilation
                static_cast<float>(problem.GetDilationH()), // dilation_h
                static_cast<float>(problem.GetDilationW()), // dilation_w

                // Batch size
                static_cast<float>(problem.GetOutBatchSize()), // batchsize

                // Layout encodings
                static_cast<float>(metadata.EncodeInLayout(problem.GetInLayout())), // in_layout
                static_cast<float>(
                    metadata.EncodeFilLayout(problem.GetWeightsLayout())),            // fil_layout
                static_cast<float>(metadata.EncodeOutLayout(problem.GetOutLayout())), // out_layout

                // Precision encoding
                static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())), // precision

                // Direction encoding
                static_cast<float>(metadata.EncodeDirection(problem.GetDirection())), // direction

                // Group count
                static_cast<float>(problem.GetGroupCount()), // group_count
            };
        }
        else if(problem.Is3d())
        {
            features = {
                // Input dimensions
                static_cast<float>(isFwd ? problem.GetInChannels()
                                         : problem.GetOutChannels()), // in_channels
                static_cast<float>(isFwd ? problem.GetInDepth() : problem.GetOutDepth()),   // in_d
                static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()), // in_h
                static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),   // in_w

                // Output dimensions
                static_cast<float>(isFwd ? problem.GetOutChannels()
                                         : problem.GetInChannels()), // out_channels
                static_cast<float>(isFwd ? problem.GetOutDepth() : problem.GetInDepth()),   // out_d
                static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()), // out_h
                static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),   // out_w

                // Filter dimensions
                static_cast<float>(problem.GetWeightsDepth()),  // fil_d
                static_cast<float>(problem.GetWeightsHeight()), // fil_h
                static_cast<float>(problem.GetWeightsWidth()),  // fil_w

                // Padding
                static_cast<float>(problem.GetPadD()), // pad_d
                static_cast<float>(problem.GetPadH()), // pad_h
                static_cast<float>(problem.GetPadW()), // pad_w

                // Stride
                static_cast<float>(problem.GetKernelStrideD()), // stride_d
                static_cast<float>(problem.GetKernelStrideH()), // stride_h
                static_cast<float>(problem.GetKernelStrideW()), // stride_w

                // Batch size
                static_cast<float>(problem.GetOutBatchSize()), // batchsize

                // Layout encodings
                static_cast<float>(metadata.EncodeInLayout(problem.GetInLayout())), // in_layout
                static_cast<float>(
                    metadata.EncodeFilLayout(problem.GetWeightsLayout())),            // fil_layout
                static_cast<float>(metadata.EncodeOutLayout(problem.GetOutLayout())), // out_layout

                // Precision encoding
                static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())), // precision

                // Direction encoding
                static_cast<float>(metadata.EncodeDirection(problem.GetDirection())), // direction

                // Group count
                static_cast<float>(problem.GetGroupCount()), // group_count
            };
        }
        else
        {
            MIOPEN_LOG_I2("Unsupported problem type for ND feature extraction");
        }

        MIOPEN_LOG_I2("TunaNetNDModel: Extracted " << features.size() << " features");
        return features;
    }

    static std::string ModelNDPath(const std::string& device, const int& dim)
    {
        auto file_path = GetSystemDbPath();

        if(dim == 3)
        {
            file_path = file_path / (device + "_3d.tn.model");
        }
        else if(dim == 2)
        {
            file_path = file_path / (device + ".tn.model");
        }
        else
        {
            MIOPEN_LOG_I2("Unsupported dim:" << dim);
        }

        if(!fs::exists(file_path))
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "Unable to load ND AI model file: " + file_path.string());
        }
        return file_path.string();
    }
};

std::unique_ptr<ModelND> GetNDModel(const std::string& device, const int& dim)
{
    MIOPEN_LOG_I2("GetNDModel called for device: " << device << " dim: " << dim);

    // Check if device is supported using the centralized helper function
    if(!HasNDTunaNetSupport(device))
    {
        MIOPEN_LOG_I2("Device " << device << " not supported for ND models");
        return nullptr;
    }

    try
    {
        // Pass device name to constructor - it will append "_3d" internally if 3D
        auto model = std::make_unique<TunaNetNDModel>(device, dim);
        MIOPEN_LOG_I2("Successfully created ND model for device: " << device);
        return model;
    }
    catch(const std::exception& e)
    {
        MIOPEN_LOG_E("Exception during ND model construction: " << e.what());
        return nullptr;
    }
    catch(...)
    {
        MIOPEN_LOG_E("Unknown exception during ND model construction");
        return nullptr;
    }
}

} // namespace immed_mode

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

//////////////////////////////////////////////////////////////////////////////////////////////////////

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace tuning {

Metadata::Metadata(const std::string& arch, const std::string& solver)
{
    const nlohmann::json metadata =
        common::LoadJSON(GetSystemDbPath() / (arch + "_" + solver + "_metadata.ktn.model"));
    predict_type = metadata["predict_type"].get<std::size_t>();
    num_tuning_params =
        metadata["num_tuning_params"].get<std::unordered_map<std::string, std::size_t>>();
    tuning_decodings =
        metadata["decodings"]["tunings"].get<std::unordered_map<std::string, std::string>>();
}

class Model
{
public:
    Metadata metadata;
    Model(const std::string& arch, const std::string& solver)
        : metadata(Metadata(arch, solver)),
          encoder(fdeep::load_model(EncoderPath(arch, solver), true, fdeep::dev_null_logger)),
          decoder(fdeep::load_model(DecoderPath(arch, solver), true, fdeep::dev_null_logger))
    {
    }
    virtual ~Model() = default;
    /**
     * Encode the input features into a "context" tensor
     *
     * @param features Input features
     * @param dim Dimension (must be equal to len(features) if transform
     *            is True and sqrt(len(features)) otherwise)
     * @param transform Reshape input features into a square matrix?
     */
    fdeep::tensors Encode(const std::vector<float>& features, std::size_t dim, bool transform) const
    {
        // if transform==True, reshape input features into a matrix of `dim x dim` dimensions.
        // otherwise, have them as a vector of size `dim`.
        const auto tensor_shape_depth = transform ? dim : 1;
        fdeep::tensor input_tensor =
            fdeep::tensor(fdeep::tensor_shape(dim, tensor_shape_depth), features);

        return encoder.predict({input_tensor});
    }
    /**
     * Decode the next token based on the previous token and the encoded context.
     *
     * Decoder predicts the next token based on the previous token and the context predicted
     * by the Encoder. A token is a representation of a kernel parameter, i.e., each unique
     * token maps to a unique kernel parameter, with the only exception being the token '-1'
     * which signals the end of the decoding process (i.e., all kernel parameters have been
     * obtained).
     *
     * @param prev_token Previous token
     * @param context Context vector obtained from encoder
     */
    fdeep::tensors Decode(const float prev_token, const fdeep::tensors& context) const
    {
        return decoder.predict(
            {{fdeep::tensor(fdeep::tensor_shape(1), std::vector<float>(1, prev_token)),
              context[0],
              context[1],
              context[2],
              context[3]}});
    }

private:
    const fdeep::model encoder;
    const fdeep::model decoder;
    static std::string EncoderPath(const std::string& arch, const std::string& solver)
    {
        const auto path = GetSystemDbPath() / (arch + "_" + solver + "_encoder.ktn.model");
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        return path.string();
    }
    static std::string DecoderPath(const std::string& arch, const std::string& solver)
    {
        const auto path = GetSystemDbPath() / (arch + "_" + solver + "_decoder.ktn.model");
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        return path.string();
    }
};

/**
 * Return the KernelTuningNet model for given architecture and solver
 *
 * KernelTuningNet models are specific to each solver and are fine-tuned for each
 * GPU skew. This function constructs the KernelTuningNet model for the given
 * architecture and solver and stores it in a static map, so that the next time
 * the same model is required it doesn't have to be constructed anew.
 *
 * @param arch GPU Architecture
 * @param solver Solver
 */
std::shared_ptr<Model> GetModel(const std::string& arch, const std::string& solver)
{
    static std::map<std::string, std::shared_ptr<Model>> models;
    auto it = models.find(solver);

    auto model_arch = arch;
    if(model_arch == "gfx950")
        model_arch = "gfx942"; // use gfx942 model for gfx950 until we have a gfx950 model

    if(it == models.end())
    {
        std::shared_ptr<Model> model = std::make_shared<Model>(model_arch, solver);
        models[solver]               = model;
        return model;
    }
    else
    {
        return it->second;
    }
}

/**
 * Set kernel parameters for given solver
 *
 * @param arch GPU Architecture
 * @param solver Solver
 * @param direction Convolution Direction
 * @param features Input features for KernelTuningNet model
 * @param transform_features Whether or not to reshape features into a square
 *                           matrix before feeding them to KernelTuningNet
 * @param validator A boolean function that accepts an index `i` and a string `v`, and returns
 *                  True iff `v` is a valid kernel parameter value at index `i`
 */
bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    miopen::conv::Direction direction,
                    const std::vector<float>& features,
                    bool transform_features,
                    std::function<bool(std::size_t, std::string)> validator)
{
    using model_type = decltype(GetModel(arch, solver));
    model_type model;
    try
    {
        model = GetModel(arch, solver);
    }
    catch(const miopen::Exception& ex)
    {
        MIOPEN_LOG_I2("[Warning] Could not retrieve model: (" << ex.what() << ")");
        return false;
    }

    // get context
    int dim = 0;
    if(transform_features)
        dim = std::sqrt(features.size());
    else
        dim = features.size();
    auto start             = std::chrono::high_resolution_clock::now();
    fdeep::tensors context = model->Encode(features, dim, transform_features);
    float decoder_input    = 0.0;

    // set direction string
    std::string dir;
    switch(direction)
    {
    case miopen::conv::Direction::Forward: dir = "fwd"; break;
    case miopen::conv::Direction::BackwardData: dir = "bwd"; break;
    case miopen::conv::Direction::BackwardWeights: dir = "wrw"; break;
    default: return false;
    }

    // run decoder to set kernel parameters
    for(size_t i = 0, num_tuning_params = 1; i < num_tuning_params; ++i)
    {
        if(i == 0 && (model->metadata.predict_type == 0u))
            num_tuning_params = model->metadata.num_tuning_params[dir];

        fdeep::tensors decoder_output = model->Decode(decoder_input, context);
        auto token_scores             = decoder_output[0].to_vector(); // token_scores[k] gives the
                                                                       // score of the k-th token
        // order tokens according to their scores
        std::priority_queue<std::pair<float, int>> pq;
        for(auto j = 0ULL; j < token_scores.size(); j++)
        {
            pq.push(std::make_pair(token_scores[j], j)); // sort by value at index
        }

        // find a token whose value is a valid kernel parameter for the i-th position
        int output_token_index = -1;
        while(!pq.empty())
        {
            // get the token with the highest score and look up its value
            int token         = pq.top().second;
            std::string value = model->metadata.tuning_decodings[std::to_string(token)];
            pq.pop();

            if(value == "-1") // if token-value is "-1", then decoding has finished
            {
                auto stop     = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                MIOPEN_LOG_I2("KTN ran for " << duration.count() << " micro-seconds. Ended at -1.");
                return false;
            }
            if(validator(i, value)) // if token-value is a valid kernel parameter, it's set
            {
                output_token_index =
                    token; // index with largest value that is valid = predicted index
                if(i == 0 && model->metadata.predict_type != 0u)
                    num_tuning_params = model->metadata.num_tuning_params[value];
                break;
            }
        }
        decoder_input = float(output_token_index);
        context       = {decoder_output.begin() + 1, decoder_output.end()};
    }

    auto stop     = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    MIOPEN_LOG_I2("KTN ran for " << duration.count() << " micro-seconds");
    return true;
}

namespace candidate_selection {

// Helper to load and cache fdeep models
const fdeep::model& GetFdeepModel(const std::string& path, const std::string& key)
{
    static std::map<std::string, std::unique_ptr<fdeep::model>> models;
    auto it = models.find(key);
    if(it == models.end())
    {
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load model file: " + path);
        auto model =
            std::make_unique<fdeep::model>(fdeep::load_model(path, true, fdeep::dev_null_logger));
        auto& ref   = *model;
        models[key] = std::move(model);
        return ref;
    }
    return *it->second;
}

std::vector<float> EncodeInputFeaturesWithFdeep(const std::vector<float>& features,
                                                const std::string& arch,
                                                const std::string& solver)
{

    fdeep::tensor input_tensor(fdeep::tensor_shape(features.size()), features);
    std::string key = arch + "_" + solver + "_input_encoder";
    std::string path =
        (GetSystemDbPath() / (arch + "_" + solver + "_input_encoder.tn.model")).string();

    MIOPEN_LOG_I2("Loading a Two-towers submodel from: " << path);
    auto tensors = GetFdeepModel(path, key).predict({input_tensor});
    if(tensors.empty())
        MIOPEN_THROW(miopenStatusInternalError, "Input encoder returned empty tensor list");
    return tensors[0].to_vector();
}

std::vector<std::vector<float>>
EncodeKernelConfigsWithFdeep(const std::vector<std::vector<float>>& encoded_candidates,
                             const std::string& arch,
                             const std::string& solver)
{

    if(encoded_candidates.empty() || encoded_candidates[0].empty())
        MIOPEN_THROW(miopenStatusInternalError,
                     "Empty candidates provided to kernel config encoder");

    std::string key = arch + "_" + solver + "_kernel_config_encoder";
    std::string path =
        (GetSystemDbPath() / (arch + "_" + solver + "_kernel_config_encoder.tn.model")).string();

    MIOPEN_LOG_I2("Loading a Two-towers submodel from: " << path);
    const auto& model = GetFdeepModel(path, key);

    // By default, use predict_multi (multi-threaded); use single-threaded loop only if env var is
    // set
    bool use_single = env::enabled(MIOPEN_AI_FDEEP_USE_SINGLE_THREAD_PREDICT);

    std::vector<std::vector<float>> result;
    std::vector<fdeep::tensors> inputs_vec;
    inputs_vec.reserve(encoded_candidates.size());
    for(const auto& candidate : encoded_candidates)
    {
        fdeep::tensor t(fdeep::tensor_shape(candidate.size()), candidate);
        inputs_vec.push_back(fdeep::tensors{t}); // wrap tensor in a vector
    }
    auto outputs = model.predict_multi(inputs_vec, !use_single); // parallelly = !use_single
    if(outputs.size() != inputs_vec.size())
        MIOPEN_THROW(miopenStatusInternalError, "predict_multi returned wrong number of outputs");
    for(const auto& out : outputs)
    {
        if(out.empty())
            MIOPEN_THROW(miopenStatusInternalError,
                         "Kernel config encoder returned empty tensor list");
        result.push_back(out[0].to_vector());
    }
    return result;
}
} // namespace candidate_selection

} // namespace tuning
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
} // namespace ai
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
