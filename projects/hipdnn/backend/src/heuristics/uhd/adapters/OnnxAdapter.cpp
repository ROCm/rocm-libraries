// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "OnnxAdapter.hpp"

#include <hipdnn_data_sdk/logging/Logger.hpp>

#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

#ifdef HIPDNN_ENABLE_ONNX

Ort::Env& OnnxAdapter::getEnv()
{
    // Process-global ONNX Runtime environment (singleton pattern).
    // ORT_LOGGING_LEVEL_WARNING keeps runtime logs minimal.
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hipdnn");
    return env;
}

std::unique_ptr<OnnxAdapter> OnnxAdapter::load(const std::string& modelPath,
                                                const std::string& expectedFeaturesHash)
{
    // 1. Check file exists
    if(!std::filesystem::exists(modelPath))
    {
        HIPDNN_SDK_LOG_ERROR("OnnxAdapter: Model file not found: " << modelPath);
        return nullptr;
    }

    try
    {
        // 2. Create session options
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1); // Single-threaded (selection is not parallel)
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // 3. Try ROCm execution provider, fall back to CPU
        try
        {
            OrtROCMProviderOptions rocmOptions{};
            options.AppendExecutionProvider_ROCM(rocmOptions);
        }
        catch(const std::exception& e)
        {
            HIPDNN_SDK_LOG_WARN("OnnxAdapter: ROCm execution provider not available ("
                                << e.what() << "), using CPU");
        }

        // 4. Load session
        auto session = std::make_shared<Ort::Session>(getEnv(), modelPath.c_str(), options);

        // 5. Get input metadata
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr inputNameAlloc = session->GetInputNameAllocated(0, allocator);
        std::string inputName(inputNameAlloc.get());

        Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
        auto tensorInfo               = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape    = tensorInfo.GetShape();

        // 6. Verify input shape: must be 2D with batch=1
        if(shape.size() != 2 || shape[0] != 1)
        {
            std::ostringstream oss;
            oss << "OnnxAdapter: Model expects 2D input with batch=1, got shape=[";
            for(size_t i = 0; i < shape.size(); ++i)
            {
                if(i > 0)
                {
                    oss << ", ";
                }
                oss << shape[i];
            }
            oss << "]. Model: " << modelPath;
            HIPDNN_SDK_LOG_ERROR(oss.str());
            return nullptr;
        }

        const size_t numFeatures = static_cast<size_t>(shape[1]);

        // 7. Get output metadata
        Ort::AllocatedStringPtr outputNameAlloc = session->GetOutputNameAllocated(0, allocator);
        std::string outputName(outputNameAlloc.get());

        // 8. Log features_hash (informational - no validation against model metadata yet)
        if(!expectedFeaturesHash.empty())
        {
            HIPDNN_SDK_LOG_INFO("OnnxAdapter: Loaded model with "
                                << numFeatures << " features, expected hash: " << expectedFeaturesHash
                                << ", model: " << modelPath);
        }
        else
        {
            HIPDNN_SDK_LOG_INFO("OnnxAdapter: Loaded model with "
                                << numFeatures << " features, model: " << modelPath);
        }

        // 9. Create adapter
        auto adapter = std::unique_ptr<OnnxAdapter>(new OnnxAdapter(numFeatures, expectedFeaturesHash));
        adapter->_session    = std::move(session);
        adapter->_inputName  = std::move(inputName);
        adapter->_outputName = std::move(outputName);

        return adapter;
    }
    catch(const Ort::Exception& e)
    {
        HIPDNN_SDK_LOG_ERROR("OnnxAdapter: ONNX Runtime exception: "
                             << e.what() << ", model: " << modelPath);
        return nullptr;
    }
    catch(const std::exception& e)
    {
        HIPDNN_SDK_LOG_ERROR("OnnxAdapter: Exception loading model: "
                             << e.what() << ", model: " << modelPath);
        return nullptr;
    }
}

#else

std::unique_ptr<OnnxAdapter> OnnxAdapter::load(const std::string& modelPath,
                                                const std::string& expectedFeaturesHash)
{
    (void)modelPath;
    (void)expectedFeaturesHash;

    HIPDNN_SDK_LOG_ERROR("OnnxAdapter: ONNX Runtime dependency not available. "
                         << "The ONNX adapter is opt-in and dependency-gated per RFC 0019 §7.3. "
                         << "To use ONNX models, build with -DHIPDNN_ENABLE_ONNX=ON and ensure "
                         << "ONNX Runtime is installed. Model path: " << modelPath);
    return nullptr;
}

#endif

OnnxAdapter::OnnxAdapter(size_t numFeatures, std::string featuresHash)
    : _numFeatures(numFeatures), _featuresHash(std::move(featuresHash))
{
}

double OnnxAdapter::score(const std::vector<double>& features) const
{
#ifdef HIPDNN_ENABLE_ONNX
    if(features.size() != _numFeatures)
    {
        std::ostringstream oss;
        oss << "OnnxAdapter: Feature count mismatch. Expected " << _numFeatures << ", got "
            << features.size();
        throw std::invalid_argument(oss.str());
    }

    try
    {
        // Create input tensor shape: [1, numFeatures]
        std::vector<int64_t> inputShape = {1, static_cast<int64_t>(_numFeatures)};

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // ONNX Runtime CreateTensor expects non-const pointer (does not modify data)
        Ort::Value inputTensor = Ort::Value::CreateTensor<double>(
            memoryInfo,
            const_cast<double*>(features.data()),
            features.size(),
            inputShape.data(),
            inputShape.size());

        // Run inference
        const char* inputNames[]  = {_inputName.c_str()};
        const char* outputNames[] = {_outputName.c_str()};

        std::vector<Ort::Value> outputTensors
            = _session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

        // Extract score (assume output is float scalar or 1D tensor with one element)
        const float* outputData = outputTensors[0].GetTensorData<float>();
        return static_cast<double>(outputData[0]);
    }
    catch(const Ort::Exception& e)
    {
        std::ostringstream oss;
        oss << "OnnxAdapter: ONNX Runtime exception during inference: " << e.what();
        throw std::runtime_error(oss.str());
    }
#else
    (void)features;
    throw std::logic_error(
        "OnnxAdapter::score() called but ONNX Runtime not available. Build with "
        "-DHIPDNN_ENABLE_ONNX=ON.");
#endif
}

} // namespace hipdnn_backend::heuristics::uhd
