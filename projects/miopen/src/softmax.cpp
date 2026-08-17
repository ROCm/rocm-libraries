// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "miopen/execution_context.hpp"
#include "miopen/softmax/problem_description.hpp"
#include <miopen/kernel_cache.hpp>
#include <miopen/softmax.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/tensor.hpp>

#include <miopen/softmax/invoke_params.hpp>
#include <miopen/softmax/solvers.hpp>
#include <miopen/find_solution.hpp>

#include <nlohmann/json.hpp>

namespace miopen {

namespace softmax {
miopen::PerformanceDb GetDb(const miopen::ExecutionContext& context,
                            const miopen::softmax::ProblemDescriptionTag&)
{
    return {
        DbKinds::PerfDb, context.GetPerfDbPath("softmax"), context.GetUserPerfDbPath("softmax")};
}
} // namespace softmax

extern "C" miopenStatus_t miopenCreateSoftmaxDescriptor(miopenSoftmaxDescriptor_t* softmaxDesc)
{
    MIOPEN_LOG_FUNCTION(softmaxDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(softmaxDesc);
        desc       = new miopen::SoftmaxDescriptor();
    });
}

extern "C" miopenStatus_t miopenSetSoftmaxDescriptor(miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float alpha,
                                                     float beta,
                                                     miopenSoftmaxAlgorithm_t algorithm,
                                                     miopenSoftmaxMode_t mode)
{

    MIOPEN_LOG_FUNCTION(softmaxDesc, alpha, beta, algorithm, mode);
    return miopen::try_(
        [&] { miopen::deref(softmaxDesc).SetParams(alpha, beta, algorithm, mode); });
}

extern "C" miopenStatus_t miopenGetSoftmaxDescriptor(const miopenSoftmaxDescriptor_t softmaxDesc,
                                                     float* alpha,
                                                     float* beta,
                                                     miopenSoftmaxAlgorithm_t* algorithm,
                                                     miopenSoftmaxMode_t* mode)
{
    MIOPEN_LOG_FUNCTION(softmaxDesc);
    return miopen::try_([&] {
        *alpha     = miopen::deref(softmaxDesc).GetAlpha();
        *beta      = miopen::deref(softmaxDesc).GetBeta();
        *algorithm = miopen::deref(softmaxDesc).GetAlgorithm();
        *mode      = miopen::deref(softmaxDesc).GetMode();
    });
}

std::ostream& operator<<(std::ostream& stream, const SoftmaxDescriptor& x)
{
    stream << "softmax," << "alpha" << x.GetAlpha() << ",beta" << x.GetBeta() << ",algorithm"
           << x.GetAlgorithm() << ",mode" << x.GetMode() << ",";

    return stream;
}

void to_json(nlohmann::json& json, const SoftmaxDescriptor& descriptor)
{
    json = nlohmann::json{
        {"alpha", descriptor.GetAlpha()},
        {"beta", descriptor.GetBeta()},
        {"algorithm", descriptor.GetAlgorithm()},
        {"mode", descriptor.GetMode()},
    };
}

void from_json(const nlohmann::json& json, SoftmaxDescriptor& descriptor)
{
    json.at("alpha").get_to(descriptor.alpha);
    json.at("beta").get_to(descriptor.beta);
    json.at("algorithm").get_to(descriptor.algorithm);
    json.at("mode").get_to(descriptor.mode);
}

miopenStatus_t SoftmaxForward(const Handle& handle,
                              const void* alpha,
                              const void* beta,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              miopenSoftmaxAlgorithm_t algorithm,
                              miopenSoftmaxMode_t mode,
                              int x_offset,
                              int y_offset)
{
    if(x == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto problem = softmax::ProblemDescription{alpha, beta, xDesc, yDesc, algorithm, mode};
    const auto invoke_params =
        softmax::InvokeParams{alpha, beta, xDesc, x, yDesc, y, algorithm, mode, x_offset, y_offset};
    const auto algo = AlgorithmName{"Softmax"};
    const auto solvers =
        solver::SolverContainer<solver::softmax::AttnSoftmax, solver::softmax::Softmax>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t SoftmaxBackward(const Handle& handle,
                               const void* alpha,
                               const TensorDescriptor& yDesc,
                               ConstData_t y,
                               const TensorDescriptor& dyDesc,
                               ConstData_t dy,
                               const void* beta,
                               const TensorDescriptor& dxDesc,
                               Data_t dx,
                               miopenSoftmaxAlgorithm_t algorithm,
                               miopenSoftmaxMode_t mode,
                               int y_offset,
                               int dy_offset,
                               int dx_offset)
{
    if(dx == nullptr || y == nullptr || dy == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm, "Null pointer for tensor.");
    }

    const auto problem =
        softmax::ProblemDescription{alpha, beta, yDesc, dyDesc, dxDesc, algorithm, mode};
    const auto invoke_params = softmax::InvokeParams{alpha,
                                                     beta,
                                                     yDesc,
                                                     y,
                                                     dyDesc,
                                                     dy,
                                                     dxDesc,
                                                     dx,
                                                     algorithm,
                                                     mode,
                                                     y_offset,
                                                     dy_offset,
                                                     dx_offset};
    const auto algo          = AlgorithmName{"Softmax"};
    const auto solvers       = solver::SolverContainer<solver::softmax::Softmax>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
