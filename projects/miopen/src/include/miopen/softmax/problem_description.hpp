// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/config.h>
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/mlo_internal.hpp>

namespace miopen {

struct NetworkConfig;

namespace softmax {

struct ProblemDescriptionTag
{
};

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase,
                                                    ProblemDescriptionTag
#if MIOPEN_ENABLE_SQLITE
    ,
                                                    SQLiteSerializable<ProblemDescription>
#endif
{
    // softmax forward constructor
    ProblemDescription(const void* alpha_,
                       const void* beta_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       miopenSoftmaxAlgorithm_t algorithm_,
                       miopenSoftmaxMode_t mode_)
        : isForward(true),
          xdxDesc(xDesc_),
          yDesc(yDesc_),

          algorithm(algorithm_),
          mode(mode_)
    {
        CheckAndAssignAlphaBeta(alpha_, beta_);

        if(xdxDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(xdxDesc.GetLengths() != yDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }
    }

    ProblemDescription(const void* alpha_,
                       const void* beta_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& dyDesc_,
                       const TensorDescriptor& dxDesc_,
                       miopenSoftmaxAlgorithm_t algorithm_,
                       miopenSoftmaxMode_t mode_)
        : isForward(false),
          xdxDesc(dxDesc_),
          yDesc(yDesc_),
          dyDesc(dyDesc_),
          algorithm(algorithm_),
          mode(mode_)
    {
        CheckAndAssignAlphaBeta(alpha_, beta_);

        if(yDesc != dyDesc)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        if(xdxDesc.GetType() != dyDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(xdxDesc.GetLengths() != dyDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }
    }

    bool IsForward() const { return isForward; }
    miopenSoftmaxAlgorithm_t GetAlgorithm() const { return algorithm; }
    miopenSoftmaxMode_t GetMode() const { return mode; }
    float GetAlpha() const { return alpha; }
    float GetBeta() const { return beta; }

    // for forward
    const TensorDescriptor& GetXDesc() const { return xdxDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    // for backward
    const TensorDescriptor& GetdYDesc() const { return dyDesc; }
    const TensorDescriptor& GetdXDesc() const { return xdxDesc; }

    void Serialize(std::ostream& stream) const { stream << MakeNetworkConfig().ToString(); }

    NetworkConfig MakeNetworkConfig() const override;

    template <class Self>
    static void Visit(Self&& self, std::function<void(int64_t, std::string)> f)
    {
        // The column names match the driver command line argument names
        f(static_cast<uint64_t>(self.isForward), "forw");
        f(self.GetBatchSize(), "batchsize");
        f(self.GetChannels(), "in_channels");
        f(self.GetHeight(), "in_h");
        f(self.GetWidth(), "in_w");
        f(static_cast<uint64_t>(self.algorithm), "algorithm");
        f(static_cast<uint64_t>(self.mode), "mode");
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(std::string, std::string)> f)
    {
        f(GetDataTypeName(self.yDesc.GetType()), "data_type");
        f(self.GetLayout(), "layout");
    }

    template <class Self, class Visitor>
    static void VisitAll(Self&& self, const Visitor& f)
    {
        Visit(std::forward<Self>(self), [&](int64_t value, std::string name) { f(value, name); });
        Visit(std::forward<Self>(self),
              [&](std::string value, std::string name) { f(value, name); });
    }

    // This declaration marks softmax as a primitive with tuning enabled.
    // Any tunable solver would be able pick it and fetch a db instance in ExecutePrimitive.
    // It has to be discoverable via ADL from problem description.
    friend auto GetDb(const ExecutionContext& context,
                      const ProblemDescriptionTag&) -> PerformanceDb;

private:
    void CheckAndAssignAlphaBeta(const void* alpha_, const void* beta_)
    {
        if(alpha_ == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Alpha value is nullptr");
        }

        if(beta_ == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Beta value is nullptr");
        }

        alpha = *(static_cast<const float*>(alpha_));
        beta  = *(static_cast<const float*>(beta_));
    }

    const bool isForward;

    float alpha;
    float beta;

    // for forward xDesc is stored in xdxDesc, for backward dxDesc is stored in xdxDesc
    TensorDescriptor xdxDesc;
    TensorDescriptor yDesc;
    TensorDescriptor dyDesc;

    const miopenSoftmaxAlgorithm_t algorithm;
    const miopenSoftmaxMode_t mode;

    std::size_t GetBatchSize() const { return yDesc.GetLengths()[0]; }
    std::size_t GetChannels() const { return yDesc.GetLengths()[1]; }
    std::size_t GetHeight() const { return yDesc.GetLengths()[2]; }
    std::size_t GetWidth() const { return yDesc.GetLengths()[3]; }
    std::string GetLayout() const { return yDesc.GetLayout_str(); }
};

} // namespace softmax
} // namespace miopen
