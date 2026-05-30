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
#pragma once

#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>
#include <miopen/mlo_internal.hpp>
#include <cassert>

namespace miopen {

struct NetworkConfig;

namespace layernorm {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescriptionTag
{
};

size_t GetStride(const TensorDescriptor& xDesc, int32_t normalized_dim);

size_t GetOuterSize(const TensorDescriptor& xDesc, int32_t normalized_dim, size_t stride);

size_t GetInnerSize(const TensorDescriptor& xDesc, int32_t normalized_dim);

struct MIOPEN_INTERNALS_EXPORT ProblemDescription : ProblemDescriptionBase,
                                                    ProblemDescriptionTag
#if MIOPEN_ENABLE_SQLITE
    ,
                                                    SQLiteSerializable<ProblemDescription>
#endif
{
    ProblemDescription(Direction direction_,
                       miopenNormMode_t mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& x2Desc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& rstdDesc_,
                       const TensorDescriptor& dyDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dwDesc_,
                       const TensorDescriptor& dbDesc_,
                       float epsilon_,
                       int32_t normalized_dim_)
        : stride(GetStride(xDesc_, normalized_dim_)),
          outer_size(GetOuterSize(xDesc_, normalized_dim_, stride)),
          inner_size(GetInnerSize(xDesc_, normalized_dim_)),
          direction(direction_),
          mode(mode_),
          xDesc(xDesc_),
          x2Desc(x2Desc_),
          weightDesc(weightDesc_),
          biasDesc(biasDesc_),
          yDesc(yDesc_),
          meanDesc(meanDesc_),
          rstdDesc(rstdDesc_),
          dyDesc(dyDesc_),
          dxDesc(dxDesc_),
          dwDesc(dwDesc_),
          dbDesc(dbDesc_),
          epsilon(epsilon_),
          normalized_dim(normalized_dim_)
    {
    }

    ProblemDescription(miopenNormMode_t mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& rstdDesc_,
                       float epsilon_,
                       int32_t normalized_dim_)
        : ProblemDescription(Direction::Forward,
                             mode_,
                             xDesc_,
                             {},
                             weightDesc_,
                             biasDesc_,
                             yDesc_,
                             meanDesc_,
                             rstdDesc_,
                             {},
                             {},
                             {},
                             {},
                             epsilon_,
                             normalized_dim_)
    {
    }

    ProblemDescription(miopenNormMode_t mode_,
                       const TensorDescriptor& dyDesc_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& rstdDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dwDesc_,
                       const TensorDescriptor& dbDesc_,
                       int32_t normalized_dim_)
        : ProblemDescription(Direction::Backward,
                             mode_,
                             xDesc_,
                             {},
                             weightDesc_,
                             {},
                             {},
                             meanDesc_,
                             rstdDesc_,
                             dyDesc_,
                             dxDesc_,
                             dwDesc_,
                             dbDesc_,
                             {},
                             normalized_dim_)
    {
    }

    ProblemDescription(miopenNormMode_t mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& x2Desc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& rstdDesc_,
                       float epsilon_,
                       int32_t normalized_dim_)
        : ProblemDescription(Direction::Forward,
                             mode_,
                             xDesc_,
                             x2Desc_,
                             weightDesc_,
                             biasDesc_,
                             yDesc_,
                             meanDesc_,
                             rstdDesc_,
                             {},
                             {},
                             {},
                             {},
                             epsilon_,
                             normalized_dim_)
    {
    }

    ProblemDescription(miopenNormMode_t mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& rstdDesc_,
                       float epsilon_)
        : ProblemDescription(Direction::Forward,
                             mode_,
                             xDesc_,
                             {},
                             weightDesc_,
                             {},
                             yDesc_,
                             {},
                             rstdDesc_,
                             {},
                             {},
                             {},
                             {},
                             epsilon_,
                             int(xDesc_.GetLengths().size()) - 1)
    {
    }

    ProblemDescription(miopenNormMode_t mode_,
                       const TensorDescriptor& dyDesc_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& rstdDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dwDesc_)
        : ProblemDescription(Direction::Backward,
                             mode_,
                             xDesc_,
                             {},
                             weightDesc_,
                             {},
                             {},
                             {},
                             rstdDesc_,
                             dyDesc_,
                             dxDesc_,
                             dwDesc_,
                             {},
                             {},
                             int(xDesc_.GetLengths().size()) - 1)
    {
    }

    Direction GetDirection() const { return direction; }
    miopenNormMode_t GetMode() const { return mode; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetX2Desc() const { return x2Desc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetBiasDesc() const { return biasDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetMeanDesc() const { return meanDesc; }
    const TensorDescriptor& GetRstdDesc() const { return rstdDesc; }
    const TensorDescriptor& GetDYDesc() const { return dyDesc; }
    const TensorDescriptor& GetDXDesc() const { return dxDesc; }
    const TensorDescriptor& GetDWDesc() const { return dwDesc; }
    const TensorDescriptor& GetDBDesc() const { return dbDesc; }
    float GetEpsilon() const { return epsilon; }
    int32_t GetNormalizedDim() const { return normalized_dim; }

    bool IsSameType() const
    {
        if(direction == Direction::Forward)
        {
            if(xDesc.GetType() != yDesc.GetType())
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "LayerNormForward: Tensor types do not match.");
#else
                return false;
#endif
            }
        }
        else
        {
            if(dyDesc.GetType() != dxDesc.GetType())
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "LayerNormBackward: Tensor types do not match.");
#else
                return false;
#endif
            }
        }
        return true;
    }

    bool IsSameLength() const
    {
        if(direction == Direction::Forward)
        {
            if(xDesc.GetLengths() != yDesc.GetLengths())
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm,
                             "LayerNormForward: Tensor dimension lengths do not match.");
#else
                return false;
#endif
            }
            return true;
        }
        else
        {
            if(dyDesc.GetLengths() != dxDesc.GetLengths())
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm,
                             "LayerNormBackward: Tensor dimension lengths do not match.");
#else
                return false;
#endif
            }
            return true;
        }
    }

    bool IsRightNormDim() const
    {
        if((normalized_dim < 0) || (normalized_dim > xDesc.GetLengths().size()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(
                miopenStatusBadParm,
                "LayerNormForward: normalized dim is greater than 0 and less than or equal "
                "Tensor dimension length.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(direction == Direction::Forward)
        {
            if(!(xDesc.IsPacked() && weightDesc.IsPacked() && biasDesc.IsPacked() &&
                 yDesc.IsPacked() && meanDesc.IsPacked() && rstdDesc.IsPacked()))
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm,
                             "LayerNormForward: Unpacked tensors not supported.");
#else
                return false;
#endif
            }
        }
        else
        {
            if(!(dyDesc.IsPacked() && xDesc.IsPacked() && weightDesc.IsPacked() &&
                 rstdDesc.IsPacked() && dxDesc.IsPacked() && dwDesc.IsPacked() &&
                 dbDesc.IsPacked()))
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm,
                             "LayerNormBackward: Unpacked tensors not supported.");
#else
                return false;
#endif
            }
        }
        return true;
    }

    bool IsLargeSize() const { return outer_size * stride > 32; }

    void Serialize(std::ostream& stream) const { stream << MakeNetworkConfig().ToString(); }

    NetworkConfig MakeNetworkConfig() const override;

    template <class Self>
    static void Visit(Self&& self, std::function<void(uint64_t, std::string)> f)
    {
        // The column names match the driver command line argument names
        f(static_cast<uint64_t>(self.direction), "direction");
        f(self.GetBatchSize(), "in_n");
        f(self.GetChannel(), "in_c");
        f(self.GetDepth(), "in_d");
        f(self.GetHeight(), "in_h");
        f(self.GetWidth(), "in_w");
        f(static_cast<uint64_t>(self.normalized_dim), "normalized_dim");
        f(static_cast<uint64_t>(self.mode), "mode");

        f(self.stride, "stride");
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(std::string, std::string)> f)
    {
        f(GetDataTypeName(self.xDesc.GetType()), "data_type");
    }

    template <class Self, class Visitor>
    static void VisitAll(Self&& self, const Visitor& f)
    {
        Visit(std::forward<Self>(self), [&](uint64_t value, std::string name) { f(value, name); });
        Visit(std::forward<Self>(self),
              [&](std::string value, std::string name) { f(value, name); });
    }

    // This declaration marks layernorm as a primitive with tuning enabled.
    // Any tunable solver would be able pick it and fetch a db instance in ExecutePrimitive.
    // It has to be discoverable via ADL from problem description.
    friend auto GetDb(const ExecutionContext& context,
                      const ProblemDescriptionTag&) -> PerformanceDb;

public:
    const size_t stride;
    const size_t outer_size;
    const size_t inner_size;

private:
    Direction direction;
    miopenNormMode_t mode;
    TensorDescriptor xDesc;
    TensorDescriptor x2Desc;
    TensorDescriptor weightDesc;
    TensorDescriptor biasDesc;
    TensorDescriptor yDesc;
    TensorDescriptor meanDesc;
    TensorDescriptor rstdDesc;
    TensorDescriptor dyDesc;
    TensorDescriptor dxDesc;
    TensorDescriptor dwDesc;
    TensorDescriptor dbDesc;

    float epsilon;
    int32_t normalized_dim;

    NetworkConfig MakeForwardNetworkConfig() const;

    std::size_t GetBatchSize() const { return xDesc.GetLengths()[0]; }
    std::size_t GetChannel() const { return xDesc.GetNumDims() > 2 ? xDesc.GetLengths()[1] : 0; }
    std::size_t GetDepth() const { return xDesc.GetNumDims() > 4 ? xDesc.GetLengths()[2] : 0; }
    std::size_t GetHeight() const
    {
        return xDesc.GetNumDims() > 2 ? xDesc.GetLengths()[xDesc.GetNumDims() - 2] : 0;
    }
    std::size_t GetWidth() const { return xDesc.GetLengths()[xDesc.GetNumDims() - 1]; }
};

} // namespace layernorm

} // namespace miopen
