// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/pooling.hpp>
#include <miopen/mlo_internal.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace pooling {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescriptionTag
{
};

struct ProblemDescription : ProblemDescriptionBase,
                            ProblemDescriptionTag
#if MIOPEN_ENABLE_SQLITE
    ,
                            SQLiteSerializable<ProblemDescription>
#endif
{
    // Forward
    ProblemDescription(const PoolingDescriptor& pooling_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       bool save_index_)
        : direction(Direction::Forward),
          pooling(pooling_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          save_index(save_index_)
    {
    }

    // Backward
    ProblemDescription(const PoolingDescriptor& pooling_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& dxDesc_,
                       const TensorDescriptor& dyDesc_)
        : direction(Direction::Backward),
          pooling(pooling_),
          xDesc(xDesc_),
          yDesc(yDesc_),
          dxDesc(dxDesc_),
          dyDesc(dyDesc_)
    {
    }

    Direction GetDirection() const { return direction; }
    const PoolingDescriptor& GetPooling() const { return pooling; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    const TensorDescriptor& GetDXDesc() const
    {
        assert(direction == Direction::Backward);
        return dxDesc;
    }

    const TensorDescriptor& GetDYDesc() const
    {
        assert(direction == Direction::Backward);
        return dyDesc;
    }

    bool SaveIndex() const
    {
        assert(direction == Direction::Forward);
        return save_index;
    }

    NetworkConfig MakeNetworkConfig() const override;

    template <class Self>
    static void Visit(Self&& self, std::function<void(int64_t, std::string)> f)
    {
        // The column names match the driver command line argument names
        f(static_cast<uint64_t>(self.direction), "direction");
        f(static_cast<uint64_t>(self.pooling.GetPaddingMode()), "pad_mode");
        f(self.pooling.GetStrides().size() > 2 ? self.pooling.GetStrides()[0] : 0, "pool_stride_d");
        f(self.pooling.GetStrides().size() > 2 ? self.pooling.GetStrides()[1]
                                               : self.pooling.GetStrides()[0],
          "pool_stride_h");
        f(self.pooling.GetStrides().size() > 2 ? self.pooling.GetStrides()[2]
                                               : self.pooling.GetStrides()[1],
          "pool_stride_w");
        f(self.pooling.GetPads().size() > 2 ? self.pooling.GetPads()[0] : 0, "pad_d");
        f(self.pooling.GetPads().size() > 2 ? self.pooling.GetPads()[1] : self.pooling.GetPads()[0],
          "pad_h");
        f(self.pooling.GetPads().size() > 2 ? self.pooling.GetPads()[2] : self.pooling.GetPads()[1],
          "pad_w");
        f(self.pooling.GetLengths().size() > 2 ? self.pooling.GetLengths()[0] : 0, "win_d");
        f(self.pooling.GetLengths().size() > 2 ? self.pooling.GetLengths()[1]
                                               : self.pooling.GetLengths()[0],
          "win_h");
        f(self.pooling.GetLengths().size() > 2 ? self.pooling.GetLengths()[2]
                                               : self.pooling.GetLengths()[1],
          "win_w");
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(std::string, std::string)> f)
    {
        f(self.GetInShape(), "in_shape");
        f(self.GetOutShape(), "out_shape");
        f(self.xDesc.GetLayout_str(), "layout");
        f(self.GetDirectionStr(), "direction");
        f(GetDataTypeName(self.xDesc.GetType()), "data_type");
        f(self.GetModeStr(), "mode");
    }

    template <class Self, class Visitor>
    static void VisitAll(Self&& self, const Visitor& f)
    {
        Visit(std::forward<Self>(self), [&](int64_t value, std::string name) { f(value, name); });
        Visit(std::forward<Self>(self),
              [&](std::string value, std::string name) { f(value, name); });
    }

    void Serialize(std::ostream& stream) const { stream << MakeNetworkConfig().ToString(); }

    // This declaration marks pooling as a primitive with tuning enabled.
    // Any tunable solver would be able pick it and fetch a db instance in ExecutePrimitive.
    // It has to be discoverable via ADL from problem description.
    friend auto GetDb(const ExecutionContext& context,
                      const ProblemDescriptionTag&) -> PerformanceDb;

private:
    Direction direction;
    PoolingDescriptor pooling;
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    TensorDescriptor dxDesc;
    TensorDescriptor dyDesc;
    bool save_index = false;
    std::size_t GetInBatchSize() const { return xDesc.GetLengths()[0]; }
    std::size_t GetInChannel() const { return xDesc.GetLengths()[1]; }
    std::size_t GetInDepth() const { return xDesc.GetNumDims() > 4 ? xDesc.GetLengths()[2] : 0; }
    std::size_t GetInHeight() const { return xDesc.GetLengths()[xDesc.GetNumDims() - 2]; }
    std::size_t GetInWidth() const { return xDesc.GetLengths()[xDesc.GetNumDims() - 1]; }
    std::size_t GetOutBatchSize() const { return yDesc.GetLengths()[0]; }
    std::size_t GetOutChannel() const { return yDesc.GetLengths()[1]; }
    std::size_t GetOutDepth() const { return yDesc.GetNumDims() > 4 ? yDesc.GetLengths()[2] : 0; }
    std::size_t GetOutHeight() const { return yDesc.GetLengths()[yDesc.GetNumDims() - 2]; }
    std::size_t GetOutWidth() const { return yDesc.GetLengths()[yDesc.GetNumDims() - 1]; }
    std::string GetInShape() const
    {
        std::string shape = std::to_string(GetInBatchSize());
        shape += "x" + std::to_string(GetInChannel());
        if(GetInDepth() != 0)
            shape += "x" + std::to_string(GetInDepth());
        shape += "x" + std::to_string(GetInHeight());
        shape += "x" + std::to_string(GetInWidth());
        return shape;
    }
    std::string GetOutShape() const
    {
        std::string shape = std::to_string(GetOutBatchSize());
        shape += "x" + std::to_string(GetOutChannel());
        if(GetOutDepth() != 0)
            shape += "x" + std::to_string(GetOutDepth());
        shape += "x" + std::to_string(GetOutHeight());
        shape += "x" + std::to_string(GetOutWidth());
        return shape;
    }
    std::string GetDirectionStr() const
    {
        switch(direction)
        {
        case Direction::Forward: return "Fwd";
        case Direction::Backward: return "Bwd";
        }

        MIOPEN_THROW(miopenStatusInvalidValue, "Wrong pooling direction provided");
    }
    std::string GetModeStr() const
    {
        switch(pooling.GetMode())
        {
        case miopenPoolingMax: return "max";
        case miopenPoolingAverage: return "avg";
        case miopenPoolingAverageInclusive: return "avg_in";
        }

        MIOPEN_THROW(miopenStatusInvalidValue, "Wrong pooling mode provided");
    }
};

} // namespace pooling

} // namespace miopen
