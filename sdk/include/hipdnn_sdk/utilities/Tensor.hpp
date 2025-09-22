// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace hipdnn_sdk
{
namespace utilities
{

struct TensorLayout
{
    std::string name;
    std::vector<int64_t> strideOrder;

    static const TensorLayout NCHW;
    static const TensorLayout NHWC;
    static const TensorLayout NCDHW;
    static const TensorLayout NDHWC;
};

inline const TensorLayout TensorLayout::NCHW{.name = "NCHW", .strideOrder = {3, 2, 1, 0}};
inline const TensorLayout TensorLayout::NHWC{.name = "NHWC", .strideOrder = strideOrderNhwc(4)};
inline const TensorLayout TensorLayout::NCDHW{.name = "NCDHW", .strideOrder = {4, 3, 2, 1, 0}};
inline const TensorLayout TensorLayout::NDHWC{.name = "NDHWC", .strideOrder = strideOrderNhwc(5)};

inline std::ostream& operator<<(std::ostream& os, const TensorLayout& layout)
{
    return os << layout.name;
}

// NOLINTBEGIN(portability-template-virtual-member-function)

// Helper to check if all types in a parameter pack satisfy a predicate
template <template <typename> class Predicate, typename... Ts>
struct AllOfTypes : std::conjunction<Predicate<Ts>...>
{
};

template <typename T>
class TensorBase
{
public:
    virtual ~TensorBase() = default;

    virtual const std::vector<int64_t>& dims() const = 0;
    virtual const std::vector<int64_t>& strides() const = 0;

    virtual IMigratableMemory<T>& memory() = 0;
    virtual const IMigratableMemory<T>& memory() const = 0;

    template <typename... Args>
    int64_t getIndex(Args... indices) const
    {
        static_assert(AllOfTypes<std::is_integral, Args...>::value,
                      "Indices must be an integral type!");

        std::vector<int64_t> indexVector = {static_cast<int64_t>(indices)...};

        return getIndex(indexVector);
    }

    template <typename IndexType>
    int64_t getIndex(const std::vector<IndexType>& indices) const
    {
        static_assert(std::is_integral<IndexType>::value, "Index type must be integral!");

        if(indices.size() > strides().size())
        {
            throw std::invalid_argument("Number of indices (" + std::to_string(indices.size())
                                        + ") must not be greater than the number of strides ("
                                        + std::to_string(strides().size()) + ")");
        }

        return std::inner_product(indices.begin(), indices.end(), strides().begin(), int64_t{0});
    }

    template <typename... Args>
    T getHostValue(Args... indices) const
    {
        int64_t index = getIndex(indices...);
        const auto* data = memory().hostData();
        return data[index];
    }

    template <typename IndexType>
    T getHostValue(const std::vector<IndexType>& indices) const
    {
        int64_t index = getIndex(indices);
        const auto* data = memory().hostData();
        return data[index];
    }

    template <typename... Args>
    void setHostValue(T value, Args... indices)
    {
        int64_t index = getIndex(indices...);
        auto* data = memory().hostData();
        data[index] = value;
    }

    template <typename IndexType>
    void setHostValue(T value, const std::vector<IndexType>& indices)
    {
        int64_t index = getIndex(indices);
        auto* data = memory().hostData();
        data[index] = value;
    }

    virtual void fillWithValue(T value) = 0;
    virtual void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) = 0;
};

// NOLINTEND(portability-template-virtual-member-function)

template <class T, class HostAlloc = HostAllocator<T>, class DeviceAlloc = DeviceAllocator<T>>
class Tensor : public TensorBase<T>
{
public:
    Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _memory(calculateItemCount(dims))
        , _dims(dims)
        , _strides(strides)
    {
        if(!isPacked(dims, strides))
        {
            throw std::invalid_argument("Tensor must be packed");
        }
    }

    Tensor(const std::vector<int64_t>& dims, const TensorLayout& layout)
        : Tensor(dims, generateStrides(dims, layout.strideOrder))
    {
    }

    Tensor(const std::vector<int64_t>& dims)
        : Tensor(dims, generateStrides(dims))
    {
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    const std::vector<int64_t>& dims() const override
    {
        return _dims;
    }

    const std::vector<int64_t>& strides() const override
    {
        return _strides;
    }

    const IMigratableMemory<T>& memory() const override
    {
        return _memory;
    }

    IMigratableMemory<T>& memory() override
    {
        return _memory;
    }

    void fillWithValue(T value) override
    {
        T* data = _memory.hostData();
        std::fill(data, data + _memory.count(), value);
    }

    void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) override
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

        auto* data = _memory.hostData();
        for(size_t i = 0; i < _memory.count(); ++i)
        {
            data[i] = static_cast<T>(distribution(generator));
        }
    }

private:
    bool isPacked(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides) const
    {
        // Item count = largest stride * item count in that dimension
        return (calculateItemCount(dims) == calculateElementSpace(dims, strides));
    }

    static size_t calculateElementSpace(const std::vector<int64_t>& dims,
                                        const std::vector<int64_t>& strides)
    {
        return static_cast<size_t>(
            std::inner_product(dims.begin(),
                               dims.end(),
                               strides.begin(),
                               1,
                               std::plus<>(),
                               [](size_t len, size_t stride) { return (len - 1) * stride; }));
    }

    static size_t calculateItemCount(const std::vector<int64_t>& dims)
    {
        if(dims.empty())
        {
            return 0;
        }

        return static_cast<size_t>(
            std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));
    }

    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
};

template <typename T>
using PinnedTensor = Tensor<T, PinnedHostAllocator<T>>;

} // namespace utilities
} // namespace hipdnn_sdk
