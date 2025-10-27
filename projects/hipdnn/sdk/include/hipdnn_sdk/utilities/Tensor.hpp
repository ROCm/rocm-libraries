// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <typeindex>
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

inline const TensorLayout TensorLayout::NCHW{"NCHW", {3, 2, 1, 0}};
inline const TensorLayout TensorLayout::NHWC{"NHWC", strideOrderNhwc(4)};
inline const TensorLayout TensorLayout::NCDHW{"NCDHW", {4, 3, 2, 1, 0}};
inline const TensorLayout TensorLayout::NDHWC{"NDHWC", strideOrderNhwc(5)};

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

class ITensor;

template <bool IsConst = false>
class ITensorIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::conditional_t<IsConst, const void*, void*>;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const void*, void*>;
    using reference = std::conditional_t<IsConst, const void*, void*>;

    using TensorType = std::conditional_t<IsConst, const ITensor&, ITensor&>;

    ITensorIterator() = default;

    template <bool C = IsConst, std::enable_if_t<!C, int> = 0>
    ITensorIterator(ITensor& tensor, bool isEnd = false)
        : _tensor(tensor)
        , _indices(_tensor.dims().size(), 0)
    {
        if(isEnd && !_tensor.dims().empty())
        {
            _indices[0] = _tensor.dims()[0];
        }
    }

    template <bool C = IsConst, std::enable_if_t<C, int> = 0>
    ITensorIterator(const ITensor& tensor, bool isEnd = false)
        : _tensor(tensor)
        , _indices(_tensor.dims().size(), 0)
    {
        if(isEnd && !_tensor.dims().empty())
        {
            _indices[0] = _tensor.dims()[0];
        }
    }

    ITensorIterator(const ITensorIterator& other)
        : _tensor(other._tensor)
        , _indices(other._indices)
    {
    }

    ITensorIterator(ITensorIterator&&) = default;

    ITensorIterator& operator=(const ITensorIterator& other)
    {
        if(this != &other)
        {
            _tensor = other._tensor;
            _indices = other._indices;
        }
        return *this;
    }

    ITensorIterator& operator=(ITensorIterator&&) = default;

    value_type operator*()
    {
        throwIfOutOfBounds("Cannot dereference end iterator");

        return _tensor.hostDataOffsetFromIndex(_tensor.getIndex(_indices));
    }

    ITensorIterator& operator++()
    {
        throwIfOutOfBounds("Iterator cannot be incremented past the end");

        const auto& dims = _tensor.dims();
        for(int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim)
        {
            auto dimIdx = static_cast<size_t>(dim);
            _indices[dimIdx]++;
            if(_indices[dimIdx] < dims[dimIdx])
            {
                return *this;
            }
            _indices[dimIdx] = 0;
        }

        //set 1 past end.
        _indices[0] = dims[0];

        return *this;
    }

    ITensorIterator operator++(int)
    {
        ITensorIterator temp = *this;
        ++(*this);
        return temp;
    }

    bool operator==(const ITensorIterator& other) const
    {
        return (&_tensor == &other._tensor) && (_indices == other._indices);
    }

    bool operator!=(const ITensorIterator& other) const
    {
        return !(*this == other);
    }

    std::vector<int64_t> indices() const
    {
        return _indices;
    }

private:
    void throwIfOutOfBounds(const std::string& reason) const
    {
        const auto& dims = _tensor.dims();
        if(dims.empty() || _indices[0] == dims[0])
        {
            throw std::out_of_range(reason);
        }
    }

    TensorType _tensor;
    std::vector<int64_t> _indices;
};

class ITensor
{
public:
    virtual ~ITensor() = default;

    virtual const std::vector<int64_t>& dims() const = 0;
    virtual const std::vector<int64_t>& strides() const = 0;

    virtual void* rawHostData() = 0;
    virtual void* rawDeviceData() = 0;

    virtual size_t elementCount() const = 0;
    virtual size_t elementSpace() const = 0;
    virtual size_t elementSize() const = 0;
    virtual void* hostDataOffsetFromIndex(int64_t index) = 0;
    virtual const void* hostDataOffsetFromIndex(int64_t index) const = 0;

    virtual void fillTensorWithValue(float value) = 0;
    virtual void
        fillTensorWithRandomValues(float min, float max, unsigned int seed = std::random_device{}())
        = 0;
    virtual size_t fillWithData(const void* data, size_t bytesCopied) = 0;

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
        static_assert(std::is_integral_v<IndexType>, "Index type must be integral!");

        if(indices.size() > strides().size())
        {
            throw std::invalid_argument("Number of indices (" + std::to_string(indices.size())
                                        + ") must not be greater than the number of strides ("
                                        + std::to_string(strides().size()) + ")");
        }

        return throwIfOutOfBounds(
            std::inner_product(indices.begin(), indices.end(), strides().begin(), int64_t{0}));
    }

    virtual ITensorIterator<false> begin() = 0;
    virtual ITensorIterator<false> end() = 0;
    virtual ITensorIterator<true> cbegin() const = 0;
    virtual ITensorIterator<true> cend() const = 0;

    virtual bool isPacked() const = 0;

    virtual void markHostModified() = 0;
    virtual void markDeviceModified() = 0;

protected:
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    int64_t throwIfOutOfBounds(int64_t index) const
    {
#ifndef NDEBUG
        if(static_cast<size_t>(index) >= elementSpace())
        {
            throw std::out_of_range("Index " + std::to_string(index)
                                    + " is out of range for tensor with "
                                    + std::to_string(elementSpace()) + " elements");
        }
#endif
        return index;
    }
};

template <typename T>
class TensorBase : public ITensor
{
public:
    void* rawHostData() override
    {
        return memory().hostData();
    }

    void* rawDeviceData() override
    {
        return memory().deviceData();
    }

    void* hostDataOffsetFromIndex(int64_t index) override
    {
        return memory().hostData() + index;
    }

    const void* hostDataOffsetFromIndex(int64_t index) const override
    {
        return memory().hostData() + index;
    }

    void fillTensorWithValue(float value) override
    {
        fillWithValue(static_cast<T>(value));
    }

    void fillTensorWithRandomValues(float min,
                                    float max,
                                    unsigned int seed = std::random_device{}()) override
    {
        fillWithRandomValues(static_cast<T>(min), static_cast<T>(max), seed);
    }

    virtual MigratableMemoryBase<T>& memory() = 0;
    virtual const MigratableMemoryBase<T>& memory() const = 0;

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

    ITensorIterator<false> begin() override
    {
        return ITensorIterator<false>(*this, false);
    }

    ITensorIterator<false> end() override
    {
        return ITensorIterator<false>(*this, true);
    }

    ITensorIterator<true> cbegin() const override
    {
        return ITensorIterator<true>(*this, false);
    }

    ITensorIterator<true> cend() const override
    {
        return ITensorIterator<true>(*this, true);
    }

    void markHostModified() override
    {
        memory().markHostModified();
    }

    void markDeviceModified() override
    {
        memory().markDeviceModified();
    }

    size_t elementSize() const override
    {
        return sizeof(T);
    }

protected:
    bool computeIsPacked(const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& strides) const
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
};

// NOLINTEND(portability-template-virtual-member-function)
template <class T, class HostAlloc = HostAllocator<T>, class DeviceAlloc = DeviceAllocator<T>>
class Tensor : public TensorBase<T>
{
public:
    Tensor(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
        : _dims(dims)
        , _strides(strides)
        , _elementCount(TensorBase<T>::calculateItemCount(dims))
    {
        validateDimsAndStridesSameSize();
        validateAllPositive(_dims, "dimension");
        validateAllPositive(_strides, "stride");

        // Set packed flag after validations since it can be incorrect if dims/strides are invalid.
        _packed = TensorBase<T>::computeIsPacked(dims, strides);

        _memory = MigratableMemory<T, HostAlloc, DeviceAlloc>(
            TensorBase<T>::calculateElementSpace(dims, strides));
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

    size_t elementCount() const override
    {
        return _elementCount;
    }

    size_t elementSpace() const override
    {
        return _memory.count();
    }

    const MigratableMemoryBase<T>& memory() const override
    {
        return _memory;
    }

    MigratableMemoryBase<T>& memory() override
    {
        return _memory;
    }

    size_t fillWithData(const void* data, size_t maxBytesCopied) override
    {
        size_t bytesCopied = std::min(maxBytesCopied, _memory.count() * sizeof(T));
        _memory.markHostModified();
        std::memcpy(_memory.hostData(), data, bytesCopied);
        return bytesCopied;
    }

    void fillWithValue(T value) override
    {
        _memory.markHostModified();
        iterateAlongDimensions(_dims, [&](const std::vector<int64_t>& indices) {
            this->setHostValue(value, indices);
        });
    }

    void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) override
    {

        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

        _memory.markHostModified();
        iterateAlongDimensions(_dims, [&](const std::vector<int64_t>& indices) {
            this->setHostValue(static_cast<T>(distribution(generator)), indices);
        });
    }

    bool isPacked() const override
    {
        return _packed;
    }

private:
    void validateDimsAndStridesSameSize() const
    {
        if(_dims.size() != _strides.size())
        {
            throw std::invalid_argument("Number of dimensions (" + std::to_string(_dims.size())
                                        + ") must match number of strides ("
                                        + std::to_string(_strides.size()) + ")");
        }
    }

    void validateAllPositive(const std::vector<int64_t>& values, const std::string& valueName) const
    {
        for(size_t i = 0; i < values.size(); ++i)
        {
            if(values[i] <= 0)
            {
                std::ostringstream oss;
                oss << "All " << valueName << "s must be positive. " << valueName << " " << i
                    << " is " << values[i];
                throw std::invalid_argument(oss.str());
            }
        }
    }

    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    size_t _elementCount;
    bool _packed;
};

template <typename T>
using PinnedTensor = Tensor<T, PinnedHostAllocator<T>>;

} // namespace utilities
} // namespace hipdnn_sdk
