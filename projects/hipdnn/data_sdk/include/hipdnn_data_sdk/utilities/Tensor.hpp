// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <variant>
#include <vector>

namespace hipdnn_data_sdk::utilities
{

/**
 * @brief Describes a tensor memory layout via stride ordering
 *
 * TensorLayout encodes how tensor dimensions map to memory. The `strideOrder` vector
 * specifies the priority of each dimension in memory layout (lower values = tighter
 * packing in memory).
 *
 * @note TensorLayout is primarily used with convolution and batch normalization tensors,
 * which follow (N, C, H, W) / (N, C, D, H, W) dimension ordering. Other operations
 * such as matmul and pointwise have their own dimension conventions. The TensorLayout
 * controls how dimensions map to contiguous memory via strides computed by
 * `generateStrides()`.
 *
 * For example, for a convolution input with dims = {1, 64, 28, 28} (N=1, C=64, H=28, W=28):
 * - TensorLayout::NCHW (stride order {3,2,1,0}) produces strides {50176, 784, 28, 1} (channel-first; N=50176, C=784, H=28, W=1)
 * - TensorLayout::NHWC (stride order {3,0,2,1}) produces strides {50176, 1, 1792, 64} (channel-last; N=50176, C=1, H=1792, W=64)
 */
struct TensorLayout
{
    std::string name; ///< Human-readable layout name (e.g., "NCHW", "NHWC")
    std::vector<int64_t> strideOrder; ///< Stride priority per dimension (lower = tighter packing)

    static const TensorLayout NCL; ///< 3D channel-first layout
    static const TensorLayout NLC; ///< 3D channel-last layout
    static const TensorLayout NCHW; ///< 4D channel-first layout
    static const TensorLayout NHWC; ///< 4D channel-last layout
    static const TensorLayout NCDHW; ///< 5D channel-first layout
    static const TensorLayout NDHWC; ///< 5D channel-last layout

    /// SDPA row-major layout for dims [batch, heads, seq_len, head_dim].
    /// Same stride order as NCHW ({3,2,1,0}): head_dim is most contiguous.
    static const TensorLayout BHSD;

    /// SDPA sequence-major layout for dims [batch, seq_len, heads, head_dim].
    /// Stride order {3,1,2,0}: head_dim contiguous, then heads, then seq_len, then batch.
    /// @note This is NOT the same stride order as NHWC. NHWC ({3,0,2,1}) would make
    /// heads contiguous, which is not the intended BSHD layout.
    static const TensorLayout BSHD;
};

// NOLINTBEGIN(bugprone-throwing-static-initialization) fixed-size layout constants
inline const TensorLayout TensorLayout::NCL{"NCL", {2, 1, 0}};
inline const TensorLayout TensorLayout::NLC{"NLC", strideOrderNhwc(3)};
inline const TensorLayout TensorLayout::NCHW{"NCHW", {3, 2, 1, 0}};
inline const TensorLayout TensorLayout::NHWC{"NHWC", strideOrderNhwc(4)};
inline const TensorLayout TensorLayout::NCDHW{"NCDHW", {4, 3, 2, 1, 0}};
inline const TensorLayout TensorLayout::NDHWC{"NDHWC", strideOrderNhwc(5)};
inline const TensorLayout TensorLayout::BHSD{"BHSD", {3, 2, 1, 0}};
inline const TensorLayout TensorLayout::BSHD{"BSHD", {3, 1, 2, 0}};
// NOLINTEND(bugprone-throwing-static-initialization)

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

// Forward declarations
class ITensor;

template <bool IsConst = false>
class ITensorIterator
{
public:
    // forward declarations
    struct LinearIndex;
    struct CompositeIndex;

    using iterator_category = std::forward_iterator_tag;
    using value_type = std::conditional_t<IsConst, const void*, void*>;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const void*, void*>;
    using reference = std::conditional_t<IsConst, const void*, void*>;

    using TensorType = std::conditional_t<IsConst,
                                          std::reference_wrapper<const ITensor>,
                                          std::reference_wrapper<ITensor>>;
    using IndexType = std::variant<LinearIndex, CompositeIndex>;

    ITensorIterator() = default;

    template <bool C = IsConst, std::enable_if_t<!C, int> = 0>
    ITensorIterator(ITensor& tensor, bool isEnd = false)
        : _tensor(tensor)
        , _index(makeIndex(_tensor, isEnd))
    {
    }

    template <bool C = IsConst, std::enable_if_t<C, int> = 0>
    ITensorIterator(const ITensor& tensor, bool isEnd = false)
        : _tensor(tensor)
        , _index(makeIndex(_tensor, isEnd))
    {
    }

    ITensorIterator(const ITensorIterator& other) = default;

    ITensorIterator(ITensorIterator&&) = default;

    ITensorIterator& operator=(const ITensorIterator& other) = default;

    ITensorIterator& operator=(ITensorIterator&&) = default;

    value_type operator*()
    {
        throwIfOutOfBounds("Cannot dereference end iterator");
        return _tensor.get().hostDataOffsetFromIndex(
            std::visit([](auto& idx) { return idx.getValue(); }, _index));
    }

    ITensorIterator& operator++()
    {
        throwIfOutOfBounds("Iterator cannot be incremented past the end");
        std::visit([](auto& idx) { ++idx; }, _index);
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
        return (&_tensor.get() == &other._tensor.get()) && (_index == other._index);
    }

    bool operator!=(const ITensorIterator& other) const
    {
        return !(*this == other);
    }

    IndexType index() const
    {
        return _index;
    }

    struct LinearIndex
    {
        LinearIndex(TensorType tensor, bool isEnd)
            : tensor(tensor)

        {
            if(isEnd && !tensor.get().dims().empty())
            {
                index = static_cast<decltype(index)>(tensor.get().elementCount());
            }
        }

        LinearIndex(const LinearIndex& other) = default;

        LinearIndex(LinearIndex&&) = default;

        LinearIndex& operator=(const LinearIndex& other) = default;

        LinearIndex& operator=(LinearIndex&& other) = default;

        LinearIndex& operator++()
        {
            ++index;
            return *this;
        }

        LinearIndex operator++(int)
        {
            auto temp{*this};
            ++(*this);
            return temp;
        }

        bool operator==(const LinearIndex& other) const
        {
            return index == other.index && &tensor.get() == &other.tensor.get();
        }

        bool operator!=(const LinearIndex& other) const
        {
            return !((*this) == other);
        }

        bool isOutOfBounds() const
        {
            return index == static_cast<decltype(index)>(tensor.get().elementCount());
        }

        int64_t getValue() const
        {
            return index;
        }

        int64_t index{0};
        TensorType tensor;
    };

    struct CompositeIndex
    {
        CompositeIndex(TensorType tensor, bool isEnd)
            : indices(tensor.get().dims().size(), 0)
            , tensor(tensor)
        {
            if(isEnd && !tensor.get().dims().empty())
            {
                indices[0] = tensor.get().dims()[0];
            }
        }

        CompositeIndex(const CompositeIndex& other) = default;

        CompositeIndex(CompositeIndex&&) = default;

        CompositeIndex& operator=(const CompositeIndex& other) = default;

        CompositeIndex& operator=(CompositeIndex&& other) = default;

        CompositeIndex& operator++()
        {
            const auto& dims = tensor.get().dims();
            for(int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim)
            {
                auto dimIdx = static_cast<size_t>(dim);
                indices[dimIdx]++;
                if(indices[dimIdx] < dims[dimIdx])
                {
                    return *this;
                }
                indices[dimIdx] = 0;
            }

            //set 1 past end.
            indices[0] = dims[0];
            return *this;
        }

        CompositeIndex operator++(int)
        {
            auto temp{*this};
            ++(*this);
            return temp;
        }

        bool operator==(const CompositeIndex& other) const
        {
            return indices == other.indices && &tensor.get() == &other.tensor.get();
        }

        bool operator!=(const CompositeIndex& other) const
        {
            return !((*this) == other);
        }

        bool isOutOfBounds() const
        {
            const auto& dims = tensor.get().dims();
            return dims.empty() || indices[0] == dims[0];
        }

        int64_t getValue() const
        {
            return tensor.get().getIndex(indices);
        }

        std::vector<int64_t> indices;
        TensorType tensor;
    };

private:
    void throwIfOutOfBounds(const std::string& reason) const
    {
        if(std::visit([](const auto& idx) { return idx.isOutOfBounds(); }, _index))
        {
            throw std::out_of_range(reason);
        }
    }

    IndexType makeIndex(TensorType tensor, bool isEnd)
    {
        if(tensor.get().isPacked())
        {
            return LinearIndex(tensor, isEnd);
        }
        return CompositeIndex(tensor, isEnd);
    }

    TensorType _tensor;
    IndexType _index;
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
    virtual void fillWithSentinelValue() = 0;
    virtual size_t fillWithData(const void* data, size_t bytesCopied) = 0;

    template <typename... Args>
    int64_t getIndex(Args... indices) const
    {
        static_assert(AllOfTypes<std::is_integral, Args...>::value,
                      "Indices must be an integral type!");

        const std::vector<int64_t> indexVector = {static_cast<int64_t>(indices)...};

        return getIndex(indexVector);
    }

    int64_t getIndex(const std::vector<int64_t>& indices) const
    {
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

    void fillWithSentinelValue() override
    {
        if constexpr(std::numeric_limits<T>::has_quiet_NaN)
        {
            fillWithValue(std::numeric_limits<T>::quiet_NaN());
        }
        else
        {
            fillWithValue(std::numeric_limits<T>::max());
        }
    }

    virtual MigratableMemoryBase<T>& memory() = 0;
    virtual const MigratableMemoryBase<T>& memory() const = 0;

    template <typename... Args>
    T getHostValue(Args... indices) const
    {
        return (*this)(indices...);
    }

    T getHostValue(const std::vector<int64_t>& indices) const
    {
        return (*this)(indices);
    }

    template <typename... Args>
    void setHostValue(T value, Args... indices)
    {
        (*this)(indices...) = value;
    }

    void setHostValue(T value, const std::vector<int64_t>& indices)
    {
        (*this)(indices) = value;
    }

    template <typename... Args>
    T& operator()(Args... indices)
    {
        const int64_t index = getIndex(indices...);
        auto* data = memory().hostData();
        return data[index];
    }

    template <typename... Args>
    const T& operator()(Args... indices) const
    {
        const int64_t index = getIndex(indices...);
        const auto* data = memory().hostData();
        return data[index];
    }

    T& operator()(const std::vector<int64_t>& indices)
    {
        const int64_t index = getIndex(indices);
        auto* data = memory().hostData();
        return data[index];
    }

    const T& operator()(const std::vector<int64_t>& indices) const
    {
        const int64_t index = getIndex(indices);
        const auto* data = memory().hostData();
        return data[index];
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
                               size_t{1},
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
            std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>()));
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

        _memory = utilities::MigratableMemory<T, HostAlloc, DeviceAlloc>(
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
        const size_t bytesCopied = std::min(maxBytesCopied, _memory.count() * sizeof(T));
        _memory.markHostModified();
        std::memcpy(_memory.hostData(), data, bytesCopied);
        return bytesCopied;
    }

    void fillWithValue(T value) override
    {
        _memory.markHostModified();
        for(auto valuePtr : (*this))
        {
            *static_cast<T*>(valuePtr) = value;
        }
    }

    void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) override
    {

        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

        _memory.markHostModified();
        for(auto valuePtr : (*this))
        {
            *static_cast<T*>(valuePtr) = static_cast<T>(distribution(generator));
        }
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

    utilities::MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    size_t _elementCount;
    bool _packed;
};

template <typename T>
using PinnedTensor = Tensor<T, PinnedHostAllocator<T>>;

template <typename T>
inline std::unique_ptr<ITensor> createTensor(const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    return std::make_unique<Tensor<T>>(dims, strides);
}

} // namespace hipdnn_data_sdk::utilities
