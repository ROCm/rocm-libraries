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

// Forward declaration of TensorBase
template <typename T>
class TensorBase;
class ITensor;

// Forward iterator for typed tensor iteration
template <typename T, bool IsConst>
class TensorIterator
{
public:
    // Iterator traits for STL compatibility
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const T*, T*>;
    using reference = std::conditional_t<IsConst, const T&, T&>;
    using tensor_type = std::conditional_t<IsConst, const TensorBase<T>, TensorBase<T>>;

    // Constructors
    TensorIterator(tensor_type* tensor, std::vector<int64_t> indices, bool isEnd = false)
        : _tensor(tensor)
        , _indices(std::move(indices))
        , _isEnd(isEnd)
    {
    }

    // Dereference operators
    reference operator*() const
    {
        if(_isEnd)
        {
            throw std::out_of_range("Cannot dereference end iterator");
        }
        int64_t index = _tensor->getIndex(_indices);
        return _tensor->memory().hostData()[index];
    }

    pointer operator->() const
    {
        if(_isEnd)
        {
            throw std::out_of_range("Cannot dereference end iterator");
        }
        int64_t index = _tensor->getIndex(_indices);
        return &_tensor->memory().hostData()[index];
    }

    // Prefix increment
    TensorIterator& operator++()
    {
        if(_isEnd)
        {
            return *this;
        }

        const auto& dims = _tensor->dims();

        // Increment indices in reverse order (rightmost dimension first)
        for(int dim = static_cast<int>(dims.size()) - 1; dim >= 0; --dim)
        {
            auto dimIdx = static_cast<size_t>(dim);
            _indices[dimIdx]++;
            if(_indices[dimIdx] < dims[dimIdx])
            {
                return *this; // Successfully incremented
            }
            _indices[dimIdx] = 0; // Carry to next dimension
        }

        // If we get here, we've incremented past the last element
        _isEnd = true;
        return *this;
    }

    // Postfix increment
    TensorIterator operator++(int)
    {
        TensorIterator temp = *this;
        ++(*this);
        return temp;
    }

    // Comparison operators
    bool operator==(const TensorIterator& other) const
    {
        if(_tensor != other._tensor)
        {
            return false;
        }
        if(_isEnd && other._isEnd)
        {
            return true;
        }
        if(_isEnd != other._isEnd)
        {
            return false;
        }
        return _indices == other._indices;
    }

    bool operator!=(const TensorIterator& other) const
    {
        return !(*this == other);
    }

    // Get current indices (useful for debugging)
    const std::vector<int64_t>& indices() const
    {
        return _indices;
    }

private:
    tensor_type* _tensor;
    std::vector<int64_t> _indices;
    bool _isEnd;
};

// Type-erased iterator for ITensor polymorphic iteration
class TypeErasedIterator
{
public:
    // Iterator traits for STL compatibility
    using iterator_category = std::forward_iterator_tag;
    using value_type = void;
    using difference_type = std::ptrdiff_t;
    using pointer = void*;
    using reference = void*;

    // Default constructor
    TypeErasedIterator() = default;

    // Copy constructor
    TypeErasedIterator(const TypeErasedIterator& other)
        : _impl(other._impl ? other._impl->clone() : nullptr)
    {
    }

    // Move constructor
    TypeErasedIterator(TypeErasedIterator&&) = default;

    // Copy assignment
    TypeErasedIterator& operator=(const TypeErasedIterator& other)
    {
        if(this != &other)
        {
            _impl = other._impl ? other._impl->clone() : nullptr;
        }
        return *this;
    }

    // Move assignment
    TypeErasedIterator& operator=(TypeErasedIterator&&) = default;

    // Dereference - returns void*
    void* operator*() const
    {
        if(!_impl)
        {
            throw std::runtime_error("Cannot dereference invalid iterator");
        }
        return _impl->get();
    }

    // Prefix increment
    TypeErasedIterator& operator++()
    {
        if(_impl)
        {
            _impl->increment();
        }
        return *this;
    }

    // Postfix increment
    TypeErasedIterator operator++(int)
    {
        TypeErasedIterator temp = *this;
        ++(*this);
        return temp;
    }

    // Comparison operators
    bool operator==(const TypeErasedIterator& other) const
    {
        if(!_impl && !other._impl)
        {
            return true;
        }
        if(!_impl || !other._impl)
        {
            return false;
        }
        return _impl->equals(other._impl.get());
    }

    bool operator!=(const TypeErasedIterator& other) const
    {
        return !(*this == other);
    }

    // Get current indices (useful for debugging)
    std::vector<int64_t> indices() const
    {
        if(!_impl)
        {
            return {};
        }
        return _impl->getIndices();
    }

    // Factory methods to create from typed iterators
    template <typename T>
    static TypeErasedIterator create(TensorIterator<T, false> iter)
    {
        TypeErasedIterator result;
        result._impl = std::make_unique<IteratorModel<T>>(std::move(iter));
        return result;
    }

    template <typename T>
    static TypeErasedIterator createConst(TensorIterator<T, true> iter)
    {
        TypeErasedIterator result;
        result._impl = std::make_unique<ConstIteratorModel<T>>(std::move(iter));
        return result;
    }

private:
    // Internal interface for type-erased operations
    struct IteratorConcept
    {
        virtual ~IteratorConcept() = default;
        virtual void increment() = 0;
        virtual void* get() = 0;
        virtual bool equals(const IteratorConcept* other) const = 0;
        virtual std::unique_ptr<IteratorConcept> clone() const = 0;
        virtual std::vector<int64_t> getIndices() const = 0;
    };

    // Concrete implementation for non-const typed iterators
    template <typename T>
    struct IteratorModel : IteratorConcept
    {
        TensorIterator<T, false> iter;

        explicit IteratorModel(TensorIterator<T, false> iter)
            : iter(std::move(iter))
        {
        }

        void increment() override
        {
            ++iter;
        }

        void* get() override
        {
            return const_cast<void*>(static_cast<const void*>(&(*iter)));
        }

        bool equals(const IteratorConcept* other) const override
        {
            auto* otherModel = dynamic_cast<const IteratorModel<T>*>(other);
            return otherModel && iter == otherModel->iter;
        }

        std::unique_ptr<IteratorConcept> clone() const override
        {
            return std::make_unique<IteratorModel<T>>(iter);
        }

        std::vector<int64_t> getIndices() const override
        {
            return iter.indices();
        }
    };

    // Concrete implementation for const typed iterators
    template <typename T>
    struct ConstIteratorModel : IteratorConcept
    {
        TensorIterator<T, true> iter;

        explicit ConstIteratorModel(TensorIterator<T, true> iter)
            : iter(std::move(iter))
        {
        }

        void increment() override
        {
            ++iter;
        }

        void* get() override
        {
            return const_cast<void*>(static_cast<const void*>(&(*iter)));
        }

        bool equals(const IteratorConcept* other) const override
        {
            auto* otherModel = dynamic_cast<const ConstIteratorModel<T>*>(other);
            return otherModel && iter == otherModel->iter;
        }

        std::unique_ptr<IteratorConcept> clone() const override
        {
            return std::make_unique<ConstIteratorModel<T>>(iter);
        }

        std::vector<int64_t> getIndices() const override
        {
            return iter.indices();
        }
    };

    std::unique_ptr<IteratorConcept> _impl;
};

class ITensor
{
public:
    virtual ~ITensor() = default;

    virtual const std::vector<int64_t>& dims() const = 0;
    virtual const std::vector<int64_t>& strides() const = 0;

    virtual void* rawHostData() = 0;

    virtual size_t elementCount() const = 0;
    virtual size_t elementSpace() const = 0;
    virtual const void* hostDataOffsetFromIndex(int64_t index) const = 0;

    virtual void fillTensorWithValue(float value) = 0;
    virtual void
        fillTensorWithRandomValues(float min, float max, unsigned int seed = std::random_device{}())
        = 0;

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

    virtual TypeErasedIterator begin() = 0;
    virtual TypeErasedIterator end() = 0;
    virtual TypeErasedIterator begin() const = 0;
    virtual TypeErasedIterator end() const = 0;

    virtual bool isPacked() const = 0;

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
    using iterator = TensorIterator<T, false>;
    using const_iterator = TensorIterator<T, true>;

    void* rawHostData() override
    {
        return memory().hostData();
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

    virtual IMigratableMemory<T>& memory() = 0;
    virtual const IMigratableMemory<T>& memory() const = 0;

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

    TypeErasedIterator begin() override
    {
        std::vector<int64_t> startIndices(dims().size(), 0);
        return TypeErasedIterator::create(iterator(this, startIndices, false));
    }

    TypeErasedIterator end() override
    {
        std::vector<int64_t> endIndices(dims().size(), 0);
        return TypeErasedIterator::create(iterator(this, endIndices, true));
    }

    TypeErasedIterator begin() const override
    {
        std::vector<int64_t> startIndices(dims().size(), 0);
        return TypeErasedIterator::createConst(const_iterator(this, startIndices, false));
    }

    TypeErasedIterator end() const override
    {
        std::vector<int64_t> endIndices(dims().size(), 0);
        return TypeErasedIterator::createConst(const_iterator(this, endIndices, true));
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
        , _packed(TensorBase<T>::computeIsPacked(dims, strides))
    {
        validateDimsAndStridesSameSize();
        validateAllPositive(_dims, "dimension");
        validateAllPositive(_strides, "stride");

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
        iterateAlongDimensions(_dims, [&](const std::vector<int64_t>& indices) {
            this->setHostValue(value, indices);
        });
    }

    void fillWithRandomValues(T min, T max, unsigned int seed = std::random_device{}()) override
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(static_cast<float>(min),
                                                           static_cast<float>(max));

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
