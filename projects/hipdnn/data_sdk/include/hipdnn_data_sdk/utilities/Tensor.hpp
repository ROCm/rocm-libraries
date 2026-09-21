// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cassert>
#include <functional>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <iostream>
#include <numeric>
#include <optional>
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

/**
 * @brief Snapshot of the state a ragged tensor iterator needs to traverse its buffer.
 *
 * `rowOffsets` is the B+1 offset table (element units), `seqAxis` is the logical axis
 * that varies within a batch's sequence, and `seqStride` is that axis's stride. All
 * three are fixed at construction so traversal performs no per-step aux reads.
 */
struct RaggedIterationInfo
{
    std::vector<int64_t> rowOffsets;
    int seqAxis;
    int64_t seqStride;
};

template <bool IsConst = false>
class ITensorIterator
{
public:
    // forward declarations
    struct LinearIndex;
    struct CompositeIndex;
    struct RaggedCompositeIndex;

    using iterator_category = std::forward_iterator_tag;
    using value_type = std::conditional_t<IsConst, const void*, void*>;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<IsConst, const void*, void*>;
    using reference = std::conditional_t<IsConst, const void*, void*>;

    using TensorType = std::conditional_t<IsConst,
                                          std::reference_wrapper<const ITensor>,
                                          std::reference_wrapper<ITensor>>;
    using IndexType = std::variant<LinearIndex, CompositeIndex, RaggedCompositeIndex>;

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

    /**
     * @brief Iterator index for ragged tensors.
     *
     * Walks each batch's full per-batch range `[ragged_offset[b], ragged_offset[b+1])`
     * in turn, visiting exactly `ragged_offset[B]` physical elements. The traversal
     * state (`rowOffsets`, `seqAxis`, `seqStride`) is snapshotted once via
     * `RaggedIterationInfo` at `begin()`/`end()`, so traversal performs no per-step
     * aux reads.
     */
    struct RaggedCompositeIndex
    {
        RaggedCompositeIndex(TensorType tensor, RaggedIterationInfo info, bool isEnd)
            : indices(tensor.get().dims().size(), 0)
            , rowOffsets(std::move(info.rowOffsets))
            , tensor(tensor)
            , seqAxis(info.seqAxis)
            , seqStride(info.seqStride)
        {
            const int64_t batchCount = numBatches();
            if(isEnd)
            {
                if(!indices.empty())
                {
                    indices[0] = batchCount;
                }
            }
            else
            {
                // Skip leading empty batches so begin() lands on a real element.
                while(indices[0] < batchCount && seqExtent(indices[0]) == 0)
                {
                    ++indices[0];
                }
            }
        }

        RaggedCompositeIndex(const RaggedCompositeIndex& other) = default;

        RaggedCompositeIndex(RaggedCompositeIndex&&) = default;

        RaggedCompositeIndex& operator=(const RaggedCompositeIndex& other) = default;

        RaggedCompositeIndex& operator=(RaggedCompositeIndex&& other) = default;

        RaggedCompositeIndex& operator++()
        {
            const auto& dims = tensor.get().dims();

            // Rightmost-first carry over the non-batch axes. The sequence axis is
            // bounded by the current batch's per-batch extent; every other non-batch
            // axis ranges fully over its dims().
            for(int dim = static_cast<int>(dims.size()) - 1; dim >= 1; --dim)
            {
                const auto dimIdx = static_cast<size_t>(dim);
                ++indices[dimIdx];
                const int64_t bound = (dim == seqAxis) ? seqExtent(indices[0]) : dims[dimIdx];
                if(indices[dimIdx] < bound)
                {
                    return *this;
                }
                indices[dimIdx] = 0;
            }

            // Carry into the batch axis, skipping empty batches.
            const int64_t batchCount = numBatches();
            do
            {
                ++indices[0];
            } while(indices[0] < batchCount && seqExtent(indices[0]) == 0);
            return *this;
        }

        RaggedCompositeIndex operator++(int)
        {
            auto temp{*this};
            ++(*this);
            return temp;
        }

        bool operator==(const RaggedCompositeIndex& other) const
        {
            return indices == other.indices && &tensor.get() == &other.tensor.get();
        }

        bool operator!=(const RaggedCompositeIndex& other) const
        {
            return !((*this) == other);
        }

        bool isOutOfBounds() const
        {
            return indices.empty() || indices[0] == numBatches();
        }

        int64_t getValue() const
        {
            return tensor.get().getIndex(indices);
        }

        std::vector<int64_t> indices;
        std::vector<int64_t> rowOffsets;
        TensorType tensor;
        int seqAxis{1};
        int64_t seqStride{1};

    private:
        int64_t numBatches() const
        {
            return static_cast<int64_t>(rowOffsets.size()) - 1;
        }

        // Per-batch sequence extent: number of sequence rows in batch b.
        int64_t seqExtent(int64_t b) const
        {
            if(b < 0 || (b + 1) >= static_cast<int64_t>(rowOffsets.size()))
            {
                return 0;
            }
            const auto bIdx = static_cast<size_t>(b);
            return (rowOffsets[bIdx + 1] - rowOffsets[bIdx]) / seqStride;
        }
    };

private:
    void throwIfOutOfBounds(const std::string& reason) const
    {
        if(std::visit([](const auto& idx) { return idx.isOutOfBounds(); }, _index))
        {
            throw std::out_of_range(reason);
        }
    }

    /// True iff walking memory linearly visits the elements in index order, i.e. the
    /// strides are the packed row-major strides for these dims.
    ///
    /// This is a question about ORDER. isPacked() is a question about EXTENT, and the
    /// two are independent: any permutation of the strides spans the same memory, so it
    /// is packed while visiting out of index order. RFC 0014 §7.2 proposes the same
    /// split from the ragged-tensor side.
    ///
    /// An axis of extent 1 is exempt, because its index is always 0 and its stride
    /// therefore never reaches an offset. NHWC activations with 1x1 spatial extent rely
    /// on this: they carry a spatial stride of C rather than 1.
    ///
    /// Defined out-of-line below: ITensor is only forward-declared at this point.
    static bool visitsInIndexOrder(const ITensor& tensor);

    IndexType makeIndex(TensorType tensor, bool isEnd)
    {
        // Both predicates are required. isPacked() keeps ragged tensors off this path
        // (RFC 0014 §4.5.7), and visitsInIndexOrder keeps stride permutations off it.
        if(tensor.get().isPacked() && visitsInIndexOrder(tensor.get()))
        {
            return LinearIndex(tensor, isEnd);
        }
        // Ragged tensors expose traversal info; dense strided tensors return nullopt
        // and fall through to the regular CompositeIndex.
        if(auto info = tensor.get().raggedIterationInfo())
        {
            return RaggedCompositeIndex(tensor, std::move(*info), isEnd);
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

        return throwIfOutOfBounds(getIndexImpl(indices));
    }

    /**
     * @brief Returns the traversal info the iterator needs for a ragged tensor.
     *
     * Dense tensors return `std::nullopt` (the iterator then uses Linear/Composite
     * indexing as today). Ragged tensors override this to expose their offset table,
     * sequence axis, and sequence stride, which the iterator snapshots once to build a
     * RaggedCompositeIndex.
     */
    virtual std::optional<RaggedIterationInfo> raggedIterationInfo() const
    {
        return std::nullopt;
    }

    virtual ITensorIterator<false> begin() = 0;
    virtual ITensorIterator<false> end() = 0;
    virtual ITensorIterator<true> cbegin() const = 0;
    virtual ITensorIterator<true> cend() const = 0;

    /// True iff the elements occupy the memory span with no gaps, i.e.
    /// elementCount == elementSpace. This says nothing about the ORDER in which a
    /// linear walk of that span visits them; a permutation of the strides is packed
    /// too, and iterating one linearly visits the right addresses in the wrong order.
    virtual bool isPacked() const = 0;

    virtual void markHostModified() = 0;
    virtual void markDeviceModified() = 0;

protected:
    /**
     * @brief Computes the physical offset for a multi-dim index.
     *
     * Default (dense) implementation is the inner product of indices and strides.
     * Ragged tensors override this to base each batch at `ragged_offset[b]`, which
     * makes every addressing path (getHostValue/setHostValue/operator(),
     * CompositeIndex::getValue, TensorView) ragged-aware at once. The argument-count
     * check stays in the non-virtual getIndex forwarder.
     */
    virtual int64_t getIndexImpl(const std::vector<int64_t>& indices) const
    {
        return std::inner_product(indices.begin(), indices.end(), strides().begin(), int64_t{0});
    }

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

/// @brief A callable that fills underlying tensor memory with user defined values.
///
/// Signature: void(T* data, size_t count)
/// The generator is called once per tensor fill and must fill `count` elements of
/// the `data` pointer. If `fillWithValues` sets `hostFill` to true, then the data
/// pointer will be a pointer to a host memory allocation. Otherwise, if `hostFill`
/// was set to false, then this is a pointer to a device allocation. `count` is
/// the element space of the tensor, i.e. it will contain padding elements if the
/// tensor is not packed.
///
/// It is not the responsibility of the generator to mark the tensor as host or
/// device modified.
///
/// For device fills the operation must be complete before the generator returns,
/// or use the same HIP stream as the migratable memory object backing the tensor.
template <typename T>
using ValueGenerator = std::function<void(T* data, size_t count)>;

template <bool IsConst>
bool ITensorIterator<IsConst>::visitsInIndexOrder(const ITensor& tensor)
{
    const auto& dims = tensor.dims();
    const auto& strides = tensor.strides();
    if(dims.size() != strides.size())
    {
        return false;
    }

    int64_t expected = 1;
    for(size_t axis = dims.size(); axis-- > 0;)
    {
        // Extent 1 pins the index to 0, so this stride never reaches an offset.
        if(dims[axis] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= dims[axis];
    }
    return true;
}

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

    // BOOLEAN has no out-of-band value; its sentinel is true and cannot detect an unwritten mask.
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

    void fillWithValues(const ValueGenerator<T>& generator, bool hostFill)
    {
        if(!generator)
        {
            throw std::invalid_argument(
                "generator must not be nullptr when calling TensorBase::fillWithValues");
        }

        MigratableMemoryBase<T>& migratableMem = memory();
        if(hostFill)
        {
            // Call will overwrite the whole allocation, mark host modified before getting the pointer
            // to avoid synchronizing device-to-host copy when calling `hostData()`
            migratableMem.markHostModified();
            generator(migratableMem.hostData(), migratableMem.count());
        }
        else
        {
            // Call will overwrite the whole allocation, mark device modified before getting
            // the pointer to avoid synchronizing host-to-device copy when calling `deviceData()`
            migratableMem.markDeviceModified();
            generator(static_cast<T*>(migratableMem.deviceData()), migratableMem.count());
        }
    }

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
