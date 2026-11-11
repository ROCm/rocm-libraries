// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_data_sdk::utilities
{

// NOLINTBEGIN(portability-template-virtual-member-function)

/// Sequence-axis index for tensors whose dims are physically ordered [B, S, H, D].
inline constexpr int BSHD_SEQ_AXIS = 1;

/// Sequence-axis index for SDPA tensors, which keep logical dims [B, H, S, D] (cuDNN
/// convention); ragged SDPA tensors carry the BSHD physical layout in their strides.
inline constexpr int SDPA_SEQ_AXIS = 2;

namespace detail
{

/// Per-batch origin (element units) and length (sequence rows) of a ragged tensor; both hold
/// B entries.
struct RaggedBatches
{
    std::vector<int64_t> origins, lengths;
};

/// Logical element count and addressed buffer span (in elements) of a ragged tensor.
struct RaggedFootprint
{
    size_t elementCount;
    size_t elementSpace;
};

/**
 * @brief Derives the per-batch origins and lengths from a ragged offset table.
 *
 * @param elementOffsets The B+1 offset table in element units (multiplier applied).
 * @param seqLens B per-batch lengths, or nullptr. With lengths, `origin[b] = elementOffsets[b]`,
 *        `len[b] = seqLens[b]` and `elementOffsets[B]` is unused. Without, the offsets must be
 *        monotonic with whole-token blocks and
 *        `len[b] = min((off[b+1] - off[b]) / seqStride, sMax)`.
 * @param seqStride Stride of the sequence axis; a token row is `seqStride` elements.
 * @param sMax Extent of the sequence axis.
 * @throws std::invalid_argument When an origin is negative or not a whole number of token rows,
 *         a given length is outside `[0, sMax]`, or (without lengths) the offsets are
 *         non-monotonic or a block is not a whole number of token rows.
 */
inline RaggedBatches raggedBatches(const std::vector<int64_t>& elementOffsets,
                                   const std::vector<int64_t>* seqLens,
                                   int64_t seqStride,
                                   int64_t sMax)
{
    if(seqStride <= 0)
    {
        throw std::invalid_argument("sequence-axis stride must be positive");
    }
    if(elementOffsets.empty())
    {
        throw std::invalid_argument("ragged offset table must hold B + 1 entries");
    }

    const size_t batchCount = elementOffsets.size() - 1;
    if(seqLens != nullptr && seqLens->size() != batchCount)
    {
        throw std::invalid_argument("seq_len must hold B entries (got "
                                    + std::to_string(seqLens->size()) + ", B is "
                                    + std::to_string(batchCount) + ")");
    }

    RaggedBatches batches;
    batches.origins.reserve(batchCount);
    batches.lengths.reserve(batchCount);
    for(size_t b = 0; b < batchCount; ++b)
    {
        const int64_t origin = elementOffsets[b];
        if(origin < 0 || origin % seqStride != 0)
        {
            throw std::invalid_argument(
                "ragged whole-token origin violated: origin " + std::to_string(origin)
                + " of batch " + std::to_string(b)
                + " must be >= 0 and a multiple of the sequence-axis stride "
                + std::to_string(seqStride));
        }

        int64_t length = 0;
        if(seqLens != nullptr)
        {
            length = (*seqLens)[b];
            if(length < 0 || length > sMax)
            {
                throw std::invalid_argument(
                    "ragged length range violated: length " + std::to_string(length) + " of batch "
                    + std::to_string(b) + " must be in [0, S_max=" + std::to_string(sMax) + "]");
            }
        }
        else
        {
            const int64_t block = elementOffsets[b + 1] - origin;
            if(block < 0)
            {
                throw std::invalid_argument("ragged_offset must be monotonic non-decreasing (batch "
                                            + std::to_string(b) + ")");
            }
            if(block % seqStride != 0)
            {
                throw std::invalid_argument(
                    "ragged whole-token block violated: per-batch block must be a whole number of "
                    "sequence rows (batch "
                    + std::to_string(b) + ")");
            }
            length = std::min(block / seqStride, sMax);
        }

        batches.origins.push_back(origin);
        batches.lengths.push_back(length);
    }
    return batches;
}

/**
 * @brief Validates the ragged layout rules and returns its element count and addressed span.
 *
 * The rules: the non-empty token intervals `[origin/seqStride, +len)` of the batches are pairwise
 * disjoint; the dense view `[maxRow+1, other non-batch dims]` satisfies the sorted-stride rule
 * (axes of extent > 1 with a non-zero stride, sorted by stride, each stride at least the span of
 * the axes below it; the batch axis is excluded and zero-stride broadcast axes are exempt); the
 * non-batch strides are non-negative.
 *
 * `elementCount = sum(len[b]) * prod(non-batch, non-seq dims)`. `elementSpace` is 0 for an empty
 * tensor, else `1 + maxRow * seqStride + sum((d_i - 1) * stride_i)` over the non-batch, non-seq
 * axes.
 *
 * @throws std::invalid_argument When `seqStride <= 0`, a rule above is violated, or the arguments
 *         are structurally inconsistent.
 */
inline RaggedFootprint raggedFootprint(const std::vector<int64_t>& dims,
                                       const std::vector<int64_t>& strides,
                                       int seqAxis,
                                       const RaggedBatches& batches)
{
    if(dims.size() < 2 || strides.size() != dims.size() || seqAxis < 1
       || seqAxis >= static_cast<int>(dims.size()))
    {
        throw std::invalid_argument("ragged layout needs rank >= 2 dims and strides of equal size "
                                    "and a sequence axis in [1, rank)");
    }
    if(batches.origins.size() != batches.lengths.size()
       || static_cast<int64_t>(batches.lengths.size()) != dims[0])
    {
        throw std::invalid_argument("ragged origins and lengths must hold dims[0] (B) entries");
    }

    const auto seqIdx = static_cast<size_t>(seqAxis);
    const int64_t seqStride = strides[seqIdx];
    if(seqStride <= 0)
    {
        throw std::invalid_argument("sequence-axis stride must be positive");
    }
    for(size_t axis = 1; axis < strides.size(); ++axis)
    {
        if(strides[axis] < 0)
        {
            throw std::invalid_argument("ragged tensor strides must be non-negative");
        }
    }

    // Overlap rule: sort the non-empty token intervals by first row.
    struct Interval
    {
        int64_t first;
        int64_t last; // one past the last row
        size_t batch;
    };
    std::vector<Interval> intervals;
    int64_t totalRows = 0;
    int64_t maxRow = -1;
    for(size_t b = 0; b < batches.lengths.size(); ++b)
    {
        const int64_t length = batches.lengths[b];
        if(length <= 0)
        {
            continue;
        }
        const int64_t first = batches.origins[b] / seqStride;
        intervals.push_back({first, first + length, b});
        totalRows += length;
        maxRow = std::max(maxRow, first + length - 1);
    }
    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        return a.first < b.first;
    });
    for(size_t i = 1; i < intervals.size(); ++i)
    {
        if(intervals[i].first < intervals[i - 1].last)
        {
            throw std::invalid_argument("ragged token intervals overlap: batch "
                                        + std::to_string(intervals[i - 1].batch) + " rows ["
                                        + std::to_string(intervals[i - 1].first) + ", "
                                        + std::to_string(intervals[i - 1].last) + ") and batch "
                                        + std::to_string(intervals[i].batch) + " rows ["
                                        + std::to_string(intervals[i].first) + ", "
                                        + std::to_string(intervals[i].last) + ")");
        }
    }

    int64_t rowCount = 1;
    for(size_t axis = 1; axis < dims.size(); ++axis)
    {
        if(axis != seqIdx)
        {
            rowCount *= dims[axis];
        }
    }
    const int64_t elementCount = totalRows * rowCount;
    // An empty tensor addresses nothing; the span formula would wrap at maxRow == -1, and a
    // zero-extent axis leaves no layout to check.
    if(elementCount == 0)
    {
        return {0, 0};
    }

    // Sorted-stride rule on the dense view; broadcast and size-1 axes are exempt.
    std::vector<std::pair<int64_t, int64_t>> axes; // (stride, extent)
    for(size_t axis = 1; axis < dims.size(); ++axis)
    {
        const int64_t extent = (axis == seqIdx) ? maxRow + 1 : dims[axis];
        if(extent > 1 && strides[axis] != 0)
        {
            axes.emplace_back(strides[axis], extent);
        }
    }
    std::sort(axes.begin(), axes.end());
    int64_t span = 1;
    for(const auto& [stride, extent] : axes)
    {
        if(stride < span)
        {
            throw std::invalid_argument(
                "ragged tensor violates the sorted-stride rule: an axis of stride "
                + std::to_string(stride) + " lies inside the " + std::to_string(span)
                + " elements spanned by the axes of smaller stride");
        }
        span += (extent - 1) * stride;
    }

    int64_t space = 1 + maxRow * seqStride;
    for(size_t axis = 1; axis < dims.size(); ++axis)
    {
        if(axis != seqIdx)
        {
            space += (dims[axis] - 1) * strides[axis];
        }
    }
    return {static_cast<size_t>(elementCount), static_cast<size_t>(space)};
}

/// Whether @p dims is `[count, 1, 1, 1]`, the shape of a ragged offset table (count = B + 1) and
/// of a seq_len tensor (count = B).
inline bool isBatchAxisVector(const std::vector<int64_t>& dims, int64_t count)
{
    return dims == std::vector<int64_t>{count, 1, 1, 1};
}

} // namespace detail

/**
 * @brief Shared, ragged-aware base for RaggedTensor<T> and ShallowRaggedTensor<T>.
 *
 * Holds the `ragged_offset` aux type-erased as a `std::shared_ptr<ITensor>` (both
 * int32 and int64 element types are accepted; the element type is discovered at
 * runtime). An optional `seq_len` aux of the same kinds gives per-batch lengths.
 * Per batch the tensor derives an origin (`offset[b] * multiplier`) and a length (`seq_len[b]`,
 * or the offset block clamped to S_max when no `seq_len` is given); the offsets and lengths are
 * snapshotted at construction. Elements are the rows `s < len[b]` of each batch at
 * `origin[b] + sum(idx * stride)`. `elementCount` counts those elements; `elementSpace` is the
 * addressed span (see `detail::raggedFootprint` for the layout rules).
 *
 * This intermediate carries everything the two concrete ragged types share — ragged addressing
 * (`getIndexImpl`), the traversal hook (`raggedIterationInfo`), geometry reporting, and
 * structural validation — leaving the concrete types responsible only for the memory carrier and
 * fill operations.
 *
 * No CRTP is required because `memory()` is already virtual on `TensorBase<T>`.
 */
template <typename T>
class RaggedTensorBase : public TensorBase<T>
{
public:
    RaggedTensorBase(std::vector<int64_t> paddedDims,
                     std::vector<int64_t> strides,
                     int seqAxis,
                     std::shared_ptr<ITensor> raggedOffset,
                     std::optional<size_t> physicalElementCount,
                     int64_t raggedOffsetMultiplier = 1,
                     const std::shared_ptr<ITensor>& raggedLength = nullptr)
        : _paddedDims(std::move(paddedDims))
        , _strides(std::move(strides))
        , _seqAxis(seqAxis)
        , _raggedOffset(std::move(raggedOffset))
        , _raggedOffsetMultiplier(raggedOffsetMultiplier)
    {
        // Structural checks first, so the aux reads below only see a supported element size and
        // _seqAxis is a valid index into _strides.
        validateRaggedStructure(raggedLength.get());
        _seqStride = _strides[static_cast<size_t>(_seqAxis)];

        // Snapshot the B+1 offset table and the optional B lengths. Reading an aux may trigger
        // a device->host sync if it lives in device memory.
        const auto batchCount = static_cast<size_t>(_paddedDims[0]);
        const std::vector<int64_t> offsets = readIndexTensor(
            *_raggedOffset, batchCount + 1, _raggedOffsetMultiplier, "ragged_offset");
        std::vector<int64_t> lengthValues;
        if(raggedLength != nullptr)
        {
            lengthValues = readIndexTensor(*raggedLength, batchCount, 1, "seq_len");
        }

        auto batches = detail::raggedBatches(offsets,
                                             raggedLength != nullptr ? &lengthValues : nullptr,
                                             _seqStride,
                                             _paddedDims[static_cast<size_t>(_seqAxis)]);
        const detail::RaggedFootprint footprint
            = detail::raggedFootprint(_paddedDims, _strides, _seqAxis, batches);

        // An explicit physicalElementCount is a redundant confirmation and must match.
        if(physicalElementCount.has_value() && *physicalElementCount != footprint.elementSpace)
        {
            throw std::invalid_argument(
                "physicalElementCount (" + std::to_string(*physicalElementCount)
                + ") must equal elementSpace (" + std::to_string(footprint.elementSpace) + ")");
        }

        _origins = std::move(batches.origins);
        _lengths = std::move(batches.lengths);
        _iteratedElementCount = footprint.elementCount;
        _physicalElementCount = footprint.elementSpace;
    }

    const std::vector<int64_t>& dims() const override
    {
        return _paddedDims;
    }

    const std::vector<int64_t>& strides() const override
    {
        return _strides;
    }

    // Iterated elements: sum(len[b]) * prod(non-batch, non-seq dims) (RFC 0014 §4.5 item 6).
    size_t elementCount() const override
    {
        return _iteratedElementCount;
    }

    // Addressed buffer span (RFC 0014 §4.5 item 6); equals elementCount() for packed layouts.
    size_t elementSpace() const override
    {
        return _physicalElementCount;
    }

    // Diverges from the dense convention `_packed = (elementCount == elementSpace)`:
    // those may be equal here, yet the buffer is neither prod(dims) elements long nor
    // regularly strided, so it must not be treated as a flat dense buffer (RFC 0014 §4.5 item 7).
    bool isPacked() const override
    {
        return false;
    }

    std::optional<RaggedIterationInfo> raggedIterationInfo() const override
    {
        return RaggedIterationInfo{_origins, _lengths, _seqAxis, _seqStride};
    }

    const ITensor* raggedOffset() const
    {
        return _raggedOffset.get();
    }

protected:
    // Ragged addressing: base each batch at its element-unit origin, then add the remaining
    // indices against the padded strides. A bare index {b} bases at the batch's origin.
    int64_t getIndexImpl(const std::vector<int64_t>& indices) const override
    {
        if(indices.empty())
        {
            return 0;
        }

        const int64_t base = _origins.at(static_cast<size_t>(indices[0]));
        return base
               + std::inner_product(std::next(indices.begin()),
                                    indices.end(),
                                    std::next(_strides.begin()),
                                    int64_t{0});
    }

    // Type-erased read of the first `count` values of an index aux, widened to int64_t and scaled
    // by `scale` (RFC 0014 §4.5 item 1). The single boundary where the offset multiplier is applied,
    // so every downstream consumer operates on element units. Value i sits at physical element
    // i * strides()[0]; a stride only matters, and so is only validated, with more than one value.
    static std::vector<int64_t>
        readIndexTensor(ITensor& aux, size_t count, int64_t scale, const char* name)
    {
        const int64_t stride = aux.strides().empty() ? 1 : aux.strides()[0];
        if(count > 1 && stride < 1)
        {
            throw std::invalid_argument(std::string(name) + " batch stride must be >= 1 (got "
                                        + std::to_string(stride) + ")");
        }
        std::vector<int64_t> values;
        values.reserve(count);
        for(size_t i = 0; i < count; ++i)
        {
            const void* p = aux.hostDataOffsetFromIndex(static_cast<int64_t>(i) * stride);
            switch(aux.elementSize())
            {
            case 4:
                values.push_back(*static_cast<const int32_t*>(p) * scale);
                break;
            case 8:
                values.push_back(*static_cast<const int64_t*>(p) * scale);
                break;
            default:
                throw std::runtime_error("ragged index aux element size must be 4 or 8 bytes");
            }
        }
        return values;
    }

    // Constructor-time structural validation, shared by both concrete types (RFC 0014 §4.5 item 5).
    void validateRaggedStructure(const ITensor* raggedLength) const
    {
        if(_raggedOffset == nullptr)
        {
            throw std::invalid_argument("ragged_offset must not be null");
        }
        if(_raggedOffsetMultiplier < 1)
        {
            throw std::invalid_argument("ragged_offset_multiplier must be >= 1 (got "
                                        + std::to_string(_raggedOffsetMultiplier) + ")");
        }
        if(_paddedDims.size() < 2)
        {
            throw std::invalid_argument("paddedDims must have rank >= 2 (got "
                                        + std::to_string(_paddedDims.size()) + ")");
        }
        if(_strides.size() != _paddedDims.size())
        {
            throw std::invalid_argument("strides size (" + std::to_string(_strides.size())
                                        + ") must equal paddedDims size ("
                                        + std::to_string(_paddedDims.size()) + ")");
        }
        if(_seqAxis < 1 || _seqAxis >= static_cast<int>(_paddedDims.size()))
        {
            throw std::invalid_argument("sequence axis (" + std::to_string(_seqAxis)
                                        + ") must be in [1, rank)");
        }
        const size_t expectedCount = static_cast<size_t>(_paddedDims[0]) + 1;
        if(_raggedOffset->elementCount() != expectedCount)
        {
            throw std::invalid_argument(
                "ragged_offset elementCount (" + std::to_string(_raggedOffset->elementCount())
                + ") must equal paddedDims[0] + 1 (" + std::to_string(expectedCount) + ")");
        }
        // Values are read along axis 0, so they must all lie on it.
        if(!detail::isBatchAxisVector(_raggedOffset->dims(), static_cast<int64_t>(expectedCount)))
        {
            throw std::invalid_argument("ragged_offset must have shape [B + 1, 1, 1, 1]");
        }
        const size_t elementSize = _raggedOffset->elementSize();
        if(elementSize != 4 && elementSize != 8)
        {
            throw std::invalid_argument("ragged_offset element size must be 4 or 8 bytes (got "
                                        + std::to_string(elementSize) + ")");
        }

        if(raggedLength == nullptr)
        {
            return;
        }
        const auto batchCount = static_cast<size_t>(_paddedDims[0]);
        if(raggedLength->elementCount() != batchCount)
        {
            throw std::invalid_argument(
                "seq_len elementCount (" + std::to_string(raggedLength->elementCount())
                + ") must equal paddedDims[0] (" + std::to_string(batchCount) + ")");
        }
        if(!detail::isBatchAxisVector(raggedLength->dims(), static_cast<int64_t>(batchCount)))
        {
            throw std::invalid_argument("seq_len must have shape [B, 1, 1, 1]");
        }
        const size_t lengthElementSize = raggedLength->elementSize();
        if(lengthElementSize != 4 && lengthElementSize != 8)
        {
            throw std::invalid_argument("seq_len element size must be 4 or 8 bytes (got "
                                        + std::to_string(lengthElementSize) + ")");
        }
    }

    std::vector<int64_t> _paddedDims;
    std::vector<int64_t> _strides;
    int _seqAxis;
    int64_t _seqStride{1};
    std::vector<int64_t> _origins; ///< per-batch origin, element units (B entries)
    std::vector<int64_t> _lengths; ///< per-batch length in sequence rows (B entries)
    size_t _iteratedElementCount{0}; ///< logical element count
    size_t _physicalElementCount{0}; ///< addressed buffer span (elementSpace)
    std::shared_ptr<ITensor> _raggedOffset; ///< non-null, fixed at construction, never reseated
    int64_t _raggedOffsetMultiplier{1}; ///< stored_offset -> element_offset scale
};

/**
 * @brief Memory-owning, ragged-aware tensor.
 *
 * Physical element count is independent of `prod(dims)`; the buffer is sized to
 * `elementSpace()`. Holds a `shared_ptr` to its `ragged_offset` aux, fixed at
 * construction; the optional `seq_len` aux gives per-batch lengths. Pinned host memory is available via
 * `RaggedTensor<T, PinnedHostAllocator<T>>` (alias `PinnedRaggedTensor<T>`).
 */
template <class T, class HostAlloc = HostAllocator<T>, class DeviceAlloc = DeviceAllocator<T>>
class RaggedTensor : public RaggedTensorBase<T>
{
public:
    RaggedTensor(std::vector<int64_t> paddedDims,
                 std::vector<int64_t> strides,
                 int seqAxis,
                 std::shared_ptr<ITensor> raggedOffset,
                 std::optional<size_t> physicalElementCount = std::nullopt,
                 int64_t raggedOffsetMultiplier = 1,
                 const std::shared_ptr<ITensor>& raggedLength = nullptr)
        : RaggedTensorBase<T>(std::move(paddedDims),
                              std::move(strides),
                              seqAxis,
                              std::move(raggedOffset),
                              physicalElementCount,
                              raggedOffsetMultiplier,
                              raggedLength)
    {
        _memory = MigratableMemory<T, HostAlloc, DeviceAlloc>(this->elementSpace());
    }

    RaggedTensor(const RaggedTensor&) = delete;
    RaggedTensor& operator=(const RaggedTensor&) = delete;

    RaggedTensor(RaggedTensor&&) = default;
    RaggedTensor& operator=(RaggedTensor&&) = default;

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

private:
    MigratableMemory<T, HostAlloc, DeviceAlloc> _memory;
};

template <typename T>
using PinnedRaggedTensor = RaggedTensor<T, PinnedHostAllocator<T>>;

// NOLINTEND(portability-template-virtual-member-function)

} // namespace hipdnn_data_sdk::utilities
