/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_AGENT_SINGLE_PASS_SCAN_OPERATORS_HPP_
#define HIPCUB_ROCPRIM_AGENT_SINGLE_PASS_SCAN_OPERATORS_HPP_

#include "../../../config.hpp"
#include "../../../util_type.hpp"

#include <hip/hip_runtime.h>

#include <rocprim/device/detail/lookback_scan_state.hpp> // IWYU pragma: export

BEGIN_HIPCUB_NAMESPACE

template<typename T, typename ScanOpT>
struct BlockScanRunningPrefixOp
{
    /// \brief Wrapped scan operator
    ScanOpT op;

    /// \brief Running block-wide prefix
    T running_total;

    /// Constructor
    HIPCUB_DEVICE
    HIPCUB_FORCEINLINE BlockScanRunningPrefixOp(ScanOpT op)
        : op(op)
    {}

    /// Constructor
    HIPCUB_DEVICE
    HIPCUB_FORCEINLINE BlockScanRunningPrefixOp(T starting_prefix, ScanOpT op)
        : op(op), running_total(starting_prefix)
    {}

    /// Prefix callback operator.  Returns the block-wide running_total in thread-0.
    ///
    /// \param block_aggregate The aggregate sum of the BlockScan inputs
    HIPCUB_DEVICE
    HIPCUB_FORCEINLINE T
        operator()(const T& block_aggregate)
    {
        T retval      = running_total;
        running_total = op(running_total, block_aggregate);
        return retval;
    }
};

// Forward declare for use in detail namespace.
// CUB uses cub::detail::is_primitive<T>::value for deducing SINGLE_WORD, but this isn't needed by the rocPRIM backend.
template<typename T,
         bool SINGLE_WORD = (sizeof(T) <= 7) /* hipcub::detail::is_primitive<T>::value */>
class ScanTileState;

namespace detail
{
/// \brief Dummy `hipcub::detail::default_delay` for API compatibility.
template<typename T>
struct default_delay_t
{};

/// \brief Wraps a `hipcub::ScanTileState`-like to a behave as
/// `rocprim::detail::lookback_scan_state`.
template<typename ScanTileState>
class ScanTileStateAsInternal
{
private:
    /// \brief The `hipcub::ScanTileState` that is being wrapped.
    ScanTileState wrapped;

public:
    using value_type           = typename ScanTileState::StatusValueT;
    using flag_underlying_type = typename ScanTileState::StatusWord;

    HIPCUB_FORCEINLINE HIPCUB_HOST_DEVICE ScanTileStateAsInternal(ScanTileState to_be_wrapped)
        : wrapped(to_be_wrapped)
    {}

    HIPCUB_FORCEINLINE HIPCUB_HOST hipError_t static create(ScanTileStateAsInternal& state,
                                                            void*                    temp_storage,
                                                            const unsigned int       num_tiles,
                                                            const hipStream_t /* stream */)
    {
        return state.wrapped.Init(num_tiles, temp_storage, 0);
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void initialize_prefix(const unsigned int block_id, const unsigned int num_tiles)
    {
        wrapped.InitializeStatus(block_id, num_tiles);
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void set_partial(const unsigned int tile_idx, value_type tile_partial)
    {
        wrapped.SetPartial(tile_idx, tile_partial);
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void set_complete(const unsigned int tile_idx, value_type tile_inclusive)
    {
        wrapped.SetInclusive(tile_idx, tile_inclusive);
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void get(const unsigned int block_id, flag_underlying_type& flag, value_type& value)
    {
        wrapped.WaitForValid(block_id, flag, value, default_delay_t<value_type>{});
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    value_type get_complete_value(const unsigned int tile_idx)
    {
        return wrapped.LoadValid(tile_idx);
    }

    // This only used for deterministic execution and not for nondeterministic.
    template<typename ScanOpT>
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    value_type get_prefix_forward(ScanOpT /* op */, const unsigned int tile_idx)
    {
        // We can ignore this because we only allow nondeterministic execution!
        __builtin_unreachable();

        return wrapped.LoadValid(tile_idx);
    }
};

/// \brief Attempts to convert a CUB-styled `ScanTileState` into a rocPRIM
/// backed tile scan state.
template<typename ScanTileState>
struct ScanTileStateConverter
{
    using type = ScanTileStateAsInternal<ScanTileState>;

    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE static type as_native(type wrapper)
    {
        return type(wrapper);
    }
};

template<typename T, bool SINGLE_WORD>
struct ScanTileStateConverter<ScanTileState<T, SINGLE_WORD>>
{
    using type = typename ScanTileState<T, SINGLE_WORD>::NativeT;

    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE static type as_native(ScanTileState<T, SINGLE_WORD> wrapper)
    {
        return wrapper.internal;
    }
};

} // namespace detail

// Since enums use the backing fundamental we have to get that backing integral
// value from original enum. A down side is that the exact values don't necessarily
// line up with CUB.
enum ScanTileStatus
{
    SCAN_TILE_OOB       = static_cast<int>(rocprim::detail::lookback_scan_prefix_flag::invalid),
    SCAN_TILE_INVALID   = static_cast<int>(rocprim::detail::lookback_scan_prefix_flag::empty),
    SCAN_TILE_PARTIAL   = static_cast<int>(rocprim::detail::lookback_scan_prefix_flag::partial),
    SCAN_TILE_INCLUSIVE = static_cast<int>(rocprim::detail::lookback_scan_prefix_flag::complete),
};

/**
 * Enum class used in CUB for specifying the memory order that shall be enforced while reading
 * and writing the tile status. Not used in rocPRIM backend, but it's present for compatibility.
 */
enum class MemoryOrder
{
    relaxed,
    acquire_release
};

/**
 * \brief Tile status, which consists of a scan status and a prefix value packed together.
 * 
 * \par Overview
 * - rocPRIM has its own implementation of the decoupled look-back, and therefore hipCUB exposes
 * this API exclusively for compatibility. That is, it is not internally used by any algorithm.
 * - Beware that some member variables may not be present and some member methods may not behave
 * as expected yet, as not all of rocPRIM's internal implementation of the decoupled look-back
 * is accesible from hipCUB.
 * 
 * \tparam T           Type of the values scanned.
 * \tparam SINGLE_WORD Whether the scan status and value type fit into one machine word that can
 * be loaded/stored using single atomic instructions.
**/
template<typename T, bool SINGLE_WORD>
class ScanTileState
{
    // Converter needs to access internal scan state.
    friend struct detail::ScanTileStateConverter<ScanTileState>;

private:
    using NativeT = rocprim::detail::lookback_scan_state<T, true, SINGLE_WORD>;
    NativeT internal;

public:
    using StatusValueT = T;
    using StatusWord   = rocprim::detail::lookback_scan_prefix_flag;

    enum
    {
        // Constant used by CUB, but unneeded by rocPRIM.
        TILE_STATUS_PADDING = HIPCUB_DEVICE_WARP_THREADS,
    };

    // Available in 'rocprim::detail::lookback_scan_state', but it's private...
    //   StatusWord*& d_tile_status;

    // Available in 'rocprim::detail::lookback_scan_state', but it's private and
    // only in the small size specialization...
    //   T*&          d_tile_partial;
    //   T*&          d_tile_inclusive;

    /// \brief Initializer
    ///
    /// \param[in] num_tiles
    /// Number of tiles
    ///
    /// \param[in] d_temp_storage
    /// Device-accessible allocation of temporary storage.
    /// When nullptr, the required allocation size is written to \p temp_storage_bytes and no work is
    /// done.
    ///
    /// \param[in] temp_storage_bytes
    /// Size in bytes of \t d_temp_storage allocation
    HIPCUB_FORCEINLINE
    HIPCUB_HOST_DEVICE
    hipError_t Init(int num_tiles, void* d_temp_storage, size_t /* temp_storage_bytes */)
    {
        // rocprim::detail::lookback_scan_state::create(...) is host only, so this function
        // won't work on device.
        return NativeT::create(internal, d_temp_storage, num_tiles, hipStreamDefault);
    }

    /// \brief Compute device memory needed for tile status
    ///
    /// \param[in] num_tiles
    ///   Number of tiles
    ///
    /// \param[out] temp_storage_bytes
    ///   Size in bytes of \t d_temp_storage allocation
    HIPCUB_FORCEINLINE
    HIPCUB_HOST_DEVICE
    static hipError_t AllocationSize(int num_tiles, size_t& temp_storage_bytes)
    {
        // rocprim::detail::lookback_scan_state::create(...) is host only, so this function
        // won't work on device.
        return NativeT::get_storage_size(num_tiles, hipStreamDefault, temp_storage_bytes);
    }

    /// \brief Initalize (from device).
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void InitializeStatus(int num_tiles)
    {
        int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        internal.initialize_prefix(tile_idx, num_tiles);
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void SetInclusive(int tile_idx, T tile_inclusive)
    {
        internal.set_complete(tile_idx, tile_inclusive);
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void SetPartial(int tile_idx, T tile_partial)
    {
        internal.set_partial(tile_idx, tile_partial);
    }

    /// \brief Wait for the corresponding tile to become non-invalid.
    template<typename DelayT = detail::default_delay_t<T>>
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    void WaitForValid(int         tile_idx,
                      StatusWord& status,
                      T&          value,
                      DelayT /* delay_or_prevent_hoisting */ = {})
    {
        internal.get(tile_idx, status, value);
    }

    /// \brief Loads and returns the tile's value.
    ///
    /// The returned value is undefined if either (a) the tile's status is invalid or
    /// (b) there is no memory fence between reading a non-invalid status and the call to LoadValid.
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    T LoadValid(int tile_idx)
    {
        return internal.get_complete_value(tile_idx);
    }
};

/**
 * \brief (Block) Scan prefix functor for retrieving the current tile prefix.
 * 
 * \par Overview
 * - rocPRIM has its own implementation of the decoupled look-back, and therefore hipCUB exposes
 * this API exclusively for compatibility. That is, it is not internally used by any algorithm.
 * - Beware that some member variables may not be present and some member methods may not behave
 * as expected yet, as not all of rocPRIM's internal implementation of the decoupled look-back
 * is accesible from hipCUB.
 *
 * \tparam T                 Type of the values scanned.
 * \tparam ScanOpT           Scan operation type.
 * \tparam ScanTileStateT    Scan status type.
 * \tparam LEGACY_PTX_ARCH   <b>[optional]</b> Unused (deprecated).
 * \tparam DelayConstructorT <b>[optional]</b> Unused (CUB's implementation detail).
 */
template<typename T,
         typename ScanOpT,
         typename ScanTileStateT,
         int LEGACY_PTX_ARCH        = 0,
         typename DelayConstructorT = void /* detail::default_delay_constructor_t<T> */>
class TilePrefixCallbackOp
{
private:
    /// \brief Convert `ScanTileStateT` into something digestable by the backing
    /// implementation in rocPRIM.
    ///
    /// This is to prevent double wrapping where a rocPRIM implementation of
    /// `ScanTileStateT` is wrapped to work with hipCUB and then wrapped again
    /// to work within rocPRIM.
    using ScanTileStateConverter =
        // rocPRIM 'rocprim::detail::offset_lookback_scan_prefix_op' expects a
        // 'rocprim::detail::lookback_scan_state'. Theoretically, a hipCUB user
        // can provide their own implementation of ScanTileStateT so we need to
        // convert 'hipcub::ScanTileState' into the expected API.
        detail::ScanTileStateConverter<ScanTileStateT>;

    /// \brief The native-like type of scan tile state after converting.
    using NativeScanTileStateT = typename ScanTileStateConverter::type;

    /// \brief The type of the native implementation in rocPRIM.
    using NativeT = rocprim::detail::offset_lookback_scan_prefix_op<
        T,
        NativeScanTileStateT,
        ScanOpT,
        // rocPRIM's deterministic lookback scan requires forward prefix which
        // CUB doesn't have. We force non-deterministism as rocPRIM might
        // change defaults.
        rocprim::detail::lookback_scan_determinism::nondeterministic>;

public:
    using TempStorage = typename NativeT::storage_type;
    using StatusWord  = rocprim::detail::lookback_scan_prefix_flag;

    // 'tile_idx' and 'scan_op' are exposed by CUB as non-const and *technically*
    // available through rocPRIM, but they're protected/private and it's not
    // worth making that public and tightly coupling this wrapper even more. We
    // instead provide a middle ground by exposing it as a const.

    /// \brief The current tile index.
    const int tile_idx;

    /// \brief Binary scan operator.
    const ScanOpT scan_op;

    TempStorage& temp_storage;

    /// \brief Interface to the tile status.
    ScanTileStateT& tile_status;

private:
    /// \brief The tile status, but wrapped to behave as required by rocPRIM.
    NativeScanTileStateT proxied_tile_status;

    /// \brief The backing implementation in rocPRIM.
    NativeT internal;

public:
    /// \brief Exclusive prefix for this tile.
    T exclusive_prefix;

    /// \brief Inclusive prefix for this tile.
    T inclusive_prefix;

    HIPCUB_FORCEINLINE HIPCUB_DEVICE TilePrefixCallbackOp(ScanTileStateT& tile_status,
                                                          TempStorage&    temp_storage,
                                                          ScanOpT         scan_op,
                                                          int             tile_idx)
        : tile_idx(tile_idx)
        , scan_op(scan_op)
        , temp_storage(temp_storage)
        , tile_status(tile_status)
        , proxied_tile_status(ScanTileStateConverter::as_native(tile_status))
        , internal(tile_idx, proxied_tile_status, temp_storage, scan_op)
    {}

    /// \brief Constructs prefix functor for a given tile index.
    HIPCUB_FORCEINLINE HIPCUB_DEVICE TilePrefixCallbackOp(ScanTileStateT& tile_status,
                                                          TempStorage&    temp_storage,
                                                          ScanOpT         scan_op)
        : TilePrefixCallbackOp(tile_status, temp_storage, scan_op, tile_idx)
    {}

    /// \brief BlockScan prefix callback functor (called by the first warp)
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    T operator()(T block_aggregate)
    {
        auto result = internal(block_aggregate);

        exclusive_prefix = GetExclusivePrefix();
        inclusive_prefix = GetInclusivePrefix();

        return result;
    }

    /// \brief Get the exclusive prefix stored in temporary storage.
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    T GetExclusivePrefix()
    {
        return internal.get_exclusive_prefix();
    }

    /// \brief Get the block aggregate stored in temporary storage.
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    T GetBlockAggregate()
    {
        return internal.get_reduction();
    }

    /// \brief Get the inclusive prefix stored in temporary storage.
    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    T GetInclusivePrefix()
    {
        return scan_op(GetExclusivePrefix(), GetBlockAggregate());
    }

    HIPCUB_FORCEINLINE HIPCUB_DEVICE
    int GetTileIdx() const
    {
        return tile_idx;
    }
};

// CUB uses cub::detail::is_primitive<T>::value for deducing SINGLE_WORD, but this isn't needed by the rocPRIM backend.
template<typename ValueT, typename KeyT, bool SINGLE_WORD = (sizeof(ValueT) + sizeof(KeyT) <= 7)>
struct ReduceByKeyScanTileState : hipcub::ScanTileState<KeyValuePair<KeyT, ValueT>>
{
    using SuperClass = hipcub::ScanTileState<KeyValuePair<KeyT, ValueT>, SINGLE_WORD>;

    /// Constructor
    HIPCUB_HOST_DEVICE
    HIPCUB_FORCEINLINE ReduceByKeyScanTileState()
        : SuperClass()
    {}
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_AGENT_SINGLE_PASS_SCAN_OPERATORS_HPP_
