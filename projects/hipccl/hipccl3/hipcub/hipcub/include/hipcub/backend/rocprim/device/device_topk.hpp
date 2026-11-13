// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Modifications Copyright (c) 2026, Advanced Micro Devices, Inc.  All rights reserved.

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_TOPK_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_TOPK_HPP_

#include "../../../config.hpp"

#include "../detail/util_env.hpp"

#include <rocprim/device/device_topk.hpp> // IWYU pragma: export
#include <rocprim/types.hpp> // IWYU pragma: export

#include _HIPCUB_LIBCXX_INCLUDE(__execution/output_ordering.h)
#include _HIPCUB_LIBCXX_INCLUDE(__execution/determinism.h)
#include _HIPCUB_LIBCXX_INCLUDE(__execution/require.h)
#include _HIPCUB_STD_INCLUDE(__execution/env.h)

BEGIN_HIPCUB_NAMESPACE

namespace detail::topk
{
template<typename EnvT>
struct requirements
{
    // Determinism and ordering requirements
    using requirements_t
        = _HIPCUB_STD_EXEC::__query_result_or_t<EnvT,
                                                _HIPCUB_LIBCXX::execution::__get_requirements_t,
                                                _HIPCUB_STD_EXEC::env<>>;
    using requested_determinism_t = _HIPCUB_STD_EXEC::__query_result_or_t<
        requirements_t,
        _HIPCUB_LIBCXX::execution::determinism::__get_determinism_t,
        _HIPCUB_LIBCXX::execution::determinism::run_to_run_t>;
    using requested_order_t = _HIPCUB_STD_EXEC::__query_result_or_t<
        requirements_t,
        _HIPCUB_LIBCXX::execution::output_ordering::__get_output_ordering_t,
        _HIPCUB_LIBCXX::execution::output_ordering::sorted_t>;

    // Ordered = false when user requested unsorted
    static constexpr auto ordered
        = !_HIPCUB_STD::is_same_v<requested_order_t,
                                  _HIPCUB_LIBCXX::execution::output_ordering::unsorted_t>;
    // Deterministic = false when user requested not_guaranteed
    static constexpr auto deterministic
        = !_HIPCUB_STD::is_same_v<requested_determinism_t,
                                  _HIPCUB_LIBCXX::execution::determinism::not_guaranteed_t>;
};
} // namespace detail::topk

struct DeviceTopK
{
    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename ValueInputIteratorT,
             typename ValueOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MaxPairs(void*                d_temp_storage,
                               size_t&              temp_storage_bytes,
                               KeyInputIteratorT    d_keys_in,
                               KeyOutputIteratorT   d_keys_out,
                               ValueInputIteratorT  d_values_in,
                               ValueOutputIteratorT d_values_out,
                               NumItemsT            num_items,
                               NumOutItemsT         k,
                               EnvT                 env = {})
    {
        using reqs = detail::topk::requirements<EnvT>;
        // CUB's behavior is to pass the user-allocated temp storage
        return detail::legacy_invoke(env,
                                     [&](auto stream)
                                     {
                                         return rocprim::topk_pairs<rocprim::default_config,
                                                                    true /*Descending*/,
                                                                    reqs::ordered,
                                                                    reqs::deterministic>(
                                             d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_keys_out,
                                             d_values_in,
                                             d_values_out,
                                             num_items,
                                             k,
                                             ::rocprim::identity_decomposer{},
                                             stream);
                                     });
    }

    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename ValueInputIteratorT,
             typename ValueOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MinPairs(void*                d_temp_storage,
                               size_t&              temp_storage_bytes,
                               KeyInputIteratorT    d_keys_in,
                               KeyOutputIteratorT   d_keys_out,
                               ValueInputIteratorT  d_values_in,
                               ValueOutputIteratorT d_values_out,
                               NumItemsT            num_items,
                               NumOutItemsT         k,
                               EnvT                 env = {})
    {
        using reqs = detail::topk::requirements<EnvT>;
        return detail::legacy_invoke(env,
                                     [&](auto stream)
                                     {
                                         return rocprim::topk_pairs<rocprim::default_config,
                                                                    false /*Descending*/,
                                                                    reqs::ordered,
                                                                    reqs::deterministic>(
                                             d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_keys_out,
                                             d_values_in,
                                             d_values_out,
                                             num_items,
                                             k,
                                             ::rocprim::identity_decomposer{},
                                             stream);
                                     });
    }

    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MaxKeys(void*              d_temp_storage,
                              size_t&            temp_storage_bytes,
                              KeyInputIteratorT  d_keys_in,
                              KeyOutputIteratorT d_keys_out,
                              NumItemsT          num_items,
                              NumOutItemsT       k,
                              EnvT               env = {})
    {
        using reqs = detail::topk::requirements<EnvT>;
        return detail::legacy_invoke(env,
                                     [&](auto stream)
                                     {
                                         return rocprim::topk<rocprim::default_config,
                                                              true /*Descending*/,
                                                              reqs::ordered,
                                                              reqs::deterministic>(
                                             d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_keys_out,
                                             num_items,
                                             k,
                                             ::rocprim::identity_decomposer{},
                                             stream);
                                     });
    }

    template<typename KeyInputIteratorT,
             typename KeyOutputIteratorT,
             typename NumItemsT,
             typename NumOutItemsT,
             typename EnvT = _HIPCUB_STD_EXEC::env<>>
  HIPCUB_RUNTIME_FUNCTION
    static hipError_t MinKeys(void*              d_temp_storage,
                              size_t&            temp_storage_bytes,
                              KeyInputIteratorT  d_keys_in,
                              KeyOutputIteratorT d_keys_out,
                              NumItemsT          num_items,
                              NumOutItemsT       k,
                              EnvT               env = {})
    {
        using reqs = detail::topk::requirements<EnvT>;
        return detail::legacy_invoke(env,
                                     [&](auto stream)
                                     {
                                         return rocprim::topk<rocprim::default_config,
                                                              false /*Descending*/,
                                                              reqs::ordered,
                                                              reqs::deterministic>(
                                             d_temp_storage,
                                             temp_storage_bytes,
                                             d_keys_in,
                                             d_keys_out,
                                             num_items,
                                             k,
                                             ::rocprim::identity_decomposer{},
                                             stream);
                                     });
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_TOPK_HPP_
