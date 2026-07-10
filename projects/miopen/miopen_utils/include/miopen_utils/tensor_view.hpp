// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// NOTE: intentionally shares the include guard GUARD_TENSOR_VIEW_HPP with the
// canonical src/kernels/tensor_view.hpp so that a translation unit which pulls
// in both (e.g. test code including a MIOpen internal header plus tensor_desc.hpp)
// sees the struct definitions only once. The two copies MUST stay in sync.
#ifndef GUARD_TENSOR_VIEW_HPP
#define GUARD_TENSOR_VIEW_HPP

// HOST-SIDE copy of tensor_view_t and tensor_layout_t.
//
// The canonical device-side copy lives at src/kernels/tensor_view.hpp and is
// registered with HIPRTC for runtime GPU kernel compilation. That copy cannot
// be moved because HIPRTC resolves includes by flat filename only.
//
// This host-side copy exists so that driver, test, and miopen_utils code can
// use tensor_view_t without reaching into MIOpen internals via relative paths.
//
// WARNING: These structs must stay in sync with src/kernels/tensor_view.hpp.
// If you change the layout, indexing, or semantics in either copy, update the
// other to match.

#include <cstdint>
#include <initializer_list>

template <int N>
struct tensor_layout_t;

template <int N>
struct tensor_view_t
{
    constexpr uint64_t get_tensor_view_idx(const tensor_layout_t<N>& tensor_layout)
    {
        static_assert(N > 0);
        uint64_t idx = 0;
        for(auto i = 0; i < N; ++i)
        {
            idx += stride[i] * tensor_layout.layout[i];
        }
        return idx;
    }
    uint64_t stride[N];
    uint64_t size[N];
};

template <int N>
struct tensor_layout_t
{
    constexpr tensor_layout_t(const tensor_view_t<N>& tensor_view, uint64_t idx)
    {
        static_assert(N > 0);
        uint64_t temp = idx;
        if constexpr(N == 1)
        {
            layout[0] = idx;
        }
        else
        {
            for(auto i = N - 1; i > 1; --i)
            {
                layout[i] = temp % tensor_view.size[i];
                temp      = temp / tensor_view.size[i];
            }
            layout[1] = temp % tensor_view.size[1];
            layout[0] = temp / tensor_view.size[1];
        }
    }

    constexpr tensor_layout_t(std::initializer_list<uint64_t> layout_)
    {
        static_assert(N > 0);
        for(auto i = 0; i < N; ++i)
        {
            layout[i] = layout_.begin()[i];
        }
    }

    uint64_t layout[N];
};

// ---------------------------------------------------------------------------
// Host-side tensor view helpers
// ---------------------------------------------------------------------------

/// Remove a dimension from a tensor view, collapsing N dims to N-1.
template <int N>
inline tensor_view_t<N - 1> get_tv_without_dim(const tensor_view_t<N>& origin_tv, int selected_dim)
{
    tensor_view_t<N - 1> res{};
    for(int i = 0; i < N; ++i)
    {
        if(i == selected_dim)
            continue;
        if(i < selected_dim)
        {
            res.size[i]   = origin_tv.size[i];
            res.stride[i] = origin_tv.stride[i];
        }
        else
        {
            res.size[i - 1]   = origin_tv.size[i];
            res.stride[i - 1] = origin_tv.stride[i];
        }
    }
    return res;
}

/// Apply slicing to a tensor view (modifies in place).
template <int N>
inline void slice_tv(tensor_view_t<N>& tensor_view, int32_t sliceCount, const int32_t* slices)
{
    for(int32_t i = 0; i < sliceCount; i++)
    {
        int32_t dim   = slices[4 * i + 0];
        int32_t start = slices[4 * i + 1];
        int32_t end   = slices[4 * i + 2];
        int32_t step  = slices[4 * i + 3];

        if(end > static_cast<int32_t>(tensor_view.size[dim]))
            end = tensor_view.size[dim];

        auto len = end - start;

        tensor_view.size[dim] = (len + step - 1) / step;
        tensor_view.stride[dim] *= step;
    }
}

#endif // GUARD_MIOPEN_UTILS_TENSOR_VIEW_HPP
