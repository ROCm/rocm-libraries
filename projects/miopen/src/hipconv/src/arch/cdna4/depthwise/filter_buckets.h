#pragma once

// Filter sizes the CDNA 4 depthwise families compile, shared by fprop/dgrad (1D Toeplitz)
// and wgrad (Hankel) so both serve the same layers.
//
// Odd squares compile exactly. 1 x kw and kh x 1 compile at the sizes common models use
// (Conformer 31/15/9/7, Mamba 4, SegNeXt 7/11/21, InceptionNeXt 11) plus fallback sizes
// that close the gaps: a layer runs on the smallest compiled size that holds its filter,
// the taps past it zero. Fallback sizes carry a shorter config ladder.

#include <array>

namespace hipconv::cdna4::depthwise_filters
{

struct FilterShape
{
    int kh;
    int kw;
    constexpr bool operator==(const FilterShape&) const = default;
};

constexpr auto square_sizes = std::array{3, 5, 7, 9, 11};

constexpr auto row_exact    = std::array{3, 4, 5, 7, 9, 11, 15, 21, 31};        // 1 x kw
constexpr auto row_fallback = std::array{23, 39, 47, 55, 63, 79, 95, 111, 127}; // 1 x kw
constexpr auto col_exact    = std::array{3, 5, 7, 11, 21};                      // kh x 1
constexpr auto col_fallback = std::array{9, 13, 15, 17, 25, 29, 31};            // kh x 1

// tf32's split operands outgrow registers in fprop/dgrad past kh x 1 25, and its two-plane
// delta ring outgrows LDS in stride-1 wgrad past 29.
constexpr int tf32_max_col_kh       = 25;
constexpr int tf32_max_col_kh_wgrad = 29;

constexpr int max_row_kw = 127;
constexpr int max_col_kh = 31;

template <std::size_t A, std::size_t B>
constexpr int smallest_at_least(int k, const std::array<int, A>& a, const std::array<int, B>& b)
{
    int best = 0;
    for(int x : a)
        if(x >= k && (best == 0 || x < best))
            best = x;
    for(int x : b)
        if(x >= k && (best == 0 || x < best))
            best = x;
    return best;
}

constexpr bool is_one_d(FilterShape f)
{
    return f.kh == 1 || f.kw == 1;
}

// Compiled at a fallback size rather than a common one.
constexpr bool is_fallback(FilterShape f)
{
    for(int x : row_fallback)
        if(f == FilterShape{1, x})
            return true;
    for(int x : col_fallback)
        if(f == FilterShape{x, 1})
            return true;
    return false;
}

// The compiled filter a layer's kh x kw runs on, or {0, 0} if none does.
constexpr FilterShape compiled_filter(int kh, int kw)
{
    if(kh == kw)
    {
        for(int k : square_sizes)
            if(k == kh)
                return {k, k};
        return {0, 0};
    }
    if(kh == 1 && kw > 1)
    {
        const int b = smallest_at_least(kw, row_exact, row_fallback);
        return b ? FilterShape{1, b} : FilterShape{0, 0};
    }
    if(kw == 1 && kh > 1)
    {
        const int b = smallest_at_least(kh, col_exact, col_fallback);
        return b ? FilterShape{b, 1} : FilterShape{0, 0};
    }
    return {0, 0};
}

// Every compiled filter: squares, then 1 x kw, then kh x 1.
constexpr auto filter_shapes = [] {
    std::array<FilterShape,
               square_sizes.size() + row_exact.size() + row_fallback.size() + col_exact.size() +
                   col_fallback.size()>
        f{};
    std::size_t i = 0;
    for(int k : square_sizes)
        f[i++] = {k, k};
    for(int k : row_exact)
        f[i++] = {1, k};
    for(int k : row_fallback)
        f[i++] = {1, k};
    for(int k : col_exact)
        f[i++] = {k, 1};
    for(int k : col_fallback)
        f[i++] = {k, 1};
    return f;
}();

} // namespace hipconv::cdna4::depthwise_filters
