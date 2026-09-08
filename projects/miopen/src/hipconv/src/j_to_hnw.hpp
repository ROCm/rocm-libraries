#pragma once

namespace hipconv
{

struct JConfig
{
    int th;          // tile size h
    int tn;          // tile size n
    int tw;          // tile size w
    int wmma_size_j; // WMMA size of J-mode
    int tiles_j;     // Number of waves in J-direction
    __device__ constexpr auto reg_stride_j() const -> int
    {
        return tn * th * tw / (wmma_size_j * tiles_j);
    }
};

// The columns of the B-matrix are called mode "J". Mode J is partioned into wave index jw, block
// index jb, and column-in-block index j. The mapping between J and (jw,jb,j) is given by
//
// J = jw * (reg_stride_j * wmma_size_j) + jb * wmma_size_j + j
//
// Moreover, we have the logical th x tn x tw tile (that is a subtensor of the height x batch x
// width tensor). The flattend tile index is computed as
//
// HNW = h * (tn * tw) + n * tw + w
//
// We want to recover h,n,w from jw,jb,j in order to fold the th x tn x tw tile into the columns of
// the B-matrix. Therefore, we need
//
// (h,n,w) = HNW^{-1}(J(jw,jb,j))
//
// JToHNW computes above mapping. Specialization can be generated with the following Python code and
// the swizzle package:
//
// import math
// from swizzle.layout import LinearLayout
//
// tiles_j = 4
// wmma_size_j = 16
// ilog2 = lambda x: int(math.log2(x))
//
// for th, tn, tw in [(8, 4, 8), (16, 1, 16)]:
//     reg_stride_j = tn * th * tw // (wmma_size_j * tiles_j)
//     hnw = LinearLayout.from_function(
//         lambda x: (x[0] * tn * tw + x[1] * tw + x[2], ), ['h', 'n', 'w'],
//         (ilog2(th), ilog2(tn), ilog2(tw)), ['offset'])
//     j = LinearLayout.from_function(
//         lambda x:
//         (x[0] * wmma_size_j * reg_stride_j + x[1] * wmma_size_j + x[2], ),
//         ['jw', 'jb', 'j'],
//         (ilog2(tiles_j), ilog2(reg_stride_j), ilog2(wmma_size_j)), ['offset'])
//
//     print(
//         f'JConfig{{.th={th},.tn={tn},.tw={tw},.wmma_size_j={wmma_size_j},.tiles_j={tiles_j}}}'
//     )
//     for f in hnw.inverted().composed(j).formula():
//         print(f'\tconst int {f[0]} = {f[1]};')
//
template <JConfig cfg>
struct JToHNW
{
    __device__ constexpr static auto convert(int jw, int jb, int j) -> std::array<int, 3>
    {
        const int idx = jw * (cfg.reg_stride_j() * cfg.wmma_size_j) + jb * cfg.wmma_size_j + j;
        const int w   = idx % cfg.tw;
        const int n   = idx / cfg.tw % cfg.tn;
        const int h   = idx / (cfg.tw * cfg.tn);
        return {h, n, w};
    }
};
template <>
struct JToHNW<JConfig{.th = 8, .tn = 4, .tw = 8, .wmma_size_j = 16, .tiles_j = 4}>
{
    __device__ constexpr static auto convert(int jw, int jb, int j) -> std::array<int, 3>
    {
        const int h = jw * 2 ^ jb / 2;
        const int n = jb % 2 * 2 ^ j / 8;
        const int w = j % 8;
        return {h, n, w};
    }
};
template <>
struct JToHNW<JConfig{.th = 16, .tn = 1, .tw = 16, .wmma_size_j = 16, .tiles_j = 4}>
{
    __device__ constexpr static auto convert(int jw, int jb, int j) -> std::array<int, 3>
    {
        const int h = jw * 4 ^ jb;
        const int n = 0;
        const int w = j;
        return {h, n, w};
    }
};

} // namespace hipconv
