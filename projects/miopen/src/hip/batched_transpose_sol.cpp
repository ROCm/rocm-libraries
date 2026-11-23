/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/env.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>
#include <miopen/magic_div.hpp>
#include <miopen/float_equal.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <limits>
#include <iostream>
#include <sstream>
#include <ranges>

#define BATCHED_TRANSPOSE_BLOCK_SIZE 256
#define BATCHED_TRANSPOSE_PERSISTENT 0

#if BATCHED_TRANSPOSE_PERSISTENT
#define BATCHED_TRANSPOSE_OCCUPANCY 4
#endif

// Compiler has bug for some of the kernel instances, where pack/ediv is not 1.
// By default we'll use a smaller list for gfx942 right now but can also force
// it by using this env var.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_FORCE_HALF_TRANSPOSE_SHORTLIST)

namespace miopen {
namespace batched_transpose {

static inline std::string GetNameTrait(std::size_t type_size)
{
    if(type_size == 1)
        return "byte";
    if(type_size == 2)
        return "half";
    if(type_size == 4)
        return "dword";
    MIOPEN_THROW("data type not supported");
}

static inline const std::vector<BatchedTransposeParam>& GetKernelList(const ExecutionContext& ctx,
                                                                      std::size_t data_size)
{
    if(data_size == 1)
    {
        static const std::vector<BatchedTransposeParam> byte_kernel_list{
            // clang-format off
            {16, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},

            {4, 64, 1, 1, 1, 1},
            {64, 4, 1, 1, 1, 1},
            {4, 128, 1, 1, 1, 1},
            {128, 4, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {256, 4, 1, 1, 1, 1},
            // clang-format on
        };
        return byte_kernel_list;
    }
    if(data_size == 2)
    {
        static const std::vector<BatchedTransposeParam> half_kernel_list_short{
            // clang-format off
            {16, 16, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},

            {4, 64, 1, 1, 1, 1},
            {64, 4, 1, 1, 1, 1},
            {4, 128, 1, 1, 1, 1},
            {128, 4, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {256, 4, 1, 1, 1, 1},
            // clang-format on
        };
        static const std::vector<BatchedTransposeParam> half_kernel_list_long{
            // clang-format off
            {16, 16, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},

            {4, 64, 1, 1, 1, 1},
            {64, 4, 1, 1, 1, 1},
            {4, 128, 1, 1, 1, 1},
            {128, 4, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {256, 4, 1, 1, 1, 1},

            {32, 32, 2, 2, 1, 1},
            {32, 32, 2, 2, 1, 2},
            {32, 32, 2, 2, 2, 1},
            {32, 32, 2, 2, 2, 2},

            {16, 64, 1, 4, 1, 2},
            {64, 16, 4, 1, 2, 1},

            {32, 64, 2, 4, 1, 2},
            {32, 64, 2, 4, 2, 2},
            {32, 64, 2, 4, 2, 4},

            {64, 32, 4, 2, 2, 1},
            {64, 32, 4, 2, 2, 2},
            {64, 32, 4, 2, 4, 2},

            {64, 64, 4, 4, 2, 2},
            {64, 64, 4, 4, 4, 4},
            // clang-format on
        };

        bool force_shortlist   = env::enabled(MIOPEN_FORCE_HALF_TRANSPOSE_SHORTLIST);
        const auto device_name = ctx.GetStream().GetDeviceName();
        if(device_name == "gfx942" || force_shortlist)
            return half_kernel_list_short;

        return half_kernel_list_long;
    }
    if(data_size == 4)
    {
        static const std::vector<BatchedTransposeParam> dword_kernel_list{
            // clang-format off
            {16, 16, 1, 1, 1, 1},
            {16, 32, 1, 1, 1, 1},
            {32, 16, 1, 1, 1, 1},
            {32, 32, 1, 1, 1, 1},

            {4, 64, 1, 1, 1, 1},
            {64, 4, 1, 1, 1, 1},
            {4, 128, 1, 1, 1, 1},
            {128, 4, 1, 1, 1, 1},
            {4, 256, 1, 1, 1, 1},
            {256, 4, 1, 1, 1, 1},
            // clang-format on
        };
        return dword_kernel_list;
    }
    MIOPEN_THROW("data type not supported");
}

static inline bool IsApplicable(uint32_t /* batch */,
                                uint32_t height,
                                uint32_t width,
                                const BatchedTransposeParam* kparam)
{
    return width % kparam->ediv_x == 0 && height % kparam->ediv_y == 0;
}

static inline bool IsSameSide(uint32_t height, uint32_t width, const BatchedTransposeParam* kparam)
{
    float radio = 0;
    if(width > height)
        radio = static_cast<float>(kparam->tile_x) / kparam->tile_y;
    else
        radio = static_cast<float>(kparam->tile_y) / kparam->tile_x;

    // E.g. for cases like width=1000, height=10
    // Allow at least 32x64, 64x64... 16x64 not allowed
    return radio >= 0.4;
}

template <typename T>
static inline float GetNormalizedRadio(T x, T y)
{
    if(y > x)
        return static_cast<float>(y) / x;
    return static_cast<float>(x) / y;
}

static inline std::string GetKernelName(std::size_t data_size, const BatchedTransposeParam* kparam)
{
    std::ostringstream kernel_name;
    std::string type_trait = GetNameTrait(data_size);
    kernel_name << "batched_transpose_" << kparam->tile_x << "x" << kparam->tile_y << "_";
    if(!(kparam->pack_x == 1 && kparam->pack_y == 1 && kparam->ediv_x == 1 && kparam->ediv_y == 1))
    {
        kernel_name << "pack_" << kparam->pack_x << "x" << kparam->pack_y << "_ediv_"
                    << kparam->ediv_x << "x" << kparam->ediv_y << "_";
    }
    kernel_name << type_trait;
    return kernel_name.str();
}

static inline std::size_t GetExtraPaddingSize(uint32_t /* batch */,
                                              uint32_t height,
                                              uint32_t width,
                                              const BatchedTransposeParam* kparam)
{
    // For simplicity and speed, we ignore batch, only compute h*w
    uint32_t padded_h = ((height + kparam->tile_y - 1) / kparam->tile_y) * kparam->tile_y;
    uint32_t padded_w = ((width + kparam->tile_x - 1) / kparam->tile_x) * kparam->tile_x;
    return static_cast<std::size_t>(padded_h) * padded_w - static_cast<std::size_t>(height) * width;
}

// The gfx1250 kernels are used on every gfx1250 target: the runtime can report "gfx1250-strict"
// instead of "gfx1250" (see the kernel guard in batched_transpose.cpp).
static inline bool IsGfx1250(const ExecutionContext& ctx)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return device_name == "gfx1250" || device_name == "gfx1250-strict";
}

// gfx1250 only: for 2-byte data where one dimension is a small multiple of 32 (32 or 64: e.g. the
// channel count of an NCHW<->NHWC transpose) and the other one is large, use the thin LDS kernels
// (256x32_pack_1x8_ediv_1x32 for the small dimension on the height, 32x512_pack_8x1_ediv_32x1 on
// the width). They issue 16-byte global/LDS accesses on the small dimension and work for any (odd)
// size of the other one.
static inline bool TryGetThinKernel(const ExecutionContext& ctx,
                                    std::size_t data_size,
                                    uint32_t batch,
                                    uint32_t height,
                                    uint32_t width,
                                    BatchedTransposeParam& out)
{
    if(data_size != 2)
        return false;
    if(!IsGfx1250(ctx))
        return false;
    constexpr uint32_t min_spatial  = 1024; // smaller spatial sizes: generic kernels are as fast
    constexpr uint64_t min_elements = 1u << 22; // small tensors: generic kernels are as fast
    if(static_cast<uint64_t>(batch) * height * width < min_elements)
        return false;
    const auto small_ok = [](uint32_t c) { return c == 32 || c == 64; };
    // Spatial sizes divisible by 4 are already served well by the packed generic kernels (ediv 2x2
    // / 4x4, 16-byte accesses); the thin kernels are for the other sizes (e.g. 149*149, 147*147,
    // 35*35), where the generic kernels fall back to 2-byte accesses.
    if(small_ok(height) && width >= min_spatial && width % 4 != 0)
    {
        // NCHW -> NHWC style: tiles of 256 columns x 32 rows, 1 element loaded, 8 stored (height %
        // 32 == 0)
        const BatchedTransposeParam param{256, 32, 1, 8, 1, 32};
        if(IsApplicable(batch, height, width, &param))
        {
            out = param;
            return true;
        }
    }
    if(small_ok(width) && height >= min_spatial && height % 4 != 0)
    {
        // NHWC -> NCHW style: tiles of 32 columns x 512 rows, 8 elements loaded, 1 stored (width %
        // 32 == 0)
        const BatchedTransposeParam param{32, 512, 8, 1, 32, 1};
        if(IsApplicable(batch, height, width, &param))
        {
            out = param;
            return true;
        }
    }
    return false;
}

// gfx1250 only: kernel selection rules fitted to measurements of the library kernels on a large set
// of convolution shapes (decision trees, cross validated over the distinct transpose sizes). For
// 2-byte data the rule chooses among the generic, thin-matrix and 16-byte register / LDS kernels,
// for 4-byte data among the vectorized fp32 kernels. Where the chosen kernel is not applicable to
// the size (ediv conditions, thin kernels need the channel count, buffer kernels need height *
// width * data size < 2^31) or the rule says "generic", the generic / thin-matrix heuristic below
// is used.
enum class Kernel2B
{
    Generic,
    Tile128x128_Pack8x8_Ediv8x8,
    Tile128x64_Pack8x8_Ediv8x8,
    Tile16x32_Pack1x1_Ediv1x1,
    Tile256x32_Pack1x8_Ediv1x32,
    Tile32x32_Pack2x2_Ediv2x1,
    Tile32x512_Pack8x1_Ediv32x1,
    Tile32x64_Pack1x8_Ediv1x8,
    Tile32x64_Pack2x4_Ediv2x4,
    Tile4x128_Pack1x1_Ediv1x1,
    Tile4x256_Pack1x1_Ediv1x1,
    Tile512x64_Pack1x8_Ediv1x64,
    Tile64x128_Pack8x4_Ediv8x4,
    Tile64x64_Pack4x4_Ediv4x4
};
enum class Kernel4B
{
    Generic,
    Tile16x32_Pack1x4_Ediv1x4,
    Tile32x256_Pack4x2_Ediv4x2,
    Tile64x128_Pack4x4_Ediv4x4,
    Tile64x32_Pack4x4_Ediv4x4,
    Tile64x64_Pack4x4_Ediv4x4
};

static inline Kernel2B SelectKernel2B(uint32_t batch, uint32_t height, uint32_t width)
{
    const uint64_t elements = static_cast<uint64_t>(batch) * height * width;
    if(height <= 96)
    {
        if(height <= 36)
        {
            if(width <= 50176)
            {
                if(width <= 27)
                {
                    if(width <= 5)
                    {
                        return Kernel2B::Tile4x256_Pack1x1_Ediv1x1;
                    }
                    else
                    {
                        return Kernel2B::Tile16x32_Pack1x1_Ediv1x1;
                    }
                }
                else
                {
                    if(elements <= 1057535)
                    {
                        return Kernel2B::Generic;
                    }
                    else
                    {
                        return Kernel2B::Tile128x64_Pack8x8_Ediv8x8;
                    }
                }
            }
            else
            {
                if(elements <= 5603328)
                {
                    return Kernel2B::Tile64x64_Pack4x4_Ediv4x4;
                }
                else
                {
                    return Kernel2B::Tile256x32_Pack1x8_Ediv1x32;
                }
            }
        }
        else
        {
            if(elements <= 51887628)
            {
                if(width % 16 != 0)
                {
                    if(height % 64 != 0)
                    {
                        return Kernel2B::Tile32x64_Pack1x8_Ediv1x8;
                    }
                    else
                    {
                        return Kernel2B::Tile512x64_Pack1x8_Ediv1x64;
                    }
                }
                else
                {
                    if(height % 8 != 0)
                    {
                        return Kernel2B::Tile32x512_Pack8x1_Ediv32x1;
                    }
                    else
                    {
                        return Kernel2B::Tile128x64_Pack8x8_Ediv8x8;
                    }
                }
            }
            else
            {
                return Kernel2B::Tile512x64_Pack1x8_Ediv1x64;
            }
        }
    }
    else
    {
        if(width % 8 != 0)
        {
            if(width <= 259)
            {
                if(width <= 8)
                {
                    if(height <= 416)
                    {
                        return Kernel2B::Tile4x128_Pack1x1_Ediv1x1;
                    }
                    else
                    {
                        return Kernel2B::Tile4x256_Pack1x1_Ediv1x1;
                    }
                }
                else
                {
                    if(width <= 73)
                    {
                        return Kernel2B::Tile32x64_Pack1x8_Ediv1x8;
                    }
                    else
                    {
                        return Kernel2B::Tile32x64_Pack2x4_Ediv2x4;
                    }
                }
            }
            else
            {
                if(width <= 2725)
                {
                    if(batch <= 9)
                    {
                        return Kernel2B::Tile64x64_Pack4x4_Ediv4x4;
                    }
                    else
                    {
                        return Kernel2B::Tile32x64_Pack1x8_Ediv1x8;
                    }
                }
                else
                {
                    if(width <= 4633)
                    {
                        return Kernel2B::Tile512x64_Pack1x8_Ediv1x64;
                    }
                    else
                    {
                        return Kernel2B::Tile64x64_Pack4x4_Ediv4x4;
                    }
                }
            }
        }
        else
        {
            if(height % 4 != 0)
            {
                if(width % 32 != 0)
                {
                    return Kernel2B::Tile32x32_Pack2x2_Ediv2x1;
                }
                else
                {
                    if(elements <= 1169669)
                    {
                        return Kernel2B::Tile16x32_Pack1x1_Ediv1x1;
                    }
                    else
                    {
                        return Kernel2B::Tile32x512_Pack8x1_Ediv32x1;
                    }
                }
            }
            else
            {
                if(elements <= 7864320)
                {
                    if(static_cast<double>(height) <= static_cast<double>(width) * 6.53333)
                    {
                        return Kernel2B::Tile32x64_Pack2x4_Ediv2x4;
                    }
                    else
                    {
                        return Kernel2B::Tile128x128_Pack8x8_Ediv8x8;
                    }
                }
                else
                {
                    if(std::max(height, width) <= 728)
                    {
                        return Kernel2B::Tile32x512_Pack8x1_Ediv32x1;
                    }
                    else
                    {
                        return Kernel2B::Tile64x128_Pack8x4_Ediv8x4;
                    }
                }
            }
        }
    }
}

static inline Kernel4B SelectKernel4B(uint32_t batch, uint32_t height, uint32_t width)
{
    const uint64_t elements = static_cast<uint64_t>(batch) * height * width;
    if(height <= 36)
    {
        return Kernel4B::Tile64x32_Pack4x4_Ediv4x4;
    }
    else
    {
        if(height <= 114)
        {
            if(elements <= 9634681)
            {
                if(static_cast<double>(height) <= static_cast<double>(width) * 0.19335)
                {
                    return Kernel4B::Tile64x32_Pack4x4_Ediv4x4;
                }
                else
                {
                    return Kernel4B::Tile16x32_Pack1x4_Ediv1x4;
                }
            }
            else
            {
                return Kernel4B::Tile64x64_Pack4x4_Ediv4x4;
            }
        }
        else
        {
            if(width <= 42)
            {
                if(elements <= 3582217)
                {
                    return Kernel4B::Tile16x32_Pack1x4_Ediv1x4;
                }
                else
                {
                    return Kernel4B::Tile32x256_Pack4x2_Ediv4x2;
                }
            }
            else
            {
                if(width % 4 != 0)
                {
                    return Kernel4B::Tile16x32_Pack1x4_Ediv1x4;
                }
                else
                {
                    return Kernel4B::Tile64x128_Pack4x4_Ediv4x4;
                }
            }
        }
    }
}

// buffer argument: the kernel uses buffer instructions (32-bit byte offsets within one batch)
static inline bool
KernelParam2B(Kernel2B kind, uint32_t height, BatchedTransposeParam& param, bool& buffer)
{
    (void)height;
    switch(kind)
    {
    case Kernel2B::Generic: return false;
    case Kernel2B::Tile128x128_Pack8x8_Ediv8x8:
        param  = BatchedTransposeParam{128, 128, 8, 8, 8, 8};
        buffer = true;
        return true;
    case Kernel2B::Tile128x64_Pack8x8_Ediv8x8:
        param  = BatchedTransposeParam{128, 64, 8, 8, 8, 8};
        buffer = true;
        return true;
    case Kernel2B::Tile16x32_Pack1x1_Ediv1x1:
        param  = BatchedTransposeParam{16, 32, 1, 1, 1, 1};
        buffer = false;
        return true;
    case Kernel2B::Tile256x32_Pack1x8_Ediv1x32:
        param  = BatchedTransposeParam{256, 32, 1, 8, 1, 32};
        buffer = false;
        return true;
    case Kernel2B::Tile32x32_Pack2x2_Ediv2x1:
        param  = BatchedTransposeParam{32, 32, 2, 2, 2, 1};
        buffer = false;
        return true;
    case Kernel2B::Tile32x512_Pack8x1_Ediv32x1:
        param  = BatchedTransposeParam{32, 512, 8, 1, 32, 1};
        buffer = false;
        return true;
    case Kernel2B::Tile32x64_Pack1x8_Ediv1x8:
        param  = BatchedTransposeParam{32, 64, 1, 8, 1, 8};
        buffer = false;
        return true;
    case Kernel2B::Tile32x64_Pack2x4_Ediv2x4:
        param  = BatchedTransposeParam{32, 64, 2, 4, 2, 4};
        buffer = false;
        return true;
    case Kernel2B::Tile4x128_Pack1x1_Ediv1x1:
        param  = BatchedTransposeParam{4, 128, 1, 1, 1, 1};
        buffer = false;
        return true;
    case Kernel2B::Tile4x256_Pack1x1_Ediv1x1:
        param  = BatchedTransposeParam{4, 256, 1, 1, 1, 1};
        buffer = false;
        return true;
    case Kernel2B::Tile512x64_Pack1x8_Ediv1x64:
        param  = BatchedTransposeParam{512, 64, 1, 8, 1, 64};
        buffer = false;
        return true;
    case Kernel2B::Tile64x128_Pack8x4_Ediv8x4:
        param  = BatchedTransposeParam{64, 128, 8, 4, 8, 4};
        buffer = true;
        return true;
    case Kernel2B::Tile64x64_Pack4x4_Ediv4x4:
        param  = BatchedTransposeParam{64, 64, 4, 4, 4, 4};
        buffer = false;
        return true;
    }
    return false;
}

static inline bool KernelParam4B(Kernel4B kind, BatchedTransposeParam& param, bool& buffer)
{
    switch(kind)
    {
    case Kernel4B::Generic: return false;
    case Kernel4B::Tile16x32_Pack1x4_Ediv1x4:
        param  = BatchedTransposeParam{16, 32, 1, 4, 1, 4};
        buffer = true;
        return true;
    case Kernel4B::Tile32x256_Pack4x2_Ediv4x2:
        param  = BatchedTransposeParam{32, 256, 4, 2, 4, 2};
        buffer = true;
        return true;
    case Kernel4B::Tile64x128_Pack4x4_Ediv4x4:
        param  = BatchedTransposeParam{64, 128, 4, 4, 4, 4};
        buffer = false;
        return true;
    case Kernel4B::Tile64x32_Pack4x4_Ediv4x4:
        param  = BatchedTransposeParam{64, 32, 4, 4, 4, 4};
        buffer = true;
        return true;
    case Kernel4B::Tile64x64_Pack4x4_Ediv4x4:
        param  = BatchedTransposeParam{64, 64, 4, 4, 4, 4};
        buffer = false;
        return true;
    }
    return false;
}

// Fall back to the generic heuristic where the fitted kernel is slower. The fitted rules above are
// not always better than the generic heuristic: for small and medium tensors (a few microseconds)
// and for some aspect ratios the generic kernels are faster, up to 1.6x for 4-byte data. The
// functions below return true for the sizes where the kernel chosen by the fitted rule was measured
// more than 10% slower than the generic heuristic.

static inline bool UseGeneric4B(uint32_t batch, uint32_t height, uint32_t width)
{
    const uint64_t elements  = static_cast<uint64_t>(batch) * height * width;
    const uint64_t per_batch = static_cast<uint64_t>(height) * width;
    // small and medium tensors that are not close to square
    return elements <= 7001088 && per_batch <= 3244032 &&
           static_cast<double>(std::max(height, width)) >
               static_cast<double>(std::min(height, width)) * 1.23526;
}

static inline bool UseGeneric2B(uint32_t batch, uint32_t height, uint32_t width)
{
    // number of blocks with 32x32 tiles
    const uint64_t blk_s =
        static_cast<uint64_t>(batch) * ((height + 31) / 32) * ((width + 31) / 32);
    if(blk_s <= 14568)
    {
        const uint32_t min_hw = std::min(height, width);
        if(height % 32 != 0)
            return min_hw > 486;
        return static_cast<double>(std::max(height, width)) <=
               static_cast<double>(min_hw) * 655.805;
    }
    return batch <= 3 && static_cast<uint64_t>(height) * width <= 11239424;
}

static inline bool TryGetFittedKernel(const ExecutionContext& ctx,
                                      std::size_t data_size,
                                      uint32_t batch,
                                      uint32_t height,
                                      uint32_t width,
                                      BatchedTransposeParam& out)
{
    if(data_size != 2 && data_size != 4)
        return false;
    if(!IsGfx1250(ctx))
        return false;
    // Small tensors: the large tiles of the fitted kernels give too few blocks to fill the GPU and
    // the generic kernels are faster.
    constexpr uint64_t min_elements_2b = uint64_t{1} << 19;
    constexpr uint64_t min_elements_4b = uint64_t{1} << 21;
    if(static_cast<uint64_t>(batch) * height * width <
       (data_size == 2 ? min_elements_2b : min_elements_4b))
        return false;

    BatchedTransposeParam param;
    bool buffer = false;
    if(data_size == 2)
    {
        if(!KernelParam2B(SelectKernel2B(batch, height, width), height, param, buffer))
            return false;
        if(UseGeneric2B(batch, height, width))
            return false;
    }
    else
    {
        if(!KernelParam4B(SelectKernel4B(batch, height, width), param, buffer))
            return false;
        if(UseGeneric4B(batch, height, width))
            return false;
    }
    if(!IsApplicable(batch, height, width, &param))
        return false;
    // Buffer instructions use 32-bit byte offsets below 2^31 within one batch (and the out-of-range
    // offset 0x7fffffff has to lie outside of the buffer).
    if(buffer && static_cast<uint64_t>(height) * width * data_size >= (uint64_t{1} << 31))
        return false;
    out = param;
    return true;
}

static inline BatchedTransposeParam HeuristicGet(const ExecutionContext& ctx,
                                                 std::size_t data_size,
                                                 uint32_t batch,
                                                 uint32_t height,
                                                 uint32_t width)
{
    /*
     * Iterate from big tile size to small tile size, and try match ediv first
     * If every kernel is applicable, then will pick up the bigest one
     * If need extra padding in h/w (due to tile size), then will pick up kernel that waste the
     * samllest.
     */

    // The gfx1250 selection rules (fitted and thin kernels) were derived from measurements of
    // tensors of at most 2^32 bytes; larger tensors use the generic heuristic below.
    if(static_cast<uint64_t>(batch) * height * width * data_size <=
       std::numeric_limits<uint32_t>::max())
    {
        {
            BatchedTransposeParam fitted;
            if(TryGetFittedKernel(ctx, data_size, batch, height, width, fitted))
                return fitted;
        }

        {
            BatchedTransposeParam thin;
            if(TryGetThinKernel(ctx, data_size, batch, height, width, thin))
                return thin;
        }
    }

    const auto& kernel_list = GetKernelList(ctx, data_size);
    BatchedTransposeParam best_kernel;
    std::size_t extra_padding_size = std::numeric_limits<std::size_t>::max();
    float hw_radio                 = GetNormalizedRadio(height, width);

    if(hw_radio >= 12 && (height <= 8 || width <= 8))
    {
        // Early heuristic for cases that has very large width, very small height (or vice versa)
        if(hw_radio <= 48)
        {
            best_kernel = (width <= 8) ? BatchedTransposeParam{4, 64, 1, 1, 1, 1}
                                       : BatchedTransposeParam{64, 4, 1, 1, 1, 1};
        }
        else if(hw_radio <= 128)
        {
            best_kernel = (width <= 8) ? BatchedTransposeParam{4, 128, 1, 1, 1, 1}
                                       : BatchedTransposeParam{128, 4, 1, 1, 1, 1};
        }
        else
        {
            best_kernel = (width <= 8) ? BatchedTransposeParam{4, 256, 1, 1, 1, 1}
                                       : BatchedTransposeParam{256, 4, 1, 1, 1, 1};
        }
        return best_kernel;
    }

    for(const auto& it : std::ranges::reverse_view(kernel_list))
    {
        if(it.tile_x == 4 || it.tile_y == 4)
        {
            // We don't want such kernel to be selected here,
            // they should be used in above cases
            continue;
        }
        if(!IsApplicable(batch, height, width, &it))
            continue;
        std::size_t current_padding_size = GetExtraPaddingSize(batch, height, width, &it);
        bool replace_current             = false;
        if(best_kernel.tile_x == 0 && best_kernel.tile_y == 0)
        {
            // 1st applicable case
            replace_current = true;
        }
        if(hw_radio > 128)
        {
            // This is for cases that h, w have a great difference
            if(!IsSameSide(height, width, &it))
                continue;
            float prev_radio = GetNormalizedRadio(
                GetNormalizedRadio(best_kernel.tile_y, best_kernel.tile_x), hw_radio);
            float curr_radio =
                GetNormalizedRadio(GetNormalizedRadio(it.tile_y, it.tile_x), hw_radio);

            if(curr_radio * current_padding_size < prev_radio * extra_padding_size)
            {
                if(curr_radio <= prev_radio)
                {
                    replace_current = true;
                }
            }
            else if(float_equal(curr_radio * current_padding_size, prev_radio * extra_padding_size))
            {
                // If width == height, a greate chance is that the kernel performance would be
                // almost the same, so ignore this case
                if((width > height && it.tile_x > it.tile_y &&
                    best_kernel.tile_x < best_kernel.tile_y) ||
                   (width < height && it.tile_x < it.tile_y &&
                    best_kernel.tile_x > best_kernel.tile_y))
                {
                    replace_current = true;
                }
            }
        }
        else
        {
            if(current_padding_size < extra_padding_size)
            {
                replace_current = true;
            }
        }

        if(replace_current)
        {
            extra_padding_size = current_padding_size;
            best_kernel        = it;
        }
    }

    assert(extra_padding_size != std::numeric_limits<std::size_t>::max()); // Impossible
    return best_kernel;
}

} // namespace batched_transpose

BatchedTransposeSolution::BatchedTransposeSolution(const ExecutionContext& ctx,
                                                   miopenDataType_t data_type_,
                                                   uint32_t batch_,
                                                   uint32_t height_,
                                                   uint32_t width_)
    : data_type(data_type_), batch(batch_), height(height_), width(width_)
{
    if(!IsApplicable(data_type))
        MIOPEN_THROW("This data type is not supported");
    num_cu                 = ctx.GetStream().GetMaxComputeUnits();
    std::size_t data_size  = miopen::GetTypeSize(data_type);
    kernel_param_heuristic = batched_transpose::HeuristicGet(ctx, data_size, batch, height, width);
}

solver::KernelInfo BatchedTransposeSolution::GetKernelInfo() const
{
    std::size_t block_size = BATCHED_TRANSPOSE_BLOCK_SIZE;
#if BATCHED_TRANSPOSE_PERSISTENT
    std::size_t grid_size = num_cu * BATCHED_TRANSPOSE_OCCUPANCY;
#else
    uint32_t dim_h = (height + kernel_param_heuristic.tile_y - 1) / kernel_param_heuristic.tile_y;
    uint32_t dim_w = (width + kernel_param_heuristic.tile_x - 1) / kernel_param_heuristic.tile_x;
    std::size_t grid_size = static_cast<std::size_t>(batch) * dim_h * dim_w;
#endif
    std::string kernel_name = GetKernelName();
    solver::KernelInfo kernel;
    kernel.kernel_file = "batched_transpose.cpp";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    kernel.g_wk.push_back(grid_size * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    MIOPEN_LOG_T(kernel_name);

    return kernel;
}

std::vector<OpKernelArg> BatchedTransposeSolution::GetKernelArg() const
{
    uint32_t dim_h = (height + kernel_param_heuristic.tile_y - 1) / kernel_param_heuristic.tile_y;
    uint32_t dim_w = (width + kernel_param_heuristic.tile_x - 1) / kernel_param_heuristic.tile_x;
    uint32_t dim_total = batch * dim_h * dim_w;
#if BATCHED_TRANSPOSE_PERSISTENT
    std::size_t grid_size = static_cast<std::size_t>(num_cu) * BATCHED_TRANSPOSE_OCCUPANCY;
#else
    std::size_t grid_size = static_cast<std::size_t>(batch) * dim_h * dim_w;
#endif

    magic_div_u32_t magic_h = magic_div_u32_gen(dim_h);
    magic_div_u32_t magic_w = magic_div_u32_gen(dim_w);

    std::vector<OpKernelArg> opArgs;
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(0); // placeholder
    opArgs.emplace_back(height);
    opArgs.emplace_back(width);
    if(grid_size != static_cast<uint32_t>(grid_size))
        MIOPEN_THROW("Variable grid size can't be casted to uint32_t safely");
    opArgs.emplace_back(static_cast<uint32_t>(grid_size));
    opArgs.emplace_back(dim_total);
    opArgs.emplace_back(magic_h.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_h.shift));
    opArgs.emplace_back(magic_w.magic);
    opArgs.emplace_back(static_cast<uint32_t>(magic_w.shift));

    return opArgs;
}

std::string BatchedTransposeSolution::GetKernelName() const
{
    std::size_t data_size = miopen::GetTypeSize(data_type);
    return batched_transpose::GetKernelName(data_size, &kernel_param_heuristic);
}

bool BatchedTransposeSolution::IsSkippable() const
{
    // If height or width is 1, actually no need to do transpose.
    // But nonthing prevent you from DO transpose...
    return height == 1 || width == 1;
}

size_t BatchedTransposeSolution::GetOutputTensorSize() const
{
    return miopen::GetTypeSize(data_type) * batch * height * width;
}

InvokerFactory BatchedTransposeSolution::MakeBatchedTransposeInvokerFactory() const
{
    std::vector<OpKernelArg> opArgs = GetKernelArg();

    miopen::InvokerFactory invoker_factory([=](const std::vector<miopen::Kernel>& kernels) mutable {
        return [=](const miopen::Handle& _handle,
                   const miopen::AnyInvokeParams& primitive_param) mutable {
            decltype(auto) invoke_params = primitive_param.CastTo<transpose_invoke_param>();

            const auto k = _handle.Run(kernels[0]);

            opArgs[0] = OpKernelArg(invoke_params.dst);
            opArgs[1] = OpKernelArg(invoke_params.src);

            k(opArgs);
        };
    });

    return invoker_factory;
}

} // namespace miopen
