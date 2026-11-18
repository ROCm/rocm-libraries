// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "engines/kernel_ingestor_engine/packs/ConvAotGeometryCommon.hpp"

/**
 * @file Gfx1250ConvFwdTypes.hpp
 * @brief The problem and launch-plan vocabulary of hipkernel:Gfx1250ConvFwdAot's geometry.
 *
 * Header-only and free of hipDNN includes, so the geometry it carries is unit-tested against
 * rocKE's dispatcher (the golden vectors of make_conv_aot_test_vectors.py --arch gfx1250)
 * without a GPU or a loaded plugin. A plan names its buffers -- the operands `x`, `w`, `y`
 * and the workspaces it sizes -- and the native pack (ConvFwdAotNative.cpp) launches it
 * through the shared conv_fwd_aot kernel ABI.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

enum class DataType
{
    FP16,
    BF16,
};

/// The dtype as the KMD's `dtype` field spells it.
inline const char* toString(DataType dtype)
{
    return dtype == DataType::BF16 ? "bf16" : "fp16";
}

/// gfx1250 target limits (arch_specs.json) are the shared conv_aot::Target and caps the
/// dgrad port also uses; the WMMA constants below are this family's.
using conv_aot::I32_MAX;
using conv_aot::MAX_DISPATCH_ITEMS;
using conv_aot::MAX_GRID_DIM;
constexpr int64_t GFX1250_WAVE_SIZE = conv_aot::GFX1250.waveSize;
constexpr int64_t GFX1250_MAX_THREADS_PER_BLOCK = conv_aot::MAX_BLOCK_THREADS;
constexpr int64_t GFX1250_LDS_BYTES = conv_aot::GFX1250.ldsCapacityBytes;
constexpr int64_t WMMA_K = 32;
constexpr int64_t CH_PAD = 8; ///< LDS row skew of a staged tile, in channels (16 bytes).
constexpr int64_t TDM_MAX_EXTENT = 0xFFFF;

/// Python's `//`: rounds toward negative infinity, as rocKE's output extents do (an empty
/// output is 0, not 1, when the numerator is negative).
inline int64_t floorDiv(int64_t a, int64_t b)
{
    const auto q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

/// A 2-D grouped forward convolution: x NHWC (N, H, W, C), w KYXC (K, Y, X, C / groups),
/// y NHWK. Every extent is a positive count; padding is per axis.
struct Problem
{
    int64_t n = 1;
    int64_t c = 1;
    int64_t k = 1;
    int64_t h = 1;
    int64_t w = 1;
    int64_t y = 1;
    int64_t x = 1;
    int64_t groups = 1;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
    DataType dtype = DataType::FP16;

    int64_t cpg() const
    {
        return c / groups;
    }
    int64_t kpg() const
    {
        return k / groups;
    }
    int64_t ho() const
    {
        return floorDiv(h + 2 * padH - dilationH * (y - 1) - 1, strideH) + 1;
    }
    int64_t wo() const
    {
        return floorDiv(w + 2 * padW - dilationW * (x - 1) - 1, strideW) + 1;
    }
};

/// One kernel argument of a launch: an i32 value, a pointer into a named buffer at a byte
/// offset, or the byte size of a named buffer.
struct Arg
{
    enum class Kind
    {
        I32,
        POINTER,
        BYTES,
    };

    std::string name;
    Kind kind = Kind::I32;
    int64_t value = 0; ///< The i32 value, or the pointer's byte offset.
    std::string buffer; ///< The pointer's or size's buffer.

    static Arg i32(std::string name, int64_t value)
    {
        return Arg{std::move(name), Kind::I32, value, {}};
    }
    static Arg pointer(std::string name, std::string buffer, int64_t offset = 0)
    {
        return Arg{std::move(name), Kind::POINTER, offset, std::move(buffer)};
    }
    static Arg bytes(std::string name, std::string buffer)
    {
        return Arg{std::move(name), Kind::BYTES, 0, std::move(buffer)};
    }
};

using Dim3 = std::array<int64_t, 3>;

/// One launch: the binary it runs (an instance label), its grid
/// (in workgroups), block (in threads) and arguments in ABI order.
struct Step
{
    std::string kernel;
    Dim3 grid{1, 1, 1};
    Dim3 block{1, 1, 1};
    std::vector<Arg> args;
};

/// The steps of one kernel on one problem, run in order on one stream, and the scratch
/// buffers they share (name -> bytes, no initial contents).
struct Plan
{
    std::vector<Step> steps;
    std::map<std::string, int64_t> workspaces;
};

/// Why a kernel refuses a problem; empty optional when it admits it.
using Refusal = std::optional<std::string>;

/// rocke.helpers.spec.ceil_div: exact ceil(a / b) for any signs; @p b != 0.
inline int64_t ceilDiv(int64_t a, int64_t b)
{
    return -floorDiv(-a, b);
}

/// rocke.helpers.spec.align_up.
inline int64_t alignUp(int64_t a, int64_t b)
{
    return ceilDiv(a, b) * b;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
