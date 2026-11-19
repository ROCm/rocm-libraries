// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engines/kernel_ingestor_engine/descriptors/conv/narrow/kernels/ConvNarrowArgs.h"

/**
 * @file ConvNarrowPlan.hpp
 * @brief The config half of CONV_NARROW_FWD / CONV_NARROW_BWD (ConvNarrow.cpp), free of the
 *        graph types: what a catalog config declares, and how it runs a problem given in the
 *        kernels' roles (the strip and band sizes, the prepared filter, the launch).
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_narrow
{

/// The narrow channels a config serves at most, and its filter extents at most.
inline constexpr int MAX_KO = 8;
inline constexpr int MAX_TAPS = 7;
/// LDS a block takes at most on @p gcnArchName: 160 KiB on gfx950, 64 KiB elsewhere.
int64_t ldsLimitOf(std::string_view gcnArchName);

/// A convolution in the kernels' roles (ConvNarrowArgs.h): `in` (n, ci, in extents) reduced into
/// `out` (n, ko, out extents). Strides, paddings and dilations are the graph's (forward roles).
struct NarrowShape
{
    bool backward = false;
    int n = 0;
    int ci = 0;
    int ko = 0;
    std::array<int, 3> in{}; ///< d, h, w (leading 1s below 3-D)
    std::array<int, 3> out{};
    std::array<int, 3> filter{}; ///< t, r, s
    std::array<int, 3> pad{}; ///< pre-padding
    std::array<int, 3> stride{};
    std::array<int, 3> dilation{};
    int flip = 0;
    int elementBytes = 0;
    std::array<long long, 5> is{}; ///< in element strides: n, c, d, h, w
    std::array<long long, 5> os{};
    std::array<long long, 5> ws{}; ///< the graph filter's: k, c, t, r, s
    int wGraphK = 0;
    int wGraphC = 0;
    int inAlign = 0; ///< byte alignment of in's pointer, at most 16
};

/// A catalog config: direction (`FWD`, `BWD`), layout (`nchw`, `nhwc`), atom (a K1 matrix-core
/// atom, or `valu`), the narrow channels it serves (kmin to ko), the filter (fd x fr x fs) and width
/// stride (sw), depth streaming (ds: 1 for unit depth stride and dilation), pixels a product row
/// (wr), in rows a step (rb), waves a block (nw), output pixels a thread (op), the reduction
/// channels it holds (cmin to cm, padded to whole k-steps), the steps its A loads run ahead (pd)
/// and whether the filter columns fold into the reduction (sf: unit width stride; matrix cores
/// only, all 0 for valu).
struct Config
{
    std::string direction;
    std::string layout;
    std::string atom;
    int kmin = 0;
    int ko = 0;
    int fd = 0;
    int fr = 0;
    int fs = 0;
    int sw = 0;
    int ds = 0;
    int wr = 0;
    int rb = 0;
    int nw = 0;
    int op = 0;
    int cmin = 0;
    int cm = 0;
    int pd = 0;
    int sf = 0;
};

/// The reduction step (k per instruction) and tile size of a matrix-core atom the kernels take,
/// or nullopt; `valu`: step 1, size 1.
struct AtomShape
{
    int ka = 0;
    int kpl = 0;
    int dim = 0;
    int wave = 0; ///< 0 for valu (the device's)
    int es = 0; ///< element bytes (0 for valu: any)

    /// k-steps a lane's 16-byte run of channels covers (ConvNarrow.cpp's U).
    int run() const
    {
        return es == 0 || kpl * es >= 16 ? 1 : 16 / (kpl * es);
    }
};
std::optional<AtomShape> atomShapeOf(const std::string& atom);

/// Validates a config's own fields (not against a problem): nullopt when malformed.
std::optional<Config> checkConfig(Config config);

/// How a config runs a problem on a device.
struct Plan
{
    Config config;
    ConvNarrowArgs args{}; ///< every field but the pointers
    int wave = 0; ///< lanes a wave
    int threads = 0; ///< a block's
    bool vectorIn = false; ///< CN_AV
    bool bRegisters = false; ///< CN_BR: the B fragments fit registers
    int ncp = 0; ///< product columns (padded)
    int ldsBytes = 0;
    long long prepareGrid = 0;
    size_t workspaceBytes = 0; ///< the prepared filter
};

/// The plan @p config runs @p shape with on a device of @p wave lanes, @p computeUnits dispatch
/// units (0: unknown) and @p ldsLimit bytes of LDS a block, or nullopt when it does not serve it.
std::optional<Plan> planFor(const NarrowShape& shape,
                            const Config& config,
                            int wave,
                            int64_t computeUnits,
                            int64_t ldsLimit);

/// The HIPRTC defines of @p plan (CN_*), for element type @p elementType (`float`, `_Float16`,
/// `unsigned short`) and the atom's K1 type name @p kernelType (empty for valu).
std::vector<std::pair<std::string, std::string>> compileDefinesFor(const Plan& plan,
                                                                   const std::string& elementType,
                                                                   const std::string& kernelType);

/// A backward-weights convolution in the kernel's roles (ConvNarrowWrwArgs): `in` (n, ci, in
/// extents) the wide operand, `nar` (n, ko, nar extents) the narrow one; strides, paddings and
/// dilations the graph's. wideIsC: an output layer (in = x, nar = dy); else a stem (in = dy,
/// nar = x).
struct WrwShape
{
    bool wideIsC = false;
    int n = 0;
    int ci = 0;
    int ko = 0;
    std::array<int, 3> in{};
    std::array<int, 3> nar{};
    std::array<int, 3> filter{};
    std::array<int, 3> pad{};
    std::array<int, 3> stride{};
    std::array<int, 3> dilation{};
    int flip = 0;
    int elementBytes = 0;
    std::array<long long, 5> is{};
    std::array<long long, 5> ns{};
    std::array<long long, 5> ws{}; ///< dw's: k, c, t, r, s
    int inAlign = 0;
};

/// A backward-weights catalog config: side (`k`: an output layer; `c`: a stem), layout, atom (or
/// `valu`), the narrow channels it serves (kmin to ko), the filter (fd x fr x fs), in columns a step (wr), in rows a step (rb),
/// waves a block (nw) and row tiles a wave (mpw; valu: column groups).
struct WrwConfig
{
    std::string side;
    std::string layout;
    std::string atom;
    int kmin = 0;
    int ko = 0;
    int fd = 0;
    int fr = 0;
    int fs = 0;
    int wr = 0;
    int rb = 0;
    int nw = 0;
    int mpw = 0;
};

/// Validates a backward-weights config's own fields: nullopt when malformed.
std::optional<WrwConfig> checkWrwConfig(WrwConfig config);

/// How a backward-weights config runs a problem.
struct WrwPlan
{
    WrwConfig config;
    ConvNarrowWrwArgs args{}; ///< every field but the pointers
    int wave = 0;
    int threads = 0;
    bool vectorIn = false; ///< CW_AV
    int ncp = 0;
    int xw = 0; ///< narrow pixels a window row (CW_XW)
    int ldsBytes = 0;
    long long reduceGrid = 0;
    size_t workspaceBytes = 0; ///< the partial sums
};

/// The plan @p config runs @p shape with (as planFor), or nullopt.
std::optional<WrwPlan> planWrw(const WrwShape& shape,
                               const WrwConfig& config,
                               int wave,
                               int64_t computeUnits,
                               int64_t ldsLimit);

/// The HIPRTC defines of @p plan (CW_*).
std::vector<std::pair<std::string, std::string>> wrwDefinesFor(const WrwPlan& plan,
                                                               const std::string& elementType,
                                                               const std::string& kernelType);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_narrow

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
