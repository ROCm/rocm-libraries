// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvIgemmNchwProblem.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::conv_igemm_nchw
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using conv_naive::ConvDirection;

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
constexpr int64_t MAX_GRID_YZ = 65535;
constexpr int64_t MAX_BLOCK = 1024;
constexpr int64_t LDS_LIMIT = 65536;
/// The FMA kernel's main loop is specialized per pad mod stride up to this stride.
constexpr int64_t MAX_STRIDE = 8;

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

/// True when @p strides (n, c, d, h, w) are packed channels-first for @p extents; extent-1 axes
/// may carry any stride.
bool channelsFirst(const long long (&strides)[CONV_NAIVE_TENSOR_DIMS],
                   const std::array<long long, CONV_NAIVE_TENSOR_DIMS>& extents)
{
    long long expected = 1;
    for(size_t axis = CONV_NAIVE_TENSOR_DIMS; axis-- > 0;)
    {
        if(extents[axis] != 1 && strides[axis] != expected)
        {
            return false;
        }
        expected *= extents[axis];
    }
    return true;
}

/// A matrix-core atom's shape: the fragment's edge and the reduction one step takes.
struct AtomShape
{
    int64_t dim = 0;
    int64_t ka = 0;
};

AtomShape shapeOf(std::string_view atom)
{
    const bool f32 = atom.size() >= 3 && atom.substr(atom.size() - 3) == "f32";
    if(atom.substr(0, 6) == "mfma16")
    {
        return {16, f32 ? 4 : 16};
    }
    if(atom.substr(0, 6) == "mfma32")
    {
        return {32, f32 ? 2 : 8};
    }
    if(atom.substr(0, 4) == "wmma")
    {
        return {16, 16};
    }
    return {};
}

bool fitsInt(int64_t value)
{
    return value >= 0 && value <= INT_LIMIT;
}

} // namespace

std::optional<NchwProblem>
    parseNchwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    auto base = conv_naive::parseConvNaiveProblem(context, ConvDirection::BACKWARD_DATA);
    if(!base.has_value())
    {
        return std::nullopt;
    }
    int elementBytes = 0;
    switch(base->dataType)
    {
    case data_objects::DataType::FLOAT:
        elementBytes = 4;
        break;
    case data_objects::DataType::HALF:
    case data_objects::DataType::BFLOAT16:
        elementBytes = 2;
        break;
    default:
        return std::nullopt;
    }
    const auto& a = base->args;
    if(a.groups != 1 || a.flipFilter != 0)
    {
        return std::nullopt;
    }
    // Depth and height: any stride, padding and filter, a dilation only at stride 1 (a dx row
    // reduces over each (dy row, filter row) pair that reaches it).
    for(int axis = 0; axis < 2; ++axis)
    {
        if(a.dilation[axis] < 1 || (a.dilation[axis] != 1 && a.stride[axis] != 1)
           || a.stride[axis] < 1 || a.prePadding[axis] < 0 || a.filter[axis] < 1
           || a.inSpatial[axis] < 1 || a.outSpatial[axis] < 1)
        {
            return std::nullopt;
        }
    }
    // Width: the convolution; dilation only at stride 1 (the taps then shift by the dilation).
    if(a.dilation[2] < 1 || (a.dilation[2] != 1 && a.stride[2] != 1) || a.stride[2] < 1
       || a.prePadding[2] < 0 || a.filter[2] < 1)
    {
        return std::nullopt;
    }
    const long long d = a.inSpatial[0], h = a.inSpatial[1];
    const long long dyD = a.outSpatial[0], dyH = a.outSpatial[1];
    if(!channelsFirst(a.xStrides, {a.n, a.c, d, h, a.inSpatial[2]})
       || !channelsFirst(a.yStrides, {a.n, a.k, dyD, dyH, a.outSpatial[2]})
       || !channelsFirst(a.wStrides, {a.k, a.c, a.filter[0], a.filter[1], a.filter[2]}))
    {
        return std::nullopt;
    }
    const long long rows = d * h;
    const long long rowsY = dyD * dyH;
    const long long filterRun = a.filter[0] * a.filter[1] * a.filter[2];
    for(const long long value : {a.n,
                                 a.c,
                                 a.k,
                                 rows,
                                 rowsY,
                                 a.inSpatial[2],
                                 a.outSpatial[2],
                                 a.filter[2],
                                 a.prePadding[2],
                                 a.stride[2],
                                 a.dilation[2],
                                 a.dilation[0],
                                 a.dilation[1],
                                 a.prePadding[0],
                                 a.prePadding[1],
                                 a.stride[0],
                                 a.stride[1],
                                 filterRun})
    {
        if(value < 0 || !fitsInt(value))
        {
            return std::nullopt;
        }
    }
    // dy is addressed by 32-bit byte offsets within an image, the filter as a whole.
    const auto dyImage = pack_features::checkedProduct({a.k, rowsY, a.outSpatial[2]});
    const auto dxImage = pack_features::checkedProduct({a.c, rows, a.inSpatial[2]});
    const auto filterElements = pack_features::checkedProduct({a.k, a.c, filterRun});
    if(!dyImage.has_value() || !dxImage.has_value() || !filterElements.has_value()
       || *dyImage > INT_LIMIT / elementBytes || *filterElements > INT_LIMIT / elementBytes
       || a.n * rows > MAX_GRID_YZ)
    {
        return std::nullopt;
    }

    NchwProblem problem;
    problem.elementBytes = elementBytes;
    problem.stride = a.stride[2];
    // Flat also reads dy as one buffer (32-bit byte offsets) and numbers the outputs of every
    // image's channel plane (n rows Wx) in 32 bits.
    problem.flat = a.filter[2] <= a.stride[2] && a.dilation[2] == 1
                   && a.n * *dyImage <= INT_LIMIT / elementBytes
                   && a.n * rows * a.inSpatial[2] <= INT_LIMIT;
    for(int axis = 0; axis < 3; ++axis)
    {
        problem.flat = problem.flat && a.prePadding[axis] == 0 && a.dilation[axis] == 1
                       && a.filter[axis] <= a.stride[axis]
                       && a.inSpatial[axis] == a.stride[axis] * a.outSpatial[axis];
    }
    problem.dxUid = base->operands.x;
    problem.wUid = base->operands.w;
    problem.dyUid = base->operands.y;
    auto& k7 = problem.args;
    k7.n = static_cast<int>(a.n);
    k7.c = static_cast<int>(a.c);
    k7.k = static_cast<int>(a.k);
    k7.rows = static_cast<int>(rows);
    k7.wx = static_cast<int>(a.inSpatial[2]);
    k7.wy = static_cast<int>(a.outSpatial[2]);
    k7.fs = static_cast<int>(a.filter[2]);
    k7.pad = static_cast<int>(a.prePadding[2]);
    k7.qBase = static_cast<int>(a.prePadding[2] / a.stride[2]);
    k7.rowsY = static_cast<int>(rowsY);
    k7.hx = static_cast<int>(h);
    k7.hy = static_cast<int>(dyH);
    k7.dyDepth = static_cast<int>(dyD);
    k7.sd = static_cast<int>(a.stride[0]);
    k7.pd = static_cast<int>(a.prePadding[0]);
    k7.fd = static_cast<int>(a.filter[0]);
    k7.sh = static_cast<int>(a.stride[1]);
    k7.ph = static_cast<int>(a.prePadding[1]);
    k7.fh = static_cast<int>(a.filter[1]);
    k7.fStride = static_cast<int>(filterRun);
    k7.dil = static_cast<int>(a.dilation[2]);
    k7.dd = static_cast<int>(a.dilation[0]);
    k7.dh = static_cast<int>(a.dilation[1]);
    k7.dyImage = *dyImage;
    k7.dxImage = *dxImage;
    problem.base = std::move(*base);
    return problem;
}

int64_t atomIdOf(std::string_view atom)
{
    const auto it = std::find(ATOMS.begin(), ATOMS.end(), atom);
    return it == ATOMS.end() ? -1 : static_cast<int64_t>(it - ATOMS.begin());
}

int64_t ldsBytesOf(const NchwConfig& config, int elementBytes)
{
    const int64_t pad = 16 / elementBytes;
    const int64_t smax = config.stride * config.taps;
    if(config.atom == "fma")
    {
        const int64_t wq = config.bmq + config.halo;
        const int64_t lda = wq + pad;
        // k7FmaPitch: an odd number of 16-byte chunks.
        const int64_t chunks = divUp(smax, pad);
        const int64_t cp = (chunks % 2 == 0 ? chunks + 1 : chunks) * pad;
        const int64_t ldb = config.bn * cp + pad;
        return config.stages * config.bk * (lda + ldb) * elementBytes;
    }
    const int64_t wq = config.bmq + config.halo;
    // The output tile: dx's element type (fp32 for three-bf16), over the operand tiles.
    const int outBytes = config.math == 1 ? 4 : elementBytes;
    const int64_t out = config.bn * (config.bmq * config.stride + 16 / outBytes) * outBytes;
    if(config.atom.size() >= 3 && config.atom.substr(config.atom.size() - 3) == "f32")
    {
        // One k per lane: k-major tiles as the FMA kernel's (no transposing stores).
        const int64_t chunks = divUp(smax, pad);
        const int64_t cp = (chunks % 2 == 0 ? chunks + 1 : chunks) * pad;
        return std::max(config.bk * ((wq + pad) + config.bn * cp + pad) * elementBytes, out);
    }
    const int64_t ldk = config.bk + pad;
    const int64_t aTile = wq * ldk + (config.vec > 0 ? wq / config.vec : 0) * pad;
    const int64_t bTile = smax * config.bn * ldk;
    return std::max((aTile + bTile) * elementBytes, out);
}

std::optional<NchwProblem> threeBf16ProblemOf(const NchwProblem& problem)
{
    NchwProblem t = problem;
    const auto& a = problem.base.args;
    const int64_t k3 = a.k * 3;
    const int64_t dyImage = problem.args.dyImage * 3;
    const auto filter = pack_features::checkedProduct({k3, a.c, problem.args.fStride});
    if(!fitsInt(k3) || dyImage > INT_LIMIT / 2 || !filter.has_value() || *filter > INT_LIMIT / 2)
    {
        return std::nullopt;
    }
    t.base.dataType = data_objects::DataType::BFLOAT16;
    t.elementBytes = 2;
    t.base.args.k = k3;
    t.args.k = static_cast<int>(k3);
    t.args.dyImage = dyImage;
    t.flat = problem.flat && a.n * dyImage <= INT_LIMIT / 2;
    return t;
}

bool threeBf16Pays(const NchwProblem& problem, std::string_view gcnArchName)
{
    // Timed against the best exact config on 12 top fp32 K7 shapes (PLAN.md): 0.05-0.74x on
    // gfx950, gfx942 and gfx90a; on gfx1201 (exact on FMA) 1.07x and 1.46x for 3-D 3x3x3 at
    // stride 2 and 0.15-0.79x elsewhere. Admitted only there.
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    const auto& a = problem.base.args;
    const bool rdna = arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
    return rdna && problem.base.spatialRank == 3 && a.stride[0] >= 2 && a.stride[1] >= 2
           && a.stride[2] >= 2 && a.filter[0] * a.filter[1] * a.filter[2] >= 27;
}

namespace
{
std::optional<NchwLaunch> launchCore(const NchwProblem& problem, const NchwConfig& config);
} // namespace

std::optional<NchwLaunch> nchwLaunchFor(const NchwProblem& problem, const NchwConfig& config)
{
    if(config.math == 0)
    {
        return launchCore(problem, config);
    }
    if(config.math != 1 || problem.base.dataType != data_objects::DataType::FLOAT
       || !conv_igemm::tf32Permitted() || conv_igemm::atomElement(config.atom) != "bf16")
    {
        return std::nullopt;
    }
    const auto tripled = threeBf16ProblemOf(problem);
    if(!tripled.has_value())
    {
        return std::nullopt;
    }
    auto launch = launchCore(*tripled, config);
    if(!launch.has_value())
    {
        return std::nullopt;
    }
    const auto& a = problem.base.args;
    conv_tf32::SplitTensor dy;
    dy.outer = a.n;
    dy.cg = a.k;
    dy.inner = static_cast<int64_t>(problem.args.rowsY) * problem.args.wy;
    dy.pattern = conv_tf32::PATTERN_A;
    conv_tf32::SplitTensor w;
    w.outer = 1;
    w.cg = a.k;
    w.inner = a.c * static_cast<int64_t>(problem.args.fStride);
    w.pattern = conv_tf32::PATTERN_B;
    launch->split = true;
    launch->plan = conv_tf32::splitPlanFor(dy, w);
    launch->args = tripled->args;
    return launch;
}

namespace
{
std::optional<NchwLaunch> launchCore(const NchwProblem& problem, const NchwConfig& config)
{
    const auto atomId = atomIdOf(config.atom);
    if(atomId < 0 || atomId != config.atomId)
    {
        return std::nullopt;
    }
    const bool fma = config.atom == "fma";
    const int eb = problem.elementBytes;
    const auto& a = problem.args;
    // The vector: a whole number of elements in at most 16 bytes, covering the taps' window
    // reach, dividing the tile's phase indices and the filter run.
    if(config.vec != 1 && config.vec != 2 && config.vec != 4 && config.vec != 8)
    {
        return std::nullopt;
    }
    if(config.vec * eb > 16 || config.bmq < 1 || config.bn < 1 || config.bk < 1
       || config.bmq % config.vec != 0 || (config.bmq * config.stride) % config.vec != 0
       || (config.bn * config.stride * config.taps) % config.vec != 0
       || (config.bn * a.fs) % config.vec != 0)
    {
        return std::nullopt;
    }
    // Stride and taps: the problem's stride (the kernels unroll at most MAX_STRIDE phases), and
    // exactly the taps of the fullest phase.
    if(config.stride != problem.stride || config.stride > MAX_STRIDE || config.taps < 1
       || divUp(a.fs, config.stride) != config.taps)
    {
        return std::nullopt;
    }
    // The window's halo: whole vectors past the tile that reach the farthest tap, (taps - 1)
    // dilation back, and one further for the phases below pad mod stride. The FMA kernel with a
    // one-vector halo unrolls its taps over a register window (unit dilation); a wider halo reads
    // each tap's window from LDS (stride 1).
    const int64_t reach = (config.taps - 1) * a.dil + (config.stride > 1 ? 1 : 0);
    if(config.halo < config.vec || config.halo % config.vec != 0 || reach > config.halo
       || (fma && config.halo == config.vec && (config.taps > config.vec || a.dil != 1))
       || (fma && config.halo > config.vec && config.stride != 1))
    {
        return std::nullopt;
    }
    // Flat: the problem flat; its filter's taps exactly the computed phases, or every phase
    // when the filter has more than one tap (a phase computed for no tap is waste, not a
    // different answer); the vector dividing an image's dy positions (no load straddles
    // images); the classes within the grid.
    const int64_t classes = static_cast<int64_t>(a.sd) * a.sh;
    if(config.flat < 0 || config.flat > config.stride
       || (config.flat > 0
           && (!problem.flat || config.flat < a.fs
               || (config.flat != a.fs && (config.flat != config.stride || a.fs == 1))
               || config.taps != 1 || (static_cast<int64_t>(a.rowsY) * a.wy) % config.vec != 0
               || classes > MAX_GRID_YZ)))
    {
        return std::nullopt;
    }
    const int64_t phases = config.flat > 0 ? config.flat : config.stride;
    NchwLaunch launch;
    int64_t threads = 0;
    if(fma)
    {
        if(config.wm != 0 || config.wn != 0 || config.wp != 0 || config.tq < 1 || config.tc < 1
           || config.bmq % config.tq != 0 || config.bn % config.tc != 0
           || (config.stages != 1 && config.stages != 2))
        {
            return std::nullopt;
        }
        threads = (config.bn / config.tc) * (config.bmq / config.tq);
    }
    else
    {
        const auto info = conv_igemm::atomNamed(config.atom);
        const auto shape = shapeOf(config.atom);
        if(!info.has_value() || info->wave == 0 || shape.dim == 0 || config.tq != 0
           || config.tc != 0 || config.wm < 1 || config.wn < 1 || config.wp < 1
           || phases % config.wp != 0 || config.stages != 1)
        {
            return std::nullopt;
        }
        const int64_t kv = eb == 2 ? 2 : 1;
        if(config.bmq % (config.wm * shape.dim) != 0 || config.bn % (config.wn * shape.dim) != 0
           || config.bk % shape.ka != 0 || config.bk % kv != 0)
        {
            return std::nullopt;
        }
        threads = config.wm * config.wn * config.wp * info->wave;
        launch.kernelType = info->kernelType;
    }
    if(threads < 1 || threads > MAX_BLOCK || ldsBytesOf(config, eb) > LDS_LIMIT)
    {
        return std::nullopt;
    }
    const int64_t span = config.bmq * config.stride;
    const int64_t positions = static_cast<int64_t>(a.n) * a.rowsY * a.wy;
    const int64_t gridX = config.flat > 0 ? divUp(positions, config.bmq) : divUp(a.wx, span);
    const int64_t gridY = divUp(a.c, config.bn);
    const int64_t gridZ = config.flat > 0 ? classes : static_cast<int64_t>(a.n) * a.rows;
    if(gridX < 1 || gridX > INT_LIMIT || gridX * span > INT_LIMIT || gridY > MAX_GRID_YZ
       || gridZ < 1 || gridZ > MAX_GRID_YZ)
    {
        return std::nullopt;
    }
    launch.gridX = static_cast<unsigned int>(gridX);
    launch.gridY = static_cast<unsigned int>(gridY);
    launch.gridZ = static_cast<unsigned int>(gridZ);
    launch.block = static_cast<unsigned int>(threads);
    launch.args = problem.args;
    return launch;
}
} // namespace

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_igemm_nchw

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
