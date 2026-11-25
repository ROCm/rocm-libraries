// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvNarrowPlan.hpp"

#include <algorithm>
#include <limits>

namespace hip_kernel_provider::kernel_ingestor_engine::conv_narrow
{

namespace
{

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
/// Byte offsets within one image are 32-bit (and 2^31 is the out-of-range marker).
constexpr int64_t IMAGE_LIMIT = int64_t{1} << 31;

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t floorDiv(int64_t a, int64_t b)
{
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

int64_t ceilDiv(int64_t a, int64_t b)
{
    return -floorDiv(-a, b);
}

/// Bytes of one image of a tensor (channels and extents with their strides), first element to
/// one past its last, or nullopt past 63 bits.
std::optional<int64_t> imageBytes(int64_t channels,
                                  const std::array<int, 3>& extents,
                                  const std::array<long long, 5>& strides,
                                  int elementBytes)
{
    const std::array<int64_t, 4> dims{channels, extents[0], extents[1], extents[2]};
    int64_t last = 0;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        const int64_t stride = strides.at(axis + 1);
        if(stride < 0 || dims.at(axis) < 1)
        {
            return std::nullopt;
        }
        const int64_t span = dims.at(axis) - 1;
        if(span != 0 && stride > (std::numeric_limits<int64_t>::max() / 4 - last) / span)
        {
            return std::nullopt;
        }
        last += span * stride;
    }
    return (last + 1) * elementBytes;
}

bool pow2(int v)
{
    return v > 0 && (v & (v - 1)) == 0;
}

} // namespace

std::optional<AtomShape> atomShapeOf(const std::string& atom)
{
    struct Entry
    {
        const char* name;
        AtomShape shape;
    };
    static const std::array<Entry, 15> s_atoms{{
        {"valu", {1, 1, 1, 0, 0}},
        {"mfma16_f16", {16, 4, 16, 64, 2}},
        {"mfma16_bf16", {16, 4, 16, 64, 2}},
        {"mfma16_bf16_908", {8, 2, 16, 64, 2}},
        {"mfma16_f32", {4, 1, 16, 64, 4}},
        {"mfma16x32_f16", {32, 8, 16, 64, 2}},
        {"mfma16x32_bf16", {32, 8, 16, 64, 2}},
        {"mfma32_f16", {8, 4, 32, 64, 2}},
        {"mfma32_bf16", {8, 4, 32, 64, 2}},
        {"mfma32_bf16_908", {4, 2, 32, 64, 2}},
        {"mfma32_f32", {2, 1, 32, 64, 4}},
        {"wmma11_f16", {16, 16, 16, 32, 2}},
        {"wmma11_bf16", {16, 16, 16, 32, 2}},
        {"wmma12_f16", {16, 8, 16, 32, 2}},
        {"wmma12_bf16", {16, 8, 16, 32, 2}},
    }};
    for(const auto& entry : s_atoms)
    {
        if(atom == entry.name)
        {
            return entry.shape;
        }
    }
    return std::nullopt;
}

std::optional<Config> checkConfig(Config config)
{
    if((config.direction != "FWD" && config.direction != "BWD")
       || (config.layout != "nchw" && config.layout != "nhwc") || !atomShapeOf(config.atom))
    {
        return std::nullopt;
    }
    if(config.kmin < 1 || config.kmin > config.ko || config.ko > MAX_KO || config.fd < 1
       || config.fd > MAX_TAPS || config.fr < 1 || config.fr > MAX_TAPS || config.fs < 1
       || config.fs > MAX_TAPS || config.sw < 1 || config.sw > 4 || config.wr < 1 || config.wr > 256
       || config.rb < 1 || config.rb > 16 || !pow2(config.nw) || config.nw > 16 || config.op < 1
       || config.op > 8 || (config.ds != 0 && config.ds != 1))
    {
        return std::nullopt;
    }
    const auto atom = atomShapeOf(config.atom);
    const bool valu = atom->wave == 0;
    if(config.sf != 0 && (config.sf != 1 || config.sw != 1))
    {
        return std::nullopt;
    }
    if(valu ? config.cmin != 0 || config.cm != 0 || config.pd != 0 || config.sf != 0
            : (config.cm < atom->ka || config.cm > 1024 || config.cm % (atom->ka * atom->run()) != 0
               || config.pd < 1 || config.pd > 3 || config.cmin < 1 || config.cmin > config.cm))
    {
        return std::nullopt;
    }
    return config;
}

int64_t ldsLimitOf(std::string_view gcnArchName)
{
    return gcnArchName.rfind("gfx950", 0) == 0 ? 160 * 1024 : 64 * 1024;
}

std::optional<Plan> planFor(const NarrowShape& shape,
                            const Config& config,
                            int wave,
                            int64_t computeUnits,
                            int64_t ldsLimit)
{
    const auto atom = atomShapeOf(config.atom);
    if(!atom || (config.direction == "BWD") != shape.backward)
    {
        return std::nullopt;
    }
    const bool valu = atom->wave == 0;
    if((wave != 32 && wave != 64) || (!valu && atom->wave != wave))
    {
        return std::nullopt;
    }
    if(shape.ko < config.kmin || shape.ko > config.ko || shape.ci < 1 || shape.n < 1
       || shape.filter[0] != config.fd || shape.filter[1] != config.fr
       || shape.filter[2] != config.fs || shape.stride[2] != config.sw)
    {
        return std::nullopt;
    }
    const bool nchw = config.layout == "nchw";
    if(nchw ? (shape.is[4] != 1 || shape.os[4] != 1) : (shape.is[1] != 1 || shape.os[1] != 1))
    {
        return std::nullopt;
    }
    const int es = shape.elementBytes;
    if((es != 2 && es != 4) || shape.inAlign < es)
    {
        return std::nullopt;
    }
    // Depth streaming walks in planes in order: unit depth stride and dilation.
    const bool stream = config.ds == 1;
    if(stream && (shape.stride[0] != 1 || shape.dilation[0] != 1))
    {
        return std::nullopt;
    }

    // Products a step: M rows of NCP columns through LDS.
    const int64_t threads = static_cast<int64_t>(config.nw) * wave;
    const int64_t m = static_cast<int64_t>(config.rb) * config.wr;
    const int64_t folded = config.sf == 1 ? config.fs : 1; // filter columns on the reduction
    const int64_t ncol = static_cast<int64_t>(stream ? config.fd : 1) * config.ko * config.fr
                         * (config.fs / folded);
    int64_t ncp = 0;
    int64_t ntl = 0;
    if(valu)
    {
        if(m % wave != 0 || threads % m != 0)
        {
            return std::nullopt;
        }
        const int64_t groups = threads / m;
        ncp = groups * divUp(ncol, groups);
    }
    else
    {
        if(m % (static_cast<int64_t>(atom->dim) * config.nw) != 0)
        {
            return std::nullopt;
        }
        ntl = divUp(ncol, atom->dim);
        ncp = ntl * atom->dim;
    }
    // The B fragments in registers where one column group's fit 64 VGPRs a lane (CN_BR), else
    // every group's filter in LDS.
    const int64_t groups = stream ? 1 : config.fd;
    const bool bRegisters
        = !valu && groups == 1
          && ncp / atom->dim * (folded * config.cm / atom->ka) * std::max(1, atom->kpl * es / 4)
                 <= 64;
    // Two product buffers and their zero run, and (no CN_BR) the filter's columns of cm channels
    // plus a 16-byte pad.
    const int64_t lds
        = (2 * m * (ncp + 1) + static_cast<int64_t>(config.ko) * config.fr * config.fs) * 4
          + (valu || bRegisters ? 0 : groups * ncp * (folded * config.cm + 16 / es) * es);
    if(lds > ldsLimit)
    {
        return std::nullopt;
    }

    // Output columns a strip: every tap of the strip's columns reads within the product row.
    const int sw = config.sw;
    const int64_t pw = shape.pad[2];
    const int64_t dw = shape.dilation[2];
    const int64_t reach = (config.fs - 1) * dw;
    int64_t tw = 0;
    if(folded > 1)
    {
        tw = config.wr; // a product row is an output column
    }
    else if(!shape.backward)
    {
        if(config.wr - 1 - reach < 0)
        {
            return std::nullopt;
        }
        tw = (config.wr - 1 - reach) / sw + 1;
    }
    else
    {
        const int64_t first = ceilDiv(pw - reach, sw);
        for(int64_t t = sw; t <= static_cast<int64_t>(config.wr) * sw + reach + sw; t += sw)
        {
            if(floorDiv(t - 1 + pw, sw) - first + 1 > config.wr)
            {
                break;
            }
            tw = t;
        }
        if(tw == 0)
        {
            return std::nullopt;
        }
    }
    const int64_t pixels = threads * config.op; // output pixels a block
    const int64_t wo = shape.out[2];
    tw = std::min(tw, shape.backward ? divUp(wo, sw) * sw : wo);
    tw = std::min(tw, shape.backward ? pixels / sw * sw : pixels);
    if(tw < 1)
    {
        return std::nullopt;
    }
    const int64_t strips = divUp(wo, tw);
    const int64_t ho = shape.out[1];
    const int64_t dOut = shape.out[0];
    // Rows a band (as many as the block's threads cover) and planes a segment (streaming: every
    // plane, else one): halved, the one whose halo grows less, while the grid has fewer than four
    // blocks a dispatch unit (down to 4 rows and 1 plane).
    const int64_t target = 4 * (computeUnits > 0 ? computeUnits : 64);
    int64_t rs = std::min(pixels / tw, ho);
    int64_t ds = stream ? dOut : 1;
    const int64_t rowHalo = (config.fr - 1) * static_cast<int64_t>(shape.dilation[1]);
    const int64_t planeHalo = stream ? config.fd - 1 : 0;
    const auto blocksOf = [&](int64_t r, int64_t d) {
        return static_cast<int64_t>(shape.n) * divUp(dOut, d) * divUp(ho, r) * strips;
    };
    while(blocksOf(rs, ds) < target && (rs > 4 || ds > 1))
    {
        const int64_t rs2 = divUp(rs, 2);
        const int64_t ds2 = divUp(ds, 2);
        // The read cost a halving adds: halo / extent.
        const double rowCost
            = rs > 4 ? static_cast<double>(rowHalo + rs2) / static_cast<double>(rs2) : 1e9;
        const double planeCost
            = ds > 1 ? static_cast<double>(planeHalo + ds2) / static_cast<double>(ds2) : 1e9;
        if(rowCost <= planeCost)
        {
            rs = rs2;
        }
        else
        {
            ds = ds2;
        }
    }
    const int64_t bands = divUp(ho, rs);
    const int64_t segs = divUp(dOut, ds);
    const int64_t blocks = blocksOf(rs, ds);
    if(rs < 1 || blocks > INT_LIMIT || strips > INT_LIMIT)
    {
        return std::nullopt;
    }

    const auto inBytes = imageBytes(shape.ci, shape.in, shape.is, es);
    const auto outBytes = imageBytes(shape.ko, shape.out, shape.os, es);
    if(!inBytes || !outBytes || *inBytes >= IMAGE_LIMIT || *outBytes >= IMAGE_LIMIT)
    {
        return std::nullopt;
    }

    // Vector loads of a pixel's channel run: channels packed, ci, the strides and the pointer
    // whole runs (matrix cores: U k-steps' runs, ConvNarrow.cpp).
    const int64_t unit = atom->run();
    const int64_t run = valu ? 16 / es : atom->kpl * unit;
    bool vectorIn = !nchw && shape.ci % run == 0 && shape.inAlign >= std::min<int64_t>(run * es, 16)
                    && (valu || run * es >= 4);
    for(const size_t axis : {size_t{0}, size_t{2}, size_t{3}, size_t{4}})
    {
        vectorIn = vectorIn && shape.is.at(axis) % run == 0;
    }

    const int64_t cpad = valu ? shape.ci : divUp(shape.ci, atom->ka * unit) * atom->ka * unit;
    if(!valu && (cpad > config.cm || cpad < config.cmin))
    {
        return std::nullopt;
    }
    const int64_t prepared = groups * cpad * ncp * folded;
    if(prepared > INT_LIMIT)
    {
        return std::nullopt;
    }

    Plan plan;
    plan.config = config;
    plan.wave = wave;
    plan.threads = static_cast<int>(threads);
    plan.vectorIn = vectorIn;
    plan.bRegisters = bRegisters;
    plan.ncp = static_cast<int>(ncp);
    plan.ldsBytes = static_cast<int>(lds);
    plan.prepareGrid = divUp(prepared, 256);
    plan.workspaceBytes = static_cast<size_t>(prepared) * static_cast<size_t>(valu ? 4 : es);
    auto& a = plan.args;
    for(size_t i = 0; i < 5; ++i)
    {
        a.is[i] = shape.is.at(i);
        a.os[i] = shape.os.at(i);
        a.ws[i] = shape.ws.at(i);
    }
    a.inImageBytes = *inBytes;
    a.outImageBytes = *outBytes;
    a.n = shape.n;
    a.ci = shape.ci;
    a.ko = shape.ko;
    a.cpad = static_cast<int>(cpad);
    a.di = shape.in[0];
    a.hi = shape.in[1];
    a.wi = shape.in[2];
    a.dOut = shape.out[0];
    a.ho = shape.out[1];
    a.wo = shape.out[2];
    a.pd = shape.pad[0];
    a.ph = shape.pad[1];
    a.pw = shape.pad[2];
    a.sd = shape.stride[0];
    a.sh = shape.stride[1];
    a.dd = shape.dilation[0];
    a.dh = shape.dilation[1];
    a.dw = shape.dilation[2];
    a.tw = static_cast<int>(tw);
    a.strips = static_cast<int>(strips);
    a.rs = static_cast<int>(rs);
    a.bands = static_cast<int>(bands);
    a.ds = static_cast<int>(ds);
    a.segs = static_cast<int>(segs);
    a.blocks = static_cast<int>(blocks);
    a.wGraphK = shape.wGraphK;
    a.wGraphC = shape.wGraphC;
    a.flip = shape.flip;
    a.prepared = static_cast<int>(prepared);
    return plan;
}

std::vector<std::pair<std::string, std::string>> compileDefinesFor(const Plan& plan,
                                                                   const std::string& elementType,
                                                                   const std::string& kernelType)
{
    const auto& c = plan.config;
    const bool valu = c.atom == "valu";
    std::vector<std::pair<std::string, std::string>> defines{{"CN_T", elementType},
                                                             {"CN_VALU", valu ? "1" : "0"}};
    if(valu)
    {
        defines.emplace_back("CN_WAVE", std::to_string(plan.wave));
    }
    else
    {
        defines.emplace_back("CN_ATOM", kernelType);
        defines.emplace_back("CN_CM", std::to_string(c.cm));
        defines.emplace_back("CN_PD", std::to_string(c.pd));
        defines.emplace_back("CN_SF", std::to_string(c.sf));
        defines.emplace_back("CN_BR", plan.bRegisters ? "1" : "0");
    }
    const std::array<std::pair<const char*, int>, 14> fields{
        {{"CN_KO", c.ko},
         {"CN_DS", c.ds},
         {"CN_FD", c.fd},
         {"CN_FR", c.fr},
         {"CN_FS", c.fs},
         {"CN_SW", c.sw},
         {"CN_BWD", c.direction == "BWD" ? 1 : 0},
         {"CN_NCHW", c.layout == "nchw" ? 1 : 0},
         {"CN_AV", plan.vectorIn ? 1 : 0},
         {"CN_WR", c.wr},
         {"CN_RB", c.rb},
         {"CN_NW", c.nw},
         {"CN_OP", c.op},
         {"CN_NCP", plan.ncp}}};
    for(const auto& [name, value] : fields)
    {
        defines.emplace_back(name, std::to_string(value));
    }
    return defines;
}

std::optional<WrwConfig> checkWrwConfig(WrwConfig config)
{
    const auto atom = atomShapeOf(config.atom);
    if((config.side != "k" && config.side != "c")
       || (config.layout != "nchw" && config.layout != "nhwc") || !atom)
    {
        return std::nullopt;
    }
    if(config.kmin < 1 || config.kmin > config.ko || config.ko > MAX_KO || config.fd < 1
       || config.fd > MAX_TAPS || config.fr < 1 || config.fr > MAX_TAPS || config.fs < 1
       || config.fs > MAX_TAPS || config.wr < 1 || config.wr > 256 || config.rb < 1
       || config.rb > 16 || !pow2(config.nw) || config.nw > 16 || config.mpw < 1 || config.mpw > 16)
    {
        return std::nullopt;
    }
    return config;
}

std::optional<WrwPlan> planWrw(const WrwShape& shape,
                               const WrwConfig& config,
                               int wave,
                               int64_t computeUnits,
                               int64_t ldsLimit)
{
    const auto atom = atomShapeOf(config.atom);
    if(!atom || (config.side == "k") != shape.wideIsC)
    {
        return std::nullopt;
    }
    const bool valu = atom->wave == 0;
    if((wave != 32 && wave != 64) || (!valu && atom->wave != wave))
    {
        return std::nullopt;
    }
    if(shape.ko < config.kmin || shape.ko > config.ko || shape.ci < 1 || shape.n < 1
       || shape.filter[0] != config.fd || shape.filter[1] != config.fr
       || shape.filter[2] != config.fs)
    {
        return std::nullopt;
    }
    const bool nchw = config.layout == "nchw";
    if(nchw ? (shape.is[4] != 1 || shape.ns[4] != 1) : (shape.is[1] != 1 || shape.ns[1] != 1))
    {
        return std::nullopt;
    }
    const int es = shape.elementBytes;
    if((es != 2 && es != 4) || shape.inAlign < es)
    {
        return std::nullopt;
    }
    const int64_t threads = static_cast<int64_t>(config.nw) * wave;
    const int64_t kr = static_cast<int64_t>(config.rb) * config.wr;
    const int64_t ncol = static_cast<int64_t>(config.ko) * config.fd * config.fr * config.fs;
    int64_t ncp = 0;
    int64_t cpb = 0; // wide channels a block
    int64_t bElem = 0; // bytes of a B element in LDS
    if(valu)
    {
        const int64_t groups = config.mpw;
        if(config.nw % groups != 0)
        {
            return std::nullopt;
        }
        ncp = groups * divUp(ncol, groups);
        cpb = threads / groups;
        bElem = 4;
        if(ncp / groups > 64) // a thread's sums in registers
        {
            return std::nullopt;
        }
    }
    else
    {
        if(kr % atom->ka != 0 || config.wr % atom->kpl != 0)
        {
            return std::nullopt;
        }
        ncp = divUp(ncol, atom->dim) * atom->dim;
        cpb = static_cast<int64_t>(config.mpw) * config.nw * atom->dim;
        bElem = es;
    }
    // The narrow window a step's taps reach: a row of it covers the step's columns' filter
    // columns.
    const int64_t reachW = static_cast<int64_t>(config.fs - 1) * shape.dilation[2];
    const int64_t sw = shape.stride[2];
    const int64_t xw
        = shape.wideIsC ? (config.wr - 1 + reachW) / sw + 2 : (config.wr - 1) * sw + reachW + 1;
    const int64_t windowElems
        = static_cast<int64_t>(config.ko) * config.fd * config.fr * config.rb * xw;
    const int64_t lds = ncp * (kr + 16 / bElem) * bElem + windowElems * es;
    if(lds > ldsLimit)
    {
        return std::nullopt;
    }
    const int64_t strips = divUp(shape.in[2], config.wr);
    const int64_t bands = divUp(shape.in[1], config.rb);
    const int64_t steps = static_cast<int64_t>(shape.n) * shape.in[0] * bands * strips;
    const int64_t cblocks = divUp(shape.ci, cpb);
    if(steps > INT_LIMIT)
    {
        return std::nullopt;
    }
    // Splits: about two blocks a dispatch unit, each at least 4 steps, the partials within
    // 256 MiB.
    const int64_t units = computeUnits > 0 ? computeUnits : 64;
    int64_t splits
        = std::clamp<int64_t>(divUp(2 * units, cblocks), 1, std::max<int64_t>(1, steps / 4));
    const int64_t partBytes = static_cast<int64_t>(shape.ci) * ncp * 4;
    splits = std::max<int64_t>(1, std::min(splits, (int64_t{256} << 20) / partBytes));
    if(splits * cblocks > INT_LIMIT)
    {
        return std::nullopt;
    }
    const auto inBytes = imageBytes(shape.ci, shape.in, shape.is, es);
    const auto narBytes = imageBytes(shape.ko, shape.nar, shape.ns, es);
    if(!inBytes || !narBytes || *inBytes >= IMAGE_LIMIT || *narBytes >= IMAGE_LIMIT)
    {
        return std::nullopt;
    }
    const int64_t taps = static_cast<int64_t>(config.fd) * config.fr * config.fs;
    const int64_t dwElems = static_cast<int64_t>(shape.ci) * shape.ko * taps;
    if(dwElems > INT_LIMIT)
    {
        return std::nullopt;
    }
    // Vector loads of a lane's pixel run: channels-first rows of whole runs.
    bool vectorIn = !valu && nchw && atom->kpl * es >= 4 && shape.in[2] % atom->kpl == 0
                    && shape.inAlign >= std::min<int64_t>(static_cast<int64_t>(atom->kpl) * es, 16);
    for(const size_t axis : {size_t{0}, size_t{1}, size_t{2}, size_t{3}})
    {
        vectorIn = vectorIn && shape.is.at(axis) % atom->kpl == 0;
    }

    WrwPlan plan;
    plan.config = config;
    plan.wave = wave;
    plan.threads = static_cast<int>(threads);
    plan.vectorIn = vectorIn;
    plan.ncp = static_cast<int>(ncp);
    plan.xw = static_cast<int>(xw);
    plan.ldsBytes = static_cast<int>(lds);
    plan.reduceGrid = divUp(dwElems, 256);
    plan.workspaceBytes = static_cast<size_t>(splits * partBytes);
    auto& a = plan.args;
    for(size_t i = 0; i < 5; ++i)
    {
        a.is[i] = shape.is.at(i);
        a.ns[i] = shape.ns.at(i);
        a.ws[i] = shape.ws.at(i);
    }
    a.inImageBytes = *inBytes;
    a.narImageBytes = *narBytes;
    a.n = shape.n;
    a.ci = shape.ci;
    a.ko = shape.ko;
    a.di = shape.in[0];
    a.hi = shape.in[1];
    a.wi = shape.in[2];
    a.dn = shape.nar[0];
    a.hn = shape.nar[1];
    a.wn = shape.nar[2];
    a.pd = shape.pad[0];
    a.ph = shape.pad[1];
    a.pw = shape.pad[2];
    a.sd = shape.stride[0];
    a.sh = shape.stride[1];
    a.sw = shape.stride[2];
    a.dd = shape.dilation[0];
    a.dh = shape.dilation[1];
    a.dw2 = shape.dilation[2];
    a.strips = static_cast<int>(strips);
    a.bands = static_cast<int>(bands);
    a.steps = static_cast<int>(steps);
    a.splits = static_cast<int>(splits);
    a.cblocks = static_cast<int>(cblocks);
    a.flip = shape.flip;
    a.wideIsC = shape.wideIsC ? 1 : 0;
    a.dwElems = static_cast<int>(dwElems);
    return plan;
}

std::vector<std::pair<std::string, std::string>> wrwDefinesFor(const WrwPlan& plan,
                                                               const std::string& elementType,
                                                               const std::string& kernelType)
{
    const auto& c = plan.config;
    const bool valu = c.atom == "valu";
    std::vector<std::pair<std::string, std::string>> defines{{"CW_T", elementType},
                                                             {"CW_VALU", valu ? "1" : "0"}};
    defines.emplace_back(valu ? "CW_WAVE" : "CW_ATOM",
                         valu ? std::to_string(plan.wave) : kernelType);
    const std::array<std::pair<const char*, int>, 13> fields{
        {{"CW_KO", c.ko},
         {"CW_XW", plan.xw},
         {"CW_FD", c.fd},
         {"CW_FR", c.fr},
         {"CW_FS", c.fs},
         {"CW_MAPB", c.side == "k" ? 1 : 0},
         {"CW_NCHW", c.layout == "nchw" ? 1 : 0},
         {"CW_AV", plan.vectorIn ? 1 : 0},
         {"CW_WR", c.wr},
         {"CW_RB", c.rb},
         {"CW_NW", c.nw},
         {"CW_MPW", c.mpw},
         {"CW_NCP", plan.ncp}}};
    for(const auto& [name, value] : fields)
    {
        defines.emplace_back(name, std::to_string(value));
    }
    return defines;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_narrow

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
