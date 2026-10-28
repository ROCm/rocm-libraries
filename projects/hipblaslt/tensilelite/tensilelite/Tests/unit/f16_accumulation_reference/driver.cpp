// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Runs the f16-accumulation reference model (client/include/F16AccumulationStep.hpp) for
// test_f16_accumulation_reference.py:
//   driver golden <file>  WMMA records; prints "<records> <differing results> <differing short steps>"
//   driver order <file>   one case per line, "depthU gsu gsuc staggerU shift mapping pgr K wg0 wg1 numTiles0 slice";
//                         prints the slice's 64-K step starts per line
//   driver reduce <file>  one line of f16 partials (hex) per case; prints the reduced f16 (hex)
//   driver tree <file>    the same with the StreamK tree fixup
//   driver chunks <file>  one case per line, "kind tree perTile tiles itersPerTile grid skTiles skItersPerWG
//                         extraIters split skItersPerWI t"; prints the StreamK chunks of tile t as "start end" pairs,
//                         or "error"
//   driver chunksteps <file> one case per line, "depthU staggerU shift mapping pgr K wg0 wg1 numTiles0 itersPerTile
//                         start end"; prints the chunk's 64-K step starts
//   driver tileorder <file> one launch per line, "wgm tiles0 tiles1 batches [spaceFillingAlgo wgmWord]"; prints the
//                         linear tile index at each workgroup
//   driver round          boundary cases of the rounding helpers; prints the number of failures

#include "F16AccumulationStep.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace TensileLite::Client;

static double fp8Value(unsigned v)
{
    unsigned e = (v >> 3) & 15, m = v & 7;
    if(e == 15 && m == 7)
        return std::nan("");
    double x = e ? std::ldexp(double(8 + m), int(e) - 10) : std::ldexp(double(m), -9);
    return (v & 0x80) ? -x : x;
}

static std::vector<double> fp8Values(std::string const& hex)
{
    std::vector<double> v(hex.size() / 2);
    for(size_t i = 0; i < v.size(); i++)
        v[i] = fp8Value(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
    return v;
}

static int golden(char const* path)
{
    std::ifstream in(path);
    std::string   kind, c, a, b, d;
    size_t        records = 0, differ = 0, differShort = 0;
    while(in >> kind >> c >> a >> b >> d)
    {
        std::vector<double> av = fp8Values(a), bv = fp8Values(b);
        uint16_t const      expect = uint16_t(std::stoul(d, nullptr, 16));
        uint16_t            acc    = uint16_t(std::stoul(c, nullptr, 16));
        for(size_t s = 0; s < av.size(); s += f16AccumulationBlockK)
            acc = f16AccumulationStep(acc, &av[s], &bv[s], f16AccumulationBlockK);
        differ += acc != expect;
        if(av.size() == f16AccumulationBlockK)
        {
            // A step shortened to its last nonzero product is the reference's zero-padded tail step.
            size_t n = 0;
            for(size_t k = 0; k < av.size(); k++)
                if(av[k] != 0 && bv[k] != 0)
                    n = k + 1;
            if(n < av.size())
                differShort += f16AccumulationStep(uint16_t(std::stoul(c, nullptr, 16)), &av[0], &bv[0], n) != expect;
        }
        records++;
    }
    std::printf("%zu %zu %zu\n", records, differ, differShort);
    return 0;
}

static int order(char const* path)
{
    std::ifstream in(path);
    std::string   line;
    while(std::getline(in, line))
    {
        std::istringstream    is(line);
        F16AccumulationLaunch l;
        unsigned long         gsuc, K, wg0, wg1, numTiles0, slice;
        is >> l.depthU >> l.gsu >> gsuc >> l.staggerU >> l.staggerUStrideShift >> l.staggerUMapping
            >> l.prefetchGlobalRead >> K >> wg0 >> wg1 >> numTiles0 >> slice;
        if(!is)
            continue;
        l.gsuCoalesced = gsuc != 0;
        for(size_t k : f16AccumulationSliceSteps(
                l, K, uint32_t(slice), uint32_t(wg0), uint32_t(wg1), uint32_t(numTiles0)))
            std::printf("%zu ", k);
        std::printf("\n");
    }
    return 0;
}

static int key(char const* path)
{
    std::ifstream in(path);
    std::string   line;
    while(std::getline(in, line))
    {
        std::istringstream    is(line);
        F16AccumulationLaunch l;
        unsigned long         gsuc;
        is >> l.depthU >> l.gsu >> gsuc >> l.staggerU >> l.staggerUStrideShift >> l.staggerUMapping
            >> l.prefetchGlobalRead >> l.macroTile0 >> l.macroTile1;
        if(!is)
            continue;
        l.gsuCoalesced = gsuc != 0;
        for(uint64_t v : f16AccumulationOrderKey(l))
            std::printf("%llu ", static_cast<unsigned long long>(v));
        std::printf("\n");
    }
    return 0;
}

static int reduce(char const* path, bool tree)
{
    std::ifstream in(path);
    std::string   line;
    while(std::getline(in, line))
    {
        std::istringstream    is(line);
        std::vector<uint16_t> partials;
        std::string           h;
        while(is >> h)
            partials.push_back(uint16_t(std::stoul(h, nullptr, 16)));
        if(!partials.empty())
            std::printf("%04x\n",
                        tree ? f16AccumulationTreeReduce(partials.data(), partials.size())
                             : f16AccumulationReduce(partials.data(), partials.size()));
    }
    return 0;
}

static int chunks(char const* path)
{
    std::ifstream in(path);
    std::string   line;
    while(std::getline(in, line))
    {
        std::istringstream     is(line);
        F16AccumulationStreamK sk;
        unsigned long          tree, perTile, t;
        is >> sk.kind >> tree >> perTile >> sk.tiles >> sk.itersPerTile >> sk.grid >> sk.skTiles >> sk.skItersPerWG
            >> sk.extraIters >> sk.split >> sk.skItersPerWI >> t;
        if(!is)
            continue;
        sk.tree          = tree != 0;
        sk.perTileExtras = perTile != 0;
        try
        {
            for(auto const& c : f16AccumulationStreamKChunks(sk, t))
                std::printf("%zu %zu ", c.start, c.end);
            std::printf("\n");
        }
        catch(std::runtime_error const&)
        {
            std::printf("error\n");
        }
    }
    return 0;
}

static int tileOrder(char const* path)
{
    std::ifstream in(path);
    std::string   line;
    while(std::getline(in, line))
    {
        std::istringstream     is(line);
        F16AccumulationStreamK sk;
        size_t                 batches;
        is >> sk.wgm >> sk.tiles0 >> sk.tiles1 >> batches;
        if(!is)
            continue;
        if(is >> sk.spaceFillingAlgo >> sk.wgmWord)
            sk.spaceFilling = true;
        sk.tiles = sk.tiles0 * sk.tiles1 * batches;
        for(size_t t : f16AccumulationTileOrder(sk))
            std::printf("%zu ", t);
        std::printf("\n");
    }
    return 0;
}

static int chunkSteps(char const* path)
{
    std::ifstream in(path);
    std::string   line;
    while(std::getline(in, line))
    {
        std::istringstream    is(line);
        F16AccumulationLaunch l;
        F16AccumulationChunk  c;
        unsigned long         K, wg0, wg1, numTiles0;
        is >> l.depthU >> l.staggerU >> l.staggerUStrideShift >> l.staggerUMapping >> l.prefetchGlobalRead >> K
            >> wg0 >> wg1 >> numTiles0 >> l.streamK.itersPerTile >> c.start >> c.end;
        if(!is)
            continue;
        f16AccumulationForEachChunkStep(
            l, K, c, uint32_t(wg0), uint32_t(wg1), uint32_t(numTiles0), [](size_t k, size_t) {
                std::printf("%zu ", k);
            });
        std::printf("\n");
    }
    return 0;
}

static int roundingCases()
{
    int  failures = 0;
    auto expect   = [&](uint16_t got, uint16_t want) { failures += got != want; };

    expect(f16AccumulationFromFloat(65504.0f), 0x7bff);
    expect(f16AccumulationFromFloat(std::nextafter(65520.0f, 0.0f)), 0x7bff);
    expect(f16AccumulationFromFloat(65520.0f), 0x7c00);
    expect(f16AccumulationFromFloat(-65520.0f), 0xfc00);
    expect(f16AccumulationFromFloat(INFINITY), 0x7c00);
    expect(f16AccumulationFromFloat(std::ldexp(1.0f, -25)), 0x0000);
    expect(f16AccumulationFromFloat(std::nextafter(std::ldexp(1.0f, -25), 1.0f)), 0x0001);
    expect(f16AccumulationFromFloat(std::ldexp(3.0f, -25)), 0x0002);
    expect(f16AccumulationFromFloat(-std::ldexp(1.0f, -24)), 0x8001);
    expect(f16AccumulationFromFloat(std::ldexp(1.0f, -14) - std::ldexp(1.0f, -25)), 0x0400);
    expect(f16AccumulationFromFloat(1.0f + std::ldexp(1.0f, -11)), 0x3c00);
    expect(f16AccumulationFromFloat(1.0f + std::ldexp(3.0f, -11)), 0x3c02);
    expect(f16AccumulationFromFloat(-0.0f), 0x8000);
    expect(f16AccumulationFromFloat(std::nanf("")) & 0x7fff, 0x7e00);

    expect(f16AccumulationRound(int64_t(65504) << 24), 0x7bff);
    expect(f16AccumulationRound((int64_t(65520) << 24) - 1), 0x7bff);
    expect(f16AccumulationRound(int64_t(65520) << 24), 0x7c00);
    expect(f16AccumulationRound(-(int64_t(65520) << 24)), 0xfc00);
    expect(f16AccumulationRound(1), 0x0001);
    expect(f16AccumulationRound(-1), 0x8001);
    expect(f16AccumulationRound(0), 0x0000);

    failures += f16AccumulationFloor(-1, 2) != -4;
    failures += f16AccumulationFloor(5, 2) != 4;
    failures += f16AccumulationFloor(-4, 2) != -4;
    failures += f16AccumulationFloor(-7, 0) != -7;

    for(uint32_t h = 0; h < 0x10000; h++)
    {
        uint16_t const back = f16AccumulationFromFloat(f16AccumulationToFloat(uint16_t(h)));
        bool const     nan  = (h & 0x7c00) == 0x7c00 && (h & 0x3ff);
        failures += nan ? (back & 0x7fff) != 0x7e00 : back != h;
    }
    // Two implementations of nearest-even: integer units of 2^-24, and f32.
    for(int64_t v = -(int64_t(1) << 24); v <= (int64_t(1) << 24); v += 4099)
        failures += f16AccumulationRound(v) != f16AccumulationFromFloat(std::ldexp(float(v), -24));

    std::printf("%d\n", failures);
    return 0;
}

int main(int argc, char** argv)
{
    std::string mode = argc > 1 ? argv[1] : "";
    if(mode == "golden" && argc > 2)
        return golden(argv[2]);
    if(mode == "order" && argc > 2)
        return order(argv[2]);
    if(mode == "reduce" && argc > 2)
        return reduce(argv[2], false);
    if(mode == "tree" && argc > 2)
        return reduce(argv[2], true);
    if(mode == "chunks" && argc > 2)
        return chunks(argv[2]);
    if(mode == "chunksteps" && argc > 2)
        return chunkSteps(argv[2]);
    if(mode == "tileorder" && argc > 2)
        return tileOrder(argv[2]);
    if(mode == "key" && argc > 2)
        return key(argv[2]);
    if(mode == "round")
        return roundingCases();
    std::fprintf(stderr, "usage: driver golden|order|reduce|tree|chunks|chunksteps|tileorder|key <file> | round\n");
    return 2;
}
