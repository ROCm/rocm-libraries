// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Host-only selection test for the AOT catalog engine's conv2d_fprop family.
// Unlike the parity test it needs no GPU: it exercises the catalog *matcher*
// (Catalog::candidatesFor -> Selection constraint eval) against hand-built
// ProblemShapes, proving the two properties the runtime-generic ("fully
// dynamic") model rests on:
//
//   1. GENERICITY -- one shape-free .co per tile config serves ANY conv shape.
//      Every kernel in the family carries NO shape 'equals'/'multiple_of' on the
//      spatial dims, so the SAME set of candidate symbols is selected for wildly
//      different geometries (tiny 1x1, large strided, non-tile-aligned M/N/K).
//      This is the conv analog of layernorm2d's runtime-N (_dyn_) kernels.
//
//   2. FAIL-CLOSED -- the family gates on exactly dtype / groups==1 / C,K
//      multiple_of vec. A grouped conv, an unsupported dtype, or a channel count
//      the vectorized loads can't service finds NO candidate (so the engine
//      declines and another engine / the fallback runs), rather than launching a
//      kernel that would mis-address memory.
//
// The published-fact contract these assertions mirror lives in
// ConvFpropAdapter::decode (dtype, groups, C, K, ... are all ProblemShape keys).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "engines/aot_catalog_engine/AotCatalogTestSupport.hpp"

namespace
{

using namespace aot_catalog_engine;

// Baked in by CMake: the build-tree copy of the catalog (<arch>/<family>/...).
const std::string CATALOG_DIR = aotResolveTestCatalogDir();
constexpr const char* ARCH = "gfx1151";

// The four facts the conv2d_fprop constraints reference. Spatial geometry is
// deliberately NOT part of selection -- that is the whole point of the family --
// so a "shape" here is only what the matcher looks at plus the extents we vary
// to prove they do not move the selection.
// C/K keep the canonical channel-dim spelling so the keys read against the
// family.json constraints; that collides with the camelBack member rule, so
// identifier-naming is disabled just for this struct.
// NOLINTBEGIN(readability-identifier-naming)
struct SelKey
{
    std::string dtype;
    int64_t groups;
    int64_t C;
    int64_t K;
};
// NOLINTEND(readability-identifier-naming)

catalog::ProblemShape makeProblem(const SelKey& s)
{
    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{s.dtype});
    problem.emplace("groups", catalog::ShapeValue{s.groups});
    problem.emplace("C", catalog::ShapeValue{s.C});
    problem.emplace("K", catalog::ShapeValue{s.K});
    return problem;
}

// Sorted candidate symbols for `s` -- sorted so two shapes' candidate SETS can be
// compared directly regardless of catalog iteration order.
std::vector<std::string> symbolsFor(const catalog::Catalog& cat, const SelKey& s)
{
    const catalog::ProblemShape problem = makeProblem(s);
    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("conv_fprop", problem);
    std::vector<std::string> symbols;
    symbols.reserve(candidates.size());
    for(const catalog::Catalog::Candidate& candidate : candidates)
    {
        symbols.push_back(candidate.kernel->symbol);
    }
    std::sort(symbols.begin(), symbols.end());
    return symbols;
}

bool allRuntimeGeneric(const std::vector<std::string>& symbols)
{
    return !symbols.empty()
           && std::all_of(symbols.begin(), symbols.end(), [](const std::string& s) {
                  return s.find("_dyn_") != std::string::npos;
              });
}

} // namespace

// One f16 problem draws several tile-config candidates, all runtime-generic
// (_dyn_) -- the multi-candidate set the tuner picks the winner from.
TEST(TestConvSelection, MultipleRuntimeGenericCandidatesForOneProblem)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    const std::vector<std::string> symbols
        = symbolsFor(cat, SelKey{"f16", /*groups=*/1, /*C=*/8, /*K=*/16});
    EXPECT_GE(symbols.size(), 2u)
        << "expected multiple tile-config candidates for one conv problem";
    EXPECT_TRUE(allRuntimeGeneric(symbols))
        << "every conv2d_fprop kernel must be a shape-free _dyn_ kernel";
}

// GENERICITY: the SAME candidate set is selected for a wide range of geometries
// -- tiny, large, strided/dilated, and non-tile-aligned M/N_gemm/K_gemm -- so
// one .co per config genuinely serves any shape. Only the four selection facts
// are held fixed (dtype/groups/C/K); the spatial geometry that varies between
// these problems is invisible to selection by construction.
TEST(TestConvSelection, OneCoServesManyShapesF16)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    // All share dtype=f16, groups=1, and C/K multiples of the vec width; they
    // differ only in the (unselected) spatial geometry they would run at.
    const SelKey base{"f16", 1, 8, 16};
    const std::vector<std::string> baseline = symbolsFor(cat, base);
    ASSERT_FALSE(baseline.empty()) << "no candidate for the baseline f16 conv problem";
    ASSERT_TRUE(allRuntimeGeneric(baseline));

    // Different C/K magnitudes (still vec-aligned) must not change WHICH kernels
    // match -- the constraint is multiple_of, not equals.
    for(const SelKey& s : {
            SelKey{"f16", 1, 8, 8}, // 1x1-ish tiny
            SelKey{"f16", 1, 64, 128}, // large
            SelKey{"f16", 1, 16, 512}, // wide output
            SelKey{"f16", 1, 320, 640}, // diffusion-scale channels
        })
    {
        const std::vector<std::string> symbols = symbolsFor(cat, s);
        EXPECT_EQ(symbols, baseline)
            << "C=" << s.C << " K=" << s.K << " selected a different candidate set";
    }
}

// bf16 draws its own shape-free candidate set (the diffusion dtype path).
TEST(TestConvSelection, OneCoServesManyShapesBf16)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    const std::vector<std::string> baseline = symbolsFor(cat, SelKey{"bf16", 1, 8, 16});
    ASSERT_FALSE(baseline.empty()) << "no bf16 conv candidate";
    ASSERT_TRUE(allRuntimeGeneric(baseline));

    const std::vector<std::string> other = symbolsFor(cat, SelKey{"bf16", 1, 128, 256});
    EXPECT_EQ(other, baseline) << "bf16 selection must be shape-independent";
}

// FAIL-CLOSED: a grouped conv (groups != 1) finds no candidate -- the gfx1151
// family is groups==1 only; a CDNA grouped kernel would opt in via its own
// constraint, with zero C++ change.
TEST(TestConvSelection, GroupedConvDeclines)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    EXPECT_TRUE(symbolsFor(cat, SelKey{"f16", /*groups=*/2, 8, 16}).empty())
        << "grouped conv must find no gfx1151 candidate";
    EXPECT_TRUE(symbolsFor(cat, SelKey{"f16", /*groups=*/8, 64, 64}).empty())
        << "grouped conv must find no gfx1151 candidate";
}

// FAIL-CLOSED: an unsupported dtype finds no candidate.
TEST(TestConvSelection, UnsupportedDtypeDeclines)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    for(const char* dtype : {"f32", "f8_e4m3", "i8"})
    {
        EXPECT_TRUE(symbolsFor(cat, SelKey{dtype, 1, 8, 16}).empty())
            << "dtype '" << dtype << "' must find no candidate";
    }
}

// FAIL-CLOSED: a channel count the vectorized (vec=8) loads can't service finds
// no candidate -- C or K not a multiple of the vec width declines, rather than
// launching a kernel that would over-read the channel-contiguous vector loads.
TEST(TestConvSelection, MisalignedChannelsDecline)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    // C not a multiple of 8 (12 = mult of 4, not 8) -> declines.
    EXPECT_TRUE(symbolsFor(cat, SelKey{"f16", 1, /*C=*/12, /*K=*/16}).empty())
        << "C not a multiple of the vec width must decline";
    // K not a multiple of 8 (20 = mult of 4, not 8) -> declines.
    EXPECT_TRUE(symbolsFor(cat, SelKey{"f16", 1, /*C=*/8, /*K=*/20}).empty())
        << "K not a multiple of the vec width must decline";
    // A vec-aligned pair on the same axes DOES match (guards against the above
    // passing for the wrong reason, e.g. an empty catalog family).
    EXPECT_FALSE(symbolsFor(cat, SelKey{"f16", 1, /*C=*/16, /*K=*/16}).empty())
        << "a vec-aligned C/K pair must still match";
}
