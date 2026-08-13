// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// GPU numeric-parity test for the AOT catalog engine's forward-convolution path
// -- the sixth op wired into the catalog (after GEMM, RMSNorm, LayerNorm,
// activation, and SDPA). Like the other parity tests it drives the substrate
// directly (Catalog load -> candidate selection -> module load -> LaunchAbi
// pack/grid via CatalogPlan::execute) against the shipped gfx1151 rocKE
// conv2d_fprop .co and compares to a CPU reference.
//
// The point of this test is the runtime-generic ("fully dynamic") model: ONE
// shape-free .co per tile config serves ANY 2-D forward-conv shape, with partial
// tiles at the M / N_gemm / K_gemm boundaries masked. So the shapes below are
// chosen to be DELIBERATELY non-tile-aligned:
//   M      = N*Ho*Wo        (must NOT be a multiple of tile_m in {64,128})
//   N_gemm = K              (must NOT be a multiple of tile_n = 64)
//   K_gemm = R*S*C          (must NOT be a multiple of tile_k in {64,32})
// If a boundary were mis-addressed instead of masked, these would fail.
//
// Layout is the one the runtime kernel addresses: input NHWC, weight KRSC
// (== KYXC), output NHWK -- channels-last packed on every operand.
//
// Semantics (cross-correlation, the DL "convolution"):
//   Y[n,ho,wo,k] = sum_{r,s,c} X[n, ho*sH-pH+r*dH, wo*sW-pW+s*dW, c] * W[k,r,s,c]
// with out-of-bounds input taps contributing 0 (zero padding).

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "catalog/Catalog.hpp"
#include "catalog/CatalogTypes.hpp"
#include "core/Handle.hpp"
#include "engines/aot_catalog_engine/AotCatalogTestSupport.hpp"
#include "launch/LaunchAbi.hpp"
#include "launch/ModuleLoader.hpp"
#include "plans/CatalogPlan.hpp"

namespace
{

using namespace aot_catalog_engine;

// Baked in by CMake: the build-tree copy of the catalog (<arch>/<family>/...).
const std::string CATALOG_DIR = aotResolveTestCatalogDir();

// The shipped .co is gfx1151-only; the test is meaningful only on that arch.
constexpr const char* ARCH = "gfx1151";

bool gpuIsArch(const std::string& arch)
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        return false;
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return false;
    }
    // gcnArchName looks like "gfx1151:sramecc+:xnack-"; match the leading token.
    const std::string name(props.gcnArchName);
    return name.rfind(arch, 0) == 0;
}

// A 2-D forward-conv problem. Extents are chosen by the caller to be non-tile-
// aligned so the partial-tile masking is genuinely exercised.
//
// The fields keep the canonical convolution symbols (N,C,K,Hi,Wi,R,S + the
// stride/pad/dilation pairs) so they read 1:1 against the runtime ABI table in
// README_conv_enablement.md; that domain spelling collides with the provider's
// camelBack member rule, so identifier-naming is disabled just for this struct.
// NOLINTBEGIN(readability-identifier-naming)
struct ConvShape
{
    int N, C, K, Hi, Wi, R, S, sH, sW, pH, pW, dH, dW;

    int Ho() const
    {
        return (Hi + 2 * pH - dH * (R - 1) - 1) / sH + 1;
    }
    int Wo() const
    {
        return (Wi + 2 * pW - dW * (S - 1) - 1) / sW + 1;
    }
    int64_t M() const
    {
        return static_cast<int64_t>(N) * Ho() * Wo();
    }
    int64_t Ngemm() const
    {
        return K;
    }
    int64_t Kgemm() const
    {
        return static_cast<int64_t>(R) * S * C;
    }
    int64_t xNumel() const
    {
        return static_cast<int64_t>(N) * Hi * Wi * C;
    }
    int64_t wNumel() const
    {
        return static_cast<int64_t>(K) * R * S * C;
    }
    int64_t yNumel() const
    {
        return static_cast<int64_t>(N) * Ho() * Wo() * K;
    }
};
// NOLINTEND(readability-identifier-naming)

// Deterministic small inputs, kept O(0.1..1) so the K_gemm-length f32
// accumulation stays comfortably inside f16/bf16 dynamic range.
float xVal(int n, int h, int w, int c)
{
    const unsigned t = (static_cast<unsigned>(n) * 131u + static_cast<unsigned>(h) * 17u
                        + static_cast<unsigned>(w) * 7u + static_cast<unsigned>(c) * 3u)
                       % 7u;
    return (static_cast<float>(t) - 3.0f) * 0.1f;
}
float wVal(int k, int r, int s, int c)
{
    const unsigned t = (static_cast<unsigned>(k) * 53u + static_cast<unsigned>(r) * 19u
                        + static_cast<unsigned>(s) * 11u + static_cast<unsigned>(c) * 5u)
                       % 5u;
    return (static_cast<float>(t) - 2.0f) * 0.125f;
}

// Flat channels-last index for a [D0,D1,D2,D3] tensor -- NHWC (input), KRSC
// (weight) and NHWK (output) all share this shape. Every term is widened to
// size_t so the whole address computation stays unsigned end-to-end (keeps the
// Superbuild's -Wsign-conversion clean, where mixing int extents into a size_t
// expression errors).
std::size_t flatIdx(int d0, int d1, int d2, int d3, int e1, int e2, int e3)
{
    return ((static_cast<std::size_t>(d0) * static_cast<std::size_t>(e1)
             + static_cast<std::size_t>(d1))
                * static_cast<std::size_t>(e2)
            + static_cast<std::size_t>(d2))
               * static_cast<std::size_t>(e3)
           + static_cast<std::size_t>(d3);
}

// f32 NHWC / KRSC / NHWK cross-correlation reference. Reads the (dtype-rounded)
// host inputs back through `readX`/`readW` so the reference sees exactly the
// values the kernel does.
template <typename ReadX, typename ReadW>
std::vector<float> convReference(const ConvShape& p, const ReadX& readX, const ReadW& readW)
{
    const int outH = p.Ho();
    const int outW = p.Wo();
    std::vector<float> ref(static_cast<size_t>(p.yNumel()), 0.0f);
    for(int n = 0; n < p.N; ++n)
    {
        for(int ho = 0; ho < outH; ++ho)
        {
            for(int wo = 0; wo < outW; ++wo)
            {
                for(int k = 0; k < p.K; ++k)
                {
                    float acc = 0.0f;
                    for(int r = 0; r < p.R; ++r)
                    {
                        const int hi = ho * p.sH - p.pH + r * p.dH;
                        if(hi < 0 || hi >= p.Hi)
                        {
                            continue;
                        }
                        for(int s = 0; s < p.S; ++s)
                        {
                            const int wi = wo * p.sW - p.pW + s * p.dW;
                            if(wi < 0 || wi >= p.Wi)
                            {
                                continue;
                            }
                            for(int c = 0; c < p.C; ++c)
                            {
                                acc += readX(n, hi, wi, c) * readW(k, r, s, c);
                            }
                        }
                    }
                    ref[flatIdx(n, ho, wo, k, outH, outW, p.K)] = acc;
                }
            }
        }
    }
    return ref;
}

// Build the (A,B,D + *_bytes + geometry) launch bindings the ConvFpropAdapter
// would produce for `p` at element size `elemBytes`. Pointer uids: A=1,B=2,D=3.
catalog::LaunchBindings makeBindings(const ConvShape& p, int64_t elemBytes)
{
    catalog::LaunchBindings bindings;
    bindings.pointerUids.emplace("A", 1);
    bindings.pointerUids.emplace("B", 2);
    bindings.pointerUids.emplace("D", 3);
    bindings.scalars.emplace("A_bytes", catalog::ScalarValue{p.xNumel() * elemBytes});
    bindings.scalars.emplace("B_bytes", catalog::ScalarValue{p.wNumel() * elemBytes});
    bindings.scalars.emplace("D_bytes", catalog::ScalarValue{p.yNumel() * elemBytes});
    bindings.scalars.emplace("N", catalog::ScalarValue{static_cast<int64_t>(p.N)});
    bindings.scalars.emplace("C", catalog::ScalarValue{static_cast<int64_t>(p.C)});
    bindings.scalars.emplace("K", catalog::ScalarValue{static_cast<int64_t>(p.K)});
    bindings.scalars.emplace("Hi", catalog::ScalarValue{static_cast<int64_t>(p.Hi)});
    bindings.scalars.emplace("Wi", catalog::ScalarValue{static_cast<int64_t>(p.Wi)});
    bindings.scalars.emplace("R", catalog::ScalarValue{static_cast<int64_t>(p.R)});
    bindings.scalars.emplace("S", catalog::ScalarValue{static_cast<int64_t>(p.S)});
    bindings.scalars.emplace("sH", catalog::ScalarValue{static_cast<int64_t>(p.sH)});
    bindings.scalars.emplace("sW", catalog::ScalarValue{static_cast<int64_t>(p.sW)});
    bindings.scalars.emplace("pH", catalog::ScalarValue{static_cast<int64_t>(p.pH)});
    bindings.scalars.emplace("pW", catalog::ScalarValue{static_cast<int64_t>(p.pW)});
    bindings.scalars.emplace("dH", catalog::ScalarValue{static_cast<int64_t>(p.dH)});
    bindings.scalars.emplace("dW", catalog::ScalarValue{static_cast<int64_t>(p.dW)});
    return bindings;
}

// Grid + selection symbols. The kernel's grid formula is
// ceil_div(N_gemm,tile_n) x ceil_div(M,tile_m); the tile literals are baked into
// each .co's grid entry, so only the symbol values cross here.
launch::SymbolTable makeGridSymbols(const ConvShape& p)
{
    launch::SymbolTable symbols;
    symbols.emplace("M", p.M());
    symbols.emplace("N_gemm", p.Ngemm());
    symbols.emplace("K_gemm", p.Kgemm());
    symbols.emplace("N", static_cast<int64_t>(p.N));
    symbols.emplace("C", static_cast<int64_t>(p.C));
    symbols.emplace("K", static_cast<int64_t>(p.K));
    symbols.emplace("Ho", static_cast<int64_t>(p.Ho()));
    symbols.emplace("Wo", static_cast<int64_t>(p.Wo()));
    return symbols;
}

uint16_t floatToBf16(float f)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t roundingBias = ((bits >> 16) & 1u) + 0x7fffu; // round to nearest even
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16ToFloat(uint16_t bpat)
{
    const uint32_t bits = static_cast<uint32_t>(bpat) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Select the first f16/bf16 candidate for `p`, run it as a single-candidate
// plan on real device buffers, and return the device output decoded to float.
// Returns false (via ADD_FAILURE at the call site) only through EXPECT_* here.
template <typename Elem, typename Encode, typename Decode>
void runAndCompare(const std::string& dtype,
                   const ConvShape& p,
                   const Encode& encode,
                   const Decode& decode,
                   float absTol,
                   float relTol)
{
    const catalog::Catalog cat = catalog::Catalog::loadForDevice(CATALOG_DIR, ARCH);
    if(cat.empty())
    {
        AOT_SKIP_OR_FAIL_ON_EMPTY_CATALOG(CATALOG_DIR);
    }

    catalog::ProblemShape problem;
    problem.emplace("dtype", catalog::ShapeValue{dtype});
    problem.emplace("groups", catalog::ShapeValue{static_cast<int64_t>(1)});
    problem.emplace("C", catalog::ShapeValue{static_cast<int64_t>(p.C)});
    problem.emplace("K", catalog::ShapeValue{static_cast<int64_t>(p.K)});

    const std::vector<catalog::Catalog::Candidate> candidates
        = cat.candidatesFor("conv_fprop", problem);
    ASSERT_FALSE(candidates.empty()) << "no conv_fprop candidate for the " << dtype << " problem";
    const catalog::KernelEntry& kernel = *candidates.front().kernel;

    std::optional<launch::HipModuleGuard> module
        = launch::loadKernelModule(kernel.coPath, kernel.symbol);
    ASSERT_TRUE(module.has_value()) << "failed to load " << kernel.coPath;

    // Host inputs (dtype-encoded) + f32 reference decoded from the same bits.
    std::vector<Elem> hostX(static_cast<size_t>(p.xNumel()));
    std::vector<Elem> hostW(static_cast<size_t>(p.wNumel()));
    std::vector<Elem> hostY(static_cast<size_t>(p.yNumel()), Elem{});
    for(int n = 0; n < p.N; ++n)
    {
        for(int h = 0; h < p.Hi; ++h)
        {
            for(int w = 0; w < p.Wi; ++w)
            {
                for(int c = 0; c < p.C; ++c)
                {
                    hostX[flatIdx(n, h, w, c, p.Hi, p.Wi, p.C)] = encode(xVal(n, h, w, c));
                }
            }
        }
    }
    for(int k = 0; k < p.K; ++k)
    {
        for(int r = 0; r < p.R; ++r)
        {
            for(int s = 0; s < p.S; ++s)
            {
                for(int c = 0; c < p.C; ++c)
                {
                    hostW[flatIdx(k, r, s, c, p.R, p.S, p.C)] = encode(wVal(k, r, s, c));
                }
            }
        }
    }
    auto readX = [&](int n, int h, int w, int c) {
        return decode(hostX[flatIdx(n, h, w, c, p.Hi, p.Wi, p.C)]);
    };
    auto readW = [&](int k, int r, int s, int c) {
        return decode(hostW[flatIdx(k, r, s, c, p.R, p.S, p.C)]);
    };
    const std::vector<float> ref = convReference(p, readX, readW);

    catalog::LaunchBindings bindings = makeBindings(p, static_cast<int64_t>(sizeof(Elem)));
    launch::SymbolTable gridSymbols = makeGridSymbols(p);
    const auto workspaceBytes
        = static_cast<size_t>(launch::evalWorkspace(kernel.workspace, gridSymbols));

    const CatalogPlan plan(std::move(*module),
                           kernel.launch,
                           std::move(bindings),
                           std::move(gridSymbols),
                           workspaceBytes,
                           kernel.symbol);

    void* deviceX = nullptr;
    void* deviceW = nullptr;
    void* deviceY = nullptr;
    ASSERT_EQ(hipMalloc(&deviceX, hostX.size() * sizeof(Elem)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceW, hostW.size() * sizeof(Elem)), hipSuccess);
    ASSERT_EQ(hipMalloc(&deviceY, hostY.size() * sizeof(Elem)), hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceX, hostX.data(), hostX.size() * sizeof(Elem), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(deviceW, hostW.data(), hostW.size() * sizeof(Elem), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(deviceY, 0, hostY.size() * sizeof(Elem)), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    Handle handle;
    handle.setStream(stream);

    const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers = {{
        {1, deviceX},
        {2, deviceW},
        {3, deviceY},
    }};

    ASSERT_NO_THROW(
        plan.execute(handle, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipMemcpy(hostY.data(), deviceY, hostY.size() * sizeof(Elem), hipMemcpyDeviceToHost),
              hipSuccess);

    for(size_t i = 0; i < hostY.size(); ++i)
    {
        const float got = decode(hostY[i]);
        const float want = ref[i];
        const float tol = std::max(absTol, relTol * std::fabs(want));
        ASSERT_NEAR(got, want, tol) << dtype << " mismatch at flat y index " << i;
    }

    (void)hipFree(deviceX);
    (void)hipFree(deviceW);
    (void)hipFree(deviceY);
    (void)hipStreamDestroy(stream);
}

// A 3x3 stride-1 pad-1 conv on a 7x7 map: M=49, N_gemm=16, K_gemm=72 -- all
// three implicit-GEMM extents are partial against every shipped tile config, so
// the boundary masking on M, N_gemm, and K_gemm is all exercised at once.
constexpr ConvShape NON_ALIGNED_3X3{/*N=*/1,
                                    /*C=*/8,
                                    /*K=*/16,
                                    /*Hi=*/7,
                                    /*Wi=*/7,
                                    /*R=*/3,
                                    /*S=*/3,
                                    /*sH=*/1,
                                    /*sW=*/1,
                                    /*pH=*/1,
                                    /*pW=*/1,
                                    /*dH=*/1,
                                    /*dW=*/1};

// A strided, dilated, asymmetric-extent conv: exercises the runtime stride/pad/
// dilation coordinate maps (not just the halo). Hi!=Wi, stride 2, dilation 2.
// Ho = (11+2-2*2-1)/2+1 = 4, Wo = (9+2-2*2-1)/2+1 = 3 -> M=1*4*3=12; N_gemm=24;
// K_gemm=2*2*8=32 (partial vs tile_m/tile_n=64; exact vs one tile_k=32 config,
// partial vs the tile_k=64 config -- both masking paths covered across configs).
constexpr ConvShape STRIDED_DILATED{/*N=*/1,
                                    /*C=*/8,
                                    /*K=*/24,
                                    /*Hi=*/11,
                                    /*Wi=*/9,
                                    /*R=*/2,
                                    /*S=*/2,
                                    /*sH=*/2,
                                    /*sW=*/2,
                                    /*pH=*/1,
                                    /*pW=*/1,
                                    /*dH=*/2,
                                    /*dW=*/2};

} // namespace

TEST(TestAotCatalogConvNumericParity, F16NonAligned3x3MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    runAndCompare<_Float16>(
        "f16",
        NON_ALIGNED_3X3,
        [](float f) { return static_cast<_Float16>(f); },
        [](_Float16 h) { return static_cast<float>(h); },
        /*absTol=*/2e-2f,
        /*relTol=*/3e-2f);
}

TEST(TestAotCatalogConvNumericParity, Bf16NonAligned3x3MatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    runAndCompare<uint16_t>(
        "bf16",
        NON_ALIGNED_3X3,
        [](float f) { return floatToBf16(f); },
        [](uint16_t b) { return bf16ToFloat(b); },
        /*absTol=*/5e-2f,
        /*relTol=*/5e-2f);
}

TEST(TestAotCatalogConvNumericParity, F16StridedDilatedMatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    runAndCompare<_Float16>(
        "f16",
        STRIDED_DILATED,
        [](float f) { return static_cast<_Float16>(f); },
        [](_Float16 h) { return static_cast<float>(h); },
        /*absTol=*/2e-2f,
        /*relTol=*/3e-2f);
}

TEST(TestAotCatalogConvNumericParity, Bf16StridedDilatedMatchesReference)
{
    if(!gpuIsArch(ARCH))
    {
        GTEST_SKIP() << "no " << ARCH << " GPU present";
    }
    runAndCompare<uint16_t>(
        "bf16",
        STRIDED_DILATED,
        [](float f) { return floatToBf16(f); },
        [](uint16_t b) { return bf16ToFloat(b); },
        /*absTol=*/5e-2f,
        /*relTol=*/5e-2f);
}
