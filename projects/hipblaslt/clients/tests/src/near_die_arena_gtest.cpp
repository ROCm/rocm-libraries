// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The near-die MXFP6 GEMM reads the closer memory only when its operands sit in a
// physically contiguous allocation. SameResultAsOrdinaryAllocation checks that
// this allocation returns the same result as an ordinary buffer. KernelTimes
// reports the time of each registered kernel.
//
// The suite name starts with "stress" so the large-memory tier runs it. A normal
// PR run does not: the allocation is 64 GiB + 2 MiB, and it is skipped unless the
// GPU is gfx950 chip 0x75a0 or 0x75a3, the driver is amdgpu 6.19 or newer, and less
// than 2 GiB of device memory is in use.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include "mxDataGen.hpp"
#include "mx_tile128x4.hpp"

#include <Tensile/hip/HipHardware.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{
    constexpr int64_t kM          = 4096;
    constexpr int64_t kN          = 4096;
    constexpr int64_t kK          = 65536;
    constexpr int64_t kKBlocks    = kK / 32;
    constexpr int     kCold       = 200;
    constexpr int     kHot        = 500;
    constexpr size_t  kWorkspace  = 256ull << 20;
    constexpr int64_t kArenaBytes = (64ll << 30) + (2ll << 20);
    constexpr int64_t kCleanBelow = 2ll << 30;
    constexpr int64_t kSlotAlign  = 512ll << 20;
    constexpr int64_t kArenaLimit = 32ll << 30;

    float       kAlpha = 1.f;
    float       kBeta  = 0.f;

    bool driverSelectsNearDie()
    {
        std::ifstream in("/sys/module/amdgpu/version");
        std::string   version;
        std::getline(in, version);
        return TensileLite::hip::page24RuleHolds(version);
    }

    // Bytes of device memory in use, or -1 when the file cannot be read.
    int64_t vramUsed(hipDeviceProp_t const& prop)
    {
        char path[160];
        std::snprintf(path,
                      sizeof(path),
                      "/sys/bus/pci/devices/%04x:%02x:%02x.0/mem_info_vram_used",
                      prop.pciDomainID,
                      prop.pciBusID,
                      prop.pciDeviceID);
        FILE* f = std::fopen(path, "r");
        if(!f)
            return -1;
        long long used = -1;
        if(std::fscanf(f, "%lld", &used) != 1)
            used = -1;
        std::fclose(f);
        return used;
    }

    struct Arena
    {
        void*   raw  = nullptr;
        char*   base = nullptr;
        int64_t next = 0;

        ~Arena()
        {
            if(raw)
                static_cast<void>(hipFree(raw));
        }

        // 512 MiB aligned, so bits 21 and 28 of an offset match the arena.
        char* slot(int64_t bytes)
        {
            int64_t const off = (next + kSlotAlign - 1) / kSlotAlign * kSlotAlign;
            if(off + bytes > kArenaLimit)
                return nullptr;
            next = off + bytes;
            return base + off;
        }
    };

    struct Owned
    {
        std::vector<void*> ptrs;
        ~Owned()
        {
            for(void* p : ptrs)
                if(p)
                    static_cast<void>(hipFree(p));
        }
        void* alloc(size_t bytes)
        {
            void* p = nullptr;
            if(hipMalloc(&p, bytes) != hipSuccess)
                return nullptr;
            ptrs.push_back(p);
            return p;
        }
    };

    bool upload(void* dst, void const* src, size_t bytes)
    {
        return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) == hipSuccess;
    }

    struct Images
    {
        std::vector<uint8_t> nearA;
        std::vector<uint8_t> nearB;
        std::vector<uint8_t> nearScaleA;
        std::vector<uint8_t> scaleB;
        std::vector<uint8_t> groupedA;
        std::vector<uint8_t> groupedB;
        std::vector<uint8_t> groupedScaleA;
        std::vector<uint8_t> planarA;
        std::vector<uint8_t> planarB;
        // Dequantized values, index k + row * K.
        std::vector<float> refA;
        std::vector<float> refB;
    };

    // trig_float, the same fill hipblaslt-bench uses. A constant byte pattern,
    // and zeros in particular, skips real multiply work and reports a higher rate.
    Images makeImages(hipDataType type)
    {
        size_t const         packed  = static_cast<size_t>(kM) * kK * 6 / 8;
        size_t const         nScales = static_cast<size_t>(kM * kKBlocks);
        std::vector<uint8_t> rawA(packed);
        std::vector<uint8_t> rawB(packed);
        std::vector<uint8_t> scalesA(nScales);
        std::vector<uint8_t> scalesB(nScales);
        Images               out;
        out.refA = generateMXInput(type,
                                   HIP_R_8F_UE8M0,
                                   rawA.data(),
                                   scalesA.data(),
                                   kK,
                                   kM,
                                   kK,
                                   true,
                                   32,
                                   1,
                                   true,
                                   MXScaleLayout::None,
                                   "trig_float");
        out.refB = generateMXInput(type,
                                   HIP_R_8F_UE8M0,
                                   rawB.data(),
                                   scalesB.data(),
                                   kK,
                                   kN,
                                   kK,
                                   false,
                                   32,
                                   1,
                                   false,
                                   MXScaleLayout::None,
                                   "trig_float");
        {
            std::vector<uint8_t> placed(mx_tile128x4::placedFp6Bytes(kM, false), 0);
            mx_tile128x4::placeFp6Data(placed.data(), rawA.data(), kM, kK, false);
            out.nearA = std::move(placed);
        }
        {
            std::vector<uint8_t> placed(mx_tile128x4::placedFp6Bytes(kN, true), 0);
            mx_tile128x4::placeFp6Data(placed.data(), rawB.data(), kN, kK, true);
            out.nearB = std::move(placed);
        }
        out.nearScaleA = mx_tile128x4::placeScales(scalesA, kKBlocks);
        out.scaleB     = mx_tile128x4::shuffleScales(scalesB, kN, kKBlocks);
        out.groupedA.resize(packed);
        out.groupedB.resize(packed);
        mx_tile128x4::tileFp6Data(
            out.groupedA.data(), rawA.data(), kM, kK, mx_tile128x4::TileOrder::P48);
        mx_tile128x4::tileFp6Data(
            out.groupedB.data(), rawB.data(), kN, kK, mx_tile128x4::TileOrder::P48);
        out.groupedScaleA = mx_tile128x4::shuffleScales(scalesA, kM, kKBlocks);
        out.planarA.resize(packed);
        mx_tile128x4::tileFp6Data(
            out.planarA.data(), rawA.data(), kM, kK, mx_tile128x4::TileOrder::P16);
        out.planarB = std::move(rawB);
        return out;
    }

    struct Timed
    {
        double                us = -1;
        std::string           name;
        std::vector<uint16_t> d;
    };

    // One GEMM, timed as the mean of kHot launches after kCold warmup launches.
    // The clock is the host clock around the queued launches, with no flush.
    Timed timeGemm(hipblasLtHandle_t            handle,
                   hipStream_t                  stream,
                   void*                        workspace,
                   void*                        a,
                   void*                        b,
                   void*                        scaleA,
                   void*                        scaleB,
                   void*                        d,
                   hipblasLtOrder_t             orderA,
                   hipblasLtOrder_t             orderB,
                   hipblasLtMatmulMatrixScale_t scaleAMode,
                   hipblasLtMatmulMatrixScale_t scaleBMode,
                   hipDataType                  type = HIP_R_6F_E2M3,
                   int                          cold = kCold,
                   int                          hot  = kHot,
                   bool                         copyOut = true)
    {
        Timed out;
        hipblaslt_ext::GemmProblemType problem(HIPBLAS_OP_T,
                                               HIPBLAS_OP_N,
                                               type,
                                               type,
                                               HIP_R_16BF,
                                               HIP_R_16BF,
                                               HIPBLAS_COMPUTE_32F);
        problem.setOrderA(orderA);
        problem.setOrderB(orderB);
        hipblaslt_ext::GemmEpilogue epilogue;
        epilogue.setScalingAType(scaleAMode);
        epilogue.setScalingBType(scaleBMode);
        hipblaslt_ext::GemmInputs inputs;
        inputs.setA(a);
        inputs.setB(b);
        inputs.setC(d);
        inputs.setD(d);
        inputs.setAlpha(&kAlpha);
        inputs.setBeta(&kBeta);
        inputs.setScaleA(scaleA);
        inputs.setScaleB(scaleB);

        hipblaslt_ext::Gemm gemm(handle,
                                 HIPBLAS_OP_T,
                                 HIPBLAS_OP_N,
                                 type,
                                 type,
                                 HIP_R_16BF,
                                 HIP_R_16BF,
                                 HIPBLAS_COMPUTE_32F);
        hipblaslt_ext::GemmPreference preference;
        preference.setMaxWorkspaceBytes(kWorkspace);
        preference.setUniformSummationOrder(1);
        std::vector<hipblasLtMatmulHeuristicResult_t> picks;
        if(gemm.setProblem(kM,
                           kN,
                           kK,
                           1,
                           kK,
                           kK,
                           kM,
                           kM,
                           kK * kM,
                           kK * kN,
                           kM * kN,
                           kM * kN,
                           epilogue,
                           inputs,
                           problem)
               != HIPBLAS_STATUS_SUCCESS
           || gemm.algoGetHeuristic(1, preference, picks) != HIPBLAS_STATUS_SUCCESS
           || picks.empty())
        {
            ADD_FAILURE() << "no GEMM for orders " << int(orderA) << " and " << int(orderB);
            return out;
        }
        out.name = hipblaslt_ext::getSolutionNameFromAlgo(handle, picks[0].algo);
        size_t workspaceBytes = 0;
        if(gemm.isAlgoSupported(picks[0].algo, workspaceBytes) != HIPBLAS_STATUS_SUCCESS
           || workspaceBytes > kWorkspace
           || gemm.initialize(picks[0].algo, workspace) != HIPBLAS_STATUS_SUCCESS)
        {
            ADD_FAILURE() << out.name << " could not be initialized";
            return out;
        }
        for(int i = 0; i < cold; ++i)
        {
            if(gemm.run(stream) != HIPBLAS_STATUS_SUCCESS)
            {
                ADD_FAILURE() << out.name << " failed during warmup";
                return out;
            }
        }
        if(hipStreamSynchronize(stream) != hipSuccess)
        {
            ADD_FAILURE() << out.name << " warmup did not finish";
            return out;
        }
        auto const t0 = std::chrono::steady_clock::now();
        for(int i = 0; i < hot; ++i)
        {
            if(gemm.run(stream) != HIPBLAS_STATUS_SUCCESS)
            {
                ADD_FAILURE() << out.name << " failed during timing";
                return out;
            }
        }
        if(hipStreamSynchronize(stream) != hipSuccess)
        {
            ADD_FAILURE() << out.name << " timed run did not finish";
            return out;
        }
        out.us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                     .count()
                 / hot;
        if(!copyOut)
            return out;
        out.d.resize(static_cast<size_t>(kM * kN));
        if(hipMemcpy(out.d.data(), d, out.d.size() * sizeof(uint16_t), hipMemcpyDeviceToHost)
           != hipSuccess)
            ADD_FAILURE() << out.name << " result copy failed";
        return out;
    }

    float bf16ToFloat(uint16_t bits)
    {
        uint32_t wide  = static_cast<uint32_t>(bits) << 16;
        float    value = 0.f;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    }

    int mismatchCount(std::vector<uint16_t> const& x, std::vector<uint16_t> const& y)
    {
        int count = 0;
        for(size_t i = 0; i < x.size(); ++i)
            count += x[i] != y[i];
        return count;
    }

    // 56 points against a double dot of the dequantized inputs. Checking every
    // output on the host is a full GEMM of this shape.
    void checkSample(char const*                 typeName,
                     char const*                 kernel,
                     std::vector<float> const&   a,
                     std::vector<float> const&   b,
                     std::vector<uint16_t> const& d)
    {
        int64_t const rows[] = {0, 1, 16, 31, 128, 255, 1024, 4095};
        int64_t const cols[] = {0, 1, 17, 64, 256, 1023, 4095};
        double        maxRel = 0;
        int           nans   = 0;
        for(int64_t m : rows)
            for(int64_t n : cols)
            {
                double dot = 0;
                for(int64_t k = 0; k < kK; ++k)
                    dot += static_cast<double>(a[static_cast<size_t>(k + m * kK)])
                           * static_cast<double>(b[static_cast<size_t>(k + n * kK)]);
                float const got = bf16ToFloat(d[static_cast<size_t>(m + n * kM)]);
                if(!std::isfinite(got) || !std::isfinite(dot))
                    ++nans;
                double const denom = std::max(std::abs(dot), 1.0);
                maxRel = std::max(maxRel, std::abs(static_cast<double>(got) - dot) / denom);
            }
        EXPECT_EQ(nans, 0) << typeName << " " << kernel;
        // bf16 keeps about three decimal digits, so a long fp32 sum still fits.
        EXPECT_LT(maxRel, 1e-2) << typeName << " " << kernel << " max relative error " << maxRel;
    }
} // namespace

// One launch is enough: the check is that the contiguous allocation and an
// ordinary buffer return the same result and select the expected kernels.
TEST(stress_near_die_arena, SameResultAsOrdinaryAllocation)
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count < 1)
        GTEST_SKIP() << "No GPU available";
    int device = 0;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, device), hipSuccess);
    if(std::string(prop.gcnArchName).rfind("gfx950", 0) != 0)
        GTEST_SKIP() << "The near-die kernels are registered for gfx950 only";
    int chip = 0;
    ASSERT_EQ(hipDeviceGetAttribute(&chip, hipDeviceAttributePciChipId, device), hipSuccess);
    if(chip != 0x75a0 && chip != 0x75a3)
        GTEST_SKIP() << "The near-die kernels are registered for chips 0x75a0 and 0x75a3 only";
    if(!driverSelectsNearDie())
        GTEST_SKIP() << "The near-die kernels are selected on amdgpu 6.19 and newer";
    int64_t const used = vramUsed(prop);
    if(used < 0)
        GTEST_SKIP() << "Could not read device memory in use";
    if(used >= kCleanBelow)
        GTEST_SKIP() << "Device memory in use is " << (used >> 20)
                     << " MiB; the contiguous allocation needs under 2 GiB";

    Arena arena;
    if(hipExtMallocWithFlags(&arena.raw, static_cast<size_t>(kArenaBytes), hipDeviceMallocContiguous)
           != hipSuccess
       || !arena.raw)
        GTEST_SKIP() << "The contiguous allocation failed";
    if(reinterpret_cast<uintptr_t>(arena.raw) % (2u << 20) != 0)
    {
        GTEST_SKIP() << "The contiguous allocation is not aligned to 2 MiB";
    }
    arena.base = static_cast<char*>(arena.raw);

    Images const images = makeImages(HIP_R_6F_E2M3);
    char* const  slotA  = arena.slot(static_cast<int64_t>(images.nearA.size()));
    char* const  slotB  = arena.slot(static_cast<int64_t>(images.nearB.size()));
    char* const  slotSA = arena.slot(static_cast<int64_t>(images.nearScaleA.size()));
    char* const  slotSB = arena.slot(static_cast<int64_t>(images.scaleB.size()));
    ASSERT_NE(slotA, nullptr);
    ASSERT_NE(slotB, nullptr);
    ASSERT_NE(slotSA, nullptr);
    ASSERT_NE(slotSB, nullptr);

    Owned ordinary;
    void* plainA  = ordinary.alloc(images.nearA.size());
    void* plainB  = ordinary.alloc(images.nearB.size());
    void* plainSA = ordinary.alloc(images.nearScaleA.size());
    void* plainSB = ordinary.alloc(images.scaleB.size());
    void* groupA  = ordinary.alloc(images.groupedA.size());
    void* groupB  = ordinary.alloc(images.groupedB.size());
    void* groupSA = ordinary.alloc(images.groupedScaleA.size());
    void* groupSB = ordinary.alloc(images.scaleB.size());
    void* dNear   = ordinary.alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
    void* dPlain  = ordinary.alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
    void* dGroup  = ordinary.alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
    void* ws      = ordinary.alloc(kWorkspace);
    if(!plainA || !plainB || !plainSA || !plainSB || !groupA || !groupB || !groupSA || !groupSB
       || !dNear || !dPlain || !dGroup || !ws)
        GTEST_SKIP() << "An ordinary device allocation failed";

    ASSERT_TRUE(upload(slotA, images.nearA.data(), images.nearA.size()));
    ASSERT_TRUE(upload(slotB, images.nearB.data(), images.nearB.size()));
    ASSERT_TRUE(upload(slotSA, images.nearScaleA.data(), images.nearScaleA.size()));
    ASSERT_TRUE(upload(slotSB, images.scaleB.data(), images.scaleB.size()));
    ASSERT_TRUE(upload(plainA, images.nearA.data(), images.nearA.size()));
    ASSERT_TRUE(upload(plainB, images.nearB.data(), images.nearB.size()));
    ASSERT_TRUE(upload(plainSA, images.nearScaleA.data(), images.nearScaleA.size()));
    ASSERT_TRUE(upload(plainSB, images.scaleB.data(), images.scaleB.size()));
    ASSERT_TRUE(upload(groupA, images.groupedA.data(), images.groupedA.size()));
    ASSERT_TRUE(upload(groupB, images.groupedB.data(), images.groupedB.size()));
    ASSERT_TRUE(upload(groupSA, images.groupedScaleA.data(), images.groupedScaleA.size()));
    ASSERT_TRUE(upload(groupSB, images.scaleB.data(), images.scaleB.size()));

    hipblasLtHandle_t handle = nullptr;
    hipStream_t       stream = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    struct Guard
    {
        hipblasLtHandle_t handle;
        hipStream_t       stream;
        ~Guard()
        {
            if(stream)
                static_cast<void>(hipStreamDestroy(stream));
            if(handle)
                hipblasLtDestroy(handle);
        }
    } guard{handle, stream};

    auto const tile = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT;
    auto const placedScale = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT;
    Timed const near        = timeGemm(handle,
                              stream,
                              ws,
                              slotA,
                              slotB,
                              slotSA,
                              slotSB,
                              dNear,
                              HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT,
                              HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT,
                              placedScale,
                              tile,
                              HIP_R_6F_E2M3,
                              0,
                              1);
    Timed const plain       = timeGemm(handle,
                               stream,
                               ws,
                               plainA,
                               plainB,
                               plainSA,
                               plainSB,
                               dPlain,
                               HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT,
                               HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT,
                               placedScale,
                               tile,
                               HIP_R_6F_E2M3,
                               0,
                               1);
    Timed const grouped     = timeGemm(handle,
                                 stream,
                                 ws,
                                 groupA,
                                 groupB,
                                 groupSA,
                                 groupSB,
                                 dGroup,
                                 HIPBLASLT_ORDER_COL256_R128_P48_EXT,
                                 HIPBLASLT_ORDER_COL256_R128_P48_EXT,
                                 tile,
                                 tile,
                                 HIP_R_6F_E2M3,
                                 0,
                                 1);
    if(HasFailure())
        return;

    EXPECT_NE(near.name.find("ka941ce3864"), std::string::npos) << near.name;
    EXPECT_NE(plain.name.find("ka941ce3864"), std::string::npos) << plain.name;
    EXPECT_NE(grouped.name.find("k067cdb88d0"), std::string::npos) << grouped.name;
    ASSERT_EQ(near.d.size(), plain.d.size());
    auto const mismatch = std::mismatch(near.d.begin(), near.d.end(), plain.d.begin());
    EXPECT_EQ(mismatch.first, near.d.end())
        << "contiguous and ordinary results differ at index "
        << (mismatch.first - near.d.begin());
    EXPECT_GT(std::count_if(near.d.begin(),
                            near.d.end(),
                            [](uint16_t v) { return v != 0; }),
              0)
        << "the result is all zeros";
}

// 1000 cold and 1000 hot launches, host clock around the batch, no flush. The
// near-die kernel's operands sit in the contiguous allocation. The other two
// kernels use ordinary allocations. Both 6-bit types.
TEST(stress_near_die_arena, KernelTimes)
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count < 1)
        GTEST_SKIP() << "No GPU available";
    int device = 0;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, device), hipSuccess);
    if(std::string(prop.gcnArchName).rfind("gfx950", 0) != 0)
        GTEST_SKIP() << "The near-die kernels are registered for gfx950 only";
    int chip = 0;
    ASSERT_EQ(hipDeviceGetAttribute(&chip, hipDeviceAttributePciChipId, device), hipSuccess);
    if(chip != 0x75a0 && chip != 0x75a3)
        GTEST_SKIP() << "The near-die kernels are registered for chips 0x75a0 and 0x75a3 only";
    if(!driverSelectsNearDie())
        GTEST_SKIP() << "The near-die kernels are selected on amdgpu 6.19 and newer";
    int64_t const used = vramUsed(prop);
    if(used < 0)
        GTEST_SKIP() << "Could not read device memory in use";
    if(used >= kCleanBelow)
        GTEST_SKIP() << "Device memory in use is " << (used >> 20)
                     << " MiB; the contiguous allocation needs under 2 GiB";

    Arena arena;
    if(hipExtMallocWithFlags(&arena.raw, static_cast<size_t>(kArenaBytes), hipDeviceMallocContiguous)
           != hipSuccess
       || !arena.raw)
        GTEST_SKIP() << "The contiguous allocation failed";
    if(reinterpret_cast<uintptr_t>(arena.raw) % (2u << 20) != 0)
        GTEST_SKIP() << "The contiguous allocation is not aligned to 2 MiB";
    arena.base = static_cast<char*>(arena.raw);

    Images images = makeImages(HIP_R_6F_E2M3);
    char* const  slotA  = arena.slot(static_cast<int64_t>(images.nearA.size()));
    char* const  slotB  = arena.slot(static_cast<int64_t>(images.nearB.size()));
    char* const  slotSA = arena.slot(static_cast<int64_t>(images.nearScaleA.size()));
    char* const  slotSB = arena.slot(static_cast<int64_t>(images.scaleB.size()));
    ASSERT_NE(slotA, nullptr);
    ASSERT_NE(slotB, nullptr);
    ASSERT_NE(slotSA, nullptr);
    ASSERT_NE(slotSB, nullptr);

    Owned ordinary;
    void* groupA  = ordinary.alloc(images.groupedA.size());
    void* groupB  = ordinary.alloc(images.groupedB.size());
    void* groupSA = ordinary.alloc(images.groupedScaleA.size());
    void* groupSB = ordinary.alloc(images.scaleB.size());
    void* planA   = ordinary.alloc(images.planarA.size());
    void* planB   = ordinary.alloc(images.planarB.size());
    void* dGroup  = ordinary.alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
    void* dPlan   = ordinary.alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
    void* dNear   = ordinary.alloc(static_cast<size_t>(kM * kN) * sizeof(uint16_t));
    void* ws      = ordinary.alloc(kWorkspace);
    if(!groupA || !groupB || !groupSA || !groupSB || !planA || !planB || !dGroup || !dPlan
       || !dNear || !ws)
        GTEST_SKIP() << "An ordinary device allocation failed";

    auto uploadAll = [&](Images const& img) {
        ASSERT_TRUE(upload(slotA, img.nearA.data(), img.nearA.size()));
        ASSERT_TRUE(upload(slotB, img.nearB.data(), img.nearB.size()));
        ASSERT_TRUE(upload(slotSA, img.nearScaleA.data(), img.nearScaleA.size()));
        ASSERT_TRUE(upload(slotSB, img.scaleB.data(), img.scaleB.size()));
        ASSERT_TRUE(upload(groupA, img.groupedA.data(), img.groupedA.size()));
        ASSERT_TRUE(upload(groupB, img.groupedB.data(), img.groupedB.size()));
        ASSERT_TRUE(upload(groupSA, img.groupedScaleA.data(), img.groupedScaleA.size()));
        ASSERT_TRUE(upload(groupSB, img.scaleB.data(), img.scaleB.size()));
        ASSERT_TRUE(upload(planA, img.planarA.data(), img.planarA.size()));
        ASSERT_TRUE(upload(planB, img.planarB.data(), img.planarB.size()));
    };

    hipblasLtHandle_t handle = nullptr;
    hipStream_t       stream = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    struct Guard
    {
        hipblasLtHandle_t handle;
        hipStream_t       stream;
        ~Guard()
        {
            if(stream)
                static_cast<void>(hipStreamDestroy(stream));
            if(handle)
                hipblasLtDestroy(handle);
        }
    } guard{handle, stream};

    auto const tile        = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT;
    auto const placedScale = HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT;
    constexpr int kTableCold = 1000;
    constexpr int kTableHot  = 1000;

    std::string driver = "unknown";
    {
        std::ifstream in("/sys/module/amdgpu/version");
        std::getline(in, driver);
    }
    std::printf("amdgpu %s  chip 0x%x  cold %d  hot %d  trig_float  no flush\n",
                driver.c_str(),
                chip,
                kTableCold,
                kTableHot);
    std::printf("%-12s %-6s %6s %6s %6s %7s %7s %7s %7s %9s %8s\n",
                "kernel",
                "type",
                "M",
                "N",
                "K",
                "orderA",
                "orderB",
                "scaleA",
                "scaleB",
                "time_us",
                "tflops");

    auto report = [&](char const*  typeName,
                      char const*  expect,
                      int          orderA,
                      char const*  orderB,
                      int          scaleA,
                      int          scaleB,
                      Timed const& timed) {
        EXPECT_NE(timed.name.find(expect), std::string::npos) << timed.name;
        double const tflops = 2.0 * kM * kN * kK / (timed.us * 1e-6) / 1e12;
        char const*  stem   = "kernel";
        if(timed.name.find("k067cdb88d0") != std::string::npos)
            stem = "k067cdb88d0";
        else if(timed.name.find("kd2977a1791") != std::string::npos)
            stem = "kd2977a1791";
        else if(timed.name.find("ka941ce3864") != std::string::npos)
            stem = "ka941ce3864";
        std::printf("%-12s %-6s %6lld %6lld %6lld %7d %7s %7d %7d %9.3f %8.1f\n",
                    stem,
                    typeName,
                    static_cast<long long>(kM),
                    static_cast<long long>(kN),
                    static_cast<long long>(kK),
                    orderA,
                    orderB,
                    scaleA,
                    scaleB,
                    timed.us,
                    tflops);
    };

    for(auto const type : {HIP_R_6F_E2M3, HIP_R_6F_E3M2})
    {
        if(type != HIP_R_6F_E2M3)
            images = makeImages(type);
        uploadAll(images);
        if(HasFailure())
            return;
        char const* const typeName = type == HIP_R_6F_E2M3 ? "f6_r" : "bf6_r";
        Timed const       grouped  = timeGemm(handle,
                                       stream,
                                       ws,
                                       groupA,
                                       groupB,
                                       groupSA,
                                       groupSB,
                                       dGroup,
                                       HIPBLASLT_ORDER_COL256_R128_P48_EXT,
                                       HIPBLASLT_ORDER_COL256_R128_P48_EXT,
                                       tile,
                                       tile,
                                       type,
                                       kTableCold,
                                       kTableHot,
                                       true);
        Timed const       planar   = timeGemm(handle,
                                      stream,
                                      ws,
                                      planA,
                                      planB,
                                      groupSA,
                                      groupSB,
                                      dPlan,
                                      HIPBLASLT_ORDER_COL256_R128_P16_EXT,
                                      HIPBLASLT_ORDER_COL,
                                      tile,
                                      tile,
                                      type,
                                      kTableCold,
                                      kTableHot,
                                      true);
        Timed const       near     = timeGemm(handle,
                                    stream,
                                    ws,
                                    slotA,
                                    slotB,
                                    slotSA,
                                    slotSB,
                                    dNear,
                                    HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT,
                                    HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT,
                                    placedScale,
                                    tile,
                                    type,
                                    kTableCold,
                                    kTableHot,
                                    true);
        if(HasFailure())
            return;
        report(typeName, "k067cdb88d0", 105, "105", 1018, 1018, grouped);
        report(typeName, "kd2977a1791", 106, "column", 1018, 1018, planar);
        report(typeName, "ka941ce3864", 107, "108", 1019, 1018, near);
        EXPECT_EQ(mismatchCount(grouped.d, planar.d), 0) << typeName;
        EXPECT_EQ(mismatchCount(grouped.d, near.d), 0) << typeName;
        ASSERT_EQ(images.refA.size(), static_cast<size_t>(kM * kK));
        ASSERT_EQ(images.refB.size(), static_cast<size_t>(kN * kK));
        checkSample(typeName, "48-byte", images.refA, images.refB, grouped.d);
        checkSample(typeName, "16-byte", images.refA, images.refB, planar.d);
        checkSample(typeName, "near-die", images.refA, images.refB, near.d);
    }
}
