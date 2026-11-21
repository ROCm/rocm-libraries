// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests of the gfx1250 batched transpose kernel instances (batched_transpose.cpp). The kernels are
// launched directly (the same arguments and grid as BatchedTransposeSolution), so that every
// instance is covered for every shape, independently of the kernel selection heuristic. The
// instances only do something on gfx1250, the tests are skipped on other devices.

#include <gtest/gtest.h>

#include <miopen/batched_transpose_sol.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/handle.hpp>
#include <miopen/invoker.hpp>
#include <miopen/magic_div.hpp>

#include "../get_handle.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t block_size = 256; // BATCHED_TRANSPOSE_BLOCK_SIZE

// Instances added for gfx1250 in batched_transpose.cpp; keep in sync with the file.
// name: <tile width>x<tile height>_pack_<load>x<store>_ediv_<width divisor>x<height divisor>
struct Instance
{
    const char* name;
    uint32_t data_size; // 2: half / bfloat16 (kernel suffix _half), 4: float (suffix _dword)
    uint32_t tile_x;
    uint32_t tile_y;
    uint32_t ediv_x;
    uint32_t ediv_y;
    bool buffer; // buffer instructions: height * width * data size has to be below 2^31

    std::string KernelName() const
    {
        return std::string("batched_transpose_") + name + (data_size == 2 ? "_half" : "_dword");
    }
};

// Printed in the test listing (ctest names the tests after it); without it gtest prints the raw
// bytes of the object, which contain a pointer that changes from run to run.
void PrintTo(const Instance& inst, std::ostream* os) { *os << inst.name; }

const std::vector<Instance>& Instances()
{
    static const std::vector<Instance> instances = {
        // 2-byte, register
        {"32x64_pack_1x8_ediv_1x8", 2, 32, 64, 1, 8, false},
        {"128x128_pack_8x8_ediv_8x8", 2, 128, 128, 8, 8, true},
        {"128x64_pack_8x8_ediv_8x8", 2, 128, 64, 8, 8, true},
        // 2-byte, LDS
        {"64x128_pack_8x4_ediv_8x4", 2, 64, 128, 8, 4, true},
        // 2-byte, LDS, thin matrices (small channel count, one position per thread)
        {"256x32_pack_1x8_ediv_1x32", 2, 256, 32, 1, 32, false},
        {"512x64_pack_1x8_ediv_1x64", 2, 512, 64, 1, 64, false},
        {"32x512_pack_8x1_ediv_32x1", 2, 32, 512, 32, 1, false},
        // 4-byte
        {"64x64_pack_4x4_ediv_4x4", 4, 64, 64, 4, 4, false},
        {"64x128_pack_4x4_ediv_4x4", 4, 64, 128, 4, 4, false},
        {"64x32_pack_4x4_ediv_4x4", 4, 64, 32, 4, 4, true},
        {"32x256_pack_4x2_ediv_4x2", 4, 32, 256, 4, 2, true},
        {"16x32_pack_1x4_ediv_1x4", 4, 16, 32, 1, 4, true},
    };
    return instances;
}

// Sizes of one dimension that are multiples of div: smaller than, equal to, not a multiple of and
// larger than typical tiles, and some that are odd / prime multiples of div.
std::vector<uint32_t> Dims(uint32_t div)
{
    std::vector<uint32_t> v;
    for(uint32_t k : {1u, 2u, 3u, 4u, 5u, 7u, 9u, 16u, 17u, 33u, 64u, 65u})
        v.push_back(div * k);
    for(uint32_t d : {8u, 31u, 100u, 129u, 257u, 1369u, 22201u}) // 22201 = 149 * 149
        if(d % div == 0)
            v.push_back(d);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

template <typename T>
T Pattern(uint64_t i)
{
    // cheap hash of the position, every element differs from most of its neighbours
    uint64_t x = i * 0x9E3779B97F4A7C15ull;
    x ^= x >> 29;
    return static_cast<T>(x);
}

struct Mismatch
{
    uint64_t b, w, h;
    uint64_t got, expected;
};

// Compares the complete output with the expected transpose: dst[b][w][h] == src[b][h][w] for every
// element of every batch. Returns the first mismatch found, if any. Even the largest tensors that
// the buffer instructions can address (2^31 bytes) are verified completely: the work is split in
// blocks of `wblock` columns that are checked by all hardware threads. Inside a block the source is
// read along the rows (contiguous) and the destination along `wblock` columns at the same time
// (`wblock` streams that each advance sequentially), which keeps the strided accesses cache
// friendly.
template <typename T>
std::optional<Mismatch> VerifyTranspose(const std::vector<T>& src,
                                        const std::vector<T>& dst,
                                        uint32_t batch,
                                        uint32_t height,
                                        uint32_t width)
{
    constexpr uint32_t wblock = 64;
    const uint64_t wblocks    = (width + wblock - 1) / wblock;
    const uint64_t items      = static_cast<uint64_t>(batch) * wblocks;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> found{false};
    std::mutex m;
    std::optional<Mismatch> first;

    const auto worker = [&] {
        for(uint64_t it = next++; it < items && !found; it = next++)
        {
            const uint64_t b  = it / wblocks;
            const uint64_t w0 = (it % wblocks) * wblock;
            const uint64_t w1 = std::min<uint64_t>(w0 + wblock, width);
            for(uint64_t h = 0; h < height; h++)
                for(uint64_t w = w0; w < w1; w++)
                {
                    const T expected = src[(b * height + h) * width + w];
                    const T got      = dst[(b * width + w) * height + h];
                    if(got != expected)
                    {
                        std::lock_guard<std::mutex> lock(m);
                        if(!first)
                            first = Mismatch{b, w, h, got, expected};
                        found = true;
                        return;
                    }
                }
        }
    };
    std::vector<std::thread> threads;
    for(unsigned i = 0, t = std::max(1u, std::thread::hardware_concurrency()); i < t; i++)
        threads.emplace_back(worker);
    for(auto& t : threads)
        t.join();
    return first;
}

// Runs the instance on a batch x height x width tensor (src[b][h][w] -> dst[b][w][h]). The test
// does two independent checks:
//   1. Correctness: every element of the output is compared with the expected transposed value
//      (also for the largest tensors, see VerifyTranspose).
//   2. No out-of-bounds writes: nothing is written behind the end of the output (see `guard`
//      below). This is a negative check, it does not influence the result of the transpose.
template <typename T>
void RunInstance(const Instance& inst, uint32_t batch, uint32_t height, uint32_t width)
{
    ASSERT_EQ(sizeof(T), inst.data_size);
    ASSERT_EQ(width % inst.ediv_x, 0u);
    ASSERT_EQ(height % inst.ediv_y, 0u);

    auto&& handle = get_handle();
    if(handle.GetDeviceName() != "gfx1250")
        GTEST_SKIP() << "gfx1250 only";

    const uint64_t n = static_cast<uint64_t>(batch) * height * width;
    // The kernels must not write outside of the n output elements (tiles that stick out of the
    // tensor are masked or skipped, out of range buffer stores are dropped by the hardware). To
    // check that, the output buffer gets `guard` extra elements behind the end: a write there is
    // harmless (it stays inside the allocation) and visible afterwards. With an output of exactly
    // n elements the overshoot would usually go unnoticed: device allocations are rounded up to
    // whole pages, so a write behind the end does not fault, it would only corrupt the padding.
    // The whole output buffer (also the first n elements) is filled with `sentinel` before the
    // launch, so that elements that the kernel did not write also show up as mismatches.
    constexpr uint64_t guard = 1u << 16;
    const T sentinel         = static_cast<T>(0xABABABABu);

    std::vector<T> src(n);
    for(uint64_t i = 0; i < n; i++)
        src[i] = Pattern<T>(i);
    std::vector<T> dst(n + guard, sentinel);

    auto src_dev = handle.Write(src);
    auto dst_dev = handle.Write(dst);

    // the arguments and the grid of BatchedTransposeSolution
    const uint32_t dim_h     = (height + inst.tile_y - 1) / inst.tile_y;
    const uint32_t dim_w     = (width + inst.tile_x - 1) / inst.tile_x;
    const uint64_t grid_size = static_cast<uint64_t>(batch) * dim_h * dim_w;
    ASSERT_LE(grid_size, 0xffffffffull);
    const miopen::magic_div_u32_t magic_h = miopen::magic_div_u32_gen(dim_h);
    const miopen::magic_div_u32_t magic_w = miopen::magic_div_u32_gen(dim_w);

    std::vector<OpKernelArg> args;
    args.emplace_back(0); // dst, src: set at the invocation
    args.emplace_back(0);
    args.emplace_back(height);
    args.emplace_back(width);
    args.emplace_back(static_cast<uint32_t>(grid_size)); // dim_stride
    args.emplace_back(static_cast<uint32_t>(grid_size)); // dim_total
    args.emplace_back(magic_h.magic);
    args.emplace_back(static_cast<uint32_t>(magic_h.shift));
    args.emplace_back(magic_w.magic);
    args.emplace_back(static_cast<uint32_t>(magic_w.shift));

    miopen::solver::KernelInfo kernel;
    kernel.kernel_file = "batched_transpose.cpp";
    kernel.kernel_name = inst.KernelName();
    kernel.g_wk        = {static_cast<size_t>(grid_size) * block_size, 1, 1};
    kernel.l_wk        = {block_size, 1, 1};

    miopen::InvokerFactory factory([=](const std::vector<miopen::Kernel>& kernels) mutable {
        return
            [=](const miopen::Handle& h, const miopen::AnyInvokeParams& primitive_param) mutable {
                decltype(auto) invoke_params = primitive_param.CastTo<transpose_invoke_param>();
                const auto k                 = h.Run(kernels[0]);
                args[0]                      = OpKernelArg(invoke_params.dst);
                args[1]                      = OpKernelArg(invoke_params.src);
                k(args);
            };
    });

    miopen::ExecutionContext ctx;
    ctx.SetStream(&handle);
    auto invoker = ctx.GetStream().PrepareInvoker(factory, {kernel});
    invoker(handle, transpose_invoke_param{src_dev.get(), dst_dev.get()});

    dst = handle.Read<T>(dst_dev, n + guard);

    const std::string where = std::string(inst.name) + " batch " + std::to_string(batch) +
                              " height " + std::to_string(height) + " width " +
                              std::to_string(width);

    // out-of-bounds check: the guard elements behind the output must still hold the sentinel
    for(uint64_t i = 0; i < guard; i++)
        ASSERT_EQ(dst[n + i], sentinel) << "write behind the output at +" << i << ": " << where;

    const auto bad = VerifyTranspose(src, dst, batch, height, width);
    if(bad)
        FAIL() << "mismatch at (b " << bad->b << ", w " << bad->w << ", h " << bad->h << "): got "
               << static_cast<uint64_t>(bad->got) << ", expected "
               << static_cast<uint64_t>(bad->expected) << ": " << where;
}

void RunInstanceTyped(const Instance& inst, uint32_t batch, uint32_t height, uint32_t width)
{
    if(inst.data_size == 2)
        RunInstance<uint16_t>(inst, batch, height, width);
    else
        RunInstance<uint32_t>(inst, batch, height, width);
}

std::vector<Instance> InstancesOfSize(uint32_t data_size, bool buffer_only)
{
    std::vector<Instance> v;
    for(const auto& inst : Instances())
        if(inst.data_size == data_size && (inst.buffer || !buffer_only))
            v.push_back(inst);
    return v;
}

// Shapes around the tile sizes, odd sizes and several batches
void RunShapes(const Instance& inst)
{
    for(uint32_t batch : {1u, 3u})
    {
        for(uint32_t height : Dims(inst.ediv_y))
        {
            for(uint32_t width : Dims(inst.ediv_x))
            {
                if(static_cast<uint64_t>(batch) * height * width > (4u << 20))
                    continue;
                RunInstanceTyped(inst, batch, height, width);
                if(::testing::Test::HasFatalFailure() || ::testing::Test::IsSkipped())
                    return;
            }
        }
    }
}

// Largest tensor per batch that the buffer instructions can address (height * width * data size
// < 2^31), long (or tall) in one dimension: the byte offsets get close to 2^31, a wrong offset type
// or out of range handling would show up here.
void RunLargestBufferRange(const Instance& inst, bool tall)
{
    ASSERT_TRUE(inst.buffer);
    const uint64_t max_elems = (uint64_t{1} << 31) / inst.data_size - 1;
    uint32_t height          = std::max(64u, inst.ediv_y) / inst.ediv_y * inst.ediv_y;
    uint32_t width           = std::max(64u, inst.ediv_x) / inst.ediv_x * inst.ediv_x;
    if(tall)
        height = static_cast<uint32_t>(max_elems / width / inst.ediv_y * inst.ediv_y);
    else
        width = static_cast<uint32_t>(max_elems / height / inst.ediv_x * inst.ediv_x);
    ASSERT_LT(static_cast<uint64_t>(height) * width * inst.data_size, uint64_t{1} << 31);
    RunInstanceTyped(inst, 1, height, width);
}

std::string InstanceName(const ::testing::TestParamInfo<Instance>& info) { return info.param.name; }

} // namespace

// The test suites are per data size: 2-byte instances (half / bfloat16 kernels, FP16) and 4-byte
// instances (float / int32 kernels, FP32). The buffer range tests only run the instances that use
// buffer instructions.
class GPU_BatchedTransposeKernels_FP16 : public ::testing::TestWithParam<Instance>
{
};
TEST_P(GPU_BatchedTransposeKernels_FP16, Shapes) { RunShapes(GetParam()); }
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BatchedTransposeKernels_FP16,
                         ::testing::ValuesIn(InstancesOfSize(2, false)),
                         InstanceName);

class GPU_BatchedTransposeKernelsBufferRange_FP16 : public ::testing::TestWithParam<Instance>
{
};
TEST_P(GPU_BatchedTransposeKernelsBufferRange_FP16, Long)
{
    RunLargestBufferRange(GetParam(), false);
}
TEST_P(GPU_BatchedTransposeKernelsBufferRange_FP16, Tall)
{
    RunLargestBufferRange(GetParam(), true);
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BatchedTransposeKernelsBufferRange_FP16,
                         ::testing::ValuesIn(InstancesOfSize(2, true)),
                         InstanceName);

class GPU_BatchedTransposeKernels_FP32 : public ::testing::TestWithParam<Instance>
{
};
TEST_P(GPU_BatchedTransposeKernels_FP32, Shapes) { RunShapes(GetParam()); }
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BatchedTransposeKernels_FP32,
                         ::testing::ValuesIn(InstancesOfSize(4, false)),
                         InstanceName);

class GPU_BatchedTransposeKernelsBufferRange_FP32 : public ::testing::TestWithParam<Instance>
{
};
TEST_P(GPU_BatchedTransposeKernelsBufferRange_FP32, Long)
{
    RunLargestBufferRange(GetParam(), false);
}
TEST_P(GPU_BatchedTransposeKernelsBufferRange_FP32, Tall)
{
    RunLargestBufferRange(GetParam(), true);
}
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_BatchedTransposeKernelsBufferRange_FP32,
                         ::testing::ValuesIn(InstancesOfSize(4, true)),
                         InstanceName);
