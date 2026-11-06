// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "AttentionGeneralBwdGpuCases.generated.hpp"
#include "ScopedPluginLogCapture.hpp"

/**
 * @file AttentionGeneralBwdGpuSuite.hpp
 * @brief The device suite of hipkernel:AttentionGeneralBwd, shared by the per-arch test files.
 *
 * Each case builds its SdpaBackward graph through the frontend, plans it on the engine only and
 * runs it on random Q, K, V and dO, with O and the stats from hipDNN's SDPA forward reference.
 * dQ, dK and dV are compared with hipDNN's backward reference (the GPU reference,
 * GpuFpReferenceSdpa::bprop, itself checked against CpuFpReferenceSdpa::backward). A THD graph is
 * run as its padded dense equivalent with per-sequence lengths, which is how the reference graph
 * executors run ragged SDPA. On a device with the arch's full compute-unit count, the kernels each
 * slot selects and the workspace query must be the ones the engine's Python reference predicts
 * (AttentionGeneralBwdGpuCases.generated.hpp).
 *
 * Set HIPDNN_TEST_EXPECTED_ARCH=gfx942 (or gfx950) on required runs so a missing or different
 * device fails instead of skipping.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration::general_bwd_gpu
{

inline constexpr const char* ENGINE = "hipkernel:AttentionGeneralBwd";

/// Output bits no kernel writes: a NaN in fp16, a large finite value in bf16; compared bitwise.
inline constexpr uint16_t SENTINEL = 0x7e00;

enum Uid : int64_t
{
    Q = 1,
    K = 2,
    V = 3,
    O = 4,
    DO = 5,
    STATS = 6,
    DQ = 7,
    DK = 8,
    DV = 9,
    OFFSETS_Q = 10,
    OFFSETS_KV = 11
};

/// The eight data operands.
enum class Role
{
    Q,
    K,
    V,
    O,
    DO,
    DQ,
    DK,
    DV
};

inline constexpr std::array<Role, 8> ROLES
    = {Role::Q, Role::K, Role::V, Role::O, Role::DO, Role::DQ, Role::DK, Role::DV};

inline int64_t uidOf(Role role)
{
    constexpr std::array<int64_t, 8> UIDS = {Q, K, V, O, DO, DQ, DK, DV};
    return UIDS[static_cast<size_t>(role)];
}

inline bool isQuerySide(Role role)
{
    return role == Role::Q || role == Role::O || role == Role::DO || role == Role::DQ;
}

inline int64_t headsOf(const GpuCase& c, Role role)
{
    if(isQuerySide(role))
    {
        return c.hQ;
    }
    return role == Role::K || role == Role::DK ? c.hK : c.hV;
}

inline int64_t seqOf(const GpuCase& c, Role role)
{
    return isQuerySide(role) ? c.sQ : c.sKv;
}

inline uint16_t toBits(float x, bool bf16)
{
    if(bf16)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &x, 4);
        bits += 0x7fffU + ((bits >> 16) & 1U);
        return static_cast<uint16_t>(bits >> 16);
    }
    const auto half = static_cast<_Float16>(x);
    uint16_t bits = 0;
    std::memcpy(&bits, &half, 2);
    return bits;
}

inline float fromBits(uint16_t bits, bool bf16)
{
    if(bf16)
    {
        const uint32_t wide = static_cast<uint32_t>(bits) << 16;
        float value = 0;
        std::memcpy(&value, &wide, 4);
        return value;
    }
    _Float16 half = 0;
    std::memcpy(&half, &bits, 2);
    return static_cast<float>(half);
}

/// The graph's element strides of @p role: BHSD, or token-major (BSHD; THD with B ignored).
inline std::vector<int64_t> stridesOf(const GpuCase& c, Role role)
{
    const int64_t h = headsOf(c, role);
    const int64_t s = seqOf(c, role);
    if(c.layout == Layout::BHSD)
    {
        return {h * s * c.d, s * c.d, c.d, 1};
    }
    return {s * h * c.d, c.d, h * c.d, 1};
}

/// Per-sequence lengths and the token offsets of the two ragged tables. A batched graph has every
/// length at its maximum and no tables.
struct Lengths
{
    std::vector<int32_t> q;
    std::vector<int32_t> kv;
    std::vector<int32_t> offsetsQ;
    std::vector<int32_t> offsetsKv;
};

inline bool masked(const GpuCase& c)
{
    return c.leftBound >= 0 || c.rightBound >= 0;
}

inline void setOffsets(Lengths& lengths)
{
    const auto prefix = [](const std::vector<int32_t>& lens) {
        std::vector<int32_t> out(lens.size() + 1, 0);
        for(size_t b = 0; b < lens.size(); ++b)
        {
            out[b + 1] = out[b] + lens[b];
        }
        return out;
    };
    lengths.offsetsQ = prefix(lengths.q);
    lengths.offsetsKv = prefix(lengths.kv);
}

/// Random lengths within the graph's maxima. Sequence 0 takes both maxima. Under a mask every
/// query row keeps a visible key: no sequence has more query rows than keys.
inline Lengths makeLengths(const GpuCase& c, uint32_t seed)
{
    Lengths lengths;
    lengths.q.assign(static_cast<size_t>(c.b), static_cast<int32_t>(c.sQ));
    lengths.kv.assign(static_cast<size_t>(c.b), static_cast<int32_t>(c.sKv));
    if(c.layout == Layout::THD)
    {
        std::mt19937 rng(seed);
        for(int64_t b = 0; b < c.b; ++b)
        {
            const auto i = static_cast<size_t>(b);
            if(b > 0)
            {
                std::uniform_int_distribution<int64_t> kv(std::max<int64_t>(1, c.sKv / 2), c.sKv);
                lengths.kv[i] = static_cast<int32_t>(kv(rng));
                std::uniform_int_distribution<int64_t> q(1, c.sQ);
                lengths.q[i] = static_cast<int32_t>(q(rng));
            }
            if(masked(c))
            {
                lengths.q[i] = std::min(lengths.q[i], lengths.kv[i]);
            }
        }
    }
    setOffsets(lengths);
    return lengths;
}

/// Storage elements of a data operand: B * S_max tokens for THD (rows past the last offset stay
/// unaddressed).
inline size_t elementsOf(const GpuCase& c, Role role)
{
    return static_cast<size_t>(c.b * headsOf(c, role) * seqOf(c, role) * c.d);
}

/// Storage index of element (b, h, s, i) of @p role.
inline size_t indexOf(
    const GpuCase& c, const Lengths& lengths, Role role, int64_t b, int64_t h, int64_t s, int64_t i)
{
    if(c.layout == Layout::THD)
    {
        const auto& offsets = isQuerySide(role) ? lengths.offsetsQ : lengths.offsetsKv;
        const int64_t token = offsets[static_cast<size_t>(b)] + s;
        return static_cast<size_t>((token * headsOf(c, role) + h) * c.d + i);
    }
    const auto st = stridesOf(c, role);
    return static_cast<size_t>(b * st[0] + h * st[1] + s * st[2] + i);
}

inline std::vector<int64_t> statsStrides(const GpuCase& c)
{
    switch(c.stats)
    {
    case Stats::RAGGED:
        return {c.sQ * c.hQ, 1, c.hQ, 1};
    case Stats::RAGGED_HEAD_MAJOR:
        return {c.sQ, c.b * c.sQ, 1, 1};
    case Stats::DENSE:
    default:
        return {c.hQ * c.sQ, c.sQ, 1, 1};
    }
}

inline size_t statsIndex(const GpuCase& c, const Lengths& lengths, int64_t b, int64_t h, int64_t s)
{
    const auto token = static_cast<int64_t>(lengths.offsetsQ[static_cast<size_t>(b)]) + s;
    switch(c.stats)
    {
    case Stats::RAGGED:
        return static_cast<size_t>(token * c.hQ + h);
    case Stats::RAGGED_HEAD_MAJOR:
        return static_cast<size_t>(h * c.b * c.sQ + token);
    case Stats::DENSE:
    default:
        return static_cast<size_t>((b * c.hQ + h) * c.sQ + s);
    }
}

/// Host data of one problem instance, in the device layout.
struct HostState
{
    Lengths lengths;
    std::array<std::vector<uint16_t>, 8> data; ///< by Role; outputs hold the engine's result
    std::vector<float> stats;
    /// The reference gradients (dQ, dK, dV) in the device layout; only addressed rows are set.
    std::array<std::vector<float>, 3> expected;
};

/// Random Q, K, V and dO for @p lengths; O, the stats and the expected gradients come from the
/// reference.
inline HostState makeState(const GpuCase& c, Lengths lengths, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    HostState s;
    s.lengths = std::move(lengths);
    for(const Role role : ROLES)
    {
        auto& v = s.data[static_cast<size_t>(role)];
        v.assign(elementsOf(c, role), SENTINEL);
        if(role == Role::Q || role == Role::K || role == Role::V || role == Role::DO)
        {
            for(auto& x : v)
            {
                x = toBits(uniform(rng), c.bf16);
            }
        }
    }
    s.stats.assign(static_cast<size_t>(c.b * c.hQ * c.sQ), 0.0F);
    return s;
}

/// The reference on the padded dense problem: forward for O and the stats (written into @p s in
/// the device layout), then backward for the expected gradients.
template <typename T>
void referenceAs(const GpuCase& c, HostState& s)
{
    using hipdnn_data_sdk::utilities::Tensor;
    const auto& lens = s.lengths;
    const auto dense = [&](Role role) {
        return std::vector<int64_t>{c.b, headsOf(c, role), seqOf(c, role), c.d};
    };
    const auto load = [&](Role role) {
        auto t = std::make_unique<Tensor<T>>(dense(role));
        T* host = t->memory().hostData();
        std::fill(host, host + t->elementCount(), static_cast<T>(0.0F));
        const auto& src = s.data[static_cast<size_t>(role)];
        const int64_t heads = headsOf(c, role);
        const int64_t seq = seqOf(c, role);
        for(int64_t b = 0; b < c.b; ++b)
        {
            const auto len = isQuerySide(role) ? lens.q[static_cast<size_t>(b)]
                                               : lens.kv[static_cast<size_t>(b)];
            for(int64_t h = 0; h < heads; ++h)
            {
                for(int64_t row = 0; row < len; ++row)
                {
                    for(int64_t i = 0; i < c.d; ++i)
                    {
                        host[((b * heads + h) * seq + row) * c.d + i] = static_cast<T>(
                            fromBits(src[indexOf(c, lens, role, b, h, row, i)], c.bf16));
                    }
                }
            }
        }
        t->memory().markHostModified();
        return t;
    };
    auto q = load(Role::Q);
    auto k = load(Role::K);
    auto v = load(Role::V);
    auto dO = load(Role::DO);
    Tensor<T> o(dense(Role::O));
    Tensor<float> lse({c.b, c.hQ, c.sQ});
    Tensor<int32_t> seqLenQ({c.b, 1, 1, 1});
    Tensor<int32_t> seqLenKv({c.b, 1, 1, 1});
    std::copy(lens.q.begin(), lens.q.end(), seqLenQ.memory().hostData());
    std::copy(lens.kv.begin(), lens.kv.end(), seqLenKv.memory().hostData());
    seqLenQ.memory().markHostModified();
    seqLenKv.memory().markHostModified();
    const std::optional<float> scale
        = c.scale < 0.0F ? std::nullopt : std::optional<float>(c.scale);
    const bool padded = c.layout == Layout::THD;

    auto fwd = hipdnn_gpu_ref::GpuSdpaFwdOptions<float>{}
                   .setScale(scale)
                   .setBand(c.leftBound, c.rightBound, !c.bottomRight)
                   .setLse(&lse);
    auto bwd = hipdnn_gpu_ref::GpuSdpaBwdOptions<float>{}
                   .setScale(scale)
                   .setBand(c.leftBound, c.rightBound, !c.bottomRight)
                   .setLse(&lse);
    if(padded)
    {
        fwd.setSeqLens(&seqLenQ, &seqLenKv);
        bwd.setSeqLens(&seqLenQ, &seqLenKv);
    }
    hipdnn_gpu_ref::GpuFpReferenceSdpa::fprop<T, T, T, T>(*q, *k, *v, o, fwd);

    // The engine reads O and the stats as stored: write them back in the device layout.
    const T* oHost = o.memory().hostData();
    const float* lseHost = lse.memory().hostData();
    auto& oDst = s.data[static_cast<size_t>(Role::O)];
    for(int64_t b = 0; b < c.b; ++b)
    {
        for(int64_t h = 0; h < c.hQ; ++h)
        {
            for(int64_t row = 0; row < lens.q[static_cast<size_t>(b)]; ++row)
            {
                s.stats[statsIndex(c, lens, b, h, row)] = lseHost[(b * c.hQ + h) * c.sQ + row];
                for(int64_t i = 0; i < c.d; ++i)
                {
                    oDst[indexOf(c, lens, Role::O, b, h, row, i)] = toBits(
                        static_cast<float>(oHost[((b * c.hQ + h) * c.sQ + row) * c.d + i]), c.bf16);
                }
            }
        }
    }

    Tensor<float> dq(dense(Role::DQ));
    Tensor<float> dk(dense(Role::DK));
    Tensor<float> dv(dense(Role::DV));
    hipdnn_gpu_ref::GpuFpReferenceSdpa::bprop<T, T, T, T, T, float, float, float>(
        *q, *k, *v, o, *dO, dq, dk, dv, bwd);

    const std::array<std::pair<Role, Tensor<float>*>, 3> grads
        = {{{Role::DQ, &dq}, {Role::DK, &dk}, {Role::DV, &dv}}};
    for(size_t g = 0; g < grads.size(); ++g)
    {
        const Role role = grads[g].first;
        const float* src = grads[g].second->memory().hostData();
        auto& dst = s.expected[g];
        dst.assign(elementsOf(c, role), 0.0F);
        const int64_t heads = headsOf(c, role);
        const int64_t seq = seqOf(c, role);
        for(int64_t b = 0; b < c.b; ++b)
        {
            const auto len = isQuerySide(role) ? lens.q[static_cast<size_t>(b)]
                                               : lens.kv[static_cast<size_t>(b)];
            for(int64_t h = 0; h < heads; ++h)
            {
                for(int64_t row = 0; row < len; ++row)
                {
                    for(int64_t i = 0; i < c.d; ++i)
                    {
                        dst[indexOf(c, lens, role, b, h, row, i)]
                            = src[((b * heads + h) * seq + row) * c.d + i];
                    }
                }
            }
        }
    }
}

inline void reference(const GpuCase& c, HostState& s)
{
    if(c.bf16)
    {
        referenceAs<hipdnn_data_sdk::types::bfloat16>(c, s);
    }
    else
    {
        referenceAs<hipdnn_data_sdk::types::half>(c, s);
    }
}

class DeviceBuffers
{
public:
    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
    DeviceBuffers(DeviceBuffers&&) = delete;
    DeviceBuffers& operator=(DeviceBuffers&&) = delete;
    ~DeviceBuffers()
    {
        for(void* p : _allocations)
        {
            EXPECT_EQ(hipFree(p), hipSuccess);
        }
    }
    void* allocate(size_t bytes)
    {
        void* p = nullptr;
        if(hipMalloc(&p, std::max<size_t>(bytes, 256)) != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed");
        }
        _allocations.push_back(p);
        return p;
    }

private:
    std::vector<void*> _allocations;
};

/// The planned graph and its device buffers.
struct Built
{
    std::unique_ptr<hipdnn_frontend::graph::Graph> graph;
    DeviceBuffers memory;
    std::unordered_map<int64_t, void*> buffers;
    void* workspace = nullptr;
    int64_t workspaceBytes = 0;
};

/// The tolerance of a gradient element: relative to the tensor's largest reference magnitude
/// and to the element's own. The kernels round P and dS to the I/O type before their GEMMs and
/// accumulate in fp32.
struct Tolerance
{
    float scaleOfMax;
    float relative;
    /// The smallest magnitude scaleOfMax applies to: a gradient whose reference is all zero
    /// (e.g. one visible key per row, so dS = 0) still admits the kernels' rounding noise.
    float floor;
};

inline Tolerance toleranceFor(bool bf16)
{
    Tolerance tolerance{};
    tolerance.scaleOfMax = bf16 ? 4.0e-2F : 1.5e-2F;
    tolerance.relative = tolerance.scaleOfMax;
    tolerance.floor = 1.0e-3F;
    return tolerance;
}

/// Fixture of one arch's suite. @p Arch names the arch, its full compute-unit count and its
/// cases.
template <typename Arch>
class AttentionGeneralBwdGpu : public ::testing::Test
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int64_t _computeUnits = 0;

    void SetUp() override
    {
        const auto expectedArch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        if(expectedArch.empty())
        {
            SKIP_IF_NO_DEVICES();
        }
        int device = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        hipDeviceProp_t props{};
        ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
        const std::string arch = props.gcnArchName;
        const auto base = arch.substr(0, arch.find(':'));
        if(!expectedArch.empty() && expectedArch == Arch::ARCH)
        {
            ASSERT_EQ(base, expectedArch);
        }
        if(base != Arch::ARCH)
        {
            GTEST_SKIP() << "the " << Arch::ARCH << " suite of " << ENGINE << "; found " << arch;
        }
        _computeUnits = props.multiProcessorCount;
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        const auto pluginPath
            = std::filesystem::weakly_canonical(
                  hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH)
                  .string();
        const char* path = pluginPath.c_str();
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(1, &path, HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        }
        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            EXPECT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    static bool ok(const hipdnn_frontend::Error& error)
    {
        EXPECT_EQ(error.code, hipdnn_frontend::ErrorCode::OK) << error.err_msg;
        return error.code == hipdnn_frontend::ErrorCode::OK;
    }

    /// Whether the device has the compute-unit count the cases' kernel expectations assume.
    bool fullDevice() const
    {
        return _computeUnits == Arch::FULL_CUS;
    }

    /// The SdpaBackward graph of @p c, operationally built (not planned).
    static std::unique_ptr<hipdnn_frontend::graph::Graph> makeGraph(const GpuCase& c)
    {
        using namespace hipdnn_frontend;
        using namespace hipdnn_frontend::graph;
        const DataType io = c.bf16 ? DataType::BFLOAT16 : DataType::HALF;
        auto graph = std::make_unique<Graph>();
        graph->set_io_data_type(io)
            .set_compute_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT);
        const auto make = [](int64_t uid,
                             DataType type,
                             const std::vector<int64_t>& dims,
                             const std::vector<int64_t>& strides) {
            return std::make_shared<TensorAttributes>(
                TensorAttributes().set_uid(uid).set_data_type(type).set_dim(dims).set_stride(
                    strides));
        };
        const bool thd = c.layout == Layout::THD;
        std::shared_ptr<TensorAttributes> offsetsQ;
        std::shared_ptr<TensorAttributes> offsetsKv;
        if(thd)
        {
            offsetsQ = make(OFFSETS_Q, DataType::INT32, {c.b + 1, 1, 1, 1}, {1, 1, 1, 1});
            offsetsKv = make(OFFSETS_KV, DataType::INT32, {c.b + 1, 1, 1, 1}, {1, 1, 1, 1});
        }
        const auto operand = [&](Role role, const std::shared_ptr<TensorAttributes>& t) {
            const auto st = stridesOf(c, role);
            t->set_uid(uidOf(role))
                .set_data_type(io)
                .set_dim({c.b, headsOf(c, role), seqOf(c, role), c.d})
                .set_stride(st)
                .set_alignment(c.alignment);
            if(thd)
            {
                t->set_ragged_offset(isQuerySide(role) ? offsetsQ : offsetsKv)
                    .set_ragged_offset_multiplier(st[2]);
            }
            return t;
        };
        std::array<std::shared_ptr<TensorAttributes>, 5> in;
        for(const Role role : {Role::Q, Role::K, Role::V, Role::O, Role::DO})
        {
            in[static_cast<size_t>(role)] = operand(role, std::make_shared<TensorAttributes>());
        }
        auto stats = make(STATS, DataType::FLOAT, {c.b, c.hQ, c.sQ, 1}, statsStrides(c));
        if(c.stats != Stats::DENSE)
        {
            stats->set_ragged_offset(offsetsQ).set_ragged_offset_multiplier(
                c.stats == Stats::RAGGED ? c.hQ : 1);
        }
        SdpaBackwardAttributes attributes;
        attributes.set_name("general_bwd");
        if(c.scale >= 0.0F)
        {
            attributes.set_attn_scale(c.scale);
        }
        if(c.leftBound >= 0)
        {
            attributes.set_diagonal_band_left_bound(c.leftBound);
        }
        if(c.rightBound >= 0)
        {
            attributes.set_diagonal_band_right_bound(c.rightBound);
        }
        if(c.bottomRight)
        {
            attributes.set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT);
        }
        auto [dq, dk, dv]
            = graph->sdpa_backward(in[0], in[1], in[2], in[3], in[4], stats, attributes);
        operand(Role::DQ, dq)->set_output(true);
        operand(Role::DK, dk)->set_output(true);
        operand(Role::DV, dv)->set_output(true);
        return graph;
    }

    /// Plans @p c on the engine only and allocates its buffers and workspace.
    void build(const GpuCase& c, Built& built)
    {
        built.graph = makeGraph(c);
        auto& graph = *built.graph;
        ASSERT_TRUE(ok(graph.validate()));
        ASSERT_TRUE(ok(graph.build_operation_graph(_handle)));
        ASSERT_TRUE(ok(graph.create_execution_plan_ext(
            hipdnn_data_sdk::utilities::engineNameToId(ENGINE), {})));
        ASSERT_TRUE(ok(graph.check_support()));
        ASSERT_TRUE(ok(graph.build_plans()));
        ASSERT_TRUE(ok(graph.get_workspace_size(built.workspaceBytes)));
        if(fullDevice())
        {
            EXPECT_EQ(built.workspaceBytes, c.workspaceBytes);
        }
        else
        {
            EXPECT_GT(built.workspaceBytes, 0);
        }
        for(const Role role : ROLES)
        {
            built.buffers[uidOf(role)] = built.memory.allocate(elementsOf(c, role) * 2);
        }
        built.buffers[STATS]
            = built.memory.allocate(static_cast<size_t>(c.b * c.hQ * c.sQ) * sizeof(float));
        if(c.layout == Layout::THD)
        {
            const auto bytes = static_cast<size_t>(c.b + 1) * sizeof(int32_t);
            built.buffers[OFFSETS_Q] = built.memory.allocate(bytes);
            built.buffers[OFFSETS_KV] = built.memory.allocate(bytes);
        }
        built.workspace = built.memory.allocate(
            static_cast<size_t>(std::max<int64_t>(built.workspaceBytes, 1)));
    }

    /// Uploads @p s (stream-ordered), with the outputs and the workspace poisoned, so the
    /// comparison sees only what this execute wrote and the plan zeroes what it accumulates into.
    void upload(const GpuCase& c, const HostState& s, Built& built)
    {
        const auto copy = [&](void* dst, const void* src, size_t bytes) {
            ASSERT_EQ(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, _stream), hipSuccess);
        };
        for(const Role role : ROLES)
        {
            const auto& v = s.data[static_cast<size_t>(role)];
            copy(built.buffers.at(uidOf(role)), v.data(), v.size() * 2);
        }
        copy(built.buffers.at(STATS), s.stats.data(), s.stats.size() * sizeof(float));
        if(c.layout == Layout::THD)
        {
            copy(built.buffers.at(OFFSETS_Q),
                 s.lengths.offsetsQ.data(),
                 s.lengths.offsetsQ.size() * sizeof(int32_t));
            copy(built.buffers.at(OFFSETS_KV),
                 s.lengths.offsetsKv.data(),
                 s.lengths.offsetsKv.size() * sizeof(int32_t));
        }
        ASSERT_EQ(hipMemsetAsync(built.workspace,
                                 0xff,
                                 static_cast<size_t>(std::max<int64_t>(built.workspaceBytes, 1)),
                                 _stream),
                  hipSuccess);
    }

    /// Reads dQ, dK and dV back and compares every addressed element with the reference; every
    /// unaddressed THD token keeps the sentinel.
    void check(const GpuCase& c, const HostState& s, const Built& built)
    {
        const auto tol = toleranceFor(c.bf16);
        const std::array<Role, 3> grads = {Role::DQ, Role::DK, Role::DV};
        for(size_t g = 0; g < grads.size(); ++g)
        {
            const Role role = grads[g];
            constexpr std::array<const char*, 3> NAMES = {"dQ", "dK", "dV"};
            const char* name = NAMES[g];
            std::vector<uint16_t> out(elementsOf(c, role));
            ASSERT_EQ(hipMemcpyAsync(out.data(),
                                     built.buffers.at(uidOf(role)),
                                     out.size() * 2,
                                     hipMemcpyDeviceToHost,
                                     _stream),
                      hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
            const auto& want = s.expected[g];
            std::vector<bool> addressed(out.size(), false);
            const int64_t heads = headsOf(c, role);
            float largest = 0.0F;
            for(int64_t b = 0; b < c.b; ++b)
            {
                const auto len = isQuerySide(role) ? s.lengths.q[static_cast<size_t>(b)]
                                                   : s.lengths.kv[static_cast<size_t>(b)];
                for(int64_t h = 0; h < heads; ++h)
                {
                    for(int64_t row = 0; row < len; ++row)
                    {
                        for(int64_t i = 0; i < c.d; ++i)
                        {
                            const auto at = indexOf(c, s.lengths, role, b, h, row, i);
                            addressed[at] = true;
                            largest = std::max(largest, std::abs(want[at]));
                        }
                    }
                }
            }
            float worst = 0.0F;
            size_t worstAt = 0;
            for(size_t at = 0; at < out.size(); ++at)
            {
                if(!addressed[at])
                {
                    ASSERT_EQ(out[at], SENTINEL)
                        << name << ": the engine wrote unaddressed element " << at;
                    continue;
                }
                const float actual = fromBits(out[at], c.bf16);
                ASSERT_TRUE(std::isfinite(actual)) << name << " element " << at;
                const float bound = tol.scaleOfMax * std::max(largest, tol.floor)
                                    + tol.relative * std::abs(want[at]);
                const float ratio = std::abs(actual - want[at]) / bound;
                if(ratio > worst)
                {
                    worst = ratio;
                    worstAt = at;
                }
            }
            std::cout << "GBWD_ERR " << c.name << " " << name << " " << worst * tol.scaleOfMax
                      << '\n';
            EXPECT_LE(worst, 1.0F)
                << name << ": element " << worstAt << " is " << fromBits(out[worstAt], c.bf16)
                << ", reference " << want[worstAt] << " (largest |reference| " << largest << ")";
        }
    }

    hipdnn_frontend::Error execute(Built& built)
    {
        return built.graph->execute(_handle, built.buffers, built.workspace);
    }

    /// Checks that each slot selected the kernel the engine's Python reference predicts.
    void
        expectSelectedKernels(const GpuCase& c,
                              const hipdnn_test_sdk::utilities::IsolatedLogRecorder& recorder) const
    {
        if(!fullDevice())
        {
            std::cout << "GBWD_NOTE " << c.name << ": " << _computeUnits
                      << " compute units, not the " << Arch::FULL_CUS
                      << " the expected kernels assume; only correctness is checked\n";
            return;
        }
        for(const auto& [slot, id] : {std::pair{"prep", c.prepKernelId}, {"main", c.mainKernelId}})
        {
            const auto line = std::string("engine '") + ENGINE + "' slot '" + slot
                              + "' selected kernel " + std::string(id);
            EXPECT_TRUE(recorder.hasLogContaining(line))
                << "expected " << line << " (" << c.mainKernel << "). Captured logs:\n"
                << recorder.getRecordedLogsAsString();
        }
    }

    /// Builds, runs and checks one accepted case.
    void runAccepted(const GpuCase& c)
    {
        const ScopedPluginLogCapture capture(this);
        Built built;
        build(c, built);
        if(::testing::Test::HasFatalFailure())
        {
            std::cout << capture.recorder().getRecordedLogsAsString() << '\n';
            return;
        }
        expectSelectedKernels(c, capture.recorder());
        auto state = makeState(c, makeLengths(c, 7), 1);
        reference(c, state);
        upload(c, state, built);
        ASSERT_TRUE(ok(execute(built)));
        check(c, state, built);
    }

    /// A declined case is not offered, and the engine logs why.
    void runDeclined(const GpuCase& c)
    {
        const ScopedPluginLogCapture capture(this);
        auto graph = makeGraph(c);
        ASSERT_TRUE(ok(graph->validate()));
        ASSERT_TRUE(ok(graph->build_operation_graph(_handle)));
        std::vector<int64_t> ranked;
        const auto result = graph->get_ranked_engine_ids(ranked);
        const auto id = hipdnn_data_sdk::utilities::engineNameToId(ENGINE);
        EXPECT_TRUE(result.code != hipdnn_frontend::ErrorCode::OK
                    || std::find(ranked.begin(), ranked.end(), id) == ranked.end())
            << ENGINE << " offered to serve " << c.name;
        const auto line = std::string(ENGINE) + " declined the graph [" + std::string(c.code) + ":";
        EXPECT_TRUE(capture.recorder().hasLogContaining(line))
            << "expected '" << line << "'. Captured logs:\n"
            << capture.recorder().getRecordedLogsAsString();
    }

    void runCase(const GpuCase& c)
    {
        if(c.code == "OK")
        {
            runAccepted(c);
        }
        else
        {
            runDeclined(c);
        }
    }

    /// One plan captured once into a HIP graph and replayed three times, each with new data and,
    /// for THD, new lengths (each within the graph's maxima) in the same device offset tables.
    /// The capture holds exactly the plan's kernel launches: nothing allocates, copies or
    /// synchronizes inside execute.
    void captureAndReplay(const GpuCase& c)
    {
        SCOPED_TRACE(std::string(c.name));
        Built built;
        build(c, built);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        auto state = makeState(c, makeLengths(c, 3), 11);
        reference(c, state);
        upload(c, state, built);
        ASSERT_TRUE(ok(execute(built)));
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        check(c, state, built);

        hipGraph_t graph = nullptr;
        ASSERT_EQ(hipStreamBeginCapture(_stream, hipStreamCaptureModeThreadLocal), hipSuccess);
        const auto status = execute(built);
        hipStreamCaptureStatus capturing = hipStreamCaptureStatusNone;
        const auto query = hipStreamIsCapturing(_stream, &capturing);
        ASSERT_EQ(hipStreamEndCapture(_stream, &graph), hipSuccess);
        ASSERT_TRUE(ok(status));
        ASSERT_EQ(query, hipSuccess);
        EXPECT_EQ(capturing, hipStreamCaptureStatusActive) << "execute invalidated the capture";
        size_t nodes = 0;
        ASSERT_EQ(hipGraphGetNodes(graph, nullptr, &nodes), hipSuccess);
        std::vector<hipGraphNode_t> all(nodes);
        ASSERT_EQ(hipGraphGetNodes(graph, all.data(), &nodes), hipSuccess);
        for(auto* node : all)
        {
            hipGraphNodeType type = hipGraphNodeTypeEmpty;
            ASSERT_EQ(hipGraphNodeGetType(node, &type), hipSuccess);
            EXPECT_EQ(type, hipGraphNodeTypeKernel) << "a captured node is not a kernel launch";
        }
        if(fullDevice())
        {
            EXPECT_EQ(static_cast<int64_t>(nodes), c.launches);
        }
        hipGraphExec_t exec = nullptr;
        ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);

        for(uint32_t step = 0; step < 3; ++step)
        {
            SCOPED_TRACE(step);
            state = makeState(c, makeLengths(c, 100 + step), 200 + step);
            reference(c, state);
            upload(c, state, built);
            ASSERT_EQ(hipGraphLaunch(exec, _stream), hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
            check(c, state, built);
        }
        ASSERT_EQ(hipGraphExecDestroy(exec), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
    }

    /// A data pointer below the alignment the graph declares makes execute fail cleanly.
    void misalignedPointerFails(const GpuCase& c)
    {
        ASSERT_GE(c.alignment, 4);
        Built built;
        build(c, built);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        auto state = makeState(c, makeLengths(c, 5), 5);
        reference(c, state);
        upload(c, state, built);
        auto misaligned = built.buffers;
        misaligned[Q] = static_cast<char*>(built.buffers.at(Q)) + 2;
        const auto status = built.graph->execute(_handle, misaligned, built.workspace);
        // The backend checks declared alignment before the engine sees the pointer.
        EXPECT_EQ(status.code, hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR)
            << "execute did not refuse a Q pointer 2 bytes past a " << c.alignment
            << "-byte declared alignment: " << status.err_msg;
        const auto refusal = "device pointer is not aligned to the required "
                             + std::to_string(c.alignment) + " bytes";
        EXPECT_NE(status.err_msg.find(refusal), std::string::npos)
            << "expected '" << refusal << "' in: " << status.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        // The plan still runs on aligned pointers.
        upload(c, state, built);
        ASSERT_TRUE(ok(execute(built)));
        check(c, state, built);
    }
};

/// The parameterised suite of one arch.
template <typename Arch>
class AttentionGeneralBwdGpuCases : public AttentionGeneralBwdGpu<Arch>,
                                    public ::testing::WithParamInterface<GpuCase>
{
};

/// The cases of @p cases in the quick tier (@p quick) or only in the full one.
template <size_t N>
std::vector<GpuCase> casesOfTier(const std::array<GpuCase, N>& cases, bool quick)
{
    std::vector<GpuCase> out;
    std::copy_if(cases.begin(), cases.end(), std::back_inserter(out), [quick](const GpuCase& c) {
        return c.quick == quick;
    });
    return out;
}

template <size_t N>
std::vector<GpuCase> casesTagged(const std::array<GpuCase, N>& cases, std::string_view tag)
{
    std::vector<GpuCase> out;
    std::copy_if(cases.begin(), cases.end(), std::back_inserter(out), [tag](const GpuCase& c) {
        return c.tags.find(tag) != std::string_view::npos;
    });
    return out;
}

inline std::string caseName(const ::testing::TestParamInfo<GpuCase>& info)
{
    return std::string(info.param.name);
}

/// hipDNN's GPU backward reference, which the suite compares against, agrees with its CPU
/// reference (CpuFpReferenceSdpa) on a small padded problem with a window and GQA.
inline void expectGpuReferenceMatchesCpu()
{
    using hipdnn_data_sdk::types::bfloat16;
    using hipdnn_data_sdk::utilities::Tensor;
    using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;
    const int64_t b = 2;
    const int64_t hq = 4;
    const int64_t hk = 2;
    const int64_t sq = 33;
    const int64_t skv = 47;
    const int64_t d = 32;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    const auto filled = [&](const std::vector<int64_t>& dims) {
        auto t = std::make_unique<Tensor<bfloat16>>(dims);
        bfloat16* host = t->memory().hostData();
        for(size_t i = 0; i < t->elementCount(); ++i)
        {
            host[i] = static_cast<bfloat16>(uniform(rng));
        }
        t->memory().markHostModified();
        return t;
    };
    auto q = filled({b, hq, sq, d});
    auto k = filled({b, hk, skv, d});
    auto v = filled({b, hk, skv, d});
    auto dO = filled({b, hq, sq, d});
    Tensor<int32_t> lenQ({b, 1, 1, 1});
    Tensor<int32_t> lenKv({b, 1, 1, 1});
    lenQ.memory().hostData()[0] = 33;
    lenQ.memory().hostData()[1] = 20;
    lenKv.memory().hostData()[0] = 47;
    lenKv.memory().hostData()[1] = 30;
    lenQ.memory().markHostModified();
    lenKv.memory().markHostModified();
    const float scale = 0.2F;

    Tensor<bfloat16> oGpu({b, hq, sq, d});
    Tensor<float> lseGpu({b, hq, sq});
    hipdnn_gpu_ref::GpuFpReferenceSdpa::fprop<bfloat16, bfloat16, bfloat16, bfloat16>(
        *q,
        *k,
        *v,
        oGpu,
        hipdnn_gpu_ref::GpuSdpaFwdOptions<float>{}
            .setScale(scale)
            .setBand(9, 3, false)
            .setSeqLens(&lenQ, &lenKv)
            .setLse(&lseGpu));
    Tensor<float> dqGpu({b, hq, sq, d});
    Tensor<float> dkGpu({b, hk, skv, d});
    Tensor<float> dvGpu({b, hk, skv, d});
    hipdnn_gpu_ref::GpuFpReferenceSdpa::
        bprop<bfloat16, bfloat16, bfloat16, bfloat16, bfloat16, float, float, float>(
            *q,
            *k,
            *v,
            oGpu,
            *dO,
            dqGpu,
            dkGpu,
            dvGpu,
            hipdnn_gpu_ref::GpuSdpaBwdOptions<float>{}
                .setScale(scale)
                .setBand(9, 3, false)
                .setSeqLens(&lenQ, &lenKv)
                .setLse(&lseGpu));

    Tensor<bfloat16> oCpu({b, hq, sq, d});
    Tensor<float> lseCpu({b, hq, sq, 1});
    Tensor<float> dqCpu({b, hq, sq, d});
    Tensor<float> dkCpu({b, hk, skv, d});
    Tensor<float> dvCpu({b, hk, skv, d});
    CpuFpReferenceSdpa::forward<bfloat16, bfloat16, bfloat16, bfloat16>(
        *q,
        *k,
        *v,
        oCpu,
        hipdnn_test_sdk::utilities::SdpaFwdOptions<float>{}
            .setScale(scale)
            .setBand(9, 3, false)
            .setSeqLens(&lenQ, &lenKv)
            .setLse(&lseCpu));
    {
        const bfloat16* g = oGpu.memory().hostData();
        const bfloat16* r = oCpu.memory().hostData();
        for(size_t i = 0; i < oGpu.elementCount(); ++i)
        {
            const auto want = static_cast<float>(r[i]);
            ASSERT_NEAR(static_cast<float>(g[i]), want, 1.0e-3F + 8.0e-3F * std::abs(want))
                << "O element " << i;
        }
    }
    // Both backwards read the same O, so they differ only in summation order.
    CpuFpReferenceSdpa::
        backward<bfloat16, bfloat16, bfloat16, bfloat16, bfloat16, float, float, float>(
            *q,
            *k,
            *v,
            oGpu,
            *dO,
            dqCpu,
            dkCpu,
            dvCpu,
            hipdnn_test_sdk::utilities::SdpaBwdOptions<float>{}
                .setScale(scale)
                .setBand(9, 3, false)
                .setSeqLens(&lenQ, &lenKv)
                .setLse(&lseCpu));
    const std::array<std::pair<Tensor<float>*, Tensor<float>*>, 3> pairs
        = {{{&dqGpu, &dqCpu}, {&dkGpu, &dkCpu}, {&dvGpu, &dvCpu}}};
    for(const auto& [gpu, cpu] : pairs)
    {
        const float* g = gpu->memory().hostData();
        const float* r = cpu->memory().hostData();
        for(size_t i = 0; i < gpu->elementCount(); ++i)
        {
            ASSERT_NEAR(g[i], r[i], 1.0e-4F + 1.0e-3F * std::abs(r[i])) << "element " << i;
        }
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration::general_bwd_gpu
