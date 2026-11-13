// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdCatalog.generated.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdConstants.generated.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdMatch.hpp"

/**
 * @file AttentionGeneralBwdGeometry.hpp
 * @brief The plan of `hipkernel:AttentionGeneralBwd`: tuning key, catalog row, runtime plan
 *        values, workspace layout, grids, blocks and every kernarg of the prep, main, convert
 *        and split-dQ launches (ABI `rocke.attn_bwd.v3`).
 *
 * A port of rocKE's `attn_bwd_plan` (general route) over the generated catalog: the tuned
 * table entry and the class pick one catalog row, and every value the plan binds at launch is
 * recomputed here from the request, the arch traits and the device CU count. The replay
 * corpus generated from the Python plan is the oracle. Pure functions of the lowered request:
 * testable with no HIP context.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::general_bwd
{

using hipdnn_plugin_sdk::ingestor::BoundTokens;

/// What one architecture fixes about the plan.
struct ArchTraits
{
    std::string_view arch;
    int64_t waveSize = 0;
};

constexpr ArchTraits archTraitsOf(std::string_view arch)
{
    for(const auto& facts : generated::ARCH_FACTS)
    {
        if(facts.arch == arch)
        {
            return {facts.arch, facts.waveSize};
        }
    }
    return {};
}

inline constexpr ArchTraits GFX942_TRAITS = archTraitsOf("gfx942");
inline constexpr ArchTraits GFX950_TRAITS = archTraitsOf("gfx950");
static_assert(GFX942_TRAITS.waveSize > 0 && GFX950_TRAITS.waveSize > 0);

/// The shipped traits of @p arch, or nullopt when the engine does not ship it.
inline std::optional<ArchTraits> shippedArchTraits(std::string_view arch)
{
    for(const auto shipped : generated::ENGINE_ARCHS)
    {
        if(shipped == arch)
        {
            return archTraitsOf(arch);
        }
    }
    return std::nullopt;
}

/// The request fields the plan reads, in Operand order. Strides are `(B, H, S, D)` element strides; under THD
/// the B stride is ignored and S is the token stride. `qMult` / `kvMult` are the element
/// multipliers of the Q and K/V offset tables (THD only).
struct PlanInput
{
    int64_t b = 0;
    int64_t hQ = 0;
    int64_t hK = 0;
    int64_t hV = 0;
    int64_t sQ = 0;
    int64_t sKv = 0;
    int64_t headSize = 0;
    generated::Dtype dtype = generated::Dtype::FP16;
    bool thd = false;
    std::array<std::array<int64_t, 4>, DATA_OPERANDS> strides{};
    std::vector<int64_t> lseDims;
    std::vector<int64_t> lseStrides;
    int64_t left = -1;
    int64_t right = -1;
    bool bottomRight = false;
    bool padding = false;
    int64_t seqLenStride = 1;
    bool offsets64 = false;
    int64_t qMult = 1;
    int64_t kvMult = 1;
    int64_t tensorAlignment = generated::DEFAULT_ALIGNMENT;
    std::optional<double> scale;
    std::optional<int64_t> numCus;
    /// THD workspace rows by storage token when both are set (hipDNN sets neither).
    std::optional<int64_t> maxTotalQ;
    std::optional<int64_t> maxTotalKv;

    const std::array<int64_t, 4>& stridesOf(Operand operand) const
    {
        return strides[index(operand)];
    }
};

struct Region
{
    int64_t offset = 0;
    int64_t bytes = 0;
};

/// The shared workspace; region keys are "lse2", "dsum", "dq_acc", "dk_acc", "dv_acc" and
/// "worklist". Main-slot scratch is empty, so the scratch offset is the shared size.
struct WorkspaceLayout
{
    int64_t bytes = 0;
    int64_t scratchOffset = 0;
    std::map<std::string, Region> regions;
};

/// Launch names: "" (the prep slot's single launch), "main", "dq", "convert_dk", "convert_dv".
enum class Launch : uint8_t
{
    PREP,
    MAIN,
    DQ,
    CONVERT_DK,
    CONVERT_DV,
};

constexpr std::string_view launchName(Launch launch)
{
    switch(launch)
    {
    case Launch::PREP:
        return "";
    case Launch::MAIN:
        return "main";
    case Launch::DQ:
        return "dq";
    case Launch::CONVERT_DK:
        return "convert_dk";
    case Launch::CONVERT_DV:
        return "convert_dv";
    default:
        return "";
    }
}

/// One kernarg in ABI order. `kind` is the ABI format letter (Q pointer, q i64, f f32, i i32).
/// A pointer carries its role (a tensor name, `ws.<region>`, or empty for a null pointer) and
/// bits 0; an integer its two's-complement value; an f32 its IEEE bit pattern.
struct Kernarg
{
    std::string_view name;
    char kind = 'i';
    uint64_t bits = 0;
    std::string_view role;

    int64_t intValue() const
    {
        return static_cast<int64_t>(bits);
    }
};

struct LaunchPlan
{
    Launch launch = Launch::PREP;
    std::string_view kernel;
    std::array<uint32_t, 3> grid{};
    uint32_t block = 0;
    std::vector<Kernarg> kernargs;
};

/// The runtime plan values bound at prepare.
struct RuntimeValues
{
    int64_t gSplit = 1;
    generated::LbOrder lbOrder = generated::LbOrder::NATURAL;
    int64_t pair = 0;
    int64_t packHeads = 0;
    bool useWorklist = false;
    int64_t scalePlacement = 0;
    double scale = 0.0;
    int64_t nKvTiles = 0;
};

struct Plan
{
    const generated::CatalogRow* row = nullptr;
    const generated::TunedKey* tunedKey = nullptr;
    RuntimeValues runtime;
    std::vector<LaunchPlan> launches;
    WorkspaceLayout workspace;
};

namespace detail
{

inline constexpr std::string_view ROLE_LSE2 = "ws.lse2";
inline constexpr std::string_view ROLE_DSUM = "ws.dsum";
inline constexpr std::string_view ROLE_DQ_ACC = "ws.dq_acc";
inline constexpr std::string_view ROLE_DK_ACC = "ws.dk_acc";
inline constexpr std::string_view ROLE_DV_ACC = "ws.dv_acc";
inline constexpr std::string_view ROLE_WORKLIST = "ws.worklist";

inline Decline unavailable(std::string detail)
{
    return Decline{generated::Reason::INSTANCE_UNAVAILABLE,
                   "no buildable instance: " + std::move(detail)};
}

inline int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

inline int64_t alignWorkspace(int64_t bytes)
{
    return ceilDiv(bytes, generated::WORKSPACE_ALIGN) * generated::WORKSPACE_ALIGN;
}

/// a * b, or nullopt when it leaves int64.
inline std::optional<int64_t> mul(int64_t a, int64_t b)
{
    int64_t out = 0;
    if(__builtin_mul_overflow(a, b, &out))
    {
        return std::nullopt;
    }
    return out;
}

inline uint32_t f32Bits(double value)
{
    const auto narrowed = static_cast<float>(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &narrowed, sizeof(bits));
    return bits;
}

inline int64_t headsOf(const PlanInput& in, Operand tensor)
{
    switch(tensor)
    {
    case Operand::K:
    case Operand::DK:
        return in.hK;
    case Operand::V:
    case Operand::DV:
        return in.hV;
    default:
        return in.hQ;
    }
}

inline int64_t seqOf(const PlanInput& in, Operand tensor)
{
    switch(tensor)
    {
    case Operand::Q:
    case Operand::O:
    case Operand::DO:
    case Operand::DQ:
        return in.sQ;
    default:
        return in.sKv;
    }
}

/// The B/H/S strides of every tensor that address more than one index.
inline std::vector<int64_t> usedStrides(const PlanInput& in)
{
    std::vector<int64_t> out;
    for(const auto tensor : DATA)
    {
        const auto& st = in.stridesOf(tensor);
        if(!in.thd && in.b > 1)
        {
            out.push_back(st[0]);
        }
        if(headsOf(in, tensor) > 1)
        {
            out.push_back(st[1]);
        }
        if(seqOf(in, tensor) > 1 || in.thd)
        {
            out.push_back(st[2]);
        }
    }
    return out;
}

/// THD rows indexed by storage token (both caller bounds given), else by sequence slot.
inline bool wsByToken(const PlanInput& in)
{
    return in.thd && in.maxTotalQ.has_value() && in.maxTotalKv.has_value();
}

/// Workspace rows (rows_q, rows_kv): B * S, or the caller's bounds when rows are tokens.
inline std::optional<std::pair<int64_t, int64_t>> rowsOf(const PlanInput& in)
{
    if(wsByToken(in))
    {
        return std::make_pair(*in.maxTotalQ, *in.maxTotalKv);
    }
    const auto rowsQ = mul(in.b, in.sQ);
    const auto rowsKv = mul(in.b, in.sKv);
    if(!rowsQ || !rowsKv)
    {
        return std::nullopt;
    }
    return std::make_pair(*rowsQ, *rowsKv);
}

inline int64_t gcdGroup(const PlanInput& in)
{
    const int64_t gk = in.hK != 0 ? in.hQ / in.hK : 0;
    const int64_t gv = in.hV != 0 ? in.hQ / in.hV : 0;
    return std::gcd(gk, gv);
}

/// Query heads per CTA: the K group in direct mode, the K/V common group in atomic mode.
inline int64_t groupOf(const PlanInput& in, bool atomicKv)
{
    if(atomicKv)
    {
        return gcdGroup(in);
    }
    return in.hK != 0 ? in.hQ / in.hK : 0;
}

/// Head units of the main grid's y axis.
inline int64_t unitsOf(const PlanInput& in, bool atomicKv)
{
    if(!atomicKv)
    {
        return in.hK;
    }
    const int64_t group = groupOf(in, true);
    return group != 0 ? in.hQ / group : 0;
}

/// Index into MASK_KINDS of the band (none, causal_tl, causal_br, window).
inline int64_t maskKindOf(const PlanInput& in)
{
    if(in.left == -1 && in.right == -1)
    {
        return 0;
    }
    if(in.left == -1 && in.right == 0)
    {
        return in.bottomRight ? 2 : 1;
    }
    return 3;
}

/// Index into HEAD_MODES (mha, gqa, hk_ne_hv).
inline int64_t headModeOf(const PlanInput& in)
{
    if(in.hK != in.hV)
    {
        return 2;
    }
    return in.hK != in.hQ ? 1 : 0;
}

/// Index into LENGTH_RELATIONS (square, unequal, s_q_1).
inline int64_t lengthRelationOf(const PlanInput& in)
{
    if(in.sQ == 1)
    {
        return 2;
    }
    return in.sQ == in.sKv ? 0 : 1;
}

/// Index into LENGTH_MODES (fixed, padded, thd).
inline int64_t lengthModeOf(const PlanInput& in)
{
    if(in.thd)
    {
        return 2;
    }
    return in.padding ? 1 : 0;
}

inline std::string kvTileBucket(int64_t tiles)
{
    for(const auto bound : generated::KV_TILE_BUCKETS)
    {
        if(tiles <= bound)
        {
            return "t" + std::to_string(bound);
        }
    }
    return "t>" + std::to_string(generated::KV_TILE_BUCKETS.back());
}

inline std::string_view fillBucket(const PlanInput& in, int64_t totalCtas)
{
    if(!in.numCus)
    {
        return generated::FILL_BUCKETS[3];
    }
    if(totalCtas < *in.numCus)
    {
        return generated::FILL_BUCKETS[0];
    }
    if(totalCtas < 2 * *in.numCus)
    {
        return generated::FILL_BUCKETS[1];
    }
    return generated::FILL_BUCKETS[2];
}

inline const generated::DefaultGeometry*
    defaultGeometry(std::string_view arch, const PlanInput& in, int64_t stageVec)
{
    const auto seq = in.thd ? generated::SeqMode::THD : generated::SeqMode::BATCHED;
    for(const auto& row : generated::DEFAULT_GEOMETRY)
    {
        if(row.arch == arch && row.headSize == in.headSize && row.dtype == in.dtype
           && row.stageVec == stageVec && row.seqMode == seq)
        {
            return &row;
        }
    }
    return nullptr;
}

template <std::size_t N>
inline int64_t indexOf(const std::array<std::string_view, N>& names, std::string_view name)
{
    for(std::size_t i = 0; i < N; ++i)
    {
        if(names[i] == name)
        {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

/// The multipliers a scale placement leaves to each kernel.
struct ScaleMults
{
    double ds = 1.0;
    double dk = 1.0;
    double dq = 1.0;
    double convDq = 1.0;
    double convDk = 1.0;
};

inline std::optional<ScaleMults> scaleMults(std::string_view placement, double scale, bool atomicKv)
{
    ScaleMults m;
    if(placement == "fold_ds")
    {
        m.ds = scale;
        return m;
    }
    if(placement == "convert")
    {
        m.dk = atomicKv ? 1.0 : scale;
        m.convDq = scale;
        m.convDk = atomicKv ? scale : 1.0;
        return m;
    }
    if(placement == "dq_before_atomic")
    {
        m.dk = atomicKv ? 1.0 : scale;
        m.dq = scale;
        m.convDk = atomicKv ? scale : 1.0;
        return m;
    }
    return std::nullopt;
}

/// The kernarg values of one launch by name, emitted in ABI order.
class KernargValues
{
public:
    void pointer(std::string_view name, std::string_view role)
    {
        _values.push_back({name, 'Q', 0, role});
    }

    void integer(std::string_view name, int64_t value)
    {
        _values.push_back({name, 'i', static_cast<uint64_t>(value), {}});
    }

    void real(std::string_view name, double value)
    {
        _values.push_back({name, 'f', f32Bits(value), {}});
    }

    /// The kernargs in @p abi order, or the name of the first parameter the values do not
    /// cover with a matching kind.
    template <std::size_t N>
    std::variant<std::vector<Kernarg>, std::string>
        ordered(const std::array<generated::AbiParam, N>& abi) const
    {
        if(_values.size() != N)
        {
            return std::string("the launch sets ") + std::to_string(_values.size())
                   + " kernargs, the ABI has " + std::to_string(N);
        }
        std::vector<Kernarg> out;
        out.reserve(N);
        for(const auto& param : abi)
        {
            const auto it = std::find_if(_values.begin(), _values.end(), [&](const Kernarg& k) {
                return k.name == param.name;
            });
            const bool isPointer = param.kind == 'Q';
            const bool isReal = param.kind == 'f';
            if(it == _values.end() || (it->kind == 'Q') != isPointer || (it->kind == 'f') != isReal)
            {
                return std::string("kernarg ") + std::string(param.name)
                       + " is not set with its ABI kind";
            }
            Kernarg arg = *it;
            arg.name = param.name;
            arg.kind = param.kind;
            out.push_back(arg);
        }
        return out;
    }

private:
    std::vector<Kernarg> _values;
};

inline void addStrides(KernargValues& values,
                       const PlanInput& in,
                       Operand tensor,
                       std::string_view b,
                       std::string_view h,
                       std::string_view t)
{
    const auto& st = in.stridesOf(tensor);
    values.integer(b, in.thd ? 0 : st[0]);
    values.integer(h, st[1]);
    values.integer(t, st[2]);
}

/// `(l_b, l_h, l_t)`: the LSE element strides the prep reads (THD drops the batch stride).
inline std::array<int64_t, 3> lseKernargStrides(const PlanInput& in)
{
    const auto& st = in.lseStrides;
    if(!in.thd)
    {
        if(st.empty())
        {
            return {in.hQ * in.sQ, in.sQ, 1};
        }
        return {st[0], st[1], st[2]};
    }
    if(st.size() == 4)
    {
        return {0, st[1], st[2]};
    }
    if(st.empty())
    {
        return {0, 1, in.hQ};
    }
    return {0, st[1], st[0]};
}

inline std::optional<std::array<uint32_t, 3>> gridOf(int64_t x, int64_t y, int64_t z)
{
    constexpr int64_t MAX = std::numeric_limits<uint32_t>::max();
    if(x < 0 || y < 0 || z < 0 || x > MAX || y > MAX || z > MAX)
    {
        return std::nullopt;
    }
    return std::array<uint32_t, 3>{
        static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)};
}

} // namespace detail

/// rocKE's `attn_bwd_stage_vec`: 8 when every used stride is a multiple of 8 and every base
/// is 16-byte aligned, else 1.
inline int64_t stageVecOf(const PlanInput& in)
{
    if(in.tensorAlignment < 16)
    {
        return 1;
    }
    for(const auto stride : detail::usedStrides(in))
    {
        if(stride % 8 != 0)
        {
            return 1;
        }
    }
    return 8;
}

/// rocKE's `attn_bwd_narrow_width`: the widest access (4, 2 or 1 elements) the strides and
/// bases of a `stage_vec = 1` request allow.
inline int64_t narrowWidthOf(const PlanInput& in)
{
    const auto used = detail::usedStrides(in);
    for(const int64_t width : {int64_t{4}, int64_t{2}})
    {
        if(in.tensorAlignment < 2 * width)
        {
            continue;
        }
        if(std::all_of(
               used.begin(), used.end(), [width](int64_t stride) { return stride % width == 0; }))
        {
            return width;
        }
    }
    return 1;
}

/// The request's tuned-table key: performance class, kv-tile bucket at the class default kN0
/// and grid-fill bucket against the CU count.
struct TuningKey
{
    int64_t headSize = 0;
    generated::Dtype dtype = generated::Dtype::FP16;
    int64_t maskKind = 0;
    int64_t headMode = 0;
    int64_t lengthRelation = 0;
    int64_t lengthMode = 0;
    int64_t stageVec = 8;
    std::string kvBucket;
    std::string fillBucket;
};

inline std::optional<TuningKey> tuningKeyOf(const PlanInput& in, const ArchTraits& traits)
{
    TuningKey key;
    key.headSize = in.headSize;
    key.dtype = in.dtype;
    key.maskKind = detail::maskKindOf(in);
    key.headMode = detail::headModeOf(in);
    key.lengthRelation = detail::lengthRelationOf(in);
    key.lengthMode = detail::lengthModeOf(in);
    key.stageVec = stageVecOf(in);
    const auto* geometry = detail::defaultGeometry(traits.arch, in, key.stageVec);
    if(geometry == nullptr)
    {
        return std::nullopt;
    }
    const int64_t tiles = detail::ceilDiv(in.sKv, geometry->blockN);
    key.kvBucket = detail::kvTileBucket(tiles);
    const bool atomicKv = in.hK != in.hV;
    const auto grid = detail::mul(in.b, detail::unitsOf(in, atomicKv));
    const auto total = grid ? detail::mul(*grid, tiles) : std::nullopt;
    key.fillBucket = detail::fillBucket(in, total.value_or(std::numeric_limits<int64_t>::max()));
    return key;
}

/// The `"|"`-joined spelling of a tuned-table key (rocKE's `tuning_key_string`).
inline std::string tuningKeyString(const TuningKey& key, const ArchTraits& traits)
{
    const std::string_view dtype = key.dtype == generated::Dtype::BF16 ? "bf16" : "fp16";
    std::string out(traits.arch);
    out += "|general|" + std::to_string(key.headSize) + "|" + std::string(dtype) + "|"
           + std::string(generated::MASK_KINDS[static_cast<std::size_t>(key.maskKind)]) + "|"
           + std::string(generated::HEAD_MODES[static_cast<std::size_t>(key.headMode)]) + "|"
           + std::string(generated::LENGTH_RELATIONS[static_cast<std::size_t>(key.lengthRelation)])
           + "|" + std::string(generated::LENGTH_MODES[static_cast<std::size_t>(key.lengthMode)])
           + "|" + std::to_string(key.stageVec) + "|" + key.kvBucket + "|" + key.fillBucket;
    return out;
}

inline std::string tuningKeyString(const PlanInput& in, const ArchTraits& traits)
{
    const auto key = tuningKeyOf(in, traits);
    return key ? tuningKeyString(*key, traits) : std::string();
}

/// The most specific tuned entry for @p key, as rocKE's `attn_bwd_table_key`: exact, then the
/// fill bucket as ANY_BUCKET, then the kv-tile bucket, then both. nullptr: the class default.
inline const generated::TunedKey* tunedEntryFor(const TuningKey& key, const ArchTraits& traits)
{
    const std::string_view any = generated::ANY_BUCKET;
    const std::array<std::pair<std::string_view, std::string_view>, 4> buckets = {{
        {key.kvBucket, key.fillBucket},
        {key.kvBucket, any},
        {any, key.fillBucket},
        {any, any},
    }};
    for(const auto& [kv, fill] : buckets)
    {
        for(const auto& entry : generated::TUNED_KEYS)
        {
            if(entry.arch == traits.arch && entry.headSize == key.headSize
               && entry.dtype == key.dtype && entry.maskKind == key.maskKind
               && entry.headMode == key.headMode && entry.lengthRelation == key.lengthRelation
               && entry.lengthMode == key.lengthMode && entry.stageVec == key.stageVec
               && entry.kvBucket == kv && entry.fillBucket == fill)
            {
                return &entry;
            }
        }
    }
    return nullptr;
}

/// The class and tuned values that name one catalog row.
struct RowKey
{
    generated::Dtype dtype = generated::Dtype::FP16;
    int64_t headSize = 0;
    generated::SeqMode seq = generated::SeqMode::BATCHED;
    int64_t stageVec = 8;
    generated::DkvMode dkv = generated::DkvMode::DIRECT;
    generated::MaskClass mask = generated::MaskClass::NONE;
    generated::DqMode dq = generated::DqMode::ATOMIC;
    bool headPack = false;
    bool lbMirror = false;
    bool lbInterleave = false;
    int64_t tuningKeyId = 0;
    bool mainWorklist = false;
    int64_t mainStaticScale = 0;
    int64_t narrowVec = 1;
};

inline const generated::CatalogRow* catalogRowFor(const RowKey& key, const ArchTraits& traits)
{
    for(const auto& row : generated::CATALOG)
    {
        if(row.arch == traits.arch && row.dtype == key.dtype && row.headSize == key.headSize
           && row.seq == key.seq && row.stageVec == key.stageVec && row.dkv == key.dkv
           && row.mask == key.mask && row.dq == key.dq && row.headPack == key.headPack
           && row.lbMirror == key.lbMirror && row.lbInterleave == key.lbInterleave
           && row.tuningKeyId == key.tuningKeyId && row.mainWorklist == key.mainWorklist
           && row.mainStaticScale == key.mainStaticScale && row.prepNarrowVec == key.narrowVec)
        {
            return &row;
        }
    }
    return nullptr;
}

/**
 * @brief The shared workspace of a plan (rocKE's `attn_bwd_workspace_layout`): fp32 sections
 *        packed in the order dq_acc (atomic dQ only), lse2, dsum, dk_acc / dv_acc (atomic
 *        dK/dV), worklist (THD work list sized for @p blockN), each rounded up to 256 bytes.
 *        nullopt when a size leaves int64.
 */
inline std::optional<WorkspaceLayout> workspaceLayoutOf(
    const PlanInput& in, bool atomicDq, bool atomicKv, bool worklist, int64_t blockN)
{
    using detail::mul;
    const auto rows = detail::rowsOf(in);
    if(!rows)
    {
        return std::nullopt;
    }
    const auto rowsQ = rows->first;
    const auto rowsKv = rows->second;
    const auto product = [](std::initializer_list<int64_t> factors) -> std::optional<int64_t> {
        int64_t out = 1;
        for(const auto f : factors)
        {
            const auto next = detail::mul(out, f);
            if(!next)
            {
                return std::nullopt;
            }
            out = *next;
        }
        return out;
    };

    std::vector<std::pair<std::string, std::optional<int64_t>>> sizes;
    if(atomicDq)
    {
        sizes.emplace_back("dq_acc", product({4, rowsQ, in.hQ, in.headSize}));
    }
    sizes.emplace_back("lse2", product({4, rowsQ, in.hQ}));
    sizes.emplace_back("dsum", product({4, rowsQ, in.hQ}));
    if(atomicKv)
    {
        sizes.emplace_back("dk_acc", product({4, rowsKv, in.hK, in.headSize}));
        sizes.emplace_back("dv_acc", product({4, rowsKv, in.hV, in.headSize}));
    }
    if(worklist)
    {
        const auto items = product({2, in.b, detail::ceilDiv(in.sKv, blockN)});
        constexpr int64_t maxItems
            = std::numeric_limits<int64_t>::max() - generated::WORKLIST_HEADER_WORDS;
        const auto words = items && *items <= maxItems
                               ? std::optional<int64_t>(generated::WORKLIST_HEADER_WORDS + *items)
                               : std::nullopt;
        sizes.emplace_back("worklist", words ? mul(4, *words) : std::nullopt);
    }

    WorkspaceLayout layout;
    int64_t offset = 0;
    for(const auto& [name, bytes] : sizes)
    {
        if(!bytes || *bytes > std::numeric_limits<int64_t>::max() - generated::WORKSPACE_ALIGN
           || offset > std::numeric_limits<int64_t>::max() - detail::alignWorkspace(*bytes))
        {
            return std::nullopt;
        }
        layout.regions[name] = Region{offset, *bytes};
        offset += detail::alignWorkspace(*bytes);
    }
    layout.bytes = offset;
    layout.scratchOffset = offset;
    return layout;
}

/**
 * @brief Why @p layout cannot be proven to fit, or empty when it can: every region is
 *        256-byte aligned, starts at or after the end of the region before it, and ends within
 *        `bytes`; `bytes` is 256-byte aligned and the scratch offset equals it.
 */
inline std::string proveWorkspace(const WorkspaceLayout& layout)
{
    constexpr int64_t ALIGN = generated::WORKSPACE_ALIGN;
    if(layout.bytes < 0 || layout.bytes % ALIGN != 0)
    {
        return "ws.bytes " + std::to_string(layout.bytes) + " is not a non-negative multiple of "
               + std::to_string(ALIGN);
    }
    if(layout.scratchOffset != layout.bytes)
    {
        return "ws.scratch_offset " + std::to_string(layout.scratchOffset) + " is not ws.bytes "
               + std::to_string(layout.bytes);
    }
    std::vector<std::pair<Region, std::string>> ordered;
    ordered.reserve(layout.regions.size());
    for(const auto& [name, region] : layout.regions)
    {
        ordered.emplace_back(region, name);
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.first.offset < b.first.offset
               || (a.first.offset == b.first.offset && a.second < b.second);
    });
    int64_t cursor = 0;
    for(const auto& [region, name] : ordered)
    {
        if(region.offset < cursor || region.offset % ALIGN != 0 || region.bytes < 0)
        {
            return "the " + name + " region at " + std::to_string(region.offset)
                   + " overlaps its predecessor or is not 256-byte aligned";
        }
        if(region.bytes > layout.bytes - region.offset)
        {
            return "the " + name + " region at " + std::to_string(region.offset) + " of "
                   + std::to_string(region.bytes) + " bytes ends past ws.bytes "
                   + std::to_string(layout.bytes);
        }
        cursor = region.offset + region.bytes;
    }
    return {};
}

namespace detail
{

/// The values every launch of one plan reads.
struct PlanValues
{
    const PlanInput* in = nullptr;
    const generated::CatalogRow* row = nullptr;
    WorkspaceLayout workspace;
    bool atomicKv = false;
    int64_t group = 0;
    int64_t dqGroup = 0;
    int64_t gSplit = 1;
    int64_t units = 0;
    int64_t gx = 0;
    int64_t nKvTiles = 0;
    generated::LbOrder lbOrder = generated::LbOrder::NATURAL;
    int64_t pair = 0;
    int64_t packHeads = 0;
    bool useWorklist = false;
    double scaleLog2 = 0.0;
    ScaleMults mults;
    int64_t rowsQ = 0;
    int64_t rowsKv = 0;
    std::array<int64_t, 3> dqGrid{};

    bool thd() const
    {
        return in->thd;
    }
    int64_t wsSeg() const
    {
        return thd() && !wsByToken(*in) ? 1 : 0;
    }
    int64_t hasLen() const
    {
        return in->padding ? 1 : 0;
    }
    int64_t qMult() const
    {
        return thd() ? in->qMult : 1;
    }
    int64_t kvMult() const
    {
        return thd() ? in->kvMult : 1;
    }
    int64_t qDiv() const
    {
        return thd() ? in->stridesOf(Operand::Q)[2] : 1;
    }
    int64_t kvDiv() const
    {
        return thd() ? in->stridesOf(Operand::K)[2] : 1;
    }
    std::string_view seqRole(bool qSide) const
    {
        if(hasLen() == 0)
        {
            return {};
        }
        return qSide ? "seq_len_q" : "seq_len_kv";
    }
    std::string_view offsetsRole(bool qSide) const
    {
        if(!thd())
        {
            return {};
        }
        return qSide ? "offsets_q" : "offsets_kv";
    }
    /// `ws.<region>`, or a null pointer when the layout has no such region.
    std::string_view workspaceRole(std::string_view region, std::string_view role) const
    {
        return workspace.regions.count(std::string(region)) != 0 ? role : std::string_view{};
    }

    void lengths(KernargValues& v) const
    {
        v.integer("has_len", hasLen());
        v.integer("len_stride", hasLen() != 0 ? in->seqLenStride : 0);
        v.integer("off64", in->offsets64 ? 1 : 0);
    }
    void sides(KernargValues& v) const
    {
        v.integer("q_mult", qMult());
        v.integer("q_div", qDiv());
        v.integer("kv_mult", kvMult());
        v.integer("kv_div", kvDiv());
    }
    void band(KernargValues& v) const
    {
        v.integer("left", in->left);
        v.integer("right", in->right);
        v.integer("bottom_right", in->bottomRight ? 1 : 0);
    }
    void heads(KernargValues& v) const
    {
        v.integer("h_q", in->hQ);
        v.integer("h_k", in->hK);
        v.integer("h_v", in->hV);
    }
};

inline KernargValues prepKernargs(const PlanValues& p)
{
    const auto& in = *p.in;
    KernargValues v;
    v.pointer("O", "o");
    v.pointer("dO", "do");
    v.pointer("LSE", "lse");
    v.pointer("SEQ_Q", p.seqRole(true));
    v.pointer("SEQ_KV", p.seqRole(false));
    v.pointer("OFF_Q", p.offsetsRole(true));
    v.pointer("OFF_KV", p.offsetsRole(false));
    v.pointer("WS_LSE2", ROLE_LSE2);
    v.pointer("WS_DSUM", ROLE_DSUM);
    v.pointer("WS_DQ", p.workspaceRole("dq_acc", ROLE_DQ_ACC));
    v.pointer("WS_DK", p.workspaceRole("dk_acc", ROLE_DK_ACC));
    v.pointer("WS_DV", p.workspaceRole("dv_acc", ROLE_DV_ACC));
    v.pointer("WORKLIST", p.workspaceRole("worklist", ROLE_WORKLIST));
    addStrides(v, in, Operand::O, "o_b", "o_h", "o_t");
    addStrides(v, in, Operand::DO, "do_b", "do_h", "do_t");
    const auto lse = lseKernargStrides(in);
    v.integer("l_b", lse[0]);
    v.integer("l_h", lse[1]);
    v.integer("l_t", lse[2]);
    p.heads(v);
    v.integer("S_q_max", in.sQ);
    v.integer("S_kv_max", in.sKv);
    p.lengths(v);
    p.sides(v);
    v.integer("ws_rows_q", p.rowsQ);
    v.integer("ws_rows_kv", p.atomicKv ? p.rowsKv : 0);
    v.integer("ws_seg", p.wsSeg());
    v.integer("zero_kv", p.atomicKv ? 1 : 0);
    v.integer("zero_dq", p.row->dq == generated::DqMode::ATOMIC ? 1 : 0);
    v.integer("use_worklist", p.useWorklist ? 1 : 0);
    v.integer("n_batch", in.b);
    v.integer("wl_kn0", p.useWorklist ? p.row->mainBlockN : 0);
    return v;
}

inline KernargValues mainKernargs(const PlanValues& p)
{
    const auto& in = *p.in;
    KernargValues v;
    v.pointer("Q", "q");
    v.pointer("K", "k");
    v.pointer("V", "v");
    v.pointer("dO", "do");
    v.pointer("dK", p.atomicKv ? "" : "dk");
    v.pointer("dV", p.atomicKv ? "" : "dv");
    v.pointer("WS_LSE2", ROLE_LSE2);
    v.pointer("WS_DSUM", ROLE_DSUM);
    v.pointer("WS_DQ", p.workspaceRole("dq_acc", ROLE_DQ_ACC));
    v.pointer("WS_DK", p.workspaceRole("dk_acc", ROLE_DK_ACC));
    v.pointer("WS_DV", p.workspaceRole("dv_acc", ROLE_DV_ACC));
    v.pointer("SEQ_Q", p.seqRole(true));
    v.pointer("SEQ_KV", p.seqRole(false));
    v.pointer("OFF_Q", p.offsetsRole(true));
    v.pointer("OFF_KV", p.offsetsRole(false));
    v.pointer("WORKLIST", p.workspaceRole("worklist", ROLE_WORKLIST));
    v.real("scale_log2", p.scaleLog2);
    v.real("ds_mult", p.mults.ds);
    v.real("dk_mult", p.mults.dk);
    v.real("dq_mult", p.mults.dq);
    addStrides(v, in, Operand::Q, "q_b", "q_h", "q_t");
    addStrides(v, in, Operand::K, "k_b", "k_h", "k_t");
    addStrides(v, in, Operand::V, "v_b", "v_h", "v_t");
    addStrides(v, in, Operand::DO, "do_b", "do_h", "do_t");
    addStrides(v, in, Operand::DK, "dk_b", "dk_h", "dk_t");
    addStrides(v, in, Operand::DV, "dv_b", "dv_h", "dv_t");
    p.heads(v);
    v.integer("G", p.group);
    v.integer("gk", in.hK != 0 ? in.hQ / in.hK : 0);
    v.integer("gv", in.hV != 0 ? in.hQ / in.hV : 0);
    v.integer("S_q_max", in.sQ);
    v.integer("S_kv_max", in.sKv);
    p.lengths(v);
    p.sides(v);
    p.band(v);
    v.integer("ws_rows_q", p.rowsQ);
    v.integer("ws_rows_kv", p.atomicKv ? p.rowsKv : 0);
    v.integer("ws_seg", p.wsSeg());
    v.integer("g_split", p.gSplit);
    v.integer("n_kv_tiles", p.nKvTiles);
    v.integer("lb_order", static_cast<int64_t>(p.lbOrder));
    v.integer("pair", p.pair);
    v.integer("xcd_n", 0);
    v.integer("xcd_chunk", 0);
    v.integer("pack_heads", p.packHeads);
    v.integer("use_worklist", p.useWorklist ? 1 : 0);
    return v;
}

/// The convert of one fp32 accumulator into its output tensor; @p launch is DQ (atomic dQ),
/// CONVERT_DK or CONVERT_DV.
inline KernargValues convertKernargs(const PlanValues& p, Launch launch)
{
    const auto& in = *p.in;
    const bool qSide = launch == Launch::DQ;
    const bool isDk = launch == Launch::CONVERT_DK;
    Operand dst = Operand::DV;
    std::string_view source = ROLE_DV_ACC;
    std::string_view destination = "dv";
    double mult = 1.0;
    if(qSide)
    {
        dst = Operand::DQ;
        source = ROLE_DQ_ACC;
        destination = "dq";
        mult = p.mults.convDq;
    }
    else if(isDk)
    {
        dst = Operand::DK;
        source = ROLE_DK_ACC;
        destination = "dk";
        mult = p.mults.convDk;
    }
    KernargValues v;
    v.pointer("SRC", source);
    v.pointer("DST", destination);
    v.pointer("SEQ", p.seqRole(qSide));
    v.pointer("OFF", p.offsetsRole(qSide));
    addStrides(v, in, dst, "d_b", "d_h", "d_t");
    v.real("mult", mult);
    v.integer("H", headsOf(in, dst));
    v.integer("S_max", seqOf(in, dst));
    p.lengths(v);
    v.integer("tok_mult", qSide ? p.qMult() : p.kvMult());
    v.integer("tok_div", qSide ? p.qDiv() : p.kvDiv());
    v.integer("ws_rows", qSide ? p.rowsQ : p.rowsKv);
    v.integer("ws_seg", p.wsSeg());
    return v;
}

/// The split dQ kernel: it replaces the dQ accumulator, its zero fill and its convert, so the
/// convert multiplier folds into dq_mult.
inline KernargValues dqKernargs(const PlanValues& p)
{
    const auto& in = *p.in;
    KernargValues v;
    v.pointer("Q", "q");
    v.pointer("K", "k");
    v.pointer("V", "v");
    v.pointer("dO", "do");
    v.pointer("dQ", "dq");
    v.pointer("WS_LSE2", ROLE_LSE2);
    v.pointer("WS_DSUM", ROLE_DSUM);
    v.pointer("SEQ_Q", p.seqRole(true));
    v.pointer("SEQ_KV", p.seqRole(false));
    v.pointer("OFF_Q", p.offsetsRole(true));
    v.pointer("OFF_KV", p.offsetsRole(false));
    v.real("scale_log2", p.scaleLog2);
    v.real("ds_mult", p.mults.ds);
    v.real("dq_mult", p.mults.dq * p.mults.convDq);
    addStrides(v, in, Operand::Q, "q_b", "q_h", "q_t");
    addStrides(v, in, Operand::K, "k_b", "k_h", "k_t");
    addStrides(v, in, Operand::V, "v_b", "v_h", "v_t");
    addStrides(v, in, Operand::DO, "do_b", "do_h", "do_t");
    addStrides(v, in, Operand::DQ, "dq_b", "dq_h", "dq_t");
    p.heads(v);
    v.integer("G", p.dqGroup);
    v.integer("S_q_max", in.sQ);
    v.integer("S_kv_max", in.sKv);
    p.lengths(v);
    p.sides(v);
    p.band(v);
    v.integer("ws_rows_q", p.rowsQ);
    v.integer("ws_seg", p.wsSeg());
    v.integer("xcd_n", 0);
    v.integer("xcd_chunk", 0);
    return v;
}

/// One launch in ABI order, or why it cannot be formed.
template <std::size_t N>
inline std::variant<LaunchPlan, std::string>
    launchPlan(Launch launch,
               std::string_view kernel,
               const std::array<int64_t, 3>& grid,
               int64_t block,
               const std::array<generated::AbiParam, N>& abi,
               const KernargValues& values)
{
    const auto fitted = gridOf(grid[0], grid[1], grid[2]);
    if(!fitted || block <= 0 || block > std::numeric_limits<uint32_t>::max())
    {
        const auto name = launchName(launch);
        return "the " + std::string(name.empty() ? "prep" : name)
               + " grid or block does not fit 32 bits";
    }
    auto ordered = values.ordered(abi);
    if(auto* why = std::get_if<std::string>(&ordered))
    {
        return std::move(*why);
    }
    return LaunchPlan{launch,
                      kernel,
                      *fitted,
                      static_cast<uint32_t>(block),
                      std::move(std::get<std::vector<Kernarg>>(ordered))};
}

/// The launches of a plan in order: prep, main, dq (split dQ kernel or the dQ convert), then
/// the dK and dV converts when dK/dV accumulate atomically.
inline std::variant<std::vector<LaunchPlan>, std::string> launchesOf(const PlanValues& p)
{
    const auto& in = *p.in;
    const auto& row = *p.row;
    std::vector<std::variant<LaunchPlan, std::string>> built;
    built.push_back(launchPlan(Launch::PREP,
                               row.prepKernel,
                               {ceilDiv(std::max<int64_t>({in.sQ, in.sKv, 1}), row.prepRowsPerCta),
                                in.hQ + (p.atomicKv ? in.hK + in.hV : 0),
                                in.b},
                               row.prepThreads,
                               generated::ABI_PREP,
                               prepKernargs(p)));
    built.push_back(launchPlan(Launch::MAIN,
                               row.mainKernel,
                               {p.gx, p.units * p.gSplit, in.b},
                               row.mainThreads,
                               generated::ABI_MAIN,
                               mainKernargs(p)));
    const auto convert = [&p, &in, &row](Launch launch, int64_t heads, int64_t sMax) {
        return launchPlan(launch,
                          row.convertKernel,
                          {ceilDiv(std::max<int64_t>(sMax, 1), row.convertRowsPerCta), heads, in.b},
                          row.convertThreads,
                          generated::ABI_CONVERT,
                          convertKernargs(p, launch));
    };
    if(row.dq == generated::DqMode::SPLIT)
    {
        built.push_back(launchPlan(
            Launch::DQ, row.dqKernel, p.dqGrid, row.dqThreads, generated::ABI_DQ, dqKernargs(p)));
    }
    else
    {
        built.push_back(convert(Launch::DQ, in.hQ, in.sQ));
    }
    if(p.atomicKv)
    {
        built.push_back(convert(Launch::CONVERT_DK, in.hK, in.sKv));
        built.push_back(convert(Launch::CONVERT_DV, in.hV, in.sKv));
    }
    std::vector<LaunchPlan> out;
    for(auto& launch : built)
    {
        if(auto* why = std::get_if<std::string>(&launch))
        {
            return std::move(*why);
        }
        out.push_back(std::move(std::get<LaunchPlan>(launch)));
    }
    return out;
}

} // namespace detail

/**
 * @brief The plan of a request that passed the request checks (rocKE's `_resolve_plan`, general
 *        route, under the engine policy), or INSTANCE_UNAVAILABLE where rocKE's plan raises.
 *
 * Selection follows rocKE's precedence: the most specific tuned-table entry of the request's
 * tuning key, else the class default; the class plus the tuned values then name one catalog
 * row. Runtime values (load-balance order and mirror pairing with its CU guard, work list,
 * head packing, scale multipliers, head split) are bound from the request and the device CU
 * count.
 */
inline std::variant<Plan, Decline> resolvePlan(const PlanInput& in, const ArchTraits& traits)
{
    using detail::unavailable;
    using generated::DqMode;
    using generated::LbOrder;

    const auto key = tuningKeyOf(in, traits);
    const auto* geometry = detail::defaultGeometry(traits.arch, in, stageVecOf(in));
    if(!key || geometry == nullptr)
    {
        return unavailable("no default geometry for the class");
    }
    const auto* tuned = tunedEntryFor(*key, traits);

    const int64_t placementIndex = tuned != nullptr ? tuned->scalePlacement : 0;
    const std::string_view placement
        = generated::SCALE_PLACEMENTS[static_cast<std::size_t>(placementIndex)];
    const int64_t staticScale = tuned != nullptr
                                    ? tuned->staticScale
                                    : detail::indexOf(generated::STATIC_SCALES, placement);
    const std::string_view staticName
        = generated::STATIC_SCALES[static_cast<std::size_t>(staticScale)];
    if(staticName != placement && staticName != "runtime")
    {
        return unavailable("static_scale " + std::string(staticName)
                           + " does not match scale_placement " + std::string(placement));
    }

    detail::PlanValues p;
    p.in = &in;
    p.gSplit = tuned != nullptr ? tuned->gSplit : 1;
    p.atomicKv = in.hK != in.hV || p.gSplit > 1;
    p.group = detail::groupOf(in, p.atomicKv);
    p.dqGroup = detail::gcdGroup(in);
    if(p.gSplit < 1 || p.group % p.gSplit != 0)
    {
        return unavailable("g_split " + std::to_string(p.gSplit) + " must divide the head group "
                           + std::to_string(p.group));
    }

    int64_t narrowVec = 1;
    if(tuned != nullptr && tuned->narrowVecCap != 0)
    {
        const int64_t cap = tuned->narrowVecCap;
        if(cap != 1 && cap != 2 && cap != 4)
        {
            return unavailable("narrow_vec " + std::to_string(cap) + " not in (1, 2, 4)");
        }
        if(key->stageVec == 1)
        {
            narrowVec = std::min(cap, narrowWidthOf(in));
        }
    }

    // Short query spans pack the group's query heads into the M rows of one q tile, unless
    // the tuned entry pins head_pack.
    const auto packedRows = detail::mul(in.sQ, p.group);
    bool headPack = false;
    if(tuned != nullptr && tuned->headPackPin >= 0)
    {
        headPack = tuned->headPackPin == 1;
    }
    else
    {
        const int64_t blockM = tuned != nullptr ? tuned->blockM : geometry->blockM;
        headPack = p.group > 1 && packedRows && *packedRows <= blockM;
    }

    p.lbOrder = static_cast<LbOrder>(tuned != nullptr ? tuned->lbOrder : 0);
    p.useWorklist = tuned != nullptr && tuned->useWorklist;

    RowKey rowKey;
    rowKey.dtype = in.dtype;
    rowKey.headSize = in.headSize;
    rowKey.seq = in.thd ? generated::SeqMode::THD : generated::SeqMode::BATCHED;
    rowKey.stageVec = key->stageVec;
    rowKey.dkv = p.atomicKv ? generated::DkvMode::ATOMIC : generated::DkvMode::DIRECT;
    rowKey.mask = key->maskKind == 0 ? generated::MaskClass::NONE : generated::MaskClass::BAND;
    rowKey.dq = tuned != nullptr ? tuned->dqMode : DqMode::ATOMIC;
    rowKey.headPack = headPack;
    rowKey.lbMirror = p.lbOrder == LbOrder::MIRROR;
    rowKey.lbInterleave = p.lbOrder == LbOrder::INTERLEAVE;
    rowKey.tuningKeyId = tuned != nullptr ? tuned->id : 0;
    rowKey.mainWorklist = p.useWorklist;
    rowKey.mainStaticScale = staticScale;
    rowKey.narrowVec = narrowVec;
    p.row = catalogRowFor(rowKey, traits);
    if(p.row == nullptr)
    {
        return unavailable("the class and tuning key " + tuningKeyString(*key, traits)
                           + " name no catalog row");
    }
    const auto& row = *p.row;

    if(headPack && (!packedRows || *packedRows > row.mainBlockM))
    {
        return unavailable("head_pack needs G * s_q <= block_m");
    }
    p.packHeads = headPack ? p.group : 0;

    p.nKvTiles = detail::ceilDiv(in.sKv, row.mainBlockN);
    p.units = detail::unitsOf(in, p.atomicKv);
    const auto unitsSplit = detail::mul(p.units, p.gSplit);
    if(!unitsSplit || *unitsSplit > generated::GRID_YZ_MAX)
    {
        return unavailable("GRID_LIMIT: main units * g_split > 65535");
    }

    p.pair = p.lbOrder == LbOrder::MIRROR ? 1 : 0;
    p.gx = p.pair != 0 ? detail::ceilDiv(p.nKvTiles, 2) : p.nKvTiles;
    if(p.pair != 0)
    {
        if(key->maskKind == 0)
        {
            return unavailable("mirror pairing needs a causal or window mask");
        }
        // No-gain guard: an unpaired grid that runs in one wave cannot be shortened.
        const auto tiles = detail::mul(p.nKvTiles, *unitsSplit);
        const auto unpaired = tiles ? detail::mul(*tiles, in.b) : std::nullopt;
        if(!in.numCus || (unpaired && *unpaired <= *in.numCus))
        {
            return unavailable("mirror pairing needs num_cus and an unpaired grid that "
                               "overfills it (more than num_cus CTAs)");
        }
        if(!generated::MIRROR_PAIRING_BUILT)
        {
            return unavailable("mirror pairing is not built");
        }
    }
    if(tuned != nullptr && tuned->xcdChunk != 0)
    {
        return unavailable("xcd_chunk needs num_xcds");
    }
    if(p.useWorklist)
    {
        if(!in.thd)
        {
            return unavailable("use_worklist serves THD requests");
        }
        if(p.lbOrder != LbOrder::NATURAL)
        {
            return unavailable("use_worklist fixes the CTA order; it needs lb_order natural");
        }
        const auto items = detail::mul(in.b, p.nKvTiles);
        const auto words = items ? detail::mul(2, *items) : std::nullopt;
        if(!words || *words > generated::I32_MAX - generated::WORKLIST_HEADER_WORDS)
        {
            return unavailable("use_worklist: the work list is not indexable in i32 words");
        }
    }

    double scale = 1.0;
    if(in.scale)
    {
        scale = *in.scale;
    }
    else if(in.headSize > 0)
    {
        scale = 1.0 / std::sqrt(static_cast<double>(in.headSize));
    }
    const auto mults = detail::scaleMults(placement, scale, p.atomicKv);
    if(!mults)
    {
        return unavailable("scale_placement " + std::string(placement) + " is unknown");
    }
    p.mults = *mults;
    p.scaleLog2 = scale * generated::LOG2E;

    if(row.dq == DqMode::SPLIT)
    {
        int64_t rows = row.dqBlockM;
        int64_t heads = in.hQ;
        if(row.dqHeadLoopGroup)
        {
            if(p.dqGroup < 1 || row.dqBlockM % p.dqGroup != 0)
            {
                return unavailable("head_loop group needs the head group to divide the dQ "
                                   "block_m");
            }
            rows = row.dqBlockM / p.dqGroup;
            heads = in.hQ / p.dqGroup;
        }
        p.dqGrid = {detail::ceilDiv(std::max<int64_t>(in.sQ, 1), rows), heads, in.b};
        if(p.dqGrid[1] > generated::GRID_YZ_MAX || p.dqGrid[0] >= (int64_t{1} << 31))
        {
            return unavailable("GRID_LIMIT: dQ grid");
        }
    }

    const auto layout = workspaceLayoutOf(
        in, row.dq == DqMode::ATOMIC, p.atomicKv, p.useWorklist, row.mainBlockN);
    if(!layout)
    {
        return unavailable("the workspace size overflows int64");
    }
    p.workspace = *layout;
    const auto rows = detail::rowsOf(in);
    p.rowsQ = rows ? rows->first : 0;
    p.rowsKv = rows ? rows->second : 0;

    auto launches = detail::launchesOf(p);
    if(auto* why = std::get_if<std::string>(&launches))
    {
        return unavailable(std::move(*why));
    }

    Plan plan;
    plan.row = p.row;
    plan.tunedKey = tuned;
    plan.runtime.gSplit = p.gSplit;
    plan.runtime.lbOrder = p.lbOrder;
    plan.runtime.pair = p.pair;
    plan.runtime.packHeads = p.packHeads;
    plan.runtime.useWorklist = p.useWorklist;
    plan.runtime.scalePlacement = placementIndex;
    plan.runtime.scale = scale;
    plan.runtime.nKvTiles = p.nKvTiles;
    plan.launches = std::move(std::get<std::vector<LaunchPlan>>(launches));
    plan.workspace = std::move(p.workspace);
    return plan;
}

/// The graph-match tokens of @p plan: the `ws.*` layout and the prep / main KMD fields of the
/// kernels the plan picked, encoded as the kernel matcher reads them.
inline BoundTokens tokensFor(const Plan& plan)
{
    BoundTokens bound;
    const auto& ws = plan.workspace;
    bound["ws.bytes"] = ws.bytes;
    bound["ws.scratch_offset"] = ws.scratchOffset;
    for(const auto& [name, region] : ws.regions)
    {
        bound["ws." + name + ".offset"] = region.offset;
    }

    const auto& row = *plan.row;
    const auto prep = [&bound](std::string_view field, int64_t value) {
        bound[std::string(generated::PREP_TOKEN_PREFIX) + std::string(field)] = value;
    };
    prep("dtype", static_cast<int64_t>(row.dtype));
    prep("head_size", row.headSize);
    prep("seq_mode", static_cast<int64_t>(row.seq));
    prep("stage_vec", row.stageVec);
    prep("worklist", row.prepWorklist ? 1 : 0);
    prep("rows_per_cta", row.prepRowsPerCta);
    prep("narrow_vec", row.prepNarrowVec);

    const auto main = [&bound](std::string_view field, int64_t value) {
        bound[std::string(generated::MAIN_TOKEN_PREFIX) + std::string(field)] = value;
    };
    main("dtype", static_cast<int64_t>(row.dtype));
    main("head_size", row.headSize);
    main("seq_mode", static_cast<int64_t>(row.seq));
    main("stage_vec", row.stageVec);
    main("dkv_mode", static_cast<int64_t>(row.dkv));
    main("mask_class", static_cast<int64_t>(row.mask));
    main("dq_mode", static_cast<int64_t>(row.dq));
    main("head_pack", row.headPack ? 1 : 0);
    main("lb_mirror", row.lbMirror ? 1 : 0);
    main("lb_interleave", row.lbInterleave ? 1 : 0);
    main("tuning_key", row.tuningKeyId);
    main("main_block_n", row.mainBlockN);
    main("main_worklist", row.mainWorklist ? 1 : 0);
    main("main_static_scale", row.mainStaticScale);
    return bound;
}

/// The plan's view of a lowered request.
inline std::optional<PlanInput> planInputOf(const Request& request)
{
    PlanInput in;
    if(request.dtype == generated::BWD_DTYPES[0])
    {
        in.dtype = generated::Dtype::FP16;
    }
    else if(request.dtype == generated::BWD_DTYPES[1])
    {
        in.dtype = generated::Dtype::BF16;
    }
    else
    {
        return std::nullopt;
    }
    in.b = request.b;
    in.hQ = request.hQ;
    in.hK = request.hK;
    in.hV = request.hV;
    in.sQ = request.sQ;
    in.sKv = request.sKv;
    in.headSize = request.dQk;
    in.thd = request.isThd();
    for(const auto operand : DATA)
    {
        in.strides[index(operand)] = request.tensorStrides(operand);
    }
    in.lseDims = request.lseDims;
    in.lseStrides = request.lseStrides;
    const auto [left, right, bottomRight] = request.resolvedBand();
    in.left = left;
    in.right = right;
    in.bottomRight = bottomRight;
    in.padding = request.padding;
    in.seqLenStride = request.seqLenStride;
    in.offsets64 = request.raggedOffsetDtype == "int64";
    std::tie(in.qMult, in.kvMult) = request.sideMults();
    in.tensorAlignment = request.tensorAlignment;
    in.scale = request.scale;
    in.numCus = request.numCus;
    in.maxTotalQ = request.maxTotalQ;
    in.maxTotalKv = request.maxTotalKv;
    return in;
}

/// resolvePlan() of a lowered request on @p arch (a resolver for evaluateRequest()).
inline std::variant<Plan, Decline> resolvePlan(const Request& request, std::string_view arch)
{
    const auto traits = shippedArchTraits(arch);
    if(!traits)
    {
        return Decline{generated::Reason::ARCH_UNSUPPORTED,
                       std::string(arch) + " is not an engine arch"};
    }
    const auto in = planInputOf(request);
    if(!in)
    {
        return detail::unavailable("dtype " + request.dtype + " has no catalog row");
    }
    return resolvePlan(*in, *traits);
}

/// The engine's verdict on a graph: lowering, request checks, plan, policy checks.
inline std::variant<Plan, Decline> evaluate(const MatchContext& context)
{
    return evaluateGraph(context, [](const Request& request, std::string_view arch) {
        return resolvePlan(request, arch);
    });
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::general_bwd
