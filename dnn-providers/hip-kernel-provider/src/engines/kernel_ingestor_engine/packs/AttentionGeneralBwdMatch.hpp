// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdConstants.generated.hpp"
#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"

/**
 * @file AttentionGeneralBwdMatch.hpp
 * @brief hipkernel:AttentionGeneralBwd's graph lowering and request checks: a hipDNN
 *        SdpaBackward node to a backward request, then rocKE's general backward predicate
 *        over it, check for check and in the same order.
 *
 * The reference is rocKE's Python: `dispatch/attention_bwd/hipdnn.py` (the lowering,
 * `from_hipdnn_graph`) and `kernels/common/attention_bwd_plan.py` (`attn_bwd_support`:
 * the request checks, the plan resolution, the policy checks; first failure wins). Every
 * constant comes from AttentionGeneralBwdConstants.generated.hpp, which is generated from
 * that Python; the replay corpus checks the two agree code for code.
 *
 * Steps, in order:
 *
 *   sdpaBackwardNode()   not a single SdpaBackward node: not this engine's graph (silent)
 *   archCheck()          a processor the engine does not ship: ARCH_UNSUPPORTED
 *   lowerGraph()         a graph the lowering cannot read: LOWERING
 *   requestChecks()      the predicate's request checks (ARCH_UNSUPPORTED .. EMPTY)
 *   (plan resolution)    the caller's resolver; a plan that does not build:
 *                        INSTANCE_UNAVAILABLE
 *   policyChecks()       the engine policy: NOT_YET_DECLARED
 *
 * evaluateGraph() runs them all around a resolver; matchGraph() also logs the decline.
 * Integer rules are evaluated in 128-bit arithmetic saturated far above every limit, so a
 * graph with extents near 2^63 is declined on the rule Python declines it on.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::general_bwd
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// The request's tensors, data operands first in rocKE's order (q k v o do dq dk dv),
/// then the stats (rocKE's `lse`).
enum class Operand : uint8_t
{
    Q,
    K,
    V,
    O,
    DO,
    DQ,
    DK,
    DV,
    LSE,
};
inline constexpr size_t DATA_OPERANDS = 8;
inline constexpr size_t OPERANDS = 9;
inline constexpr std::array<std::string_view, OPERANDS> OPERAND_NAMES
    = {{"q", "k", "v", "o", "do", "dq", "dk", "dv", "lse"}};
inline constexpr std::array<Operand, DATA_OPERANDS> DATA = {{Operand::Q,
                                                             Operand::K,
                                                             Operand::V,
                                                             Operand::O,
                                                             Operand::DO,
                                                             Operand::DQ,
                                                             Operand::DK,
                                                             Operand::DV}};
inline constexpr std::array<Operand, 3> OUTPUTS = {{Operand::DQ, Operand::DK, Operand::DV}};

constexpr size_t index(Operand operand)
{
    return static_cast<size_t>(operand);
}

/// Q's side reads Q's offset table (q o do dq), K's side K's (k v dk dv).
constexpr bool isQSide(Operand operand)
{
    return operand == Operand::Q || operand == Operand::O || operand == Operand::DO
           || operand == Operand::DQ;
}

inline std::string_view reasonName(generated::Reason code)
{
    return generated::REASON_NAMES[static_cast<size_t>(code)];
}

struct Decline
{
    generated::Reason code;
    std::string detail;
};

/// "hipkernel:AttentionGeneralBwd declined the graph [<CODE>: <detail>]"
inline std::ostream& operator<<(std::ostream& out, const Decline& decline)
{
    return out << generated::ENGINE_NAME << " declined the graph [" << reasonName(decline.code)
               << ": " << decline.detail << "]";
}

inline void logDecline(const Decline& decline)
{
    HIPDNN_PLUGIN_LOG_INFO(decline);
}

namespace checks
{

/// Signed 128-bit values saturated at +-2^125: every product of two int64 values and every
/// sum of two saturated values stays representable, and the order against any limit the
/// rules compare with (all far below 2^125) is the exact order.
__extension__ typedef __int128 Wide;
inline constexpr Wide WIDE_CAP = Wide(1) << 125;
inline constexpr Wide I31 = Wide(1) << 31;
inline constexpr Wide I63 = Wide(1) << 63;

constexpr Wide saturate(Wide value)
{
    if(value > WIDE_CAP)
    {
        return WIDE_CAP;
    }
    return value < -WIDE_CAP ? -WIDE_CAP : value;
}

constexpr Wide magnitude(Wide value)
{
    return value < 0 ? -value : value;
}

constexpr Wide mul(Wide left, Wide right)
{
    left = saturate(left);
    right = saturate(right);
    if(left == 0 || right == 0)
    {
        return 0;
    }
    if(magnitude(left) > WIDE_CAP / magnitude(right))
    {
        return (left < 0) != (right < 0) ? -WIDE_CAP : WIDE_CAP;
    }
    return left * right;
}

constexpr Wide add(Wide left, Wide right)
{
    return saturate(saturate(left) + saturate(right));
}

/// Python's floor division (`a // b`) for b > 0.
constexpr Wide floorDiv(Wide numerator, Wide denominator)
{
    Wide quotient = numerator / denominator;
    if(numerator % denominator != 0 && numerator < 0)
    {
        --quotient;
    }
    return quotient;
}

/// Sufficient non-overlap test over (extent, stride) pairs with extent > 1: each stride
/// clears the extent of the smaller ones (false proves the layout disjoint).
inline bool overlapsSorted(std::vector<std::pair<Wide, Wide>> live)
{
    std::sort(live.begin(), live.end(), [](const auto& left, const auto& right) {
        return left.second != right.second ? left.second < right.second : left.first < right.first;
    });
    Wide extent = 1;
    for(const auto& [extentOf, stride] : live)
    {
        if(stride < extent)
        {
            return true;
        }
        extent = add(extent, mul(extentOf - 1, stride));
    }
    return false;
}

/// Whether two distinct indices of a non-negatively strided tensor share an element: the
/// sorted test, then an exact test with the unit-stride dim and the largest other dim
/// solved in closed form and the rest enumerated (more than OVERLAP_BUDGET combinations
/// declines conservatively).
inline bool overlaps(const std::vector<Wide>& dims, const std::vector<Wide>& strides)
{
    std::vector<std::pair<Wide, Wide>> live;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(dims[i] > 1)
        {
            live.emplace_back(dims[i], strides[i]);
        }
    }
    if(!overlapsSorted(live))
    {
        return false;
    }
    for(const auto& [extentOf, stride] : live)
    {
        if(stride == 0)
        {
            return true;
        }
    }

    Wide unitExtent = 1;
    std::vector<std::pair<Wide, Wide>> rest;
    bool unitTaken = false;
    for(const auto& entry : live)
    {
        if(!unitTaken && entry.second == 1)
        {
            unitExtent = entry.first;
            unitTaken = true;
            continue;
        }
        rest.push_back(entry);
    }
    if(rest.empty())
    {
        return false;
    }

    size_t largest = 0;
    for(size_t k = 1; k < rest.size(); ++k)
    {
        if(rest[k].first > rest[largest].first)
        {
            largest = k;
        }
    }
    const auto [nj, sj] = rest[largest];
    std::vector<std::pair<Wide, Wide>> others;
    for(size_t k = 0; k < rest.size(); ++k)
    {
        if(k != largest)
        {
            others.push_back(rest[k]);
        }
    }

    Wide combos = 1;
    for(const auto& [extentOf, stride] : others)
    {
        combos = mul(combos, 2 * extentOf - 1);
        if(combos > generated::OVERLAP_BUDGET)
        {
            return true;
        }
    }

    std::vector<Wide> delta;
    delta.reserve(others.size());
    for(const auto& [extentOf, stride] : others)
    {
        delta.push_back(-(extentOf - 1));
    }
    while(true)
    {
        Wide x = 0;
        bool anyNonZero = false;
        for(size_t k = 0; k < others.size(); ++k)
        {
            x = add(x, mul(delta[k], others[k].second));
            anyNonZero = anyNonZero || delta[k] != 0;
        }
        const Wide lo = std::max(-(nj - 1), -floorDiv(unitExtent - 1 + x, sj));
        const Wide hi = std::min(nj - 1, floorDiv(unitExtent - 1 - x, sj));
        if(!anyNonZero)
        {
            if(std::max(lo, Wide(1)) <= hi || lo <= std::min(hi, Wide(-1)))
            {
                return true;
            }
        }
        else if(lo <= hi)
        {
            return true;
        }

        size_t k = 0;
        for(; k < others.size(); ++k)
        {
            if(delta[k] < others[k].first - 1)
            {
                ++delta[k];
                break;
            }
            delta[k] = -(others[k].first - 1);
        }
        if(k == others.size())
        {
            return false;
        }
    }
}

/// Prep / convert rows per CTA: the arch table, else one block of max(wave, 64) threads
/// with D / 8 lanes per row.
inline int64_t rowsPerCta(std::string_view arch, int64_t headSize)
{
    for(const auto& row : generated::AUX_ROWS_PER_CTA)
    {
        if(row.arch == arch && row.headSize == headSize)
        {
            return row.rowsPerCta;
        }
    }
    int64_t wave = 64;
    for(const auto& facts : generated::ARCH_FACTS)
    {
        if(facts.arch == arch)
        {
            wave = facts.waveSize;
        }
    }
    return std::max<int64_t>(1, std::max<int64_t>(wave, 64) / std::max<int64_t>(1, headSize / 8));
}

template <typename T, size_t N>
bool contains(const std::array<T, N>& values, const T& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

template <typename T>
std::string listOf(const T& values)
{
    std::ostringstream out;
    out << "[";
    bool first = true;
    for(const auto& value : values)
    {
        out << (first ? "" : ", ") << static_cast<int64_t>(value);
        first = false;
    }
    out << "]";
    return out.str();
}

} // namespace checks

/// rocKE's AttnBwdRequest, field for field (camelCase), as the lowering fills it.
///
/// Every data operand's strides are explicit (THD: the batch entry 0). An empty
/// per-operand dtype means "the same as dtype". raggedOffsetMultipliers holds the per-tensor
/// multipliers of a THD request (rocKE's mapping; all nullopt is rocKE's None).
struct Request
{
    using Strides = std::array<int64_t, 4>;

    int64_t b = 0;
    int64_t hQ = 0;
    int64_t hK = 0;
    int64_t hV = 0;
    int64_t sQ = 0;
    int64_t sKv = 0;
    int64_t dQk = 0;
    int64_t dV = 0;
    std::string dtype = "fp16";
    std::string kDtype;
    std::string vDtype;
    std::string oDtype;
    std::string doDtype;
    std::string dqDtype;
    std::string dkDtype;
    std::string dvDtype;
    std::array<Strides, DATA_OPERANDS> strides{};
    std::vector<int64_t> lseDims;
    std::vector<int64_t> lseStrides;
    std::string lseDtype = "fp32";
    std::string layout = "dense";
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool bottomRight = false;
    bool padding = false;
    bool hasSeqLenQ = false;
    bool hasSeqLenKv = false;
    std::string seqLenDtype = "int32";
    std::vector<int64_t> seqLenDims;
    int64_t seqLenStride = 1;
    std::string raggedOffsetDtype = "int32";
    std::vector<int64_t> raggedOffsetDims;
    int64_t raggedOffsetMultiplier = 1;
    bool raggedSameTable = true;
    std::array<std::optional<int64_t>, OPERANDS> raggedOffsetMultipliers{};
    std::optional<bool> lseRagged;
    std::optional<int64_t> maxTotalQ;
    std::optional<int64_t> maxTotalKv;
    int64_t tensorAlignment = generated::DEFAULT_ALIGNMENT;
    std::optional<double> scale;
    bool scaleIsDeviceTensor = false;
    bool hasBias = false;
    bool wantDbias = false;
    double dropout = 0.0;
    bool deterministic = false;
    bool alibi = false;
    bool paged = false;
    bool fp8 = false;
    std::vector<std::string> otherFeatures;
    std::optional<int64_t> numCus;
    std::optional<int64_t> numXcds;

    bool isThd() const
    {
        return layout == "thd";
    }

    const std::string& tensorDtype(Operand operand) const
    {
        const std::array<const std::string*, DATA_OPERANDS> named
            = {{&dtype, &kDtype, &vDtype, &oDtype, &doDtype, &dqDtype, &dkDtype, &dvDtype}};
        const std::string& value = *named[index(operand)];
        return value.empty() ? dtype : value;
    }

    int64_t tensorHeads(Operand operand) const
    {
        switch(operand)
        {
        case Operand::K:
        case Operand::DK:
            return hK;
        case Operand::V:
        case Operand::DV:
            return hV;
        default:
            return hQ;
        }
    }

    int64_t tensorSeq(Operand operand) const
    {
        return isQSide(operand) ? sQ : sKv;
    }

    const Strides& tensorStrides(Operand operand) const
    {
        return strides[index(operand)];
    }

    /// (left, right, bottom_right) in rocKE's units (left: the FlashAttention window).
    std::tuple<int64_t, int64_t, bool> resolvedBand() const
    {
        return {leftBound, rightBound, bottomRight};
    }

    /// "none", "causal_tl", "causal_br" or "window".
    std::string_view maskKind() const
    {
        if(leftBound == -1 && rightBound == -1)
        {
            return generated::MASK_KINDS[0];
        }
        if(leftBound == -1 && rightBound == 0)
        {
            return bottomRight ? generated::MASK_KINDS[2] : generated::MASK_KINDS[1];
        }
        return generated::MASK_KINDS[3];
    }

    /// The element-offset multiplier of an operand's offset table.
    int64_t raggedMult(Operand operand) const
    {
        return raggedOffsetMultipliers[index(operand)].value_or(raggedOffsetMultiplier);
    }

    /// (q_mult, kv_mult): the multipliers of Q's and K's tables.
    std::pair<int64_t, int64_t> sideMults() const
    {
        return {raggedMult(Operand::Q), raggedMult(Operand::K)};
    }

    /// dK / dV through fp32 workspace atomics: split K / V heads or a head split.
    bool dkvAtomic(int64_t gSplit = 1) const
    {
        return hK != hV || gSplit > 1;
    }

    /// Query heads per main CTA: h_q / h_k, or gcd over K and V heads when atomic.
    int64_t group(bool atomic) const
    {
        const int64_t gk = hK != 0 ? hQ / hK : 0;
        const int64_t gv = hV != 0 ? hQ / hV : 0;
        return atomic ? std::gcd(gk, gv) : gk;
    }

    /// Main grid head units.
    int64_t units(bool atomic) const
    {
        if(!atomic)
        {
            return hK;
        }
        const int64_t g = group(atomic);
        return g != 0 ? hQ / g : 0;
    }

    /// THD workspace rows indexed by storage token (both caller bounds given).
    bool wsByToken() const
    {
        return isThd() && maxTotalQ.has_value() && maxTotalKv.has_value();
    }

    std::pair<checks::Wide, checks::Wide> wideRows() const
    {
        if(wsByToken())
        {
            return {*maxTotalQ, *maxTotalKv};
        }
        return {checks::mul(b, sQ), checks::mul(b, sKv)};
    }

    /// The B / H / S strides of an operand that address more than one index.
    std::vector<int64_t> usedStrides(Operand operand) const
    {
        const auto& st = tensorStrides(operand);
        std::vector<int64_t> out;
        if(!isThd() && b > 1)
        {
            out.push_back(st[0]);
        }
        if(tensorHeads(operand) > 1)
        {
            out.push_back(st[1]);
        }
        if(tensorSeq(operand) > 1 || isThd())
        {
            out.push_back(st[2]);
        }
        return out;
    }

    bool everyUsedStrideDivisibleBy(int64_t width) const
    {
        for(const auto operand : DATA)
        {
            for(const auto stride : usedStrides(operand))
            {
                if(stride % width != 0)
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// 8 when every used stride is a multiple of 8 and every base is 16 B aligned, else 1.
    int64_t stageVec() const
    {
        if(tensorAlignment < 16)
        {
            return 1;
        }
        return everyUsedStrideDivisibleBy(8) ? 8 : 1;
    }

    bool hasZeroExtent() const
    {
        return std::min({b, hQ, hK, hV, sQ, sKv, dQk}) == 0;
    }

    struct LseLayout
    {
        std::string why; ///< empty: accepted
        std::array<int64_t, 3> strides{}; ///< (l_b, l_h, l_t) the prep reads it with
    };

    /// The stats form and the element strides the prep reads it with, or why it is declined.
    LseLayout lseLayout() const
    {
        const auto decline = [](std::string why) { return LseLayout{std::move(why), {}}; };
        const auto& dims = lseDims;
        const auto& st = lseStrides;
        if(lseDtype != "fp32")
        {
            return decline("LSE dtype " + lseDtype + ", not fp32");
        }
        for(const auto stride : st)
        {
            if(stride < 0)
            {
                return decline("LSE strides " + checks::listOf(st) + " hold a negative stride");
            }
        }
        const std::vector<int64_t> batched = {b, hQ, sQ, 1};
        if(!isThd())
        {
            if(lseRagged.value_or(false))
            {
                return decline("a ragged LSE needs the THD layout");
            }
            if(!dims.empty() && dims != batched)
            {
                return decline("dense LSE dims " + checks::listOf(dims) + ", want () or "
                               + checks::listOf(batched));
            }
            if(!st.empty() && st.size() != 4)
            {
                return decline("dense LSE strides " + checks::listOf(st)
                               + " not of rank 4 (b, h, t, 1)");
            }
            if(st.empty())
            {
                return LseLayout{"", {hQ * sQ, sQ, 1}};
            }
            return LseLayout{"", {st[0], st[1], st[2]}};
        }
        if(dims == batched)
        {
            if(lseRagged.has_value() && !*lseRagged)
            {
                return decline("dense per-batch LSE " + checks::listOf(batched)
                               + " under THD is not read per sequence (the prep reads it at "
                                 "the sequence's first token); pass it ragged on the Q offset "
                                 "table or packed [T_q, H_q, 1]");
            }
            if(st.size() != 4)
            {
                return decline("THD LSE in the batched shape needs its rank-4 (b, h, t, 1) "
                               "strides, got "
                               + checks::listOf(st));
            }
            return LseLayout{"", {0, st[1], st[2]}};
        }
        if(!dims.empty() && (dims.size() != 3 || dims[1] != hQ || dims[2] != 1))
        {
            return decline("THD LSE dims " + checks::listOf(dims) + ", want (), (T_q, H_q, 1) or "
                           + checks::listOf(batched));
        }
        if(!dims.empty() && dims[0] < 0)
        {
            return decline("THD LSE dims " + checks::listOf(dims) + " hold a negative token count");
        }
        if(!st.empty() && st.size() != 3)
        {
            return decline("THD LSE strides " + checks::listOf(st) + " not of rank 3 (t, h, 1)");
        }
        if(st.empty())
        {
            return LseLayout{"", {0, 1, hQ}};
        }
        return LseLayout{"", {0, st[1], st[0]}};
    }

    /// The functional feature groups the request needs, in FEATURE_GROUPS order.
    std::array<bool, generated::FEATURE_GROUPS.size()> featureGroups() const
    {
        std::array<bool, generated::FEATURE_GROUPS.size()> groups{};
        const auto set = [&groups](std::string_view name) {
            for(size_t i = 0; i < groups.size(); ++i)
            {
                if(generated::FEATURE_GROUPS[i] == name)
                {
                    groups[i] = true;
                }
            }
        };
        set("base");
        const bool defaultScale
            = !scale.has_value()
              || (dQk > 0 && *scale == 1.0 / std::sqrt(static_cast<double>(dQk)));
        if(leftBound != -1 || rightBound != -1 || !defaultScale || dQk == 32 || sQ == 1)
        {
            set("masks_and_stats");
        }
        if(hK != hV)
        {
            set("split_kv_heads");
        }
        else if(hK != hQ)
        {
            set("grouped_heads");
        }
        if(padding)
        {
            set("padding");
        }
        if(isThd())
        {
            set("ragged");
        }
        if(stageVec() == 1)
        {
            set("narrow_access");
        }
        return groups;
    }
};

/// The graph's uid of every operand, and Q's / K's offset tables.
struct TensorRoles
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t dO = 0;
    int64_t lse = 0;
    int64_t dq = 0;
    int64_t dk = 0;
    int64_t dv = 0;
    std::optional<int64_t> offsetsQ;
    std::optional<int64_t> offsetsKv;

    int64_t uidOf(Operand operand) const
    {
        const std::array<int64_t, OPERANDS> uids = {{q, k, v, o, dO, dq, dk, dv, lse}};
        return uids[index(operand)];
    }
};

struct Lowered
{
    Request request;
    TensorRoles roles;
    /// Each operand's guaranteed base alignment in bytes (data operands, then the stats).
    std::array<int64_t, OPERANDS> alignment{};
};

/// The node the engine reads, or nullptr if the graph is not a single SdpaBackward node.
inline const data_objects::SdpaBackwardAttributes* sdpaBackwardNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaBackwardAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::SdpaBackwardAttributes>();
}

/// The device's processor: gcnArchName before its first ':' ("gfx942:sramecc+:xnack-").
inline std::string_view processorOf(const MatchContext& context)
{
    const std::string_view name = context.deviceProperties.gcnArchName;
    return name.substr(0, name.find(':'));
}

inline int64_t computeUnitsOf(const MatchContext& context)
{
    return context.deviceProperties.multiProcessorCount;
}

inline std::optional<Decline> archCheck(std::string_view arch)
{
    if(!checks::contains(generated::ENGINE_ARCHS, arch))
    {
        std::ostringstream detail;
        detail << generated::ENGINE_NAME << " ships " << generated::ENGINE_ARCHS[0] << " and "
               << generated::ENGINE_ARCHS[1];
        return Decline{generated::Reason::ARCH_UNSUPPORTED, detail.str()};
    }
    return std::nullopt;
}

namespace lowering
{

inline Decline fail(std::string detail)
{
    return Decline{generated::Reason::LOWERING, std::move(detail)};
}

/// rocKE's spelling of a hipDNN data type: the JSON name through HIPDNN_DTYPES, else the
/// lower-case name (which the predicate declines by name).
inline std::string dtypeName(data_objects::DataType dataType)
{
    std::string name = data_objects::EnumNameDataType(dataType);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    for(const auto& entry : generated::HIPDNN_DTYPES)
    {
        if(entry.hipdnn == name)
        {
            return std::string(entry.rocke);
        }
    }
    return name;
}

inline std::vector<int64_t> ints(const flatbuffers::Vector<int64_t>* values)
{
    std::vector<int64_t> out;
    if(values != nullptr)
    {
        out.assign(values->begin(), values->end());
    }
    return out;
}

inline bool virtualOrPassByValue(const data_objects::TensorAttributes& tensor)
{
    return tensor.virtual_() || tensor.is_runtime_pass_by_value();
}

/// The byte alignment a declared `alignment` guarantees: its largest power-of-two divisor
/// (a pointer that is a multiple of 24 is only 8-byte aligned). Values below 1 pass through.
inline int64_t guaranteedAlignment(int64_t declared)
{
    return declared < 1 ? declared : declared & -declared;
}

inline bool isOneByte(data_objects::DataType dtype)
{
    switch(dtype)
    {
    case data_objects::DataType::UINT8:
    case data_objects::DataType::INT8:
    case data_objects::DataType::FP8_E4M3:
    case data_objects::DataType::FP8_E5M2:
    case data_objects::DataType::FP8_E8M0:
    case data_objects::DataType::FP8_E4M3_FNUZ:
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return true;
    default:
        return false;
    }
}

/// A scalar tensor's stored value as a number; nullopt for a non-numeric (bool) value.
inline std::optional<double> numericValue(const data_objects::TensorAttributes& tensor)
{
    switch(tensor.value_type())
    {
    case data_objects::TensorValue::Float32Value:
        return static_cast<double>(tensor.value_as_Float32Value()->value());
    case data_objects::TensorValue::Float16Value:
        return static_cast<double>(tensor.value_as_Float16Value()->value());
    case data_objects::TensorValue::BFloat16Value:
        return static_cast<double>(tensor.value_as_BFloat16Value()->value());
    case data_objects::TensorValue::Float8Value:
        return static_cast<double>(tensor.value_as_Float8Value()->value());
    case data_objects::TensorValue::Int32Value:
        return static_cast<double>(tensor.value_as_Int32Value()->value());
    case data_objects::TensorValue::Int64Value:
        return static_cast<double>(tensor.value_as_Int64Value()->value());
    case data_objects::TensorValue::Float64Value:
        return tensor.value_as_Float64Value()->value();
    default:
        return std::nullopt;
    }
}

/// hipDNN's band attributes as (bottom_right, left, right), its units; nullopt with the
/// reason for a band outside hipDNN's domain.
struct Band
{
    bool bottomRight = false;
    int64_t left = -1;
    int64_t right = -1;
};

inline std::variant<Band, std::string>
    resolveHipdnnBand(const data_objects::SdpaBackwardAttributes& node)
{
    const auto alignment = node.diagonal_alignment();
    if(alignment != data_objects::DiagonalAlignment::TOP_LEFT
       && alignment != data_objects::DiagonalAlignment::BOTTOM_RIGHT)
    {
        return std::string("unknown diagonal_alignment");
    }
    asm_sdpa_engine::plan_utils::DiagonalBand diagonal{};
    try
    {
        diagonal = asm_sdpa_engine::plan_utils::resolveDiagonalBand(node);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        return std::string(e.what());
    }
    // resolveDiagonalBand returns the FlashAttention window (L - 1); this band keeps
    // hipDNN's units.
    Band band;
    band.bottomRight = diagonal.alignment == data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    band.left = diagonal.leftBound < 0 ? -1 : diagonal.leftBound + 1;
    band.right = diagonal.rightBound;
    return band;
}

} // namespace lowering

/**
 * @brief The backward request of an SdpaBackward graph, or LOWERING for a graph the
 *        lowering cannot read: an operand uid naming no tensor, a virtual or pass-by-value
 *        operand, a rank or dim mismatch, an offset table that is missing, virtual,
 *        pass-by-value, itself ragged, not rank 4 or not unit-strided, a ragged tensor that
 *        is not an operand, a stats alignment below 4 bytes, a band outside hipDNN's
 *        domain, or an optional uid naming no tensor.
 *
 * @p numCus is the device's compute-unit count (<= 0: unknown).
 */
inline std::variant<Lowered, Decline> lowerGraph(const MatchContext& context, int64_t numCus)
{
    using lowering::fail;
    const auto* node = sdpaBackwardNode(context);
    if(node == nullptr)
    {
        return fail("not an SdpaBackward graph");
    }

    Lowered out;
    TensorRoles& roles = out.roles;
    roles.q = node->q_tensor_uid();
    roles.k = node->k_tensor_uid();
    roles.v = node->v_tensor_uid();
    roles.o = node->o_tensor_uid();
    roles.dO = node->do_tensor_uid();
    roles.lse = node->stats_tensor_uid();
    roles.dq = node->dq_tensor_uid();
    roles.dk = node->dk_tensor_uid();
    roles.dv = node->dv_tensor_uid();

    // operands
    std::array<const data_objects::TensorAttributes*, OPERANDS> ent{};
    std::array<std::vector<int64_t>, OPERANDS> dims;
    std::array<std::vector<int64_t>, OPERANDS> strides;
    for(size_t i = 0; i < OPERANDS; ++i)
    {
        const auto operand = static_cast<Operand>(i);
        const int64_t uid = roles.uidOf(operand);
        ent[i] = sdpa_graph::findTensor(context, uid);
        if(ent[i] == nullptr)
        {
            return fail(std::string(OPERAND_NAMES[i]) + " uid " + std::to_string(uid)
                        + " names no tensor");
        }
        if(lowering::virtualOrPassByValue(*ent[i]))
        {
            return fail(std::string(OPERAND_NAMES[i]) + " (uid " + std::to_string(uid)
                        + ") is virtual or pass-by-value");
        }
        dims[i] = lowering::ints(ent[i]->dims());
        strides[i] = lowering::ints(ent[i]->strides());
    }
    for(size_t i = 0; i < OPERANDS; ++i)
    {
        if(dims[i].size() != sdpa_graph::SDPA_RANK || strides[i].size() != sdpa_graph::SDPA_RANK)
        {
            return fail(std::string(OPERAND_NAMES[i]) + " dims " + checks::listOf(dims[i])
                        + " / strides " + checks::listOf(strides[i]) + " not rank 4");
        }
        if(*std::min_element(dims[i].begin(), dims[i].end()) < 0)
        {
            return fail(std::string(OPERAND_NAMES[i]) + " dims " + checks::listOf(dims[i])
                        + " negative");
        }
    }
    {
        const auto& q = dims[index(Operand::Q)];
        const auto& k = dims[index(Operand::K)];
        const auto& v = dims[index(Operand::V)];
        const int64_t batch = q[0];
        const int64_t hq = q[1];
        const int64_t sq = q[2];
        const int64_t dqk = q[3];
        const std::array<std::pair<Operand, std::vector<int64_t>>, 7> want = {{
            {Operand::K, {batch, k[1], k[2], dqk}},
            {Operand::V, {batch, v[1], k[2], v[3]}},
            {Operand::O, {batch, hq, sq, v[3]}},
            {Operand::DO, {batch, hq, sq, v[3]}},
            {Operand::DQ, q},
            {Operand::DK, k},
            {Operand::DV, v},
        }};
        for(const auto& [operand, wanted] : want)
        {
            if(dims[index(operand)] != wanted)
            {
                return fail(std::string(OPERAND_NAMES[index(operand)]) + " dims "
                            + checks::listOf(dims[index(operand)]) + ", want "
                            + checks::listOf(wanted));
            }
        }
        const auto& stats = dims[index(Operand::LSE)];
        if(stats[0] != batch || stats[1] != hq || stats[2] != sq)
        {
            return fail("stats dims " + checks::listOf(stats) + ", want [" + std::to_string(batch)
                        + ", " + std::to_string(hq) + ", " + std::to_string(sq) + ", 1]");
        }
    }

    // ragged offsets
    std::array<std::optional<int64_t>, OPERANDS> offsets{};
    for(size_t i = 0; i < OPERANDS; ++i)
    {
        const auto offset = ent[i]->ragged_offset_tensor_uid();
        if(!offset.has_value())
        {
            continue;
        }
        const int64_t off = offset.value();
        const std::string what
            = std::string(OPERAND_NAMES[i]) + " offset table uid " + std::to_string(off);
        const auto* table = sdpa_graph::findTensor(context, off);
        if(table == nullptr)
        {
            return fail(std::string(OPERAND_NAMES[i]) + " ragged offset uid " + std::to_string(off)
                        + " names no tensor");
        }
        if(lowering::virtualOrPassByValue(*table))
        {
            return fail(what + " is virtual or pass-by-value");
        }
        if(table->ragged_offset_tensor_uid().has_value())
        {
            return fail(what + " is itself ragged");
        }
        const auto tableDims = lowering::ints(table->dims());
        const auto tableStrides = lowering::ints(table->strides());
        if(tableDims.size() != sdpa_graph::SDPA_RANK
           || tableStrides.size() != sdpa_graph::SDPA_RANK)
        {
            return fail(what + " is not rank 4");
        }
        if(tableDims[0] > 1 && tableStrides[0] != 1)
        {
            return fail(what + " stride " + std::to_string(tableStrides[0]) + " is not 1");
        }
        offsets[i] = off;
    }
    {
        std::set<int64_t> consumed;
        for(size_t i = 0; i < OPERANDS; ++i)
        {
            consumed.insert(roles.uidOf(static_cast<Operand>(i)));
        }
        for(const auto& [uid, tensor] : context.graph.getTensorMap())
        {
            if(consumed.count(uid) == 0 && tensor != nullptr
               && tensor->ragged_offset_tensor_uid().has_value())
            {
                return fail("tensor uid " + std::to_string(uid)
                            + " is ragged; only the backward operands are read through an "
                              "offset table");
            }
        }
    }

    for(size_t i = 0; i < OPERANDS; ++i)
    {
        out.alignment[i] = lowering::guaranteedAlignment(ent[i]->alignment());
    }
    if(out.alignment[index(Operand::LSE)] < generated::LSE_MIN_ALIGNMENT)
    {
        return fail("stats alignment below 4 bytes");
    }

    const auto band = lowering::resolveHipdnnBand(*node);
    if(const auto* why = std::get_if<std::string>(&band))
    {
        return fail("band: " + *why);
    }
    const auto& hipdnnBand = std::get<lowering::Band>(band);

    // request fields
    Request& req = out.request;
    const auto& q = dims[index(Operand::Q)];
    const auto& k = dims[index(Operand::K)];
    const auto& v = dims[index(Operand::V)];
    req.b = q[0];
    req.hQ = q[1];
    req.hK = k[1];
    req.hV = v[1];
    req.sQ = q[2];
    req.sKv = k[2];
    req.dQk = q[3];
    req.dV = v[3];
    if(numCus > 0)
    {
        req.numCus = numCus;
    }
    req.leftBound = hipdnnBand.left == -1 ? -1 : hipdnnBand.left - 1;
    req.rightBound = hipdnnBand.right;
    req.bottomRight = hipdnnBand.bottomRight;

    std::array<std::string, OPERANDS> dts;
    for(size_t i = 0; i < OPERANDS; ++i)
    {
        dts[i] = lowering::dtypeName(ent[i]->data_type());
    }
    req.dtype = dts[index(Operand::Q)];
    req.kDtype = dts[index(Operand::K)];
    req.vDtype = dts[index(Operand::V)];
    req.oDtype = dts[index(Operand::O)];
    req.doDtype = dts[index(Operand::DO)];
    req.dqDtype = dts[index(Operand::DQ)];
    req.dkDtype = dts[index(Operand::DK)];
    req.dvDtype = dts[index(Operand::DV)];
    req.fp8 = std::any_of(DATA.begin(), DATA.end(), [&dts](Operand operand) {
        return std::string_view(dts[index(operand)]).substr(0, 3) == "fp8";
    });
    req.lseDtype = dts[index(Operand::LSE)];
    if(context.graph.getNodeWrapper(0).computeDataType() != data_objects::DataType::FLOAT)
    {
        req.otherFeatures = {"compute_data_type"};
    }

    const bool thd = std::any_of(DATA.begin(), DATA.end(), [&offsets](Operand operand) {
        return offsets[index(operand)].has_value();
    });
    for(const auto operand : DATA)
    {
        const auto& st = strides[index(operand)];
        req.strides[index(operand)] = {thd ? 0 : st[0], st[1], st[2], st[3]};
    }
    if(thd)
    {
        req.layout = "thd";

        const auto& qTable = offsets[index(Operand::Q)];
        bool same = std::all_of(DATA.begin(), DATA.end(), [&offsets](Operand operand) {
            return offsets[index(operand)].has_value();
        });
        if(same)
        {
            const auto& kTable = offsets[index(Operand::K)];
            for(const auto operand : DATA)
            {
                same = same && offsets[index(operand)] == (isQSide(operand) ? qTable : kTable);
            }
            const auto& lseTable = offsets[index(Operand::LSE)];
            same = same && (!lseTable.has_value() || lseTable == qTable);
        }
        req.raggedSameTable = same;

        std::vector<int64_t> used;
        for(size_t i = 0; i < OPERANDS; ++i)
        {
            if(offsets[i].has_value()
               && std::find(used.begin(), used.end(), *offsets[i]) == used.end())
            {
                used.push_back(*offsets[i]);
            }
        }
        std::vector<std::string> tableDtypes;
        for(const auto uid : used)
        {
            const auto name
                = lowering::dtypeName(sdpa_graph::findTensor(context, uid)->data_type());
            if(std::find(tableDtypes.begin(), tableDtypes.end(), name) == tableDtypes.end())
            {
                tableDtypes.push_back(name);
            }
        }
        req.raggedOffsetDtype = tableDtypes.size() == 1 ? tableDtypes[0] : "mixed";

        // [B + 1, 1, 1, 1] reads as (B + 1,); the first other form is kept as given
        const auto canonical = [&req](const std::vector<int64_t>& tableDims) {
            return tableDims.size() == sdpa_graph::SDPA_RANK
                   && checks::Wide(tableDims[0]) == checks::Wide(req.b) + 1 && tableDims[1] == 1
                   && tableDims[2] == 1 && tableDims[3] == 1;
        };
        for(const auto uid : used)
        {
            auto tableDims = lowering::ints(sdpa_graph::findTensor(context, uid)->dims());
            if(!canonical(tableDims))
            {
                req.raggedOffsetDims = std::move(tableDims);
                break;
            }
        }
        if(req.raggedOffsetDims.empty())
        {
            req.raggedOffsetDims = {req.b + 1};
        }
        for(size_t i = 0; i < OPERANDS; ++i)
        {
            if(offsets[i].has_value())
            {
                req.raggedOffsetMultipliers[i] = ent[i]->ragged_offset_multiplier();
            }
        }
    }
    req.lseDims = dims[index(Operand::LSE)];
    req.lseStrides = strides[index(Operand::LSE)];
    req.lseRagged = offsets[index(Operand::LSE)].has_value();
    req.tensorAlignment = out.alignment[0];
    for(const auto operand : DATA)
    {
        req.tensorAlignment = std::min(req.tensorAlignment, out.alignment[index(operand)]);
    }
    roles.offsetsQ = offsets[index(Operand::Q)];
    roles.offsetsKv = offsets[index(Operand::K)];

    // scale
    if(node->scale_tensor_uid().has_value())
    {
        const int64_t uid = node->scale_tensor_uid().value();
        const auto* scaleTensor = sdpa_graph::findTensor(context, uid);
        if(scaleTensor == nullptr)
        {
            return fail("scale_tensor_uid uid " + std::to_string(uid) + " names no tensor");
        }
        if(scaleTensor->is_runtime_pass_by_value()
           || scaleTensor->value_type() == data_objects::TensorValue::NONE)
        {
            req.scaleIsDeviceTensor = true;
        }
        else if(lowering::isOneByte(scaleTensor->data_type())
                || scaleTensor->value_type() == data_objects::TensorValue::Float8Value)
        {
            // hipDNN stores a 1-byte constant as its encoded byte, not its value.
            return fail("scale tensor of a 1-byte dtype");
        }
        else
        {
            const auto value = lowering::numericValue(*scaleTensor);
            if(!value.has_value())
            {
                return fail("scale tensor value is not a number");
            }
            req.scale = value;
        }
    }
    else if(node->attn_scale_value().has_value())
    {
        req.scale = static_cast<double>(node->attn_scale_value().value());
    }
    else
    {
        // hipDNN (cuDNN) applies no scaling when neither is given.
        req.scale = 1.0;
    }

    // features
    req.hasBias = node->attn_mask_tensor_uid().has_value();
    req.wantDbias = node->dbias_tensor_uid().has_value();
    req.alibi = node->alibi_mask();
    const std::array<flatbuffers::Optional<int64_t>, 5> dropoutInputs
        = {{node->seed_tensor_uid(),
            node->offset_tensor_uid(),
            node->dropout_mask_tensor_uid(),
            node->dropout_scale_tensor_uid(),
            node->dropout_scale_inv_tensor_uid()}};
    const double probability = node->dropout_probability().has_value()
                                   ? static_cast<double>(node->dropout_probability().value())
                                   : 0.0;
    if(probability != 0.0)
    {
        req.dropout = probability;
    }
    else if(std::any_of(dropoutInputs.begin(), dropoutInputs.end(), [](const auto& uid) {
                return uid.has_value();
            }))
    {
        // dropout requested through its tensors only
        req.dropout = 1.0;
    }
    const std::array<std::pair<std::string_view, flatbuffers::Optional<int64_t>>, 10> optional
        = {{{"scale_tensor_uid", node->scale_tensor_uid()},
            {"attn_mask_tensor_uid", node->attn_mask_tensor_uid()},
            {"seq_len_q_tensor_uid", node->seq_len_q_tensor_uid()},
            {"seq_len_kv_tensor_uid", node->seq_len_kv_tensor_uid()},
            {"seed_tensor_uid", node->seed_tensor_uid()},
            {"offset_tensor_uid", node->offset_tensor_uid()},
            {"dropout_mask_tensor_uid", node->dropout_mask_tensor_uid()},
            {"dropout_scale_tensor_uid", node->dropout_scale_tensor_uid()},
            {"dropout_scale_inv_tensor_uid", node->dropout_scale_inv_tensor_uid()},
            {"dbias_tensor_uid", node->dbias_tensor_uid()}}};
    for(const auto& [key, uid] : optional)
    {
        if(uid.has_value() && sdpa_graph::findTensor(context, uid.value()) == nullptr)
        {
            return fail(std::string(key) + " uid " + std::to_string(uid.value())
                        + " names no tensor");
        }
    }

    // padding
    req.padding = node->padding_mask();
    struct SeqLenForm
    {
        std::string dtype;
        std::vector<int64_t> dims;
        int64_t stride;
        bool operator!=(const SeqLenForm& other) const
        {
            return dtype != other.dtype || dims != other.dims || stride != other.stride;
        }
    };
    std::vector<SeqLenForm> forms;
    for(const auto& [uid, has] : {std::pair{node->seq_len_q_tensor_uid(), &req.hasSeqLenQ},
                                  std::pair{node->seq_len_kv_tensor_uid(), &req.hasSeqLenKv}})
    {
        if(!uid.has_value())
        {
            continue;
        }
        *has = true;
        const auto* tensor = sdpa_graph::findTensor(context, uid.value());
        const auto st = lowering::ints(tensor->strides());
        forms.push_back({lowering::dtypeName(tensor->data_type()),
                         lowering::ints(tensor->dims()),
                         st.empty() ? 0 : st[0]});
    }
    if(!forms.empty())
    {
        req.seqLenDtype = forms[0].dtype;
        if(std::any_of(forms.begin(), forms.end(), [&forms](const SeqLenForm& form) {
               return form != forms[0];
           }))
        {
            req.seqLenDtype = "mixed";
        }
        req.seqLenDims = forms[0].dims;
        req.seqLenStride = forms[0].stride;
    }
    return out;
}

namespace checks
{

/// The first out-of-scope feature (the engine policy does not serve deterministic).
inline std::optional<Decline> featureDecline(const Request& req)
{
    using generated::Reason;
    std::ostringstream dropout;
    dropout << "dropout=" << req.dropout;
    std::string features;
    for(const auto& feature : req.otherFeatures)
    {
        features += (features.empty() ? "" : ", ") + feature;
    }
    const std::array<std::tuple<Reason, bool, std::string>, 8> rows = {{
        {Reason::DROPOUT, req.dropout != 0.0, dropout.str()},
        {Reason::BIAS_IN_BACKWARD, req.hasBias, "bias / attn_mask tensor in the backward graph"},
        {Reason::DBIAS, req.wantDbias, "dBias requested"},
        {Reason::ALIBI, req.alibi, "ALiBi requested"},
        {Reason::PAGED, req.paged, "paged KV is served only by the rocKE-native unified entry"},
        {Reason::FP8, req.fp8, "fp8 tensors requested"},
        {Reason::DETERMINISTIC, req.deterministic, "deterministic backward requested"},
        {Reason::UNSUPPORTED_FEATURE, !req.otherFeatures.empty(), "features (" + features + ")"},
    }};
    for(const auto& [code, hit, detail] : rows)
    {
        if(hit)
        {
            return Decline{code, detail};
        }
    }
    return std::nullopt;
}

inline std::optional<std::string> bandBound(const Request& req)
{
    for(const auto& [name, value] :
        {std::pair{"left_bound", req.leftBound}, std::pair{"right_bound", req.rightBound}})
    {
        if(value < -1 || value >= generated::I32_FAR)
        {
            return std::string(name) + "=" + std::to_string(value) + " outside [-1, 2^30)";
        }
    }
    return std::nullopt;
}

/// The i64 kernargs and the i64 byte offsets the kernels form.
inline std::optional<std::string> wideRange(const Request& req, const std::array<int64_t, 3>& lse)
{
    for(const auto operand : DATA)
    {
        const auto& [sb, sh, st, sd] = req.tensorStrides(operand);
        (void)sd;
        const std::string name(OPERAND_NAMES[index(operand)]);
        if(Wide(sh) >= I63 || (!req.isThd() && Wide(sb) >= I63))
        {
            return name + ": batch / head stride does not fit the i64 kernarg";
        }
        Wide lastOffset = add(mul(Wide(req.tensorHeads(operand)) - 1, sh), req.dQk);
        if(!req.isThd())
        {
            lastOffset
                = add(lastOffset,
                      add(mul(Wide(req.b) - 1, sb), mul(Wide(req.tensorSeq(operand)) - 1, st)));
        }
        if(mul(lastOffset, 2) >= I63)
        {
            return name + ": largest byte offset >= 2^63";
        }
    }
    const auto [lb, lh, lt] = lse;
    if(Wide(lb) >= I63 || Wide(lh) >= I63)
    {
        return std::string("lse: batch / head stride does not fit the i64 kernarg");
    }
    Wide lastOffset = mul(Wide(req.hQ) - 1, lh);
    if(!req.isThd())
    {
        lastOffset = add(lastOffset, add(mul(Wide(req.b) - 1, lb), mul(req.sQ, lt)));
    }
    if(mul(add(lastOffset, 1), 4) >= I63)
    {
        return std::string("lse: largest byte offset >= 2^63");
    }
    return std::nullopt;
}

inline std::optional<std::string> indexRange(const Request& req, bool atomicKv)
{
    const Wide d = req.dQk;
    for(const auto operand : DATA)
    {
        const Wide ss = magnitude(req.tensorStrides(operand)[2]);
        if(add(mul(req.tensorSeq(operand), ss), d) >= I31)
        {
            return std::string(OPERAND_NAMES[index(operand)]) + ": S_max * stride_s + D >= 2^31";
        }
    }
    const auto lse = req.lseLayout().strides;
    if(add(mul(req.sQ, magnitude(lse[2])), 1) >= I31)
    {
        return std::string("lse: S_q * stride_s + 1 >= 2^31");
    }
    if(auto why = wideRange(req, lse))
    {
        return why;
    }
    const auto [rowsQ, rowsKv] = req.wideRows();
    if(rowsQ >= I31)
    {
        return std::string("workspace rows_q >= 2^31");
    }
    if(atomicKv && rowsKv >= I31)
    {
        return std::string("workspace rows_kv >= 2^31");
    }
    for(const int64_t sMax : {req.sQ, req.sKv})
    {
        if(mul(sMax, d) >= I31)
        {
            return std::string("S_max * D >= 2^31 (in-slab workspace index)");
        }
        if(sMax >= generated::I32_FAR)
        {
            return std::string("S_max >= 2^30");
        }
    }
    return std::nullopt;
}

inline std::optional<std::string>
    gridLimit(const Request& req, std::string_view arch, bool atomicKv)
{
    if(req.b > generated::GRID_YZ_MAX)
    {
        return "B=" + std::to_string(req.b) + " > 65535";
    }
    const int64_t units = req.units(atomicKv);
    if(units > generated::GRID_YZ_MAX)
    {
        return "main units=" + std::to_string(units) + " > 65535";
    }
    const Wide prepY = add(req.hQ, atomicKv ? add(req.hK, req.hV) : Wide(0));
    if(prepY > generated::GRID_YZ_MAX)
    {
        return std::string("prep grid y > 65535");
    }
    const Wide rows = rowsPerCta(arch, req.dQk);
    const Wide longest = std::max(req.sQ, req.sKv);
    if(floorDiv(longest + rows - 1, rows) >= I31)
    {
        return std::string("prep / convert grid x >= 2^31");
    }
    return std::nullopt;
}

inline std::optional<std::string> seqLenFormat(const Request& req)
{
    const bool hasAny = req.hasSeqLenQ || req.hasSeqLenKv;
    if(hasAny && !req.padding)
    {
        return std::string("SEQ_LEN tensors without padding_mask");
    }
    if(req.padding && !(req.hasSeqLenQ && req.hasSeqLenKv))
    {
        return std::string("padding_mask without both SEQ_LEN tensors");
    }
    if(hasAny)
    {
        if(req.seqLenDtype != "int32")
        {
            return "SEQ_LEN dtype " + req.seqLenDtype;
        }
        const std::vector<int64_t> perBatch = {req.b, 1, 1, 1};
        if(!req.seqLenDims.empty() && req.seqLenDims != perBatch)
        {
            return "SEQ_LEN dims " + listOf(req.seqLenDims);
        }
        if(req.seqLenStride < 0 || req.seqLenStride > generated::I32_MAX)
        {
            return "SEQ_LEN stride " + std::to_string(req.seqLenStride);
        }
    }
    return std::nullopt;
}

inline std::optional<std::string> raggedFormat(const Request& req)
{
    if(!req.isThd())
    {
        return std::nullopt;
    }
    if(req.raggedOffsetDtype == "mixed")
    {
        return std::string("the q-side and kv-side offset tables differ in dtype");
    }
    if(req.raggedOffsetDtype != "int32" && req.raggedOffsetDtype != "int64")
    {
        return "offset dtype " + req.raggedOffsetDtype;
    }
    if(!req.raggedOffsetDims.empty()
       && (req.raggedOffsetDims.size() != 1 || Wide(req.raggedOffsetDims[0]) != Wide(req.b) + 1))
    {
        return "offset dims " + listOf(req.raggedOffsetDims);
    }
    if(!req.raggedSameTable)
    {
        return std::string("every tensor must be ragged: q o do dq (and a ragged lse) on one "
                           "offset table, k v dk dv on one offset table");
    }
    const auto inRange = [](int64_t value) { return value >= 1 && value <= generated::I32_MAX; };
    if(!inRange(req.raggedOffsetMultiplier))
    {
        return "multiplier " + std::to_string(req.raggedOffsetMultiplier) + " outside [1, 2^31)";
    }
    for(size_t i = 0; i < OPERANDS; ++i)
    {
        const auto& mult = req.raggedOffsetMultipliers[i];
        if(mult.has_value() && !inRange(*mult))
        {
            return std::string(OPERAND_NAMES[i]) + " multiplier " + std::to_string(*mult)
                   + " outside [1, 2^31)";
        }
    }
    for(const auto lead : {Operand::Q, Operand::K})
    {
        if(!inRange(req.tensorStrides(lead)[2]))
        {
            return std::string(OPERAND_NAMES[index(lead)]) + " token stride "
                   + std::to_string(req.tensorStrides(lead)[2]) + " outside [1, 2^31)";
        }
    }
    for(const auto operand : DATA)
    {
        if(req.tensorStrides(operand)[2] < 1)
        {
            return std::string(OPERAND_NAMES[index(operand)]) + " token stride must be positive";
        }
    }
    // Each tensor's multiplier must land on its side lead's token:
    // m_t * div == m_lead * token_stride_t.
    const auto [qMult, kvMult] = req.sideMults();
    const int64_t qDiv = req.tensorStrides(Operand::Q)[2];
    const int64_t kvDiv = req.tensorStrides(Operand::K)[2];
    for(const auto operand : DATA)
    {
        const auto& mult = req.raggedOffsetMultipliers[index(operand)];
        if(!mult.has_value())
        {
            continue;
        }
        const bool qSide = isQSide(operand);
        const int64_t lead = qSide ? qMult : kvMult;
        const int64_t div = qSide ? qDiv : kvDiv;
        const int64_t tokenStride = req.tensorStrides(operand)[2];
        if(mul(*mult, div) != mul(lead, tokenStride))
        {
            return std::string(OPERAND_NAMES[index(operand)]) + " multiplier "
                   + std::to_string(*mult) + " does not address " + (qSide ? "q" : "k")
                   + "'s token";
        }
    }
    const auto& lseMult = req.raggedOffsetMultipliers[index(Operand::LSE)];
    if(lseMult.has_value())
    {
        if(req.lseRagged.has_value() && !*req.lseRagged)
        {
            return std::string("an lse multiplier for a dense per-batch lse");
        }
        const int64_t lseToken = req.lseLayout().strides[2];
        if(mul(*lseMult, qDiv) != mul(qMult, lseToken))
        {
            return "lse multiplier " + std::to_string(*lseMult) + " does not address q's token";
        }
    }
    if(!req.hasZeroExtent())
    {
        for(const auto& [name, bound] :
            {std::pair{"max_total_q", req.maxTotalQ}, std::pair{"max_total_kv", req.maxTotalKv}})
        {
            if(bound.has_value() && *bound < 1)
            {
                return std::string(name) + "=" + std::to_string(*bound) + " must be at least 1";
            }
        }
    }
    return std::nullopt;
}

} // namespace checks

/**
 * @brief rocKE's request checks (`_request_checks`), in order: the first decline, or
 *        nullopt when the request reaches plan resolution.
 *
 * The engine policy is fixed: every engine arch validated, deterministic not served, no
 * resource-gate exclusion, so ARCH_NOT_VALIDATED and the instance list never decline.
 */
inline std::optional<Decline> requestChecks(const Request& req, std::string_view arch)
{
    using checks::Wide;
    using generated::Reason;
    if(!checks::contains(generated::ENGINE_ARCHS, arch))
    {
        return Decline{Reason::ARCH_UNSUPPORTED, std::string(arch) + " is not a backward target"};
    }
    if(auto decline = checks::featureDecline(req))
    {
        return decline;
    }
    if(req.scaleIsDeviceTensor)
    {
        return Decline{Reason::SCALE_DEVICE_TENSOR, "scale is a device-buffer tensor"};
    }
    const std::array<Operand, 5> inputs
        = {{Operand::Q, Operand::K, Operand::V, Operand::O, Operand::DO}};
    const bool knownDtype
        = std::find(generated::BWD_DTYPES.begin(), generated::BWD_DTYPES.end(), req.dtype)
          != generated::BWD_DTYPES.end();
    if(!knownDtype || std::any_of(inputs.begin(), inputs.end(), [&req](Operand operand) {
           return req.tensorDtype(operand) != req.dtype;
       }))
    {
        std::string names;
        for(const auto operand : inputs)
        {
            names += (names.empty() ? "" : ", ") + req.tensorDtype(operand);
        }
        return Decline{Reason::DTYPE, "inputs [" + names + "] not all fp16 or all bf16"};
    }
    for(const auto operand : OUTPUTS)
    {
        if(req.tensorDtype(operand) != req.dtype)
        {
            return Decline{Reason::OUT_DTYPE,
                           std::string(OPERAND_NAMES[index(operand)]) + " dtype "
                               + req.tensorDtype(operand) + " differs from " + req.dtype};
        }
    }
    const auto lse = req.lseLayout();
    if(!lse.why.empty())
    {
        return Decline{Reason::LSE_FORMAT, lse.why};
    }
    if(req.dQk != req.dV)
    {
        return Decline{Reason::DQK_NE_DV,
                       "d_qk=" + std::to_string(req.dQk) + " d_v=" + std::to_string(req.dV)};
    }
    if(req.dQk == generated::HEAD_DIM_256)
    {
        return Decline{Reason::HEAD_DIM_256_BWD, "head dim 256 has no backward"};
    }
    if(!checks::contains(generated::BWD_HEAD_SIZES, req.dQk))
    {
        return Decline{Reason::HEAD_DIM,
                       "head dim " + std::to_string(req.dQk) + " not in "
                           + checks::listOf(generated::BWD_HEAD_SIZES)};
    }
    if(req.hK > 0 && req.hV > 0 && (req.hQ % req.hK != 0 || req.hQ % req.hV != 0))
    {
        return Decline{Reason::GQA_RATIO,
                       "h_q=" + std::to_string(req.hQ) + " not a multiple of h_k="
                           + std::to_string(req.hK) + ", h_v=" + std::to_string(req.hV)};
    }
    if(auto why = checks::bandBound(req))
    {
        return Decline{Reason::BAND_BOUND, *why};
    }
    for(const auto operand : DATA)
    {
        if(req.tensorStrides(operand)[3] != 1)
        {
            return Decline{Reason::STRIDE_D,
                           std::string(OPERAND_NAMES[index(operand)]) + " D stride "
                               + std::to_string(req.tensorStrides(operand)[3])};
        }
    }
    for(const auto operand : DATA)
    {
        const auto& st = req.tensorStrides(operand);
        if(std::any_of(st.begin(), st.end(), [](int64_t stride) { return stride < 0; }))
        {
            return Decline{Reason::STRIDE_NEGATIVE,
                           std::string(OPERAND_NAMES[index(operand)]) + " strides "
                               + checks::listOf(st)};
        }
    }
    const auto [rowsQ, rowsKv] = req.wideRows();
    for(const auto operand : OUTPUTS)
    {
        const auto& [sb, sh, ss, sd] = req.tensorStrides(operand);
        const Wide heads = req.tensorHeads(operand);
        std::vector<Wide> dims;
        std::vector<Wide> strides;
        if(req.isThd())
        {
            dims = {operand == Operand::DQ ? rowsQ : rowsKv, heads, req.dQk};
            strides = {ss, sh, sd};
        }
        else
        {
            dims = {req.b, heads, req.tensorSeq(operand), req.dQk};
            strides = {sb, sh, ss, sd};
        }
        if(*std::min_element(dims.begin(), dims.end()) == 0)
        {
            continue;
        }
        if(checks::overlaps(dims, strides))
        {
            return Decline{Reason::OUTPUT_OVERLAP,
                           std::string(OPERAND_NAMES[index(operand)]) + " strides "
                               + checks::listOf(req.tensorStrides(operand)) + " self-overlap"};
        }
    }
    if(req.tensorAlignment < 2)
    {
        return Decline{Reason::STRIDE_ALIGN,
                       "base alignment " + std::to_string(req.tensorAlignment)
                           + " B < element size"};
    }
    const bool atomicKv = req.dkvAtomic();
    if(auto why = checks::indexRange(req, atomicKv))
    {
        return Decline{Reason::INDEX_RANGE, *why};
    }
    if(auto why = checks::gridLimit(req, arch, atomicKv))
    {
        return Decline{Reason::GRID_LIMIT, *why};
    }
    if(auto why = checks::seqLenFormat(req))
    {
        return Decline{Reason::SEQ_LEN_FORMAT, *why};
    }
    if(auto why = checks::raggedFormat(req))
    {
        return Decline{Reason::RAGGED_FORMAT, *why};
    }
    if(req.hasZeroExtent())
    {
        return Decline{Reason::EMPTY, "a batch, head, sequence or head-dim extent is 0"};
    }
    return std::nullopt;
}

/// rocKE's policy checks (`_policy_checks`) under the engine policy: no performance gate,
/// so only the feature groups it does not declare decline (NOT_YET_DECLARED).
inline std::optional<Decline> policyChecks(const Request& req, std::string_view /*arch*/)
{
    const auto groups = req.featureGroups();
    std::string missing;
    for(size_t i = 0; i < groups.size(); ++i)
    {
        if(groups[i] && !generated::FEATURE_GROUP_DECLARED[i])
        {
            missing += (missing.empty() ? "'" : ", '") + std::string(generated::FEATURE_GROUPS[i])
                       + "'";
        }
    }
    if(!missing.empty())
    {
        return Decline{generated::Reason::NOT_YET_DECLARED,
                       "feature groups not declared yet: [" + missing + "]"};
    }
    return std::nullopt;
}

/**
 * @brief The predicate's order around a plan resolver: the request checks, the plan
 *        (INSTANCE_UNAVAILABLE when it does not build), then the policy checks.
 *
 * @p resolve is called as `resolve(request, arch)` and returns `std::variant<Plan, Decline>`.
 */
template <typename Resolve>
auto evaluateRequest(const Request& request,
                     std::string_view arch,
                     Resolve&& resolve) -> std::decay_t<decltype(resolve(request, arch))>
{
    using Result = std::decay_t<decltype(resolve(request, arch))>;
    if(auto decline = requestChecks(request, arch))
    {
        return Result{std::move(*decline)};
    }
    Result result = resolve(request, arch);
    if(auto* decline = std::get_if<Decline>(&result))
    {
        if(decline->code != generated::Reason::INSTANCE_UNAVAILABLE)
        {
            decline->detail = "no buildable instance: " + std::string(reasonName(decline->code))
                              + ": " + decline->detail;
            decline->code = generated::Reason::INSTANCE_UNAVAILABLE;
        }
        return result;
    }
    if(auto decline = policyChecks(request, arch))
    {
        return Result{std::move(*decline)};
    }
    return result;
}

/**
 * @brief Every step on a graph: the engine's arch, the lowering, then evaluateRequest().
 *        A graph that is not a single SdpaBackward node declines LOWERING here; graph
 *        matchers test sdpaBackwardNode() first and stay silent on it.
 */
template <typename Resolve>
auto evaluateGraph(const MatchContext& context, Resolve&& resolve)
    -> std::decay_t<decltype(resolve(std::declval<const Request&>(), std::string_view{}))>
{
    using Result
        = std::decay_t<decltype(resolve(std::declval<const Request&>(), std::string_view{}))>;
    if(sdpaBackwardNode(context) == nullptr)
    {
        return Result{lowering::fail("not an SdpaBackward graph")};
    }
    const std::string_view arch = processorOf(context);
    if(auto decline = archCheck(arch))
    {
        return Result{std::move(*decline)};
    }
    auto lowered = lowerGraph(context, computeUnitsOf(context));
    if(auto* decline = std::get_if<Decline>(&lowered))
    {
        return Result{std::move(*decline)};
    }
    return evaluateRequest(std::get<Lowered>(lowered).request, arch, resolve);
}

/// evaluateGraph() for a graph matcher: nullopt, silently for a graph that is not an
/// SdpaBackward node, else with the decline logged at INFO.
template <typename Resolve>
auto matchGraph(const MatchContext& context, Resolve&& resolve)
{
    using Result
        = std::decay_t<decltype(resolve(std::declval<const Request&>(), std::string_view{}))>;
    using Plan = std::variant_alternative_t<0, Result>;
    std::optional<Plan> out;
    if(sdpaBackwardNode(context) == nullptr)
    {
        return out;
    }
    auto result = evaluateGraph(context, resolve);
    if(auto* decline = std::get_if<Decline>(&result))
    {
        logDecline(*decline);
        return out;
    }
    out.emplace(std::move(std::get<Plan>(result)));
    return out;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::general_bwd
