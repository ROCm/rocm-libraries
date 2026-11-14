// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IRunnableKernel.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdGeometry.hpp"

/**
 * @file AttentionGeneralBwdNative.cpp
 * @brief The hipkernel:AttentionGeneralBwd engine's native half: graph match, kernel match,
 *        per-pack architecture criteria and the slot dispatch handlers.
 *
 * The engine runs rocKE's general SDPA backward as two slots. `prep` builds the LSE2/Dsum
 * rows, zeroes the shared fp32 accumulators and, under THD, the work list; `main` runs the
 * main kernel, then the dQ launch (the split dQ kernel or the dQ convert) and, when dK/dV
 * accumulate atomically, the dK and dV converts. Every main kernel declares the four main
 * launches; the dK/dV converts of a direct-dK/dV plan are never built or launched.
 *
 * The graph match computes the whole plan once (AttentionGeneralBwdGeometry.hpp) and
 * publishes it as tokens: the shared workspace layout (`ws.*`) and the KMD fields of the
 * prep and main kernels the plan names. The kernel match selects exactly those kernels.
 * Each handler recomputes the plan from the graph at prepare, refuses one that disagrees
 * with the tokens or with its kernel, and binds every runtime value as a kernarg.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

namespace generated = general_bwd::generated;

static_assert(generated::ENGINE_NAME == ATTENTION_GENERAL_BWD_ENGINE_NAME,
              "the generated constants name another engine");

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.attention_general_bwd.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.attention_general_bwd.kernel_match";
constexpr std::string_view PREP_DISPATCH_SYMBOL = "hipkernel.attention_general_bwd.prep_dispatch";
constexpr std::string_view MAIN_DISPATCH_SYMBOL = "hipkernel.attention_general_bwd.main_dispatch";
constexpr std::string_view PREP_GFX942_SYMBOL = "hipkernel.attention_general_bwd.prep_gfx942_match";
constexpr std::string_view PREP_GFX950_SYMBOL = "hipkernel.attention_general_bwd.prep_gfx950_match";
constexpr std::string_view MAIN_GFX942_SYMBOL = "hipkernel.attention_general_bwd.main_gfx942_match";
constexpr std::string_view MAIN_GFX950_SYMBOL = "hipkernel.attention_general_bwd.main_gfx950_match";
constexpr std::string_view MAIN_SPLITDQ_GFX950_SYMBOL
    = "hipkernel.attention_general_bwd.main_splitdq_gfx950_match";

/// The launch names of every main kernel's `sources`.
constexpr std::array<std::string_view, 4> MAIN_LAUNCHES
    = {{"main", "dq", "convert_dk", "convert_dv"}};

/// The workspace base the regions are offset from. The regions themselves sit on 256-byte
/// boundaries of it; the kernels read the fp32 sections with 16-byte vectors.
constexpr uintptr_t WORKSPACE_BASE_ALIGNMENT = 16;

[[noreturn]] void failBadParam(const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, std::string(generated::ENGINE_NAME) + ": " + message);
}

// ---------------------------------------------------------------------------
// Graph match
// ---------------------------------------------------------------------------

std::variant<general_bwd::Plan, general_bwd::Decline>
    resolvePlanOf(const general_bwd::Request& request, std::string_view arch)
{
    return general_bwd::resolvePlan(request, arch);
}

/**
 * @brief Graph-scoped applicability: rocKE's backward predicate and plan over the lowered
 *        graph, published as the `ws.*` layout and the class tokens of the picked kernels.
 *
 * Silent on a graph that is not a single SdpaBackward node; every other decline is logged
 * at INFO as `<engine> declined the graph [<CODE>: <detail>]`.
 */
std::optional<BoundTokens> attentionGeneralBwdGraphMatches(const MatchContext& context)
{
    auto plan = general_bwd::matchGraph(context, &resolvePlanOf);
    if(!plan.has_value())
    {
        return std::nullopt;
    }
    if(const auto why = general_bwd::proveWorkspace(plan->workspace); !why.empty())
    {
        general_bwd::logDecline(
            general_bwd::Decline{generated::Reason::LOWERING, "workspace: " + why});
        return std::nullopt;
    }
    return general_bwd::tokensFor(*plan);
}

// ---------------------------------------------------------------------------
// Pack criteria
// ---------------------------------------------------------------------------

enum class ArchId : uint8_t
{
    GFX942,
    GFX950,
};

constexpr std::string_view archName(ArchId arch)
{
    return arch == ArchId::GFX942 ? generated::ENGINE_ARCHS[0] : generated::ENGINE_ARCHS[1];
}

static_assert(generated::ENGINE_ARCHS[0] == "gfx942" && generated::ENGINE_ARCHS[1] == "gfx950");

/// Each pack holds one architecture's code objects, so it serves only its own device.
template <ArchId ARCH>
bool packArchMatches(const MatchContext& context, const BoundTokens& /*bound*/)
{
    return general_bwd::processorOf(context) == archName(ARCH);
}

// ---------------------------------------------------------------------------
// Kernel match
// ---------------------------------------------------------------------------

/// How a KMD field's value encodes into its token.
enum class FieldKind : uint8_t
{
    INT,
    BOOL,
    DTYPE,
    SEQ_MODE,
    DKV_MODE,
    MASK_CLASS,
    DQ_MODE,
    STATIC_SCALE,
    TUNING_KEY,
};

struct SlotField
{
    std::string_view name;
    FieldKind kind;
};

constexpr std::array<SlotField, generated::PREP_TOKENS.size()> PREP_FIELDS = {{
    {"dtype", FieldKind::DTYPE},
    {"head_size", FieldKind::INT},
    {"seq_mode", FieldKind::SEQ_MODE},
    {"stage_vec", FieldKind::INT},
    {"worklist", FieldKind::BOOL},
    {"rows_per_cta", FieldKind::INT},
    {"narrow_vec", FieldKind::INT},
}};

constexpr std::array<SlotField, generated::MAIN_TOKENS.size()> MAIN_FIELDS = {{
    {"dtype", FieldKind::DTYPE},
    {"head_size", FieldKind::INT},
    {"seq_mode", FieldKind::SEQ_MODE},
    {"stage_vec", FieldKind::INT},
    {"dkv_mode", FieldKind::DKV_MODE},
    {"mask_class", FieldKind::MASK_CLASS},
    {"dq_mode", FieldKind::DQ_MODE},
    {"head_pack", FieldKind::BOOL},
    {"lb_mirror", FieldKind::BOOL},
    {"lb_interleave", FieldKind::BOOL},
    {"tuning_key", FieldKind::TUNING_KEY},
    {"main_block_n", FieldKind::INT},
    {"main_worklist", FieldKind::BOOL},
    {"main_static_scale", FieldKind::STATIC_SCALE},
}};

template <std::size_t N>
constexpr bool fieldsFollowTokens(const std::array<SlotField, N>& fields,
                                  const std::array<std::string_view, N>& tokens)
{
    for(std::size_t i = 0; i < N; ++i)
    {
        if(fields[i].name != tokens[i])
        {
            return false;
        }
    }
    return true;
}

static_assert(fieldsFollowTokens(PREP_FIELDS, generated::PREP_TOKENS),
              "the prep fields drifted from the generated token list");
static_assert(fieldsFollowTokens(MAIN_FIELDS, generated::MAIN_TOKENS),
              "the main fields drifted from the generated token list");

/// The field only a main kernel's record carries.
constexpr std::string_view MAIN_ONLY_FIELD = "dq_mode";

struct Spelling
{
    std::string_view text;
    int64_t code;
};

template <typename E>
constexpr int64_t codeOf(E value)
{
    return static_cast<int64_t>(value);
}

// The KMD spells dtype as the descriptor vocabulary maps it (fp16 -> FP16).
constexpr std::array<Spelling, 2> DTYPE_SPELLINGS
    = {{{"FP16", codeOf(generated::Dtype::FP16)}, {"BF16", codeOf(generated::Dtype::BF16)}}};
constexpr std::array<Spelling, 2> SEQ_MODE_SPELLINGS = {
    {{"batched", codeOf(generated::SeqMode::BATCHED)}, {"thd", codeOf(generated::SeqMode::THD)}}};
constexpr std::array<Spelling, 2> DKV_MODE_SPELLINGS
    = {{{"direct", codeOf(generated::DkvMode::DIRECT)},
        {"atomic", codeOf(generated::DkvMode::ATOMIC)}}};
constexpr std::array<Spelling, 2> MASK_CLASS_SPELLINGS = {
    {{"none", codeOf(generated::MaskClass::NONE)}, {"band", codeOf(generated::MaskClass::BAND)}}};
constexpr std::array<Spelling, 2> DQ_MODE_SPELLINGS = {
    {{"atomic", codeOf(generated::DqMode::ATOMIC)}, {"split", codeOf(generated::DqMode::SPLIT)}}};

std::optional<int64_t> spellingCode(const std::array<Spelling, 2>& spellings, std::string_view text)
{
    for(const auto& spelling : spellings)
    {
        if(spelling.text == text)
        {
            return spelling.code;
        }
    }
    return std::nullopt;
}

std::optional<int64_t> staticScaleCode(std::string_view text)
{
    for(std::size_t i = 0; i < generated::STATIC_SCALES.size(); ++i)
    {
        if(generated::STATIC_SCALES[i] == text)
        {
            return static_cast<int64_t>(i);
        }
    }
    return std::nullopt;
}

/// The per-arch id of a tuning-key string; 0 is the class default ("").
std::optional<int64_t> tuningKeyCode(std::string_view text, std::string_view arch)
{
    if(text.empty())
    {
        return 0;
    }
    for(const auto& entry : generated::TUNED_KEYS)
    {
        if(entry.arch == arch && entry.key == text)
        {
            return entry.id;
        }
    }
    return std::nullopt;
}

/// @p field of @p kernel as its token encodes it, or nullopt when the record lacks the field
/// or holds a value the encoding has no code for. Total: a malformed record never matches.
std::optional<int64_t>
    encodeField(const KernelDefinition& kernel, const SlotField& field, std::string_view arch)
{
    const auto it = kernel.metadata.find(std::string(field.name));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto& value = it->second;
    const auto* integer = std::get_if<int64_t>(&value);
    const auto* flag = std::get_if<bool>(&value);
    const auto* text = std::get_if<std::string>(&value);

    switch(field.kind)
    {
    case FieldKind::INT:
        return integer != nullptr ? std::optional<int64_t>(*integer) : std::nullopt;
    case FieldKind::BOOL:
        return flag != nullptr ? std::optional<int64_t>(*flag ? 1 : 0) : std::nullopt;
    case FieldKind::DTYPE:
        return text != nullptr ? spellingCode(DTYPE_SPELLINGS, *text) : std::nullopt;
    case FieldKind::SEQ_MODE:
        return text != nullptr ? spellingCode(SEQ_MODE_SPELLINGS, *text) : std::nullopt;
    case FieldKind::DKV_MODE:
        return text != nullptr ? spellingCode(DKV_MODE_SPELLINGS, *text) : std::nullopt;
    case FieldKind::MASK_CLASS:
        return text != nullptr ? spellingCode(MASK_CLASS_SPELLINGS, *text) : std::nullopt;
    case FieldKind::DQ_MODE:
        return text != nullptr ? spellingCode(DQ_MODE_SPELLINGS, *text) : std::nullopt;
    case FieldKind::STATIC_SCALE:
        return text != nullptr ? staticScaleCode(*text) : std::nullopt;
    case FieldKind::TUNING_KEY:
        return text != nullptr ? tuningKeyCode(*text, arch) : std::nullopt;
    default:
        return std::nullopt;
    }
}

/// One INFO line per process for a record no graph can select, so a catalog built with a
/// vocabulary this engine does not know is visible without flooding every match.
void logUnencodableOnce(const KernelDefinition& kernel, std::string_view field)
{
    static std::atomic<bool> s_logged{false};
    if(!s_logged.exchange(true))
    {
        HIPDNN_PLUGIN_LOG_INFO(generated::ENGINE_NAME
                               << ": kernel '" << kernel.name << "' carries no " << field
                               << " value this engine encodes, so no graph selects it");
    }
}

/// The name of the token @p field of a slot binds: `<prefix><field>`.
std::string tokenName(std::string_view prefix, std::string_view field)
{
    return std::string(prefix) + std::string(field);
}

template <std::size_t N>
bool fieldsMatch(const std::array<SlotField, N>& fields,
                 std::string_view prefix,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel,
                 std::string_view arch)
{
    for(const auto& field : fields)
    {
        const auto expected = tryGetBoundInt(bound, tokenName(prefix, field.name));
        if(!expected.has_value())
        {
            return false;
        }
        const auto actual = encodeField(kernel, field, arch);
        if(!actual.has_value())
        {
            logUnencodableOnce(kernel, field.name);
            return false;
        }
        if(*actual != *expected)
        {
            return false;
        }
    }
    return true;
}

bool isMainKernel(const KernelDefinition& kernel)
{
    return kernel.metadata.count(std::string(MAIN_ONLY_FIELD)) != 0;
}

/// Kernel-scoped applicability: the kernel whose every slot field equals the token the graph
/// match bound for it. The plan names one prep and one main kernel, so exactly one of each
/// matches across the architecture's packs.
bool attentionGeneralBwdKernelMatches(const MatchContext& context,
                                      const BoundTokens& bound,
                                      const KernelDefinition& kernel)
{
    const std::string_view arch = general_bwd::processorOf(context);
    if(isMainKernel(kernel))
    {
        return fieldsMatch(MAIN_FIELDS, generated::MAIN_TOKEN_PREFIX, bound, kernel, arch);
    }
    return fieldsMatch(PREP_FIELDS, generated::PREP_TOKEN_PREFIX, bound, kernel, arch);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The slot a handler serves.
enum class Role : uint8_t
{
    PREP,
    MAIN,
};

/// The kernarg ABI of one launch kind (`rocke.attn_bwd.v3`).
enum class Abi : uint8_t
{
    PREP,
    MAIN,
    CONVERT,
    DQ,
};

template <std::size_t N>
constexpr bool hasKnownKinds(const std::array<generated::AbiParam, N>& abi)
{
    for(const auto& param : abi)
    {
        if(param.kind != 'Q' && param.kind != 'q' && param.kind != 'i' && param.kind != 'f')
        {
            return false;
        }
    }
    return true;
}

static_assert(hasKnownKinds(generated::ABI_PREP) && hasKnownKinds(generated::ABI_MAIN)
                  && hasKnownKinds(generated::ABI_CONVERT) && hasKnownKinds(generated::ABI_DQ),
              "an ABI parameter has a format letter launch() does not marshal");

/// The argument list a launch marshals, each argument at its natural alignment. Named, so
/// requireSignatureMatch also catches a permutation of same-sized arguments.
template <std::size_t N>
std::vector<KernelArgument> signatureOf(const std::array<generated::AbiParam, N>& abi)
{
    std::vector<KernelArgument> signature;
    signature.reserve(N);
    uint32_t offset = 0;
    for(const auto& param : abi)
    {
        const bool pointer = param.kind == 'Q';
        const uint32_t size = pointer || param.kind == 'q' ? 8U : 4U;
        offset = (offset + size - 1) / size * size;
        signature.push_back(KernelArgument{
            pointer ? "global_buffer" : "by_value", size, offset, std::string(param.name)});
        offset += size;
    }
    return signature;
}

std::vector<KernelArgument> signatureOf(Abi abi)
{
    switch(abi)
    {
    case Abi::PREP:
        return signatureOf(generated::ABI_PREP);
    case Abi::MAIN:
        return signatureOf(generated::ABI_MAIN);
    case Abi::CONVERT:
        return signatureOf(generated::ABI_CONVERT);
    case Abi::DQ:
        return signatureOf(generated::ABI_DQ);
    default:
        failBadParam("unknown launch ABI");
    }
}

std::size_t abiSize(Abi abi)
{
    switch(abi)
    {
    case Abi::PREP:
        return generated::ABI_PREP.size();
    case Abi::MAIN:
        return generated::ABI_MAIN.size();
    case Abi::CONVERT:
        return generated::ABI_CONVERT.size();
    case Abi::DQ:
        return generated::ABI_DQ.size();
    default:
        failBadParam("unknown launch ABI");
    }
}

/// The dQ launch is the split dQ kernel when the plan splits dQ, otherwise the convert.
Abi abiOf(general_bwd::Launch launch, generated::DqMode dqMode)
{
    switch(launch)
    {
    case general_bwd::Launch::PREP:
        return Abi::PREP;
    case general_bwd::Launch::MAIN:
        return Abi::MAIN;
    case general_bwd::Launch::DQ:
        return dqMode == generated::DqMode::SPLIT ? Abi::DQ : Abi::CONVERT;
    case general_bwd::Launch::CONVERT_DK:
    case general_bwd::Launch::CONVERT_DV:
        return Abi::CONVERT;
    default:
        failBadParam("unknown launch");
    }
}

/// Where a pointer kernarg points at launch.
struct PointerArg
{
    std::size_t position = 0;
    /// A tensor's uid, or nullopt for a workspace region.
    std::optional<int64_t> uid;
    /// Bytes past the unshifted workspace base, for a workspace region.
    int64_t offset = 0;
    /// The alignment the launch requires of the real pointer, in bytes.
    int64_t alignment = 1;
    std::string_view role;
};

/// One launch, ready to go: its code object with grid and block applied, its kernargs in ABI
/// order (every pointer slot 0 until launch), and where its pointers come from.
struct PreparedLaunch
{
    general_bwd::Launch launch = general_bwd::Launch::PREP;
    Abi abi = Abi::PREP;
    IngestorKernelCode code;
    std::vector<uint64_t> words;
    std::vector<PointerArg> pointers;
};

class PreparedAttentionGeneralBwd : public PreparedDispatch
{
public:
    explicit PreparedAttentionGeneralBwd(std::vector<PreparedLaunch> launches)
        : _launches(std::move(launches))
    {
    }

    const std::vector<PreparedLaunch>& launches() const
    {
        return _launches;
    }

private:
    std::vector<PreparedLaunch> _launches;
};

/// The graph's plan with the lowering it came from, which carries the operand uids and their
/// guaranteed alignments.
struct GraphPlan
{
    general_bwd::Lowered lowered;
    general_bwd::Plan plan;
};

[[noreturn]] void failNoLongerMatches(const general_bwd::Decline& decline)
{
    failBadParam("the graph no longer matches ["
                 + std::string(general_bwd::reasonName(decline.code)) + ": " + decline.detail
                 + "]");
}

/// The graph match's verdict recomputed at prepare. A graph that no longer yields a plan is a
/// caller that reached the handler without matching: it faults rather than declines.
GraphPlan graphPlanOf(const MatchContext& context)
{
    if(general_bwd::sdpaBackwardNode(context) == nullptr)
    {
        failBadParam("the graph is not a single SdpaBackward node");
    }
    const std::string_view arch = general_bwd::processorOf(context);
    if(auto decline = general_bwd::archCheck(arch))
    {
        failNoLongerMatches(*decline);
    }
    auto lowered = general_bwd::lowerGraph(context, general_bwd::computeUnitsOf(context));
    if(const auto* decline = std::get_if<general_bwd::Decline>(&lowered))
    {
        failNoLongerMatches(*decline);
    }
    auto& lowering = std::get<general_bwd::Lowered>(lowered);
    auto result = general_bwd::evaluateRequest(lowering.request, arch, &resolvePlanOf);
    if(const auto* decline = std::get_if<general_bwd::Decline>(&result))
    {
        failNoLongerMatches(*decline);
    }
    auto& plan = std::get<general_bwd::Plan>(result);
    if(const auto why = general_bwd::proveWorkspace(plan.workspace); !why.empty())
    {
        failBadParam("the workspace of the graph's plan does not prove: " + why);
    }
    return GraphPlan{std::move(lowering), std::move(plan)};
}

bool isEngineToken(std::string_view token)
{
    const auto startsWith
        = [token](std::string_view prefix) { return token.substr(0, prefix.size()) == prefix; };
    return startsWith("ws.") || startsWith(generated::PREP_TOKEN_PREFIX)
           || startsWith(generated::MAIN_TOKEN_PREFIX);
}

/// The handlers address the workspace by the layout they recompute, and the runtime sized
/// and checked it by the bound tokens: every token the plan publishes must be bound with its
/// value, and no other engine token may be.
void requireTokens(const BoundTokens& bound, const BoundTokens& expected)
{
    for(const auto& [token, value] : expected)
    {
        const auto* want = std::get_if<int64_t>(&value);
        const auto got = tryGetBoundInt(bound, token);
        if(want == nullptr || !got.has_value() || *got != *want)
        {
            failBadParam("token '" + token + "' is "
                         + (got.has_value() ? std::to_string(*got) : std::string("missing"))
                         + ", the plan of the graph has "
                         + (want != nullptr ? std::to_string(*want) : std::string("none")));
        }
    }
    for(const auto& [token, value] : bound)
    {
        static_cast<void>(value);
        if(isEngineToken(token) && expected.count(token) == 0)
        {
            failBadParam("token '" + token
                         + "' is bound but the plan of the graph has no such "
                           "value");
        }
    }
}

/// The operand a tensor role names, in the geometry's role spelling.
std::optional<general_bwd::Operand> operandOfRole(std::string_view role)
{
    for(std::size_t i = 0; i < general_bwd::OPERANDS; ++i)
    {
        if(general_bwd::OPERAND_NAMES[i] == role)
        {
            return static_cast<general_bwd::Operand>(i);
        }
    }
    return std::nullopt;
}

/// Where the pointer kernarg @p kernarg points at launch, or nullopt for a null pointer.
std::optional<PointerArg> pointerArgOf(std::size_t position,
                                       const general_bwd::Kernarg& kernarg,
                                       const GraphPlan& graphPlan)
{
    const std::string_view role = kernarg.role;
    if(role.empty())
    {
        return std::nullopt;
    }
    PointerArg pointer;
    pointer.position = position;
    pointer.role = role;

    constexpr std::string_view WORKSPACE_PREFIX = "ws.";
    if(role.substr(0, WORKSPACE_PREFIX.size()) == WORKSPACE_PREFIX)
    {
        const auto& regions = graphPlan.plan.workspace.regions;
        const auto region = regions.find(std::string(role.substr(WORKSPACE_PREFIX.size())));
        if(region == regions.end())
        {
            failBadParam("kernarg " + std::string(kernarg.name) + " names workspace region '"
                         + std::string(role) + "', which the plan's layout lacks");
        }
        pointer.offset = region->second.offset;
        return pointer;
    }

    const auto& lowered = graphPlan.lowered;
    if(const auto operand = operandOfRole(role))
    {
        pointer.uid = lowered.roles.uidOf(*operand);
        pointer.alignment = *operand == general_bwd::Operand::LSE
                                ? generated::LSE_MIN_ALIGNMENT
                                : lowered.alignment[general_bwd::index(*operand)];
        return pointer;
    }

    const std::optional<int64_t>* table = nullptr;
    if(role == "offsets_q")
    {
        table = &lowered.roles.offsetsQ;
    }
    else if(role == "offsets_kv")
    {
        table = &lowered.roles.offsetsKv;
    }
    if(table == nullptr || !table->has_value())
    {
        failBadParam("kernarg " + std::string(kernarg.name) + " is bound to '" + std::string(role)
                     + "', which this engine does not serve");
    }
    pointer.uid = *table;
    pointer.alignment = lowered.request.raggedOffsetDtype == "int64" ? 8 : 4;
    return pointer;
}

/// The kernargs of @p launchPlan as launch words: pointers 0 (filled at launch), i64 and f32
/// as recorded, i32 as its low 32 bits after a range check.
struct BoundKernargs
{
    std::vector<uint64_t> words;
    std::vector<PointerArg> pointers;
};

BoundKernargs
    bindKernargs(const general_bwd::LaunchPlan& launchPlan, Abi abi, const GraphPlan& graphPlan)
{
    const auto& kernargs = launchPlan.kernargs;
    if(kernargs.size() != abiSize(abi))
    {
        failBadParam("the " + std::string(general_bwd::launchName(launchPlan.launch))
                     + " launch binds " + std::to_string(kernargs.size())
                     + " kernargs, its ABI takes " + std::to_string(abiSize(abi)));
    }
    BoundKernargs bound;
    bound.words.assign(kernargs.size(), 0);
    for(std::size_t i = 0; i < kernargs.size(); ++i)
    {
        const auto& kernarg = kernargs[i];
        switch(kernarg.kind)
        {
        case 'Q':
            if(auto pointer = pointerArgOf(i, kernarg, graphPlan))
            {
                bound.pointers.push_back(*pointer);
            }
            break;
        case 'i':
        {
            const int64_t value = kernarg.intValue();
            if(value < std::numeric_limits<int32_t>::min()
               || value > std::numeric_limits<int32_t>::max())
            {
                failBadParam("kernarg " + std::string(kernarg.name) + " = " + std::to_string(value)
                             + " does not fit its i32 slot");
            }
            bound.words[i] = static_cast<uint32_t>(static_cast<int32_t>(value));
            break;
        }
        case 'q':
        case 'f':
            bound.words[i] = kernarg.bits;
            break;
        default:
            failBadParam("kernarg " + std::string(kernarg.name) + " has an unknown kind");
        }
    }
    return bound;
}

/// A launch's code object, checked against the symbol the plan names and the ABI it
/// marshals, with the plan's grid and block applied.
IngestorKernelCode buildLaunchCode(const compilation::KpackKernelLoader& kpackLoader,
                                   const MatchContext& context,
                                   const KernelDefinition& definition,
                                   const general_bwd::LaunchPlan& launchPlan,
                                   Abi abi)
{
    const auto& source = definition.source;
    if(source.kind == KernelSourceKind::KPACK && !source.symbol.empty()
       && source.symbol != launchPlan.kernel)
    {
        const auto name = general_bwd::launchName(launchPlan.launch);
        failBadParam(describeDescriptor("kernel", definition.name, definition.kernelId)
                     + " launch '" + std::string(name.empty() ? "prep" : name) + "' is symbol '"
                     + source.symbol + "', but the plan of the graph names '"
                     + std::string(launchPlan.kernel) + "'");
    }
    auto code = buildIngestorKernelCode(kpackLoader, context, definition, signatureOf(abi));
    code.setBlockSize(launchPlan.block, 1, 1);
    code.setGridSize(launchPlan.grid[0], launchPlan.grid[1], launchPlan.grid[2]);
    return code;
}

void requireAligned(const void* pointer, int64_t alignment, std::string_view role)
{
    if(alignment > 1
       && reinterpret_cast<uintptr_t>(pointer) % static_cast<uintptr_t>(alignment) != 0)
    {
        failBadParam("the " + std::string(role) + " pointer is not " + std::to_string(alignment)
                     + "-byte aligned, the alignment the graph declares and the kernels read "
                       "it with");
    }
}

template <std::size_t... I>
void launchWords(const compilation::IRunnableKernel& kernel,
                 hipStream_t stream,
                 const uint64_t* words,
                 std::index_sequence<I...> /*positions*/)
{
    // Each argument is read at its own size from the start of its slot: the low half of a
    // slot holds an i32 or f32.
    kernel.launch(stream, words[I]...);
}

void launchImage(Abi abi,
                 const compilation::IRunnableKernel& kernel,
                 hipStream_t stream,
                 const std::vector<uint64_t>& words)
{
    switch(abi)
    {
    case Abi::PREP:
        launchWords(
            kernel, stream, words.data(), std::make_index_sequence<generated::ABI_PREP.size()>{});
        return;
    case Abi::MAIN:
        launchWords(
            kernel, stream, words.data(), std::make_index_sequence<generated::ABI_MAIN.size()>{});
        return;
    case Abi::CONVERT:
        launchWords(kernel,
                    stream,
                    words.data(),
                    std::make_index_sequence<generated::ABI_CONVERT.size()>{});
        return;
    case Abi::DQ:
        launchWords(
            kernel, stream, words.data(), std::make_index_sequence<generated::ABI_DQ.size()>{});
        return;
    default:
        failBadParam("unknown launch ABI");
    }
}

/**
 * @brief The native dispatch behind one slot's UDD.
 *
 * PREP launches the prep slot's single kernel. MAIN launches the main slot's kernels in plan
 * order: main, dq, then convert_dk and convert_dv only when dK/dV accumulate atomically.
 * Neither slot owns scratch: everything it shares lives in the graph's `ws.*` regions.
 *
 * launch() allocates nothing on the device, does not synchronize and reads no device data,
 * so it is HIP-graph capturable. Plans are not saved: supportsSaving() is the SDK default.
 */
template <Role ROLE>
class AttentionGeneralBwdDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit AttentionGeneralBwdDispatchHandler(const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    std::set<std::string> launchNames() const override
    {
        std::set<std::string> names;
        if constexpr(ROLE == Role::MAIN)
        {
            for(const auto name : MAIN_LAUNCHES)
            {
                names.emplace(name);
            }
        }
        return names;
    }

    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto graphPlan = graphPlanOf(context);
        const auto& plan = graphPlan.plan;

        const auto expected = general_bwd::tokensFor(plan);
        requireTokens(bound, expected);

        // The kernel the runtime selected must be the one the plan names: its kernargs are
        // the plan's.
        const std::string_view arch = general_bwd::processorOf(context);
        const bool selected
            = ROLE == Role::MAIN
                  ? fieldsMatch(MAIN_FIELDS, generated::MAIN_TOKEN_PREFIX, expected, kernel, arch)
                  : fieldsMatch(PREP_FIELDS, generated::PREP_TOKEN_PREFIX, expected, kernel, arch);
        if(!selected || isMainKernel(kernel) != (ROLE == Role::MAIN))
        {
            failBadParam(describeDescriptor("kernel", kernel.name, kernel.kernelId) + " is not the "
                         + (ROLE == Role::MAIN ? "main" : "prep")
                         + " kernel the plan of the graph names");
        }

        std::vector<PreparedLaunch> launches;
        launches.reserve(plan.launches.size());
        for(const auto& launchPlan : plan.launches)
        {
            if((launchPlan.launch == general_bwd::Launch::PREP) != (ROLE == Role::PREP))
            {
                continue;
            }
            const Abi abi = abiOf(launchPlan.launch, plan.row->dq);
            const std::string name(general_bwd::launchName(launchPlan.launch));
            auto code = ROLE == Role::PREP
                            ? buildLaunchCode(_kpackLoader, context, kernel, launchPlan, abi)
                            : buildLaunchCode(_kpackLoader,
                                              context,
                                              launchKernelDefinition(kernel, name),
                                              launchPlan,
                                              abi);
            auto kernargs = bindKernargs(launchPlan, abi, graphPlan);
            launches.push_back(PreparedLaunch{launchPlan.launch,
                                              abi,
                                              std::move(code),
                                              std::move(kernargs.words),
                                              std::move(kernargs.pointers)});
        }
        if(launches.empty())
        {
            failBadParam("the plan of the graph has no launch for this slot");
        }
        return std::make_unique<PreparedAttentionGeneralBwd>(std::move(launches));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedBwd = dynamic_cast<const PreparedAttentionGeneralBwd&>(prepared);
        if(workspace == nullptr)
        {
            failBadParam("the shared workspace is null");
        }
        requireAligned(workspace, static_cast<int64_t>(WORKSPACE_BASE_ALIGNMENT), "workspace");

        // Every pointer of every launch is resolved and checked before the first launch, so a
        // missing or misaligned buffer fails with nothing enqueued.
        std::vector<std::vector<uint64_t>> images;
        images.reserve(preparedBwd.launches().size());
        for(const auto& step : preparedBwd.launches())
        {
            auto words = step.words;
            for(const auto& pointer : step.pointers)
            {
                void* address = nullptr;
                if(pointer.uid.has_value())
                {
                    address = hipdnn_plugin_sdk::findDeviceBuffer(
                                  *pointer.uid, deviceBuffers, numDeviceBuffers)
                                  .ptr;
                }
                else
                {
                    address = static_cast<char*>(workspace) + pointer.offset;
                }
                requireAligned(address, pointer.alignment, pointer.role);
                words[pointer.position]
                    = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
            }
            images.push_back(std::move(words));
        }

        const auto stream = handle.getStream();
        for(std::size_t i = 0; i < images.size(); ++i)
        {
            const auto& step = preparedBwd.launches()[i];
            launchImage(step.abi, step.code.kernelForStream(stream), stream, images[i]);
        }
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& attentionGeneralBwdKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

/// Process-lifetime: the registry holds non-owning pointers to the two handlers, which share
/// one loader and the pack's module cache.
template <Role ROLE>
const AttentionGeneralBwdDispatchHandler<ROLE>& attentionGeneralBwdDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        attentionGeneralBwdKpackModuleCache());
    static const AttentionGeneralBwdDispatchHandler<ROLE> s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void resetAttentionGeneralBwdModuleCache()
{
    attentionGeneralBwdKpackModuleCache().clear();
}

void registerAttentionGeneralBwdSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &attentionGeneralBwdGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &attentionGeneralBwdKernelMatches);
    scope.add(std::string(PREP_GFX942_SYMBOL), &packArchMatches<ArchId::GFX942>);
    scope.add(std::string(PREP_GFX950_SYMBOL), &packArchMatches<ArchId::GFX950>);
    scope.add(std::string(MAIN_GFX942_SYMBOL), &packArchMatches<ArchId::GFX942>);
    scope.add(std::string(MAIN_GFX950_SYMBOL), &packArchMatches<ArchId::GFX950>);
    scope.add(std::string(MAIN_SPLITDQ_GFX950_SYMBOL), &packArchMatches<ArchId::GFX950>);
    scope.add(std::string(PREP_DISPATCH_SYMBOL), &attentionGeneralBwdDispatchHandler<Role::PREP>());
    scope.add(std::string(MAIN_DISPATCH_SYMBOL), &attentionGeneralBwdDispatchHandler<Role::MAIN>());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
