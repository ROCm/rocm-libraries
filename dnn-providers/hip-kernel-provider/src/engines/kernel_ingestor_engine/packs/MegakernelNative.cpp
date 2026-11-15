// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// hipkernel:Megakernel serves a hipDNN custom op `megakernel.<model>` with a persistent
// decode megakernel: one precompiled code object per binary key, launched with one
// workgroup per CU. Built only with ENABLE_MEGAKERNEL_ENGINE.
//
// Payload (UTF-8 JSON object, schema 1):
//   {"schema": 1, "model": str, "src": str, "arch": str, "cus": int, "kv": str,
//    "block": int, "M": int, "tp": int, "rt": int, "lm_head": bool, "nlayers": int}
// Every key but `nlayers` is part of the binary key and must equal the KMD field of the
// same name on the kernel (`arch` is checked against the device; the kpack arch table
// checks the kernel). `nlayers` is a launch argument.
//
// Inputs: [tab, glob], two device buffers whose addresses the kernel takes by value.
// Kernel: kdec(tab: i64, glob: i64, nlayers: i32). Grid (cus, 1, 1), block
// (block_threads, 1, 1), dynamic LDS lds_bytes, all from the kernel's KMD record.
//
// Co-residency (the kernel spins on a grid barrier, so every workgroup must be resident):
// graph_match pins `cus` to the build device's multiProcessorCount. prepare() declines,
// and launch() refuses, when the device's CU count is not `cus` or the occupancy of the
// loaded function is below one workgroup per CU. launch() checks the device it resolves
// from the handle's stream, once per device and plan, so a plan built on one device does
// not launch on a same-arch device with fewer CUs. This does not protect against other
// work on the same CUs: the caller runs the decode step on one stream and other processes
// on the GPU are not supported.

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(ENABLE_MEGAKERNEL_ENGINE)

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>
#include <nlohmann/json.hpp>

#include "compilation/Kernel.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/MegakernelCoResidency.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

constexpr std::string_view GRAPH_MATCH_SYMBOL = "megakernel.match_custom_op";
constexpr std::string_view KERNEL_MATCH_SYMBOL = "megakernel.kernel_matches";
constexpr std::string_view DISPATCH_SYMBOL = "megakernel.dispatch";
constexpr std::string_view OP_ID_PREFIX = "megakernel.";
constexpr int64_t PAYLOAD_SCHEMA = 1;

// Bound tokens that are not payload keys.
constexpr std::string_view TAB_UID = "megakernel.tab_uid";
constexpr std::string_view GLOB_UID = "megakernel.glob_uid";
constexpr std::string_view NLAYERS = "nlayers";

// The binary key compared with the kernel's KMD record, in the order a decline names the
// first differing field. `src` is last: it differs whenever anything else does, so a
// specific field is the more useful report.
constexpr std::array<std::string_view, 3> STRING_KEYS = {"model", "kv", "src"};
constexpr std::array<std::string_view, 5> INT_KEYS = {"cus", "block", "M", "tp", "rt"};
constexpr std::string_view BOOL_KEY = "lm_head";
constexpr std::array<std::string_view, 9> KERNEL_KEY_ORDER
    = {"model", "cus", "kv", "block", "M", "tp", "rt", "lm_head", "src"};

// Launch geometry of one code object, from its KMD record.
constexpr std::string_view BLOCK_THREADS = "block_threads";
constexpr std::string_view LDS_BYTES = "lds_bytes";

/// The head of a decline line: "hipkernel:Megakernel declined <what> [<field>]: ". The
/// field is the stable key a log filter or test reads; the text after it is for people.
struct Declined
{
    std::string what;
    std::string_view field;
};

std::ostream& operator<<(std::ostream& out, const Declined& declined)
{
    return out << MEGAKERNEL_ENGINE_NAME << " declined " << declined.what << " [" << declined.field
               << "]: ";
}

const data_objects::CustomOpAttributes* customOpNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::CustomOpAttributes)
    {
        return nullptr;
    }
    return &node.attributesAs<data_objects::CustomOpAttributes>();
}

std::string toText(const MetadataValue& value)
{
    return std::visit(
        [](const auto& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, std::string>)
            {
                return v;
            }
            else if constexpr(std::is_same_v<T, bool>)
            {
                return v ? "true" : "false";
            }
            else if constexpr(std::is_same_v<T, int64_t> || std::is_same_v<T, double>)
            {
                return std::to_string(v);
            }
            else
            {
                return "<list>";
            }
        },
        value);
}

/// Reads payload key @p key of JSON type @p want into @p bound, or names it as the cause.
template <typename T>
bool bindKey(const nlohmann::json& payload,
             std::string_view key,
             BoundTokens& bound,
             std::string& cause)
{
    const auto it = payload.find(std::string(key));
    bool ok = it != payload.end();
    if(ok)
    {
        if constexpr(std::is_same_v<T, std::string>)
        {
            ok = it->is_string();
        }
        else if constexpr(std::is_same_v<T, bool>)
        {
            ok = it->is_boolean();
        }
        else
        {
            ok = it->is_number_integer();
        }
    }
    if(!ok)
    {
        cause = key;
        return false;
    }
    bound[std::string(key)] = it->get<T>();
    return true;
}

/// GRAPH scope. Total: every malformed payload declines and logs which key is wrong.
std::optional<BoundTokens> matchCustomOp(const MatchContext& context)
{
    const auto* op = customOpNode(context);
    if(op == nullptr || op->custom_op_id() == nullptr)
    {
        return std::nullopt;
    }
    const std::string_view opId = op->custom_op_id()->string_view();
    if(opId.substr(0, OP_ID_PREFIX.size()) != OP_ID_PREFIX)
    {
        return std::nullopt;
    }
    // From here on the graph is ours, so each decline says why.
    const auto decline = [&](std::string_view field, const auto& detail) {
        // Named: a braced initializer's comma would split the macro argument.
        const Declined head{"custom op '" + std::string(opId) + "'", field};
        HIPDNN_PLUGIN_LOG_INFO(head << detail);
        return std::nullopt;
    };

    const auto* inputs = op->input_tensor_uids();
    if(inputs == nullptr || inputs->size() != 2)
    {
        return decline("inputs", "expected exactly two inputs [tab, glob]");
    }
    const auto* bytes = op->data();
    if(bytes == nullptr || bytes->empty())
    {
        return decline("payload", "no payload");
    }
    const auto payload = nlohmann::json::parse(bytes->begin(), bytes->end(), nullptr, false);
    if(payload.is_discarded() || !payload.is_object())
    {
        return decline("payload", "payload is not a JSON object");
    }
    const auto schema = payload.find("schema");
    if(schema == payload.end() || !schema->is_number_integer()
       || schema->get<int64_t>() != PAYLOAD_SCHEMA)
    {
        return decline("schema",
                       "payload schema "
                           + (schema == payload.end() ? std::string("missing") : schema->dump())
                           + ", this engine reads schema " + std::to_string(PAYLOAD_SCHEMA));
    }

    BoundTokens bound;
    std::string cause;
    bool ok = bindKey<std::string>(payload, "arch", bound, cause);
    for(const auto key : STRING_KEYS)
    {
        ok = ok && bindKey<std::string>(payload, key, bound, cause);
    }
    for(const auto key : INT_KEYS)
    {
        ok = ok && bindKey<int64_t>(payload, key, bound, cause);
    }
    ok = ok && bindKey<bool>(payload, BOOL_KEY, bound, cause);
    ok = ok && bindKey<int64_t>(payload, NLAYERS, bound, cause);
    if(!ok)
    {
        return decline(cause, "payload key '" + cause + "' is missing or has the wrong type");
    }

    const auto& model = std::get<std::string>(bound.at("model"));
    if(opId.substr(OP_ID_PREFIX.size()) != model)
    {
        return decline("model", "payload model '" + model + "' is not the op id's");
    }
    const auto& arch = std::get<std::string>(bound.at("arch"));
    const auto deviceArch
        = hipdnn_plugin_sdk::stripArchFeatures(context.deviceProperties.gcnArchName);
    if(arch != deviceArch)
    {
        return decline("arch", "payload arch " + arch + ", device arch " + std::string(deviceArch));
    }
    const auto cus = std::get<int64_t>(bound.at("cus"));
    if(cus != context.deviceProperties.multiProcessorCount)
    {
        return decline("cus",
                       "payload cus " + std::to_string(cus) + ", device multiProcessorCount "
                           + std::to_string(context.deviceProperties.multiProcessorCount));
    }
    const auto nlayers = std::get<int64_t>(bound.at(std::string(NLAYERS)));
    if(nlayers <= 0 || nlayers > std::numeric_limits<int32_t>::max())
    {
        return decline("nlayers", "nlayers " + std::to_string(nlayers) + " is not a positive i32");
    }

    bound[std::string(TAB_UID)] = inputs->Get(0);
    bound[std::string(GLOB_UID)] = inputs->Get(1);
    return bound;
}

const MetadataValue& requireBound(const BoundTokens& bound, std::string_view token)
{
    const auto it = bound.find(std::string(token));
    if(it == bound.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       std::string(MEGAKERNEL_ENGINE_NAME)
                                                           + ": missing bound token '"
                                                           + std::string(token) + "'");
    }
    return it->second;
}

int64_t boundInt(const BoundTokens& bound, std::string_view token)
{
    return std::get<int64_t>(requireBound(bound, token));
}

/// KERNEL scope: every binary-key field of the kernel equals the payload's. A kernel of
/// another model is skipped silently; for one of the payload's model, the first differing
/// field is logged, so a missing binary reads as "no code object for M=8", not as a bare
/// "no engine configurations".
bool kernelMatches(const MatchContext& /*context*/,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    for(const auto field : KERNEL_KEY_ORDER)
    {
        const std::string name(field);
        const auto have = kernel.tryGetMetadata(name);
        const auto& want = requireBound(bound, field);
        if(have.has_value() && *have == want)
        {
            continue;
        }
        if(field != "model")
        {
            const Declined head{"kernel '" + kernel.name + "'", field};
            HIPDNN_PLUGIN_LOG_INFO(head << "payload " << name << " " << toText(want) << ", kernel "
                                        << (have.has_value() ? toText(*have) : "<none>"));
        }
        return false;
    }
    return true;
}

/// The loaded function behind @p runnable, for its occupancy query.
const compilation::Kernel& loadedKernel(const compilation::IRunnableKernel& runnable,
                                        const std::string& kernelName)
{
    const auto* loaded = dynamic_cast<const compilation::Kernel*>(&runnable);
    if(loaded == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(MEGAKERNEL_ENGINE_NAME) + ": kernel '" + kernelName
                + "' did not resolve to a loaded function");
    }
    return *loaded;
}

/// Workgroups of @p threads threads and @p lds bytes of dynamic LDS that fit per CU. A
/// failed HIP query is a runtime fault (INTERNAL_ERROR), not "does not fit".
int blocksPerCu(const compilation::Kernel& loaded,
                unsigned threads,
                unsigned lds,
                const std::string& kernelName)
{
    int blocks = 0;
    const auto status = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, loaded.function(), static_cast<int>(threads), lds);
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string(MEGAKERNEL_ENGINE_NAME) + ": occupancy query for kernel '" + kernelName
                + "' failed: " + hipGetErrorString(status));
    }
    return blocks;
}

/// Throws NOT_APPLICABLE, with the decline head, when the grid cannot be co-resident.
void requireCoResident(int64_t cus,
                       int deviceCus,
                       int blocks,
                       const std::string& kernelName,
                       unsigned threads,
                       unsigned lds)
{
    const auto problem = megakernel::coResidencyProblem(cus, deviceCus, blocks);
    if(problem.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
            std::string(MEGAKERNEL_ENGINE_NAME) + " declined kernel '" + kernelName + "' ["
                + std::string(problem->field) + "]: " + problem->detail + " ("
                + std::to_string(threads) + " threads, " + std::to_string(lds) + " B dynamic LDS)");
    }
}

class PreparedMegakernel : public IngestorPreparedDispatch
{
public:
    struct Launch
    {
        int64_t tabUid;
        int64_t globUid;
        int32_t nlayers;
        int64_t cus;
        unsigned threads;
        unsigned lds;
        std::string kernelName;
    };

    PreparedMegakernel(IngestorKernelCode code, Launch launch, int checkedDevice)
        : _code(std::move(code))
        , _launch(std::move(launch))
        , _checkedDevices{checkedDevice}
    {
    }

    const IngestorKernelCode& kernelCode() const override
    {
        return _code;
    }

    const Launch& launch() const
    {
        return _launch;
    }

    /// The kernel to launch on @p deviceOrdinal, after the co-residency check of that
    /// device. The check runs once per device: its answer does not change.
    compilation::IRunnableKernel& kernelForDevice(int deviceOrdinal) const
    {
        auto& kernel = _code.kernelFor(deviceOrdinal);
        const std::lock_guard<std::mutex> lock(_mutex);
        if(_checkedDevices.count(deviceOrdinal) != 0)
        {
            return kernel;
        }
        int deviceCus = 0;
        const auto status = hipDeviceGetAttribute(
            &deviceCus, hipDeviceAttributeMultiprocessorCount, deviceOrdinal);
        if(status != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string(MEGAKERNEL_ENGINE_NAME) + ": CU count query for device "
                    + std::to_string(deviceOrdinal) + " failed: " + hipGetErrorString(status));
        }
        // kernelFor() admitted only the plan's arch, so the KMD launch geometry holds.
        const auto blocks = blocksPerCu(loadedKernel(kernel, _launch.kernelName),
                                        _launch.threads,
                                        _launch.lds,
                                        _launch.kernelName);
        requireCoResident(
            _launch.cus, deviceCus, blocks, _launch.kernelName, _launch.threads, _launch.lds);
        _checkedDevices.insert(deviceOrdinal);
        return kernel;
    }

private:
    IngestorKernelCode _code;
    Launch _launch;
    mutable std::mutex _mutex;
    mutable std::set<int> _checkedDevices;
};

/// kdec(tab: Int64, glob: Int64, nlayers: Int32), all by value.
const std::vector<KernelArgument>& megakernelSignature()
{
    static const std::vector<KernelArgument> s_signature{
        {"by_value", 8, 0, ""}, {"by_value", 8, 0, ""}, {"by_value", 4, 0, ""}};
    return s_signature;
}

unsigned launchDim(const KernelDefinition& kernel, std::string_view field)
{
    const auto value = kernel.getIntMetadata(std::string(field));
    if(value < 0 || value > std::numeric_limits<int32_t>::max())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            std::string(MEGAKERNEL_ENGINE_NAME) + ": kernel '" + kernel.name + "' has "
                + std::string(field) + " " + std::to_string(value));
    }
    return static_cast<unsigned>(value);
}

class MegakernelDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit MegakernelDispatchHandler(const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
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
        const auto cus = boundInt(bound, "cus");
        const auto threads = launchDim(kernel, BLOCK_THREADS);
        const auto lds = launchDim(kernel, LDS_BYTES);

        auto code = buildIngestorKernelCode(_kpackLoader, context, kernel, megakernelSignature());
        code.setGridSize(static_cast<unsigned>(cus));
        code.setBlockSize(threads);
        code.setSharedMemBytes(lds);

        const auto blocks = blocksPerCu(
            loadedKernel(code.kernelFor(context.deviceId), kernel.name), threads, lds, kernel.name);
        requireCoResident(
            cus, context.deviceProperties.multiProcessorCount, blocks, kernel.name, threads, lds);

        PreparedMegakernel::Launch launch{boundInt(bound, TAB_UID),
                                          boundInt(bound, GLOB_UID),
                                          static_cast<int32_t>(boundInt(bound, NLAYERS)),
                                          cus,
                                          threads,
                                          lds,
                                          kernel.name};
        return std::make_unique<PreparedMegakernel>(
            std::move(code), std::move(launch), context.deviceId);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& mk = dynamic_cast<const PreparedMegakernel&>(prepared);
        const auto& args = mk.launch();
        // Resolve the device as kernelForStream() does, then re-check co-residency there.
        auto& kernel = mk.kernelForDevice(launchDeviceOrdinal(handle.getStream()));
        const auto tab
            = hipdnn_plugin_sdk::findDeviceBuffer(args.tabUid, deviceBuffers, numDeviceBuffers);
        const auto glob
            = hipdnn_plugin_sdk::findDeviceBuffer(args.globUid, deviceBuffers, numDeviceBuffers);
        const auto tabAddress = static_cast<int64_t>(reinterpret_cast<intptr_t>(tab.ptr));
        const auto globAddress = static_cast<int64_t>(reinterpret_cast<intptr_t>(glob.ptr));
        kernel.launch(handle.getStream(), tabAddress, globAddress, args.nlayers);
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

compilation::KpackModuleCache& megakernelKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

const MegakernelDispatchHandler& megakernelDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(megakernelKpackModuleCache());
    static const MegakernelDispatchHandler s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void resetMegakernelModuleCache()
{
    megakernelKpackModuleCache().clear();
}

void registerMegakernelSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCH_SYMBOL), &matchCustomOp);
    scope.add(std::string(KERNEL_MATCH_SYMBOL), &kernelMatches);
    scope.add(std::string(DISPATCH_SYMBOL), &megakernelDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif
