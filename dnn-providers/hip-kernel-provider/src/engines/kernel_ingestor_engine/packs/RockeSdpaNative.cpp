// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "compilation/RockeRecipe.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <map>
#include <mutex>

namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
namespace data = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::HipdnnPluginException;

void check(bool condition, const std::string& message)
{
    if(!condition)
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE, message);
}

std::optional<BoundTokens> matchGraph(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1
       || !archSupports({"gfx950"}, context.deviceProperties.gcnArchName))
        return std::nullopt;
    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data::NodeAttributes::SdpaAttributes
       || node.computeDataType() != data::DataType::FLOAT)
        return std::nullopt;
    const auto& attrs = node.attributesAs<data::SdpaAttributes>();
    // This pack implements only the exact dense inference semantics it advertises.
    if(attrs.attn_mask_tensor_uid().has_value() || attrs.scale_tensor_uid().has_value()
       || attrs.seq_len_q_tensor_uid().has_value() || attrs.seq_len_kv_tensor_uid().has_value()
       || attrs.seed_tensor_uid().has_value() || attrs.offset_tensor_uid().has_value()
       || attrs.dropout_mask_tensor_uid().has_value()
       || attrs.dropout_scale_tensor_uid().has_value()
       || attrs.page_table_k_tensor_uid().has_value() || attrs.page_table_v_tensor_uid().has_value()
       || attrs.block_mask_tensor_uid().has_value() || attrs.sink_token_tensor_uid().has_value()
       || attrs.descale_q_tensor_uid().has_value() || attrs.descale_k_tensor_uid().has_value()
       || attrs.descale_v_tensor_uid().has_value() || attrs.descale_s_tensor_uid().has_value()
       || attrs.scale_s_tensor_uid().has_value() || attrs.scale_o_tensor_uid().has_value()
       || attrs.stats_tensor_uid().has_value() || attrs.max_tensor_uid().has_value()
       || attrs.sum_exp_tensor_uid().has_value() || attrs.rng_dump_tensor_uid().has_value()
       || attrs.amax_s_tensor_uid().has_value() || attrs.amax_o_tensor_uid().has_value()
       || attrs.generate_stats().value_or(false) || attrs.alibi_mask() || attrs.padding_mask()
       || !attrs.causal_mask() || attrs.causal_mask_bottom_right()
       || attrs.dropout_probability().has_value() || attrs.left_bound().has_value()
       || attrs.right_bound().has_value() || attrs.max_seq_len_kv().has_value()
       || attrs.diagonal_alignment() != data::DiagonalAlignment::TOP_LEFT
       || attrs.mma_core_mode() != data::DataType::UNSET
       || attrs.implementation() != data::AttentionImplementation::AUTO
       || !attrs.attn_scale_value().has_value()
       || attrs.attn_scale_value().value() != 1.0f / std::sqrt(128.0f))
        return std::nullopt;
    const std::array<int64_t, 4> uids{
        attrs.q_tensor_uid(), attrs.k_tensor_uid(), attrs.v_tensor_uid(), attrs.o_tensor_uid()};
    const auto& tensors = context.graph.getTensorMap();
    int64_t sequence = 0;
    for(auto uid : uids)
    {
        const auto it = tensors.find(uid);
        if(it == tensors.end())
            return std::nullopt;
        const auto& tensor = *it->second;
        const auto* dims = tensor.dims();
        const auto* strides = tensor.strides();
        if(!dims || !strides || dims->size() != 4 || strides->size() != 4
           || tensor.data_type() != data::DataType::BFLOAT16)
            return std::nullopt;
        if(sequence == 0)
            sequence = dims->Get(2);
        if(sequence <= 0 || sequence > 1024 || dims->Get(0) != 1 || dims->Get(1) != 4
           || dims->Get(2) != sequence || dims->Get(3) != 128)
            return std::nullopt;
        const std::array<int64_t, 4> expected{sequence * 4 * 128, 128, 4 * 128, 1};
        for(size_t i = 0; i < 4; ++i)
            if(strides->Get(static_cast<uint32_t>(i)) != expected[i])
                return std::nullopt;
    }
    BoundTokens bound;
    bound["S"] = sequence;
    for(size_t i = 0; i < 4; ++i)
        bound[std::array<const char*, 4>{"q_ptr", "k_ptr", "v_ptr", "o_ptr"}[i]] = uids[i];
    return bound;
}

std::shared_ptr<const compilation::RecipeBytes> artifact(const KernelDefinition& kernel)
{
    check(kernel.source.recipe.has_value(), "recipe source payload absent");
    const auto origin = std::filesystem::weakly_canonical(kernel.originDirectory);
    const auto boundary
        = std::filesystem::weakly_canonical(kernel.treeRoot.empty() ? origin : kernel.treeRoot);
    const auto path = std::filesystem::weakly_canonical(origin / kernel.source.recipe->bundle);
    const auto relative = path.lexically_relative(boundary);
    check(!relative.empty() && *relative.begin() != ".." && !relative.is_absolute(),
          "recipe bundle lies outside descriptor tree");
    // One immutable snapshot is shared by admission and preparation. Fresh installs
    // are discovered in a new process; changing a loaded artifact is unsupported.
    static std::mutex mutex;
    static std::map<std::filesystem::path, std::shared_ptr<const compilation::RecipeBytes>>
        snapshots;
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = snapshots.find(path);
    if(found != snapshots.end())
        return found->second;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    check(file.good(), "cannot open recipe bundle: " + path.string());
    const auto size = file.tellg();
    check(size > 0 && size <= 64 * 1024 * 1024, "invalid recipe bundle size");
    auto bytes = std::make_shared<compilation::RecipeBytes>(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes->data()), size);
    check(file.good(), "cannot read recipe bundle: " + path.string());
    snapshots.emplace(path, bytes);
    return bytes;
}
int64_t boundInt(const BoundTokens& bound, const std::string& name)
{
    auto value = tryGetBoundInt(bound, name);
    check(value.has_value(), "missing SDPA binding: " + name);
    return *value;
}
bool matchKernel(const MatchContext& context,
                 const BoundTokens& bound,
                 const KernelDefinition& kernel)
{
    if(kernel.source.kind != KernelSourceKind::ROCKE_RECIPE)
        return false;
    return compilation::recipeAdmits(*artifact(kernel),
                                     kernel.source.recipe->recipeKey,
                                     context.deviceProperties.gcnArchName,
                                     boundInt(bound, "S"));
}
double score(const MatchContext&, const BoundTokens&, const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}
struct PreparedSdpa : PreparedDispatch
{
    BoundTokens bindings;
    compilation::RockeRecipe recipe;
    PreparedSdpa(const BoundTokens& bound,
                 const KernelDefinition& kernel,
                 const MatchContext& context)
        : bindings(bound)
        , recipe(*artifact(kernel),
                 kernel.source.recipe->recipeKey,
                 context.deviceProperties.gcnArchName,
                 boundInt(bound, "S"))
    {
    }
};
class Handler : public IKernelDispatchHandler<Handle>
{
public:
    bool supportsSourceKind(KernelSourceKind kind) const override
    {
        return kind == KernelSourceKind::ROCKE_RECIPE;
    }
    size_t workspaceBytes(const MatchContext&,
                          const BoundTokens&,
                          const KernelDefinition&) const override
    {
        return 0;
    }
    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        return std::make_unique<PreparedSdpa>(bound, kernel, context);
    }
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* buffers,
                uint32_t numBuffers,
                void*) const override
    {
        const auto& state = dynamic_cast<const PreparedSdpa&>(prepared);
        const auto* plan = state.recipe.plan();
        const size_t bytes = static_cast<size_t>(boundInt(state.bindings, "S")) * 4 * 128 * 2;
        std::map<std::string, void*> pointers;
        std::array<uintptr_t, 4> starts{};
        size_t n = 0;
        for(const auto* name : {"q_ptr", "k_ptr", "v_ptr", "o_ptr"})
        {
            const auto buffer = hipdnn_plugin_sdk::findDeviceBuffer(
                boundInt(state.bindings, name), buffers, numBuffers);
            const auto address = reinterpret_cast<uintptr_t>(buffer.ptr);
            check(address && address % 16 == 0 && address <= UINTPTR_MAX - bytes,
                  "invalid SDPA pointer alignment/range");
            for(size_t i = 0; i < n; ++i)
                check(address >= starts[i] + bytes || starts[i] >= address + bytes,
                      "SDPA buffers overlap");
            starts[n++] = address;
            pointers.emplace(name, buffer.ptr);
        }
        size_t size = rocke_launch_plan_kernarg_size(plan);
        std::vector<unsigned char> args(size, 0);
        const float scale = 1.0f / std::sqrt(128.0f);
        for(int i = 0; i < rocke_launch_plan_num_args(plan); ++i)
        {
            const auto* arg = rocke_launch_plan_arg(plan, i);
            check(arg && arg->name && arg->offset <= size && arg->size <= size - arg->offset,
                  "invalid recipe argument layout");
            if(arg->kind == ROCKE_ARG_POINTER)
            {
                const auto it = pointers.find(arg->name);
                check(it != pointers.end() && arg->size == sizeof(void*),
                      "unknown SDPA pointer argument");
                std::memcpy(args.data() + arg->offset, &it->second, sizeof(void*));
            }
            else
            {
                check(std::string(arg->name) == "scale" && arg->kind == ROCKE_ARG_F32
                          && arg->size == sizeof(float),
                      "unknown SDPA scalar argument");
                std::memcpy(args.data() + arg->offset, &scale, sizeof(float));
            }
        }
        void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                         args.data(),
                         HIP_LAUNCH_PARAM_BUFFER_SIZE,
                         &size,
                         HIP_LAUNCH_PARAM_END};
        state.recipe.launch(extra, handle.getStream());
    }
};
} // namespace
void registerRockeSdpaSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    static const Handler handler;
    scope.add("hipkernel.rocke_sdpa.graph_match", &matchGraph);
    scope.add("hipkernel.rocke_sdpa.kernel_match", &matchKernel);
    scope.add("hipkernel.rocke_sdpa.score", &score);
    scope.add("hipkernel.rocke_sdpa.dispatch", &handler);
}
} // namespace hip_kernel_provider::kernel_ingestor_engine
