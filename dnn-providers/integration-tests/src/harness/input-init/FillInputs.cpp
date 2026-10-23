// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/input-init/FillInputs.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>

#if defined(USE_ROCRAND)
#include <hip/hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFpReferenceCommon.hpp>
#endif

namespace hipdnn_integration_tests
{
namespace
{

using TensorDataTypes = std::unordered_map<int64_t, hipdnn_flatbuffers_sdk::data_objects::DataType>;

TensorDataTypes collectDataTypes(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
{
    TensorDataTypes dataTypes;
    if(graph.tensors() == nullptr)
    {
        return dataTypes;
    }
    for(const auto* tensor : *graph.tensors())
    {
        dataTypes.emplace(tensor->uid(), tensor->data_type());
    }
    return dataTypes;
}

// Symmetric range over every finite value of T, so a uniform draw rounds onto
// each representable magnitude, including the extremes.
template <typename T>
FillRecipe symmetricFullRange()
{
    const auto max = static_cast<float>(std::numeric_limits<T>::max());
    return FillRecipe::free(-max, max);
}

// ── Fill dispatch ───────────────────────────────────────────────────────────

FillResult fill(hipdnn_data_sdk::utilities::ITensor& tensor,
                const FillRecipe& recipe,
                unsigned int seed,
                DeviceInputFiller* device)
{
    switch(recipe.kind)
    {
    case FillRecipe::Kind::FREE:
        if(device != nullptr && device->tryFill(tensor, recipe, seed))
        {
            return FillResult::ok(/*deviceFilled=*/1);
        }
        tensor.fillTensorWithRandomValues(recipe.lo, recipe.hi, seed);
        return FillResult::ok();
    case FillRecipe::Kind::FIXED:
        tensor.fillTensorWithValue(recipe.value);
        return FillResult::ok();
    default:
        return FillResult::unsupported("unknown FillRecipe kind");
    }
}

// ── Per-op init defaults ────────────────────────────────────────────────────
// Each function sets defaults for one node type via recipes.setDefault().
// setDefault uses try_emplace — if the test already set() a uid, the default
// is silently skipped.

// ── Batchnorm ────────────────────────────────────────────────────────────────

void setBatchnormInferenceInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
}

void setBatchnormInferenceVarianceInitDefaults(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node, InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setBatchnormTrainingInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
    recipes.setDefault(a->prev_running_mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->prev_running_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
    recipes.setDefault(a->momentum_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

void setBatchnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(-0.1f, 0.1f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.5f, 1.5f));
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

void setLayernormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                              InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setLayernormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->mean_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->inv_variance_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

// ── RMSNorm ──────────────────────────────────────────────────────────────────

void setRmsnormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                            InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->epsilon_tensor_uid(), FillRecipe::fixed(1e-5f));
}

void setRmsnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                    InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    recipes.setDefault(a->inv_rms_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

// ── Block-scale quantization ─────────────────────────────────────────────────

// Largest magnitude a recipe can produce.
float recipeMagnitude(const FillRecipe& recipe)
{
    if(recipe.kind == FillRecipe::Kind::FIXED)
    {
        return std::fabs(recipe.value);
    }
    return std::max(std::fabs(recipe.lo), std::fabs(recipe.hi));
}

void setBlockScaleDequantizeInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                         InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return;
    }

    // Scale is normalized against the operand range in effect (its data-type
    // default or a test override), as OCP MX does: scale ~ 2^-floor(log2(amax)) for
    // amax >= 1. The dequantized block then peaks between 2 and 4 (FP4/FP6:
    // 3 to 3.75), so products stay within FP16 range; an operand with
    // amax < 1 keeps e = 0 and peaks at 2 * amax. UE8M0 has no mantissa bits,
    // so the [0.5, 2] * 2^-e draw discretizes to three powers of two. An
    // operand in [-1, 1] gives e = 0, i.e. [0.5, 2].
    const float amax = recipeMagnitude(recipes.fill(a->x_tensor_uid()));
    const int e = amax >= 1.0f ? std::ilogb(amax) : 0;
    recipes.setDefault(a->scale_tensor_uid(),
                       FillRecipe::free(std::ldexp(0.5f, -e), std::ldexp(2.0f, -e)));
}

// ── SDPA ─────────────────────────────────────────────────────────────────────

void setSdpaForwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(a == nullptr)
    {
        return;
    }

    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
}

void setSdpaBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                 InputFillRecipes& recipes)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }

    recipes.setDefault(a->scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
    recipes.setDefault(a->dropout_scale_tensor_uid(), FillRecipe::free(0.1f, 1.0f));
    recipes.setDefault(a->dropout_scale_inv_tensor_uid(), FillRecipe::free(0.1f, 1.0f));

    recipes.setDefault(a->o_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
    recipes.setDefault(a->stats_tensor_uid(), FillRecipe::free(0.0f, 1.0f));
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

bool applyDefaultFills(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                       InputFillRecipes& recipes)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::BatchnormInferenceAttributes:
        setBatchnormInferenceInitDefaults(node, recipes);
        return true;
    case NA::BatchnormInferenceAttributesVarianceExt:
        setBatchnormInferenceVarianceInitDefaults(node, recipes);
        return true;
    case NA::BatchnormAttributes:
        setBatchnormTrainingInitDefaults(node, recipes);
        return true;
    case NA::BatchnormBackwardAttributes:
        setBatchnormBackwardInitDefaults(node, recipes);
        return true;
    case NA::LayernormAttributes:
        setLayernormInitDefaults(node, recipes);
        return true;
    case NA::LayernormBackwardAttributes:
        setLayernormBackwardInitDefaults(node, recipes);
        return true;
    case NA::RMSNormAttributes:
        setRmsnormInitDefaults(node, recipes);
        return true;
    case NA::RMSNormBackwardAttributes:
        setRmsnormBackwardInitDefaults(node, recipes);
        return true;
    case NA::BlockScaleDequantizeAttributes:
        setBlockScaleDequantizeInitDefaults(node, recipes);
        return true;
    case NA::SdpaAttributes:
        setSdpaForwardInitDefaults(node, recipes);
        return true;
    case NA::SdpaBackwardAttributes:
        setSdpaBackwardInitDefaults(node, recipes);
        return true;
    // Ops whose inputs need no special init; they get defaultFillFor(dtype).
    case NA::PointwiseAttributes:
    case NA::ConvolutionFwdAttributes:
    case NA::ConvolutionBwdAttributes:
    case NA::ConvolutionWrwAttributes:
    case NA::MatmulAttributes:
    case NA::ReductionAttributes:
    case NA::ResampleFwdAttributes:
    case NA::ResampleBwdAttributes:
    case NA::BlockScaleQuantizeAttributes:
    case NA::CustomOpAttributes:
    case NA::MoeGroupedMatmulAttributes:
    case NA::MoeGroupedMatmulBwdAttributes:
    case NA::NONE:
        return true;
    default:
        return false;
    }
}

} // anonymous namespace

#if defined(USE_ROCRAND)
using RocRandGenerator = hipdnn_gpu_ref::common::RocRandGenerator;
#endif

struct DeviceInputFiller::Impl
{
#if defined(USE_ROCRAND)
    std::unique_ptr<RocRandGenerator> generator;
    bool fillPending = false;

    // Starts the fill when `tensor` is a T tensor; creates the generator on first use,
    // so a tensor of a type rocRAND does not fill never costs one.
    template <class T>
    bool fillOnDevice(hipdnn_data_sdk::utilities::ITensor& tensor,
                      const FillRecipe& recipe,
                      unsigned int seed)
    {
        auto* typed = dynamic_cast<hipdnn_data_sdk::utilities::TensorBase<T>*>(&tensor);
        if(typed == nullptr)
        {
            return false;
        }

        if(generator == nullptr)
        {
            generator = std::make_unique<RocRandGenerator>(ROCRAND_RNG_PSEUDO_DEFAULT);
        }

        // Set before the launch: a fill that throws partway may already have queued work.
        fillPending = true;
        hipdnn_gpu_ref::common::gpu_fp_reference_tensor::gpuFillWithRandomValues<T>(
            *typed,
            static_cast<T>(recipe.lo),
            static_cast<T>(recipe.hi),
            seed,
            *generator,
            /*synchronize=*/false);
        return true;
    }
#endif
};

DeviceInputFiller::DeviceInputFiller()
    : _impl(std::make_unique<Impl>())
{
}

DeviceInputFiller::~DeviceInputFiller()
{
    // The generator is about to be destroyed; a fill must not still be using it.
    try
    {
        waitForFills();
    }
    catch(...) // NOLINT(bugprone-empty-catch): nothing useful to do about it here
    {
    }
}

bool DeviceInputFiller::isSupported()
{
#if defined(USE_ROCRAND)
    return true;
#else
    return false;
#endif
}

bool DeviceInputFiller::tryFill([[maybe_unused]] hipdnn_data_sdk::utilities::ITensor& tensor,
                                [[maybe_unused]] const FillRecipe& recipe,
                                [[maybe_unused]] unsigned int seed)
{
#if defined(USE_ROCRAND)
    if(tensor.elementSpace() < minElements())
    {
        return false;
    }

    try
    {
        auto& impl = *_impl;
        return impl.fillOnDevice<float>(tensor, recipe, seed)
               || impl.fillOnDevice<hipdnn_data_sdk::types::half>(tensor, recipe, seed)
               || impl.fillOnDevice<hipdnn_data_sdk::types::bfloat16>(tensor, recipe, seed)
               || impl.fillOnDevice<double>(tensor, recipe, seed);
    }
    catch(const std::exception& e)
    {
        // Generator creation, the scaling kernel's compile, scratch allocation and the
        // launches are all device work; none of it is a fault in the graph under test.
        throw DeviceInputError(std::string("device input fill failed: ") + e.what());
    }
#else
    return false;
#endif
}

void DeviceInputFiller::waitForFills()
{
#if defined(USE_ROCRAND)
    if(!_impl->fillPending)
    {
        return;
    }
    _impl->fillPending = false;

    const hipError_t status = hipDeviceSynchronize();
    if(status != hipSuccess)
    {
        throw DeviceInputError(std::string("device input fill failed: ")
                               + hipGetErrorString(status));
    }
#endif
}

namespace
{

// Waits for the device fills on every way out of fillInputs(), including an exception
// from a fill that threw partway: the tensors are about to be dropped, and fills
// already queued must not still be writing into memory being freed. Errors are
// swallowed here, since the one already propagating is the one worth reporting; the
// normal path waits explicitly and sees them.
class WaitForFillsOnExit
{
public:
    explicit WaitForFillsOnExit(DeviceInputFiller* device)
        : _device(device)
    {
    }

    ~WaitForFillsOnExit()
    {
        if(_device == nullptr)
        {
            return;
        }
        try
        {
            _device->waitForFills();
        }
        catch(...) // NOLINT(bugprone-empty-catch): see above
        {
        }
    }

    WaitForFillsOnExit(const WaitForFillsOnExit&) = delete;
    WaitForFillsOnExit& operator=(const WaitForFillsOnExit&) = delete;
    WaitForFillsOnExit(WaitForFillsOnExit&&) = delete;
    WaitForFillsOnExit& operator=(WaitForFillsOnExit&&) = delete;

private:
    DeviceInputFiller* _device;
};

} // anonymous namespace

FillRecipe defaultFillFor(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    namespace types = hipdnn_data_sdk::types;

    switch(dataType)
    {
    case DataType::FLOAT:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::DOUBLE:
        return FillRecipe{};
    // Few enough codes that a uniform draw over the full range reaches all of them;
    // [-1, 1] would only reach 3 of the 8 FP4 magnitudes.
    case DataType::FP4_E2M1:
        return symmetricFullRange<types::fp4_e2m1>();
    case DataType::FP6_E2M3:
        return symmetricFullRange<types::fp6_e2m3>();
    case DataType::FP6_E3M2:
        return symmetricFullRange<types::fp6_e3m2>();
    // FP8 codes are spread logarithmically, so no uniform range reaches most of
    // them: a wide range leaves the small exponents empty. Full coverage needs a
    // log-uniform or code-uniform distribution, which ITensor cannot draw yet.
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return FillRecipe{};
    // Power-of-two scale; [0.5, 2] discretizes to {0.5, 1, 2}.
    case DataType::FP8_E8M0:
        return FillRecipe::free(0.5f, 2.0f);
    // Integer draws from [-1, 1] collapse to a few values. A meaningful range is
    // op-specific (indices, offsets, masks), so it belongs in per-op defaults.
    case DataType::UINT8:
    case DataType::INT32:
    case DataType::INT8:
    case DataType::INT4:
    case DataType::INT64:
    case DataType::BOOLEAN:
        return FillRecipe{};
    case DataType::UNSET:
    default:
        throw std::invalid_argument("defaultFillFor: no fill default for data type "
                                    + std::string(EnumNameDataType(dataType)));
    }
}

FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes,
                      DeviceInputFiller* device)
{
    // Dtype defaults sit below test and per-op recipes. Only those that differ
    // from the generic FREE[-1, 1] are recorded, so meta.inputs stays unchanged
    // for float graphs and names the range for narrow formats. Registered
    // before the per-op pass so a default derived from another input's range
    // (block-scale dequantize's scale) sees that input's final range.
    const auto dataTypes = collectDataTypes(graph);
    for(const int64_t uid : ownedUids)
    {
        const auto dataType = dataTypes.find(uid);
        if(dataType == dataTypes.end())
        {
            continue;
        }
        const auto recipe = defaultFillFor(dataType->second);
        if(recipe != FillRecipe{})
        {
            recipes.setTypeDefault(uid, recipe);
        }
    }

    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        const auto& node = *graph.nodes()->Get(i);
        if(!applyDefaultFills(node, recipes))
        {
            const auto* name = node.name();
            return FillResult::unsupported(
                "no input fill registered for op "
                + std::string(name != nullptr ? name->c_str() : "(unnamed)"));
        }
    }

    // Sort so the rng sequence is deterministic regardless of discovery order.
    auto sortedUids = ownedUids;
    std::sort(sortedUids.begin(), sortedUids.end());

    std::mt19937 rng(recipes.globalSeed());

    const WaitForFillsOnExit waitOnExit(device);

    std::size_t deviceFilled = 0;
    for(const int64_t uid : sortedUids)
    {
        const unsigned int seed
            = recipes.resolveSeed(uid).value_or(static_cast<unsigned int>(rng()));
        auto fillResult = fill(*inputs.at(uid), recipes.fill(uid), seed, device);
        if(!fillResult.filled)
        {
            return FillResult::unsupported("uid " + std::to_string(uid) + ": " + fillResult.reason);
        }
        deviceFilled += fillResult.deviceFilled;
    }

    // One wait for every device fill, instead of one per tensor.
    if(device != nullptr)
    {
        device->waitForFills();
    }
    return FillResult::ok(deviceFilled);
}

} // namespace hipdnn_integration_tests
