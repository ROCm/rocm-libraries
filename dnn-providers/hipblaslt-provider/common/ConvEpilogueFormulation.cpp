// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/ConvEpilogueFormulation.hpp"

#include "common/GemmFormulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace hipblaslt_plugin
{

namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace conv_epilogue = hipdnn_plugin_sdk::conv_epilogue;
using conv_epilogue::OpCode;
using data_objects::DataType;

/// Workspace tensors start on this boundary, as the repack plans' own do.
constexpr size_t ALIGNMENT = 256;

size_t alignUp(size_t value)
{
    return (value + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}

int operandTypeCode(DataType dataType)
{
    switch(dataType)
    {
    case DataType::HALF:
        return 1;
    case DataType::BFLOAT16:
        return 2;
    default:
        return 0;
    }
}

/// The run-time menu form of @p epilogue (ConvRepackEpilogue), its values the plan-time ones;
/// operand pointers and runtime values are filled at execution.
ConvRepackEpilogue repackEpilogue(const conv_epilogue::ConvEpilogue& epilogue, int channelAxis)
{
    ConvRepackEpilogue e{};
    for(size_t i = 0; i < epilogue.operands.size() && i < CONV_REPACK_EPI_OPERANDS; ++i)
    {
        e.operandType[i] = operandTypeCode(epilogue.operands[i].dataType);
        e.operandPerChannel[i] = epilogue.operands[i].perChannel ? 1 : 0;
    }
    for(size_t i = 0; i < epilogue.scalars.size() && i < CONV_REPACK_EPI_SCALARS; ++i)
    {
        e.scalar[i] = epilogue.scalars[i].value;
    }
    for(size_t i = 0; i < epilogue.ops.size() && i < CONV_REPACK_EPI_OPS; ++i)
    {
        e.op[i] = static_cast<int>(epilogue.ops[i].code);
        e.slot[i] = epilogue.ops[i].slot;
    }
    e.ops = static_cast<int>(std::min<size_t>(epilogue.ops.size(), CONV_REPACK_EPI_OPS));
    e.channelAxis = channelAxis;
    return e;
}

/// The epilogue cast over @p dims from @p srcStrides (fp32) to @p dstStrides; channels on axis 1.
StridedCastEpilogueArgs epilogueCast(const ConvProblem& problem,
                                     const std::vector<int64_t>& dims,
                                     const std::vector<int64_t>& srcStrides,
                                     const std::vector<int64_t>& dstStrides)
{
    StridedCastEpilogueArgs args{};
    args.cast.rank = static_cast<int>(dims.size());
    args.cast.total = 1;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        args.cast.dims[axis] = dims[axis];
        args.cast.srcStrides[axis] = srcStrides[axis];
        args.cast.dstStrides[axis] = dstStrides[axis];
        args.cast.total *= dims[axis];
    }
    args.cast.batches = 1;
    args.epilogue = repackEpilogue(problem.epilogue, 1);
    return args;
}

int64_t freshUid(const ConvProblem& problem, const ConvGemmPlanCalls& plan, int64_t after)
{
    int64_t last = std::max({problem.x.uid, problem.w.uid, problem.y.uid, after});
    for(const auto& operand : problem.epilogue.operands)
    {
        last = std::max(last, operand.uid);
    }
    for(const auto& tensor : plan.workspace)
    {
        last = std::max(last, tensor.uid);
    }
    return last + 16;
}

bool nativeRows(const ConvProblem& problem, const GemmCall& call)
{
    // The rows of d are y's channels: one group, channels unit-stride, m all of them.
    // The libraries' limits must leave the call whole: the epilogue applies to it once.
    return problem.groups == 1 && call.batchCount == 1 && !call.accumulate
           && (problem.k == 1 || problem.y.strides[1] == 1) && call.m == problem.k
           && fitLibraryLimits({call}).size() == 1;
}

std::optional<ConvGemmPlanCalls>
    fuseNative(const ConvProblem& problem, ConvGemmPlanCalls plan, const NativeEpilogue& native)
{
    bool any = false;
    for(const auto& stage : plan.stages)
    {
        for(const auto& step : stage.after)
        {
            if(step.dstUid == problem.y.uid)
            {
                return std::nullopt;
            }
        }
        for(const auto& step : stage.before)
        {
            if(step.dstUid == problem.y.uid)
            {
                return std::nullopt;
            }
        }
        for(const auto& call : stage.calls)
        {
            if(call.d.uid == problem.y.uid)
            {
                if(!nativeRows(problem, call))
                {
                    return std::nullopt;
                }
                any = true;
            }
        }
    }
    if(!any)
    {
        return std::nullopt;
    }
    for(auto& stage : plan.stages)
    {
        for(auto& call : stage.calls)
        {
            if(call.d.uid == problem.y.uid)
            {
                call.epilogue = native;
            }
        }
    }
    plan.epilogueForm = EpilogueForm::NATIVE;
    return plan;
}

/// The staged channels-last y: the copies into y become epilogue casts of an fp32 staging.
std::optional<ConvGemmPlanCalls>
    fuseIntoStagedCopy(const ConvProblem& problem,
                       ConvGemmPlanCalls plan,
                       const std::shared_ptr<const conv_epilogue::ConvEpilogue>& epilogue)
{
    const int64_t bytes = elementBytes(problem.dataType);
    std::vector<int64_t> staged;
    for(const auto& stage : plan.stages)
    {
        for(const auto& call : stage.calls)
        {
            if(call.d.uid == problem.y.uid)
            {
                return std::nullopt;
            }
        }
        for(const auto* steps : {&stage.before, &stage.after})
        {
            for(const auto& step : *steps)
            {
                if(step.dstUid != problem.y.uid)
                {
                    continue;
                }
                const auto* copy = std::get_if<StridedCopyArgs>(&step.args);
                if(copy == nullptr || static_cast<size_t>(copy->rank) != problem.y.dims.size()
                   || copy->dims[1] != problem.k || step.srcByteOffset % bytes != 0)
                {
                    return std::nullopt;
                }
                if(std::find(staged.begin(), staged.end(), step.srcUid) == staged.end())
                {
                    staged.push_back(step.srcUid);
                }
            }
        }
    }
    if(staged.empty())
    {
        return std::nullopt;
    }
    // Each staged tensor gets an fp32 twin at the end of the workspace; the GEMMs write it.
    std::vector<std::pair<int64_t, int64_t>> twins;
    int64_t next = freshUid(problem, plan, 0);
    for(const auto uid : staged)
    {
        const auto found = std::find_if(plan.workspace.begin(),
                                        plan.workspace.end(),
                                        [uid](const WorkspaceTensor& t) { return t.uid == uid; });
        if(found == plan.workspace.end())
        {
            return std::nullopt;
        }
        const size_t offset = alignUp(plan.workspaceBytes);
        const size_t twinBytes = found->bytes / static_cast<size_t>(bytes) * sizeof(float);
        plan.workspace.push_back(WorkspaceTensor{next, offset, twinBytes});
        plan.workspaceBytes = offset + twinBytes;
        twins.emplace_back(uid, next);
        ++next;
    }
    const auto twinOf = [&twins](int64_t uid) -> std::optional<int64_t> {
        for(const auto& [from, to] : twins)
        {
            if(from == uid)
            {
                return to;
            }
        }
        return std::nullopt;
    };
    for(auto& stage : plan.stages)
    {
        for(auto& call : stage.calls)
        {
            if(const auto twin = twinOf(call.d.uid))
            {
                call.d.uid = *twin;
                call.d.dataType = DataType::FLOAT;
            }
            // Nothing else reads the staging forward.
            if(twinOf(call.a.uid) || twinOf(call.b.uid))
            {
                return std::nullopt;
            }
        }
        for(auto* steps : {&stage.before, &stage.after})
        {
            for(auto& step : *steps)
            {
                const auto twin = twinOf(step.srcUid);
                if(twinOf(step.dstUid))
                {
                    return std::nullopt;
                }
                if(!twin.has_value())
                {
                    continue;
                }
                if(step.dstUid != problem.y.uid)
                {
                    return std::nullopt;
                }
                const auto copy = std::get<StridedCopyArgs>(step.args);
                const std::vector<int64_t> dims(copy.dims, copy.dims + copy.rank);
                const std::vector<int64_t> src(copy.srcStrides, copy.srcStrides + copy.rank);
                const std::vector<int64_t> dst(copy.dstStrides, copy.dstStrides + copy.rank);
                step.args = epilogueCast(problem, dims, src, dst);
                step.dataType = problem.dataType;
                step.srcUid = *twin;
                step.srcByteOffset
                    = step.srcByteOffset / bytes * static_cast<int64_t>(sizeof(float));
            }
        }
    }
    plan.epilogueForm = EpilogueForm::FUSED_PASS;
    plan.epilogue = epilogue;
    return plan;
}

/// Everything that writes y writes an fp32 copy of it instead; one cast applies the epilogue.
std::optional<ConvGemmPlanCalls>
    fuseInExtraPass(const ConvProblem& problem,
                    ConvGemmPlanCalls plan,
                    const std::shared_ptr<const conv_epilogue::ConvEpilogue>& epilogue)
{
    const int64_t bytes = elementBytes(problem.dataType);
    int64_t span = 1;
    for(size_t axis = 0; axis < problem.y.dims.size(); ++axis)
    {
        int64_t reach = 0;
        if(__builtin_mul_overflow(problem.y.dims[axis] - 1, problem.y.strides[axis], &reach)
           || __builtin_add_overflow(span, reach, &span))
        {
            return std::nullopt;
        }
    }
    const int64_t uid = freshUid(problem, plan, 0);
    const size_t offset = alignUp(plan.workspaceBytes);
    const size_t copyBytes = static_cast<size_t>(span) * sizeof(float);
    for(auto& stage : plan.stages)
    {
        for(auto& call : stage.calls)
        {
            if(call.a.uid == problem.y.uid || call.b.uid == problem.y.uid)
            {
                return std::nullopt;
            }
            if(call.d.uid == problem.y.uid)
            {
                call.d.uid = uid;
                call.d.dataType = DataType::FLOAT;
            }
        }
        for(auto* steps : {&stage.before, &stage.after})
        {
            for(auto& step : *steps)
            {
                if(step.srcUid == problem.y.uid)
                {
                    return std::nullopt;
                }
                if(step.dstUid != problem.y.uid)
                {
                    continue;
                }
                // Only a sum of fp32 partials writes y forward; it now writes the fp32 copy.
                if(!std::holds_alternative<StridedCastArgs>(step.args)
                   || step.dstByteOffset % bytes != 0)
                {
                    return std::nullopt;
                }
                step.dstUid = uid;
                step.dataType = DataType::FLOAT;
                step.dstByteOffset
                    = step.dstByteOffset / bytes * static_cast<int64_t>(sizeof(float));
            }
        }
    }
    plan.workspace.push_back(WorkspaceTensor{uid, offset, copyBytes});
    plan.workspaceBytes = offset + copyBytes;
    ConvGemmStage last;
    last.after.push_back(
        RepackStep{epilogueCast(problem, problem.y.dims, problem.y.strides, problem.y.strides),
                   problem.dataType,
                   uid,
                   problem.y.uid,
                   0,
                   0});
    plan.stages.push_back(std::move(last));
    plan.epilogueForm = EpilogueForm::EXTRA_PASS;
    plan.epilogue = epilogue;
    return plan;
}

} // namespace

std::optional<NativeEpilogue> nativeEpilogueFor(const ConvProblem& problem)
{
    const auto& epilogue = problem.epilogue;
    const auto& ops = epilogue.ops;
    if(ops.empty() || ops.size() > 2)
    {
        return std::nullopt;
    }
    NativeEpilogue native;
    size_t next = 0;
    if(ops[0].code == OpCode::ADD_OPERAND)
    {
        const auto& bias = epilogue.operands[static_cast<size_t>(ops[0].slot)];
        if(!bias.perChannel
           || (bias.dataType != DataType::FLOAT && bias.dataType != problem.dataType))
        {
            return std::nullopt;
        }
        native.biasUid = bias.uid;
        native.biasType = bias.dataType;
        native.mode = NATIVE_EPILOGUE_BIAS;
        next = 1;
    }
    if(next == ops.size())
    {
        return native;
    }
    if(next + 1 != ops.size())
    {
        return std::nullopt;
    }
    const bool withBias = native.biasUid.has_value();
    const auto& op = ops[next];
    const auto value = [&epilogue](int slot) -> std::optional<float> {
        const auto& scalar = epilogue.scalars[static_cast<size_t>(slot)];
        return scalar.runtime.has_value() ? std::nullopt : std::make_optional(scalar.value);
    };
    switch(op.code)
    {
    case OpCode::RELU:
    {
        const auto lower = value(op.slot);
        const auto upper = value(op.slot + 1);
        const auto slope = value(op.slot + 2);
        if(!lower || !upper || !slope || *slope != 0.0F)
        {
            return std::nullopt;
        }
        if(std::isinf(*upper) && *upper > 0.0F)
        {
            if(*lower != 0.0F)
            {
                return std::nullopt;
            }
            native.mode = withBias ? NATIVE_EPILOGUE_RELU_BIAS : NATIVE_EPILOGUE_RELU;
            return native;
        }
        if(!std::isfinite(*lower) || !std::isfinite(*upper) || *lower > *upper)
        {
            return std::nullopt;
        }
        native.mode = withBias ? NATIVE_EPILOGUE_CLAMP_BIAS : NATIVE_EPILOGUE_CLAMP;
        native.act0 = *lower;
        native.act1 = *upper;
        return native;
    }
    case OpCode::GELU_TANH:
        native.mode = withBias ? NATIVE_EPILOGUE_GELU_BIAS : NATIVE_EPILOGUE_GELU;
        return native;
    case OpCode::SWISH:
    {
        const auto beta = value(op.slot);
        if(!beta || *beta != 1.0F)
        {
            return std::nullopt;
        }
        native.mode = withBias ? NATIVE_EPILOGUE_SWISH_BIAS : NATIVE_EPILOGUE_SWISH;
        return native;
    }
    case OpCode::SIGMOID:
        if(withBias)
        {
            return std::nullopt;
        }
        native.mode = NATIVE_EPILOGUE_SIGMOID;
        return native;
    default:
        break;
    }
    return std::nullopt;
}

std::optional<ConvGemmPlanCalls>
    fuseEpilogue(const ConvProblem& problem, ConvGemmPlanCalls plan, bool native)
{
    if(problem.epilogue.empty())
    {
        return plan;
    }
    if(problem.direction != ConvDirection::FORWARD)
    {
        return std::nullopt;
    }
    if(native)
    {
        if(const auto mapped = nativeEpilogueFor(problem))
        {
            auto fused = fuseNative(problem, plan, *mapped);
            if(fused.has_value())
            {
                return fused;
            }
        }
    }
    const auto epilogue = std::make_shared<const conv_epilogue::ConvEpilogue>(problem.epilogue);
    auto staged = fuseIntoStagedCopy(problem, plan, epilogue);
    if(staged.has_value())
    {
        return staged;
    }
    return fuseInExtraPass(problem, std::move(plan), epilogue);
}

} // namespace hipblaslt_plugin
