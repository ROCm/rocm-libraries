// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "common/ConvEpilogueFormulation.hpp"

#include "common/GemmFormulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    args.cast.batches = 1;
    int64_t total = 1;
    int64_t span = 1;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        total *= dims[axis];
        span += (dims[axis] - 1) * dstStrides[axis];
    }
    args.cast.total = total;
    if(srcStrides == dstStrides && span == total)
    {
        // Both sides the same dense layout: one flat walk over memory, coalesced on both, with
        // the channel of element o, (o / channel stride) % K, carried along without a division.
        args.cast.rank = 1;
        args.cast.dims[0] = total;
        args.cast.srcStrides[0] = 1;
        args.cast.dstStrides[0] = 1;
        args.epilogue = repackEpilogue(problem.epilogue, -1);
        args.epilogue.flatChannelStride = std::max<int64_t>(dstStrides[1], 1);
        args.epilogue.flatChannels = dims[1];
        return args;
    }
    // Otherwise y's axes, innermost the one y is densest along, so stores coalesce.
    std::vector<size_t> order(dims.size());
    for(size_t axis = 0; axis < order.size(); ++axis)
    {
        order[axis] = axis;
    }
    std::stable_sort(order.begin(), order.end(), [&dstStrides](size_t lhs, size_t rhs) {
        return dstStrides[lhs] > dstStrides[rhs];
    });
    int channelAxis = 0;
    args.cast.rank = static_cast<int>(dims.size());
    for(size_t i = 0; i < order.size(); ++i)
    {
        args.cast.dims[i] = dims[order[i]];
        args.cast.srcStrides[i] = srcStrides[order[i]];
        args.cast.dstStrides[i] = dstStrides[order[i]];
        if(order[i] == 1)
        {
            channelAxis = static_cast<int>(i);
        }
    }
    args.epilogue = repackEpilogue(problem.epilogue, channelAxis);
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

/// The call is whole after the libraries' limits: an epilogue applies to it once.
bool whole(const GemmCall& call)
{
    return !call.accumulate && fitLibraryLimits({call}).size() == 1;
}

/// The rows of d are y's channels: one group, channels unit-stride, m all of them.
bool channelRows(const ConvProblem& problem, const GemmCall& call)
{
    return problem.groups == 1 && call.batchCount == 1
           && (problem.k == 1 || problem.y.strides[1] == 1) && call.m == problem.k;
}

/// The columns of d are y's channels: one group, column j at channel j.
bool channelColumns(const ConvProblem& problem, const GemmCall& call)
{
    return problem.groups == 1 && call.n == problem.k
           && (problem.k == 1 || call.d.leadingDimension == problem.y.strides[1]);
}

/// Whether @p value is exact in @p dataType (the infinities are).
bool exactIn(float value, DataType dataType)
{
    if(std::isinf(value) || dataType == DataType::FLOAT)
    {
        return !std::isnan(value);
    }
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const int exponent = static_cast<int>((bits >> 23) & 0xFFU) - 127;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if(value == 0.0F)
    {
        return true;
    }
    if(dataType == DataType::BFLOAT16)
    {
        // fp32's range, 7 fraction bits.
        return (mantissa & 0xFFFFU) == 0;
    }
    // fp16: 10 fraction bits, normal down to 2^-14, subnormal steps of 2^-24, up to 65504.
    if(std::fabs(value) > 65504.0F || exponent < -24)
    {
        return false;
    }
    const int dropped = exponent >= -14 ? 13 : 13 + (-14 - exponent);
    return (mantissa & ((1U << dropped) - 1U)) == 0;
}

/// Elements of y and whether they fill its span without gaps (any order of axes).
std::optional<int64_t> denseElements(const ConvOperand& y)
{
    int64_t total = 1;
    int64_t span = 1;
    for(size_t axis = 0; axis < y.dims.size(); ++axis)
    {
        total *= y.dims[axis];
        span += (y.dims[axis] - 1) * y.strides[axis];
    }
    if(span != total)
    {
        return std::nullopt;
    }
    return total;
}

/// The GEMMs that write y apply @p gemm themselves; a relu / clamp tail clamps y after them.
std::optional<ConvGemmPlanCalls> fuseIntoGemm(const ConvProblem& problem,
                                              ConvGemmPlanCalls plan,
                                              const GemmEpilogue& gemm,
                                              const GemmEpilogueSupport& support)
{
    const DataType yType = problem.dataType;
    // A tail alone changes no GEMM: it clamps y whoever wrote it, once everything has.
    const bool inGemm = gemm.bias.has_value() || gemm.activation != 0;
    // Where each call writing y takes the bias (true: as an addend), checked before any change.
    std::vector<std::vector<bool>> addend(plan.stages.size());
    int64_t tileRows = 0;
    bool any = false;
    for(size_t s = 0; s < plan.stages.size(); ++s)
    {
        const auto& stage = plan.stages[s];
        for(const auto* steps : {&stage.before, &stage.after})
        {
            for(const auto& step : *steps)
            {
                if(step.srcUid == problem.y.uid || (inGemm && step.dstUid == problem.y.uid))
                {
                    return std::nullopt;
                }
            }
        }
        for(const auto& call : stage.calls)
        {
            if(call.a.uid == problem.y.uid || call.b.uid == problem.y.uid)
            {
                return std::nullopt;
            }
            bool asAddend = false;
            if(call.d.uid == problem.y.uid && inGemm)
            {
                if(!whole(call) || (gemm.activation != 0 && !support.libraryEpilogue))
                {
                    return std::nullopt;
                }
                if(gemm.bias.has_value())
                {
                    // hipBLASLt has no sigmoid with a bias: that bias is an addend.
                    const bool rows = support.libraryEpilogue && channelRows(problem, call)
                                      && gemm.activation != NATIVE_EPILOGUE_SIGMOID;
                    asAddend = !rows;
                    // A vector-shaped call may run as the library's GEMV, which reads c as d: no
                    // addend there.
                    if(asAddend
                       && (!channelColumns(problem, call) || gemm.bias->dataType != yType
                           || call.m == 1 || call.n == 1))
                    {
                        return std::nullopt;
                    }
                }
                if(asAddend)
                {
                    tileRows = std::max(tileRows, call.m);
                }
            }
            any = any || call.d.uid == problem.y.uid;
            addend[s].push_back(asAddend);
        }
    }
    if(!any)
    {
        return std::nullopt;
    }
    std::optional<int64_t> yElements;
    if(gemm.clamp.has_value())
    {
        yElements = denseElements(problem.y);
        if(!yElements.has_value())
        {
            return std::nullopt;
        }
    }

    // The broadcast bias: tileRows x K, column-major, every row the bias; one fill before the
    // first stage, read by every call through batch stride 0.
    std::optional<GemmOperand> tile;
    if(tileRows > 0)
    {
        const int64_t uid = freshUid(problem, plan, 0);
        const size_t offset = alignUp(plan.workspaceBytes);
        const auto bytes = static_cast<size_t>(tileRows * problem.k * elementBytes(yType));
        plan.workspace.push_back(WorkspaceTensor{uid, offset, bytes});
        plan.workspaceBytes = offset + bytes;
        BroadcastArgs fill{};
        fill.rows = tileRows;
        fill.channels = problem.k;
        fill.srcStride = 1;
        auto& first = plan.stages.front().before;
        first.insert(first.begin(), RepackStep{fill, yType, gemm.bias->uid, uid, 0, 0});
        tile = GemmOperand{uid, yType, tileRows, 0, 0};
    }

    for(size_t s = 0; s < plan.stages.size(); ++s)
    {
        for(size_t i = 0; i < plan.stages[s].calls.size(); ++i)
        {
            auto& call = plan.stages[s].calls[i];
            if(call.d.uid != problem.y.uid || !inGemm)
            {
                continue;
            }
            const bool biasHere = gemm.bias.has_value() && !addend[s][i];
            if(addend[s][i])
            {
                call.c = tile;
            }
            if(gemm.activation == 0 && !biasHere)
            {
                continue;
            }
            NativeEpilogue native;
            native.mode = gemm.activation + (biasHere ? NATIVE_EPILOGUE_BIAS : 0);
            native.act0 = gemm.act0;
            native.act1 = gemm.act1;
            if(biasHere)
            {
                native.biasUid = gemm.bias->uid;
                native.biasType = gemm.bias->dataType;
            }
            call.epilogue = native;
        }
    }
    if(gemm.clamp.has_value())
    {
        ClampArgs clamp{};
        clamp.total = *yElements;
        clamp.lower = gemm.clamp->first;
        clamp.upper = gemm.clamp->second;
        plan.stages.back().after.push_back(
            RepackStep{clamp, yType, problem.y.uid, problem.y.uid, 0, 0});
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

ConvGemmPlanCalls splitFp32OutputReductions(ConvGemmPlanCalls plan, int64_t chunks)
{
    for(auto& stage : plan.stages)
    {
        std::vector<GemmCall> calls;
        for(const auto& call : stage.calls)
        {
            if(call.d.dataType != DataType::FLOAT || call.a.dataType == DataType::FLOAT
               || call.epilogue.has_value() || chunks < 2 || call.k < chunks)
            {
                calls.push_back(call);
                continue;
            }
            const int64_t length = (call.k + chunks - 1) / chunks;
            for(int64_t first = 0; first < call.k; first += length)
            {
                GemmCall part = call;
                part.k = std::min(length, call.k - first);
                part.a.offset += call.transposeA ? first : first * call.a.leadingDimension;
                part.b.offset += call.transposeB ? first * call.b.leadingDimension : first;
                part.accumulate = call.accumulate || first > 0;
                calls.push_back(part);
            }
        }
        stage.calls = std::move(calls);
    }
    return plan;
}

std::optional<GemmEpilogue> gemmEpilogueFor(const ConvProblem& problem,
                                            const GemmEpilogueSupport& support)
{
    const auto& epilogue = problem.epilogue;
    const auto& ops = epilogue.ops;
    if(ops.empty() || ops.size() > 2)
    {
        return std::nullopt;
    }
    GemmEpilogue gemm;
    size_t next = 0;
    if(ops[0].code == OpCode::ADD_OPERAND)
    {
        const auto& bias = epilogue.operands[static_cast<size_t>(ops[0].slot)];
        if(!bias.perChannel
           || (bias.dataType != DataType::FLOAT && bias.dataType != problem.dataType))
        {
            return std::nullopt;
        }
        gemm.bias = bias;
        next = 1;
    }
    if(next == ops.size())
    {
        return gemm;
    }
    if(next + 1 != ops.size())
    {
        return std::nullopt;
    }
    const auto& op = ops[next];
    const auto value = [&](int offset) -> std::optional<float> {
        const auto& slot
            = epilogue.scalars[static_cast<size_t>(op.slot) + static_cast<size_t>(offset)];
        if(slot.runtime.has_value())
        {
            return std::nullopt;
        }
        return slot.value;
    };
    const bool fp32Output = problem.dataType == DataType::FLOAT;
    switch(op.code)
    {
    case OpCode::RELU:
    {
        const auto lower = value(0);
        const auto upper = value(1);
        const auto slope = value(2);
        if(!lower.has_value() || !upper.has_value() || !slope.has_value() || *slope != 0.0F)
        {
            return std::nullopt;
        }
        if(support.libraryEpilogue && support.libraryActivations)
        {
            if(*lower == 0.0F && std::isinf(*upper) && *upper > 0.0F)
            {
                gemm.activation = NATIVE_EPILOGUE_RELU;
            }
            else
            {
                gemm.activation = NATIVE_EPILOGUE_CLAMP;
                gemm.act0 = *lower;
                gemm.act1 = *upper;
            }
            return gemm;
        }
        // Clamping the rounded result is clamping before the rounding when the bounds are
        // exact in y's type: the rounding is monotonic and keeps them.
        if(!exactIn(*lower, problem.dataType) || !exactIn(*upper, problem.dataType))
        {
            return std::nullopt;
        }
        gemm.clamp = std::make_pair(*lower, *upper);
        return gemm;
    }
    case OpCode::SIGMOID:
        // The 16-bit form of the chain's sigmoid; an fp32 y takes a few-ulp one.
        if(!support.libraryEpilogue || (fp32Output && !support.libraryActivations))
        {
            return std::nullopt;
        }
        gemm.activation = NATIVE_EPILOGUE_SIGMOID;
        return gemm;
    case OpCode::GELU_TANH:
        if(!support.libraryEpilogue || !support.libraryActivations)
        {
            return std::nullopt;
        }
        gemm.activation = NATIVE_EPILOGUE_GELU;
        return gemm;
    case OpCode::SWISH:
    {
        const auto beta = value(0);
        if(!support.libraryEpilogue || !support.libraryActivations || !beta.has_value()
           || *beta != 1.0F)
        {
            return std::nullopt;
        }
        gemm.activation = NATIVE_EPILOGUE_SWISH;
        return gemm;
    }
    default:
        break;
    }
    // Everything else (exact GELU, which hipBLASLt does not have, tanh, ELU, ...) runs in a pass.
    return std::nullopt;
}

bool usesLibraryActivation(const ConvProblem& problem)
{
    const auto library = gemmEpilogueFor(problem, GemmEpilogueSupport{true, true});
    if(!library.has_value() || library->activation == 0)
    {
        return false;
    }
    const auto exact = gemmEpilogueFor(problem, GemmEpilogueSupport{true, false});
    return !exact.has_value() || exact->activation != library->activation;
}

std::optional<ConvGemmPlanCalls> fuseEpilogue(const ConvProblem& problem,
                                              ConvGemmPlanCalls plan,
                                              const std::optional<GemmEpilogueSupport>& gemm)
{
    if(problem.epilogue.empty())
    {
        return plan;
    }
    if(problem.direction != ConvDirection::FORWARD)
    {
        return std::nullopt;
    }
    if(gemm.has_value())
    {
        if(const auto mapped = gemmEpilogueFor(problem, *gemm))
        {
            auto fused = fuseIntoGemm(problem, plan, *mapped, *gemm);
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
