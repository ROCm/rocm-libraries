// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "solution_index_case.hpp"

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_scaling_format.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace
{
    std::optional<hipDataType> hipType(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
            return HIP_R_32F;
        case rocisa::DataType::Double:
            return HIP_R_64F;
        case rocisa::DataType::Half:
            return HIP_R_16F;
        case rocisa::DataType::BFloat16:
            return HIP_R_16BF;
        case rocisa::DataType::Float8_fnuz:
            return HIP_R_8F_E4M3_FNUZ;
        case rocisa::DataType::BFloat8_fnuz:
            return HIP_R_8F_E5M2_FNUZ;
        case rocisa::DataType::Float8:
            return HIP_R_8F_E4M3;
        case rocisa::DataType::BFloat8:
            return HIP_R_8F_E5M2;
        case rocisa::DataType::Int8:
            return HIP_R_8I;
        case rocisa::DataType::Int32:
            return HIP_R_32I;
        case rocisa::DataType::ComplexFloat:
            return HIP_C_32F;
        case rocisa::DataType::ComplexDouble:
            return HIP_C_64F;
        case rocisa::DataType::Float4:
            return static_cast<hipDataType>(HIP_R_4F_E2M1);
        case rocisa::DataType::Float6:
            return static_cast<hipDataType>(HIP_R_6F_E2M3);
        case rocisa::DataType::BFloat6:
            return static_cast<hipDataType>(HIP_R_6F_E3M2);
        default:
            return std::nullopt;
        }
    }

    std::optional<hipblasComputeType_t> computeType(rocisa::DataType type)
    {
        switch(type)
        {
        case rocisa::DataType::Float:
        case rocisa::DataType::ComplexFloat:
            return HIPBLAS_COMPUTE_32F;
        case rocisa::DataType::Double:
        case rocisa::DataType::ComplexDouble:
            return HIPBLAS_COMPUTE_64F;
        case rocisa::DataType::Int32:
            return HIPBLAS_COMPUTE_32I;
        default:
            return std::nullopt;
        }
    }

    // A trailing C on the A/B index names marks a conjugated operand.
    std::string operationIdentifier(TensileLite::ContractionSolution::ProblemType const& p,
                                    bool                                                 conjugateA,
                                    bool                                                 conjugateB)
    {
        return std::string("Contraction_l_A") + (p.transA ? "lik" : "ilk") + (conjugateA ? "C" : "")
               + "_B" + (p.transB ? "jlk" : "ljk") + (conjugateB ? "C" : "") + "_Cijk_Dijk";
    }

    char operation(bool transposed, bool conjugated)
    {
        return conjugated ? 'C' : transposed ? 'T' : 'N';
    }

    // The activations a solution's predicate allows, when it restricts them.
    std::optional<std::vector<TensileLite::ActivationType>>
        allowedActivations(TensileLite::ContractionSolution const& solution)
    {
        using Problem   = TensileLite::ContractionProblemGemm;
        using WhiteList = TensileLite::Predicates::Contraction::ActivationEnumWhiteList;
        std::vector<std::shared_ptr<TensileLite::Predicates::Predicate<Problem>>> pending{
            solution.problemPredicate};
        while(!pending.empty())
        {
            auto predicate = pending.back();
            pending.pop_back();
            if(auto list = std::dynamic_pointer_cast<WhiteList>(predicate))
                return list->value;
            if(auto all
               = std::dynamic_pointer_cast<TensileLite::Predicates::And<Problem>>(predicate))
                pending.insert(pending.end(), all->value.begin(), all->value.end());
        }
        return std::nullopt;
    }

    std::optional<hipblaslt_bias_source> biasSource(int code)
    {
        // Serialized bias-source codes; 2 (C) does not occur in shipped solutions.
        switch(code)
        {
        case 0:
            return hipblaslt_bias_source::a;
        case 1:
            return hipblaslt_bias_source::b;
        case 3:
            return hipblaslt_bias_source::d;
        default:
            return std::nullopt;
        }
    }

    // The client's scaling mode for an MX operand, or nullopt when it has none.
    std::optional<hipblaslt_scaling_format> mxScaling(
        int block, rocisa::DataType scaleType, int format, bool transposed, bool isA, bool gfx1250)
    {
        using S = hipblaslt_scaling_format;
        if(format == 1)
        {
            // The gfx950 host-preswizzled layout needs A transposed and B normal.
            if(block == 32 && scaleType == rocisa::DataType::E8 && transposed == isA)
                return S::Block_32_UE8M0_32_8_EXT;
            return std::nullopt;
        }
        // The client prepares the gfx1250 tensor-data-movement layout itself.
        if((format == 2) != gfx1250 || (format != 0 && format != 2))
            return std::nullopt;
        bool const b32 = block == 32;
        if(block != 32 && block != 16)
            return std::nullopt;
        switch(scaleType)
        {
        case rocisa::DataType::E8:
            return b32 ? S::Block_32_UE8M0 : S::Block_16_UE8M0;
        case rocisa::DataType::Float8:
            return b32 ? S::Block_32_UE4M3 : S::Block_16_UE4M3;
        case rocisa::DataType::E5M3:
            return b32 ? S::Block_32_UE5M3 : S::Block_16_UE5M3;
        default:
            return std::nullopt;
        }
    }

    bool swizzleSupported(rocisa::DataType type)
    {
        // The client ignores swizzle flags for other storage types.
        return type == rocisa::DataType::Half || type == rocisa::DataType::BFloat16
               || type == rocisa::DataType::Float8_fnuz || type == rocisa::DataType::Float4;
    }

    using ProblemPredicate
        = TensileLite::Predicates::Predicate<TensileLite::ContractionProblemGemm>;

    bool isSizePredicate(ProblemPredicate const& predicate)
    {
        auto const type = predicate.type();
        return type.find("Size") != std::string::npos || type == "GlobalSplitUCheckMinK";
    }

    // The solution's dimension checks, or nullopt when one sits under Or or Not,
    // where a single bound cannot be chosen.
    std::optional<std::vector<std::shared_ptr<ProblemPredicate>>>
        sizePredicates(std::shared_ptr<ProblemPredicate> const& predicate)
    {
        using Problem = TensileLite::ContractionProblemGemm;
        std::vector<std::shared_ptr<ProblemPredicate>> result;
        if(auto all = std::dynamic_pointer_cast<TensileLite::Predicates::And<Problem>>(predicate))
        {
            for(auto const& child : all->value)
            {
                auto sizes = sizePredicates(child);
                if(!sizes)
                    return std::nullopt;
                result.insert(result.end(), sizes->begin(), sizes->end());
            }
            return result;
        }
        if(auto any = std::dynamic_pointer_cast<TensileLite::Predicates::Or<Problem>>(predicate))
        {
            for(auto const& child : any->value)
                if(!sizePredicates(child) || !sizePredicates(child)->empty())
                    return std::nullopt;
            return result;
        }
        if(auto none = std::dynamic_pointer_cast<TensileLite::Predicates::Not<Problem>>(predicate))
        {
            auto inner = sizePredicates(none->value);
            if(!inner || !inner->empty())
                return std::nullopt;
            return result;
        }
        if(predicate && isSizePredicate(*predicate))
            result.push_back(predicate);
        return result;
    }

    struct Sizes
    {
        int64_t m, n, k;
    };

    // The size closest to the preferred one that the solution's own size
    // predicates accept. Bounds and multiples named by the predicates suggest
    // candidates; the predicates themselves decide.
    std::optional<Sizes> chooseSizes(TensileLite::ContractionSolution const&               solution,
                                     std::vector<std::shared_ptr<ProblemPredicate>> const& checks,
                                     Sizes   preferred,
                                     int64_t kMultiple)
    {
        namespace C   = TensileLite::Predicates::Contraction;
        auto const& p = solution.problemType;
        // Problem size order: free0 (M), free1 (N), batch, bound (K).
        std::array<int64_t, 4>              want{preferred.m, preferred.n, 1, preferred.k};
        std::array<int64_t, 4>              multiple{1, 1, 1, kMultiple};
        std::array<std::vector<int64_t>, 4> hints;
        for(auto const& check : checks)
        {
            if(auto c = std::dynamic_pointer_cast<C::SizeEqual>(check))
                hints[c->index].push_back(c->value);
            else if(auto c = std::dynamic_pointer_cast<C::SizeGreaterThan>(check))
                hints[c->index].push_back(c->value + 1);
            else if(auto c = std::dynamic_pointer_cast<C::SizeLessThan>(check))
                hints[c->index].push_back(c->value - 1);
            else if(auto c = std::dynamic_pointer_cast<C::SizeInRange>(check))
            {
                hints[c->index].push_back(static_cast<int64_t>(c->value.min));
                if(c->value.max != std::numeric_limits<size_t>::max())
                    hints[c->index].push_back(static_cast<int64_t>(c->value.max) - 1);
            }
            else if(auto c = std::dynamic_pointer_cast<C::SizeMultiple>(check))
                multiple[c->index] = std::lcm(multiple[c->index], int64_t(c->value));
            else if(auto c = std::dynamic_pointer_cast<C::Free0SizeMultiple>(check))
                multiple[0] = std::lcm(multiple[0], int64_t(c->value));
            else if(auto c = std::dynamic_pointer_cast<C::Free1SizeMultiple>(check))
                multiple[1] = std::lcm(multiple[1], int64_t(c->value));
            else if(auto c = std::dynamic_pointer_cast<C::BoundSizeMultiple>(check))
                multiple[3] = std::lcm(multiple[3], int64_t(c->value));
            // With the problem's split unset, the minimum K is the solution's
            // split times a per-split minimum; one split or an automatic split
            // (-1) imposes none.
            else if(auto c = std::dynamic_pointer_cast<C::GlobalSplitUCheckMinK>(check))
            {
                if(c->value[1] > 1)
                    hints[3].push_back(int64_t(c->value[0]) * c->value[1]);
            }
        }
        std::array<std::vector<int64_t>, 4> candidates;
        for(int d = 0; d < 4; ++d)
        {
            std::vector<int64_t> values{want[d]};
            values.insert(values.end(), hints[d].begin(), hints[d].end());
            for(int64_t value : values)
            {
                int64_t const below = value / multiple[d] * multiple[d];
                for(int64_t v : {below, below + multiple[d]})
                    if(v > 0
                       && std::find(candidates[d].begin(), candidates[d].end(), v)
                              == candidates[d].end())
                        candidates[d].push_back(v);
            }
            // Closest to the preferred size first; the larger one on a tie.
            std::stable_sort(candidates[d].begin(), candidates[d].end(), [&](int64_t a, int64_t b) {
                auto da = std::abs(a - want[d]), db = std::abs(b - want[d]);
                return da != db ? da < db : a > b;
            });
        }
        std::optional<Sizes>    best;
        std::pair<int, int64_t> bestCost{5, 0};
        for(int64_t m : candidates[0])
            for(int64_t n : candidates[1])
                for(int64_t k : candidates[3])
                {
                    auto problem
                        = TensileLite::ContractionProblemGemm::GEMM_Strides(p.transA,
                                                                            p.transB,
                                                                            p.aType,
                                                                            p.bType,
                                                                            p.cType,
                                                                            p.dType,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            1,
                                                                            p.transA ? k : m,
                                                                            m * k,
                                                                            p.transB ? n : k,
                                                                            n * k,
                                                                            m,
                                                                            m * n,
                                                                            m,
                                                                            m * n,
                                                                            0);
                    bool accepted
                        = std::all_of(checks.begin(), checks.end(), [&](auto const& check) {
                              return (*check)(problem);
                          });
                    if(!accepted)
                        continue;
                    std::array<int64_t, 3> chosen{m, n, k}, preferredMNK{want[0], want[1], want[3]};
                    std::pair<int, int64_t> cost{0, 0};
                    for(int d = 0; d < 3; ++d)
                    {
                        cost.first += chosen[d] != preferredMNK[d];
                        cost.second += std::abs(chosen[d] - preferredMNK[d]);
                    }
                    if(cost < bestCost)
                        best = Sizes{m, n, k}, bestCost = cost;
                }
        return best;
    }

}

bool requiresCUCount(TensileLite::ContractionSolution const& solution,
                     TensileLite::Hardware const&            hardware)
{
    auto const* gpu = dynamic_cast<TensileLite::AMDGPU const*>(&hardware);
    if(!gpu)
        return false;
    TensileLite::AMDGPU other = *gpu;
    other.computeUnitCount += 1;
    auto const& predicate = *solution.hardwarePredicate;
    return predicate(hardware) && !predicate(other);
}

std::variant<Arguments, UnsupportedSolution> solutionIndexCase(
    int index, TensileLite::ContractionSolution const& solution, SolutionCaseOptions const& options)
{
    using Result        = std::variant<Arguments, UnsupportedSolution>;
    auto unsupported    = [](std::string reason) { return Result{UnsupportedSolution{reason}}; };
    auto const& p       = solution.problemType;
    auto const& mapping = solution.sizeMapping;

    if(p.sparse || p.useGateResidual || p.fusedGemmA2A)
        return unsupported("sparse, gate residual or fused all-to-all");
    if(p.useInitialStridesAB || p.useInitialStridesCD || !p.stridedBatched)
        return unsupported("unusual strides or batching");
    if(mapping.adaptiveGemmGSUA)
        return unsupported("adaptive split-K");
    if(p.activationType != TensileLite::ActivationType::None
       && p.activationType != TensileLite::ActivationType::All
       && p.activationType != TensileLite::ActivationType::Hipblaslt_all)
        return unsupported("fixed activation");
    if(p.cType != p.dType)
        return unsupported("different C and D types");
    if(mapping.globalSplitU < -1 || (mapping.globalSplitU == 0 && !mapping.isPersistent()))
        return unsupported("invalid globalSplitU");
    bool conjA = false, conjB = false, matched = false;
    for(bool a : {false, true})
        for(bool b : {false, true})
            if(p.operationIdentifier == operationIdentifier(p, a, b))
                conjA = a, conjB = b, matched = true;
    if(!matched)
        return unsupported("contraction " + p.operationIdentifier);
    if((conjA && !p.transA) || (conjB && !p.transB))
        return unsupported("conjugation without transpose");

    auto a   = hipType(p.aType);
    auto b   = hipType(p.bType);
    auto c   = hipType(p.cType);
    auto cmp = computeType(p.computeType);
    if(!(a && b && c && cmp))
        return unsupported("datatype the client cannot express");

    // Sizes: one macro tile, and K covering reductionBlocks complete DepthU
    // blocks per split. Persistent schedules distribute the blocks
    // themselves; an automatic split (-1) is given two splits' worth.
    int64_t const fixedSplit
        = mapping.isPersistent() || mapping.globalSplitU == -1 ? 1 : mapping.globalSplitU;
    int64_t const kSplits  = mapping.isPersistent()       ? 1
                             : mapping.globalSplitU == -1 ? 2
                                                          : mapping.globalSplitU;
    int64_t       multiple = static_cast<int64_t>(mapping.depthU) * fixedSplit;
    for(int block : {p.mxBlockA, p.mxBlockB})
        if(block)
            multiple = std::lcm(multiple, int64_t(block));
    int64_t const preferredK
        = options.reductionBlocks * static_cast<int64_t>(mapping.depthU) * kSplits;
    auto checks = sizePredicates(solution.problemPredicate);
    if(!checks)
        return unsupported("size predicates under Or or Not");
    auto sizes = chooseSizes(solution,
                             *checks,
                             {static_cast<int64_t>(mapping.macroTile.x),
                              static_cast<int64_t>(mapping.macroTile.y),
                              (preferredK + multiple - 1) / multiple * multiple},
                             multiple);
    if(!sizes)
        return unsupported("no size satisfies the solution's size predicates");
    int64_t const m = sizes->m, n = sizes->n, k = sizes->k;

    // Value-initialized: init() leaves fields such as batch_mode, the batch
    // offsets and bias_stride unset.
    Arguments arg{};
    arg.init();
    std::strcpy(arg.function, "matmul");
    std::snprintf(arg.name, sizeof(arg.name), "idx%d", index);
    std::strcpy(arg.category, "solution_index");
    arg.a_type = *a;
    arg.b_type = *b;
    arg.c_type = *c;
    arg.d_type = *c;
    // Real float and integer computation use their own scalar types;
    // double and complex scalars match C.
    arg.scale_type = p.computeType == rocisa::DataType::Float   ? HIP_R_32F
                     : p.computeType == rocisa::DataType::Int32 ? HIP_R_32I
                                                                : *c;
    arg.compute_type
        = p.f32XdlMathOp == rocisa::DataType::XFloat32 ? HIPBLAS_COMPUTE_32F_FAST_TF32 : *cmp;
    arg.transA      = operation(p.transA, conjA);
    arg.transB      = operation(p.transB, conjB);
    arg.batch_count = 1;
    arg.alpha       = options.alpha;
    arg.beta        = options.beta;
    // Bounded inputs keep MX products within FP16 outputs at large DepthU.
    arg.initialization         = (p.mxBlockA || p.mxBlockB) ? hipblaslt_initialization::hpl
                                                            : hipblaslt_initialization::rand_int;
    arg.algo_method            = 2;
    arg.solution_index         = index;
    arg.requested_solution_num = 1;
    arg.norm_check             = 1;
    arg.norm_check_assert      = true;
    // The norm check is the verdict; an exact comparison flags reference
    // rounding in scaled mixed FP8 and MX outputs.
    arg.unit_check     = 0;
    arg.allclose_check = 1;
    // The requested solution must run even if it does not guarantee a
    // uniform summation order.
    arg.uniform_summation_order  = 0;
    arg.user_allocated_workspace = 128 * 1024 * 1024;

    // Explicit compute inputs: the client accepts narrow and FP32 types only;
    // with both unset, Int8 storage keeps Float computation.
    if(p.computeType == rocisa::DataType::Float
       && !(p.computeInputTypeA == p.aType && p.computeInputTypeB == p.bType
            && p.aType == rocisa::DataType::Int8 && p.bType == rocisa::DataType::Int8))
    {
        auto inputA = hipType(p.computeInputTypeA);
        auto inputB = hipType(p.computeInputTypeB);
        if(!inputA || !inputB || *inputA == HIP_R_8I || *inputB == HIP_R_8I)
            return unsupported("compute input type the client cannot express");
        arg.compute_input_typeA = *inputA;
        arg.compute_input_typeB = *inputB;
    }

    // Outputs the solution always writes: a gradient needs its bias; E is
    // used with ReLU (forward) or its derivative (gradient), with a bias too.
    bool const gradient = p.useGradient, aux = p.useE;
    if(gradient || aux)
    {
        if(!p.useBias)
            return unsupported("gradient or E epilogue without bias support");
        std::optional<hipblaslt_bias_source> source;
        for(int code : p.biasSrcWhiteList)
            if((source = biasSource(code)))
                break;
        auto const& types = p.biasDataTypeWhiteList;
        if(!source || types.empty())
            return unsupported("no usable bias source or type");
        bool const floatAllowed
            = std::find(types.begin(), types.end(), rocisa::DataType::Float) != types.end();
        auto biasType = hipType(floatAllowed ? rocisa::DataType::Float : types.front());
        if(!biasType)
            return unsupported("bias datatype the client cannot express");
        arg.bias_vector = true;
        arg.bias_type   = *biasType;
        arg.bias_source = *source;
        arg.gradient    = gradient;
        if(aux)
        {
            auto required
                = gradient ? TensileLite::ActivationType::DRelu : TensileLite::ActivationType::Relu;
            auto allowed = allowedActivations(solution);
            if(allowed && std::find(allowed->begin(), allowed->end(), required) == allowed->end())
                return unsupported("required ReLU activation not allowed");
            auto auxType = hipType(p.eType);
            if(!auxType)
                return unsupported("E datatype the client cannot express");
            arg.use_e           = true;
            arg.aux_type        = *auxType;
            arg.activation_type = hipblaslt_activation_type::relu;
        }
    }
    arg.amaxD = p.outputAmaxD;

    for(bool isA : {true, false})
    {
        int  block = isA ? p.mxBlockA : p.mxBlockB;
        bool swz   = isA ? p.swizzleTensorA : p.swizzleTensorB;
        if(swz && !swizzleSupported(isA ? p.aType : p.bType))
            return unsupported("swizzled storage type the client ignores");
        (isA ? arg.swizzle_a : arg.swizzle_b) = swz;
        if(!block)
            continue;
        auto mode = mxScaling(block,
                              isA ? p.mxTypeA : p.mxTypeB,
                              p.mxScaleFormat,
                              isA ? p.transA : p.transB,
                              isA,
                              options.gfx1250);
        if(!mode)
            return unsupported("MX scale layout the client cannot express");
        (isA ? arg.scaleA : arg.scaleB) = *mode;
    }

    // A grouped solution runs two independent members of the chosen shape.
    int const members = p.groupedGemm ? 2 : 1;
    if(p.groupedGemm)
    {
        arg.grouped_gemm = members;
        arg.use_ext      = true;
        if(p.useBias && !arg.bias_vector)
        {
            auto const& sources = p.biasSrcWhiteList;
            if(p.useBias != 1 || std::find(sources.begin(), sources.end(), 3) == sources.end())
                return unsupported("grouped bias outside D rows");
            auto const&      types = p.biasDataTypeWhiteList;
            rocisa::DataType biasType
                = types.empty()
                          || std::find(types.begin(), types.end(), p.computeType) != types.end()
                      ? p.computeType
                      : types.front();
            auto hipBias = hipType(biasType);
            if(!hipBias)
                return unsupported("bias datatype the client cannot express");
            arg.bias_vector = true;
            arg.bias_type   = *hipBias;
            arg.bias_source = hipblaslt_bias_source::d;
        }
        if(p.useScaleAB == "Scalar")
            arg.scaleA = arg.scaleB = hipblaslt_scaling_format::Scalar;
        else if(!p.useScaleAB.empty())
            return unsupported("grouped input scaling " + p.useScaleAB);
        if(p.aType != p.bType)
        {
            // The descriptor-based grouped API rejects mixed A/B storage; the
            // dimension-based C++ API derives compute inputs from the compute enum.
            if(p.computeType == rocisa::DataType::Float
               && p.computeInputTypeA == rocisa::DataType::Half
               && p.computeInputTypeB == rocisa::DataType::Half)
                arg.compute_type = HIPBLAS_COMPUTE_32F_FAST_16F;
            else if(p.computeType == rocisa::DataType::Float
                    && p.computeInputTypeA == rocisa::DataType::BFloat16
                    && p.computeInputTypeB == rocisa::DataType::BFloat16)
                arg.compute_type = HIPBLAS_COMPUTE_32F_FAST_16BF;
            else
                return unsupported("mixed grouped compute inputs");
            arg.use_ext_setproblem = true;
        }
    }
    for(int i = 0; i < members; ++i)
    {
        arg.M[i] = m;
        arg.N[i] = n;
        arg.K[i] = k;
    }
    arg.raise_to_packed_layout(true);
    return arg;
}
