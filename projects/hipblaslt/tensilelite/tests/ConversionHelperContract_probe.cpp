// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <Tensile/Activation.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

using namespace TensileLite;

namespace
{
    const char* activationName(ActivationType value)
    {
        switch(value)
        {
        case ActivationType::None: return "none";
        case ActivationType::Relu: return "relu";
        case ActivationType::Gelu: return "gelu";
        case ActivationType::Sigmoid: return "sigmoid";
        case ActivationType::All: return "all";
        case ActivationType::Hipblaslt_all: return "hipblaslt_all";
        default: throw std::runtime_error("unexpected activation in contract fixture");
        }
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
        return 2;
    try
    {
        auto library = std::dynamic_pointer_cast<
            MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>>(
                LoadLibraryFilePreload<ContractionProblemGemm>(argv[1], {LazyLoadingInit::All}));
        if(!library || library->solutions.empty())
            throw std::runtime_error("generated library has no materialized solutions");
        for(auto const& entry : library->solutions)
        {
            auto const& solution = *entry.second;
            auto const& type = solution.problemType;
            TensorDescriptor a("a", type.aType,
                               {type.transA ? 256UL : 128UL, type.transA ? 128UL : 256UL, 1},
                               {1, type.transA ? 256UL : 128UL, 128 * 256});
            TensorDescriptor b("b", type.bType,
                               {type.transB ? 64UL : 256UL, type.transB ? 256UL : 64UL, 1},
                               {1, type.transB ? 64UL : 256UL, 64 * 256});
            TensorDescriptor c("c", type.cType, {128, 64, 1}, {1, 128, 128 * 64});
            TensorDescriptor d("d", type.dType, {128, 64, 1}, {1, 128, 128 * 64});
            auto problem = ContractionProblemGemm::GEMM(
                type.transA, type.transB, a, {}, b, {}, c, {}, d, {},
                type.useBeta ? 1.0 : 0.0);
            problem.setComputeInputTypeA(type.computeInputTypeA);
            problem.setComputeInputTypeB(type.computeInputTypeB);
            problem.setUseBias(type.useBias);
            if(type.useBias)
                problem.setBias(type.biasDataTypeWhiteList.at(0), 128, 128);
            problem.setUseScaleCD(type.useScaleCD);
            problem.setUseScaleAlphaVec(type.useScaleAlphaVec);
            problem.setActivationComputeType(type.activationComputeDataType);
            constexpr ActivationType activations[]{ActivationType::None, ActivationType::Relu,
                                                    ActivationType::Gelu, ActivationType::Sigmoid};
            bool generalActivation = type.activationType == ActivationType::All
                                     || type.activationType == ActivationType::Hipblaslt_all;
            for(size_t ai = 0; ai < (generalActivation ? 4UL : 1UL); ++ai)
            {
                auto activation = generalActivation ? activations[ai] : type.activationType;
                problem.setActivationType(activation);
                problem.setParams().setActivationEnum(activation);
                ContractionInputs inputs;
                for(size_t gsu : {2, 4, 8, 16})
                {
                    for(size_t vw : {1, 2, 4})
                    {
                        auto name = solution.outputConversionKernelName(problem, inputs, vw, gsu);
                        std::cout << entry.first << '\t' << solution.bespokeNarrowing << '\t'
                                  << type.useBeta << '\t' << type.groupedGemm << '\t'
                                  << type.useBias << '\t' << type.useScaleCD << '\t'
                                  << type.useScaleAlphaVec << '\t'
                                  << solution.sizeMapping.globalAccumulation << '\t'
                                  << rocisa::TypeAbbrev(type.dType) << '\t'
                                  << activationName(type.activationType)
                                  << '\t' << activationName(activation) << '\t' << gsu << '\t'
                                  << vw << '\t' << name << '\n';
                    }
                }
            }
        }
        return 0;
    }
    catch(std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
