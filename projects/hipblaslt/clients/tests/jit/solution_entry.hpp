// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-library.hpp"
#include "test_helpers.hpp"
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// The solutions the tests publish from the committed assembly under data/.
// Each record is the kernel's own, read from its custom.config.
namespace hipblaslt_jit_test
{
    namespace fs = std::filesystem;

    using Solution  = std::shared_ptr<TensileLite::ContractionSolution>;
    using Predicate = std::shared_ptr<
        TensileLite::Predicates::Predicate<TensileLite::ContractionProblemGemm>>;

    // Solutions from one assembled kernel: the source that links its
    // relocatable into an executable, and the K values at which the tests
    // publish them.
    struct Prepared
    {
        std::string                        name;
        hipblaslt_jit::TensileliteAssembly assembly;
        hipblaslt_jit::SourceCodeBundle    source;
        std::vector<size_t>                ks;
    };

    // A copy of solution named <kernel><suffix> with workGroupMapping and the
    // extra problem predicates.
    inline Solution variant(const Solution&        solution,
                            const std::string&     suffix,
                            int                    workGroupMapping,
                            std::vector<Predicate> extra)
    {
        using And = TensileLite::Predicates::And<TensileLite::ContractionProblemGemm>;
        auto copy                          = hipblaslt_jit::copySolution(solution);
        copy->solutionName                 = solution->kernelName + suffix;
        copy->sizeMapping.workGroupMapping = workGroupMapping;
        if(!extra.empty())
        {
            auto terms = extra;
            if(const auto all = std::dynamic_pointer_cast<And>(solution->problemPredicate))
                terms.insert(terms.begin(), all->value.begin(), all->value.end());
            else
                terms.insert(terms.begin(), solution->problemPredicate);
            copy->problemPredicate = std::make_shared<And>(terms);
        }
        return copy;
    }

    // Assembles the device's committed plain kernel once and returns plain,
    // its solution as the kernel records it, and plain-pair, that kernel as two
    // solutions: the first only for K a multiple of 512, with WorkGroupMapping
    // 8, the second for any K, with WorkGroupMapping 1.
    inline std::vector<Prepared> prepareSolutions(const fs::path&              data,
                                                  const hipblaslt_jit::Target& target,
                                                  const fs::path&              scratch = {})
    {
        std::vector<hipblaslt_jit::Artifact> units;
        const auto sources = data / fs::u8path(target.processorName()) / "plain" / "sources";
        {
            INFO(("No committed plain kernel in " + sources.u8string()));
            REQUIRE((fs::is_directory(sources)));
        }
        for(const auto& file : fs::directory_iterator(sources))
            if(file.path().extension() == ".s")
            {
                const auto text = readFile(file.path());
                units.push_back({file.path().filename().u8string(),
                                 hipblaslt_jit::Artifact::Kind::Assembly,
                                 std::vector<uint8_t>(text.begin(), text.end())});
            }
        {
            INFO(("A source bundle has one main kernel in " + sources.u8string()));
            REQUIRE((units.size() == 1));
        }

        hipblaslt_jit::SourceCodeBundle source;
        source.mainKernel = units.front();
        source.target     = target;
        hipblaslt_jit::TensileliteAssembly assembly;
        hipblaslt_jit::Artifact            relocatable;
        auto                               status = hipblaslt_jit::TensileliteAssembly::make(
            *hipblaslt_jit::makeComgrBuilder(), source, assembly, relocatable);
        hipblaslt_jit::keepBuildLog(status, scratch);
        {
            INFO(("Cannot read the solution of " + source.mainKernel.name + ": " + status.message));
            REQUIRE((status.ok() && assembly.solutions.size() == 1));
        }
        const auto& solution = assembly.solutions.front();
        source.mainKernel    = relocatable;
        source.kernelNames   = {solution->kernelName};

        auto pair      = assembly;
        pair.solutions = {
            variant(solution,
                    "_K512_WGM8",
                    8,
                    {std::make_shared<TensileLite::Predicates::Contraction::BoundSizeMultiple>(
                        0, 512)}),
            variant(solution, "_WGM1", 1, {})};
        return {{"plain", assembly, source, {512}}, {"plain-pair", pair, source, {1024, 256}}};
    }

    // The solutions whose hardware and problem predicates accept problem.
    inline std::vector<Solution> supported(const std::vector<Solution>&               solutions,
                                           const TensileLite::ContractionProblemGemm& problem,
                                           const TensileLite::Hardware&               hardware)
    {
        std::vector<Solution> result;
        for(const auto& solution : solutions)
            if((*solution->hardwarePredicate)(hardware) && (*solution->problemPredicate)(problem))
                result.push_back(solution);
        return result;
    }
}
