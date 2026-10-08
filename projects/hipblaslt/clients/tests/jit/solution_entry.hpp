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

    // The workspace the tests offer a solution.
    constexpr size_t workspaceLimit = 64 << 20;

    // One assembled kernel: the bundle that links its relocatable into an
    // executable, with the solutions that executable runs, and the K values at
    // which the tests publish them.
    struct Prepared
    {
        std::string                     name;
        hipblaslt_jit::SourceCodeBundle source;
        std::vector<size_t>             ks;
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

    // Reads the device's committed plain bundle once with
    // readTensileSourceBundle and returns plain, its solution as the kernel
    // records it, and plain-pair, that kernel as two solutions: the first only
    // for K a multiple of 512, with WorkGroupMapping 8, the second for any K,
    // with WorkGroupMapping 1.
    inline std::vector<Prepared> prepareSolutions(const fs::path&              data,
                                                  const hipblaslt_jit::Target& target,
                                                  const fs::path&              scratch = {})
    {
        const auto bundle = data / fs::u8path(target.processorName()) / "plain";
        hipblaslt_jit::SourceCodeBundle source;
        auto status = hipblaslt_jit::readTensileSourceBundle(*hipblaslt_jit::makeComgrBuilder(),
                                                             bundle,
                                                             target,
                                                             hipblaslt_jit::jitCodeObjectVersion,
                                                             source);
        hipblaslt_jit::keepBuildLog(status, scratch);
        {
            INFO(("Cannot read the solution of " + bundle.u8string() + ": " + status.message));
            REQUIRE((status.ok() && source.solutions.size() == 1
                     && source.mainKernel.kind == hipblaslt_jit::Artifact::Kind::Relocatable));
        }

        const auto solution = source.solutions.front();
        auto       pair     = source;
        pair.solutions      = {
            variant(solution,
                    "_K512_WGM8",
                    8,
                    {std::make_shared<TensileLite::Predicates::Contraction::BoundSizeMultiple>(
                        0, 512)}),
            variant(solution, "_WGM1", 1, {})};
        return {{"plain", source, {512}}, {"plain-pair", pair, {1024, 256}}};
    }
}
