// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include <Tensile/Tensile.hpp>
#include <stdexcept>

namespace hipblaslt_jit
{
    namespace
    {
        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }
    }

    std::shared_ptr<TensileLibrary> loadGemmLibrary(const std::vector<uint8_t>& entry)
    {
        auto library = std::dynamic_pointer_cast<TensileLibrary>(
            TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(entry));
        require(library && !library->solutions.empty(),
                "Expected a non-lazy library containing local solutions");
        int expected = 0;
        for(const auto& [index, solution] : library->solutions)
            require(index == expected++ && solution && solution->index == index,
                    "Expected local solutions 0 to N-1");
        return library;
    }

    SourceBundleRead readTensileSourceBundle(const std::filesystem::path& bundle)
    {
        namespace artifacts = hipblaslt_jit::source_bundle;
        SourceBundleRead result;
        result.source      = artifacts::readSourceBundle(bundle);
        const auto library = loadGemmLibrary(
            artifacts::readArtifact(artifacts::artifact(bundle, "library/TensileLibrary.dat")));
        result.kernelName = library->solutions.at(0)->kernelName;
        return result;
    }
}
