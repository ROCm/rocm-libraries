// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include <Tensile/Tensile.hpp>
#include <algorithm>
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

        void checkHip(hipError_t status, const char* operation)
        {
            if(status != hipSuccess)
                throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
        }

        bool contains(const std::vector<std::string>& names, const std::string& name)
        {
            return std::find(names.begin(), names.end(), name) != names.end();
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

    TensileSource readTensileSourceBundle(const std::filesystem::path& bundle)
    {
        namespace artifacts = hipblaslt_jit::source_bundle;
        auto          sources = artifacts::readSourceBundle(bundle);
        TensileSource result;
        result.solution.entry = std::move(sources.library);
        result.library        = loadGemmLibrary(result.solution.entry);
        for(const auto& [index, solution] : result.library->solutions)
            if(!contains(result.solution.kernelNames, solution->kernelName))
                result.solution.kernelNames.push_back(solution->kernelName);
        for(auto& file : sources.assembly)
            result.solution.units.push_back(
                {std::move(file.name), std::move(file.bytes), BuildUnit::Kind::Assembly, {}});
        std::vector<IncludeFile> includes;
        for(auto& header : sources.headers)
            includes.push_back({std::move(header.name), std::move(header.bytes)});
        for(auto& file : sources.hip)
            result.solution.units.push_back(
                {std::move(file.name), std::move(file.bytes), BuildUnit::Kind::Hip, includes});
        return result;
    }

    std::shared_ptr<TensileBundle> parseTensileBundle(const BuiltSolution&                   built,
                                                      std::shared_ptr<TensileLite::Hardware> hardware)
    {
        require(hardware != nullptr, "TensileLite hardware is required");
        auto bundle      = std::make_shared<TensileBundle>();
        bundle->hardware = std::move(hardware);
        bundle->kernels  = built.generated.kernelNames;
        bundle->library  = loadGemmLibrary(built.generated.entry);
        std::vector<std::string> used;
        for(const auto& [index, solution] : bundle->library->solutions)
        {
            require(contains(bundle->kernels, solution->kernelName),
                    "Library solution " + std::to_string(index) + " names kernel "
                        + solution->kernelName + ", which the build does not define");
            used.push_back(solution->kernelName);
        }
        for(const auto& kernel : bundle->kernels)
            require(contains(used, kernel), "No library solution names the built kernel " + kernel);
        return bundle;
    }

    void loadTensileBundle(TensileBundle& bundle, const BuiltSolution& built)
    {
        bundle.adapter = std::make_shared<TensileLite::hip::SolutionAdapter>(false, "jit-gemm");
        checkHip(bundle.adapter->loadCodeObjectBytes(built.object.bytes),
                 "Load generated code object");
        for(const auto& kernel : bundle.kernels)
            checkHip(bundle.adapter->initKernel(kernel), "Resolve generated kernel symbol");
    }
}
