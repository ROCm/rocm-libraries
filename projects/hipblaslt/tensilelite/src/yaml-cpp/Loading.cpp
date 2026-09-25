// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/yaml-cpp/Loading.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/yaml-cpp/YAML.hpp>

namespace TensileLite
{
    namespace
    {
        template <typename MyProblem, typename MySolution>
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
            loadYaml(YAML::Node const&                   root,
                     std::string const&                  filename,
                     const std::vector<LazyLoadingInit>& preloaded)
        {
            std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> library;
            LibraryIOContext<MySolution> context{filename, preloaded, nullptr};
            Serialization::YamlCppInput  input(root, &context);
            input.input(library);

            if(!input.error.empty())
            {
                if(Debug::Instance().printDataInit())
                {
                    std::cout << "Error loading YAML data:" << std::endl;
                    for(auto const& error : input.error)
                        std::cout << error << std::endl;
                }
                return nullptr;
            }
            return library;
        }
    } // namespace

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        YamlCppLoadLibraryFile(std::string const&                  filename,
                               const std::vector<LazyLoadingInit>& preloaded)
    {
        try
        {
            return loadYaml<MyProblem, MySolution>(YAML::LoadFile(filename), filename, preloaded);
        }
        catch(std::exception const& exception)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading " << filename << " (YAML):" << std::endl
                          << exception.what() << std::endl;
            return nullptr;
        }
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        YamlCppLoadLibraryData(std::vector<uint8_t> const& data, std::string filename)
    {
        try
        {
            std::string yaml(reinterpret_cast<char const*>(data.data()), data.size());
            return loadYaml<MyProblem, MySolution>(YAML::Load(yaml), filename, {});
        }
        catch(std::exception const& exception)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading YAML data:" << std::endl
                          << exception.what() << std::endl;
            return nullptr;
        }
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        YamlCppLoadLibraryFile<ContractionProblemGemm, ContractionSolution>(
            std::string const& filename, const std::vector<LazyLoadingInit>& preloaded);

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        YamlCppLoadLibraryData<ContractionProblemGemm, ContractionSolution>(
            std::vector<uint8_t> const& data, std::string filename);
} // namespace TensileLite
