
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

#include <rocRoller/Assemblers/InProcessAssembler.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    RegisterComponent(InProcessAssembler);
    static_assert(Component::Component<InProcessAssembler>);

    bool InProcessAssembler::Match(Argument arg)
    {
        return arg == AssemblerType::InProcess;
    }

    AssemblerPtr InProcessAssembler::Build(Argument arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<InProcessAssembler>();
    }

    std::string InProcessAssembler::name() const
    {
        return Name;
    }

    std::vector<char> InProcessAssembler::assembleMachineCode(const std::string& machineCode,
                                                              const GPUArchitectureTarget& target)
    {
        return assembleMachineCode(machineCode, target, "");
    }

    std::vector<char> InProcessAssembler::assembleMachineCode(const std::string& machineCode,
                                                              const GPUArchitectureTarget& target,
                                                              const std::string& kernelName)
    {
        // Time assembleMachineCode function
        TIMER(t, "Assembler::assembleMachineCode");

        // Create a temporary directory to hold the files provided to and created by the linker.
        auto  tmpFolderTemplate = std::filesystem::temp_directory_path() / "rocroller-XXXXXX";
        char* tmpFolder         = mkdtemp(const_cast<char*>(tmpFolderTemplate.c_str()));
        if(!tmpFolder)
            throw std::runtime_error("Unable to create temporary directory");

        std::string outputName = kernelName.size() ? kernelName : "a";
        outputName += "_" + target.toString();
        std::replace(outputName.begin(), outputName.end(), ':', '-');
        std::string tmpObjectFile       = std::string(tmpFolder) + "/" + outputName + ".o";
        std::string tmpSharedObjectFile = std::string(tmpFolder) + "/" + outputName + ".so";

        try
        {
            assemble(machineCode.c_str(),
                     target.getVersionString().c_str(),
                     target.getLLVMFeatureString().c_str(),
                     tmpObjectFile.c_str());
            link(tmpObjectFile.c_str(), tmpSharedObjectFile.c_str());
        }
        catch(const std::runtime_error& e)
        {
            // Remove temporary directory if there were any errors.
            std::filesystem::remove_all(tmpFolder);
            throw;
        }

        // Read in shared object into vector
        std::ifstream     fstream(tmpSharedObjectFile, std::ios::in | std::ios::binary);
        std::vector<char> result((std::istreambuf_iterator<char>(fstream)),
                                 std::istreambuf_iterator<char>());

        // Remove temporary directory
        std::filesystem::remove_all(tmpFolder);

        return result;
    }
}
