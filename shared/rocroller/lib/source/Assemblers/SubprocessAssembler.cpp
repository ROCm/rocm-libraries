/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <rocRoller/Assemblers/SubprocessAssembler.hpp>

#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    RegisterComponent(SubprocessAssembler);
    static_assert(Component::Component<SubprocessAssembler>);

    bool SubprocessAssembler::Match(Argument arg)
    {
        return arg == AssemblerType::Subprocess;
    }

    AssemblerPtr SubprocessAssembler::Build(Argument arg)
    {
        if(!Match(arg))
            return nullptr;

        return std::make_shared<SubprocessAssembler>();
    }

    std::string SubprocessAssembler::name() const
    {
        return Name;
    }

    std::tuple<int, std::string> SubprocessAssembler::execute(std::string const& command)
    {
        std::array<char, 128> buffer;
        std::string           result;
        FILE*                 pipe(popen(command.c_str(), "r"));
        if(!pipe)
        {
            return {-1, ""};
        }
        while(fgets(buffer.data(), buffer.size(), pipe) != nullptr)
        {
            result += buffer.data();
        }
        int ret_code = pclose(pipe);

        return {ret_code, result};
    }

    void SubprocessAssembler::executeChecked(std::string const&           command,
                                             std::function<void()> const& cleanupCall)
    {
        auto [retCode, output] = execute(command);

        if(retCode != 0)
        {
            cleanupCall();
            Throw<FatalError>(ShowValue(command), ShowValue(retCode), ShowValue(output));
        }
    }

    std::string SubprocessAssembler::makeTempFolder()
    {
        auto  tmpFolderTemplate = std::filesystem::temp_directory_path() / "rocroller-XXXXXX";
        char* tmpFolder         = mkdtemp(const_cast<char*>(tmpFolderTemplate.c_str()));
        if(!tmpFolder)
            throw std::runtime_error("Unable to create temporary directory");

        return tmpFolder;
    }

    template <typename Container>
    std::string joinArgs(Container args)
    {
        std::ostringstream msg;
        streamJoin(msg, args, " ");
        return msg.str();
    }

    std::vector<char> SubprocessAssembler::assembleMachineCode(const std::string& machineCode,
                                                               const GPUArchitectureTarget& target,
                                                               const std::string& kernelName)
    {
        TIMER(t, "Assembler::assembleMachineCode");

        std::filesystem::path tmpFolder = makeTempFolder();

        auto deleteDir = [tmpFolder]() { std::filesystem::remove_all(tmpFolder); };

        auto assemblyFile   = tmpFolder / "kernel.s";
        auto objectFile     = tmpFolder / "kernel.o";
        auto codeObjectFile = tmpFolder / "kernel.co";

        {
            std::ofstream file(assemblyFile);
            file << machineCode;
        }

        std::string assemblerPath = Settings::getInstance()->get(Settings::SubprocessAssemblerPath);
        if(assemblerPath.empty())
        {
            std::string rocmPath = Settings::getInstance()->get(Settings::ROCMPath);
            assemblerPath        = rocmPath + "/bin/amdclang++";
        }

        {

            auto const arch          = GPUArchitectureLibrary::getInstance()->GetArch(target);
            auto const wavefrontSize = arch.GetCapability(GPUCapability::DefaultWavefrontSize);

            std::vector<std::string> args = {assemblerPath,
                                             "-x",
                                             "assembler",
                                             "-target",
                                             "amdgcn-amd-amdhsa",
                                             "-mcode-object-version=5",
                                             concatenate("-mcpu=", toString(target)),
                                             (wavefrontSize == 64) ? "-mwavefrontsize64" : "",
                                             "-c",
                                             "-o",
                                             objectFile,
                                             assemblyFile};

            auto command = joinArgs(args);

            executeChecked(command, deleteDir);
        }

        {
            std::vector<std::string> args
                = {assemblerPath, "-target", "amdgcn-amd-amdhsa", "-o", codeObjectFile, objectFile};

            auto command = joinArgs(args);

            executeChecked(command, deleteDir);
        }

        auto fileContents = readFile(codeObjectFile);

        deleteDir();

        return fileContents;
    }

    std::vector<char> SubprocessAssembler::assembleMachineCode(const std::string& machineCode,
                                                               const GPUArchitectureTarget& target)
    {
        return assembleMachineCode(machineCode, target, "");
    }

}
