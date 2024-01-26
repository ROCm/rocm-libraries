

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <rocRoller/Assemblers/SubprocessAssembler.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    RegisterComponent(SubprocessAssembler);
    static_assert(Component::Component<SubprocessAssembler>);

    SubprocessAssembler::SubprocessAssembler() {}

    SubprocessAssembler::~SubprocessAssembler() {}

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

    void SubprocessAssembler::executeChecked(std::string const& command)
    {
        auto [retCode, output] = execute(command);

        AssertFatal(retCode == 0, ShowValue(command), ShowValue(retCode), ShowValue(output));
    }

    std::string SubprocessAssembler::makeTempFolder()
    {
        auto  tmpFolderTemplate = std::filesystem::temp_directory_path() / "rocroller-XXXXXX";
        char* tmpFolder         = mkdtemp(const_cast<char*>(tmpFolderTemplate.c_str()));
        if(!tmpFolder)
            throw std::runtime_error("Unable to create temporary directory");

        return tmpFolder;
    }

    std::vector<char> SubprocessAssembler::readFile(std::string const& filename)
    {
        std::ifstream file(filename);

        AssertFatal(file.good(), "Could not read ", filename);

        std::array<char, 4096> buffer;

        std::vector<char> rv;

        file.read(buffer.data(), buffer.size());

        while(file.good() && !file.eof())
        {
            rv.insert(rv.end(), buffer.begin(), buffer.end());

            file.read(buffer.data(), buffer.size());
        }

        auto numRead = file.gcount();
        AssertFatal(numRead <= buffer.size());

        rv.insert(rv.end(), buffer.begin(), buffer.begin() + numRead);

        return rv;
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

        int foo = 5;

        std::filesystem::path tmpFolder = makeTempFolder();

        auto deleteDir = [tmpFolder](auto*) { std::filesystem::remove_all(tmpFolder); };

        // std::shared_ptr<int> tmpFolderGuard(&foo, deleteDir);

        auto assemblyFile   = tmpFolder / "kernel.s";
        auto objectFile     = tmpFolder / "kernel.o";
        auto codeObjectFile = tmpFolder / "kernel.co";

        {
            std::ofstream file(assemblyFile);
            file << machineCode;
        }

        std::string assemblerPath = "/opt/rocm/llvm/bin/clang++";

        {
            std::vector<std::string> args = {assemblerPath,
                                             "-x",
                                             "assembler",
                                             "-target",
                                             "amdgcn-amd-amdhsa",
                                             "-mcode-object-version=5",
                                             concatenate("-mcpu=", target.toString()),
                                             "-mwavefrontsize64",
                                             "-c",
                                             "-o",
                                             objectFile,
                                             assemblyFile};

            auto command = joinArgs(args);

            executeChecked(command);
        }

        {
            std::vector<std::string> args
                = {assemblerPath, "-target", "amdgcn-amd-amdhsa", "-o", codeObjectFile, objectFile};

            auto command = joinArgs(args);

            executeChecked(command);
        }

        return readFile(codeObjectFile);
    }

    std::vector<char> SubprocessAssembler::assembleMachineCode(const std::string& machineCode,
                                                               const GPUArchitectureTarget& target)
    {
        return assembleMachineCode(machineCode, target, "");
    }

}
