// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

namespace hipblaslt_jit
{
    namespace
    {
        namespace co = code_object;
        namespace fs = std::filesystem;

        std::string text(const std::vector<uint8_t>& bytes)
        {
            return {bytes.begin(), bytes.end()};
        }

        // The target ID of the first `.amdgcn_target "amdgcn-amd-amdhsa--<id>"`, or empty.
        std::string declaredTarget(const std::string& source)
        {
            const std::string directive = ".amdgcn_target", triple = "amdgcn-amd-amdhsa--";
            for(auto at = source.find(directive); at != std::string::npos;
                at      = source.find(directive, at + 1))
            {
                const auto lineEnd = source.find('\n', at);
                const auto start   = source.find(triple, at);
                if(start == std::string::npos || start > lineEnd)
                    continue;
                const auto begin = start + triple.size();
                const auto end   = source.find('"', begin);
                if(end != std::string::npos && end < lineEnd)
                    return source.substr(begin, end - begin);
            }
            return {};
        }

        // A declared target is compatible when it names the device's processor and
        // only features the device has; it is then rewritten to the device's target ID.
        bool compatible(const std::string& declared, const co::Target& device)
        {
            const auto target = co::Target::fromTargetId(declared);
            return target.gfx == device.gfx
                   && std::all_of(target.features.begin(),
                                  target.features.end(),
                                  [&](const std::string& feature) {
                                      return std::count(device.features.begin(),
                                                        device.features.end(),
                                                        feature)
                                             != 0;
                                  });
        }

        // 32 or 64 from `.amdhsa_wavefront_size32 <0|1>`, or 0 for the processor default.
        int declaredWavefrontSize(const std::string& source)
        {
            const std::string directive = ".amdhsa_wavefront_size32";
            const auto        at        = source.find(directive);
            if(at == std::string::npos)
                return 0;
            auto value = source.find_first_not_of(" \t", at + directive.size());
            if(value == std::string::npos)
                return 0;
            return source[value] == '1' ? 32 : source[value] == '0' ? 64 : 0;
        }

        std::string firstError(const std::string& log)
        {
            std::istringstream lines(log);
            std::string        line;
            while(std::getline(lines, line))
                if(line.find("error") != std::string::npos)
                    return line.substr(0, 512);
            return {};
        }

        class ComgrBuilder final : public CodeObjectBuilder
        {
        public:
            Status build(const GeneratedSolution& solution,
                         const GenerationRequest& request,
                         BuiltSolution&           built) const override
            {
                built             = {};
                const auto target = co::Target::fromTargetId(request.target.targetId);
                co::Options options;
                options.codeObjectVersion = request.codeObjectVersion;
                options.retargetAssembly  = true;
                options.compilerFlags     = {"-D__HIP_HCC_COMPAT_MODE__=1"};
                options.linkerFlags       = {"-Xlinker", "--build-id=sha1"};
                options.rocmPath          = co::rocmPath();

                std::vector<co::AssemblySource> assembly;
                std::vector<co::HipSource>      hip;
                std::set<std::string>           includes;
                int                             wavefrontSize = 0;
                bool                            main          = false;
                for(const auto& unit : solution.units)
                {
                    main        = main || unit.role == BuildUnit::Role::Main;
                    auto source = text(unit.bytes);
                    if(unit.kind == BuildUnit::Kind::Hip)
                    {
                        for(const auto& include : unit.includes)
                            if(includes.insert(include.name).second)
                                options.includes.push_back({include.name, text(include.bytes)});
                        hip.push_back({unit.name, std::move(source)});
                        continue;
                    }
                    const auto declared = declaredTarget(source);
                    if(!declared.empty() && !compatible(declared, target))
                        return failure(request,
                                       solution,
                                       "Assembly " + unit.name + " targets " + declared + ", not "
                                           + target.targetId(),
                                       {});
                    const auto wave = declaredWavefrontSize(source);
                    if(wave != 0 && wavefrontSize != 0 && wave != wavefrontSize)
                        return failure(
                            request, solution, "Assembly units disagree on the wavefront size", {});
                    wavefrontSize = wave != 0 ? wave : wavefrontSize;
                    assembly.push_back({unit.name, std::move(source)});
                }
                if(!main || solution.kernelName.empty())
                    return failure(request,
                                   solution,
                                   "A generated solution needs a main unit and a kernel name",
                                   {});

                std::string                  log;
                std::vector<co::Relocatable> objects;
                if(!assembly.empty())
                {
                    auto assemblyTarget          = target;
                    assemblyTarget.wavefrontSize = wavefrontSize;
                    auto result = co::assembleRelocatables(assembly, assemblyTarget, options);
                    log += result.log;
                    if(!result.ok())
                        return failure(request, solution, "comgr could not assemble", log);
                    objects = std::move(result.objects);
                }
                if(!hip.empty())
                {
                    auto result = co::compileHipRelocatables(hip, target, options);
                    log += result.log;
                    if(!result.ok())
                        return failure(request, solution, "comgr could not compile HIP", log);
                    for(auto& object : result.objects)
                        objects.push_back(std::move(object));
                }
                auto linked = co::link(objects, target, options);
                log += linked.log;
                if(!linked.ok())
                    return failure(request, solution, "comgr could not link", log);

                const auto metadata = co::readMetadata(linked.bytes);
                if(!metadata.ok())
                    return failure(
                        request, solution, "Cannot read the built code object", log + metadata.log);
                if(metadata.metadata.isaName != target.isaName())
                    return failure(request,
                                   solution,
                                   "Built code object targets " + metadata.metadata.isaName
                                       + ", not " + target.isaName(),
                                   log);
                const auto& kernels = metadata.metadata.kernels;
                if(std::none_of(kernels.begin(), kernels.end(), [&](const co::KernelMetadata& k) {
                       return k.name == solution.kernelName;
                   }))
                    return failure(request,
                                   solution,
                                   "Built code object does not define kernel "
                                       + solution.kernelName,
                                   log);
                built.generated = solution;
                built.object.bytes.assign(linked.bytes.begin(), linked.bytes.end());
                return {};
            }

        private:
            // Appends the comgr log to <scratch>/comgr.log and names it in the message.
            static Status failure(const GenerationRequest&  request,
                                  const GeneratedSolution&  solution,
                                  std::string               message,
                                  const std::string&        log)
            {
                Status status{Status::Code::Failed, Stage::Build, std::move(message)};
                const auto error = firstError(log);
                if(!error.empty())
                    status.message += ": " + error;
                if(log.empty() || request.scratch.empty())
                    return status;
                const auto    path = request.scratch / "comgr.log";
                std::ofstream output(path, std::ios::app);
                output << "== " << solution.kernelName << "\n" << log;
                if(output.flush())
                {
                    std::error_code ignored;
                    status.logPath = fs::absolute(path, ignored).u8string();
                    status.message += "; see " + status.logPath;
                }
                return status;
            }
        };
    }

    std::shared_ptr<const CodeObjectBuilder> makeComgrBuilder()
    {
        return std::make_shared<const ComgrBuilder>();
    }
}
