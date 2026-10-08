// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include <algorithm>
#include <fstream>
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
        bool compatible(const Target& declared, const Target& device)
        {
            const auto agrees = [](TargetFeature declared, TargetFeature device) {
                return declared == TargetFeature::Any || declared == device;
            };
            return declared.processor == device.processor
                   && agrees(declared.sramecc, device.sramecc)
                   && agrees(declared.xnack, device.xnack);
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
            Status build(const SourceCodeBundle& source, BuildKind kind, Artifact& out) const override
            {
                out                = {};
                const auto& target = source.target;
                co::Options options;
                options.codeObjectVersion = source.codeObjectVersion;
                options.retargetAssembly  = true;
                options.linkerFlags       = {"-Xlinker", "--build-id=sha1"};
                options.rocmPath          = co::rocmPath();

                if(source.mainKernel.name.empty() || source.mainKernel.bytes.empty()
                   || std::any_of(source.kernelNames.begin(),
                                  source.kernelNames.end(),
                                  [](const std::string& name) { return name.empty(); }))
                    return failure("A generated solution needs a main kernel", {});
                if(target.processor == Processor::gfx000)
                    return failure("A build needs the device's target processor", {});
                if(kind == BuildKind::Relocatable && !source.helpers.empty())
                    return failure("A relocatable build compiles the main kernel alone", {});

                if(source.mainKernel.kind == Artifact::Kind::Assembly)
                {
                    const auto sourceText = text(source.mainKernel.bytes);
                    const auto declared   = declaredTarget(sourceText);
                    if(!declared.empty())
                    {
                        const auto parsed = Target::fromTargetId(declared);
                        if(!parsed || !compatible(*parsed, target))
                            return failure("Assembly " + source.mainKernel.name + " targets "
                                               + declared + ", not " + target.targetId(),
                                           {});
                    }
                    options.wavefrontSize = declaredWavefrontSize(sourceText);
                }

                auto compile = [&](const Artifact& kernel, Artifact& object) -> Status {
                    SourceCodeBundle unit;
                    unit.mainKernel = kernel;
                    unit.includes   = source.includes;
                    unit.hipFlags   = source.hipFlags;
                    auto compiled   = co::compileRelocatable(unit, target, options);
                    if(!compiled.ok())
                        return failure("comgr could not compile " + kernel.name, compiled.log);
                    object = std::move(compiled.artifact);
                    return {};
                };

                Artifact mainObject;
                if(auto status = compile(source.mainKernel, mainObject); !status.ok())
                    return status;
                if(kind == BuildKind::Relocatable)
                {
                    out = std::move(mainObject);
                    return {};
                }

                std::vector<Artifact> objects{std::move(mainObject)};
                for(const auto& helper : source.helpers)
                {
                    if(helper.name.empty() || helper.bytes.empty())
                        return failure("A helper has no name or no bytes", {});
                    if(helper.kind == Artifact::Kind::Relocatable)
                        objects.push_back(helper);
                    else if(helper.kind == Artifact::Kind::Hip)
                    {
                        Artifact object;
                        if(auto status = compile(helper, object); !status.ok())
                            return status;
                        objects.push_back(std::move(object));
                    }
                    else
                        return failure("Helper " + helper.name
                                           + " is neither HIP source nor a relocatable",
                                       {});
                }

                auto linked = co::linkRelocatables(objects, target, options);
                if(!linked.ok())
                    return failure("comgr could not link the executable", linked.log);

                const auto metadata = co::inspect(linked.artifact.bytes);
                if(!metadata.ok())
                    return failure("Cannot read the built executable", linked.log + metadata.log);
                if(metadata.isaName != target.isaName())
                    return failure("The executable targets " + metadata.isaName + ", not "
                                       + target.isaName(),
                                   linked.log);
                if(metadata.codeObjectVersion != source.codeObjectVersion)
                    return failure("The executable is code-object version "
                                       + std::to_string(metadata.codeObjectVersion) + ", not "
                                       + std::to_string(source.codeObjectVersion),
                                   linked.log);
                const auto& names = metadata.kernelNames;
                for(const auto& name : source.kernelNames)
                    if(std::find(names.begin(), names.end(), name) == names.end())
                        return failure("The executable does not define kernel " + name,
                                       linked.log);
                out      = std::move(linked.artifact);
                out.name = names.empty()                 ? source.mainKernel.name
                           : source.kernelNames.empty() ? names.front()
                                                        : source.kernelNames.front();
                return {};
            }

        private:
            // Names the first error line of log in the message and keeps log.
            static Status failure(std::string message, std::string log)
            {
                Status     status{Status::Code::Failed, Stage::Build, std::move(message)};
                const auto error = firstError(log);
                if(!error.empty())
                    status.message += ": " + error;
                status.log = std::move(log);
                return status;
            }
        };
    }

    std::shared_ptr<const CodeObjectBuilder> makeComgrBuilder()
    {
        return std::make_shared<const ComgrBuilder>();
    }

    void keepBuildLog(Status& status, const fs::path& scratch)
    {
        if(status.log.empty() || scratch.empty())
            return;
        const auto    path = scratch / "comgr.log";
        std::ofstream output(path, std::ios::app);
        output << "== " << status.message << "\n" << status.log;
        if(output.flush())
        {
            std::error_code ignored;
            status.logPath = fs::absolute(path, ignored).u8string();
            status.message += "; see " + status.logPath;
        }
    }
}
