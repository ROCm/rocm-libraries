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
            Status build(const SourceCodeBundle& source,
                         const BuildRequest&     request,
                         CompiledCode&           built) const override
            {
                built              = {};
                const auto  target = co::Target::fromTargetId(request.targetId);
                co::Options options;
                options.codeObjectVersion = request.codeObjectVersion;
                options.retargetAssembly  = true;
                options.linkerFlags       = {"-Xlinker", "--build-id=sha1"};
                options.rocmPath          = co::rocmPath();

                if(source.mainKernel.name.empty() || source.mainKernel.bytes.empty()
                   || std::any_of(request.kernelNames.begin(),
                                  request.kernelNames.end(),
                                  [](const std::string& name) { return name.empty(); }))
                    return failure(request, "A generated solution needs a main kernel", {});

                int wavefrontSize = 0;
                if(source.mainKernel.kind == Artifact::Kind::Assembly)
                {
                    const auto sourceText = text(source.mainKernel.bytes);
                    const auto declared   = declaredTarget(sourceText);
                    if(!declared.empty() && !compatible(declared, target))
                        return failure(request,
                                       "Assembly " + source.mainKernel.name + " targets " + declared
                                           + ", not " + target.targetId(),
                                       {});
                    wavefrontSize = declaredWavefrontSize(sourceText);
                }

                auto assemblyTarget          = target;
                assemblyTarget.wavefrontSize = wavefrontSize;

                // Helpers left on the bundle are compiled here. JitLibrary is the
                // helper cache: a caller that has a library acquires each helper
                // from it and does not leave that source on the bundle. This
                // builder keeps no cache of its own. comgr's file cache is not
                // a substitute.
                auto compileHelper = [&](const Artifact& helper, Artifact& out) -> Status {
                    if(helper.name.empty() || helper.kind != Artifact::Kind::Hip)
                        return failure(request, "A helper source has no HIP name", {});
                    SourceCodeBundle helperSource;
                    helperSource.mainKernel = helper;
                    helperSource.includes   = source.includes;
                    helperSource.hipFlags   = source.hipFlags;
                    auto helperObject       = co::build(helperSource, assemblyTarget, options);
                    if(!helperObject.ok())
                        return failure(request,
                                       "comgr could not build helper " + helper.name,
                                       helperObject.log);
                    const auto helperMeta = co::inspect(helperObject.codeObject.bytes);
                    if(!helperMeta.ok() || helperMeta.isaName != target.isaName()
                       || helperMeta.codeObjectVersion != request.codeObjectVersion)
                        return failure(request,
                                       "Helper " + helper.name + " is not a code object for "
                                           + target.isaName(),
                                       helperObject.log + helperMeta.log);
                    out      = std::move(helperObject.codeObject);
                    out.name = helper.name;
                    out.kind = Artifact::Kind::CodeObject;
                    return {};
                };
                for(const auto& helper : source.helpers)
                {
                    Artifact builtHelper;
                    const auto helperStatus = compileHelper(helper, builtHelper);
                    if(!helperStatus.ok())
                        return helperStatus;
                    built.helpers.push_back(std::move(builtHelper));
                }

                auto linked = co::build(source, assemblyTarget, options);
                if(!linked.ok())
                    return failure(request, "comgr could not build the code object", linked.log);

                const auto metadata = co::inspect(linked.codeObject.bytes);
                if(!metadata.ok())
                    return failure(
                        request, "Cannot read the built code object", linked.log + metadata.log);
                if(metadata.isaName != target.isaName())
                    return failure(request,
                                   "Built code object targets " + metadata.isaName + ", not "
                                       + target.isaName(),
                                   linked.log);
                if(metadata.codeObjectVersion != request.codeObjectVersion)
                    return failure(request,
                                   "Built code object is version "
                                       + std::to_string(metadata.codeObjectVersion) + ", not "
                                       + std::to_string(request.codeObjectVersion),
                                   linked.log);
                const auto& names = metadata.kernelNames;
                if(!request.kernelNames.empty())
                {
                    for(const auto& name : request.kernelNames)
                        if(std::find(names.begin(), names.end(), name) == names.end())
                            return failure(request,
                                           "Built code object does not define kernel " + name,
                                           linked.log);
                }
                built.codeObject      = std::move(linked.codeObject);
                built.codeObject.kind = Artifact::Kind::CodeObject;
                built.codeObject.name = names.empty() ? source.mainKernel.name
                                      : request.kernelNames.empty() ? names.front()
                                                                    : request.kernelNames.front();
                return {};
            }

        private:
            // Appends the comgr log to <scratch>/comgr.log and names it in the message.
            static Status
                failure(const BuildRequest& request, std::string message, const std::string& log)
            {
                Status     status{Status::Code::Failed, Stage::Build, std::move(message)};
                const auto error = firstError(log);
                if(!error.empty())
                    status.message += ": " + error;
                if(log.empty() || request.scratch.empty())
                    return status;
                const auto    path = request.scratch / "comgr.log";
                std::ofstream output(path, std::ios::app);
                output << "==";
                for(const auto& name : request.kernelNames)
                    output << ' ' << name;
                output << "\n" << log;
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
