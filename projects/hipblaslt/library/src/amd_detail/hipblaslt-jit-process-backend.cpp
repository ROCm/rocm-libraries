// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-debug.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "rocblaslt_secure_env.hpp"
#include <algorithm>
#include <exception>
#include <functional>
#include <sstream>
#include <utility>

namespace hipblaslt_jit
{
    namespace
    {
        struct Candidate
        {
            ProcessBackendName                     name;
            bool                                   optIn = false;
            std::function<Status(ProcessBackend&)> make;
        };

        std::vector<Candidate> candidates()
        {
            std::vector<Candidate> all{
                {defaultProcessBackendName(), false, makeDefaultProcessBackend}};
            for(auto& backend : optInProcessBackends())
                all.push_back({std::move(backend.name), true, backend.make});
            return all;
        }

        // In HIPBLASLT_JIT_BACKENDS order, or the default-on candidates when it
        // is unset or empty.
        std::vector<const Candidate*> select(const std::vector<Candidate>& all,
                                             const std::string&            variable)
        {
            std::vector<const Candidate*> selected;
            if(variable.empty())
            {
                for(const auto& candidate : all)
                    if(!candidate.optIn)
                        selected.push_back(&candidate);
                return selected;
            }
            std::istringstream ids(variable);
            for(std::string id; std::getline(ids, id, ',');)
            {
                if(id.empty())
                    continue;
                const auto found = std::find_if(all.begin(), all.end(), [&](const Candidate& c) {
                    return c.name.id == id;
                });
                if(found == all.end())
                    report(Severity::Warning,
                           "backend " + id
                               + " in HIPBLASLT_JIT_BACKENDS is not in this build; ignored");
                else if(std::find(selected.begin(), selected.end(), &*found) == selected.end())
                    selected.push_back(&*found);
            }
            return selected;
        }

        ProcessJit buildProcessJit(const Candidate& candidate)
        {
            ProcessJit process{candidate.name, nullptr, {}};
            try
            {
                ProcessBackend made;
                process.configured = candidate.make(made);
                if(!process.configured.ok())
                {
                    process.configured.stage = Stage::Configure;
                    return process;
                }
                debug::Phase storePhase("store");
                auto         store = makeLibraryStore(
                    JitLibrary::process(), made.backend->info(), jitCodeObjectVersion);
                storePhase.stop();
                debug::Phase components("components");
                process.jit = std::make_shared<const Jit>(Jit::Components{std::move(made.backend),
                                                                          std::move(made.predictor),
                                                                          std::move(made.knowledge),
                                                                          makeComgrBuilder(),
                                                                          makeTensileLoader(),
                                                                          std::move(store)});
            }
            catch(const std::exception& e)
            {
                process.configured = {Status::Code::Failed, Stage::Configure, e.what()};
            }
            return process;
        }

        std::vector<ProcessJit> buildProcessJits()
        {
            const char*       value = rocblaslt_secure_getenv("HIPBLASLT_JIT_BACKENDS");
            const std::string variable(value ? value : "");
            const auto        all = candidates();
            std::vector<ProcessJit> jits;
            for(const auto* candidate : select(all, variable))
                jits.push_back(buildProcessJit(*candidate));
            if(jits.empty())
                jits.push_back({{},
                                nullptr,
                                {Status::Code::NotSupported,
                                 Stage::Configure,
                                 "HIPBLASLT_JIT_BACKENDS=" + variable
                                     + " names no backend of this build"}});
            return jits;
        }

        // With timing, a setup line reports the first processJits call.
        std::vector<ProcessJit> makeProcessJits()
        {
            if(!debug::on(debug::Timing))
                return buildProcessJits();
            debug::Record record;
            auto          made = [&] {
                debug::Scope scope(&record);
                return buildProcessJits();
            }();
            const auto failed = std::find_if(made.begin(), made.end(), [](const ProcessJit& p) {
                return !p.configured.ok();
            });
            debug::Line line(debug::Timing, "setup");
            line.add("status", failed == made.end() ? "ok" : "failed");
            if(failed != made.end())
                line.add("message", failed->configured.message);
            uint64_t total = 0;
            for(const char* phase : {"tool_check", "backend", "store", "components"})
                total += record.nanoseconds(phase);
            const std::pair<const char*, uint64_t> first{"total", total};
            record.write(line, &first);
            line.write();
            return made;
        }
    }

    const std::vector<ProcessJit>& processJits()
    {
        static const auto jits = makeProcessJits();
        return jits;
    }
}
