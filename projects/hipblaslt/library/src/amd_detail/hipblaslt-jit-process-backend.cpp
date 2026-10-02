// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-debug.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-loader.hpp"
#include <exception>
#include <utility>

namespace hipblaslt_jit
{
    namespace
    {
        std::pair<std::shared_ptr<const Jit>, Status> buildProcessJit()
        {
            try
            {
                ProcessBackend made;
                auto           status = makeDefaultProcessBackend(made);
                if(!status.ok())
                {
                    status.stage = Stage::Configure;
                    return {nullptr, std::move(status)};
                }
                debug::Phase storePhase("store");
                auto         store = makeLibraryStore(
                    JitLibrary::process(), made.backend->info(), jitCodeObjectVersion);
                storePhase.stop();
                debug::Phase components("components");
                return {std::make_shared<const Jit>(Jit::Components{std::move(made.backend),
                                                                    std::move(made.predictor),
                                                                    std::move(made.knowledge),
                                                                    makeComgrBuilder(),
                                                                    makeTensileLoader(),
                                                                    std::move(store)}),
                        {}};
            }
            catch(const std::exception& e)
            {
                return {nullptr, {Status::Code::Failed, Stage::Configure, e.what()}};
            }
        }

        // With timing, a setup line reports the first processJit call.
        std::pair<std::shared_ptr<const Jit>, Status> makeProcessJit()
        {
            if(!debug::on(debug::Timing))
                return buildProcessJit();
            debug::Record record;
            auto          made = [&] {
                debug::Scope scope(&record);
                return buildProcessJit();
            }();
            debug::Line line(debug::Timing, "setup");
            line.add("status", made.second.ok() ? "ok" : "failed");
            if(!made.second.ok())
                line.add("message", made.second.message);
            uint64_t total = 0;
            for(const char* phase : {"tool_check", "backend", "store", "components"})
                total += record.nanoseconds(phase);
            const std::pair<const char*, uint64_t> first{"total", total};
            record.write(line, &first);
            line.write();
            return made;
        }
    }

    std::shared_ptr<const Jit> processJit(Status& why)
    {
        static const auto process = makeProcessJit();
        why                       = process.second;
        return process.first;
    }
}
