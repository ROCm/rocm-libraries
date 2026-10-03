// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
                auto store = makeLibraryStore(JitLibrary::process(), jitCodeObjectVersion);
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
    }

    std::shared_ptr<const Jit> processJit(Status& why)
    {
        static const auto process = buildProcessJit();
        why                       = process.second;
        return process.first;
    }
}
