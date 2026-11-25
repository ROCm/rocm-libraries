// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-only: cudnn_frontend_EngineConfigGenerator.h alone and first in its
// TU, as a consumer outside the umbrella includes it.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_EngineConfigGenerator.h>

namespace hipdnn_shim_test
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// External linkage forces the generator and filter to be emitted, not just
// parsed.
// NOLINTNEXTLINE(misc-use-internal-linkage)
fe::executionPlans_t generateWithEngineConfigGeneratorAlone(const fe::GeneratorSource& source,
                                                            fe::OperationGraph& graph,
                                                            const fe::Predicate& predicate,
                                                            fe::executionPlans_t& plans)
{
    fe::EngineConfigGenerator generator(1, &source);
    static_cast<void>(generator.generate_engine_config(graph));
    return fe::filter(predicate, plans);
}

} // namespace hipdnn_shim_test
