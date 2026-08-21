// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/FixedLinearArbiterLibrary.hpp>
#include <Tensile/Serialization/Base.hpp>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename MyProblem, typename MySolution, typename IO>
        struct MappingTraits<FixedLinearArbiterLibrary<MyProblem, MySolution>, IO>
        {
            using Library = FixedLinearArbiterLibrary<MyProblem, MySolution>;
            using iot     = IOTraits<IO>;

            static void mapping(IO& io, Library& lib)
            {
                iot::mapRequired(io, "model_id", lib.modelId);
                iot::mapRequired(io, "feature_schema", lib.featureSchema);
                iot::mapRequired(io, "weights", lib.weights);
                iot::mapRequired(io, "cu_count", lib.cuCount);
                iot::mapRequired(io, "g0_library", lib.g0Library);
                iot::mapRequired(io, "o3_library", lib.o3Library);

                if(!iot::outputting(io))
                {
                    if(lib.weights.size() != 22)
                        iot::setError(io, "FixedLinearArbiter requires exactly 22 weights.");
                    if(!lib.g0Library || !lib.o3Library)
                        iot::setError(io, "FixedLinearArbiter requires both child libraries.");
                    if(lib.cuCount <= 0.0)
                        iot::setError(io, "FixedLinearArbiter requires positive cu_count.");
                }
            }

            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace TensileLite
