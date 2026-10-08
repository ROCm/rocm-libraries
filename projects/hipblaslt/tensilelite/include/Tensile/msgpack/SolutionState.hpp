// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <msgpack.hpp>

#include <Tensile/ContractionSolution.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
     * Builds the library record of a TensileLite kernel from its solution
     * state, the map TensileLite writes as the custom.config of the kernel's
     * metadata, as Solution.FromOriginalState in Tensile/Contractions.py does.
     *
     * The state may be read from a code object's metadata note. The assembler
     * types each scalar of the note from its YAML text, so an empty string
     * arrives as 0.0 and a null as the string "null"; both are read as the
     * values the state holds.
     *
     * As when loading a library, the WorkgroupMappingXCCCheck predicate reads
     * the current HIP device's compute unit count.
     *
     * Throws std::runtime_error naming the key when a key the record needs is
     * missing or holds a value of the wrong kind.
     */
    TENSILELITEHOST_EXPORT std::shared_ptr<ContractionSolution>
        solutionFromState(msgpack::object const& state);
}
