// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <msgpack.hpp>

#include <Tensile/ContractionSolution_fwd.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
     * Builds the library record of one fully derived solution state, the map
     * Solution.FromOriginalState receives after defaults and derived parameters
     * are filled in.
     *
     * The input is not a raw custom.config delta and not an unexpanded logic
     * entry. It must already contain ProblemType.TotalIndices,
     * IndicesSummation, IndicesFree, IndicesBatch, IndexAssignmentsA,
     * IndexAssignmentsB, and the other derived fields the record reads. A
     * metadata note may carry that derived map. A caller that passes an
     * unexpanded map gets a missing-key error.
     *
     * The assembler types each scalar of a metadata note from its YAML text,
     * so an empty string arrives as 0.0 and a null as the string "null"; both
     * are read as the values the state holds.
     *
     * The CustomKernel block is read as the library reads a record's
     * customKernel, with what that reader accepts, ignores and defaults.
     *
     * WorkgroupMappingXCCCheck stores only its value. Its compute-unit count
     * stays 0 until the predicate is evaluated, which is when the device is
     * read. Building the record does not query a GPU.
     *
     * Throws std::runtime_error naming the key when a key the record needs is
     * missing or holds a value of the wrong kind.
     *
     * It also throws for states Solution.FromOriginalState accepts, because
     * the library cannot load or launch them:
     *  - with PersistentLoopArgsVersion 2 (StreamK with a cluster): a
     *    ClusterDim that is not x by y by 1 with x and y powers of two up to 8
     *    and 2 to 16 CUs in all, StreamKAtomic or PrefetchAcrossPersistent set,
     *    OutputAmaxD, a bias gradient, or a hand-written custom kernel
     *    (ContractionSolution::validatePersistentLoopArgs);
     *  - a custom kernel with a PersistentGrid argument at
     *    PersistentLoopArgsVersion 0, or at version 1 without ItersPerTile and
     *    PersistentGrid as adjacent unpadded uint32 arguments, with a legacy
     *    scheduling or workspace argument, or with a partial workspace;
     *  - a StreamKAtomic other than 0 or 1;
     *  - a legacy StreamK or StreamKForceDPOnly selector, which Python
     *    translates (a PersistentLoop selector, which Python refuses as
     *    well, also throws);
     *  - an Int4EncodingA other than Signed or UnsignedBias8;
     *  - no ISA, which Python takes from the device for an assembly kernel, or
     *    an ISA naming a processor TensileLite does not know;
     *  - a KernelLanguage that TensileLite has no name for;
     *  - a CustomKernelSizePredicate tag other than SizeGreaterThan,
     *    SizeLessThan and SizeEqual;
     *  - an AssertSize* or AssertStride* entry whose index or value is not an
     *    integer >= 0 (a value of -1 is skipped, as in Python);
     *  - an Ideals key that is not an integer, or a decimal string in int
     *    range;
     *  - a WorkGroup, ThreadTile or ClusterDim of more than 3 entries, a
     *    MatrixInstruction of more than 4, or an MIWaveGroup of more than 2.
     */
    TENSILELITEHOST_EXPORT std::shared_ptr<ContractionSolution>
                           solutionFromState(msgpack::object const& state);
}
