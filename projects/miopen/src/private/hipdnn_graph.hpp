// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The part of the hipDNN forwarding that does not depend on the operation. The
// first half is what the rest of the wrapper calls: probing the backend,
// releasing a handle's hipDNN state and prefixing forwarded error messages. The
// second half is what each operation's file (hipdnn_conv.cpp) uses: it
// describes its problem as a PlanKey and supplies the function that builds the
// graph for it, and RunCachedGraph does the rest.
//
// Compiled only into the public wrapper library; never installed.
#pragma once

#include <miopen/miopen.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// Declared rather than included so that wrapper.cpp and routing.cpp, which only
// need the first half of this header, do not pull in the hipDNN frontend.
namespace hipdnn_frontend {
namespace graph {
class Graph;
} // namespace graph
} // namespace hipdnn_frontend

namespace miopen {
namespace wrapper {
namespace hipdnn {

// True when hipDNN can serve a forwarded call. Loads the backend on first use, so
// a process that forwards nothing should never reach it. The answer is cached for
// the process, and a false one also prints one line to stderr saying why.
bool IsAvailable();

// Drops whatever hipDNN state was created for this MIOpen handle. Called from
// the miopenDestroy stub on both routes, because a handle can be destroyed after
// MIOPEN_DISABLE_HIPDNN_FOR took its entry points back off the hipDNN path.
// Like miopenDestroy, it must not run while another call is using the handle:
// calls use the handle's state without holding the map lock.
void ReleaseHandle(miopenHandle_t handle);

// Replacement text for miopenGetErrorString when the last forwarded call on this
// thread failed with `status`, or null when it did not. The result has
// thread-local storage duration, matching what miopenGetErrorString promises its
// callers.
//
// This exists so a forwarded failure is distinguishable from the same status
// raised by MIOpen itself, without adding a public symbol to do it.
const char* PrefixedErrorString(miopenStatus_t status, const char* nativeMessage);

// One value per kind of graph the wrapper builds. Kept in one list so that the
// values stay distinct as operations are added: it is the first element of
// every PlanKey.
enum class GraphKind : int64_t
{
    ConvFprop,
    ConvDgrad,
    ConvWgrad,
    ConvBiasActivation,
};

// Identifies one cached graph. `problem` is everything that tells two problems
// apart, flattened to integers by the operation. It starts with the GraphKind,
// and every variable-length list in it is preceded by its length, so two
// different problems never flatten to the same sequence.
struct PlanKey
{
    miopenHandle_t handle = nullptr;
    std::vector<int64_t> problem;

    bool operator==(const PlanKey& other) const
    {
        return handle == other.handle && problem == other.problem;
    }
};

// Records a forwarded failure for miopenGetErrorString to report, and returns
// `status`.
miopenStatus_t RecordFailure(miopenStatus_t status, std::string message);

using PopulateGraphFn = std::function<bool(hipdnn_frontend::graph::Graph&)>;
using VariantPack     = std::unordered_map<int64_t, void*>;

// Runs the graph for `key` on `handle`'s stream. On a cache miss, `populate`
// fills in a new graph, which is then built and cached. Failures are recorded
// with RecordFailure.
miopenStatus_t RunCachedGraph(miopenHandle_t handle,
                              const PlanKey& key,
                              const PopulateGraphFn& populate,
                              VariantPack& variantPack);

} // namespace hipdnn
} // namespace wrapper
} // namespace miopen
