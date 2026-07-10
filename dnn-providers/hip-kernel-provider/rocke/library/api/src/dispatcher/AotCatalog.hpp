// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "dispatcher/AotInstance.hpp"

namespace rocke_client::dispatcher
{

// The set of AOT-built kernel instances the dispatcher can select from.
//
// PHASE 1 (this ticket): the production catalog is ALWAYS EMPTY. The rocKE AOT
// producer (PR #8866) currently emits loose build-tree HSACO + sidecar files
// only -- it has no install rules, no kpack packaging, and no runtime catalog.
// Until that lands, `loadDefault()` returns an empty catalog and the engine
// therefore declines every graph (a deliberate no-op). The selection logic is
// still real and fully exercised in unit tests via catalogs constructed from
// fixture instances.
class AotCatalog
{
public:
    AotCatalog() = default;
    explicit AotCatalog(std::vector<AotInstance> instances);

    // The production catalog source.
    //
    // TODO(kpack): once PR #8866's kpack packaging + install rules land, this must:
    //   1. resolve the installed artifact root (see defaultArtifactRoot());
    //   2. discover per-arch instance lists (kernels/<op>/<family>/<arch>/aot_list.json
    //      or the kpack index -- source format TBD by the kpack ticket);
    //   3. parse each instance (compile_spec + selection.batch + attribute_constraints)
    //      into AotInstance, mirroring rocke_client_aot.instance_schema semantics;
    //   4. (plan-construction, separate) index the matching sidecars by cache_key.
    // For now it logs the deferral and returns an empty catalog.
    static AotCatalog loadDefault();

    // Instances whose op and arch match, in stable (insertion) order. Returns
    // non-owning references borrowed from this catalog (valid for its lifetime,
    // which spans the owning engine) so the selection path copies no instances.
    std::vector<std::reference_wrapper<const AotInstance>>
        candidatesFor(const std::string& op, const std::string& arch) const;

    bool empty() const
    {
        return _instances.empty();
    }

    std::size_t size() const
    {
        return _instances.size();
    }

private:
    std::vector<AotInstance> _instances;
};

// The placeholder root under which installed AOT artifacts are expected to live.
//
// TODO(kpack): confirm with the kpack/packaging work where PR #8866 artifacts are
// installed (predicted: an install-prefix-relative share/ path resolved next to
// the loaded plugin, overridable by an env var for testing). This is a
// placeholder string only; nothing reads from it yet.
const char* defaultArtifactRoot();

} // namespace rocke_client::dispatcher
