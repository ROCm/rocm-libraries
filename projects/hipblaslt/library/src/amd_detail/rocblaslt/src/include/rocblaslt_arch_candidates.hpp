// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

// Choosing the architecture stepping from the ASIC revision instead of from the
// name the runtime reported.
//
// Background. A gfx1250 A0 part is the same ISA as a B0 part but lacks some of
// its features, so #11777 gives it its own compiler target, `gfx1250-strict`,
// with its own device library, tuning logic and lazy-loading entry. Which of the
// two names ROCr reports for an A0 is decided by HSA_DISABLE_GFX12_STRICT:
// `"0"` reports gfx1250-strict, anything else -- including unset, which is
// today's default -- reports gfx1250. That is a poor thing to key a library off.
// The variable is proposed to change default (rocm-systems#11575), so the same
// binary would change behaviour on the same silicon; and it is process-global
// ROCr state, so a library cannot set it on its own behalf without changing what
// every other component in the process sees.
//
// hipDeviceProp_t::asicRevision is the alternative, and it is not a second
// opinion: it is HSA_AMD_AGENT_INFO_ASIC_REVISION, which ROCr fills from KFD
// topology `capability` bits 25:22 -- the very bits it consults before deciding
// whether to append the suffix. Reading it cannot disagree with ROCr about which
// silicon this is, only about what ROCr chose to call it.
//
// Header-only, and free of HIP types on purpose. The rule is pure string and
// integer work, and keeping it that way is what lets a unit test pin every case
// -- including steppings this machine does not have -- with no GPU, no driver
// and no device library. The HIP_VERSION guard around asicRevision lives with
// the caller that reads the property, not here.

// The suffix ROCr appends to a rev-0 agent's processor name. Spelled once so the
// strip and the re-derive below cannot drift apart.
inline constexpr const char* rocblaslt_arch_stepping_suffix = "-strict";

// The architecture without its feature suffix: everything from the first ':' on,
// which is how `:xnack-` and `:sramecc+` arrive. Every consumer in this tree cuts
// there before using the name.
inline std::string rocblaslt_arch_strip_features(const std::string& gcnArchName)
{
    return gcnArchName.substr(0, gcnArchName.find(':'));
}

// The architecture without a trailing stepping suffix. A name that does not carry
// one is returned unchanged, so this is safe to apply unconditionally.
inline std::string rocblaslt_arch_strip_stepping(const std::string& arch)
{
    const std::string suffix(rocblaslt_arch_stepping_suffix);

    if(arch.size() > suffix.size()
       && arch.compare(arch.size() - suffix.size(), suffix.size(), suffix) == 0)
        return arch.substr(0, arch.size() - suffix.size());

    return arch;
}

// The architecture with both decorations removed.
//
// The order is the whole trick. One A0 has two spellings -- `gfx1250` and
// `gfx1250-strict`, and either may arrive with `:xnack-` attached -- and
// stripping features first, stepping second, collapses both onto one base.
// Reversing it leaves the feature suffix attached during the stepping test, which
// then never matches, and the name comes back out as `gfx1250-strict-strict`.
inline std::string rocblaslt_arch_base_name(const std::string& gcnArchName)
{
    return rocblaslt_arch_strip_stepping(rocblaslt_arch_strip_features(gcnArchName));
}

// The architecture names that may serve a device, best first.
//
// Two candidates rather than one answer, because the preferred name is not always
// built. During the transition ROCm ships gfx1250-only packages, and those must
// keep working on an A0 that the runtime reports as gfx1250; committing to
// `gfx1250-strict` there would send the library-path lookup, the mapping
// filename and the code-object filter at a directory the build never produced,
// turning today's shipping configuration into a hard failure. So the caller takes
// the first candidate it can actually serve.
//
// The last entry is always the base architecture, so the list is never empty and
// the caller's fallback always terminates.
//
// Not restricted to gfx1250: ROCr's own gate is on the revision alone, so an
// architecture that gains a stepping variant later is picked up without a second
// edit here. Where no such variant exists the extra candidate simply never
// resolves, at the cost of one failed lookup during initialisation.
inline std::vector<std::string> rocblaslt_arch_name_candidates(const std::string& gcnArchName,
                                                               int                asicRevision)
{
    const std::string base = rocblaslt_arch_base_name(gcnArchName);

    // Revision 0 is the first stepping of a part -- the only one ROCr renames.
    if(asicRevision == 0)
        return {base + rocblaslt_arch_stepping_suffix, base};

    return {base};
}
