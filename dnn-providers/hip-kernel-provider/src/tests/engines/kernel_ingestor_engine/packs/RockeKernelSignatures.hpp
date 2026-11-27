// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

/**
 * @file RockeKernelSignatures.hpp
 * @brief The kernel argument lists the rocKE attention packs launch, as rocKE declares them.
 *
 * RockeKernelSignaturesTable.inc is generated at build time from rocKE's own launch
 * signatures by gen_rocke_kernel_signatures.py (kernel_ingestor_engine/CMakeLists.txt) and is
 * never committed, so the provider's tests compare each pack's C++ argument list with what
 * rocKE declares rather than with a second hand-written copy, and a rename in rocKE fails here
 * until the pack follows. The generator needs only rocKE's Python, so the table exists whether
 * or not the rocKE engine is built.
 *
 * Keys are `<arch>.<family>[.<form>]`: `gfx942.unified_2d`, `gfx950.unified_2d.lse`,
 * `gfx950.unified_3d.segment`, `gfx942.dense.persistent.qb_major.lse`, and so on (the
 * generator lists them). Every offset is zero: the packs compare kind, size and name, never
 * the offset.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

struct RockeKernelSignature
{
    const char* key;
    std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> arguments;
};

/// Every argument list in the generated table.
inline const std::vector<RockeKernelSignature>& rockeKernelSignatures()
{
    static const std::vector<RockeKernelSignature> s_signatures{
#include "RockeKernelSignaturesTable.inc"
    };
    return s_signatures;
}

/// The argument list rocKE declares for @p key. Throws std::out_of_range, naming the key, when
/// the table has no such list (a key the generator stopped producing, or a misspelling).
inline std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>
    rockeKernelSignature(std::string_view key)
{
    for(const auto& signature : rockeKernelSignatures())
    {
        if(key == signature.key)
        {
            return signature.arguments;
        }
    }
    throw std::out_of_range("RockeKernelSignaturesTable.inc has no kernel argument list '"
                            + std::string(key) + "'");
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
