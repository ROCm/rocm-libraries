// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#ifdef HIPDNN_ENGINE_ASM_SDPA

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestAsmSdpaCodeObjects.cpp
 * @brief Every code object the ASM descriptors name exists in the kernel tree.
 *
 * A `hsaco_file` kernel names a path relative to the tree root, and nothing checks it
 * until a dispatch loads it -- which happens only on the architecture that pack targets,
 * after matching, past the point applicability promised the graph. A path that is merely
 * wrong therefore looks exactly like a working engine everywhere except one machine.
 *
 * That is not hypothetical: the backward companion tables shipped bare filenames against
 * an arch-nested tree, and every backward dispatch on MI300X died in prepare() with
 * `file not found` while every test on every other machine stayed green.
 *
 * Resolved against the source tree rather than an install or build prefix, so this holds
 * wherever the provider is checked out and needs no environment.
 */
namespace
{

/// The in-tree kernel root, from this file's own location. ASM_KERNEL_SOURCE_DIR is set
/// by CMake next to the AITER_ASM_DIR the engine compiles in.
std::filesystem::path kernelRoot()
{
    return std::filesystem::path{ASM_KERNEL_SOURCE_DIR};
}

} // namespace

TEST(TestAsmSdpaCodeObjects, EveryDescriptorKernelNamesAFileThatExists)
{
    using namespace hipdnn_plugin_sdk::ingestor;

    const auto root = kernelRoot();
    ASSERT_TRUE(std::filesystem::is_directory(root))
        << "kernel source tree not found at " << root
        << "; this test resolves paths against the checkout, so that is a wiring error";

    const auto& sets = hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets();
    ASSERT_FALSE(sets.empty()) << "no descriptor sets discovered, so nothing was asserted";

    size_t checked = 0;
    for(const auto& set : sets)
    {
        for(const auto& pack : set.packs)
        {
            for(const auto& kernel : pack.kernels)
            {
                if(kernel.source.kind != KernelSourceKind::HSACO_FILE)
                {
                    continue;
                }
                ++checked;

                // The gfx942 forward kernels are the one exception: their .co lives under
                // a per-die MI300/ or MI308/ subdirectory that the dispatch handler
                // splices in from a PCI chip-id probe, because two devices reporting the
                // same arch want different files. Accept either die.
                const auto direct = root / kernel.source.codeObjectFile;
                if(std::filesystem::exists(direct))
                {
                    continue;
                }

                const std::filesystem::path relative{kernel.source.codeObjectFile};
                const auto spliced = [&relative, &root](const char* die) {
                    return root / relative.parent_path() / die / relative.filename();
                };

                EXPECT_TRUE(std::filesystem::exists(spliced("MI300"))
                            && std::filesystem::exists(spliced("MI308")))
                    << "kernel '" << kernel.name << "' names " << kernel.source.codeObjectFile
                    << ", which is neither a file nor a per-die pair under " << root;
            }
        }
    }

    // A rename or a dropped pack would otherwise make every assertion above vacuous.
    EXPECT_GT(checked, 0U) << "no hsaco_file kernels were checked";
}

#endif // HIPDNN_ENGINE_ASM_SDPA
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
