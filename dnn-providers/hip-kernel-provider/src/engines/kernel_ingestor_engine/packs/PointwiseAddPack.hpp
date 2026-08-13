// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>

#include "core/Handle.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

/**
 * @file PointwiseAddPack.hpp
 * @brief The pointwise-add descriptor set, built in memory.
 *
 * Stands in for what a loader will produce from installed files: one engine (UED),
 * its metadata schema (KMD) and heuristic (UHD), two matchers (UMDs), one dispatch
 * descriptor (UDD), and one pack (KDP) binding them over three kernels (UKDs).
 *
 * ALMIOPEN-2401 deletes this file: a descriptor set becomes parsed data rather than
 * code. Nothing outside it depends on that distinction, since it returns the same
 * generic DescriptorSet a loader will.
 */

/// @brief Builds this pack's descriptor set.
hipdnn_plugin_sdk::ingestor::DescriptorSet buildPointwiseAddDescriptorSet();

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
