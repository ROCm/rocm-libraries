// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::serialization
{

/// True when a plan whose kernel has source kind @p kind can be saved.
///
/// Saving accepts only kpack kernels. `embedded_source` kernels have no recorded argument
/// signature, and their code bytes are not exposed. `hsaco_file` and `rocke_builder`
/// kernels have no adapter that loads them. The payload format accepts any source kind.
/// Decide saveability by source kind only through this function.
constexpr bool isSerializableSourceKind(hipdnn_plugin_sdk::ingestor::KernelSourceKind kind)
{
    return kind == hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::serialization

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
