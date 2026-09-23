// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <amd_comgr/amd_comgr.h>

namespace rocRoller
{
    namespace Serialization
    {
        namespace ELFDetail
        {
            /**
             * Parses ELF as amd_comgr_metadata_node_t into a T.
             */
            template <typename T>
            T fromELFData(amd_comgr_metadata_node_t metadata);
        } // namespace ELFDetail
    } // namespace Serialization
} // namespace rocRoller

#include <rocRoller/Serialization/ELF_impl.hpp>
