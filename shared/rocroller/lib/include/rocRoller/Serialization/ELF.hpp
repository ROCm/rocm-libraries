// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <istream>
#include <string>

#include <amd_comgr/amd_comgr.h>

namespace rocRoller
{
    namespace Serialization
    {
        /**
         * Note that the functions declared here will assume that you have included the deserialization headers for any
         * type(s) you want to deserialize.
         */

        /**
         * Parses ELF from a filepath into a T.
         */
        template <typename T>
        T fromELFFile(std::string const& filename);

        /**
         * Parses ELF from a file into a T.
         */
        template <typename T>
        T fromELF(std::string const& filename);
    }
}

#include <rocRoller/Serialization/ELF_impl.hpp>
