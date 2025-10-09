/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <string>
#include <rocRoller/Serialization/ELF.hpp>
#include <rocRoller/Serialization/comgr/comgr.hpp>
#include <amd_comgr/amd_comgr.h>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename T>
        T fromELF(std::string const& elf)
        {
            T rv;
            
            amd_comgr_data_t elfData;
            amd_comgr_metadata_node_t metadataNode;
            
            auto status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &elfData);
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to create COMGR data object");
            
            status = amd_comgr_set_data(elfData, elf.size(), elf.data());
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to set ELF data");
            
            status = amd_comgr_get_data_metadata(elfData, &metadataNode);
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to extract metadata from ELF");
            
            Serialization::ComgrNodeInput comgrNodeInput(metadataNode, nullptr);
            comgrNodeInput.input(metadataNode, rv);
            
            amd_comgr_destroy_metadata(metadataNode);
            amd_comgr_release_data(elfData);
            
            return rv;
        }
    } // namespace Serialization
} // namespace rocRoller