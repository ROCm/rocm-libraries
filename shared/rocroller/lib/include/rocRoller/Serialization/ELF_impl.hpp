/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <fstream>
#include <string>
#include <vector>

#include <rocRoller/Serialization/ELF.hpp>
#include <rocRoller/Serialization/ELF_detail.hpp>
#include <rocRoller/Serialization/comgr/comgr.hpp>

#include <amd_comgr/amd_comgr.h>

namespace rocRoller
{
    namespace Serialization
    {
        namespace ELFDetail
        {
            template <typename T>
            T fromELFData(amd_comgr_metadata_node_t metadataNode)
            {
                T rv;

                Serialization::ComgrNodeInput comgrNodeInput(metadataNode, nullptr);
                comgrNodeInput.input(metadataNode, rv);
                amd_comgr_destroy_metadata(metadataNode);

                return rv;
            }
        } // namespace ELFDetail

        template <typename T>
        T fromELFFile(std::string const& filename)
        {
            amd_comgr_data_t          data;
            amd_comgr_metadata_node_t node;
            std::vector<char>         buffer;

            std::ifstream file(filename, std::ios::binary | std::ios::ate);
            if(!file.is_open())
            {
                throw std::runtime_error("Failed to open ELF file: " + filename);
            }

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            buffer.resize(size);
            if(!file.read(buffer.data(), size))
            {
                throw std::runtime_error("Failed to read ELF file: " + filename);
            }
            file.close();

            auto status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data);
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to create COMGR data object");

            status = amd_comgr_set_data(data, buffer.size(), buffer.data());
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to set ELF data");

            status = amd_comgr_set_data_name(data, NULL);
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to set ELF data name");

            status = amd_comgr_get_data_metadata(data, &node);
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get ELF metadata");

            try
            {
                return ELFDetail::fromELFData<T>(node);
            }
            catch(...)
            {
                std::cerr << "Error processing ELF file: " << filename << std::endl;
                amd_comgr_release_data(data);
                amd_comgr_destroy_metadata(node);
                throw;
            }
        }
    } // namespace Serialization
} // namespace rocRoller
