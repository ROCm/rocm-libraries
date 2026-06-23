/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/msgpack/MessagePack.hpp>

#include <Tensile/msgpack/Loading.hpp>

#include <filesystem>
#include <fstream>

#include <zlib.h>

namespace TensileLite
{
    namespace Serialization
    {
        void objectToMap(const msgpack::object&                            object,
                         std::unordered_map<std::string, msgpack::object>& result)
        {
            if(object.type != msgpack::type::object_type::MAP)
                throw std::runtime_error(concatenate("Expected MAP, found ", object.type));

            for(uint32_t i = 0; i < object.via.map.size; i++)
            {
                auto& element = object.via.map.ptr[i];

                std::string key;
                switch(element.key.type)
                {
                case msgpack::type::object_type::STR:
                {
                    element.key.convert(key);
                    break;
                }
                case msgpack::type::object_type::POSITIVE_INTEGER:
                {
                    auto iKey = element.key.as<uint32_t>();
                    key       = std::to_string(iKey);
                    break;
                }
                default:
                    throw std::runtime_error("Unexpected map key type");
                }

                result[key] = std::move(element.val);
            }
        }
    }

    namespace
    {
    bool readCompressedMsgObject(std::string const&     gz_filename,
                                 msgpack::object_handle& result)
    {
        std::ifstream in(gz_filename, std::ios::binary | std::ios::ate);
        if(!in.is_open())
        {
            return false;
        }

        auto compressed_size = static_cast<size_t>(in.tellg());
        in.seekg(0);
        std::vector<uint8_t> compressed(compressed_size);
        in.read(reinterpret_cast<char*>(compressed.data()), compressed_size);
        if(!in)
        {
            return false;
        }

        z_stream strm{};
        if(inflateInit(&strm) != Z_OK)
            return false;

        strm.next_in  = compressed.data();
        strm.avail_in = static_cast<uInt>(compressed_size);

        // Stream inflate output directly into msgpack::unpacker in chunks,
        // mirroring the uncompressed path so decompress and parse overlap.
        msgpack::unpacker unp;
        constexpr size_t  buffer_size = 1 << 19;
        bool              finished_parsing = false;
        int               ret;

        do
        {
            unp.reserve_buffer(buffer_size);
            strm.next_out  = reinterpret_cast<uint8_t*>(unp.buffer());
            strm.avail_out = static_cast<uInt>(buffer_size);

            ret = inflate(&strm, Z_NO_FLUSH);
            if(ret != Z_OK && ret != Z_STREAM_END)
            {
                inflateEnd(&strm);
                return false;
            }

            size_t produced = buffer_size - strm.avail_out;
            unp.buffer_consumed(produced);
            finished_parsing = unp.next(result);
        } while(!finished_parsing && ret == Z_OK);

        inflateEnd(&strm);

        return finished_parsing;
    }
    } // anonymous namespace

    bool fileToMsgObject(std::string const& filename, msgpack::object_handle& result)
    {
        try
        {
            // Probe for a zlib-compressed variant first
            std::string gz_filename = filename + ".zlib";
            if(std::filesystem::exists(gz_filename))
            {
                if(readCompressedMsgObject(gz_filename, result))
                    return true;

                if(Debug::Instance().printDataInit())
                    std::cout << "Warning: failed to decompress " << gz_filename
                              << ", falling back to uncompressed" << std::endl;
            }

            // Fall back to uncompressed file
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            if(!in.is_open())
            {
                if(Debug::Instance().printDataInit())
                    std::cout << "Error loading " << filename << " (msgpack):\nFailed to open file"
                              << std::endl;

                return false;
            }

            msgpack::unpacker unp;
            bool              finished_parsing;
            constexpr size_t  buffer_size = 1 << 19;
            do
            {
                unp.reserve_buffer(buffer_size);
                in.read(unp.buffer(), buffer_size);
                unp.buffer_consumed(in.gcount());
                finished_parsing = unp.next(result);
            } while(!finished_parsing && !in.fail());

            if(!finished_parsing)
            {
                if(Debug::Instance().printDataInit())
                {
                    const char* const error_str
                        = in.eof() ? "Unexpected end of file" : "Read failure";
                    std::cout << "Error loading " << filename << " (msgpack):\n"
                              << error_str << std::endl;
                }

                return false;
            }
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading msgpack data:\n" << exc.what() << std::endl;

            return false;
        }
        return true;
    }

    std::map<int, std::string> MessagePackLoadLibraryMapping(std::string const& filename)
    {
        if(Debug::Instance().printDataInit())
            std::cout << "Loading library mapping from file: " << filename << std::endl;
        msgpack::object_handle result;
        if(!fileToMsgObject(filename, result))
            return {};

        std::map<int, std::string> libraryMapping;
        try
        {
            std::unordered_map<std::string, msgpack::object> objectMap;
            Serialization::objectToMap(result.get(), objectMap);

            for(auto const& pair : objectMap)
            {
                int         key = std::stoi(pair.first);
                std::string value;
                pair.second.convert(value);
                libraryMapping[key] = value;
            }
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading library mapping: " << exc.what() << std::endl;

            return {};
        }

        return libraryMapping;
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        MessagePackLoadLibraryFile(std::string const&                  filename,
                                   const std::vector<LazyLoadingInit>& preloaded)
    {
        msgpack::object_handle result;
        if(!fileToMsgObject(filename, result))
            return nullptr;

        // copy data from msgpack::object_handle into MasterSolutionLibrary
        try
        {
            std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

            LibraryIOContext<MySolution>    context{filename, preloaded, nullptr};
            Serialization::MessagePackInput min(result.get(), &context);

            Serialization::PointerMappingTraits<TensileLite::MasterContractionLibrary,
                                                Serialization::MessagePackInput>::mapping(min, rv);

            if(!min.error.empty())
            {
                std::ostringstream msg;
                msg << "Error loading msgpack data:\n";
                for(auto const& err : min.error)
                    msg << err << std::endl;

                throw std::runtime_error(msg.str());
            }

            return rv;
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading msgpack data:\n" << exc.what() << std::endl;

            return nullptr;
        }
    }

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        MessagePackLoadLibraryData(std::vector<uint8_t> const& data)
    {
        try
        {
            std::shared_ptr<MasterSolutionLibrary<MyProblem, MySolution>> rv;

            auto result = msgpack::unpack((const char*)data.data(), data.size());
            LibraryIOContext<MySolution>    context{std::string(""), {}, nullptr};
            Serialization::MessagePackInput min(result.get(), &context);

            Serialization::PointerMappingTraits<TensileLite::MasterContractionLibrary,
                                                Serialization::MessagePackInput>::mapping(min, rv);

            if(!min.error.empty())
            {
                std::ostringstream msg;
                msg << "Error loading msgpack data:" << std::endl;
                for(auto const& err : min.error)
                    msg << err << std::endl;

                throw std::runtime_error(msg.str());
            }

            return rv;
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading msgpack data:" << std::endl << exc.what() << std::endl;

            return nullptr;
        }
    }

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        MessagePackLoadLibraryFile<ContractionProblemGemm, ContractionSolution>(
            std::string const& filename, const std::vector<LazyLoadingInit>& preloaded);

    template std::shared_ptr<SolutionLibrary<ContractionProblemGemm, ContractionSolution>>
        MessagePackLoadLibraryData<ContractionProblemGemm, ContractionSolution>(
            std::vector<uint8_t> const& data);
}
