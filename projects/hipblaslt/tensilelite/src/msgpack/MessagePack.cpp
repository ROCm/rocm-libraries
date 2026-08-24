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

#include <charconv>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>
#include <vector>

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
    // Upper bound on bytes inflated from a single .dat.zlib, so a tiny but
    // highly compressible file cannot exhaust memory (a "zip bomb"). Fixed at
    // 16 GiB. TENSILE_MAX_DECOMPRESSED_BYTES may only *lower* the bound (for
    // tests or stricter deployments); it can never raise it, so it cannot be
    // used to weaken the guard.
    size_t maxDecompressedBytes()
    {
        constexpr size_t defaultMax = size_t(16) << 30;
        if(const char* env = std::getenv("TENSILE_MAX_DECOMPRESSED_BYTES"))
        {
            char*              end    = nullptr;
            unsigned long long parsed = std::strtoull(env, &end, 0);
            if(end != env && parsed > 0)
            {
                size_t requested = static_cast<size_t>(parsed);
                return requested < defaultMax ? requested : defaultMax;
            }
        }
        return defaultMax;
    }

    bool readCompressedMsgObject(std::string const&     gz_filename,
                                 msgpack::object_handle& result)
    {
        try
        {
            std::ifstream in(gz_filename, std::ios::binary | std::ios::ate);
            if(!in.is_open())
            {
                return false;
            }

            auto pos = in.tellg();
            if(pos < 0)
                return false;

            auto compressed_size = static_cast<size_t>(pos);
            if(compressed_size > std::numeric_limits<uInt>::max())
                return false;

            in.seekg(0);
            std::vector<uint8_t> compressed(compressed_size);
            in.read(reinterpret_cast<char*>(compressed.data()), compressed_size);
            if(!in)
            {
                return false;
            }

            z_stream strm{};
            if(inflateInit(&strm) != Z_OK)
            {
                return false;
            }

            struct InflateGuard
            {
                z_stream* stream;
                ~InflateGuard()
                {
                    inflateEnd(stream);
                }
            } guard{&strm};

            strm.next_in  = compressed.data();
            strm.avail_in = static_cast<uInt>(compressed_size);

            // Stream inflate output directly into msgpack::unpacker in chunks,
            // mirroring the uncompressed path so decompress and parse overlap.
            msgpack::unpacker unp;
            constexpr size_t  buffer_size = 1 << 19;
            bool              finished_parsing = false;
            bool              inflate_complete = false;
            int               ret;
            size_t            total_produced   = 0;
            const size_t      max_produced     = maxDecompressedBytes();

            do
            {
                unp.reserve_buffer(buffer_size);
                strm.next_out  = reinterpret_cast<uint8_t*>(unp.buffer());
                strm.avail_out = static_cast<uInt>(buffer_size);

                ret = inflate(&strm, Z_NO_FLUSH);
                if(ret != Z_OK && ret != Z_STREAM_END)
                    return false;

                if(ret == Z_STREAM_END)
                {
                    // A single zlib stream is expected; leftover input means
                    // trailing bytes after the stream (corruption/tampering).
                    if(strm.avail_in != 0)
                        return false;
                    inflate_complete = true;
                }

                size_t produced = buffer_size - strm.avail_out;
                total_produced += produced;
                if(total_produced > max_produced)
                    return false;
                unp.buffer_consumed(produced);
                if(!finished_parsing)
                    finished_parsing = unp.next(result);
            } while(ret == Z_OK);

            return inflate_complete && finished_parsing && unp.nonparsed_size() == 0;
        }
        catch(std::runtime_error const&)
        {
            return false;
        }
    }

    constexpr char   zlib_suffix[]    = ".zlib";
    constexpr size_t zlib_suffix_size = sizeof(zlib_suffix) - 1;

    bool hasZlibSuffix(std::string const& filename)
    {
        return filename.size() >= zlib_suffix_size
               && filename.compare(
                      filename.size() - zlib_suffix_size, zlib_suffix_size, zlib_suffix)
                      == 0;
    }

    std::string logicalLibraryName(std::string const& filename)
    {
        if(hasZlibSuffix(filename))
            return filename.substr(0, filename.size() - zlib_suffix_size);
        return filename;
    }
    } // anonymous namespace

    bool fileToMsgObject(std::string const& filename, msgpack::object_handle& result)
    {
        try
        {
            std::string base_filename = filename;
            if(hasZlibSuffix(base_filename))
            {
                if(readCompressedMsgObject(base_filename, result))
                    return true;

                base_filename.resize(base_filename.size() - zlib_suffix_size);
            }

            // Probe for a zlib-compressed variant first
            std::string gz_filename = base_filename + ".zlib";
            if(std::filesystem::exists(gz_filename))
            {
                if(readCompressedMsgObject(gz_filename, result))
                    return true;

                if(Debug::Instance().printDataInit())
                    std::cout << "Warning: failed to decompress " << gz_filename
                              << ", falling back to uncompressed" << std::endl;
            }

            // Fall back to uncompressed file
            std::ifstream in(base_filename, std::ios::in | std::ios::binary);
            if(!in.is_open())
            {
                if(Debug::Instance().printDataInit())
                    std::cout << "Error loading " << base_filename
                              << " (msgpack):\nFailed to open file" << std::endl;

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
                    std::cout << "Error loading " << base_filename << " (msgpack):\n"
                              << error_str << std::endl;
                }

                return false;
            }

            if(unp.nonparsed_size() != 0
               || in.peek() != std::ifstream::traits_type::eof())
            {
                if(Debug::Instance().printDataInit())
                    std::cout << "Error loading " << base_filename << " (msgpack):\n"
                              << "Trailing bytes after object" << std::endl;

                return false;
            }
        }
        catch(std::exception const& exc)
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
                int         key   = 0;
                char const* first = pair.first.data();
                char const* last  = first + pair.first.size();
                auto const  conv  = std::from_chars(first, last, key);
                if(conv.ec != std::errc{} || conv.ptr != last)
                {
                    if(Debug::Instance().printDataInit())
                        std::cout << "Error loading library mapping: invalid key \""
                                  << pair.first << "\"" << std::endl;
                    return {};
                }
                std::string value;
                pair.second.convert(value);
                libraryMapping[key] = value;
            }
        }
        catch(std::exception const& exc)
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

            LibraryIOContext<MySolution>    context{logicalLibraryName(filename), preloaded, nullptr};
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
