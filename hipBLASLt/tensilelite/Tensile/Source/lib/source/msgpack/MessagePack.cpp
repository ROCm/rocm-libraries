/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <fstream>

namespace TensileLite
{
    namespace Serialization
    {
        void objectToMap(msgpack::object&                                  object,
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

    template <typename MyProblem, typename MySolution>
    std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>
        MessagePackLoadLibraryFile(std::string const&                  filename,
                                   const std::vector<LazyLoadingInit>& preloaded)
    {
        // parse file into a msgpack::object_handle
        msgpack::object_handle result;
        try
        {
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            if(!in.is_open())
            {
                if(Debug::Instance().printDataInit())
                    std::cout << "Error loading " << filename << " (msgpack):\nFailed to open file"
                              << std::endl;

                return nullptr;
            }

            msgpack::unpacker unp;
            bool              finished_parsing;
            constexpr size_t  buffer_size = 1 << 19;
            do
            {
                unp.reserve_buffer(buffer_size);
                in.read(unp.buffer(), buffer_size);
                unp.buffer_consumed(in.gcount());
                finished_parsing = unp.next(result); // may throw msgpack::parse_error
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

                return nullptr;
            }
        }
        catch(std::runtime_error const& exc)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Error loading msgpack data:\n" << exc.what() << std::endl;

            return nullptr;
        }

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
