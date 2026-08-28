/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblaslt-auxiliary.h"
#include <Tensile/Serialization.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/msgpack/Loading.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/MessagePack.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <msgpack.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hipblaslt_ext
{

    class SoftmaxProblem;
    class SoftmaxSolution;

    class SoftmaxSolution : public TensileLite::Solution
    {
    public:
        friend struct TensileLite::Serialization::
            MappingTraits<SoftmaxSolution, TensileLite::Serialization::MessagePackInput>;

        using Problem = SoftmaxProblem;
        std::string name() const override
        {
            return kernelName;
        }

        std::string description() const override
        {
            std::stringstream ss;
            ss << "Softmax, (Datatype, tileM, tileN) = "
               << "(" << TensileLite::ToString(datatype) << ", " << tileM << ", " << tileN << ")";
            return ss.str();
        }

        bool isFallbackForHW(TensileLite::Hardware const&) const override
        {
            return false;
        }

        std::uint32_t getTileM() const
        {
            return tileM;
        }

        std::uint32_t getTileN() const
        {
            return tileN;
        }

        std::uint32_t getNumWorkitems() const
        {
            return numWorkitems;
        }

        std::string getCodeObjectPath() const
        {
            return coPath;
        }

        rocisa::DataType getDatatype() const
        {
            return datatype;
        }

    private:
        std::size_t      tileM{};
        std::size_t      tileN{};
        std::size_t      numWorkitems{};
        std::string      coPath;
        std::string      kernelName;
        rocisa::DataType datatype;
    };

} // namespace hipblaslt
template <typename IO>
struct TensileLite::Serialization::MappingTraits<hipblaslt_ext::SoftmaxSolution, IO>
{
    using iot = IOTraits<IO>;
    static void mapping(IO& io, hipblaslt_ext::SoftmaxSolution& s)
    {
        iot::mapRequired(io, "func_name", s.kernelName);
        std::string datatypeStr;
        iot::mapRequired(io, "io_type", datatypeStr);

        if(datatypeStr == "S")
        {
            s.datatype = rocisa::DataType::Float;
        }
        else
        {
            throw std::runtime_error("Invalid datatype in ext op library");
        }

        iot::mapRequired(io, "num_rows", s.tileM);
        iot::mapRequired(io, "num_cols", s.tileN);
        iot::mapRequired(io, "num_workitems", s.numWorkitems);
        iot::mapRequired(io, "co_path", s.coPath);
    }

    const static bool flow = false;
};

namespace hipblaslt_ext
{

    class SoftmaxProblem : public TensileLite::Problem
    {
    public:
        using Solution = SoftmaxSolution;
        SoftmaxProblem(uint32_t m, uint32_t n, rocisa::DataType datatype)
            : m(m)
            , n(n)
            , datatype(datatype)
        {
        }

        ~SoftmaxProblem() override {}

        std::string description() const override
        {
            std::stringstream ss;
            ss << "Softmax Problem(" << TensileLite::ToString(datatype) << ", " << m << ", " << n
               << ")";
            return ss.str();
        }

        std::uint32_t getM() const
        {
            return m;
        }

        std::uint32_t getN() const
        {
            return n;
        }

    private:
        std::uint32_t    m{};
        std::uint32_t    n{};
        rocisa::DataType datatype{rocisa::DataType::Float};
    };

    struct ExtOpLibrary
    {
        virtual ~ExtOpLibrary()                 = default;
        virtual std::string type() const        = 0;
        virtual std::string description() const = 0;

        template <typename T>
        T& as()
        {
            return dynamic_cast<T&>(*this);
        }

        template <typename T>
        const T& as() const
        {
            return dynamic_cast<T&>(*this);
        }
    };

    class SoftmaxSolutionLibrary : public ExtOpLibrary
    {
    public:
        static constexpr char opName[] = "Softmax";

        ~SoftmaxSolutionLibrary() override {}
        void addSolution(SoftmaxSolution& sol)
        {
            solutions.push_back(std::make_shared<SoftmaxSolution>(sol));
        }

        std::string type() const override
        {
            return "SoftmaxSolutionLibrary";
        }

        std::string description() const override
        {
            return "SoftmaxSolutionLibrary";
        }

        std::shared_ptr<SoftmaxSolution> findBestSolution(const SoftmaxProblem&        problem,
                                                          const TensileLite::Hardware& hardware,
                                                          double* fitness = nullptr) const
        {
            auto bestSolIter = std::lower_bound(
                begin(solutions), end(solutions), problem.getN(), [](const auto& it, auto v) {
                    return it->getTileN() < v;
                });

            return *bestSolIter;
        }

        void sortSolutions()
        {
            std::sort(begin(solutions), end(solutions), [](const auto& lhs, const auto& rhs) {
                return lhs->getTileN() < rhs->getTileN();
            });
        }

    private:
        TensileLite::SolutionVector<SoftmaxSolution> solutions;
    };

    class ExtOpMasterLibrary
    {
    public:
        using ExtOpLibraryPtr = std::unique_ptr<ExtOpLibrary>;
        explicit ExtOpMasterLibrary(const std::string& libPath)
            : libPath(libPath)
        {
            libDir = std::filesystem::path(this->libPath).parent_path().string();
            load(libPath);
        }

        const ExtOpLibraryPtr& getLibrary(const std::string& archName,
                                          const std::string& opName,
                                          const std::string& typeName) const
        {
            return libraries.at(archName).at(opName).at(typeName);
        }

        const std::string getLibraryPath() const
        {
            return libPath;
        }

        const std::string getLibraryFolder() const
        {
            return libDir;
        }

    private:
        bool load(const std::string& libPath)
        {
            msgpack::object_handle handle;

            if(!TensileLite::fileToMsgObject(libPath, handle))
            {
                throw std::runtime_error("Failed to load ext op library: " + libPath);
            }

            msgpack::object                                  root = handle.get();
            std::unordered_map<std::string, msgpack::object> objMap;
            TensileLite::Serialization::objectToMap(root, objMap);

            for(auto& archObj : objMap)
            {
                libraries.emplace(archObj.first,
                                  std::map<std::string, std::map<std::string, ExtOpLibraryPtr>>());

                std::unordered_map<std::string, msgpack::object> opMap;
                TensileLite::Serialization::objectToMap(archObj.second, opMap);

                for(auto& opObj : opMap)
                {
                    libraries.at(archObj.first)
                        .emplace(opObj.first, std::map<std::string, ExtOpLibraryPtr>());

                    std::unordered_map<std::string, msgpack::object> typeMap;
                    TensileLite::Serialization::objectToMap(opObj.second, typeMap);

                    for(auto& typeLib : typeMap)
                    {
                        auto& rawKernels = typeLib.second;

                        if(rawKernels.type != msgpack::type::ARRAY)
                        {
                            throw std::runtime_error("Invalid ext op lib format");
                        }

                        const auto numKernels = rawKernels.via.array.size;

                        if(opObj.first == "Softmax")
                        {
                            libraries.at(archObj.first)
                                .at(opObj.first)
                                .emplace(typeLib.first, std::make_unique<SoftmaxSolutionLibrary>());
                            auto& lib = libraries.at(archObj.first)
                                            .at(opObj.first)
                                            .at(typeLib.first)
                                            ->as<SoftmaxSolutionLibrary>();

                            for(uint32_t i = 0; i < numKernels; ++i)
                            {
                                auto&           rawKernel = rawKernels.via.array.ptr[i];
                                SoftmaxSolution solution;
                                TensileLite::Serialization::MessagePackInput msgInput(rawKernel);
                                TensileLite::Serialization::MappingTraits<
                                    SoftmaxSolution,
                                    TensileLite::Serialization::MessagePackInput>::
                                    mapping(msgInput, solution);

                                lib.addSolution(solution);
                            }

                            lib.sortSolutions();
                        }
                    }
                }
            }

            return true;
        }

    private:
        std::map<std::string, std::map<std::string, std::map<std::string, ExtOpLibraryPtr>>>
                    libraries;
        std::string libPath;
        std::string libDir;
    };

} // namespace hipblaslt_ext
