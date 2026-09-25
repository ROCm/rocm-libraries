/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "UserDrivenTuningParser.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <utility>

#ifndef TO_STR2
#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#endif

namespace TensileLite
{
    namespace
    {
        const char* const kGitVersionHeader = "Git Version: ";

        // The build this library was compiled from, in the form hipblaslt-bench
        // writes on a tuning file's first line.
        const std::string& currentBuildStamp()
        {
#ifdef HIPBLASLT_VERSION_TWEAK
            static const std::string stamp = TO_STR(HIPBLASLT_VERSION_TWEAK);
#else
            static const std::string stamp;
#endif
            return stamp;
        }

        std::string trimmed(const std::string& s)
        {
            const auto first = s.find_first_not_of(" \t\n\r\f\v");
            if(first == std::string::npos)
                return {};
            const auto last = s.find_last_not_of(" \t\n\r\f\v");
            return s.substr(first, last - first + 1);
        }

        std::vector<std::string> splitCsv(const std::string& line)
        {
            std::vector<std::string> cells;
            std::stringstream        split(line);
            std::string              cell;
            while(std::getline(split, cell, ','))
                cells.push_back(trimmed(cell));
            return cells;
        }

        // A row header names at least the columns an entry is keyed and resolved
        // on. Anything else in the file is ignored.
        bool isHeaderRow(const std::string& line)
        {
            const auto cells = splitCsv(line);
            return std::find(cells.begin(), cells.end(), "transA") != cells.end()
                   && std::find(cells.begin(), cells.end(), "solution_index") != cells.end();
        }

        // Pairs cells by column name rather than by position, so the columns
        // hipblaslt-bench writes around the ones used here can come in any order.
        std::map<std::string, std::string> zipRow(const std::vector<std::string>& names,
                                                  const std::vector<std::string>& values)
        {
            std::map<std::string, std::string> row;
            for(size_t i = 0; i < names.size() && i < values.size(); i++)
                row.emplace(names[i], values[i]);
            return row;
        }

        std::optional<std::pair<ProblemOverride, TunedEntry>>
            problemFromRow(const std::map<std::string, std::string>& row)
        {
            auto field = [&](const char* name) -> const std::string* {
                auto it = row.find(name);
                return it == row.end() ? nullptr : &it->second;
            };

            for(const char* name : {"transA",
                                    "transB",
                                    "batch_count",
                                    "m",
                                    "n",
                                    "k",
                                    "a_type",
                                    "b_type",
                                    "c_type",
                                    "compute_type",
                                    "solution_index"})
                if(!field(name))
                    return std::nullopt;

            const bool transA = (*field("transA") != "N");
            const bool transB = (*field("transB") != "N");

            size_t           m, n, b, k;
            rocisa::DataType inputTypeA    = rocisa::DataType::None;
            rocisa::DataType inputTypeB    = rocisa::DataType::None;
            rocisa::DataType outputType    = rocisa::DataType::None;
            rocisa::DataType computeType   = rocisa::DataType::None;
            int              solutionIndex = -1;

            try
            {
                b           = std::stol(*field("batch_count"));
                m           = std::stol(*field("m"));
                n           = std::stol(*field("n"));
                k           = std::stol(*field("k"));
                inputTypeA  = hipDataType_to_tensile_type(string_to_hip_datatype(*field("a_type")));
                inputTypeB  = hipDataType_to_tensile_type(string_to_hip_datatype(*field("b_type")));
                outputType  = hipDataType_to_tensile_type(string_to_hip_datatype(*field("c_type")));
                computeType = rocComputeType_to_tensile_type(
                    (rocblaslt_compute_type)string_to_hipblas_computetype(*field("compute_type")));
                solutionIndex = std::stoi(*field("solution_index"));
            }
            catch(std::invalid_argument const&)
            {
                return std::nullopt;
            }
            catch(std::out_of_range const&)
            {
                return std::nullopt;
            }

            if(inputTypeA == rocisa::DataType::None || inputTypeB == rocisa::DataType::None
               || outputType == rocisa::DataType::None || computeType == rocisa::DataType::None)
                return std::nullopt;

            // Index 0 is a real solution in the shipped logic; only a negative
            // index is meaningless.
            if(solutionIndex < 0)
                return std::nullopt;

            TunedEntry entry;
            entry.solutionIndex = solutionIndex;
            if(auto name = field("kernel_name"); name && !name->empty())
                entry.kernelName = *name;
            if(auto name = field("solution_name"); name && !name->empty())
                entry.solutionName = *name;

            return std::make_pair(
                ProblemOverride(
                    transA, transB, inputTypeA, inputTypeB, computeType, outputType, m, n, k, b),
                entry);
        }
    } // namespace

    void getContractionProblemsFromFile(const std::string& path)
    {
        OverrideMap&                m_override = OverrideMap::getMap();
        std::mutex&                 map_guard  = m_override.getLock();
        std::lock_guard<std::mutex> lock(map_guard);

        if(m_override.isLoaded(path))
            return;

        std::ifstream file_read(path);
        std::string   fileBuildStamp;
        std::string   line;
        std::string   pendingHeader;

        while(true)
        {
            if(!pendingHeader.empty())
            {
                line = std::move(pendingHeader);
                pendingHeader.clear();
            }
            else if(!std::getline(file_read, line))
            {
                break;
            }

            const std::string header = trimmed(line);
            if(header.empty())
                continue;

            if(fileBuildStamp.empty())
            {
                const auto pos = header.find(kGitVersionHeader);
                if(pos != std::string::npos)
                {
                    fileBuildStamp = header.substr(pos + std::strlen(kGitVersionHeader));
                    continue;
                }
            }

            if(!isHeaderRow(header))
                continue;

            std::string valueLine;
            if(!std::getline(file_read, valueLine))
                break;

            const std::string value = trimmed(valueLine);
            if(isHeaderRow(value))
            {
                // An interrupted append can leave a header with nothing under it.
                // Hand the next header back to the loop rather than reading it as
                // this row's values, so the row that follows it survives.
                pendingHeader = value;
                continue;
            }

            auto parsed = problemFromRow(zipRow(splitCsv(header), splitCsv(value)));
            if(!parsed)
                continue;

            const auto& [key, entry] = *parsed;

            // A row that records a name is checked at replay, where its index is
            // resolved in the running library and the name must still match. A
            // row without one has nothing to check it against, so it is trusted
            // only when the file was written by this build.
            if(!entry.kernelName && !entry.solutionName && fileBuildStamp != currentBuildStamp())
                continue;

            m_override.addIfAbsent(key, entry);
        }

        // Only a clean read counts as loaded. A read that stopped on an I/O error
        // partway through would otherwise leave a partial map that is never
        // completed.
        if(!file_read.bad())
            m_override.markLoaded(path);
    }

    ProblemOverride::ProblemOverride()
        : m_transA(false)
        , m_transB(false)
        , m_inputTypeA(rocisa::DataType::None)
        , m_inputTypeB(rocisa::DataType::None)
        , m_computeType(rocisa::DataType::None)
        , m_outputType(rocisa::DataType::None)
        , m_m(0)
        , m_n(0)
        , m_k(0)
        , m_batchSize(0)
    {
    }

    ProblemOverride::ProblemOverride(bool             transA,
                                     bool             transB,
                                     rocisa::DataType inputTypeA,
                                     rocisa::DataType inputTypeB,
                                     rocisa::DataType computeType,
                                     rocisa::DataType outputType,
                                     size_t           m,
                                     size_t           n,
                                     size_t           k,
                                     size_t           batchSize)
        : m_transA(transA)
        , m_transB(transB)
        , m_inputTypeA(inputTypeA)
        , m_inputTypeB(inputTypeB)
        , m_computeType(computeType)
        , m_outputType(outputType)
        , m_m(m)
        , m_n(n)
        , m_k(k)
        , m_batchSize(batchSize)
    {
    }

    ProblemOverride::ProblemOverride(const ProblemOverride& problem)
    {

        m_transA      = problem.transA();
        m_transB      = problem.transB();
        m_inputTypeA  = problem.inputTypeA();
        m_inputTypeB  = problem.inputTypeB();
        m_computeType = problem.computeType();
        m_outputType  = problem.outputType();
        m_m           = problem.m();
        m_n           = problem.n();
        m_k           = problem.k();
        m_batchSize   = problem.batchSize();
    }

};
