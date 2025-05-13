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
#include <fstream>
#include <shared_mutex>
#include <sstream>
#include <utility>

namespace TensileLite
{

    inline const char* HeaderFieldToString(HeaderFields field)
    {
        switch(field)
        {
        case HeaderFields::transA:
            return "transA";
        case HeaderFields::transB:
            return "transB";
        case HeaderFields::batch_count:
            return "batch_count";
        case HeaderFields::m:
            return "m";
        case HeaderFields::n:
            return "n";
        case HeaderFields::k:
            return "k";
        case HeaderFields::a_type:
            return "a_type";
        case HeaderFields::b_type:
            return "b_type";
        case HeaderFields::c_type:
            return "c_type";
        case HeaderFields::compute_type:
            return "compute_type";
        case HeaderFields::solution_index:
            return "solution_index";
        default:
            return "";
        }
    }

    void getContractionProblemsFromFile(const std::string& path)
    {
        OverrideMap&                m_override = OverrideMap::getMap();
        std::mutex&                 map_guard  = m_override.getLock();
        std::lock_guard<std::mutex> lock(map_guard);

        if(m_override.size() == 0)
        {

            std::ifstream file_read(path);
            std::string   header_line, header;
            std::string   value_line, value;
            const auto    delim = ',';

            while(std::getline(file_read, header_line))
            {
                // Ignore lines without delimiter
                header_line.erase(0, header_line.find_first_not_of(" \t\n\r\f\v"));
                HeaderFields current_field = HeaderFields::transA;

                if(header_line.find(HeaderFieldToString(current_field)) != std::string::npos)
                {

                    if(std::getline(file_read, value_line))
                    {
                        value_line.erase(0, value_line.find_first_not_of(" \t\n\r\f\v"));
                        std::vector<std::string> entries{};
                        entries.reserve(
                            static_cast<size_t>(static_cast<size_t>(HeaderFields::count)));
                        std::stringstream header_split(header_line);
                        std::stringstream value_split(value_line);

                        while(std::getline(header_split, header, delim)
                              && std::getline(value_split, value, delim))
                        {
                            if(header == HeaderFieldToString(current_field))
                            {
                                entries.push_back(value);
                                current_field = static_cast<HeaderFields>(
                                    static_cast<int>(current_field) + 1);
                            }

                            if(current_field == HeaderFields::count)
                                break;
                        }

                        auto problemSolution = problemFromEntries(entries);

                        if(problemSolution.second > 0)
                        {
                            auto sol_iter       = m_override.find(problemSolution.first);
                            bool duplicate_find = false;

                            for(auto sol_idx = sol_iter.first; sol_idx != sol_iter.second;
                                sol_idx++)
                            {
                                if(sol_idx->second == problemSolution.second)
                                {
                                    duplicate_find = true;
                                    break;
                                }
                            }

                            if(!duplicate_find)
                            {
                                m_override.add(problemSolution);
                            }
                        }
                    }
                }
            }
        }
    }

    std::pair<ProblemOverride, int> problemFromEntries(const std::vector<std::string>& entries)
    {

        const size_t entries_n = entries.size();
        if(entries_n != static_cast<size_t>(HeaderFields::count))
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        //Expected format: transA,transB,batch_count,M,N,K,input_type,output_type,compute_type,solution_index
        bool transA = (entries[static_cast<size_t>(HeaderFields::transA)] != "N");
        bool transB = (entries[static_cast<size_t>(HeaderFields::transB)] != "N");

        size_t           m, n, b, k;
        rocisa::DataType inputTypeA  = rocisa::DataType::None;
        rocisa::DataType inputTypeB  = rocisa::DataType::None;
        rocisa::DataType outputType  = rocisa::DataType::None;
        rocisa::DataType computeType = rocisa::DataType::None;

        int solution_idx = -1;

        try
        {

            // TODO: are any additional mapping parameters needed?

            b          = std::stol(entries[static_cast<size_t>(HeaderFields::batch_count)]);
            m          = std::stol(entries[static_cast<size_t>(HeaderFields::m)]);
            n          = std::stol(entries[static_cast<size_t>(HeaderFields::n)]);
            k          = std::stol(entries[static_cast<size_t>(HeaderFields::k)]);
            inputTypeA = hipDataType_to_tensile_type(
                string_to_hip_datatype(entries[static_cast<size_t>(HeaderFields::a_type)]));
            inputTypeB = hipDataType_to_tensile_type(
                string_to_hip_datatype(entries[static_cast<size_t>(HeaderFields::b_type)]));
            outputType = hipDataType_to_tensile_type(
                string_to_hip_datatype(entries[static_cast<size_t>(HeaderFields::c_type)]));
            computeType = rocComputeType_to_tensile_type(
                (rocblaslt_compute_type)string_to_hipblas_computetype(
                    entries[static_cast<size_t>(HeaderFields::compute_type)]));
            solution_idx = std::stoi(entries[static_cast<size_t>(HeaderFields::solution_index)]);
        }
        catch(std::invalid_argument const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }
        catch(std::out_of_range const& ex)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        if(inputTypeA == rocisa::DataType::None || inputTypeB == rocisa::DataType::None
           || outputType == rocisa::DataType::None || computeType == rocisa::DataType::None)
        {
            return std::make_pair(ProblemOverride{}, -1);
        }

        ProblemOverride po(
            transA, transB, inputTypeA, inputTypeB, computeType, outputType, m, n, k, b);

        return std::make_pair(po, solution_idx);
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
