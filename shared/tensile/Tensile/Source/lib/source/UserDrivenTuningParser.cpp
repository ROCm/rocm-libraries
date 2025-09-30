/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <Tensile/UserDrivenTuningParser.hpp>

#include <fstream>
#include <sstream>
#include <utility>

namespace Tensile
{
    int dataTypeSize(DataType dt)
    {
        switch(dt)
        {
        case DataType::Int8:
            return 1;

        case DataType::Half:
        case DataType::BFloat16:
            return 2;

        case DataType::Float:
        case DataType::Int32:
            return 4;

        case DataType::Double:
        case DataType::ComplexFloat:
            return 8;

        case DataType::ComplexDouble:
            return 18;

        default:
        case DataType::None:
            return 0;
        };
    }

    DataType convertToDataType(const std::string& DataTypeStr)
    {
        return DataTypeStr == "f16_r" || DataTypeStr == "h"   ? DataType::Half
               : DataTypeStr == "f32_r" || DataTypeStr == "s" ? DataType::Float
               : DataTypeStr == "f64_r" || DataTypeStr == "d" ? DataType::Double
               : DataTypeStr == "bf16_r"                      ? DataType::BFloat16
               : DataTypeStr == "f16_c"                       ? DataType::None
               : DataTypeStr == "f32_c" || DataTypeStr == "c" ? DataType::ComplexFloat
               : DataTypeStr == "f64_c" || DataTypeStr == "z" ? DataType::ComplexDouble
               : DataTypeStr == "bf16_c"                      ? DataType::None
               : DataTypeStr == "i8_r"                        ? DataType::Int8
               : DataTypeStr == "i32_r"                       ? DataType::Int32
               : DataTypeStr == "i8_c"                        ? DataType::None
               : DataTypeStr == "i32_c"                       ? DataType::None
               : DataTypeStr == "u8_r"                        ? DataType::None
               : DataTypeStr == "u32_r"                       ? DataType::None
               : DataTypeStr == "u8_c"                        ? DataType::None
               : DataTypeStr == "u32_c"                       ? DataType::None
                                                              : DataType::None;
    }

    template <>
    std::pair<ProblemOverride<ContractionProblem>, int>
        problemFromEntries(const std::vector<std::string>& entries)
    {
        const size_t entries_n = entries.size();
        if((entries_n != 15) && (entries_n != 18))
        {
            return std::make_pair(ProblemOverride<ContractionProblem>{}, -1);
        }

        // Common
        bool transA = (entries[0] != "N");
        bool transB = (entries[1] != "N");

        size_t m, n, b, k;
        size_t ldA, ldB, ldC;
        size_t strideA, strideB, strideC;
        double alpha, beta;

        DataType inputType   = DataType::None;
        DataType outputType  = DataType::None;
        DataType computeType = DataType::None;

        int solution_idx = -1;

        try
        {
            m = std::stol(entries[2]);
            n = std::stol(entries[3]);
            b = std::stol(entries[4]);
            k = std::stol(entries[5]);

            beta = std::stod(entries[7]);

            ldA = std::stol(entries[8]);
            ldB = std::stol(entries[9]);
            ldC = std::stol(entries[10]);

            strideA = 1;
            strideB = 1;
            strideC = 1;

            if(b > 1)
            {
                strideA = ldA * (transA ? m : k);
                strideB = ldB * (transB ? k : n);
                strideC = ldC * n;
            }

            if(entries_n == 15)
            {
                // Expected layout: transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,input_type,output_type,compute_type,solution_index
                inputType   = convertToDataType(entries[11]);
                outputType  = convertToDataType(entries[12]);
                computeType = convertToDataType(entries[13]);

                solution_idx = std::stoi(entries[14]);
            }
            else if(entries_n == 18)
            {
                // Expected layout: transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,stride_a,stride_b,stride_c,input_type,output_type,compute_type,solution_index
                strideA = std::stol(entries[11]);
                strideB = std::stol(entries[12]);
                strideC = std::stol(entries[13]);

                inputType   = convertToDataType(entries[14]);
                outputType  = convertToDataType(entries[15]);
                computeType = convertToDataType(entries[16]);

                solution_idx = std::stoi(entries[17]);
            }
        }
        catch(std::invalid_argument const& ex)
        {
            return std::make_pair(ProblemOverride<ContractionProblem>{}, -1);
        }
        catch(std::out_of_range const& ex)
        {
            return std::make_pair(ProblemOverride<ContractionProblem>{}, -1);
        }

        if(inputType == DataType::None || outputType == DataType::None
           || computeType == DataType::None)
        {
            return std::make_pair(ProblemOverride<ContractionProblem>{}, -1);
        }

        bool HPA = (dataTypeSize(computeType) > dataTypeSize(inputType));

        ProblemOverride<ContractionProblem> po(transA,
                                               transB,
                                               inputType,
                                               outputType,
                                               HPA,
                                               m,
                                               n,
                                               k,
                                               b,
                                               beta,
                                               ldA,
                                               strideA,
                                               ldB,
                                               strideB,
                                               ldC,
                                               strideC);

        return std::make_pair(po, solution_idx);
    }

    template <typename MyProblem>
    ProblemOverride<MyProblem>::ProblemOverride()
        : m_transA(false)
        , m_transB(false)
        , m_inputType(DataType::None)
        , m_outputType(DataType::None)
        , m_HPA(false)
        , m_m(0)
        , m_n(0)
        , m_k(0)
        , m_batchSize(0)
        , m_beta(0)
        , m_ldA(0)
        , m_strideA(0)
        , m_ldB(0)
        , m_strideB(0)
        , m_ldC(0)
        , m_strideC(0)
    {
    }

    template <typename MyProblem>
    ProblemOverride<MyProblem>::ProblemOverride(bool     transA,
                                                bool     transB,
                                                DataType inputType,
                                                DataType outputType,
                                                bool     HPA,
                                                size_t   m,
                                                size_t   n,
                                                size_t   k,
                                                size_t   batchSize,
                                                double   beta,
                                                size_t   ldA,
                                                size_t   strideA,
                                                size_t   ldB,
                                                size_t   strideB,
                                                size_t   ldC,
                                                size_t   strideC)
        : m_transA(transA)
        , m_transB(transB)
        , m_inputType(inputType)
        , m_outputType(outputType)
        , m_HPA(HPA)
        , m_m(m)
        , m_n(n)
        , m_k(k)
        , m_batchSize(batchSize)
        , m_beta(beta)
        , m_ldA(ldA)
        , m_strideA(strideA)
        , m_ldB(ldB)
        , m_strideB(strideB)
        , m_ldC(ldC)
        , m_strideC(strideC)
    {
    }

    template <>
    ProblemOverride<ContractionProblem>::ProblemOverride(const ContractionProblem& problem)
    {
        m_transA     = problem.transA();
        m_transB     = problem.transB();
        m_inputType  = problem.a().dataType();
        m_outputType = problem.c().dataType();
        m_HPA        = problem.highPrecisionAccumulate();
        m_m          = problem.freeSizeA(0);
        m_n          = problem.freeSizeB(0);
        m_k          = problem.boundSize(0);
        m_batchSize  = problem.batchSize(0);
        m_beta       = problem.beta();
        m_ldA        = problem.a().strides()[1];
        m_strideA    = problem.a().strides()[2];
        m_ldB        = problem.b().strides()[1];
        m_strideB    = problem.b().strides()[2];
        m_ldC        = problem.c().strides()[1];
        m_strideC    = problem.c().strides()[2];
    }

    template <>
    ContractionProblem ProblemOverride<ContractionProblem>::problem() const
    {
        auto problem = ContractionProblem::GEMM_Strides(m_transA,
                                                        m_transB,
                                                        m_inputType,
                                                        m_inputType,
                                                        m_outputType,
                                                        m_outputType,
                                                        m_m,
                                                        m_n,
                                                        m_k,
                                                        m_batchSize,
                                                        m_ldA,
                                                        m_strideA,
                                                        m_ldB,
                                                        m_strideB,
                                                        m_ldC,
                                                        m_strideC,
                                                        m_ldC,
                                                        m_strideC,
                                                        m_beta);
        problem.setHighPrecisionAccumulate(m_HPA);

        return problem;
    }

    template <>
    std::vector<std::pair<ProblemOverride<ContractionProblem>, int>>
        getContractionProblemsFromFile(const std::string& path)
    {
        std::vector<std::pair<ProblemOverride<ContractionProblem>, int>> out;

        std::ifstream file(path);
        std::string   line, entry;

        const auto delim         = ',';
        const auto first_heading = "transA";
        const int  max_entries   = 18;

        while(std::getline(file, line))
        {
            // Ignore lines without delimiter
            if(line.find(delim) == std::string::npos)
            {
                continue;
            }

            // Check for section start
            if(line.find(first_heading) != std::string::npos)
            {
                // TODO: Get param index from headings?
                continue;
            }

            std::vector<std::string> entries{};
            entries.reserve(max_entries);

            std::stringstream line_ss(line);
            while(getline(line_ss, entry, delim))
            {
                entries.push_back(entry);
            }

            auto problemSolution = problemFromEntries<ContractionProblem>(entries);
            if(problemSolution.second > 0)
            {
                out.push_back(problemSolution);
            }
        }

        return out;
    }
};
