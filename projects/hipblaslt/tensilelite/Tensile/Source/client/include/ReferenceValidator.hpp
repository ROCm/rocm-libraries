/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "RunListener.hpp"

#include <boost/program_options.hpp>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

#include "DataInitialization.hpp"

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        namespace po = boost::program_options;

        class ReferenceValidator : public RunListener
        {
        public:
            ReferenceValidator(po::variables_map const&            args,
                               std::shared_ptr<DataInitialization> dataInit);

            virtual bool needMoreBenchmarkRuns() const override;
            virtual void preBenchmarkRun() override;
            virtual void postBenchmarkRun() override;

            virtual void preProblem(ContractionProblem* const problem) override;
            virtual void postProblem() override;

            virtual void preSolution(ContractionSolution const& solution) override;
            virtual void postSolution() override;

            virtual bool needMoreRunsInSolution() const override;

            virtual size_t numWarmupRuns() override;
            virtual void   setNumWarmupRuns(size_t count) override;
            virtual void   preWarmup() override;
            virtual void   postWarmup(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override;
            virtual void   validateWarmups(std::shared_ptr<ProblemInputs> inputs,
                                           TimingEvents const&            startEvents,
                                           TimingEvents const&            stopEvents) override;

            virtual size_t numSyncs() override
            {
                return 0;
            }
            virtual void setNumSyncs(size_t count) override {}
            virtual void preSyncs() override {}
            virtual void postSyncs() override {}

            virtual size_t numEnqueuesPerSync() override
            {
                return 0;
            }
            virtual void setNumEnqueuesPerSync(size_t count) override {}
            virtual void preEnqueues(hipStream_t const& stream) override {}
            virtual void postEnqueues(TimingEvents const& startEvents,
                                      TimingEvents const& stopEvents,
                                      hipStream_t const&  stream) override
            {
            }
            virtual void validateEnqueues(std::shared_ptr<ProblemInputs> inputs,
                                          TimingEvents const&            startEvents,
                                          TimingEvents const&            stopEvents) override
            {
            }

            bool validate(ContractionProblemGemm const& problem,
                          ContractionInputs const&      reference,
                          ContractionInputs const&      result);

            bool checkResults(TensorDescriptor const& tensor,
                              void const*             refPtr,
                              void const*             resPtr,
                              size_t                  maxElements,
                              bool                    isgpu,
                              size_t                  validationStride);

            template <typename ValidType>
            bool checkResultsTyped(TensorDescriptor const& tensor,
                                   ValidType const*        reference,
                                   ValidType const*        result,
                                   size_t                  maxElement,
                                   bool                    isgpu,
                                   size_t                  validationStride);

            void printTensors(ContractionProblemGemm const& problem,
                              ContractionInputs const&      reference,
                              ContractionInputs const&      result);

            virtual void finalizeReport() override;

            virtual int error() const override;

        private:
            void allocateResultBuffer(size_t bytes);

            std::shared_ptr<DataInitialization> m_dataInit;
            std::shared_ptr<ProblemInputs>      m_referenceInputs;

            size_t                   m_cpuResultBufferSize = 0;
            std::shared_ptr<uint8_t> m_cpuResultBuffer;

            ContractionProblem* m_problem;

            bool m_enabled;

            int  m_elementsToValidate;
            bool m_printValids;
            int  m_printMax;

            bool m_printTensorA;
            bool m_printTensorB;
            bool m_printTensorC;
            bool m_printTensorD;
            bool m_printTensorRef;
            bool m_printTensorBias;
            bool m_printTensorScaleAlphaVec;
            bool m_printTensorAmaxD;
            bool m_printAny;

            int m_numBenchmarkRuns = 0;

            bool   m_validatedSolution = false;
            bool   m_errorInSolution   = false;
            bool   m_error             = false;
            bool   m_executedSolution  = false;
            size_t m_errorsReported    = 0;

            bool validateSolution(std::shared_ptr<ProblemInputs> inputs);
        };
    } // namespace Client
} // namespace TensileLite
