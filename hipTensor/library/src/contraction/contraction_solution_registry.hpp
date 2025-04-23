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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_REGISTRY_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_REGISTRY_HPP

#include <memory>
#include <unordered_map>
#include <vector>

#include "contraction_types.hpp"
#include "data_types.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    // @cond
    class ContractionSolution;

    class ContractionSolutionRegistry
    {
    public:
        class Query
        {
        public:
            friend class ContractionSolutionRegistry;
            using Uid    = std::size_t;
            using HashId = std::size_t;

            Query()  = default;
            ~Query() = default;
            Query(Query const& other);
            Query& operator=(Query const& other);

            /// Subsequent queries that may be performed on the current query object.
            /// E.g. in this context, query further parameters.

            // By full solution type
            Query query(int32_t                dimsM,
                        int32_t                dimsN,
                        int32_t                dimsK,
                        hipDataType            typeA,
                        hipDataType            typeB,
                        hipDataType            typeC,
                        hipDataType            typeD,
                        hiptensorOperator_t    opA,
                        hiptensorOperator_t    opB,
                        ContractionOpId_t      opCDE,
                        hiptensorComputeType_t typeCompute) const;

            // By dimensions
            Query query(int32_t dimsM, int32_t dimsN, int32_t dimsK) const;

            // By data types
            Query query(hipDataType            typeA,
                        hipDataType            typeB,
                        hipDataType            typeC,
                        hipDataType            typeD,
                        hiptensorComputeType_t typeCompute) const;

            // By element-wise operations
            Query query(hiptensorOperator_t opA, hiptensorOperator_t opB) const;

            // By contraction operation
            Query query(ContractionOpId_t opCDE) const;

            // union
            Query operator||(Query const& other) const;

            // intersection
            Query operator&&(Query const& other) const;

            // Full map of Uid to ContractionSolution*
            std::unordered_map<Uid, ContractionSolution*> const& solutions() const;

            uint32_t solutionCount() const;

            // Internal ctor
            Query(std::vector<ContractionSolution*> const& solutions);

        private:
            // Query by explicit hash
            Query query(HashId queryHash) const;

            // Hashing helpers
            static HashId hashSolution(int32_t                dimsM,
                                       int32_t                dimsN,
                                       int32_t                dimsK,
                                       hipDataType            typeA,
                                       hipDataType            typeB,
                                       hipDataType            typeC,
                                       hipDataType            typeD,
                                       hiptensorOperator_t    opA,
                                       hiptensorOperator_t    opB,
                                       ContractionOpId_t      opCDE,
                                       hiptensorComputeType_t typeCompute);

            static HashId hashDimsMNK(int32_t dimsM, int32_t dimsN, int32_t dimsK);
            static HashId hashTypesComputeABCD(hipDataType            typeA,
                                               hipDataType            typeB,
                                               hipDataType            typeC,
                                               hipDataType            typeD,
                                               hiptensorComputeType_t typeCompute);
            static HashId hashElementOps(hiptensorOperator_t opA, hiptensorOperator_t opB);
            static HashId hashContractionOps(ContractionOpId_t opCDE);

            // Adding solutions to the query
            void addSolution(ContractionSolution* solution);
            void addSolutions(std::vector<ContractionSolution*> const& solutions);
            void addSolutions(std::unordered_map<Uid, ContractionSolution*> const& solutions);

        private: // members
            // This is the has of all solutions, by unique Uid
            // [Key = KernelUid, element = ContractionSolution*]
            std::unordered_map<Uid, ContractionSolution*> mAllSolutions;

            // This is the contextual query hash
            // [Key = Query hash, Element = ContractionSolution*]
            std::unordered_map<HashId, std::vector<ContractionSolution*>> mSolutionHash;
        };

    protected:
        // Move only
        ContractionSolutionRegistry()                                              = default;
        ContractionSolutionRegistry(ContractionSolutionRegistry&&)                 = default;
        ContractionSolutionRegistry& operator=(ContractionSolutionRegistry&&)      = default;
        ContractionSolutionRegistry(ContractionSolutionRegistry const&)            = delete;
        ContractionSolutionRegistry& operator=(ContractionSolutionRegistry const&) = delete;

        // Import contraction solutions for the registry to manage
        void registerSolutions(std::vector<std::unique_ptr<ContractionSolution>>&& solutions);

    public:
        virtual ~ContractionSolutionRegistry() = default;

        template <typename... Ts>
        Query querySolutions(Ts... ts)
        {
            return mSolutionQuery.query(ts...);
        }

        Query const& allSolutions() const;

        uint32_t solutionCount() const;

    private:
        std::vector<std::unique_ptr<ContractionSolution>> mSolutionStorage;
        Query                                             mSolutionQuery;
    };
    // @endcond

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_REGISTRY_HPP
