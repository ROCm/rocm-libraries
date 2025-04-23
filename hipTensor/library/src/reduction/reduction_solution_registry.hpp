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

#ifndef HIPTENSOR_REDUCTION_SOLUTION_REGISTRY_HPP
#define HIPTENSOR_REDUCTION_SOLUTION_REGISTRY_HPP

#include <memory>
#include <unordered_map>
#include <vector>

#include "data_types.hpp"
#include "reduction_types.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    // @cond
    class ReductionSolution;

    class ReductionSolutionRegistry
    {
    public:
        class Query
        {
        public:
            friend class ReductionSolutionRegistry;
            using Uid    = std::size_t;
            using HashId = std::size_t;

            Query()  = default;
            ~Query() = default;
            Query(Query const& other);
            Query& operator=(Query const& other);

            /// Subsequent queries that may be performed on the current query object.
            /// E.g. in this context, query further parameters.

            // By solution type
            Query query(hipDataType            typeIn,
                        hiptensorComputeType_t typeAcc,
                        hipDataType            typeOut,
                        int                    rank,
                        int                    numReduceDim,
                        hiptensorOperator_t    opReduce,
                        bool                   propagateNan,
                        bool                   outputIndex) const;

            // union
            Query operator||(Query const& other) const;

            // intersection
            Query operator&&(Query const& other) const;

            // Full map of Uid to ReductionSolution*
            std::unordered_map<Uid, ReductionSolution*> const& solutions() const;

            uint32_t solutionCount() const;

            // Internal ctor
            Query(std::vector<ReductionSolution*> const& solutions);

        private:
            // Query by explicit hash
            Query query(HashId queryHash) const;

            // Hashing helpers
            static HashId hashSolution(hipDataType            typeIn,
                                       hiptensorComputeType_t typeAcc,
                                       hipDataType            typeOut,
                                       int                    rank,
                                       int                    numReduceDim,
                                       hiptensorOperator_t    opReduce,
                                       bool                   propagateNan,
                                       bool                   outputIndex);

            // Adding solutions to the query
            void addSolution(ReductionSolution* solution);
            void addSolutions(std::vector<ReductionSolution*> const& solutions);
            void addSolutions(std::unordered_map<Uid, ReductionSolution*> const& solutions);

        private: // members
            // This is the has of all solutions, by unique Uid
            // [Key = KernelUid, element = ReductionSolution*]
            std::unordered_map<Uid, ReductionSolution*> mAllSolutions;

            // This is the contextual query hash
            // [Key = Query hash, Element = ReductionSolution*]
            std::unordered_map<HashId, std::vector<ReductionSolution*>> mSolutionHash;
        };

    protected:
        // Move only
        ReductionSolutionRegistry()                                            = default;
        ReductionSolutionRegistry(ReductionSolutionRegistry&&)                 = default;
        ReductionSolutionRegistry& operator=(ReductionSolutionRegistry&&)      = default;
        ReductionSolutionRegistry(ReductionSolutionRegistry const&)            = delete;
        ReductionSolutionRegistry& operator=(ReductionSolutionRegistry const&) = delete;

        // Import reduction solutions for the registry to manage
        void registerSolutions(std::vector<std::unique_ptr<ReductionSolution>>&& solutions);

    public:
        virtual ~ReductionSolutionRegistry() = default;

        template <typename... Ts>
        Query querySolutions(Ts... ts)
        {
            return mSolutionQuery.query(ts...);
        }

        uint32_t solutionCount() const;

    private:
        std::vector<std::unique_ptr<ReductionSolution>> mSolutionStorage;
        Query                                           mSolutionQuery;
    };
    // @endcond

} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_SOLUTION_REGISTRY_HPP
