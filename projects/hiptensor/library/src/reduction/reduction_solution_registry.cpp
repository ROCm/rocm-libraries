/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "reduction_solution_registry.hpp"
#include "reduction_solution.hpp"

namespace hiptensor
{
    ////////////////////////////////////////////////
    /// Class ReductionSolutionRegistry::Query ///
    ////////////////////////////////////////////////

    // @cond
    ReductionSolutionRegistry::Query
        ReductionSolutionRegistry::Query::query(hiptensorDataType_t          typeIn,
                                                hiptensorComputeDescriptor_t typeAcc,
                                                hiptensorDataType_t          typeOut,
                                                int                          rank,
                                                int                          numReduceDim,
                                                hiptensorOperator_t          opReduce,
                                                bool                         propagateNan,
                                                bool                         outputIndex) const
    {
        auto solutionHash = hashSolution(
            typeIn, typeAcc, typeOut, rank, numReduceDim, opReduce, propagateNan, outputIndex);

        if(auto solutions = mSolutionHash.find(solutionHash); solutions != mSolutionHash.end())
        {
            return Query(mSolutionHash.at(solutionHash));
        }

        return Query();
    }

    std::unordered_map<ReductionSolutionRegistry::Query::Uid, ReductionSolution*> const&
        ReductionSolutionRegistry::Query::solutions() const
    {
        return mAllSolutions;
    }

    uint32_t ReductionSolutionRegistry::Query::solutionCount() const
    {
        return mAllSolutions.size();
    }

    ///////////////
    /// Private ///
    ///////////////

    ReductionSolutionRegistry::Query::Query(std::vector<ReductionSolution*> const& solutions)
    {
        addSolutions(solutions);
    }

    /* static */
    ReductionSolutionRegistry::Query::HashId
        ReductionSolutionRegistry::Query::hashSolution(hiptensorDataType_t          typeIn,
                                                       hiptensorComputeDescriptor_t typeAcc,
                                                       hiptensorDataType_t          typeOut,
                                                       int                          rank,
                                                       int                          numReduceDim,
                                                       hiptensorOperator_t          opReduce,
                                                       bool                         propagateNan,
                                                       bool                         outputIndex)
    {
        return Hash{}(
            typeIn, typeAcc, typeOut, rank, numReduceDim, opReduce, propagateNan, outputIndex);
    }

    void ReductionSolutionRegistry::Query::addSolution(ReductionSolution* solution)
    {
        // Acquire unique ID and category ID per solution
        auto solutionUid = solution->uid();

        if(auto const& result = mAllSolutions.emplace(std::make_pair(solutionUid, solution));
           result.second == true)
        {
            auto solutionHash = std::hash<hiptensor::ReductionSolution>{}(*solution);
            mSolutionHash[solutionHash].push_back(solution);
        }
        else
        {
#if !NDEBUG
            std::cout << "Unique solution: " << solutionUid << " already exists!" << std::endl;
#endif // !NDEBUG
        }
    }

    void ReductionSolutionRegistry::Query::addSolutions(
        std::vector<ReductionSolution*> const& solutions)
    {
        for(auto* soln : solutions)
        {
            addSolution(soln);
        }
    }

    /////////////////////////////////////////
    /// Class ReductionSolutionRegistry ///
    /////////////////////////////////////////

    void ReductionSolutionRegistry::registerSolutions(
        std::vector<std::unique_ptr<ReductionSolution>>&& solutions)
    {
        for(auto&& solution : solutions)
        {
            // Register with the query then take ownership
            mSolutionQuery.addSolution(solution.get());
            mSolutionStorage.push_back(std::move(solution));
        }
    }

    uint32_t ReductionSolutionRegistry::solutionCount() const
    {
        return mSolutionStorage.size();
    }

    void ReductionSolutionRegistry::clear()
    {
        mSolutionStorage.clear();
        mSolutionQuery = Query();
    }
    // @endcond

} // namespace hiptensor
