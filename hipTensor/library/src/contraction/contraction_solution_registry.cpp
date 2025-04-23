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

#include "contraction_solution_registry.hpp"
#include "contraction_solution.hpp"

namespace hiptensor
{
    ////////////////////////////////////////////////
    /// Class ContractionSolutionRegistry::Query ///
    ////////////////////////////////////////////////

    // @cond
    ContractionSolutionRegistry::Query::Query(Query const& other)
        : mAllSolutions(other.mAllSolutions)
        , mSolutionHash(other.mSolutionHash)
    {
    }

    ContractionSolutionRegistry::Query&
        ContractionSolutionRegistry::Query::operator=(Query const& other)
    {
        if(&other != this)
        {
            mAllSolutions = other.mAllSolutions;
            mSolutionHash = other.mSolutionHash;
        }

        return *this;
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::query(int32_t                dimsM,
                                                  int32_t                dimsN,
                                                  int32_t                dimsK,
                                                  hipDataType            typeA,
                                                  hipDataType            typeB,
                                                  hipDataType            typeC,
                                                  hipDataType            typeD,
                                                  hiptensorOperator_t    opA,
                                                  hiptensorOperator_t    opB,
                                                  ContractionOpId_t      opCDE,
                                                  hiptensorComputeType_t typeCompute) const
    {
        auto solutionHash = hashSolution(
            dimsM, dimsN, dimsK, typeA, typeB, typeC, typeD, opA, opB, opCDE, typeCompute);

        if(auto solutions = mSolutionHash.find(solutionHash); solutions != mSolutionHash.end())
        {
            return Query(mSolutionHash.at(solutionHash));
        }

        return Query();
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::query(int32_t dimsM, int32_t dimsN, int32_t dimsK) const
    {
        return query(hashDimsMNK(dimsM, dimsN, dimsK));
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::query(hipDataType            typeA,
                                                  hipDataType            typeB,
                                                  hipDataType            typeC,
                                                  hipDataType            typeD,
                                                  hiptensorComputeType_t typeCompute) const
    {
        return query(hashTypesComputeABCD(typeA, typeB, typeC, typeD, typeCompute));
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::query(hiptensorOperator_t opA,
                                                  hiptensorOperator_t opB) const
    {
        return query(hashElementOps(opA, opB));
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::query(ContractionOpId_t opCDE) const
    {
        return query(hashContractionOps(opCDE));
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::operator||(Query const& other) const
    {
        auto newQuery = *this;
        newQuery.addSolutions(other.mAllSolutions);
        return newQuery;
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::operator&&(Query const& other) const
    {
        auto newQuery = Query();

        // Add only if both queries have the solution
        for(auto& solutionPair : other.mAllSolutions)
        {
            if(auto solution = mAllSolutions.find(solutionPair.first);
               solution != mAllSolutions.end())
            {
                newQuery.addSolution(solutionPair.second);
            }
        }

        return newQuery;
    }

    std::unordered_map<ContractionSolutionRegistry::Query::Uid, ContractionSolution*> const&
        ContractionSolutionRegistry::Query::solutions() const
    {
        return mAllSolutions;
    }

    uint32_t ContractionSolutionRegistry::Query::solutionCount() const
    {
        return mAllSolutions.size();
    }

    ///////////////
    /// Private ///
    ///////////////

    ContractionSolutionRegistry::Query::Query(std::vector<ContractionSolution*> const& solutions)
    {
        addSolutions(solutions);
    }

    ContractionSolutionRegistry::Query
        ContractionSolutionRegistry::Query::query(HashId queryHash) const
    {
        if(auto solutions = mSolutionHash.find(queryHash); solutions != mSolutionHash.end())
        {
            return Query(mSolutionHash.at(queryHash));
        }

        return Query();
    }

    /* static */
    ContractionSolutionRegistry::Query::HashId
        ContractionSolutionRegistry::Query::hashSolution(int32_t                dimsM,
                                                         int32_t                dimsN,
                                                         int32_t                dimsK,
                                                         hipDataType            typeA,
                                                         hipDataType            typeB,
                                                         hipDataType            typeC,
                                                         hipDataType            typeD,
                                                         hiptensorOperator_t    opA,
                                                         hiptensorOperator_t    opB,
                                                         ContractionOpId_t      opCDE,
                                                         hiptensorComputeType_t typeCompute)
    {
        return Hash{}(
            dimsM, dimsN, dimsK, typeA, typeB, typeC, typeD, opA, opB, opCDE, typeCompute);
    }

    /* static */
    ContractionSolutionRegistry::Query::HashId
        ContractionSolutionRegistry::Query::hashDimsMNK(int32_t dimsM, int32_t dimsN, int32_t dimsK)
    {
        return Hash{}(dimsM, dimsN, dimsK);
    }

    /* static */
    ContractionSolutionRegistry::Query::HashId
        ContractionSolutionRegistry::Query::hashTypesComputeABCD(hipDataType            typeA,
                                                                 hipDataType            typeB,
                                                                 hipDataType            typeC,
                                                                 hipDataType            typeD,
                                                                 hiptensorComputeType_t typeCompute)
    {
        return Hash{}(typeA, typeB, typeC, typeD, typeCompute);
    }

    /* static */
    ContractionSolutionRegistry::Query::HashId
        ContractionSolutionRegistry::Query::hashElementOps(hiptensorOperator_t opA,
                                                           hiptensorOperator_t opB)
    {
        return Hash{}(opA, opB);
    }

    /* static */
    ContractionSolutionRegistry::Query::HashId
        ContractionSolutionRegistry::Query::hashContractionOps(ContractionOpId_t opCDE)
    {
        return Hash{}(opCDE);
    }

    void ContractionSolutionRegistry::Query::addSolution(ContractionSolution* solution)
    {
        // Acquire unique ID and category ID per solution
        auto  solutionUid = solution->uid();
        auto& params      = solution->params();

        if(auto const& result = mAllSolutions.emplace(std::make_pair(solutionUid, solution));
           result.second == true)
        {
            auto solutionHash = hashSolution(params->dimsM(),
                                             params->dimsN(),
                                             params->dimsK(),
                                             params->typeA(),
                                             params->typeB(),
                                             params->typeC(),
                                             params->typeD(),
                                             params->opA(),
                                             params->opB(),
                                             params->opCDE(),
                                             params->typeCompute());

            auto dimsMNKHash = hashDimsMNK(params->dimsM(), params->dimsN(), params->dimsK());

            auto typesComputeABCDHash = hashTypesComputeABCD(params->typeA(),
                                                             params->typeB(),
                                                             params->typeC(),
                                                             params->typeD(),
                                                             params->typeCompute());

            auto elementOpsHash = hashElementOps(params->opA(), params->opB());

            auto contactionOpsHash = hashContractionOps(params->opCDE());

            // Hash contraction solutions into categories and then register
            // into master list.
            mAllSolutions[solutionUid] = solution;
            mSolutionHash[solutionHash].push_back(solution);
            mSolutionHash[dimsMNKHash].push_back(solution);
            mSolutionHash[typesComputeABCDHash].push_back(solution);
            mSolutionHash[elementOpsHash].push_back(solution);
            mSolutionHash[contactionOpsHash].push_back(solution);
        }
        else
        {
#if !NDEBUG
            std::cout << "Unique solution: " << solutionUid << " already exists!" << std::endl;
#endif // !NDEBUG
        }
    }

    void ContractionSolutionRegistry::Query::addSolutions(
        std::vector<ContractionSolution*> const& solutions)
    {
        for(auto* soln : solutions)
        {
            addSolution(soln);
        }
    }

    void ContractionSolutionRegistry::Query::addSolutions(
        std::unordered_map<Uid, ContractionSolution*> const& solutions)
    {
        for(auto& solutionPair : solutions)
        {
            addSolution(solutionPair.second);
        }
    }

    /////////////////////////////////////////
    /// Class ContractionSolutionRegistry ///
    /////////////////////////////////////////

    void ContractionSolutionRegistry::registerSolutions(
        std::vector<std::unique_ptr<ContractionSolution>>&& solutions)
    {
        for(auto&& solution : solutions)
        {
            // Register with the query then take ownership
            mSolutionQuery.addSolution(solution.get());
            mSolutionStorage.push_back(std::move(solution));
        }
    }

    ContractionSolutionRegistry::Query const& ContractionSolutionRegistry::allSolutions() const
    {
        return mSolutionQuery;
    }

    uint32_t ContractionSolutionRegistry::solutionCount() const
    {
        return mSolutionStorage.size();
    }
    // @endcond

} // namespace hiptensor
