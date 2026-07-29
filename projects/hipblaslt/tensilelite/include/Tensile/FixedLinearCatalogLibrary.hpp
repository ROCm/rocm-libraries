// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/FixedLinearArbiterLibrary.hpp>

namespace TensileLite
{
    /** Applies the unchanged frozen 22-feature ranker to every kernel in one catalog. */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct FixedLinearCatalogLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        std::vector<std::pair<int, std::shared_ptr<MySolution>>> solutions;
        std::string modelId;
        std::string featureSchema;
        std::vector<double> weights;
        double cuCount = 96.0;
        mutable std::atomic<bool> lastFindTopRetAll = false;

        static std::string Type() { return "FixedLinearCatalog"; }
        std::string type() const override { return Type(); }
        std::string description() const override { return concatenate(Type(), ", model=", modelId, ", solutions=", solutions.size()); }

        std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const&, Hardware const&, int index) const override
        {
            auto found = std::find_if(solutions.begin(), solutions.end(), [index](auto const& item) { return item.first == index; });
            return found == solutions.end() ? nullptr : found->second;
        }

        std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem, Hardware const& hardware, double* fitness=nullptr) const override
        {
            auto result=findTopSolutions(problem,hardware,1);return result.empty()?nullptr:result.front();
        }

        SolutionVector<MySolution> findTopSolutions(MyProblem const& problem, Hardware const& hardware, int count) const override
        {
            SolutionVector<MySolution> result;
            FixedLinearArbiterLibrary<MyProblem,MySolution> scorer;scorer.weights=weights;scorer.cuCount=cuCount;
            std::vector<std::pair<double,std::shared_ptr<MySolution>>> ranked;ranked.reserve(solutions.size());
            for(auto const& item:solutions)
            {
                auto const& solution=item.second;
                if((*(solution->hardwarePredicate))(hardware)&&(*(solution->problemPredicate))(problem))
                    ranked.emplace_back(scorer.score(problem,*solution),solution);
            }
            std::stable_sort(ranked.begin(),ranked.end(),[](auto const&a,auto const&b){return a.first>b.first;});
            for(auto const& item:ranked)
            {
                item.second->tag=MySolution::MatchingTag::FixedLinearCatalog;result.push_back(item.second);
                if(result.size()==static_cast<size_t>(count))break;
            }
            if(Debug::Instance().printPropertyEvaluation()&&!ranked.empty())
                std::cout<<"FixedLinearCatalog: selected index="<<ranked.front().second->index<<" score="<<ranked.front().first<<" from="<<ranked.size()<<std::endl;
            lastFindTopRetAll=result.size()<static_cast<size_t>(count);return result;
        }

        SolutionSet<MySolution> findAllSolutions(MyProblem const&,Hardware const&,SolutionLibrarySearchType=SolutionLibrarySearchType::DEFAULT) const override
        {SolutionSet<MySolution> result;for(auto const& item:solutions)result.insert(item.second);return result;}
        SolutionSet<MySolution> findAllSolutionsGroupedGemm(std::vector<MyProblem> const&,Hardware const&,SolutionLibrarySearchType=SolutionLibrarySearchType::DEFAULT) const override{return {};}
        bool lastFindTopAlreadyRetAll() const override{return lastFindTopRetAll;}
    };
}
