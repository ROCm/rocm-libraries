/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022 Advanced Micro Devices, Inc.
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

#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include <algorithm>

namespace Tensile
{
    // Which placeholder libraries should be initialized at start
    // To be extended in the future
    enum class LazyLoadingInit
    {
        None,
        gfx803,
        gfx900,
        gfx906,
        gfx908,
        gfx90a,
        gfx942,
        gfx950,
        gfx1010,
        gfx1011,
        gfx1012,
        gfx1030,
        gfx1031,
        gfx1032,
        gfx1034,
        gfx1035,
        gfx1100,
        gfx1101,
        gfx1102,
        gfx1103,
        gfx1150,
        gfx1151,
        gfx1200,
        gfx1201,
        All
    };

    //Regex patterns for initializing libraries on startup
    inline std::string RegexPattern(LazyLoadingInit condition)
    {
        switch(condition)
        {
        case LazyLoadingInit::All:
            return "TensileLibrary_*";
        case LazyLoadingInit::gfx803:
            return "TensileLibrary_*_gfx803";
        case LazyLoadingInit::gfx900:
            return "TensileLibrary_*_gfx900";
        case LazyLoadingInit::gfx906:
            return "TensileLibrary_*_gfx906";
        case LazyLoadingInit::gfx908:
            return "TensileLibrary_*_gfx908";
        case LazyLoadingInit::gfx90a:
            return "TensileLibrary_*_gfx90a";
        case LazyLoadingInit::gfx942:
            return "TensileLibrary_*_gfx942";
        case LazyLoadingInit::gfx950:
            return "TensileLibrary_*_gfx950";
        case LazyLoadingInit::gfx1010:
            return "TensileLibrary_*_gfx1010";
        case LazyLoadingInit::gfx1011:
            return "TensileLibrary_*_gfx1011";
        case LazyLoadingInit::gfx1012:
            return "TensileLibrary_*_gfx1012";
        case LazyLoadingInit::gfx1030:
            return "TensileLibrary_*_gfx1030";
        case LazyLoadingInit::gfx1031:
            return "TensileLibrary_*_gfx1031";
        case LazyLoadingInit::gfx1032:
            return "TensileLibrary_*_gfx1032";
        case LazyLoadingInit::gfx1034:
            return "TensileLibrary_*_gfx1034";
        case LazyLoadingInit::gfx1035:
            return "TensileLibrary_*_gfx1035";
        case LazyLoadingInit::gfx1100:
            return "TensileLibrary_*_gfx1100";
        case LazyLoadingInit::gfx1101:
            return "TensileLibrary_*_gfx1101";
        case LazyLoadingInit::gfx1102:
            return "TensileLibrary_*_gfx1102";
        case LazyLoadingInit::gfx1103:
            return "TensileLibrary_*_gfx1103";
        case LazyLoadingInit::gfx1150:
            return "TensileLibrary_*_gfx1150";
        case LazyLoadingInit::gfx1151:
            return "TensileLibrary_*_gfx1151";
        case LazyLoadingInit::gfx1200:
            return "TensileLibrary_*_gfx1200";
        case LazyLoadingInit::gfx1201:
            return "TensileLibrary_*_gfx1201";
        case LazyLoadingInit::None:
            return "";
        }
        return "";
    }

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct PlaceholderLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        mutable std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> library;
        mutable SolutionMap<MySolution>*                                masterSolutions;
        mutable std::mutex*                                             solutionsGuard;
        mutable std::mutex                                              lazyLoadingGuard;
        std::string                                                     filePrefix;
        std::string                                                     suffix;
        std::string                                                     libraryDirectory;

        PlaceholderLibrary() = default;

        bool loadPlaceholderLibrary() const
        {
            std::lock_guard<std::mutex> lock(lazyLoadingGuard);
            // If condition in case two threads got into this function
            if(!library)
            {
                auto newLibrary = LoadLibraryFile<MyProblem, MySolution>(
                    (libraryDirectory + "/" + filePrefix + suffix).c_str());
                auto mLibrary
                    = static_cast<MasterSolutionLibrary<MyProblem, MySolution>*>(newLibrary.get());
                library = mLibrary->library;
                std::lock_guard<std::mutex> lock(*solutionsGuard);
                masterSolutions->insert(mLibrary->solutions.begin(), mLibrary->solutions.end());

                return mLibrary;
            }

            return false;
        }

        std::string getCodeObjectFileName(Hardware const&   hardware,
                                          MySolution const& solution) const
        {
            std::string coFileDependency = filePrefix;

            if(solution.isSourceKernel())
            {
                std::string arch = hardware.archName();
                //Remove xnack information if present
                auto pos = arch.find(":");
                if(pos != std::string::npos)
                    arch.resize(pos);

                if(coFileDependency.find("fallback") != std::string::npos)
                    coFileDependency += std::string("_") + arch + std::string(".hsaco");
                else
                    coFileDependency += std::string(".hsaco");
            }
            else
                coFileDependency += std::string(".co");

            return coFileDependency;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            if(!library)
                loadPlaceholderLibrary();

            auto solution = library->findBestSolution(problem, hardware, fitness);

            if(solution)
                solution->codeObjectFilename = getCodeObjectFileName(hardware, *solution);

            return solution;
        }

        /**
         * Returns all `Solution` objects that are capable of correctly solving this
         * `problem` on this `hardware`.
         *
         * May return an empty set if no such object exists.
         */
        virtual SolutionSet<MySolution> findAllSolutions(MyProblem const& problem,
                                                         Hardware const&  hardware) const override
        {
            if(!library)
            {
                loadPlaceholderLibrary();
            }

            auto solutions = library->findAllSolutions(problem, hardware);

            for(auto& solution : solutions)
            {
                solution->codeObjectFilename = getCodeObjectFileName(hardware, *solution);
            }

            return solutions;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsMatchingType(MyProblem const& problem,
                                         Hardware const&  hardware) const override
        {
            if(!library)
            {
                loadPlaceholderLibrary();
            }

            auto solutions = library->findAllSolutionsMatchingType(problem, hardware);

            for(auto& solution : solutions)
            {
                solution->codeObjectFilename = getCodeObjectFileName(hardware, *solution);
            }

            return solutions;
        }

        static std::string Type()
        {
            return "Placeholder";
        }

        virtual std::string type() const override
        {
            return Type();
        }

        virtual std::string description() const override
        {
            return this->type();
        }
    };

} // namespace Tensile
