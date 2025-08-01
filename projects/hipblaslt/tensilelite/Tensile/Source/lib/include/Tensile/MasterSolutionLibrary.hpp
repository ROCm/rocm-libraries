/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <filesystem>
#include <map>
#include <memory>

#include <Tensile/Debug.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

namespace fs = std::filesystem;

namespace TensileLite
{

    /**
 * \ingroup SolutionLibrary
 */
    template <typename MySolution>
    using SolutionMap = std::map<int, std::shared_ptr<MySolution>>;

    template <typename MySolution>
    struct LibraryIOContext
    {
        std::string                  filename;
        std::vector<LazyLoadingInit> preloaded;
        // If lazy loading is used, this may be updated in const functions
        SolutionMap<MySolution>* solutions;
        std::mutex*              solutionsGuard;
        std::set<std::string>*   loadedFiles;

        void* indexLoadedLibraries;
    };

    /**
 * \ingroup SolutionLibrary
 *
 * Root level library object. Contains all individual solutions in a map
 * for serialization purposes.
 */
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct MasterSolutionLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        static std::string Type()
        {
            return "Master";
        }
        std::string type() const override
        {
            return Type();
        }
        std::string description() const override
        {
            if(library == nullptr)
                return concatenate(
                    type(), " (", solutions.size(), " solutions, next level: nullptr)");
            else
                return concatenate(type(),
                                   " (",
                                   solutions.size(),
                                   " solutions, next level: ",
                                   library->type(),
                                   ")");
        }

        std::string                   libraryDirectory;
        std::string                   suffix;
        std::map<int, std::string>    libraryMapping;
        mutable std::set<std::string> loadedFiles;

        mutable std::map<std::string, std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>>
            indexLoadedLibraries;

        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> library;
        mutable SolutionMap<MySolution>                         solutions;
        std::string                                             version;
        mutable std::mutex                                      solutionsGuard;

        MasterSolutionLibrary() = default;

        bool initLibraryMapping(const std::string& tensileLibPath)
        {
            fs::path path(tensileLibPath);
            libraryDirectory = path.parent_path().string();
            suffix           = path.extension().string();
            path             = fs::path(libraryDirectory) / "TensileLiteLibrary_lazy_Mapping.dat";

            // libraryMapping
            libraryMapping = LoadLibraryMapping(path.string());
            if(libraryMapping.empty())
            {
                std::cout << "No library mapping found in " << libraryDirectory << std::endl;
                return false;
            }
            return true;
        }

        void loadLibrary(const int index) const
        {
            auto it = libraryMapping.upper_bound(index);
            if(it != libraryMapping.begin())
            {
                --it;
                std::string filePrefix = it->second;
                // load the file here directly and push the library for later use.
                {
                    std::lock_guard<std::mutex> lock(solutionsGuard);
                    if(loadedFiles.find(filePrefix) != loadedFiles.end())
                        return;
                }
                if(Debug::Instance().printDataInit())
                    std::cout << "Loading library for index " << index
                              << " from file: " << filePrefix << std::endl;

                fs::path path(libraryDirectory);
                path = path / (filePrefix + suffix);

                auto newLibrary = LoadLibraryFile<MyProblem, MySolution>(path.string());
                auto mLibrary
                    = static_cast<MasterSolutionLibrary<MyProblem, MySolution>*>(newLibrary.get());

                using std::begin;
                using std::end;

                std::lock_guard<std::mutex> lock(solutionsGuard);
                if(loadedFiles.find(filePrefix) != loadedFiles.end())
                    return;
                // Push to cache
                indexLoadedLibraries[filePrefix] = mLibrary->library;

                std::transform(begin(mLibrary->solutions),
                               end(mLibrary->solutions),
                               std::inserter(solutions, end(solutions)),
                               [this, filePrefix](auto& i) {
                                   i.second->codeObjectFilename = filePrefix + ".co";
                                   return i;
                               });
                loadedFiles.insert(filePrefix);

                if(Debug::Instance().printCodeObjectInfo())
                    std::cout << "load placeholder library " << path << std::endl
                              << mLibrary->solutions.size() << " solutions loaded" << std::endl;
            }
            else
            {
                std::cerr << "No library found for index " << index << std::endl;
            }
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            if(solutions.find(index) == solutions.end())
            {
                return std::shared_ptr<MySolution>();
            }
            auto solution = solutions.at(index);
            if(solution->requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
            {
                solution->requiredHostWorkspaceSizePerProblem
                    = solution->requiredHostSizeGroupedGemmSingle(problem, hardware);
            }
            return solution;
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(Hardware const& hardware,
                                                               const int       index) const override
        {
            loadLibrary(index);
            if(solutions.find(index) == solutions.end())
            {
                return std::shared_ptr<MySolution>();
            }
            auto solution = solutions.at(index);
            if(solution->requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
            {
                auto problem
                    = MyProblem::createDefaultProblem(solution->problemType.transA,
                                                      solution->problemType.transB,
                                                      solution->problemType.aType,
                                                      solution->problemType.bType,
                                                      solution->problemType.cType,
                                                      solution->problemType.dType,
                                                      solution->problemType.computeType,
                                                      solution->problemType.computeType,
                                                      solution->problemType.computeInputType,
                                                      solution->problemType.computeType,
                                                      1.0,
                                                      1.0,
                                                      solution->problemType.useBias,
                                                      solution->problemType.useGradient,
                                                      solution->problemType.biasDataTypeWhiteList,
                                                      solution->problemType.biasSrcWhiteList,
                                                      solution->problemType.groupedGemm,
                                                      std::numeric_limits<size_t>::max());
                solution->requiredHostWorkspaceSizePerProblem
                    = solution->requiredHostSizeGroupedGemmSingle(problem, hardware);
            }
            return solution;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            if(Debug::Instance().printSolutionSelectionTime())
            {
                auto start  = std::chrono::steady_clock::now();
                auto result = findBestSolution_runner(problem, hardware, fitness);
                auto end    = std::chrono::steady_clock::now();

                double time = std::chrono::duration<double, std::micro>(end - start).count();
                std::cout << "Solution selection time: " << time << " us" << std::endl;

                return result;
            }
            else
            {
                return findBestSolution_runner(problem, hardware, fitness);
            }
        }

        std::shared_ptr<MySolution> findBestSolution_runner(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            double* fitness = nullptr) const
        {
            const int                   solution_index = Debug::Instance().getSolutionIndex();
            std::shared_ptr<MySolution> rv;

            if(solution_index >= 0)
            {
                std::cout << "Tensile will use solution index: " << solution_index << std::endl;
                std::cout
                    << "Warning: Tensile will only work for a particular transpose and data type."
                    << std::endl;
                std::cout << "Set TENSILE_SOLUTION_INDEX to a negative number to restore the "
                             "default behavior."
                          << std::endl;
                {
                    std::lock_guard<std::mutex> guard(solutionsGuard);
                    auto                        selected_solution = solutions.at(solution_index);
                    Task                        task(hardware, problem, *(selected_solution));
                    if((*selected_solution->problemPredicate)(problem)
                       && (*selected_solution->taskPredicate)(task)
                       && (*selected_solution->hardwarePredicate)(hardware))
                        rv = selected_solution;
                    else
                        return nullptr;
                }
            }
            else
                rv = library->findBestSolution(problem, hardware, fitness);

            if(Debug::Instance().printLibraryLogicIndex())
            {
                if(rv)
                    std::cout << "Library logic solution index of winning solution: "
                              << rv->libraryLogicIndex << std::endl;
                else
                    std::cout << "No solution found" << std::endl;
            }
            return rv;
        }
        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            return library->findAllSolutions(problem, hardware, searchType);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return library->findAllSolutionsGroupedGemm(problems, hardware, searchType);
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            if(Debug::Instance().printSolutionSelectionTime())
            {
                auto   start  = std::chrono::steady_clock::now();
                auto   result = library->findTopSolutions(problem, hardware, numSolutions);
                auto   end    = std::chrono::steady_clock::now();
                double time   = std::chrono::duration<double, std::micro>(end - start).count();
                std::cout << "Solution selection time: " << time << " us" << std::endl;

                return result;
            }
            else
            {
                return library->findTopSolutions(problem, hardware, numSolutions);
            }
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            return library->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
        }
    };

} // namespace TensileLite
