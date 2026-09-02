/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/SolutionBlobCache.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TensorOps.hpp>

#include <tensilelitehost/export.h>

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

        // Set only when loading an indexed (format_version 2) file. Leaf nodes
        // capture this instead of resolving an index against `solutions`, which
        // stays empty for indexed files.
        std::shared_ptr<SolutionBlobCache<MySolution>> blobCache;

        // Points at the owning master's shard caches, keyed by file prefix, so a
        // placeholder can publish a shard's indices after loading it. Defaulted
        // because the existing aggregate initializers in the loaders pass only
        // the first few fields.
        std::map<std::string, std::shared_ptr<SolutionBlobCache<MySolution>>>* solutionSources
            = nullptr;
    };

    /**
 * Resolves a solution index against a load context, covering both layouts:
 * the eager `solutions` map for legacy files and the blob cache for indexed
 * ones. Returns nullptr for an unknown index so callers can raise a
 * load-time error, matching legacy behaviour.
 *
 * Nodes that rank across a whole index set use this and therefore materialize
 * their solutions at load. That is deliberate: in shipped gfx942 libraries
 * GranularitySelection and MLPClassification do not appear at all, and
 * Prediction only appears in small dedicated shards where it owns the entire
 * file, so per-node laziness would buy nothing there. Leaf nodes, which own
 * effectively all solutions, stay lazy via SingleSolutionLibrary.
 */
    template <typename MySolution>
    std::shared_ptr<MySolution> resolveContextSolution(LibraryIOContext<MySolution>* ctx,
                                                       int                           index)
    {
        if(ctx == nullptr)
            return std::shared_ptr<MySolution>();

        if(ctx->blobCache)
            return ctx->blobCache->get(index);

        if(ctx->solutions == nullptr)
            return std::shared_ptr<MySolution>();

        auto iter = ctx->solutions->find(index);
        if(iter == ctx->solutions->end())
            return std::shared_ptr<MySolution>();

        return iter->second;
    }

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
            // For an indexed library `solutions` only holds what has been
            // materialized so far, so report the cache's total instead of
            // printing 0 for a fully loaded file.
            size_t count = solutions.size();
            if(blobCache)
                count = blobCache->size();

            if(library == nullptr)
                return concatenate(type(), " (", count, " solutions, next level: nullptr)");
            else
                return concatenate(
                    type(), " (", count, " solutions, next level: ", library->type(), ")");
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
        mutable std::atomic<bool>                               lastFindTopRetAll = false;

        // Indexed-format state. `blobCache` serves this file's own solutions;
        // `solutionSources` collects the caches of lazily loaded shards, keyed by
        // file prefix. Keyed by shard rather than by solution index on purpose:
        // registering every index would rebuild the very table this format
        // exists to avoid, and only a handful of shards are ever resident. The
        // key also makes registration idempotent per shard, so a shard reached
        // through both loadLibrary and a placeholder cannot retain two blobs.
        std::shared_ptr<SolutionBlobCache<MySolution>> blobCache;
        mutable std::map<std::string, std::shared_ptr<SolutionBlobCache<MySolution>>>
            solutionSources;

        MasterSolutionLibrary() = default;

        bool initLibraryMapping(const std::string& tensileLibPath)
        {
            // Invariant: tensileLibPath is the logical library name ending in a
            // single extension (".dat" or ".yaml"), never the compressed
            // ".dat.zlib" name. fileToMsgObject() resolves the ".zlib" variant
            // internally by appending it as a probe suffix, so callers always
            // pass the bare ".dat" name. extension()/stem() below rely on this:
            // a double-extension name would yield suffix ".zlib" and an arch of
            // "<arch>.dat", producing a wrong per-arch mapping path.
            fs::path path(tensileLibPath);
            libraryDirectory = path.parent_path().string();
            suffix           = path.extension().string();

            // Derive the arch from the master library filename, e.g.
            // "TensileLibrary_lazy_gfx1100.dat" -> "gfx1100", and load the
            // matching per-arch mapping file. Per-arch mapping files are
            // required so single-arch shard installs don't collide on a
            // shared "TensileLiteLibrary_lazy_Mapping.dat".
            std::string       stem   = path.stem().string();
            const std::string prefix = "TensileLibrary_lazy_";
            if(stem.compare(0, prefix.size(), prefix) != 0)
            {
                std::cout << "Cannot derive arch from " << tensileLibPath << std::endl;
                return false;
            }
            const std::string arch = stem.substr(prefix.size());
            path                   = fs::path(libraryDirectory)
                   / ("TensileLiteLibrary_lazy_" + arch + "_Mapping.dat");

            libraryMapping = LoadLibraryMapping(path.string());
            if(libraryMapping.empty())
            {
                std::cout << "No library mapping found at " << path.string() << std::endl;
                return false;
            }
            return true;
        }

        void loadLibrary(const int index) const
        {
            // TODO(#7080): point-key + upper_bound misses on above-largest and
            // gap-between-keys; switch to range-encoded mapping.
            auto it = libraryMapping.upper_bound(index);
            if(it == libraryMapping.begin())
            {
                if(Debug::Instance().printDataInit())
                {
                    std::cout << "Index " << index << " not in this arch's mapping range"
                              << std::endl;
                }
                return;
            }
            --it;
            std::string filePrefix = it->second;
            // load the file here directly and push the library for later use.
            {
                std::lock_guard<std::mutex> lock(solutionsGuard);
                if(loadedFiles.find(filePrefix) != loadedFiles.end())
                {
                    return;
                }
            }
            if(Debug::Instance().printDataInit())
            {
                std::cout << "Loading library for index " << index
                          << " from file: " << filePrefix << std::endl;
            }

            fs::path path(libraryDirectory);
            path = path / (filePrefix + suffix);

            auto newLibrary = LoadLibraryFile<MyProblem, MySolution>(path.string());
            auto mLibrary
                = static_cast<MasterSolutionLibrary<MyProblem, MySolution>*>(newLibrary.get());

            using std::begin;
            using std::end;

            // An indexed shard carries no materialized solutions to copy, so
            // stamp the code object name on its cache instead and let it apply
            // as solutions are parsed.
            if(mLibrary->blobCache)
                mLibrary->blobCache->setCodeObjectFilename(filePrefix + ".co");

            std::lock_guard<std::mutex> lock(solutionsGuard);
            if(loadedFiles.find(filePrefix) != loadedFiles.end())
            {
                return;
            }
            // Push to cache
            indexLoadedLibraries[filePrefix] = mLibrary->library;

            // Publish the shard's cache so index lookups can reach solutions
            // this shard owns but has not parsed yet. One entry per shard, not
            // per solution; keyed by prefix so this is idempotent even if a
            // placeholder loads the same shard independently.
            if(mLibrary->blobCache)
                solutionSources.emplace(filePrefix, mLibrary->blobCache);

            std::transform(begin(mLibrary->solutions),
                           end(mLibrary->solutions),
                           std::inserter(solutions, end(solutions)),
                           [this, filePrefix](auto& i) {
                               i.second->codeObjectFilename = filePrefix + ".co";
                               return i;
                           });
            loadedFiles.insert(filePrefix);

            if(Debug::Instance().printCodeObjectInfo())
            {
                std::cout << "load placeholder library " << path << std::endl
                          << mLibrary->solutions.size() << " solutions loaded" << std::endl;
            }
        }

        /// Single point of truth for turning a solution index into an object.
        ///
        /// `solutions` is only fully populated for legacy files. For an indexed
        /// file it starts empty and fills in as queries materialize solutions,
        /// so every index lookup has to be prepared to fall through to this
        /// file's blob cache or to a loaded shard's cache.
        std::shared_ptr<MySolution> resolveSolutionByIndex(const int index) const
        {
            // Idempotent, and a no-op unless this is a lazy-loading master.
            loadLibrary(index);

            {
                std::lock_guard<std::mutex> lock(solutionsGuard);
                auto                        iter = solutions.find(index);
                if(iter != solutions.end())
                    return iter->second;
            }

            std::shared_ptr<MySolution> solution;
            if(blobCache && blobCache->contains(index))
            {
                solution = blobCache->get(index);
            }
            else
            {
                std::map<std::string, std::shared_ptr<SolutionBlobCache<MySolution>>> sources;
                {
                    std::lock_guard<std::mutex> lock(solutionsGuard);
                    sources = solutionSources;
                }
                for(auto const& entry : sources)
                {
                    if(entry.second && entry.second->contains(index))
                    {
                        solution = entry.second->get(index);
                        break;
                    }
                }
            }

            if(!solution)
                return std::shared_ptr<MySolution>();

            // Publish so the workspace fixups below run once per solution and
            // later lookups take the map hit.
            std::lock_guard<std::mutex> lock(solutionsGuard);
            return solutions.emplace(index, solution).first->second;
        }

        /// Parses every solution this library can serve and publishes it into
        /// `solutions`.
        ///
        /// Materializing the caches is not enough on its own: leaf nodes resolve
        /// through a cache and never touch `solutions`, so for an indexed file
        /// that map stays empty however much has been parsed. Enumeration
        /// consumers index it directly, so they need it populated.
        ///
        /// For enumeration paths only. Anything measuring selection latency must
        /// not call this, or it pays the very cost the indexed layout defers.
        void materializeAllSolutions() const
        {
            std::vector<std::shared_ptr<SolutionBlobCache<MySolution>>> sources;
            if(blobCache)
                sources.push_back(blobCache);
            {
                std::lock_guard<std::mutex> lock(solutionsGuard);
                for(auto const& entry : solutionSources)
                    sources.push_back(entry.second);
            }

            for(auto const& source : sources)
            {
                if(!source)
                    continue;

                source->materializeAll();

                // Collected before taking the map lock: get() takes the cache's
                // own lock, and no other path nests these two.
                std::vector<std::pair<int, std::shared_ptr<MySolution>>> parsed;
                parsed.reserve(source->size());
                for(int index : source->indices())
                {
                    if(auto solution = source->get(index))
                        parsed.emplace_back(index, solution);
                }

                std::lock_guard<std::mutex> lock(solutionsGuard);
                for(auto const& entry : parsed)
                    solutions.emplace(entry.first, entry.second);
            }
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            auto solution = resolveSolutionByIndex(index);
            if(!solution)
            {
                return std::shared_ptr<MySolution>();
            }
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
            auto solution = resolveSolutionByIndex(index);
            if(!solution)
            {
                return std::shared_ptr<MySolution>();
            }

            TensileLite::TensorOps nop;

            if(solution->requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
            {
                const auto& pt = solution->problemType;

                bool isComplexInput
                    = (pt.aType == rocisa::DataType::ComplexFloat || pt.aType == rocisa::DataType::ComplexDouble);
                    
                rocisa::DataType alphaBetaType = isComplexInput ? pt.aType : pt.computeType;

                auto problem
                    = MyProblem::createDefaultProblem(solution->problemType.transA,
                                                      solution->problemType.transB,
                                                      solution->problemType.aType,
                                                      solution->problemType.bType,
                                                      solution->problemType.cType,
                                                      solution->problemType.dType,
                                                      alphaBetaType,
                                                      alphaBetaType,
                                                      solution->problemType.computeInputTypeA,
                                                      solution->problemType.computeInputTypeB,
                                                      solution->problemType.computeType,
                                                      1.0,
                                                      1.0,
                                                      solution->problemType.useBias,
                                                      solution->problemType.useGradient,
                                                      solution->problemType.biasDataTypeWhiteList,
                                                      solution->problemType.biasSrcWhiteList,
                                                      solution->problemType.groupedGemm,
                                                      std::numeric_limits<size_t>::max(),
                                                      nop,
                                                      nop,
                                                      nop,
                                                      nop,
                                                      solution->problemType.useGateResidual,
                                                      solution->problemType.gateResidualDataTypeWhiteList);
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
                    // Goes through the resolver: for an indexed library the
                    // forced index will not be in `solutions` yet.
                    auto selected_solution = resolveSolutionByIndex(solution_index);
                    if(!selected_solution)
                    {
                        std::cout << "Solution index " << solution_index
                                  << " not found in this library." << std::endl;
                        return nullptr;
                    }
                    Task task(hardware, problem, *(selected_solution));
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
                lastFindTopRetAll = library->lastFindTopAlreadyRetAll();

                return result;
            }
            else
            {
                const auto& result = library->findTopSolutions(problem, hardware, numSolutions);
                lastFindTopRetAll = library->lastFindTopAlreadyRetAll();

                return result;
            }
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
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

