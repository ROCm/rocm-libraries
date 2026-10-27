/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include <Tensile/CachingLibrary.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PredictionLibrary.hpp>
#include <Tensile/SolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <tensilelitehost/export.h>

namespace TensileLite
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
        gfx940,
        gfx941,
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
        gfx1152,
        gfx1153,
        gfx1170,
        gfx1171,
        gfx1200,
        gfx1201,
        gfx1250,
        gfx1250_strict,
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
        case LazyLoadingInit::gfx940:
            return "TensileLibrary_*_gfx940";
        case LazyLoadingInit::gfx941:
            return "TensileLibrary_*_gfx941";
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
        case LazyLoadingInit::gfx1152:
            return "TensileLibrary_*_gfx1152";
        case LazyLoadingInit::gfx1153:
            return "TensileLibrary_*_gfx1153";
        case LazyLoadingInit::gfx1170:
            return "TensileLibrary_*_gfx1170";
        case LazyLoadingInit::gfx1171:
            return "TensileLibrary_*_gfx1171";
        case LazyLoadingInit::gfx1200:
            return "TensileLibrary_*_gfx1200";
        case LazyLoadingInit::gfx1201:
            return "TensileLibrary_*_gfx1201";
        case LazyLoadingInit::gfx1250:
            return "TensileLibrary_*_gfx1250";
        // These two share an ISA but not an ELF machine code, so neither may
        // preload the other's libraries. Callers match this whole-string, so the
        // pattern above stops at the bare name and this one requires the suffix.
        case LazyLoadingInit::gfx1250_strict:
            return "TensileLibrary_*_gfx1250-strict";
        case LazyLoadingInit::None:
            return "";
        }
        return "";
    }

    /// `filePrefix` without the CU-count and chip-ID parts the library build puts
    /// before the architecture: library files with equal keys hold the same
    /// problem type for the same architecture.
    inline std::string libraryProblemTypeKey(std::string const& filePrefix)
    {
        auto const archPos = filePrefix.rfind('_');
        if(archPos == std::string::npos)
            return filePrefix;

        std::string head = filePrefix.substr(0, archPos);
        auto        drop = [&head](std::string const& tag, bool hex) {
            auto const pos = head.rfind('_');
            if(pos == std::string::npos)
                return;
            auto const part = head.substr(pos + 1);
            if(part.size() <= tag.size() || part.compare(0, tag.size(), tag) != 0)
                return;
            for(size_t i = tag.size(); i < part.size(); i++)
            {
                unsigned char const c = part[i];
                if(!(hex ? (std::isxdigit(c) || c == '-') : std::isdigit(c)))
                    return;
            }
            head.resize(pos);
        };
        drop("ID", true);
        drop("CU", false);
        return head + filePrefix.substr(archPos);
    }

    // Calls visit(library, kind) for the library of every row of the Problem
    // library `library` is (rows that are Problem libraries themselves are
    // descended into instead).
    template <typename MyProblem, typename MySolution, typename Visit>
    void forEachProblemRow(std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> const& library,
                           Visit&&                                                        visit)
    {
        auto const* problemRows
            = dynamic_cast<ProblemSelectionLibrary<MyProblem, MySolution> const*>(
                withoutCache(library).get());
        if(problemRows == nullptr)
            return;

        for(size_t i = 0; i < problemRows->rows.size(); i++)
        {
            auto const& rowLibrary = problemRows->rows[i].second;
            if(dynamic_cast<ProblemSelectionLibrary<MyProblem, MySolution> const*>(
                   withoutCache(rowLibrary).get()))
                forEachProblemRow(rowLibrary, visit);
            else
                visit(rowLibrary, problemRows->rowKind(i));
        }
    }

    template <typename MyProblem, typename MySolution>
    struct PlaceholderLibrary;

    /// The placeholders of one master library, by file prefix.
    template <typename MyProblem, typename MySolution>
    struct PlaceholderRegistry
    {
        std::map<std::string, PlaceholderLibrary<MyProblem, MySolution> const*> byPrefix;

        /// The libraries of the Equality rows of the files whose problem-type key
        /// is that of `filePrefix`, in file-prefix order. Loads those files.
        std::vector<std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>>
            equalityLibraries(std::string const& filePrefix) const
        {
            std::vector<std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>> libraries;
            auto const key = libraryProblemTypeKey(filePrefix);
            for(auto const& entry : byPrefix)
            {
                if(libraryProblemTypeKey(entry.first) != key)
                    continue;
                forEachProblemRow(entry.second->loadedLibrary(),
                                  [&libraries](auto const& rowLibrary, LibraryRowKind kind) {
                                      if(kind == LibraryRowKind::EqualityMatching)
                                          libraries.push_back(rowLibrary);
                                  });
            }
            return libraries;
        }
    };

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    struct PlaceholderLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        mutable std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> library;
        mutable std::set<std::string>*                                  loadedFiles;
        mutable SolutionMap<MySolution>*                                masterSolutions;
        mutable std::mutex*                                             solutionsGuard;
        mutable std::mutex                                              lazyLoadingGuard;
        std::string                                                     filePrefix;

        // Where to publish an indexed shard's blob cache, keyed by file prefix,
        // so the owning master can resolve indices this shard has not parsed
        // yet. Without it the shard loads but its solutions stay unreachable by
        // index.
        mutable std::map<std::string, std::shared_ptr<SolutionBlobCache<MySolution>>>*
            solutionSources = nullptr;
        std::string                                                     suffix;
        std::string                                                     libraryDirectory;
        mutable std::atomic<bool>                                       lastFindTopRetAll = false;

        mutable std::map<std::string, std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>>*
            indexLoadedLibraries;

        // The registry of the master this placeholder belongs to.
        std::shared_ptr<PlaceholderRegistry<MyProblem, MySolution>> placeholders;

        PlaceholderLibrary() = default;

        bool loadPlaceholderLibrary() const
        {
            std::lock_guard<std::mutex> lock(lazyLoadingGuard);
            // If condition in case two threads got into this function
            if(!library)
            {
                std::string path       = (libraryDirectory + "/" + filePrefix + suffix).c_str();
                auto        newLibrary = LoadLibraryFile<MyProblem, MySolution>(path);
                auto        mLibrary
                    = static_cast<MasterSolutionLibrary<MyProblem, MySolution>*>(newLibrary.get());
                // Results are cached by the master that reaches this placeholder;
                // the shard's own cache would only hold the same entries again.
                auto shard = withoutCache(mLibrary->library);

                // Indexed shards publish solutions later; stamp the name on
                // the cache so it is applied as each one is parsed.
                if(mLibrary->blobCache)
                    mLibrary->blobCache->setCodeObjectFilename(getCodeObjectFileName());

                std::lock_guard<std::mutex> lock(*solutionsGuard);

                // Publishes this shard's cache so the owning master can resolve
                // indices the shard has not parsed yet. Keyed by prefix, so a
                // shard the master already published is not registered twice --
                // two caches for one shard would retain two copies of its blob
                // and hand out two objects for the same solution index.
                auto publishShardCache = [&]() {
                    if(mLibrary->blobCache && solutionSources != nullptr)
                        solutionSources->emplace(filePrefix, mLibrary->blobCache);
                };

                if(loadedFiles->find(filePrefix) != loadedFiles->end())
                {
                    if(indexLoadedLibraries->find(filePrefix) == indexLoadedLibraries->end())
                    {
                        // Keeping this shard's own tree, so its cache is the one
                        // the leaves reference.
                        publishShardCache();
                        for(auto const& entry : mLibrary->solutions)
                            entry.second->codeObjectFilename = getCodeObjectFileName();
                        adopt(std::move(shard));
                        return true;
                    }
                    // Adopting the tree the master loaded means adopting the
                    // cache the master already published; ours is dropped.
                    adopt(withoutCache((*indexLoadedLibraries)[filePrefix]));
                    indexLoadedLibraries->erase(filePrefix);

                    // If the library is already loaded, just return it
                    if(Debug::Instance().printCodeObjectInfo())
                        std::cout << "load placeholder library " << filePrefix
                                  << " from MasterLibrary cache" << std::endl;
                    return true;
                }

                publishShardCache();

                using std::begin;
                using std::end;

                std::transform(begin(mLibrary->solutions),
                               end(mLibrary->solutions),
                               std::inserter(*masterSolutions, end(*masterSolutions)),
                               [this](auto& i) {
                                   i.second->codeObjectFilename = getCodeObjectFileName();
                                   return i;
                               });
                loadedFiles->insert(filePrefix);

                if(Debug::Instance().printCodeObjectInfo())
                    std::cout << "load placeholder library " << path << std::endl
                              << mLibrary->solutions.size() << " solutions loaded" << std::endl;

                adopt(std::move(shard));
                return mLibrary;
            }

            return false;
        }

        /// The library this placeholder stands for, loaded first if it is not yet.
        std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> loadedLibrary() const
        {
            loadPlaceholderLibrary();
            std::lock_guard<std::mutex> lock(lazyLoadingGuard);
            return library;
        }

    private:
        // Publishes `shard` once the bank-mode Prediction libraries in it can
        // reach the Equality rows of their problem type.
        void adopt(std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> shard) const
        {
            if(placeholders && Debug::Instance().useTilewright())
            {
                forEachProblemRow(shard, [this](auto const& rowLibrary, LibraryRowKind) {
                    auto prediction = std::dynamic_pointer_cast<
                        ProblemPredictionLibrary<MyProblem, MySolution>>(rowLibrary);
                    if(prediction && prediction->tilewright_bank)
                        prediction->tilewright_bank_sources
                            = [registry = placeholders, prefix = filePrefix]() {
                                  return registry->equalityLibraries(prefix);
                              };
                });
            }
            library = std::move(shard);
        }

    public:
        std::string getCodeObjectFileName() const
        {
            return filePrefix + ".co";
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            if(!library)
                loadPlaceholderLibrary();

            return library->getSolutionByIndex(problem, hardware, index);
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            if(!library)
                loadPlaceholderLibrary();

            return library->findBestSolution(problem, hardware, fitness);
        }

        /**
         * Returns all `Solution` objects that are capable of correctly solving this
         * `problem` on this `hardware`.
         *
         * May return an empty set if no such object exists.
         */
        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(!library)
            {
                loadPlaceholderLibrary();
            }

            return library->findAllSolutions(problem, hardware, searchType);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            if(!library)
            {
                loadPlaceholderLibrary();
            }

            return library->findAllSolutionsGroupedGemm(problems, hardware, searchType);
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            if(!library)
            {
                loadPlaceholderLibrary();
            }

            auto solutions = library->findTopSolutions(problem, hardware, numSolutions);

            // can't reach the requested number, means findTop already done its best
            lastFindTopRetAll = (solutions.size() < numSolutions);
            return solutions;
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
            if(!library)
            {
                loadPlaceholderLibrary();
            }

            return library->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
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

        // Get the library filename for debug output
        virtual std::string getLibraryFileName() const override
        {
            return filePrefix + suffix;
        }
    };

} // namespace TensileLite

