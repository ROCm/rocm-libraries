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
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/Predicates.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
 * \addtogroup SolutionLibrary
 * @{
 */
    template <typename MyProblem, typename MySolution>
    using LibraryEntry = std::shared_ptr<SolutionLibrary<MyProblem, MySolution>>;
    template <typename MyProblem, typename MySolution, typename MyPredicate>
    using LibraryRow = std::pair<MyPredicate, LibraryEntry<MyProblem, MySolution>>;

    /// What a row of an ExactLogicLibrary selects on, named by its predicate type.
    enum class LibraryRowKind : std::uint8_t
    {
        Other,
        AMDGPU,
        ExperimentalStreamK,
        EqualityMatching,
        RangeMatching,
        GridBasedMatching,
        FreeSizeMatching,
        PredictionMatching,
    };

    template <typename RowPredicate>
    LibraryRowKind libraryRowKind(RowPredicate const& predicate)
    {
        if(!predicate.value)
            return LibraryRowKind::Other;

        std::string const type = predicate.value->type();
        if(type == "AMDGPU")
            return LibraryRowKind::AMDGPU;
        if(type == "ExperimentalStreamK")
            return LibraryRowKind::ExperimentalStreamK;
        if(type == "EqualityMatching")
            return LibraryRowKind::EqualityMatching;
        if(type == "RangeMatching")
            return LibraryRowKind::RangeMatching;
        if(type == "GridBasedMatching")
            return LibraryRowKind::GridBasedMatching;
        if(type == "FreeSizeMatching")
            return LibraryRowKind::FreeSizeMatching;
        if(type == "PredictionMatching")
            return LibraryRowKind::PredictionMatching;
        return LibraryRowKind::Other;
    }

    /**
 * The order in which a findTopSolutions call searches the rows below it while
 * TENSILE_USE_TILEWRIGHT is set, so that a GridBased row of a few kernels does
 * not answer a problem in place of a loaded tilewright model. From the first
 * GridBased row the problem reaches, the search looks ahead to the next
 * Prediction row: when a tilewright model ranks that row's library, the row is
 * searched before the GridBased rows in between, the other rows in between
 * (Equality rows answering their exact sizes, for one) keep their place before
 * it, and the rows after it follow in their order. Without such a model every
 * row keeps its place.
 *
 * The outermost problem-level ExactLogicLibrary of the call makes one and walks
 * its rows up to three times; every ExactLogicLibrary below it asks it, on the
 * same thread, about each row that matches the problem, so that no row that
 * holds kernels is searched twice. The first walk searches the rows in their
 * order up to the first GridBased row, then skips the rows of the kinds that
 * hold kernels (`selects`) up to the next Prediction row: when its model is
 * loaded and only GridBased rows were skipped, the walk searches it and stops;
 * otherwise it stops before it. The second walk searches the rows the first
 * skipped other than GridBased ones, up to and including a Prediction row with
 * a model; the last searches every row not searched yet, in their order. A row
 * of another kind that asked about no row below it holds kernels itself: one
 * that returns solutions while the first walk skips ends the reordering, the
 * walk stopping without them, and the last walk searches every row in its
 * order, that one included.
 */
    class TENSILELITEHOST_EXPORT TilewrightRowOrder
    {
    public:
        enum class Step : std::uint8_t
        {
            Search,
            Skip,
            Stop
        };

        TilewrightRowOrder()
            : m_outer(slot())
        {
            slot() = this;
        }

        ~TilewrightRowOrder()
        {
            slot() = m_outer;
        }

        TilewrightRowOrder(TilewrightRowOrder const&)            = delete;
        TilewrightRowOrder& operator=(TilewrightRowOrder const&) = delete;

        /// The order of the findTopSolutions call this thread is in, or null.
        static TilewrightRowOrder* current()
        {
            return slot();
        }

        /// Whether rows of `kind` hold kernels rather than other rows.
        static bool selects(LibraryRowKind kind)
        {
            switch(kind)
            {
            case LibraryRowKind::EqualityMatching:
            case LibraryRowKind::RangeMatching:
            case LibraryRowKind::GridBasedMatching:
            case LibraryRowKind::FreeSizeMatching:
            case LibraryRowKind::PredictionMatching:
                return true;
            default:
                return false;
            }
        }

        /// What to do with row `row` of `library`, of a kind that `selects`,
        /// which matched the problem; `model`: a tilewright model ranks the
        /// library of this Prediction row.
        Step step(void const* library, size_t row, LibraryRowKind kind, bool model)
        {
            Row const at{library, row};
            m_steps++;
            if(m_walk == Walk::First)
            {
                if(!m_skipping && kind != LibraryRowKind::GridBasedMatching)
                    return search(at, false);
                m_skipping = true;
                if(kind != LibraryRowKind::PredictionMatching)
                {
                    m_skipped.push_back({at, kind});
                    return Step::Skip;
                }
                m_skipping = false;
                if(model)
                {
                    m_model = at;
                    if(std::all_of(m_skipped.begin(), m_skipped.end(), [](Skipped const& s) {
                           return s.kind == LibraryRowKind::GridBasedMatching;
                       }))
                        return search(at, true);
                }
                m_skipped.push_back({at, kind});
                m_stop = true;
                return Step::Stop;
            }
            if(searched(at))
                return Step::Skip;
            if(m_walk == Walk::Lead)
            {
                auto const skipped = std::find_if(m_skipped.begin(),
                                                  m_skipped.end(),
                                                  [&at](Skipped const& s) { return s.at == at; });
                if(skipped == m_skipped.end() || skipped->kind == LibraryRowKind::GridBasedMatching)
                    return Step::Skip;
                return search(at, m_model && *m_model == at);
            }
            return search(at, false);
        }

        /// Whether a walk skips row `row` of `library`, of a kind that does not
        /// `selects`: one searched before.
        bool skips(void const* library, size_t row) const
        {
            return searched(Row{library, row});
        }

        /// Called after searching row `row` of `library`, of a kind that does
        /// not `selects`, when no row below it was asked about (`steps`
        /// unchanged): whether to keep the solutions it returned (`returned`).
        /// While the first walk skips, such a row cannot wait for the Prediction
        /// row ahead: the walk stops without them, and the last walk searches
        /// every row in its order.
        bool keep(void const* library, size_t row, bool returned)
        {
            if(m_skipping && returned)
            {
                m_skipping = false;
                m_stop     = true;
                return false;
            }
            m_searched.push_back(Row{library, row});
            return true;
        }

        /// The number of rows `step` was asked about so far.
        size_t steps() const
        {
            return m_steps;
        }

        /// Whether the walk has stopped: every library returns what it has.
        bool stopped() const
        {
            return m_stop;
        }

        /// Sets up the next walk after one that did not return the solutions
        /// asked for; false when no rows are left.
        bool nextWalk()
        {
            m_stop     = false;
            m_skipping = false;
            if(m_walk == Walk::First && !m_skipped.empty())
            {
                m_walk = m_model && !searched(*m_model) ? Walk::Lead : Walk::Rest;
                return true;
            }
            if(m_walk == Walk::Lead)
            {
                m_walk = Walk::Rest;
                return true;
            }
            return false;
        }

    private:
        enum class Walk : std::uint8_t
        {
            First,
            Lead,
            Rest
        };

        struct Row
        {
            void const* library;
            size_t      row;

            bool operator==(Row const& other) const
            {
                return library == other.library && row == other.row;
            }
        };

        struct Skipped
        {
            Row            at;
            LibraryRowKind kind;
        };

        Step search(Row const& at, bool last)
        {
            m_searched.push_back(at);
            m_stop = m_stop || last;
            return Step::Search;
        }

        bool searched(Row const& at) const
        {
            return std::find(m_searched.begin(), m_searched.end(), at) != m_searched.end();
        }

        static TilewrightRowOrder*& slot();

        TilewrightRowOrder*  m_outer;
        Walk                 m_walk     = Walk::First;
        bool                 m_skipping = false;
        bool                 m_stop     = false;
        size_t               m_steps    = 0;
        std::optional<Row>   m_model;
        std::vector<Row>     m_searched;
        std::vector<Skipped> m_skipped;
    };

    /// Data types of A, B, C, D and the compute input types A and B, as compared
    /// by Predicates::Contraction::TypesEqual.
    using LibraryRowTypes = std::array<rocisa::DataType, 6>;

    /**
 * Which rows of a library match a device, remembered for a few devices, for
 * rows whose predicates read only the fields AMDGPU::operator== compares.
 * Bit i of a mask is row i.
 */
    class LibraryRowsByDevice
    {
    public:
        template <typename Compute>
        uint64_t matches(AMDGPU const& device, Compute&& compute) const
        {
            Device const key{device.processor, device.computeUnitCount, device.pciChipId()};
            int const    known = m_count.load(std::memory_order_acquire);
            for(int i = 0; i < known; i++)
            {
                if(m_entries[i].device == key)
                    return m_entries[i].rows;
            }

            uint64_t const              rows = compute();
            std::lock_guard<std::mutex> lock(m_guard);
            int const                   count = m_count.load(std::memory_order_relaxed);
            for(int i = 0; i < count; i++)
            {
                if(m_entries[i].device == key)
                    return m_entries[i].rows;
            }
            if(count < Capacity)
            {
                m_entries[count] = {key, rows};
                m_count.store(count + 1, std::memory_order_release);
            }
            return rows;
        }

    private:
        struct Device
        {
            AMDGPU::Processor  processor        = AMDGPU::Processor::gfx900;
            int                computeUnitCount = 0;
            std::optional<int> pciChipId;

            bool operator==(Device const& other) const
            {
                return processor == other.processor && computeUnitCount == other.computeUnitCount
                       && pciChipId == other.pciChipId;
            }
        };

        struct Entry
        {
            Device   device;
            uint64_t rows = 0;
        };

        static constexpr int Capacity = 4;

        mutable std::array<Entry, Capacity> m_entries;
        mutable std::atomic<int>            m_count{0};
        mutable std::mutex                  m_guard;
    };

    /**
 * Represents a set of sub-libraries, each with associated predicates. It
 * should be placed in order of best to worst solutions. We assume the best
 * solution is the first one where we match the predicates.
 *
 * Examples: Picking solutions written for a particular GPU, solutions that
 * assume that a particular size is a multiple of something, etc.
 */
    template <typename MyProblem, typename MySolution, typename MyPredicate>
    struct ExactLogicLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
        using Row = LibraryRow<MyProblem, MySolution, MyPredicate>;
        std::vector<Row>          rows;
        mutable std::atomic<bool> lastFindTopRetAll = false;

        ExactLogicLibrary() = default;
        ExactLogicLibrary(std::initializer_list<Row> init)
            : rows(init)
        {
            prepareRows();
        }

        ExactLogicLibrary(std::vector<Row> const& init)
            : rows(init)
        {
            prepareRows();
        }

        /// Caches the kind of every row and, when every row predicate starts
        /// with a TypesEqual term, indexes the rows by those types; when every
        /// row predicate reads only the device, remembers which rows match a
        /// device. `rows` must not change afterwards, except by appending rows,
        /// which then go unindexed.
        void prepareRows()
        {
            rowKinds.clear();
            rowKinds.reserve(rows.size());
            for(auto const& row : rows)
                rowKinds.push_back(libraryRowKind(row.first));

            rowConjunctions.clear();
            if constexpr(!MyPredicate::SelectsHardware)
            {
                rowConjunctions.reserve(rows.size());
                for(auto const& row : rows)
                    rowConjunctions.push_back(
                        dynamic_cast<Predicates::And<MyProblem> const*>(row.first.value.get()));
            }

            rowsSelectDevices = !rows.empty() && rows.size() <= 64
                                && std::all_of(rows.begin(), rows.end(), [](Row const& row) {
                                       return row.first.readsOnlyTheDevice();
                                   });

            rowsByTypes.clear();
            rowsByTypes.reserve(rows.size());
            for(size_t i = 0; i < rows.size(); i++)
            {
                auto const types = rows[i].first.leadingTypes();
                if(!types)
                {
                    rowsByTypes.clear();
                    break;
                }
                rowsByTypes.emplace_back(*types, i);
            }
            std::sort(rowsByTypes.begin(), rowsByTypes.end());
        }

        LibraryRowKind rowKind(size_t row) const
        {
            return row < rowKinds.size() ? rowKinds[row] : libraryRowKind(rows[row].first);
        }

        /// Row `row`'s predicate when it is an And of problem predicates, else null.
        Predicates::And<MyProblem> const* rowConjunction(size_t row) const
        {
            if constexpr(MyPredicate::SelectsHardware)
                return nullptr;
            else if(row < rowConjunctions.size())
                return rowConjunctions[row];
            else
                return dynamic_cast<Predicates::And<MyProblem> const*>(rows[row].first.value.get());
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            std::shared_ptr<MySolution> rv;
            const bool                  streamK = Debug::Instance().useExperimentalSelection() == 2;

            for(size_t i = 0; i < rows.size(); i++)
            {
                auto const& row = rows[i];
                if(rowKind(i) == LibraryRowKind::ExperimentalStreamK && !streamK)
                    continue;

                if(row.first(problem, hardware))
                {
                    rv = row.second->getSolutionByIndex(problem, hardware, index);
                    if(rv)
                        return rv;
                }
            }

            return rv;
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            std::shared_ptr<MySolution> fallbackRv;
            const bool                  streamK = Debug::Instance().useExperimentalSelection() == 2;
            const bool                  rowDebug   = MyPredicate::debugging();
            auto const                  deviceRows = rowsMatchingDevice(hardware, rowDebug);

            // Return exact match immediately; otherwise keep first successful fallback.
            auto visit = [&](size_t i) {
                auto const& row = rows[i];
                if(rowKind(i) == LibraryRowKind::ExperimentalStreamK && !streamK)
                    return false;

                if(!rowMatches(i, problem, hardware, rowDebug, deviceRows))
                    return false;

                bool const equality = rowKind(i) == LibraryRowKind::EqualityMatching;
                std::optional<MatchedPredicateScope> matched;
                if(auto const* conjunction = rowConjunction(i))
                    matched.emplace(conjunction);
                if(!row.first.isFallbackMatch(hardware))
                {
                    auto rv = row.second->findBestSolution(problem, hardware, fitness);

                    if(rv && equality)
                        rv->tag = MySolution::MatchingTag::Equal;

                    if(rv)
                    {
                        if(Debug::Instance().printDeviceSelection())
                        {
                            std::cout << "  Solution found (exact): " << rv->name()
                                      << " [MatchingTag: " << rv->matchingTag() << "]" << std::endl;
                        }
                        fallbackRv = std::move(rv);
                        return true;
                    }
                }
                else if(!fallbackRv)
                {
                    fallbackRv = row.second->findBestSolution(problem, hardware, fitness);

                    if(fallbackRv && equality)
                        fallbackRv->tag = MySolution::MatchingTag::Equal;
                }
                return false;
            };

            if(!forEachCandidateRow(problem, rowDebug, visit))
            {
                if(fallbackRv && Debug::Instance().printDeviceSelection())
                {
                    std::cout << "  Solution found (fallback): " << fallbackRv->name()
                              << " [MatchingTag: " << fallbackRv->matchingTag() << "]" << std::endl;
                }
            }

            return fallbackRv;
        }

    public:

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;
            const bool              debug       = Debug::Instance().printPropertyEvaluation();
            const bool              streamK     = Debug::Instance().useExperimentalSelection() == 2;
            const auto&             excludedLib = Debug::Instance().excludedLibFromGetAll();

            auto amdGPU             = static_cast<AMDGPU const*>(&hardware);
            bool isStandardCUDevice = amdGPU->isStandardCU();

            for(size_t i = 0; i < rows.size(); i++)
            {
                auto const& row  = rows[i];
                auto const  kind = rowKind(i);
                // we want to exclude this lib from getAll. If the Set is not empty,
                // it means we want to skip searched GridBased, Prediction.
                // But if this is a non-standard-CU GPU, we still need to search all for CU-Fallback solutions.
                // In this case, we don't skip the excludedLib.
                if(!excludedLib.empty() && isStandardCUDevice)
                {
                    if(excludedLib.count(row.first.value->type()))
                        continue;
                }

                if(kind == LibraryRowKind::ExperimentalStreamK && !streamK)
                    continue;

                if(kind == LibraryRowKind::AMDGPU && !row.first(problem, hardware))
                    continue;

                auto rowSolutions = row.second->findAllSolutions(problem, hardware, searchType);

                tagSolutions(kind, debug, rowSolutions);

                rv.insert(rowSolutions.begin(), rowSolutions.end());
            }

            return rv;
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            SolutionSet<MySolution> rv;

            for(size_t i = 0; i < rows.size(); i++)
            {
                auto const& row = rows[i];
                if(rowKind(i) == LibraryRowKind::AMDGPU && !row.first(problems[0], hardware))
                    continue;

                auto rowSolutions
                    = row.second->findAllSolutionsGroupedGemm(problems, hardware, searchType);
                rv.insert(rowSolutions.begin(), rowSolutions.end());
            }

            return rv;
        }

        virtual std::string description() const override
        {
            return concatenate(this->type(), " library (", rows.size(), " rows)");
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            bool const          tilewright = Debug::Instance().useTilewright();
            TilewrightRowOrder* order      = tilewright ? TilewrightRowOrder::current() : nullptr;
            if constexpr(!MyPredicate::SelectsHardware)
            {
                if(tilewright && order == nullptr && numSolutions > 0)
                    return findTopSolutionsInTilewrightOrder(problem, hardware, numSolutions);
            }
            return findTopRows(problem, hardware, numSolutions, order);
        }

    private:
        // findTopSolutions of the outermost problem-level library of a call
        // while TENSILE_USE_TILEWRIGHT is set: the walks of its
        // TilewrightRowOrder, each adding what the ones before did not return.
        SolutionVector<MySolution> findTopSolutionsInTilewrightOrder(MyProblem const& problem,
                                                                     Hardware const&  hardware,
                                                                     int numSolutions) const
        {
            TilewrightRowOrder order;
            auto               rv = findTopRows(problem, hardware, numSolutions, &order);
            while(rv.size() < static_cast<size_t>(numSolutions) && order.nextWalk())
            {
                // A tilewright bank pool can hold solutions an earlier walk returned.
                SelectedSolutionsScope selected(rv);
                auto const             more = findTopRows(
                    problem, hardware, numSolutions - static_cast<int>(rv.size()), &order);
                rv.insert(std::end(rv), std::begin(more), std::end(more));
            }
            lastFindTopRetAll = rv.size() < static_cast<size_t>(numSolutions);
            return rv;
        }

        // The rows of the problem in row order, those `order` (when not null)
        // lets this walk search.
        SolutionVector<MySolution> findTopRows(MyProblem const&    problem,
                                               Hardware const&     hardware,
                                               int                 numSolutions,
                                               TilewrightRowOrder* order) const
        {
            SolutionVector<MySolution> rv, solutions;
            Debug const&               settings   = Debug::Instance();
            const bool                 debug      = settings.printPropertyEvaluation();
            const bool                 streamK    = settings.useExperimentalSelection() == 2;
            const bool                 rowDebug   = MyPredicate::debugging();
            auto const                 deviceRows = rowsMatchingDevice(hardware, rowDebug);

            const auto forceDynamic = settings.streamK5ForceMode();
            const bool effectiveDynamic = 
                (forceDynamic == 1) ||
                (forceDynamic != 0 && problem.getParams().streamKTileSchedulingMode() != 0);
            const bool predictionOnly = settings.usePredictionLibrary();
            const bool predictionLib  = predictionOnly || effectiveDynamic;

            // false in case of early return;
            lastFindTopRetAll = false;

            // A tilewright bank pool can hold solutions an earlier row returned.
            std::optional<SelectedSolutionsScope> selected;

            auto visit = [&](size_t i) {
                auto const& row  = rows[i];
                auto const  kind = rowKind(i);
                if(kind == LibraryRowKind::ExperimentalStreamK && !streamK)
                    return false;

                if(predictionLib
                   && (kind == LibraryRowKind::EqualityMatching
                       || kind == LibraryRowKind::RangeMatching))
                    return false;

                // Only an explicit TENSILE_PREDICTION_LIB drops these rows; prediction
                // implied by dynamic Stream-K scheduling still returns their solutions.
                if(predictionOnly
                   && (kind == LibraryRowKind::GridBasedMatching
                       || kind == LibraryRowKind::FreeSizeMatching))
                    return false;

                if(!rowMatches(i, problem, hardware, rowDebug, deviceRows))
                    return false;

                bool const selects = TilewrightRowOrder::selects(kind);
                if(order != nullptr)
                {
                    if(selects)
                    {
                        auto const step = order->step(this,
                                                      i,
                                                      kind,
                                                      kind == LibraryRowKind::PredictionMatching
                                                          && row.second->hasTilewrightModel());
                        if(step != TilewrightRowOrder::Step::Search)
                            return step == TilewrightRowOrder::Step::Stop;
                    }
                    else if(order->skips(this, i))
                        return false;
                }
                size_t const steps = order != nullptr ? order->steps() : 0;

                if(!rv.empty() && !selected)
                    selected.emplace(rv);
                std::optional<MatchedPredicateScope> matched;
                if(auto const* conjunction = rowConjunction(i))
                    matched.emplace(conjunction);
                solutions
                    = row.second->findTopSolutions(problem, hardware, numSolutions - rv.size());

                // hipblaslt_ext::matmulIsTuned() -> rocblaslt_matmul_is_tuned() needs this Equal test
                tagSolutions(kind, debug, solutions);

                if(order != nullptr && !selects && order->steps() == steps
                   && (order->stopped() || !order->keep(this, i, !solutions.empty())))
                    return true;
                rv.insert(std::end(rv), std::begin(solutions), std::end(solutions));
                return rv.size() == numSolutions || (order != nullptr && order->stopped());
            };

            if(forEachCandidateRow(problem, rowDebug, visit))
                return rv;

            // can't reach the requested number, means findTop already done its best
            lastFindTopRetAll = (rv.size() < numSolutions);
            return rv;
        }

    public:
        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            SolutionVector<MySolution> rv, solutions;

            for(auto const& row : rows)
            {
                if(row.first(problems[0], hardware))
                {
                    solutions = row.second->findTopSolutionsGroupedGemm(
                        problems, hardware, numSolutions - rv.size());
                    rv.insert(std::end(rv), std::begin(solutions), std::end(solutions));
                    if(rv.size() == numSolutions)
                        return rv;
                }
            }

            return rv;
        }

    private:
        // Bit i set when row i matches `hardware`; nullopt when the rows are
        // not looked up by device.
        std::optional<uint64_t> rowsMatchingDevice(Hardware const& hardware, bool rowDebug) const
        {
            if constexpr(MyPredicate::SelectsHardware)
            {
                if(!rowsSelectDevices || rowDebug || rowKinds.size() != rows.size())
                    return std::nullopt;
                auto const* device = dynamic_cast<AMDGPU const*>(&hardware);
                if(device == nullptr)
                    return std::nullopt;
                return devices.matches(*device, [&]() {
                    uint64_t matching = 0;
                    for(size_t i = 0; i < rows.size(); i++)
                    {
                        if(rows[i].first.evaluate(hardware, hardware))
                            matching |= uint64_t(1) << i;
                    }
                    return matching;
                });
            }
            return std::nullopt;
        }

        template <typename Any>
        bool rowMatches(size_t                         i,
                        Any const&                     problem,
                        Hardware const&                hardware,
                        bool                           rowDebug,
                        std::optional<uint64_t> const& deviceRows) const
        {
            if(deviceRows)
                return (*deviceRows >> i) & 1;
            return rowDebug ? rows[i].first(problem, hardware)
                            : rows[i].first.evaluate(problem, hardware);
        }

        // Calls visit(i) for the rows that can match `problem`, in row order,
        // until it returns true; returns whether it did. Without predicate
        // debugging, rows indexed under other data types are skipped.
        template <typename Visit>
        bool forEachCandidateRow(MyProblem const& problem, bool rowDebug, Visit&& visit) const
        {
            if(!rowDebug && !rowsByTypes.empty() && rowKinds.size() == rows.size())
            {
                auto const types = MyPredicate::problemTypes(problem);
                auto const first = std::lower_bound(
                    rowsByTypes.begin(), rowsByTypes.end(), std::make_pair(types, size_t(0)));
                for(auto it = first; it != rowsByTypes.end() && it->first == types; ++it)
                {
                    if(visit(it->second))
                        return true;
                }
                return false;
            }

            for(size_t i = 0; i < rows.size(); i++)
            {
                if(visit(i))
                    return true;
            }
            return false;
        }

        template <typename Solutions>
        static void tagSolutions(LibraryRowKind kind, bool debug, Solutions& solutions)
        {
            if(kind == LibraryRowKind::EqualityMatching)
            {
                for(auto& sol : solutions)
                    sol->tag = MySolution::MatchingTag::Equal;
                return;
            }
            // except for Equal, we test others only when debug mode.
            if(!debug)
                return;

            switch(kind)
            {
            case LibraryRowKind::GridBasedMatching:
                for(auto& sol : solutions)
                    sol->tag = MySolution::MatchingTag::GridBased;
                break;
            case LibraryRowKind::RangeMatching:
                for(auto& sol : solutions)
                    sol->tag = MySolution::MatchingTag::Range;
                break;
            case LibraryRowKind::FreeSizeMatching:
                for(auto& sol : solutions)
                    sol->tag = MySolution::MatchingTag::FreeSize;
                break;
            case LibraryRowKind::PredictionMatching:
                for(auto& sol : solutions)
                    sol->tag = MySolution::MatchingTag::Prediction;
                break;
            default:
                break;
            }
        }

        std::vector<LibraryRowKind>                     rowKinds;
        std::vector<Predicates::And<MyProblem> const*>  rowConjunctions;
        std::vector<std::pair<LibraryRowTypes, size_t>> rowsByTypes;
        bool                                            rowsSelectDevices = false;
        LibraryRowsByDevice                             devices;
    };

    struct HardwarePredicate
    {
        std::shared_ptr<Predicates::Predicate<Hardware>> value;

        // The chip IDs this predicate targets, if any. Set by callers that
        // know the chip IDs (e.g. makeHwPred, deserialization). When empty
        // the predicate has no chip-ID constraint and every match is exact.
        std::set<int> targetPciChipIds;

        HardwarePredicate() = default;
        HardwarePredicate(std::shared_ptr<Predicates::Predicate<Hardware>> init,
                          std::optional<int> chipId = std::nullopt)
            : value(init)
        {
            if(chipId)
                targetPciChipIds.insert(chipId.value());
        }

        HardwarePredicate(std::shared_ptr<Predicates::Predicate<Hardware>> init,
                          std::set<int>                                 chipIds)
            : value(init)
            , targetPciChipIds(chipIds)
        {
        }

        template <typename Any>
        bool operator()(Any const& problem, Hardware const& hardware) const
        {
            bool debug = Debug::Instance().printDeviceSelection();
            bool rv    = (*value)(hardware);

            if(debug)
            {
                PredicateDebugger::printHeader(std::cout, "ExactLogic: Hardware");
                value->debugEval(hardware, std::cout);
                PredicateDebugger::printFooter(std::cout, rv);
            }

            return rv;
        }

        static constexpr bool SelectsHardware = true;

        /// Whether operator() prints its evaluation.
        static bool debugging()
        {
            return Debug::Instance().printDeviceSelection();
        }

        /// operator() without the debug output.
        template <typename Any>
        bool evaluate(Any const&, Hardware const& hardware) const
        {
            return (*value)(hardware);
        }

        /// Whether the predicate reads no more of the hardware than the fields
        /// AMDGPU::operator== compares.
        bool readsOnlyTheDevice() const
        {
            return readsOnlyTheDevice(value.get());
        }

        std::optional<LibraryRowTypes> leadingTypes() const
        {
            return std::nullopt;
        }

        template <typename Any>
        static LibraryRowTypes problemTypes(Any const&)
        {
            return {};
        }

        // A match is a fallback when the predicate targets a specific chip ID
        // and the GPU's chip ID is different (operator() already confirmed
        // compatibility via ChipIdRegistry::canUseSolution).
        bool isFallbackMatch(Hardware const& hardware) const
        {
            if(targetPciChipIds.empty())
                return false;

            auto gpuChipId = hardware.pciChipId();
            if(!gpuChipId)
                return false;

            return targetPciChipIds.count(gpuChipId.value()) == 0;
        }

    private:
        template <typename Terms>
        static bool readsOnlyTheDeviceInAll(Terms const& terms)
        {
            return std::all_of(terms.begin(), terms.end(), [](auto const& term) {
                return readsOnlyTheDevice(term.get());
            });
        }

        static bool readsOnlyTheDevice(Predicates::Predicate<Hardware> const* predicate)
        {
            using namespace Predicates;
            if(dynamic_cast<True<Hardware> const*>(predicate)
               || dynamic_cast<False<Hardware> const*>(predicate))
                return true;
            if(auto const* all = dynamic_cast<And<Hardware> const*>(predicate))
                return readsOnlyTheDeviceInAll(all->value);
            if(auto const* any = dynamic_cast<Or<Hardware> const*>(predicate))
                return readsOnlyTheDeviceInAll(any->value);
            if(auto const* negated = dynamic_cast<Not<Hardware> const*>(predicate))
                return readsOnlyTheDevice(negated->value.get());
            if(auto const* gpu = dynamic_cast<IsSubclass<Hardware, AMDGPU> const*>(predicate))
                return readsOnlyTheDevice(gpu->value.get());
            return false;
        }

        static bool readsOnlyTheDevice(Predicates::Predicate<AMDGPU> const* predicate)
        {
            using namespace Predicates;
            if(dynamic_cast<True<AMDGPU> const*>(predicate)
               || dynamic_cast<False<AMDGPU> const*>(predicate)
               || dynamic_cast<GPU::ProcessorEqual const*>(predicate)
               || dynamic_cast<GPU::CUCountEqual const*>(predicate)
               || dynamic_cast<GPU::PciChipIdEqual const*>(predicate)
               || dynamic_cast<GPU::RunsKernelTargeting const*>(predicate))
                return true;
            if(auto const* all = dynamic_cast<And<AMDGPU> const*>(predicate))
                return readsOnlyTheDeviceInAll(all->value);
            if(auto const* any = dynamic_cast<Or<AMDGPU> const*>(predicate))
                return readsOnlyTheDeviceInAll(any->value);
            if(auto const* negated = dynamic_cast<Not<AMDGPU> const*>(predicate))
                return readsOnlyTheDevice(negated->value.get());
            return false;
        }
    };

    template <typename MyProblem, typename MySolution>
    struct HardwareSelectionLibrary
        : public ExactLogicLibrary<MyProblem, MySolution, HardwarePredicate>
    {
        using Base = ExactLogicLibrary<MyProblem, MySolution, HardwarePredicate>;

        HardwareSelectionLibrary() = default;
        HardwareSelectionLibrary(std::initializer_list<typename Base::Row> init)
            : Base(init)
        {
        }

        HardwareSelectionLibrary(std::vector<typename Base::Row> const& init)
            : Base(init)
        {
        }

        static std::string Type()
        {
            return "Hardware";
        }
        virtual std::string type() const override
        {
            return Type();
        }
    };

    template <typename MyProblem>
    struct ProblemPredicate
    {
        std::shared_ptr<Predicates::Predicate<MyProblem>> value;

        ProblemPredicate() = default;
        ProblemPredicate(std::shared_ptr<Predicates::Predicate<MyProblem>> init)
            : value(init)
        {
        }

        bool operator()(MyProblem const& problem, Hardware const& hardware) const
        {
            bool debug = Debug::Instance().printPredicateEvaluation();
            bool rv    = (*value)(problem);

            if(debug)
            {
                PredicateDebugger::printHeader(std::cout, "ExactLogic: Problem");
                value->debugEval(problem, std::cout);
                PredicateDebugger::printFooter(std::cout, rv);
            }

            return rv;
        }

        static constexpr bool SelectsHardware = false;

        /// Whether operator() prints its evaluation.
        static bool debugging()
        {
            return Debug::Instance().printPredicateEvaluation();
        }

        /// operator() without the debug output.
        bool evaluate(MyProblem const& problem, Hardware const&) const
        {
            return (*value)(problem);
        }

        bool readsOnlyTheDevice() const
        {
            return false;
        }

        /// The types an And predicate whose first term is TypesEqual requires;
        /// such a predicate is false for a problem of other types.
        std::optional<LibraryRowTypes> leadingTypes() const
        {
            if constexpr(std::is_same_v<MyProblem, ContractionProblemGemm>)
            {
                auto const* all = dynamic_cast<Predicates::And<MyProblem> const*>(value.get());
                if(all && !all->value.empty())
                {
                    if(auto const* types = dynamic_cast<Predicates::Contraction::TypesEqual const*>(
                           all->value.front().get()))
                        return types->value;
                }
            }
            return std::nullopt;
        }

        static LibraryRowTypes problemTypes(MyProblem const& problem)
        {
            if constexpr(std::is_same_v<MyProblem, ContractionProblemGemm>)
                return {problem.a().dataType(),
                        problem.b().dataType(),
                        problem.c().dataType(),
                        problem.d().dataType(),
                        problem.computeInputTypeA(),
                        problem.computeInputTypeB()};
            else
                return {};
        }

        // Problem predicates never involve hardware fallback.
        bool isFallbackMatch(Hardware const&) const
        {
            return false;
        }
    };

    template <typename MyProblem, typename MySolution>
    struct ProblemSelectionLibrary
        : public ExactLogicLibrary<MyProblem, MySolution, ProblemPredicate<MyProblem>>
    {
        using Base = ExactLogicLibrary<MyProblem, MySolution, ProblemPredicate<MyProblem>>;

        ProblemSelectionLibrary() = default;
        ProblemSelectionLibrary(std::initializer_list<typename Base::Row> init)
            : Base(init)
        {
        }

        ProblemSelectionLibrary(std::vector<typename Base::Row> const& init)
            : Base(init)
        {
        }

        static std::string Type()
        {
            return "Problem";
        }
        virtual std::string type() const
        {
            return Type();
        }
    };

    /**
 * @}
 */

} // namespace TensileLite

