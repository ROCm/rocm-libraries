/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <origami/nn/boost/library_stamp.hpp>
#include <origami/nn/boost/types.hpp>
#include <origami/nn/types.hpp>

#include <Tensile/ContractionProblem_fwd.hpp>
#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/Utils.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
     * Origami Boost (origami/nn/boost): a learned correction that re-orders Origami's top
     * candidates in ProblemPredictionLibrary, as the boost backend of origami::rank_configs
     * (inference_mode_t::analytical_rerank).
     *
     * Off unless HIPBLASLT_ORIGAMI_BOOST=1. A prediction row's model is the one origami_nn_index
     * in its library directory maps the row's logic stem to (backend boost), loaded when the row
     * is deserialized, once per model file, and used only if its stamps match this build and
     * its manifest diff accepts that library (libraryManifest, checkLibrary): kernels may be
     * removed, and added when their gated structural values were seen in training. Dtype
     * classes: "bf16" (A and B BFloat16), "tf32" (A/B/C/D Float with F32XdlMathOp XFloat32,
     * i.e. compute type f32_fast_xf32 / c_xf32_r) and "fp8" (plain F8BS: A/B Float8 with Float8
     * compute input, C/D BFloat16, A/B scales unset or scalar, no alpha vector, MX blocks or
     * swizzled operands). Strict f32, every other dtype, other FP8 problems and every call the
     * model declines keep Origami's order.
     *
     * Tuned entries take precedence: Equality and Range rows are walked before the
     * PredictionMatching row (ExactLogicLibrary), their picks are returned unchanged, and the
     * model only re-orders the prediction row's own candidates.
     */
    namespace OrigamiBoost
    {
        using Logger = std::function<void(std::string const&)>;

        /// HIPBLASLT_ORIGAMI_BOOST=1 (read once).
        TENSILELITEHOST_EXPORT bool enabled();

        /// The dtype classes a model can be loaded for.
        TENSILELITEHOST_EXPORT std::vector<std::string> const& dtypes();

        /// Dtype class of the logic files named TensileLibrary_<BB|SS|F8F8>_*: "bf16", "tf32",
        /// "fp8" or null.
        TENSILELITEHOST_EXPORT char const* logicClass(std::string const& logicStem);

        /// Dtype class of a problem: "bf16", "tf32", "fp8" or null.
        TENSILELITEHOST_EXPORT char const* problemClass(ContractionProblemGemm const& problem);

        /**
         * The boost model origami_nn_index in @p libdir maps @p logicStem to
         * (origami::nn::index_weights_path), loaded with origami::nn::load_boost_model for the
         * stem's arch and dtype class and the kernel library in @p libdir. Each model file is
         * loaded once; its outcome goes to the load loggers.
         * @return origami::nn::invalid_handle if the stem has no model or it is refused.
         */
        TENSILELITEHOST_EXPORT origami::nn::model_handle_t
                               loadForLogic(std::string const& logicStem, std::string const& libdir);

        /**
         * Canonical manifest of the PredictionMatching rows of libdir for a dtype class: the
         * files TensileLibrary_<BB|SS|F8F8>_*<arch>*.dat[.zlib] (not the lazy master), loaded with
         * the regular deserializer; their prediction rows' kernels whose problem type is in the
         * class, as origami::nn::boost::canonical_manifest. The model exporter's library check
         * reads the same files and rows.
         * @return false with @p error set if a file cannot be read or nothing is found.
         */
        TENSILELITEHOST_EXPORT bool
            libraryManifest(std::string const&                                 libdir,
                            std::string const&                                 arch,
                            std::string const&                                 dtype,
                            std::vector<origami::nn::boost::manifest_entry_t>& out,
                            std::string&                                       error);

        /**
         * The kernel-library part of loading the model at @p modelPath: the diff of libdir
         * against the model's training manifest (the model exporter's library check reports the
         * same).
         * @return false with @p error set if the model or the library cannot be read.
         */
        TENSILELITEHOST_EXPORT bool checkLibrary(std::string const&                   libdir,
                                                 std::string const&                   arch,
                                                 std::string const&                   modelPath,
                                                 origami::nn::boost::library_diff_t& diff,
                                                 std::string&                         error);

        /**
         * The Stream-K launch of @p solution on @p problem, as the model's streamk_launch
         * columns read it: solution.computeStreamKDecisions(problem, hardware), the snapshot
         * solve() takes its Stream-K launch from (same reduction, grid and workspace-fallback
         * helpers as resolveStreamKSettings), plus problem.getItersPerTile and the kernel's
         * sizeMapping values. The problem is the one hipBLASLt builds for the call, so its CU
         * budget (smCountTarget), workspace size, Stream-K tile-scheduling mode and
         * uniform-summation-order flag are those of the call; the hardware is the device's,
         * with its TENSILE_* persistent-grid settings.
         *
         * The launch takes the same decision as long as it runs this problem on this device
         * with at least the workspace the problem carries here. hipBLASLt sets that to the
         * heuristic's maximum workspace (HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES) and launches
         * with the workspace passed to the matmul call: a smaller launch workspace can move the
         * grid (Origami's grid selection reads it) or trigger the workspace fallback to a
         * data-parallel grid, which the columns did not see.
         *
         * @return valid = false if the device is not an AMDGPU or the decision throws.
         */
        TENSILELITEHOST_EXPORT origami::nn::boost::streamk_launch_t
                               streamKLaunch(ContractionSolution const&    solution,
                                             ContractionProblemGemm const& problem,
                                             Hardware const&               hardware);

        /// Per entry of a prediction row's solution list, the position of the first entry with
        /// the same solutionName.
        template <typename SolutionList>
        std::vector<std::uint32_t> solutionNameIds(SolutionList const& solutions)
        {
            std::unordered_map<std::string_view, std::uint32_t> first;
            std::vector<std::uint32_t>                          ids(solutions.size());
            for(std::size_t i = 0; i < solutions.size(); ++i)
                ids[i] = first.emplace(solutions[i].second->solutionName,
                                       static_cast<std::uint32_t>(i))
                             .first->second;
            return ids;
        }

        /// Sinks for model loads (accepted, refused) and per-call outcomes; empty disables them.
        TENSILELITEHOST_EXPORT void          setLoadLoggers(Logger accepted, Logger refused);
        TENSILELITEHOST_EXPORT void          setCallLogger(Logger logger);
        TENSILELITEHOST_EXPORT Logger const& callLogger();

        /**
         * The caller's side of the re-rank in ProblemPredictionLibrary::findTopSolutions
         * (origami::nn::boost::rerank_source_t). prepare() applies the row's own adjustment of
         * the analytical order (@p adjust) before the re-rank sees it. A row config is a
         * candidate if its solution passes the row's predicates (@p matches) and is the first of
         * its name; its kernel fields are its sizeMapping's NumLoadsCoalesced and its Stream-K
         * launch on this problem. The predicate verdicts it takes are kept for the row's walk
         * (verdict()).
         */
        template <typename MyProblem, typename MySolution, typename Matches, typename Adjust>
        class RerankSource final : public origami::nn::boost::rerank_source_t
        {
        public:
            using SolutionList = std::vector<std::pair<int, std::shared_ptr<MySolution>>>;

            RerankSource(MyProblem const&                  problem,
                         Hardware const&                   hardware,
                         SolutionList const&               solutions,
                         std::vector<std::uint32_t> const& nameIds,
                         Matches const&                    matches,
                         Adjust const&                     adjust)
                : m_problem(problem)
                , m_hardware(hardware)
                , m_solutions(solutions)
                , m_nameIds(nameIds)
                , m_matches(matches)
                , m_adjust(adjust)
            {
            }

            /// Starts a re-rank of @p origamiProblem: candidate() and verdict() use the thread's
            /// scratch from here on.
            void activate(origami::problem_t const& origamiProblem)
            {
                m_origamiProblem = &origamiProblem;
                verdicts().clear();
                accepted().clear();
                acceptedNames().clear();
            }

            void prepare(std::vector<origami::prediction_result_t>& results) override
            {
                m_adjust(results);
                m_prepared = true;
            }

            /// Whether prepare() ran, i.e. the results carry the row's adjustment.
            bool prepared() const
            {
                return m_prepared;
            }

            bool candidate(origami::config_t const&              config,
                           origami::nn::boost::kernel_fields_t& fields) override
            {
                if(config.index >= m_solutions.size())
                    return false;
                auto const& solution = m_solutions[config.index].second;
                const bool  passed   = m_matches(solution);
                verdicts().emplace_back(solution.get(), passed);
                if(!passed)
                    return false;
                // By the row's name ids when it has them (solutionNameIds), else by name.
                const bool          byId = m_nameIds.size() == m_solutions.size();
                const std::uint32_t name = byId ? m_nameIds[config.index] : 0;
                for(std::size_t i = 0; i < accepted().size(); ++i)
                    if(byId ? acceptedNames()[i] == name
                            : accepted()[i]->solutionName == solution->solutionName)
                        return false;
                accepted().push_back(solution.get());
                acceptedNames().push_back(name);
                fields.num_loads_coalesced_a = solution->sizeMapping.NumLoadsCoalescedA;
                fields.num_loads_coalesced_b = solution->sizeMapping.NumLoadsCoalescedB;
                fields.streamk               = streamKLaunch(*solution, m_problem, m_hardware);
                return true;
            }

            void outcome(origami::nn::boost::status_t status,
                         std::string const&           detail,
                         std::vector<size_t> const&   order) override
            {
                auto const& log = callLogger();
                if(!log)
                    return;
                auto const& p = *m_origamiProblem;
                if(status != origami::nn::boost::status_t::ok)
                {
                    log(concatenate("Origami Boost fallback (",
                                    origami::nn::boost::to_string(status),
                                    detail.empty() ? "" : ": ",
                                    detail,
                                    ") M=",
                                    p.size.m,
                                    " N=",
                                    p.size.n,
                                    " K=",
                                    p.size.k,
                                    " batch=",
                                    p.batch));
                    return;
                }
                log(concatenate("Origami Boost ok M=",
                                p.size.m,
                                " N=",
                                p.size.n,
                                " K=",
                                p.size.k,
                                " batch=",
                                p.batch,
                                ": top-1 ",
                                accepted()[0]->name(),
                                " -> ",
                                accepted()[order[0]]->name(),
                                " (Origami rank ",
                                order[0],
                                ")"));
            }

            /// 1 or 0 if candidate() took @p solution's predicate verdict, else -1.
            int verdict(MySolution const* solution) const
            {
                if(m_origamiProblem == nullptr)
                    return -1;
                for(auto const& v : verdicts())
                    if(v.first == solution)
                        return v.second ? 1 : 0;
                return -1;
            }

        private:
            static std::vector<std::pair<MySolution const*, bool>>& verdicts()
            {
                thread_local std::vector<std::pair<MySolution const*, bool>> v;
                return v;
            }
            static std::vector<MySolution const*>& accepted()
            {
                thread_local std::vector<MySolution const*> a;
                return a;
            }
            static std::vector<std::uint32_t>& acceptedNames()
            {
                thread_local std::vector<std::uint32_t> n;
                return n;
            }

            MyProblem const&                  m_problem;
            Hardware const&                   m_hardware;
            SolutionList const&               m_solutions;
            std::vector<std::uint32_t> const& m_nameIds;
            Matches const&                    m_matches;
            Adjust const&                     m_adjust;
            origami::problem_t const*         m_origamiProblem = nullptr;
            bool                              m_prepared       = false;
        };
    } // namespace OrigamiBoost
} // namespace TensileLite
