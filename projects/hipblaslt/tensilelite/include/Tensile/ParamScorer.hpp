// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /**
     * Optional, environment-gated "param scorer" for ProblemPredictionLibrary.
     *
     * TENSILE_PARAM_MODEL=<file> names a msgpack model (small MLP over problem
     * features with one softmax head per kernel parameter). Each catalog
     * solution is scored by sum_h w_h * log p_h(value of parameter h), and the
     * Prediction library walks solutions in that order instead of the Origami
     * ranking. With the variable unset nothing here is loaded or called.
     *
     * TENSILE_PARAM_TOPK=<k>      k > 1: rerank the top k by origami::rank_configs
     *                             (unset: model topk_default; 0 or 1: pure param order)
     * TENSILE_PARAM_DEBUG=1       print the decision per findTopSolutions call to stderr
     * TENSILE_PARAM_SELFTEST=1    verify the model's test_vectors at load, abort on mismatch
     *
     * Model key "math": "f32" runs the MLP in float (log-softmax and scores stay
     * double); absent or "f64" keeps double math.
     */
    namespace ParamScorer
    {
        struct Model;

        //! What the binding gate needs to know about one Prediction solution.
        struct SolutionInfo
        {
            int32_t     libraryLogicIndex = -1;
            std::string name;
            size_t      mt0 = 0, mt1 = 0, depthU = 0;
            int         mi0 = 0, mi1 = 0;
            size_t      wg0 = 0, wg1 = 0, wg2 = 0;
            int         wgm = 0;
            int         ldsb = 0;
            int         pgr = 0;
        };

        //! Model loaded once from TENSILE_PARAM_MODEL; nullptr when unset or invalid.
        TENSILELITEHOST_EXPORT Model const* activeModel();

        /**
         * Bind a Prediction library (its solution_list in order) to the model.
         * Returns the model row of every solution_list position, or an empty
         * vector when the model does not belong to this library. Prints one
         * stderr line stating ACTIVE or INACTIVE (+ reason).
         */
        TENSILELITEHOST_EXPORT std::vector<int>
            bindLibrary(Model const& model, std::vector<SolutionInfo> const& solutions);

        /**
         * Model rows ordered by score (descending; ties by ascending row).
         * If scores != nullptr it receives the score of every row.
         * sortedPrefix < row count: only the first sortedPrefix entries are in
         * that order (the exact top rows); the rest are the remaining rows in an
         * unspecified order -- call sortRowsTail before using them.
         */
        TENSILELITEHOST_EXPORT std::vector<int> rankRows(Model const&         model,
                                                         size_t               m,
                                                         size_t               n,
                                                         size_t               k,
                                                         size_t               b,
                                                         std::vector<double>* scores = nullptr,
                                                         size_t sortedPrefix = size_t(-1));

        //! Sort rows[from..] by (score descending, row ascending), completing rankRows.
        TENSILELITEHOST_EXPORT void sortRowsTail(std::vector<double> const& scores,
                                                 std::vector<int>&          rows,
                                                 size_t                     from);

        //! Top-k for the Origami rerank (TENSILE_PARAM_TOPK, else model topk_default).
        TENSILELITEHOST_EXPORT int topK(Model const& model);

        //! How many rows rankRows should sort up front for the Prediction walk: max(16, topK).
        TENSILELITEHOST_EXPORT size_t sortedPrefix(Model const& model);

        //! TENSILE_PARAM_DEBUG=1.
        TENSILELITEHOST_EXPORT bool debug();
    } // namespace ParamScorer
} // namespace TensileLite
