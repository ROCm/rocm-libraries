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
     * "Param scorer" for ProblemPredictionLibrary.
     *
     * A model is a msgpack file (small MLP over problem features with one
     * softmax head per kernel parameter). Each catalog solution is scored by
     * sum_h w_h * log p_h(value of parameter h), and the Prediction library
     * walks solutions in that order through the usual predicate check: the
     * next-best kernel is the fallback. Each Prediction library binds the
     * model whose problem_key and row count match it (one model per type).
     *
     * Models are loaded once per process from
     *   TENSILE_PARAM_MODEL=<file>[:<file>...] | <dir>   explicit override
     *   TENSILE_PARAM_MODEL=none                         load nothing
     *   otherwise <Tensile library dir>/ParamModel/*.msgpack (setDefaultModelDir)
     *
     * An arch whose library dir has a ParamModel/ directory is "param-only":
     * its Prediction libraries never use the Origami ranking. A library with
     * no bound model walks its catalog in libraryLogicIndex order (row 0 is a
     * robust kernel). Other arches keep the stock Origami behaviour.
     *
     * TENSILE_PARAM_TOPK=<k>      k > 1: rerank the top k by origami::rank_configs
     *                             (unset: model topk_default; 0 or 1: pure param order)
     * TENSILE_PARAM_DEBUG=1       print the decision per findTopSolutions call to stderr
     * TENSILE_PARAM_SELFTEST=1    verify the model's test_vectors at load, abort on mismatch
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

        /**
         * Directory searched for *.msgpack models when TENSILE_PARAM_MODEL is unset
         * (rocblaslt passes <Tensile library dir>/ParamModel before the library loads).
         * Ignored once the models have been loaded.
         */
        TENSILELITEHOST_EXPORT void setDefaultModelDir(std::string const& dir);

        //! All models, loaded once (see above); empty when none or all invalid.
        TENSILELITEHOST_EXPORT std::vector<Model const*> const& models();

        //! The default model directory exists: Prediction libraries never use Origami.
        TENSILELITEHOST_EXPORT bool paramOnly();

        //! The model whose problem key and row count fit this library, or nullptr.
        TENSILELITEHOST_EXPORT Model const* modelFor(std::vector<SolutionInfo> const& solutions);

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
         */
        TENSILELITEHOST_EXPORT std::vector<int> rankRows(Model const&         model,
                                                         size_t               m,
                                                         size_t               n,
                                                         size_t               k,
                                                         size_t               b,
                                                         std::vector<double>* scores = nullptr);

        //! Top-k for the Origami rerank (TENSILE_PARAM_TOPK, else model topk_default).
        TENSILELITEHOST_EXPORT int topK(Model const& model);

        //! TENSILE_PARAM_DEBUG=1.
        TENSILELITEHOST_EXPORT bool debug();
    } // namespace ParamScorer
} // namespace TensileLite
