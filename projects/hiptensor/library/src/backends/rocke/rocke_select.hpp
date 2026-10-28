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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Kernel selection of the rocKE backend at plan creation, after the capability predicates
// (enumerateCandidates admits only candidates the pack can run, without launching anything):
//
//   1. rocKE's tuned route table: a candidate from it (an exact problem rocKE measured) wins;
//      otherwise the ring GETT (rocKE's native contraction body reading the operands in place)
//      wins when the problem has one, and on a target without the GEMM routes the direct GETT
//      of a 16-bit or f32 compute class. The routes the models were not trained on (the ring
//      GETT, the direct GETT) are not scored: they follow the scored candidates in the static
//      order;
//   2. the selection model of the problem's cell in the pack: gradient-boosted trees that
//      predict each candidate's slowdown against the best candidate (log2) from integer
//      features of the problem and the candidate; the lowest prediction wins, ties by the
//      static order;
//   3. the static order (rocKE's body and class rules) when the pack has no model for the cell
//      or the model does not load.
//
// No timing, no allocation and no device call. Features are integers (log2 values in 1/256
// steps), so the selection is the same on every host for the same problem, device and pack.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_SELECT_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_SELECT_HPP

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rocke_plan.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        // Version of the feature schema. A model names the schema it was trained on and the
        // features it reads; a model of another schema is not used.
        inline constexpr int kSelectSchema = 1;

        // log2(max(x, 1)) in 1/256 steps, exact integer arithmetic.
        int64_t log2Q8(uint64_t x);

        // The names of the features, in the order features() writes them.
        std::vector<char const*> const& featureNames();

        // The feature vector of a candidate of `problem`.
        void features(backend::Problem const& problem,
                      Candidate const&        candidate,
                      std::vector<int64_t>&   out);

        // The model cells a problem's selection looks up, most specific first (route table
        // keys select.model.<cell>). Empty for operations without a model (elementwise).
        std::vector<std::string> selectionCells(backend::Problem const& problem);

        enum class SelectSource : uint8_t
        {
            Static,
            Model,
            Tuned,
            Rule, // a candidate the rules put first (the ring GETT, the direct GETT)
        };

        char const* sourceName(SelectSource source);

        struct Selection
        {
            // candidate indices, best first
            std::vector<int> order;
            // per candidate: the model's prediction (log2 slowdown), when a model scored it
            std::vector<double> scores;
            SelectSource        source = SelectSource::Static;
            // the model used: "<cell>@<id>", or why none was used
            std::string model;
        };

        // Orders the candidates (see the top of this file). `useModel` false skips step 2.
        Selection rankCandidates(backend::Problem const&       problem,
                                 Pack const&                   pack,
                                 std::vector<Candidate> const& candidates,
                                 bool                          useModel);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_SELECT_HPP
