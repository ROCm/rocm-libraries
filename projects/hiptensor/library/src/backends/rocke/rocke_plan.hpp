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

// Plan-time construction of the rocKE backend's launch programs: canonical problems, the
// candidate programs of a problem (one per kernel or composite of kernels the pack can run it
// with), kernel arguments, launch grids and workspace layout. Host only.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_PLAN_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_PLAN_HPP

#include <functional>
#include <string>
#include <vector>

#include "../../backend/problem.hpp"
#include "rocke_runtime.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        // Strategy of a candidate program.
        enum class Route : uint8_t
        {
            Elementwise = 0,
            Reduction   = 1,
            // contraction: a launch record of rocKE's tuned GETT route table
            TunedGett = 2,
            // contraction: pack -> universal GEMM with an f32 output -> epilogue; operands f16 or
            // bf16 (H16), f32 split into bf16 parts (X6: the x3 / x4 / x6 / x9 recipes), complex
            // f32 (CF32, the same recipes)
            H16  = 3,
            X6   = 4,
            CF32 = 5,
            // contraction: pack -> f64 / complex f64 strided GEMM with runtime sizes (-> epilogue
            // unless it writes D directly); also the F64 route of the other data types
            Gett64 = 6,
            // contraction: rocKE's ring GETT body with runtime index maps (f16 / bf16, gfx950):
            // the operands read and D written in place, one launch, no workspace
            RingGett = 7,
            // contraction: the direct GETT (any data type, operator and layout), the operands
            // read and D written in place, one launch, no workspace
            DirectGett = 8,
            // contraction: the f64 strided GEMM with f32 / complex f32 words (f32 MFMA products,
            // f32 accumulation): the native route of f32 data with the 32F / C32F compute type
            Gett32 = 9,
        };

        // How contractions of f32 / complex f32 (and mixed real) data with the 32F / C32F compute
        // type multiply (HIPTENSOR_ROCKE_F32_RECIPE): Native, the f32 strided GEMM (f32 products
        // and accumulation); X3 / X4 / X6 / X9, the universal GEMM with every operand split into
        // bf16 parts (x3 and x4: two parts, 3 or 4 products; x6: three parts, 6 products; x9:
        // three parts, every product); F64, the f64 strided GEMM (operands converted exactly,
        // f64 accumulation). The f64 route stays a candidate of every recipe (it is at least as
        // accurate as each of them).
        enum class F32Recipe : uint8_t
        {
            Native,
            X3,
            X4,
            X6,
            X9,
            F64,
        };

        char const* f32RecipeName(F32Recipe recipe);

        // Parses a recipe name (native, x3, x4, x6, x9, f64); false for any other text.
        bool parseF32Recipe(std::string const& text, F32Recipe& recipe);

        // What the selector knows of a candidate besides its program: the facts its features
        // are computed from. Contraction extents are the M, N, K, L group sizes; reduction
        // extents are M = outputs, K = reduced elements.
        struct CandidateFacts
        {
            Route   route     = Route::Elementwise;
            int     body      = 0; // GEMM body / f64 GEMM class / reduction mode (0 row, 1 col)
            bool    f64Route  = false; // operands converted to f64 (complex f64)
            bool    direct    = false; // the GEMM writes D (no epilogue pass)
            bool    kBlock    = false;
            bool    streamK   = false;
            bool    bilinear  = false;
            bool    operandOp = false; // a unary operator on an input
            int64_t M = 1, N = 1, K = 1, L = 1;
            int     modesM = 0, modesN = 0, modesK = 0, modesL = 0;
            // the input's (or output's) stride-1 mode: 0 none, 1 M, 2 N, 3 K, 4 L
            int               unitA = 0, unitB = 0, unitD = 0;
            int64_t           inner     = 1; // reduction: extent of the innermost reduced mode
            int64_t           innerKept = 1; // reduction: extent of the innermost kept mode
            int64_t           tileM = 0, tileN = 0, tileK = 0, block = 0, persistent = 0;
            int64_t           ctas         = 0; // CTAs of the main kernel
            int64_t           paddedVolume = 0; // contraction: padded M * N * K * L of the GEMM
            int64_t           split = 1, chunk = 0;
            uint64_t          bytes = 0; // estimated device bytes read and written by all steps
            uint64_t          flops = 0;
            PackKernel const* main  = nullptr; // the kernel that does most of the work
            // The f32 recipe that decided the routes of a contraction, when it is not x6: the
            // pack's selection models were trained on the x6 route set, so these candidates are
            // ranked by the static order (selection cells <cell>.<tag>, which no pack has).
            std::string selectTag;
        };

        // One way to run a problem: a complete program (kernel functions not loaded yet, see
        // loadFunctions) and its facts. `key` names it among the candidates of the problem and
        // is stable across processes for the same pack (plan cache, logs, KERNEL_RANK).
        struct Candidate
        {
            std::string    key;
            Program        program;
            CandidateFacts facts;
            // position in the static order (0 = the rule's choice)
            int  staticRank = 0;
            bool tuned      = false;
            // a candidate the rules put first when no tuned candidate exists (the ring GETT:
            // the selection models were not trained on its route)
            bool preferred = false;
        };

        struct CandidateOptions
        {
            // Contractions: only the F64 route (HIPTENSOR_ROCKE_CONTRACTION_ROUTE=f64).
            bool f64Only = false;
            // Contractions: skip rocKE's tuned GETT route table (HIPTENSOR_ROCKE_DISABLE_TUNED).
            bool disableTuned = false;
            // Contractions of f32 / complex f32 data with the 32F / C32F compute type: the recipe
            // (HIPTENSOR_ROCKE_F32_RECIPE; native by default).
            F32Recipe f32Recipe = F32Recipe::Native;
            // Trinary contractions: orders the candidates of one of the two steps (indices, best
            // first; the selection of rocke_select.hpp). Empty: the static order.
            std::function<std::vector<int>(backend::Problem const&, std::vector<Candidate> const&)>
                rankStep;
        };

        // Every candidate of `problem` the pack admits within `workspaceLimit` bytes of
        // workspace, in the static order. HIPTENSOR_STATUS_SUCCESS with at least one candidate,
        // HIPTENSOR_STATUS_NOT_SUPPORTED with `reason` when there is none, or another status
        // (an API error, a pack defect) that ends the plan creation.
        //
        // A trinary contraction E = alpha * opA(A) * opB(B) * opC(C) + beta * opD(D) runs as
        // hipTensor's two chained contractions: T = opA(A) * opB(B), stored in E's data type in
        // the first bytes of the workspace (T's modes and packed column-major layout as the
        // library defines them), then E = alpha * T * opC(C) + beta * opD(D). Its candidates
        // pair a candidate of each step (see intermediateOf / enumerateTrinary in
        // rocke_plan.cpp); the steps share the workspace after T.
        hiptensorStatus_t enumerateCandidates(backend::Problem const& problem,
                                              Pack const&             pack,
                                              CandidateOptions const& options,
                                              uint64_t                workspaceLimit,
                                              std::vector<Candidate>& out,
                                              std::string&            reason);

        char const* routeName(Route route);

        // Loads the code objects of the program's kernels on the current device.
        hiptensorStatus_t loadFunctions(Program& program);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_PLAN_HPP
