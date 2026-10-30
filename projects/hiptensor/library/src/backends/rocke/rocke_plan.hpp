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
#include <memory>
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
            // reduction: a tuned class of rocKE's native strided reduction (the geometry as
            // kernel arguments), with its split finalize
            TunedReduction = 10,
            // contraction: the route rocKE's tensor GETT planner chooses for a problem the GETT
            // route tables do not hold (rocke_gett_jit.hpp), its kernels built for the problem's
            // sizes and strides and compiled at plan creation (HIPTENSOR_JIT_MODE_DEFAULT)
            JitGett = 11,
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
            // JitGett: the kernel launches of the planned route (known before they are built),
            // the route (1 MN-layout GETT, 2 MN-layout GETT after tensor_pack copies,
            // 3 K-contiguous GETT after copies, 4 ring body, 5 K-contiguous GETT on the operands
            // in place) and the planner's predicted time in nanoseconds (0: none)
            int     launches    = 0;
            int     gettKind    = 0;
            int64_t predictedNs = 0;
            // the plan specializes its elementwise launches with the JIT
            // (HIPTENSOR_JIT_MODE_DEFAULT, the JIT not turned off): the operand packing and
            // epilogue launches of the GEMM routes then run faster than the pack's (set by the
            // backend for the selection; a feature from schema 3)
            bool jitMode = false;
            // The f32 recipe that decided the routes of a contraction, when it is not x6: its
            // candidates are ranked by the models of the selection cells <cell>.<tag> (the packs
            // have them for the default recipe, native), else by the static order.
            std::string selectTag;
        };

        struct GettJitPlan;

        // One way to run a problem: a complete program (kernel functions not loaded yet, see
        // loadFunctions) and its facts. `key` names it among the candidates of the problem and
        // is stable across processes for the same pack (plan cache, logs, KERNEL_RANK).
        //
        // A JitGett candidate's program is pending (Program::pending) until its kernels are
        // built (buildJitCandidate): its workspace, checks and fallback are set, its launches
        // are not.
        struct Candidate
        {
            std::string    key;
            Program        program;
            CandidateFacts facts;
            // position in the static order (0 = the rule's choice)
            int  staticRank = 0;
            bool tuned      = false;
            // a candidate the rules put first when no tuned candidate exists and the cell's
            // selection model does not score its route (rocke_select.hpp)
            bool preferred = false;
            // JitGett: the planned route its launches are built from
            std::shared_ptr<GettJitPlan const> jitGett;
        };

        struct CandidateOptions
        {
            // Contractions: only the F64 route (HIPTENSOR_ROCKE_CONTRACTION_ROUTE=f64).
            bool f64Only = false;
            // Contractions: skip rocKE's tuned and AOT GETT route tables; reductions: skip rocKE's
            // tuned reduction classes (route table and untuned choice), the generic kernels only;
            // elementwise operations: no flat kernel (HIPTENSOR_ROCKE_DISABLE_TUNED).
            bool disableTuned = false;
            // Contractions of f32 / complex f32 data with the 32F / C32F compute type: the recipe
            // (HIPTENSOR_ROCKE_F32_RECIPE; native by default).
            F32Recipe f32Recipe = F32Recipe::Native;
            // Contractions the GETT route tables do not hold: the JitGett candidate of rocKE's
            // tensor GETT planner (plans with HIPTENSOR_JIT_MODE_DEFAULT in a library built with
            // the JIT; not the steps of a trinary contraction)
            bool jitGett = false;
            // Contractions: consult the pack's tuned and AOT GETT route tables (false with
            // HIPTENSOR_ROCKE_GETT_TABLES=off: rocKE's tensor GETT planner decides the problems
            // they hold too, which compares the JIT route with the tables)
            bool gettTables = true;
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

        // Loads the code objects of the program's kernels on the current device (the pack's
        // kernels; a JIT launch has its function already).
        hiptensorStatus_t loadFunctions(Program& program);

        // Builds the launches of a pending candidate (a JitGett candidate: its kernels from the
        // JIT's kernel cache, else compiled now while `budgetMs` (0: none) minus `spentMs` is not
        // spent) and loads its functions; `spentMs` grows by the time the build took.
        // HIPTENSOR_STATUS_SUCCESS (also for a candidate that is not pending);
        // HIPTENSOR_STATUS_NOT_SUPPORTED with `reason` when the kernels cannot be had (the
        // candidate stays pending and must not run). `note` gets the JIT log line.
        hiptensorStatus_t buildJitCandidate(Candidate&                candidate,
                                            backend::DeviceKey const& device,
                                            double                    budgetMs,
                                            double&                   spentMs,
                                            std::string&              note,
                                            std::string&              reason);
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_PLAN_HPP
