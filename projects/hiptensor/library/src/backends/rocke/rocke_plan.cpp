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

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <tuple>

#include "rocke_plan.hpp"
#include "rocke_rreduce_rule.hpp"
#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
#include "rocke_gett_jit.hpp"
#endif

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            using namespace rrule;
            using backend::OpKind;
            using backend::Problem;
            using backend::TensorArg;

            // Rank bucket R8 of the generic kernels: at most 8 modes after canonicalization,
            // modes 0..2 on the thread space, 3..7 on the row space.
            constexpr int      kMaxRank         = 8;
            constexpr int      kInnerSlots      = 3;
            constexpr int64_t  kI32Max          = INT32_MAX;
            constexpr uint32_t kGridYZMax       = 65535u;
            constexpr uint64_t kWorkspaceAlign  = 256u;
            constexpr uint64_t kDtlBufferRange  = 0x7FFF0000ull;
            constexpr int64_t  kGett64SpanLimit = (int64_t(1) << 31) - 16;
            constexpr int      kWave            = 64;
            constexpr char     kEw[]            = "ew";
            constexpr char     kReduce[]        = "reduce";
            constexpr char     kGemm[]          = "gemm";
            constexpr char     kGett64[]        = "gett64";
            constexpr char     kGett32[]        = "gett32";

            std::string dtypeName(hiptensorDataType_t type)
            {
                switch(type)
                {
                case HIPTENSOR_R_16F:
                    return "f16";
                case HIPTENSOR_R_16BF:
                    return "bf16";
                case HIPTENSOR_R_32F:
                case HIPTENSOR_C_32F:
                    return "f32";
                case HIPTENSOR_R_64F:
                case HIPTENSOR_C_64F:
                    return "f64";
                default:
                    return "";
                }
            }

            bool isComplex(hiptensorDataType_t type)
            {
                return type == HIPTENSOR_C_32F || type == HIPTENSOR_C_64F;
            }

            uint64_t elementBytes(std::string const& name, bool complexElement = false)
            {
                uint64_t bytes = name == "f16" || name == "bf16" ? 2 : name == "f32" ? 4 : 8;
                return complexElement ? 2 * bytes : bytes;
            }

            // The unary operators only the kernel pack's "math" kernels implement (f64 programs
            // of rocke.helpers.elementary); the other kernels keep their registers without them.
            bool mathUnary(int op)
            {
                switch(op)
                {
                case HIPTENSOR_OP_SIN:
                case HIPTENSOR_OP_COS:
                case HIPTENSOR_OP_TAN:
                case HIPTENSOR_OP_SINH:
                case HIPTENSOR_OP_COSH:
                case HIPTENSOR_OP_ASIN:
                case HIPTENSOR_OP_ACOS:
                case HIPTENSOR_OP_ATAN:
                case HIPTENSOR_OP_ASINH:
                case HIPTENSOR_OP_ACOSH:
                case HIPTENSOR_OP_ATANH:
                case HIPTENSOR_OP_CEIL:
                case HIPTENSOR_OP_FLOOR:
                    return true;
                default:
                    return false;
                }
            }

            bool supportedUnary(hiptensorOperator_t op)
            {
                switch(op)
                {
                case HIPTENSOR_OP_IDENTITY:
                case HIPTENSOR_OP_SQRT:
                case HIPTENSOR_OP_RELU:
                case HIPTENSOR_OP_CONJ:
                case HIPTENSOR_OP_RCP:
                case HIPTENSOR_OP_SIGMOID:
                case HIPTENSOR_OP_TANH:
                case HIPTENSOR_OP_EXP:
                case HIPTENSOR_OP_LOG:
                case HIPTENSOR_OP_ABS:
                case HIPTENSOR_OP_NEG:
                    return true;
                default:
                    return mathUnary(op);
                }
            }

            bool supportedBinary(hiptensorOperator_t op)
            {
                return op == HIPTENSOR_OP_ADD || op == HIPTENSOR_OP_MUL || op == HIPTENSOR_OP_MAX
                       || op == HIPTENSOR_OP_MIN;
            }

            uint64_t alignUp(uint64_t value, uint64_t to)
            {
                return (value + to - 1) / to * to;
            }

            int64_t roundUp(int64_t value, int64_t to)
            {
                return (value + to - 1) / to * to;
            }

            int64_t ceilDiv(int64_t a, int64_t b)
            {
                return (a + b - 1) / b;
            }

            // FNV-1a 64 of a route key (the pack builder hashes the same keys)
            uint64_t fnv1a64(std::string const& text)
            {
                uint64_t hash = 0xCBF29CE484222325ull;
                for(unsigned char c : text)
                {
                    hash = (hash ^ c) * 0x100000001B3ull;
                }
                return hash;
            }

            // One mode of a canonical problem: its extent and its stride in each tensor.
            struct Dim
            {
                int64_t                extent = 1;
                std::array<int64_t, 4> stride{};
            };

            // Drops extent-1 modes, orders the modes by the strides of tensor `key` (smallest
            // first) and merges consecutive modes that are contiguous in every tensor.
            void canonicalize(std::vector<Dim>& dims, int tensors, int key)
            {
                dims.erase(std::remove_if(dims.begin(),
                                          dims.end(),
                                          [](Dim const& d) { return d.extent == 1; }),
                           dims.end());
                std::stable_sort(dims.begin(), dims.end(), [key](Dim const& a, Dim const& b) {
                    return std::llabs(a.stride[key]) < std::llabs(b.stride[key]);
                });
                std::vector<Dim> merged;
                for(auto const& d : dims)
                {
                    if(!merged.empty())
                    {
                        auto& last       = merged.back();
                        bool  contiguous = true;
                        for(int t = 0; t < tensors; t++)
                        {
                            contiguous = contiguous && d.stride[t] == last.stride[t] * last.extent;
                        }
                        if(contiguous && last.extent * d.extent <= kI32Max)
                        {
                            last.extent *= d.extent;
                            continue;
                        }
                    }
                    merged.push_back(d);
                }
                dims = std::move(merged);
            }

            int64_t spanOf(std::vector<Dim> const& dims, int tensor)
            {
                int64_t span = 0;
                for(auto const& d : dims)
                {
                    span += (d.extent - 1) * std::llabs(d.stride[tensor]);
                }
                return span;
            }

            int64_t product(std::vector<Dim> const& dims, size_t first, size_t last)
            {
                int64_t p = 1;
                for(size_t i = first; i < last && i < dims.size(); i++)
                {
                    p *= dims[i].extent;
                }
                return p;
            }

            // Puts canonical modes into the 8 kernel slots: up to three on the thread space, the
            // rest on the row space. False when they do not fit.
            bool placeSlots(std::vector<Dim> const& dims,
                            int                     tensors,
                            std::array<Dim, 8>&     slots,
                            int64_t&                nIn,
                            int64_t&                nOut)
            {
                int rank  = static_cast<int>(dims.size());
                int inner = std::min(kInnerSlots, rank);
                while(inner > 1 && product(dims, 0, inner) > kI32Max)
                {
                    inner--;
                }
                if(rank > 0 && dims[0].extent > kI32Max)
                {
                    return false;
                }
                if(rank - inner > kMaxRank - kInnerSlots)
                {
                    return false;
                }
                for(auto& s : slots)
                {
                    s = Dim{};
                }
                for(int i = 0; i < inner; i++)
                {
                    slots[i] = dims[i];
                }
                for(int i = inner; i < rank; i++)
                {
                    slots[kInnerSlots + i - inner] = dims[i];
                }
                nIn  = 1;
                nOut = 1;
                for(int i = 0; i < kMaxRank; i++)
                {
                    (i < kInnerSlots ? nIn : nOut) *= slots[i].extent;
                }
                static_cast<void>(tensors);
                return nIn <= kI32Max && nOut <= kI32Max;
            }

            bool setMagic(ArgWriter& w, char const* m, char const* s, int64_t divisor)
            {
                auto magic = magicU31(static_cast<uint32_t>(divisor));
                return w.setU32(m, magic.first) && w.set(s, magic.second);
            }

            // A tensor of a generic elementwise step: the kernel argument name, where it lives at
            // execution, and its scale.
            struct EwOperand
            {
                std::string name;
                BufferRole  role  = BufferRole::None;
                uint64_t    bytes = 0;
                ScalarRole  scale = ScalarRole::One;
                int         op    = HIPTENSOR_OP_IDENTITY;
            };

            hiptensorStatus_t finishStep(LaunchStep& step, ArgWriter const& w, std::string& reason)
            {
                std::string missing;
                if(!w.complete(missing))
                {
                    reason = "kernel " + step.kernel->name + ": argument " + missing
                             + " is not set (pack and runtime disagree)";
                    return HIPTENSOR_STATUS_INTERNAL_ERROR;
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // Applies `remap` (buffer roles and offsets, scalar roles) to a launch and to every
            // launch of its fallback, which execution takes in its place (the fallback list is
            // shared between copies of a launch: the remapped launch gets its own).
            template <typename Remap>
            void remapLaunch(LaunchStep& step, Remap const& remap)
            {
                remap(step);
                if(step.fallback != nullptr)
                {
                    auto fallback = std::make_shared<std::vector<LaunchStep>>(*step.fallback);
                    for(auto& f : *fallback)
                    {
                        remapLaunch(f, remap);
                    }
                    step.fallback = std::move(fallback);
                }
            }

            // One launch of the generic elementwise kernel over canonical modes `dims` whose
            // stride slots follow `operands` (inputs in kernel order, then the output).
            hiptensorStatus_t ewStep(Pack const&                                      pack,
                                     std::vector<std::pair<std::string, int64_t>>     ints,
                                     std::vector<std::pair<std::string, std::string>> strs,
                                     std::vector<Dim> const&                          dims,
                                     std::vector<EwOperand> const&                    operands,
                                     std::array<int, 2> const&                        binaryOps,
                                     LaunchStep&                                      step,
                                     std::string&                                     reason)
            {
                int const          tensors = static_cast<int>(operands.size());
                std::array<Dim, 8> slots;
                int64_t            nIn = 0, nOut = 0;
                if(!placeSlots(dims, tensors, slots, nIn, nOut))
                {
                    reason = "more than 8 modes after merging, or a mode too long for the generic "
                             "elementwise kernel";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                // 32-bit offsets address every element below 2^31; a complex kernel indexes the
                // components (twice the element offset)
                bool const    cplxRow = std::any_of(ints.begin(), ints.end(), [](auto const& kv) {
                    return kv.first == "cplx" && kv.second == 1;
                });
                int64_t const limit   = cplxRow ? kI32Max / 2 : kI32Max;
                bool          idx64   = false;
                for(int t = 0; t < tensors; t++)
                {
                    idx64 = idx64 || spanOf(dims, t) > limit;
                }
                ints.emplace_back("idx64", idx64 ? 1 : 0);
                bool const math = std::any_of(operands.begin(),
                                              operands.end() - 1,
                                              [](EwOperand const& o) { return mathUnary(o.op); });
                ints.emplace_back("math", math ? 1 : 0);
                step.kernel = pack.find(kEw, ints, strs);
                if(step.kernel == nullptr)
                {
                    reason = "the pack has no elementwise kernel for this data type row";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                ArgWriter   w(step);
                bool        ok     = true;
                int const   arity  = tensors - 1;
                char const* scal[] = {"alpha", "beta", "gamma"};
                bool const  cplx   = step.kernel->attr("cplx") == 1;
                for(int t = 0; t < tensors; t++)
                {
                    auto const& o = operands[t];
                    ok            = ok && w.pointer(o.name, o.role, o.bytes);
                    if(t == tensors - 1)
                    {
                        continue;
                    }
                    ok = ok && w.set("op" + o.name, o.op);
                    // the kernel's scale argument of this input: alpha (A), beta (B), gamma (C)
                    std::string s = o.name == "A" ? scal[0] : o.name == "B" ? scal[1] : scal[2];
                    if(cplx)
                    {
                        ok = ok && w.scalar(s + "_re", o.scale, false)
                             && w.scalar(s + "_im", o.scale, true);
                    }
                    else
                    {
                        ok = ok && w.scalar(s, o.scale);
                    }
                }
                if(arity >= 2)
                {
                    ok = ok && w.set("opAB", binaryOps[0]);
                }
                if(arity == 3)
                {
                    ok = ok && w.set("opABC", binaryOps[1]);
                }
                uint32_t const block = static_cast<uint32_t>(step.kernel->attr("block", 256));
                int64_t const  gridY = std::min<int64_t>(nOut, kGridYZMax);
                ok = ok && w.set("n_in", nIn) && w.set("n_out", nOut) && w.set("ystep", gridY);
                for(int i = 0; i < kMaxRank; i++)
                {
                    ok = ok && w.set("e" + std::to_string(i), slots[i].extent);
                    if(arity == 1)
                    {
                        ok = ok && w.set("lim" + std::to_string(i), slots[i].extent);
                    }
                    for(int t = 0; t < tensors; t++)
                    {
                        ok = ok
                             && w.set("s" + operands[t].name + std::to_string(i),
                                      slots[i].extent == 1 ? 0 : slots[i].stride[t]);
                    }
                }
                ok = ok && setMagic(w, "m0", "s0", slots[0].extent)
                     && setMagic(w, "m1", "s1", slots[1].extent);
                for(int i = 3; i <= 6; i++)
                {
                    auto n = std::to_string(i);
                    ok = ok && setMagic(w, ("m" + n).c_str(), ("s" + n).c_str(), slots[i].extent);
                }
                if(!ok)
                {
                    reason = "kernel " + step.kernel->name
                             + ": an argument does not fit its kernel argument";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                step.grid = dim3(
                    static_cast<uint32_t>(ceilDiv(nIn, block)), static_cast<uint32_t>(gridY), 1);
                step.block = dim3(block, 1, 1);
                return finishStep(step, w, reason);
            }

            // Most launches one elementwise program (or one reduction) is split into.
            constexpr int64_t kMaxLaunches = 4096;
            // Piece length of a mode longer than the kernels' i32 coordinates.
            constexpr int64_t kLongModePiece = int64_t(1) << 30;

            // The modes `dims` with mode `i` restricted to [first, first + extent), and the
            // element offset of that restriction in each of `tensors` tensors.
            std::vector<Dim> restrictMode(std::vector<Dim> const& dims,
                                          size_t                  i,
                                          int64_t                 first,
                                          int64_t                 extent,
                                          int                     tensors,
                                          std::vector<int64_t>&   offsets)
            {
                offsets.assign(tensors, 0);
                for(int t = 0; t < tensors; t++)
                {
                    offsets[t] = first * dims[i].stride[t];
                }
                auto out      = dims;
                out[i].extent = extent;
                if(extent == 1)
                {
                    out.erase(out.begin() + i);
                }
                return out;
            }

            // The element size of each elementwise operand (`strs` keys "a", "b", "c", "d" name
            // the operands' data types); 0 when unknown.
            std::vector<uint64_t>
                operandBytes(std::vector<std::pair<std::string, std::string>> const& strs,
                             std::vector<EwOperand> const&                           operands,
                             bool                                                    cplx)
            {
                std::vector<uint64_t> bytes(operands.size(), 0);
                for(size_t t = 0; t < operands.size(); t++)
                {
                    std::string key = operands[t].name;
                    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char ch) {
                        return static_cast<char>(std::tolower(ch));
                    });
                    for(auto const& s : strs)
                    {
                        if(s.first == key)
                        {
                            bytes[t] = elementBytes(s.second, cplx);
                        }
                    }
                }
                return bytes;
            }

            // See ewSteps. `budget`: launches left.
            hiptensorStatus_t ewPieces(Pack const&                                             pack,
                                       std::vector<std::pair<std::string, int64_t>> const&     ints,
                                       std::vector<std::pair<std::string, std::string>> const& strs,
                                       std::vector<Dim>                                        dims,
                                       std::vector<EwOperand> const& operands,
                                       std::vector<uint64_t> const&  bytes,
                                       std::array<int, 2> const&     binaryOps,
                                       std::vector<LaunchStep>&      steps,
                                       int64_t&                      budget,
                                       std::string&                  reason)
            {
                int const tensors = static_cast<int>(operands.size());
                // Host-iterated pieces of mode `i`, `piece` long: one ewPieces each, with the
                // operands offset to the piece.
                auto pieces = [&](size_t i, int64_t piece, char const* why) {
                    for(int t = 0; t < tensors; t++)
                    {
                        if(bytes[t] == 0 || dims[i].stride[t] < 0)
                        {
                            reason = std::string(why)
                                     + ", with an operand the pieces of the generic elementwise "
                                       "kernel cannot address";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                    }
                    for(int64_t first = 0; first < dims[i].extent; first += piece)
                    {
                        std::vector<int64_t> offsets;
                        auto const           sub   = restrictMode(dims,
                                                      i,
                                                      first,
                                                      std::min(piece, dims[i].extent - first),
                                                      tensors,
                                                      offsets);
                        auto                 moved = operands;
                        for(int t = 0; t < tensors; t++)
                        {
                            moved[t].bytes += uint64_t(offsets[t]) * bytes[t];
                        }
                        auto status = ewPieces(
                            pack, ints, strs, sub, moved, bytes, binaryOps, steps, budget, reason);
                        if(status != HIPTENSOR_STATUS_SUCCESS)
                        {
                            if(budget < 0)
                            {
                                reason = std::string(why) + ", and more than "
                                         + std::to_string(kMaxLaunches)
                                         + " launches of the generic elementwise kernel";
                            }
                            return status;
                        }
                    }
                    return HIPTENSOR_STATUS_SUCCESS;
                };

                // a mode past the i32 coordinates: pieces of 2^30
                for(size_t i = 0; i < dims.size(); i++)
                {
                    if(dims[i].extent > kI32Max)
                    {
                        return pieces(i, kLongModePiece, "a mode of more than 2^31 elements");
                    }
                }
                // modes past the kernel's 8 slots (the outermost ones, as canonicalize orders
                // them): one launch per index of the outermost mode
                if(dims.size() > size_t(kMaxRank))
                {
                    return pieces(dims.size() - 1, 1, "more than 8 modes after merging");
                }
                std::array<Dim, 8> slots;
                int64_t            nIn = 0, nOut = 0;
                if(!dims.empty() && !placeSlots(dims, tensors, slots, nIn, nOut))
                {
                    // more than 2^31 rows: halves of the outermost mode
                    auto const last = dims.size() - 1;
                    return pieces(last,
                                  (dims[last].extent + 1) / 2,
                                  "more than 2^31 rows of the generic elementwise kernel");
                }
                if(--budget < 0)
                {
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                LaunchStep step;
                auto status = ewStep(pack, ints, strs, dims, operands, binaryOps, step, reason);
                if(status == HIPTENSOR_STATUS_SUCCESS)
                {
                    steps.push_back(std::move(step));
                }
                return status;
            }

            // The launches of the generic elementwise kernel over canonical modes `dims` (see
            // ewStep), appended to `steps`. What one launch cannot address is split into
            // host-iterated pieces, each a launch with its operands offset to the piece: modes
            // past the kernel's 8 slots (one piece per index of the outermost mode), modes of
            // more than 2^31 elements (pieces of 2^30), and row spaces of more than 2^31 rows
            // (halves of the outermost mode). At most kMaxLaunches launches. `strs` gives the
            // data type of each operand's kernel argument ("a", "b", "c", "d").
            hiptensorStatus_t ewSteps(Pack const&                                      pack,
                                      std::vector<std::pair<std::string, int64_t>>     ints,
                                      std::vector<std::pair<std::string, std::string>> strs,
                                      std::vector<Dim> const&                          dims,
                                      std::vector<EwOperand> const&                    operands,
                                      std::array<int, 2> const&                        binaryOps,
                                      bool                                             cplx,
                                      std::vector<LaunchStep>&                         steps,
                                      std::string&                                     reason)
            {
                auto const bytes  = operandBytes(strs, operands, cplx);
                int64_t    budget = kMaxLaunches;
                return ewPieces(
                    pack, ints, strs, dims, operands, bytes, binaryOps, steps, budget, reason);
            }

            // The extent of `mode` in `tensor`, or 0 when the tensor does not have it.
            int64_t extentOf(TensorArg const& tensor, int32_t mode, int64_t* stride = nullptr)
            {
                for(int i = 0; i < tensor.numModes(); i++)
                {
                    if(tensor.modes[i] == mode)
                    {
                        if(stride != nullptr)
                        {
                            *stride = i < static_cast<int>(tensor.strides.size())
                                          ? tensor.strides[i]
                                          : 0;
                        }
                        return tensor.extents[i];
                    }
                }
                return 0;
            }

            bool realScalars(Problem const& problem, std::string& reason)
            {
                if(isComplex(problem.scalarType) || dtypeName(problem.scalarType).empty())
                {
                    reason = "scalar type not supported by the rocKE kernels";
                    return false;
                }
                return true;
            }
        } // namespace

        char const* routeName(Route route)
        {
            switch(route)
            {
            case Route::Elementwise:
                return "elementwise";
            case Route::Reduction:
                return "reduction";
            case Route::TunedGett:
                return "tuned-gett";
            case Route::H16:
                return "h16";
            case Route::X6:
                return "x6";
            case Route::CF32:
                return "cf32";
            case Route::Gett64:
                return "gett64";
            case Route::RingGett:
                return "rgett";
            case Route::DirectGett:
                return "dgett";
            case Route::TunedReduction:
                return "tuned-reduction";
            case Route::Gett32:
                return "gett32";
            case Route::JitGett:
                return "jit-gett";
            }
            return "?";
        }

        char const* f32RecipeName(F32Recipe recipe)
        {
            switch(recipe)
            {
            case F32Recipe::Native:
                return "native";
            case F32Recipe::X3:
                return "x3";
            case F32Recipe::X4:
                return "x4";
            case F32Recipe::X6:
                return "x6";
            case F32Recipe::X9:
                return "x9";
            case F32Recipe::F64:
                return "f64";
            }
            return "?";
        }

        bool parseF32Recipe(std::string const& text, F32Recipe& recipe)
        {
            for(auto r : {F32Recipe::Native,
                          F32Recipe::X3,
                          F32Recipe::X4,
                          F32Recipe::X6,
                          F32Recipe::X9,
                          F32Recipe::F64})
            {
                if(text == f32RecipeName(r))
                {
                    recipe = r;
                    return true;
                }
            }
            return false;
        }

        namespace
        {
            // ---------------------------------------------------------------------- elementwise

            // The canonical problem of an elementwise operation, as planElementwise built it
            // (inputs in kernel order, then D).
            struct EwLayout
            {
                std::vector<Dim>                                 dims;
                std::vector<EwOperand>                           operands;
                std::vector<std::pair<std::string, std::string>> strs;
                std::array<int, 2>                               binaryOps{};
            };

            hiptensorStatus_t planElementwise(Problem const&  problem,
                                              Pack const&     pack,
                                              Program&        program,
                                              CandidateFacts& facts,
                                              std::string&    reason,
                                              EwLayout*       layout = nullptr)
            {
                // Kernel tensor names and problem slots: permute A -> B, binary A, C -> D, trinary
                // A, B, C -> D.
                std::vector<std::pair<std::string, int>> inputs;
                switch(problem.kind)
                {
                case OpKind::Permute:
                    inputs = {{"A", 0}};
                    break;
                case OpKind::ElementwiseBinary:
                    inputs = {{"A", 0}, {"C", 2}};
                    break;
                case OpKind::ElementwiseTrinary:
                    inputs = {{"A", 0}, {"B", 1}, {"C", 2}};
                    break;
                default:
                    reason = "not an elementwise problem";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(!realScalars(problem, reason))
                {
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const& out = problem.output();
                if(out.unaryOp != HIPTENSOR_OP_IDENTITY && out.unaryOp != HIPTENSOR_OP_UNKNOWN)
                {
                    reason = "an operator on the output tensor is not supported";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(problem.kind != OpKind::Permute
                   && (!supportedBinary(problem.opAB)
                       || (problem.kind == OpKind::ElementwiseTrinary
                           && !supportedBinary(problem.opABC))))
                {
                    reason = "binary operator not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }

                // Modes of the output; inputs are read through their own strides (0 broadcasts).
                int const        tensors = static_cast<int>(inputs.size()) + 1;
                std::vector<Dim> dims;
                for(int i = 0; i < out.numModes(); i++)
                {
                    Dim d;
                    d.extent              = out.extents[i];
                    d.stride[tensors - 1] = out.strides[i];
                    for(int t = 0; t < tensors - 1; t++)
                    {
                        auto const& in     = problem.t[inputs[t].second];
                        int64_t     stride = 0;
                        auto        extent = extentOf(in, out.modes[i], &stride);
                        if(extent != 0 && extent != d.extent)
                        {
                            reason = "input and output extents differ";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                        d.stride[t] = extent == 0 ? 0 : stride;
                    }
                    dims.push_back(d);
                }
                std::vector<EwOperand>                       operands;
                std::vector<std::pair<std::string, int64_t>> ints{
                    {"arity", static_cast<int64_t>(inputs.size())}, {"cplx", 0}};
                std::vector<std::pair<std::string, std::string>> strs;
                ScalarRole const                                 scales[]
                    = {ScalarRole::Alpha, ScalarRole::Beta, ScalarRole::Gamma};
                BufferRole const roles[] = {BufferRole::A, BufferRole::B, BufferRole::C};
                for(auto const& [name, slot] : inputs)
                {
                    auto const& in = problem.t[slot];
                    for(int i = 0; i < in.numModes(); i++)
                    {
                        if(extentOf(out, in.modes[i]) == 0 && in.extents[i] != 1)
                        {
                            reason = "an input mode is not an output mode";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                    }
                    if(!supportedUnary(in.unaryOp) || in.unaryOp == HIPTENSOR_OP_CONJ)
                    {
                        reason = "unary operator not supported by the rocKE kernels";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    auto dt = dtypeName(in.type);
                    if(dt.empty() || isComplex(in.type))
                    {
                        reason = "data type not supported by the rocKE kernels";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    std::string key = name == "A" ? "a" : name == "B" ? "b" : "c";
                    strs.emplace_back(key, dt);
                    EwOperand o;
                    o.name  = name;
                    o.role  = roles[slot == 0 ? 0 : slot == 1 ? 1 : 2];
                    o.scale = scales[slot == 0 ? 0 : slot == 1 ? 1 : 2];
                    o.op    = static_cast<int>(in.unaryOp);
                    operands.push_back(o);
                }
                auto outType = dtypeName(out.type);
                if(outType.empty() || isComplex(out.type))
                {
                    reason = "data type not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                strs.emplace_back("d", outType);
                EwOperand d;
                d.name = "D";
                d.role = BufferRole::Out;
                operands.push_back(d);
                canonicalize(dims, tensors, tensors - 1);

                // The data type rows of the API (the same as the other backend's)
                auto const sc  = problem.scalarType;
                auto const ta  = problem.t[0].type;
                bool       row = false;
                if(problem.kind == OpKind::Permute)
                {
                    row = (ta == HIPTENSOR_R_16F
                           && (sc == HIPTENSOR_R_16F || sc == HIPTENSOR_R_32F))
                          || (ta == HIPTENSOR_R_32F && sc == HIPTENSOR_R_32F)
                          || (ta == HIPTENSOR_R_64F && sc == HIPTENSOR_R_64F);
                }
                else if(problem.kind == OpKind::ElementwiseBinary)
                {
                    auto const tc = problem.t[2].type;
                    row           = tc == out.type && ta == tc
                          && (((ta == HIPTENSOR_R_16F || ta == HIPTENSOR_R_16BF)
                               && (sc == ta || sc == HIPTENSOR_R_32F))
                              || (ta == HIPTENSOR_R_32F && sc == HIPTENSOR_R_32F)
                              || (ta == HIPTENSOR_R_64F && sc == HIPTENSOR_R_64F));
                }
                else
                {
                    row = ta == problem.t[1].type && ta == problem.t[2].type && sc == ta
                          && (ta == HIPTENSOR_R_16F || ta == HIPTENSOR_R_32F
                              || ta == HIPTENSOR_R_64F);
                }
                if(!row)
                {
                    reason = "data type combination not supported by the API";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }

                program             = Program{};
                program.pack        = &pack;
                program.scalarType  = problem.scalarType;
                program.description = "elementwise";
                program.nullStatus  = HIPTENSOR_STATUS_NOT_INITIALIZED;
                for(auto const& input : inputs)
                {
                    int const slot = input.second;
                    int const s    = slot == 0 ? 0 : slot == 1 ? 1 : 2;
                    program.requiredScalars.push_back(s);
                    program.requiredBuffers.emplace_back(slot == 0   ? BufferRole::A
                                                         : slot == 1 ? BufferRole::B
                                                                     : BufferRole::C,
                                                         s);
                }
                program.requiredBuffers.emplace_back(BufferRole::Out, -1);
                std::vector<LaunchStep> steps;
                auto                    status
                    = ewSteps(pack,
                              ints,
                              strs,
                              dims,
                              operands,
                              {static_cast<int>(problem.opAB), static_cast<int>(problem.opABC)},
                              false,
                              steps,
                              reason);
                if(status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                auto const& step    = steps.front();
                program.description = step.kernel->name;
                facts               = CandidateFacts{};
                facts.route         = Route::Elementwise;
                facts.main          = step.kernel;
                facts.ctas          = int64_t(step.grid.x) * step.grid.y * step.grid.z;
                facts.block         = step.block.x;
                for(auto& s : steps)
                {
                    program.steps.push_back(std::move(s));
                }
                if(layout != nullptr)
                {
                    layout->dims     = dims;
                    layout->operands = operands;
                    layout->strs     = strs;
                    layout->binaryOps
                        = {static_cast<int>(problem.opAB), static_cast<int>(problem.opABC)};
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }

            constexpr char kEwt[] = "ewt";
            constexpr char kEwv[] = "ewv";
            constexpr char kEwf[] = "ewf";

            // The value of integer argument `name` of a prepared launch.
            bool stepArg(LaunchStep const& step, std::string const& name, int64_t& value)
            {
                int const index = step.kernel->argIndex(name);
                if(index < 0)
                {
                    return false;
                }
                auto const& a = step.kernel->args[index];
                if(a.pointer || a.offset + uint64_t(a.size) > step.args.size())
                {
                    return false;
                }
                if(a.size == 4)
                {
                    int32_t v;
                    std::memcpy(&v, step.args.data() + a.offset, 4);
                    value = v;
                    return true;
                }
                if(a.size == 8)
                {
                    std::memcpy(&value, step.args.data() + a.offset, 8);
                    return true;
                }
                return false;
            }

            std::string argNameAtOffset(PackKernel const& kernel, uint32_t offset)
            {
                for(auto const& a : kernel.args)
                {
                    if(a.offset == offset)
                    {
                        return a.name;
                    }
                }
                return "";
            }

            // The vector class of a launch of the generic elementwise kernel (the kernel pack's
            // ewv kernels: kernels.common.hiptensor_aot.jit_ew with its geometry as arguments, the
            // rule of the JIT's vector classes): each thread handles `vec` elements of D's fastest
            // mode (16 bytes of D's type) when D's stride there is 1, the extent of that mode a
            // multiple of `vec`, and the other strides of D and of every input read with vector
            // loads (stride 1 along that mode) multiples of `vec`. At least one input must be read
            // with vector loads. The generic launch is the fallback (unaligned buffers).
            hiptensorStatus_t planEwv(EwLayout const& lay,
                                      Pack const&     pack,
                                      Program const&  generic,
                                      Program&        program,
                                      CandidateFacts& facts,
                                      std::string&    reason)
            {
                if(generic.steps.size() != 1)
                {
                    reason = "the generic program has several launches";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const& g     = generic.steps.front();
                auto const& k     = *g.kernel;
                int const   nt    = static_cast<int>(lay.operands.size());
                int const   arity = nt - 1;
                if(k.attr("idx64") != 0 || k.attr("math") != 0 || k.attr("cplx") != 0)
                {
                    reason = "no vector class of 64-bit-offset, math or complex rows";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                std::vector<int64_t> bytes;
                for(auto const& op : lay.operands)
                {
                    std::string const key(1, char(std::tolower(op.name[0])));
                    bytes.push_back(int64_t(elementBytes(k.attrStr(key))));
                }
                int64_t const vec = 16 / bytes.back();
                int64_t       e0 = 0, sD0 = 0, lim0 = 0, nIn = 0;
                bool ok = stepArg(g, "e0", e0) && stepArg(g, "sD0", sD0) && stepArg(g, "n_in", nIn)
                          && sD0 == 1 && e0 % vec == 0 && e0 >= vec;
                if(ok && arity == 1)
                {
                    ok = stepArg(g, "lim0", lim0) && lim0 % vec == 0;
                }
                std::vector<bool> vector(nt, false);
                vector.back() = true;
                for(int t = 0; ok && t < nt; t++)
                {
                    auto const& n  = lay.operands[t].name;
                    int64_t     s0 = 0;
                    ok             = stepArg(g, "s" + n + "0", s0);
                    if(t + 1 < nt)
                    {
                        vector[t] = s0 == 1;
                    }
                    for(int i = 1; ok && vector[t] && i < kMaxRank; i++)
                    {
                        int64_t si = 0;
                        ok         = stepArg(g, "s" + n + std::to_string(i), si) && si % vec == 0;
                    }
                }
                for(auto const& p : g.pointers)
                {
                    auto const name = argNameAtOffset(k, p.offset);
                    for(int t = 0; ok && t < nt; t++)
                    {
                        if(name == lay.operands[t].name && vector[t])
                        {
                            ok = p.bytes % uint64_t(bytes[t] * vec) == 0;
                        }
                    }
                }
                bool const vectorInput
                    = std::find(vector.begin(), vector.end() - 1, true) != vector.end() - 1;
                if(!ok || !vectorInput)
                {
                    reason = "no vector class for this layout";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                std::string unit;
                for(int t = 0; t + 1 < nt; t++)
                {
                    unit += vector[t] ? 'u' : 'g';
                }
                std::vector<std::pair<std::string, std::string>> strs = lay.strs;
                strs.emplace_back("unit", unit);
                auto const* kernel = pack.find(kEwv, {{"arity", int64_t(arity)}}, strs);
                if(kernel == nullptr)
                {
                    reason = "the pack has no vector class " + unit + " for this data type row";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                LaunchStep step;
                step.kernel = kernel;
                ArgWriter                                    w(step);
                bool                                         good = true;
                std::vector<std::pair<BufferRole, uint32_t>> align;
                for(auto const& p : g.pointers)
                {
                    auto const name = argNameAtOffset(k, p.offset);
                    good            = good && w.pointer(name, p.role, p.bytes);
                    for(int t = 0; t < nt; t++)
                    {
                        if(name == lay.operands[t].name && vector[t])
                        {
                            align.emplace_back(p.role, uint32_t(bytes[t] * vec));
                        }
                    }
                }
                for(auto const& sc : g.scalars)
                {
                    good = good && w.scalar(argNameAtOffset(k, sc.offset), sc.role, sc.imag);
                }
                auto const magic = magicU31(static_cast<uint32_t>(e0 / vec));
                for(auto const& a : kernel->args)
                {
                    if(a.pointer || a.name == "alpha" || a.name == "beta" || a.name == "gamma")
                    {
                        continue;
                    }
                    int64_t v = 0;
                    if(a.name == "n_in")
                    {
                        v = nIn / vec;
                    }
                    else if(a.name == "e0")
                    {
                        v = e0 / vec;
                    }
                    else if(a.name == "m0")
                    {
                        good = good && w.setU32("m0", magic.first);
                        continue;
                    }
                    else if(a.name == "s0")
                    {
                        v = magic.second;
                    }
                    else if(!stepArg(g, a.name, v))
                    {
                        good = false;
                        break;
                    }
                    good = good && w.set(a.name, v);
                }
                if(!good)
                {
                    reason = "vector class arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                uint32_t const block = static_cast<uint32_t>(kernel->attr("block", 256));
                step.block           = dim3(block, 1, 1);
                step.grid
                    = dim3(static_cast<uint32_t>(ceilDiv(nIn / vec, block)), g.grid.y, g.grid.z);
                if(auto st = finishStep(step, w, reason); st != HIPTENSOR_STATUS_SUCCESS)
                {
                    return st;
                }
                Program prog;
                prog.pack            = &pack;
                prog.scalarType      = generic.scalarType;
                prog.nullStatus      = generic.nullStatus;
                prog.requiredScalars = generic.requiredScalars;
                prog.requiredBuffers = generic.requiredBuffers;
                prog.steps.push_back(std::move(step));
                prog.description   = kernel->name;
                prog.fallback      = std::make_shared<Program>(generic);
                prog.fallbackAlign = align;
                facts              = CandidateFacts{};
                facts.route        = Route::Elementwise;
                facts.main         = kernel;
                facts.ctas
                    = int64_t(prog.steps[0].grid.x) * prog.steps[0].grid.y * prog.steps[0].grid.z;
                facts.block = block;
                program     = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // The flat kernel (the kernel pack's ewf classes, kernels.common.hiptensor_aot.ewf) of
            // an elementwise operation whose canonical problem is one mode with stride 1 in every
            // tensor (every tensor contiguous, in the same order, no broadcast): element i of
            // every tensor at offset i, no index arithmetic, two 16-byte vectors of D per thread
            // with streaming accesses (gfx942 / gfx950 packs only). The element count must be a
            // multiple of the vector (and, as the generic launch's, below 2^31; a permutation reads
            // A everywhere: no padding limit). The generic launch is the fallback (buffers not
            // aligned to the vectors).
            hiptensorStatus_t planEwf(EwLayout const& lay,
                                      Pack const&     pack,
                                      Program const&  generic,
                                      Program&        program,
                                      CandidateFacts& facts,
                                      std::string&    reason)
            {
                if(generic.steps.size() != 1)
                {
                    reason = "the generic program has several launches";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const& g     = generic.steps.front();
                auto const& k     = *g.kernel;
                int const   nt    = static_cast<int>(lay.operands.size());
                int const   arity = nt - 1;
                if(k.attr("idx64") != 0 || k.attr("math") != 0 || k.attr("cplx") != 0)
                {
                    reason = "no flat class of 64-bit-offset, math or complex rows";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                bool ok = lay.dims.size() == 1;
                for(int t = 0; ok && t < nt; t++)
                {
                    ok = lay.dims[0].stride[t] == 1;
                }
                if(!ok)
                {
                    reason = "the tensors are not one contiguous run in the same order";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const        count = lay.dims[0].extent;
                std::vector<int64_t> bytes;
                for(auto const& op : lay.operands)
                {
                    std::string const key(1, char(std::tolower(op.name[0])));
                    bytes.push_back(int64_t(elementBytes(k.attrStr(key))));
                }
                int64_t const vec  = 16 / bytes.back();
                int64_t       lim0 = 0;
                ok                 = count % vec == 0 && count < kI32Max
                     && (arity != 1 || (stepArg(g, "lim0", lim0) && lim0 == count));
                for(auto const& p : g.pointers)
                {
                    auto const name = argNameAtOffset(k, p.offset);
                    for(int t = 0; ok && t < nt; t++)
                    {
                        if(name == lay.operands[t].name)
                        {
                            ok = p.bytes % uint64_t(bytes[t] * vec) == 0;
                        }
                    }
                }
                if(!ok)
                {
                    reason = "no flat class for this element count";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const* kernel = pack.find(kEwf, {{"arity", int64_t(arity)}}, lay.strs);
                if(kernel == nullptr)
                {
                    reason = "the pack has no flat class for this data type row";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                LaunchStep step;
                step.kernel = kernel;
                ArgWriter                                    w(step);
                bool                                         good = true;
                std::vector<std::pair<BufferRole, uint32_t>> align;
                for(auto const& p : g.pointers)
                {
                    auto const name = argNameAtOffset(k, p.offset);
                    good            = good && w.pointer(name, p.role, p.bytes);
                    for(int t = 0; t < nt; t++)
                    {
                        if(name == lay.operands[t].name)
                        {
                            align.emplace_back(p.role, uint32_t(bytes[t] * vec));
                        }
                    }
                }
                for(auto const& sc : g.scalars)
                {
                    good = good && w.scalar(argNameAtOffset(k, sc.offset), sc.role, sc.imag);
                }
                for(auto const& a : kernel->args)
                {
                    if(a.pointer || a.name == "alpha" || a.name == "beta" || a.name == "gamma")
                    {
                        continue;
                    }
                    int64_t v = 0;
                    if(a.name == "n")
                    {
                        v = count / vec;
                    }
                    else if(!stepArg(g, a.name, v)) // the operator codes
                    {
                        good = false;
                        break;
                    }
                    good = good && w.set(a.name, v);
                }
                if(!good)
                {
                    reason = "flat class arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                uint32_t const block  = static_cast<uint32_t>(kernel->attr("block", 256));
                int64_t const  unroll = kernel->attr("unroll", 1);
                step.block            = dim3(block, 1, 1);
                step.grid = dim3(static_cast<uint32_t>(ceilDiv(count / vec, block * unroll)), 1, 1);
                if(auto st = finishStep(step, w, reason); st != HIPTENSOR_STATUS_SUCCESS)
                {
                    return st;
                }
                Program prog;
                prog.pack            = &pack;
                prog.scalarType      = generic.scalarType;
                prog.nullStatus      = generic.nullStatus;
                prog.requiredScalars = generic.requiredScalars;
                prog.requiredBuffers = generic.requiredBuffers;
                prog.steps.push_back(std::move(step));
                prog.description   = kernel->name;
                prog.fallback      = std::make_shared<Program>(generic);
                prog.fallbackAlign = align;
                facts              = CandidateFacts{};
                facts.route        = Route::Elementwise;
                facts.main         = kernel;
                facts.ctas         = int64_t(prog.steps[0].grid.x);
                facts.block        = block;
                program            = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // The tiled kernel of elementwise operations whose inputs are fastest along another
            // mode than D (kernels.common.hiptensor_aot.ewt, ewt_plan): D's stride-1 mode o and
            // the stride-1 mode t of the first input fastest elsewhere form the tile plane, the
            // other modes the batch space. Each input is read through LDS along t (kind 2), with
            // vector loads along o (kind 1) or one load per element (kind 0). The generic program
            // is the fallback (buffers not aligned to the vectors).
            // `transposed` false: also a layout whose inputs are all fastest along D's mode or
            // gathered (the tile's other mode is then D's next fastest mode): one launch for the
            // modes past the generic kernel's slots.
            hiptensorStatus_t planEwt(EwLayout const& lay,
                                      Pack const&     pack,
                                      Program const&  generic,
                                      Program&        program,
                                      CandidateFacts& facts,
                                      std::string&    reason,
                                      bool            transposed = true)
            {
                auto const& dims  = lay.dims;
                int const   nt    = static_cast<int>(lay.operands.size());
                int const   nd    = nt - 1;
                auto        dtype = [&lay](std::string const& name) {
                    std::string const key(1, char(std::tolower(name[0])));
                    for(auto const& s : lay.strs)
                    {
                        if(s.first == key)
                        {
                            return s.second;
                        }
                    }
                    return std::string();
                };
                for(int k = 0; k < nd; k++)
                {
                    if(mathUnary(lay.operands[k].op))
                    {
                        reason = "the tiled elementwise kernel has no math operators";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                std::vector<std::pair<std::string, std::string>> strs;
                for(auto const& op : lay.operands)
                {
                    strs.emplace_back(std::string(1, char(std::tolower(op.name[0]))),
                                      dtype(op.name));
                }
                auto const* kernel = pack.find(kEwt, {{"arity", int64_t(nd)}, {"fold", 0}}, strs);
                if(kernel == nullptr)
                {
                    reason = "the pack has no tiled elementwise kernel for this data type row";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int o = -1, t = -1;
                for(int i = 0; i < int(dims.size()); i++)
                {
                    if(dims[i].stride[nd] == 1)
                    {
                        o = i;
                        break;
                    }
                }
                for(int k = 0; o >= 0 && t < 0 && k < nd; k++)
                {
                    for(int i = 0; i < int(dims.size()); i++)
                    {
                        if(i != o && dims[i].stride[k] == 1)
                        {
                            t = i;
                            break;
                        }
                    }
                }
                if(o >= 0 && t < 0 && !transposed)
                {
                    for(int i = 0; i < int(dims.size()); i++)
                    {
                        if(i != o && (t < 0 || dims[i].stride[nd] < dims[t].stride[nd]))
                        {
                            t = i;
                        }
                    }
                }
                if(o < 0 || t < 0)
                {
                    reason = "no input is fastest along another mode than the output";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const    vo = 16 / int64_t(elementBytes(dtype("D")));
                int64_t const    eo = dims[o].extent, et = dims[t].extent;
                std::vector<int> others;
                for(int i = 0; i < int(dims.size()); i++)
                {
                    if(i != o && i != t)
                    {
                        others.push_back(i);
                    }
                }
                int const slots = int(kernel->attr("slots_b"));
                bool      ok
                    = int(others.size()) <= slots && eo % vo == 0 && dims[t].stride[nd] % vo == 0;
                for(int i : others)
                {
                    ok = ok && dims[i].stride[nd] % vo == 0;
                }
                if(!ok)
                {
                    reason = "the output's layout does not keep the tiled kernel's vectors aligned";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                std::vector<int> kinds;
                // D's 16-byte stores (D in the workspace: an operand pack of a contraction)
                std::vector<std::pair<BufferRole, uint32_t>> align{{lay.operands[nd].role, 16u}};
                for(int k = 0; k < nd; k++)
                {
                    auto const&   op = lay.operands[k];
                    int64_t const es = int64_t(elementBytes(dtype(op.name)));
                    int64_t const vt = 16 / es;
                    auto          st = [&dims, k](int i) { return dims[i].stride[k]; };
                    bool          tk = st(t) == 1 && et % vt == 0 && st(o) % vt == 0;
                    for(int i : others)
                    {
                        tk = tk && st(i) % vt == 0;
                    }
                    if(tk)
                    {
                        kinds.push_back(2);
                        align.emplace_back(op.role, 16u);
                        continue;
                    }
                    int64_t const wv = std::min(vo, vt);
                    bool          vk = st(o) == 1 && st(t) % wv == 0;
                    for(int i : others)
                    {
                        vk = vk && st(i) % wv == 0;
                    }
                    if(vk)
                    {
                        kinds.push_back(1);
                        align.emplace_back(op.role, uint32_t(wv * es));
                        continue;
                    }
                    kinds.push_back(0);
                }
                if(transposed && std::find(kinds.begin(), kinds.end(), 2) == kinds.end())
                {
                    reason = "no input is read along the transposed mode";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                for(int k = 0; k < nt; k++)
                {
                    if(spanOf(dims, k) > kI32Max)
                    {
                        reason = "a tensor past the tiled kernel's 32-bit offsets";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    for(auto const& d : dims)
                    {
                        if(d.stride[k] < 0)
                        {
                            reason = "a negative stride";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                    }
                }
                int64_t const tt = kernel->attr("tile_t"), to = kernel->attr("tile_o");
                int64_t       tilesO = ceilDiv(eo, to), tilesT = ceilDiv(et, tt);
                int64_t       nb = 1;
                for(int i : others)
                {
                    nb *= dims[i].extent;
                }
                // a plane smaller than half a tile: the folded class, fo x ft planes of
                // consecutive batch indices per tile (kernels.common.hiptensor_aot.ewt.ewt_fold)
                int64_t fo = 1, ft = 1;
                if(eo <= to && et <= tt && (to / eo) * (tt / et) >= 2)
                {
                    auto const* folded
                        = pack.find(kEwt, {{"arity", int64_t(nd)}, {"fold", 1}}, strs);
                    if(folded != nullptr && (to / eo) * (tt / et) <= folded->attr("max_planes"))
                    {
                        kernel = folded;
                        fo     = to / eo;
                        ft     = tt / et;
                        tilesO = tilesT = 1;
                    }
                }
                bool const    fold   = fo * ft > 1;
                int64_t const groups = fold ? ceilDiv(nb, fo * ft) : nb;
                if(tilesO * tilesT > kI32Max || nb > kI32Max)
                {
                    reason = "the tiled kernel's grid is past its limits";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                Program prog;
                prog.pack            = &pack;
                prog.scalarType      = generic.scalarType;
                prog.nullStatus      = generic.nullStatus;
                prog.requiredScalars = generic.requiredScalars;
                prog.requiredBuffers = generic.requiredBuffers;
                LaunchStep step;
                step.kernel = kernel;
                ArgWriter   w(step);
                char const* scal[] = {"alpha", "beta", "gamma"};
                bool        good   = true;
                for(int k = 0; k < nt; k++)
                {
                    auto const& op = lay.operands[k];
                    good           = good && w.pointer(op.name, op.role, op.bytes)
                           && w.set("s" + op.name + "o", dims[o].stride[k])
                           && w.set("s" + op.name + "t", dims[t].stride[k]);
                    for(int i = 0; i < slots; i++)
                    {
                        int64_t const s = i < int(others.size()) && dims[others[i]].extent != 1
                                              ? dims[others[i]].stride[k]
                                              : 0;
                        good            = good && w.set("s" + op.name + std::to_string(i), s);
                    }
                    if(k == nd)
                    {
                        continue;
                    }
                    std::string const sname = op.name == "A"   ? scal[0]
                                              : op.name == "B" ? scal[1]
                                                               : scal[2];
                    good = good && w.scalar(sname, op.scale) && w.set("op" + op.name, op.op)
                           && w.set("k" + op.name, kinds[k]);
                }
                if(nd >= 2)
                {
                    good = good && w.set("opAB", lay.binaryOps[0]);
                }
                if(nd == 3)
                {
                    good = good && w.set("opABC", lay.binaryOps[1]);
                }
                for(int i = 0; i < slots; i++)
                {
                    int64_t const e = i < int(others.size()) ? dims[others[i]].extent : 1;
                    auto const    n = std::to_string(i);
                    good            = good && w.set("be" + n, e);
                    if(i < slots - 1)
                    {
                        good = good && setMagic(w, ("bm" + n).c_str(), ("bs" + n).c_str(), e);
                    }
                }
                int64_t const ystep = std::min<int64_t>(groups, kGridYZMax);
                good = good && w.set("eo", eo) && w.set("et", et) && w.set("tiles_o", tilesO)
                       && setMagic(w, "to_m", "to_s", tilesO) && w.set("nb", groups)
                       && w.set("ystep", ystep);
                if(fold)
                {
                    good = good && w.set("fo", fo) && w.set("ft", ft)
                           && setMagic(w, "eo_m", "eo_s", eo) && setMagic(w, "et_m", "et_s", et)
                           && w.set("nbt", nb);
                }
                if(!good)
                {
                    reason = "tiled elementwise arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                step.grid
                    = dim3(static_cast<uint32_t>(tilesO * tilesT), static_cast<uint32_t>(ystep), 1);
                step.block = dim3(static_cast<uint32_t>(kernel->attr("block", 256)), 1, 1);
                if(auto st = finishStep(step, w, reason); st != HIPTENSOR_STATUS_SUCCESS)
                {
                    return st;
                }
                prog.steps.push_back(std::move(step));
                prog.description   = kernel->name;
                prog.fallback      = std::make_shared<Program>(generic);
                prog.fallbackAlign = align;
                facts              = CandidateFacts{};
                facts.route        = Route::Elementwise;
                facts.main         = kernel;
                facts.ctas         = tilesO * tilesT * ystep;
                facts.block        = kernel->attr("block", 256);
                program            = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // The tiled kernel in place of the launches of `steps` from `first` on, the launches
            // of the generic kernel over `dims` (real data; one per piece when the modes exceed
            // its slots) when planEwt serves the layout: an operand pack or epilogue of a
            // contraction that reads an input along another mode than it writes, or any layout
            // the generic kernel needs several launches for. The generic launches are its
            // fallback (pointers not aligned to the tiled kernel's vectors).
            void tileLaunch(Pack const&                                             pack,
                            std::vector<std::pair<std::string, std::string>> const& strs,
                            std::vector<Dim> const&                                 dims,
                            std::vector<EwOperand> const&                           operands,
                            std::array<int, 2> const&                               binaryOps,
                            bool                                                    cplx,
                            std::vector<LaunchStep>&                                steps,
                            size_t                                                  first)
            {
                if(cplx || steps.size() <= first)
                {
                    return;
                }
                for(size_t i = first; i < steps.size(); i++)
                {
                    if(steps[i].kernel == nullptr || steps[i].kernel->family != kEw)
                    {
                        return;
                    }
                }
                EwLayout lay;
                lay.dims      = dims;
                lay.operands  = operands;
                lay.strs      = strs;
                lay.binaryOps = binaryOps;
                Program generic;
                generic.pack = &pack;
                generic.steps.assign(steps.begin() + first, steps.end());
                Program        tiled;
                CandidateFacts facts;
                std::string    why;
                bool const     pieces = generic.steps.size() > 1;
                if(planEwt(lay, pack, generic, tiled, facts, why, !pieces)
                       != HIPTENSOR_STATUS_SUCCESS
                   || tiled.steps.size() != 1)
                {
                    return;
                }
                LaunchStep step = std::move(tiled.steps.front());
                for(auto& ptr : step.pointers)
                {
                    for(auto const& [role, align] : tiled.fallbackAlign)
                    {
                        if(ptr.role == role)
                        {
                            ptr.align = std::max(ptr.align, align);
                        }
                    }
                }
                step.fallback
                    = std::make_shared<std::vector<LaunchStep> const>(std::move(generic.steps));
                steps.resize(first);
                steps.push_back(std::move(step));
            }

            // ------------------------------------------------------------------------ reduction

            // Reduction modes of the generic kernels: a wave per output reading the innermost
            // reduced mode (row), or a thread per output (col).
            constexpr int kRuleMode = -1;
            constexpr int kRowMode  = 0;
            constexpr int kColMode  = 1;

            // A piece of a reduction's kept or reduced modes that one launch addresses, and the
            // element offset of the piece in each tensor.
            struct ReductionPiece
            {
                std::vector<Dim>     dims;
                std::vector<int64_t> offsets;
            };

            // Splits canonical modes into pieces one launch addresses: a mode of more than 2^31
            // elements into pieces of 2^30, modes past the kernels' 8 slots one index of the
            // outermost mode at a time, more than 2^31 elements into halves of the outermost
            // mode. False past kMaxLaunches pieces.
            bool reductionPieces(std::vector<Dim> const&      dims,
                                 int                          tensors,
                                 std::vector<ReductionPiece>& out)
            {
                out.clear();
                std::function<bool(std::vector<Dim> const&, std::vector<int64_t> const&)> walk;
                walk = [&](std::vector<Dim> const& ds, std::vector<int64_t> const& base) {
                    auto pieces = [&](size_t i, int64_t piece) {
                        for(int64_t first = 0; first < ds[i].extent; first += piece)
                        {
                            std::vector<int64_t> off;
                            auto const           sub = restrictMode(
                                ds, i, first, std::min(piece, ds[i].extent - first), tensors, off);
                            for(int t = 0; t < tensors; t++)
                            {
                                off[t] += base[t];
                            }
                            if(!walk(sub, off))
                            {
                                return false;
                            }
                        }
                        return true;
                    };
                    for(size_t i = 0; i < ds.size(); i++)
                    {
                        if(ds[i].extent > kI32Max)
                        {
                            return pieces(i, kLongModePiece);
                        }
                    }
                    if(ds.size() > size_t(kMaxRank))
                    {
                        return pieces(ds.size() - 1, 1);
                    }
                    if(product(ds, 0, ds.size()) > kI32Max)
                    {
                        return pieces(ds.size() - 1, (ds.back().extent + 1) / 2);
                    }
                    if(int64_t(out.size()) >= kMaxLaunches)
                    {
                        return false;
                    }
                    out.push_back({ds, base});
                    return true;
                };
                return walk(dims, std::vector<int64_t>(tensors, 0));
            }

            // The API checks of a reduction and its canonical kept and reduced modes (strides 0 A,
            // 1 C, 2 D; innermost first): A's and D's data type names in `a` / `d`. A decline is
            // HIPTENSOR_STATUS_NOT_SUPPORTED with `reason`; an API error another status.
            hiptensorStatus_t reductionForm(Problem const&    problem,
                                            std::string&      a,
                                            std::string&      d,
                                            std::vector<Dim>& kept,
                                            std::vector<Dim>& red,
                                            std::string&      reason)
            {
                auto const& A = problem.t[0];
                auto const& C = problem.t[2];
                auto const& D = problem.t[3];
                if(problem.kind != OpKind::Reduction || !A.present || !D.present)
                {
                    reason = "not a reduction";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(!realScalars(problem, reason))
                {
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(!supportedBinary(problem.opReduce))
                {
                    reason = "reduction operator not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto opC = C.present ? C.unaryOp : HIPTENSOR_OP_IDENTITY;
                if(!supportedUnary(A.unaryOp) || !supportedUnary(opC)
                   || A.unaryOp == HIPTENSOR_OP_CONJ || opC == HIPTENSOR_OP_CONJ)
                {
                    reason = "unary operator not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                a = dtypeName(A.type);
                d = dtypeName(D.type);
                if(a.empty() || d.empty() || isComplex(A.type) || isComplex(D.type)
                   || (C.present && C.type != D.type))
                {
                    reason = "data type not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                // The API's reduction rules (the same as the other backend's): the data type rows,
                // C and D with one descriptor and one mode list, modes of C among A's, no mode twice.
                {
                    auto const cp = problem.compute;
                    bool const typeRow
                        = A.type == D.type
                          && ((A.type == HIPTENSOR_R_16F
                               && (cp == HIPTENSOR_COMPUTE_DESC_16F
                                   || cp == HIPTENSOR_COMPUTE_DESC_32F))
                              || (A.type == HIPTENSOR_R_16BF
                                  && (cp == HIPTENSOR_COMPUTE_DESC_16BF
                                      || cp == HIPTENSOR_COMPUTE_DESC_32F))
                              || (A.type == HIPTENSOR_R_32F && cp == HIPTENSOR_COMPUTE_DESC_32F)
                              || (A.type == HIPTENSOR_R_64F && cp == HIPTENSOR_COMPUTE_DESC_64F));
                    auto unique = [](std::vector<int32_t> modes) {
                        std::sort(modes.begin(), modes.end());
                        return std::adjacent_find(modes.begin(), modes.end()) == modes.end();
                    };
                    bool shape = C.present && C.extents == D.extents && C.strides == D.strides
                                 && C.modes == D.modes && unique(A.modes) && unique(C.modes)
                                 && A.numModes() >= C.numModes();
                    for(auto m : C.modes)
                    {
                        shape = shape && extentOf(A, m) != 0;
                    }
                    if(!typeRow || !shape)
                    {
                        reason = "reduction not supported by the API (data types, or C and D "
                                 "descriptors and modes)";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                // Strides: 0 A, 1 C, 2 D.
                kept.clear();
                red.clear();
                for(int i = 0; i < A.numModes(); i++)
                {
                    Dim     dim;
                    int64_t ds = 0, cs = 0;
                    dim.extent    = A.extents[i];
                    dim.stride[0] = A.strides[i];
                    auto e        = extentOf(D, A.modes[i], &ds);
                    if(e != 0)
                    {
                        if(e != dim.extent)
                        {
                            // an API error, not a decline
                            reason = "a mode of C and D has another extent than in A";
                            return HIPTENSOR_STATUS_INVALID_VALUE;
                        }
                        if(C.present && extentOf(C, A.modes[i], &cs) != e)
                        {
                            reason = "C and D modes differ";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                        dim.stride[1] = C.present ? cs : 0;
                        dim.stride[2] = ds;
                        kept.push_back(dim);
                    }
                    else
                    {
                        red.push_back(dim);
                    }
                }
                for(int i = 0; i < D.numModes(); i++)
                {
                    if(extentOf(A, D.modes[i]) == 0 && D.extents[i] != 1)
                    {
                        reason = "an output mode is not a mode of A";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                canonicalize(kept, 3, 0);
                canonicalize(red, 1, 0);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // `mode` is kRuleMode (row when the innermost reduced mode is the innermost mode of A),
            // kRowMode or kColMode. The split of the reduced elements targets `splitWaves` times the
            // waves (row) or blocks (col) that fill the device; 0 does not split.
            hiptensorStatus_t planReduction(Problem const&  problem,
                                            Pack const&     pack,
                                            uint64_t        workspaceLimit,
                                            int             mode,
                                            int             splitWaves,
                                            Program&        program,
                                            CandidateFacts& facts,
                                            std::string&    reason)
            {
                auto const&      A   = problem.t[0];
                auto const&      C   = problem.t[2];
                auto const       opC = C.present ? C.unaryOp : HIPTENSOR_OP_IDENTITY;
                std::string      a, d;
                std::vector<Dim> kept, red;
                if(auto status = reductionForm(problem, a, d, kept, red, reason);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                int64_t const nOutAll = product(kept, 0, kept.size());
                int64_t const nRedAll = product(red, 0, red.size());
                Dim const     kept0   = kept.empty() ? Dim{} : kept[0];
                Dim const     red0    = red.empty() ? Dim{} : red[0];
                bool const    ruleRow
                    = nRedAll > 1
                      && (nOutAll == 1 || std::llabs(red0.stride[0]) < std::llabs(kept0.stride[0]));
                bool const row = mode == kRuleMode ? ruleRow : mode == kRowMode;
                if(row && nRedAll <= 1)
                {
                    reason = "the row reduction needs a reduced mode";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                // What one launch cannot address is split into host-iterated pieces: pieces of
                // the kept modes are separate reductions (each writes its own outputs), pieces
                // of the reduced modes write partial results that the final kernel combines.
                std::vector<ReductionPiece> keptPieces, redPieces;
                if(!reductionPieces(kept, 3, keptPieces) || !reductionPieces(red, 1, redPieces)
                   || int64_t(keptPieces.size() * (redPieces.size() + 1)) > kMaxLaunches)
                {
                    reason = "the reduction needs more than " + std::to_string(kMaxLaunches)
                             + " launches of the generic kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                bool idx64 = spanOf(kept, 0) + spanOf(red, 0) > kI32Max || spanOf(kept, 1) > kI32Max
                             || spanOf(kept, 2) > kI32Max;
                bool const math = mathUnary(A.unaryOp) || mathUnary(opC);
                std::vector<std::pair<std::string, int64_t>>     ints{{"idx64", idx64 ? 1 : 0},
                                                                      {"math", math ? 1 : 0}};
                std::vector<std::pair<std::string, std::string>> strs{
                    {"mode", row ? "row" : "col"}, {"a", a}, {"d", d}};
                auto const* kernel = pack.find(kReduce, ints, strs);
                strs[0].second     = "final";
                auto const* final  = pack.find(kReduce, ints, strs);
                if(kernel == nullptr || final == nullptr)
                {
                    reason = "the pack has no reduction kernel for this data type row";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const  block  = kernel->attr("block", 256);
                int64_t const  fblock = final->attr("block", 256);
                int64_t const  cus    = std::max(1, problem.device.cuCount);
                uint64_t const wbytes = d == "f64" || a == "f64" ? 8 : 4;
                uint64_t const aBytes = elementBytes(a), dBytes = elementBytes(d);

                program                 = Program{};
                program.pack            = &pack;
                program.scalarType      = problem.scalarType;
                program.description     = kernel->name;
                program.nullStatus      = HIPTENSOR_STATUS_NOT_INITIALIZED;
                program.requiredScalars = {0, 1};
                program.requiredBuffers
                    = {{BufferRole::A, 0}, {BufferRole::C, 1}, {BufferRole::Out, -1}};

                // Kernel arguments of a launch over kept modes `kd` and reduced modes `rd` (8
                // each), with A, C, D offset by `off` elements and the partials at W + `wOff`.
                auto geometry = [&](ArgWriter&                       w,
                                    std::array<Dim, kMaxRank> const& kd,
                                    std::array<Dim, kMaxRank> const& rd,
                                    bool                             withRed) {
                    bool ok = true;
                    for(int i = 0; i < kMaxRank; i++)
                    {
                        auto n = std::to_string(i);
                        ok     = ok && w.set("ke" + n, kd[i].extent)
                             && w.set("kC" + n, kd[i].extent == 1 ? 0 : kd[i].stride[1])
                             && w.set("kD" + n, kd[i].extent == 1 ? 0 : kd[i].stride[2]);
                        if(withRed)
                        {
                            ok = ok && w.set("kA" + n, kd[i].extent == 1 ? 0 : kd[i].stride[0])
                                 && w.set("re" + n, rd[i].extent)
                                 && w.set("rA" + n, rd[i].extent == 1 ? 0 : rd[i].stride[0]);
                        }
                        if(i < kMaxRank - 1)
                        {
                            ok = ok
                                 && setMagic(
                                     w, ("km" + n).c_str(), ("ks" + n).c_str(), kd[i].extent);
                            if(withRed)
                            {
                                ok = ok
                                     && setMagic(
                                         w, ("rm" + n).c_str(), ("rs" + n).c_str(), rd[i].extent);
                            }
                        }
                    }
                    return ok;
                };
                auto common = [&](ArgWriter&                    w,
                                  bool                          withA,
                                  std::array<int64_t, 3> const& off,
                                  uint64_t                      wOff,
                                  int64_t                       nOut,
                                  int64_t                       split) {
                    bool ok = w.pointer("C",
                                        C.present ? BufferRole::C : BufferRole::None,
                                        uint64_t(off[1]) * dBytes)
                              && w.pointer("D", BufferRole::Out, uint64_t(off[2]) * dBytes)
                              && w.pointer("W", BufferRole::Workspace, wOff)
                              && w.scalar("alpha", ScalarRole::Alpha)
                              && w.scalar("beta", C.present ? ScalarRole::Beta : ScalarRole::Zero)
                              && w.set("opC", static_cast<int>(opC))
                              && w.set("opR", static_cast<int>(problem.opReduce))
                              && w.set("n_out", nOut) && w.set("split", split);
                    if(withA)
                    {
                        ok = ok && w.pointer("A", BufferRole::A, uint64_t(off[0]) * aBytes)
                             && w.set("opA", static_cast<int>(A.unaryOp));
                    }
                    return ok;
                };
                auto padded = [](std::vector<Dim> const& dims) {
                    std::array<Dim, kMaxRank> out{};
                    for(auto& x : out)
                    {
                        x = Dim{};
                    }
                    for(size_t i = 0; i < dims.size() && i < size_t(kMaxRank); i++)
                    {
                        out[i] = dims[i];
                    }
                    return out;
                };
                // The split of one launch: `split` ranges of `chunk` reduced trips (row) or
                // elements (col), within `maxSplit` partials.
                struct Split
                {
                    int64_t split = 1, chunk = 0, nchunk = 0, gridX = 0, gridY = 1, trips = 0;
                };
                auto splitOf = [&](std::array<Dim, kMaxRank> const& kd,
                                   std::array<Dim, kMaxRank> const& rd,
                                   int64_t                          maxSplit) {
                    Split   s;
                    int64_t nOut = 1, nRed = 1;
                    for(int i = 0; i < kMaxRank; i++)
                    {
                        nOut *= kd[i].extent;
                        nRed *= rd[i].extent;
                    }
                    if(row)
                    {
                        s.nchunk       = ceilDiv(rd[0].extent, kWave);
                        s.trips        = s.nchunk * (nRed / rd[0].extent);
                        int64_t target = int64_t(splitWaves) * 8 * cus * 4; // waves
                        s.split        = std::clamp<int64_t>(ceilDiv(target, nOut), 1, s.trips);
                        s.gridX        = ceilDiv(nOut, block / kWave);
                        maxSplit       = std::min<int64_t>(maxSplit, kGridYZMax);
                    }
                    else
                    {
                        int64_t outer  = nOut / kd[0].extent;
                        s.trips        = nRed;
                        s.gridX        = ceilDiv(kd[0].extent, block);
                        s.gridY        = std::min<int64_t>(outer, kGridYZMax);
                        int64_t target = int64_t(splitWaves) * 4 * cus;
                        s.split  = std::clamp<int64_t>(ceilDiv(target, s.gridX * s.gridY), 1, nRed);
                        maxSplit = std::min<int64_t>(maxSplit, kGridYZMax);
                    }
                    s.split = std::clamp<int64_t>(s.split, 1, std::max<int64_t>(maxSplit, 1));
                    s.chunk = ceilDiv(s.trips, s.split);
                    s.split = ceilDiv(s.trips, s.chunk);
                    return s;
                };
                // The main kernel's launch of one piece (split > 1 or `partial`: partials at W +
                // wOff; else D written).
                auto mainStep = [&](std::array<Dim, kMaxRank> const& kd,
                                    std::array<Dim, kMaxRank> const& rd,
                                    std::array<int64_t, 3> const&    off,
                                    Split const&                     s,
                                    int64_t                          splitArg,
                                    uint64_t                         wOff,
                                    std::string&                     why) {
                    int64_t nOut = 1;
                    for(auto const& x : kd)
                    {
                        nOut *= x.extent;
                    }
                    LaunchStep step;
                    step.kernel = kernel;
                    ArgWriter w(step);
                    bool      ok
                        = common(w, true, off, wOff, nOut, splitArg) && geometry(w, kd, rd, true);
                    if(row)
                    {
                        auto magic = magicU31(static_cast<uint32_t>(s.nchunk));
                        ok         = ok && w.set("nchunk", s.nchunk) && w.set("jchunk", s.chunk)
                             && w.setU32("jm", magic.first) && w.set("js", magic.second);
                        step.grid = dim3(
                            static_cast<uint32_t>(s.gridX), static_cast<uint32_t>(s.split), 1);
                    }
                    else
                    {
                        ok = ok && w.set("n_red", s.trips) && w.set("rchunk", s.chunk)
                             && w.set("ystep", s.gridY) && w.set("n_outer", nOut / kd[0].extent);
                        step.grid = dim3(static_cast<uint32_t>(s.gridX),
                                         static_cast<uint32_t>(s.gridY),
                                         static_cast<uint32_t>(s.split));
                    }
                    step.block = dim3(static_cast<uint32_t>(block), 1, 1);
                    if(!ok)
                    {
                        why = "reduction arguments do not fit the kernel arguments";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    if(auto status = finishStep(step, w, why); status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                    program.steps.push_back(std::move(step));
                    return HIPTENSOR_STATUS_SUCCESS;
                };
                auto finalStep = [&](std::array<Dim, kMaxRank> const& kd,
                                     std::array<int64_t, 3> const&    off,
                                     int64_t                          split,
                                     std::string&                     why) {
                    int64_t nOut = 1;
                    for(auto const& x : kd)
                    {
                        nOut *= x.extent;
                    }
                    LaunchStep fin;
                    fin.kernel = final;
                    ArgWriter fw(fin);
                    if(!(common(fw, false, off, 0, nOut, split) && geometry(fw, kd, kd, false)))
                    {
                        why = "reduction arguments do not fit the kernel arguments";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    fin.grid  = dim3(static_cast<uint32_t>(ceilDiv(nOut, fblock)), 1, 1);
                    fin.block = dim3(static_cast<uint32_t>(fblock), 1, 1);
                    if(auto status = finishStep(fin, fw, why); status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                    program.steps.push_back(std::move(fin));
                    return HIPTENSOR_STATUS_SUCCESS;
                };

                uint64_t workspace  = 0;
                int64_t  firstSplit = 1, firstChunk = 0;
                bool     usedFinal = false;
                for(size_t kp = 0; kp < keptPieces.size(); kp++)
                {
                    auto const& kpiece = keptPieces[kp];
                    auto const  kd     = padded(kpiece.dims);
                    int64_t     nOut   = 1;
                    for(auto const& x : kd)
                    {
                        nOut *= x.extent;
                    }
                    // partials that fit the workspace limit and the i32 partial index
                    // (the limit rounded down to the workspace alignment, which the program's
                    // workspace size is rounded up to)
                    uint64_t const usable      = workspaceLimit / kWorkspaceAlign * kWorkspaceAlign;
                    int64_t const  maxPartials = std::min<int64_t>(
                        kI32Max / nOut,
                        static_cast<int64_t>(std::min<uint64_t>(usable / (uint64_t(nOut) * wbytes),
                                                                uint64_t(kI32Max))));
                    std::vector<Split> splits;
                    int64_t            total = 0;
                    for(auto const& rpiece : redPieces)
                    {
                        splits.push_back(splitOf(kd, padded(rpiece.dims), maxPartials));
                        total += splits.back().split;
                    }
                    bool const partial = redPieces.size() > 1;
                    if(partial && total > maxPartials)
                    {
                        // fewer, longer ranges until the partials of every piece fit
                        if(int64_t(redPieces.size()) > maxPartials)
                        {
                            reason = "the partial results of the reduction need more workspace "
                                     "than the limit";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                        total              = 0;
                        int64_t const each = maxPartials / int64_t(redPieces.size());
                        for(size_t rp = 0; rp < redPieces.size(); rp++)
                        {
                            splits[rp] = splitOf(kd, padded(redPieces[rp].dims), each);
                            total += splits[rp].split;
                        }
                    }
                    int64_t const splitArg
                        = partial ? std::max<int64_t>(total, 2) : splits[0].split;
                    uint64_t base = 0; // partials of the pieces before this one
                    for(size_t rp = 0; rp < redPieces.size(); rp++)
                    {
                        auto const&            rpiece = redPieces[rp];
                        std::array<int64_t, 3> off{kpiece.offsets[0] + rpiece.offsets[0],
                                                   kpiece.offsets[1],
                                                   kpiece.offsets[2]};
                        std::string            why;
                        auto                   status = mainStep(kd,
                                               padded(rpiece.dims),
                                               off,
                                               splits[rp],
                                               partial ? splitArg : splits[rp].split,
                                               base * uint64_t(nOut) * wbytes,
                                               why);
                        if(status != HIPTENSOR_STATUS_SUCCESS)
                        {
                            reason = why;
                            return status;
                        }
                        base += uint64_t(splits[rp].split);
                    }
                    int64_t const partials = partial ? total : splits[0].split;
                    if(partials > 1)
                    {
                        std::array<int64_t, 3> off{0, kpiece.offsets[1], kpiece.offsets[2]};
                        std::string            why;
                        if(auto status = finalStep(kd, off, partials, why);
                           status != HIPTENSOR_STATUS_SUCCESS)
                        {
                            reason = why;
                            return status;
                        }
                        usedFinal = true;
                        workspace
                            = std::max(workspace, uint64_t(partials) * uint64_t(nOut) * wbytes);
                    }
                    if(kp == 0)
                    {
                        firstSplit = partials;
                        firstChunk = splits[0].chunk;
                    }
                }
                if(usedFinal)
                {
                    program.description += "+" + final->name;
                }
                if(program.steps.size() > 2)
                {
                    program.description += "x" + std::to_string(program.steps.size());
                }
                program.workspaceBytes = alignUp(workspace, kWorkspaceAlign);
                if(program.workspaceBytes > workspaceLimit)
                {
                    reason = "the partial results of the reduction need more workspace than the "
                             "limit";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                facts           = CandidateFacts{};
                facts.route     = Route::Reduction;
                facts.body      = row ? kRowMode : kColMode;
                facts.M         = nOutAll;
                facts.K         = nRedAll;
                facts.modesM    = static_cast<int>(kept.size());
                facts.modesK    = static_cast<int>(red.size());
                facts.inner     = red0.extent;
                facts.innerKept = kept0.extent;
                facts.unitA     = red0.extent > 1 && std::llabs(red0.stride[0]) == 1     ? 3
                                  : kept0.extent > 1 && std::llabs(kept0.stride[0]) == 1 ? 1
                                                                                         : 0;
                facts.split     = firstSplit;
                facts.chunk     = firstChunk;
                facts.block     = block;
                facts.ctas      = int64_t(program.steps[0].grid.x) * program.steps[0].grid.y
                             * program.steps[0].grid.z;
                facts.main      = kernel;
                facts.operandOp = A.unaryOp != HIPTENSOR_OP_IDENTITY;
                facts.bilinear  = C.present;
                facts.bytes     = uint64_t(nOutAll) * uint64_t(nRedAll) * aBytes
                              + uint64_t(nOutAll) * dBytes * (C.present ? 2 : 1) + 2 * workspace;
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // Fills a slot set of a class launch (rreduce_host._slots): `dims` innermost first,
            // padded to `n` slots with extent 1, magic (0, 0) and stride 0.
            bool rSlots(ArgWriter&                               w,
                        std::string const&                       prefix,
                        std::vector<std::vector<int64_t>> const& dims,
                        int64_t                                  n,
                        std::vector<std::string> const&          tensors)
            {
                bool ok = int64_t(dims.size()) <= n;
                for(int64_t i = 0; ok && i < n; i++)
                {
                    std::vector<int64_t> x(tensors.size() + 1, 0);
                    x[0] = 1;
                    if(i < int64_t(dims.size()))
                    {
                        x = dims[i];
                    }
                    auto const s = std::to_string(i);
                    ok           = ok && w.set(prefix + "e" + s, x[0]);
                    for(size_t t = 0; ok && t < tensors.size(); t++)
                    {
                        ok = w.set(prefix + tensors[t] + s, x[0] == 1 ? 0 : x[1 + t]);
                    }
                    if(ok && i < n - 1)
                    {
                        ok = setMagic(
                            w, (prefix + "m" + s).c_str(), (prefix + "s" + s).c_str(), x[0]);
                    }
                }
                return ok;
            }

            // The pack class of knobs `k` (nullptr when the pack has none).
            PackKernel const*
                rClass(Pack const& pack, RProblem const& p, RKnobs const& k, bool una, bool idx64)
            {
                int64_t const nr = rSlotCount(k.row ? rChunkDims(p, k.vec).size()
                                                    : std::max<size_t>(1, p.red.size()));
                if(nr < 0)
                {
                    return nullptr;
                }
                return pack.find(
                    kRreduce,
                    {{"nr", nr},
                     {"block", k.block},
                     {"vec", k.vec},
                     {"lanes", k.lanes},
                     {"unroll", k.unroll},
                     {"rows", k.row ? 0 : k.rows},
                     {"nt", k.nt ? 1 : 0},
                     {"una", una ? 1 : 0},
                     {"idx64", idx64 ? 1 : 0}},
                    {{"mode", k.row ? "row" : "col"}, {"a", p.a}, {"d", p.d}, {"op", p.op}});
            }

            // rreduce_host.plan: the launches of `k` (main class kernel, then the finalize when
            // split > 1) with A, C, D, W of the problem.
            hiptensorStatus_t planRreduce(Problem const&    problem,
                                          Pack const&       pack,
                                          RProblem const&   p,
                                          RKnobs const&     k,
                                          PackKernel const* kernel,
                                          bool              idx64,
                                          Program&          program,
                                          std::string&      reason)
            {
                auto const&   A    = problem.t[0];
                auto const&   C    = problem.t[2];
                auto const    opC  = C.present ? C.unaryOp : HIPTENSOR_OP_IDENTITY;
                bool const    una  = kernel->attr("una", 0) != 0;
                int64_t const nOut = rExtents(p.kept);
                int64_t       nIdx = 0, step = 0, per = 0, nIt = 0;
                rLoop(p, k, nIdx, step, per, nIt);

                program                 = Program{};
                program.pack            = &pack;
                program.scalarType      = problem.scalarType;
                program.description     = kernel->name;
                program.nullStatus      = HIPTENSOR_STATUS_NOT_INITIALIZED;
                program.requiredScalars = {0, 1};
                program.requiredBuffers
                    = {{BufferRole::A, 0}, {BufferRole::C, 1}, {BufferRole::Out, -1}};

                auto common = [&](ArgWriter& w) {
                    return w.pointer("C", C.present ? BufferRole::C : BufferRole::None)
                           && w.pointer("D", BufferRole::Out)
                           && w.pointer("W", BufferRole::Workspace)
                           && w.scalar("alpha", ScalarRole::Alpha)
                           && w.scalar("beta", C.present ? ScalarRole::Beta : ScalarRole::Zero)
                           && w.set("opC", static_cast<int>(opC)) && w.set("n_out", nOut);
                };
                LaunchStep step0;
                step0.kernel = kernel;
                ArgWriter w(step0);
                bool      ok = w.pointer("A", BufferRole::A) && common(w)
                          && (!una || w.set("opA", static_cast<int>(A.unaryOp)))
                          && w.set("n_idx", nIdx) && w.set("per", per) && w.set("n_it", nIt)
                          && w.set("split", k.split);
                std::vector<std::vector<int64_t>> finalKept;
                int64_t const                     nr = kernel->attr("nr", 0);
                if(k.row)
                {
                    std::vector<std::vector<int64_t>> kd, rd;
                    for(auto const& x : p.kept)
                    {
                        kd.push_back({x[0], x[1], x[2]});
                        finalKept.push_back({x[0], x[2]});
                    }
                    for(auto const& x : rChunkDims(p, k.vec))
                    {
                        rd.push_back({x[0], x[1]});
                    }
                    ok = ok && rSlots(w, "k", kd, kMaxRank, {"A", "D"})
                         && rSlots(w, "r", rd, nr, {"A"});
                }
                else
                {
                    size_t const                      ng = rColGroup(p);
                    std::vector<std::vector<int64_t>> gd, xd, rd;
                    int64_t                           m = 1;
                    for(size_t i = 0; i < p.kept.size(); i++)
                    {
                        auto const& x = p.kept[i];
                        if(i < ng)
                        {
                            gd.push_back({x[0], x[2]});
                            m *= x[0];
                        }
                        else
                        {
                            xd.push_back({x[0], x[1], x[2]});
                        }
                    }
                    for(auto const& x : gd)
                    {
                        finalKept.push_back(x);
                    }
                    for(auto const& x : xd)
                    {
                        finalKept.push_back({x[0], x[2]});
                    }
                    for(auto const& x : p.red)
                    {
                        rd.push_back({x[0], x[1]});
                    }
                    if(rd.empty())
                    {
                        rd.push_back({1, 0});
                    }
                    int64_t const tiles = ceilDiv(m, k.lanes * k.vec);
                    ok = ok && w.set("m", m) && w.set("ka", p.kept[0][1]) && w.set("tiles", tiles)
                         && setMagic(w, "tm", "ts", tiles) && w.set("n_items", (nOut / m) * tiles)
                         && rSlots(w, "g", gd, kMaxRank, {"D"})
                         && rSlots(w, "x", xd, kMaxRank, {"A", "D"})
                         && rSlots(w, "r", rd, nr, {"A"});
                }
                step0.grid
                    = dim3(static_cast<uint32_t>(rGridX(p, k)), static_cast<uint32_t>(k.split), 1);
                step0.block = dim3(static_cast<uint32_t>(k.block), 1, 1);
                if(!ok)
                {
                    reason = "tuned reduction arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(auto status = finishStep(step0, w, reason); status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                program.steps.push_back(std::move(step0));
                uint64_t const cb = p.a == "f64" || p.d == "f64" ? 8 : 4;
                if(k.split > 1)
                {
                    int64_t const lanes = rFinalLanes(nOut, k.split);
                    auto const*   fin   = pack.find(kRreduceFinal,
                                                    {{"lanes", lanes}, {"idx64", idx64 ? 1 : 0}},
                                                    {{"a", p.a}, {"d", p.d}, {"op", p.op}});
                    if(fin == nullptr)
                    {
                        reason = "the pack has no finalize class for this reduction";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    LaunchStep step1;
                    step1.kernel = fin;
                    ArgWriter fw(step1);
                    bool      fok = common(fw) && fw.set("split", k.split)
                               && rSlots(fw, "k", finalKept, kMaxRank, {"D"});
                    // `W` of the finalize is readonly; common() set it like the main step's
                    int64_t const fblock = fin->attr("block", 256);
                    step1.grid  = dim3(static_cast<uint32_t>(ceilDiv(nOut, fblock / lanes)), 1, 1);
                    step1.block = dim3(static_cast<uint32_t>(fblock), 1, 1);
                    if(!fok)
                    {
                        reason = "tuned reduction finalize arguments do not fit";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    if(auto status = finishStep(step1, fw, reason);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                    program.steps.push_back(std::move(step1));
                    program.description += "+" + fin->name;
                    program.workspaceBytes
                        = alignUp(uint64_t(nOut) * uint64_t(k.split) * cb, kWorkspaceAlign);
                }
                // vector accesses: A's loads; col classes store D (unit stride) and the
                // partials as vectors
                uint64_t const eb = elementBytes(p.a);
                if(k.vec > 1)
                {
                    program.fallbackAlign.emplace_back(BufferRole::A, uint32_t(k.vec * eb));
                    if(!k.row)
                    {
                        program.fallbackAlign.emplace_back(BufferRole::Out,
                                                           uint32_t(k.vec * elementBytes(p.d)));
                        if(k.split > 1)
                        {
                            program.fallbackAlign.emplace_back(BufferRole::Workspace,
                                                               uint32_t(k.vec * cb));
                        }
                    }
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // The tuned reduction classes and prototypes of a pack, read once per pack (Pack::load
            // has checked the reduce.* routes with rCheckRoutes, so every record parses)
            struct RTables
            {
                std::vector<RClass>            classes;
                std::vector<PackKernel const*> kernels; // per class
                std::vector<RProto>            protos;
            };

            RTables const& rTables(Pack const& pack)
            {
                static std::mutex                     lock;
                static std::map<Pack const*, RTables> cache;
                std::lock_guard<std::mutex>           guard(lock);
                auto                                  it = cache.find(&pack);
                if(it != cache.end())
                {
                    return it->second;
                }
                RTables t;
                for(auto const& kernel : pack.kernels())
                {
                    if(kernel.family != kRreduce)
                    {
                        continue;
                    }
                    RClass c;
                    c.name   = kernel.name;
                    c.mode   = kernel.attrStr("mode");
                    c.a      = kernel.attrStr("a");
                    c.d      = kernel.attrStr("d");
                    c.op     = kernel.attrStr("op");
                    c.nr     = kernel.attr("nr", 0);
                    c.block  = kernel.attr("block", 0);
                    c.vec    = kernel.attr("vec", 0);
                    c.lanes  = kernel.attr("lanes", 0);
                    c.unroll = kernel.attr("unroll", 0);
                    c.rows   = kernel.attr("rows", 0);
                    c.nt     = kernel.attr("nt", 0) != 0;
                    c.una    = kernel.attr("una", 0) != 0;
                    c.idx64  = kernel.attr("idx64", 0) != 0;
                    t.classes.push_back(c);
                    t.kernels.push_back(&kernel);
                }
                int64_t const n
                    = std::strtoll(pack.route("reduce.proto.count", "0").c_str(), nullptr, 10);
                for(int64_t i = 0; i < n; i++)
                {
                    RProto r;
                    if(rParseProto(pack.route("reduce.proto." + std::to_string(i)), r))
                    {
                        t.protos.push_back(r);
                    }
                }
                return cache.emplace(&pack, std::move(t)).first->second;
            }

            // The tuned reduction candidates of a problem: its tuned route (table problems), else
            // up to kRreduceCandidates class launches of the untuned choice. A split whose
            // partial results exceed the workspace limit is lowered to the largest power of two
            // that fits (no split when the limit is zero). Every program falls back to the
            // generic no-split program when a buffer misses the alignment of its vector accesses.
            hiptensorStatus_t enumerateTunedReduction(Problem const&          problem,
                                                      Pack const&             pack,
                                                      uint64_t                workspaceLimit,
                                                      std::vector<Candidate>& out)
            {
                std::string      a, d, why;
                std::vector<Dim> kept, red;
                if(reductionForm(problem, a, d, kept, red, why) != HIPTENSOR_STATUS_SUCCESS)
                {
                    return HIPTENSOR_STATUS_SUCCESS; // the generic candidates report it
                }
                auto const& A   = problem.t[0];
                auto const& C   = problem.t[2];
                auto const  opA = A.unaryOp;
                auto const  opC = C.present ? C.unaryOp : HIPTENSOR_OP_IDENTITY;
                if(mathUnary(opA) || mathUnary(opC) || a != d)
                {
                    return HIPTENSOR_STATUS_SUCCESS;
                }
                bool const una = opA != HIPTENSOR_OP_IDENTITY && opA != HIPTENSOR_OP_CONJ;
                RProblem   p;
                p.a  = a;
                p.d  = d;
                p.eb = int64_t(elementBytes(a));
                switch(problem.opReduce)
                {
                case HIPTENSOR_OP_ADD:
                    p.op = "add";
                    break;
                case HIPTENSOR_OP_MUL:
                    p.op = "mul";
                    break;
                case HIPTENSOR_OP_MAX:
                    p.op = "max";
                    break;
                case HIPTENSOR_OP_MIN:
                    p.op = "min";
                    break;
                default:
                    return HIPTENSOR_STATUS_SUCCESS;
                }
                for(auto const& x : kept)
                {
                    p.kept.push_back({x.extent, x.stride[0], x.stride[2]});
                }
                for(auto const& x : red)
                {
                    p.red.push_back({x.extent, x.stride[0]});
                }
                bool const    streaming = pack.arch() == "gfx942" || pack.arch() == "gfx950";
                int64_t const cus       = std::max(1, problem.device.cuCount);

                // the generic no-split program: the fallback of a misaligned buffer
                Program        generic;
                CandidateFacts gfacts;
                std::string    gwhy;
                if(planReduction(problem, pack, workspaceLimit, kRuleMode, 0, generic, gfacts, gwhy)
                   != HIPTENSOR_STATUS_SUCCESS)
                {
                    return HIPTENSOR_STATUS_SUCCESS;
                }
                auto fallback = std::make_shared<Program>(std::move(generic));

                // `kernel`: the class to run `k` with (nullptr: the class of exactly the problem's
                // slots and offset width)
                auto make
                    = [&](RKnobs const& k, bool tuned, PackKernel const* kernel, Candidate& c) {
                          if(!rCheck(p, k, streaming).empty())
                          {
                              return false;
                          }
                          if(kernel == nullptr)
                          {
                              kernel = rClass(pack, p, k, una, rIdx64(p, k));
                          }
                          if(kernel == nullptr)
                          {
                              return false;
                          }
                          bool const  idx64 = kernel->attr("idx64", 0) != 0;
                          std::string reason;
                          if(planRreduce(problem, pack, p, k, kernel, idx64, c.program, reason)
                             != HIPTENSOR_STATUS_SUCCESS)
                          {
                              return false;
                          }
                          if(!c.program.fallbackAlign.empty())
                          {
                              c.program.fallback = fallback;
                              c.program.workspaceBytes
                                  = std::max(c.program.workspaceBytes, fallback->workspaceBytes);
                          }
                          if(c.program.workspaceBytes > workspaceLimit)
                          {
                              return false;
                          }
                          auto& f     = c.facts;
                          f.route     = Route::TunedReduction;
                          f.body      = k.row ? kRowMode : kColMode;
                          f.M         = rExtents(p.kept);
                          f.K         = p.red.empty() ? 1 : rExtents(p.red);
                          f.modesM    = static_cast<int>(p.kept.size());
                          f.modesK    = static_cast<int>(p.red.size());
                          f.inner     = p.red.empty() ? 1 : p.red[0][0];
                          f.innerKept = p.kept.empty() ? 1 : p.kept[0][0];
                          f.split     = k.split;
                          f.block     = k.block;
                          f.ctas      = rGridX(p, k) * k.split;
                          f.main      = kernel;
                          f.operandOp = una;
                          f.bilinear  = C.present;
                          f.bytes     = uint64_t(f.M) * uint64_t(f.K) * uint64_t(p.eb)
                                    + uint64_t(f.M) * elementBytes(p.d) * (C.present ? 2 : 1)
                                    + 2 * c.program.workspaceBytes;
                          c.key = std::string("rreduce:") + (tuned ? "tuned:" : "") + kernel->name
                                  + ":split" + std::to_string(k.split);
                          c.tuned = tuned;
                          return true;
                      };

                // 1. rocKE's reduction route table (problems with opA IDENTITY): the tuned knobs,
                // the split lowered to the workspace limit
                if(!una)
                {
                    auto const key = rKey(p);
                    char       hex[17];
                    snprintf(
                        hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(fnv1a64(key)));
                    std::string const route = std::string("reduce.tuned.") + hex;
                    RKnobs            k;
                    if(pack.route(route + ".key") == key && rParseRecord(pack.route(route), k))
                    {
                        k.split = rCapSplit(p, k.split, workspaceLimit);
                        Candidate c;
                        if(make(k, true, nullptr, c))
                        {
                            c.staticRank = static_cast<int>(out.size());
                            out.push_back(std::move(c));
                            return HIPTENSOR_STATUS_SUCCESS;
                        }
                    }
                }

                // 2. the untuned choice (rocke_rreduce_rule.hpp untunedChoices, rocKE's
                // rreduce_host.untuned_choices), the first preferred
                auto const&  tables = rTables(pack);
                size_t const first  = out.size();
                for(auto const& choice : untunedChoices(
                        p, tables.classes, tables.protos, una, cus, streaming, workspaceLimit))
                {
                    Candidate c;
                    if(!make(choice.knobs, false, tables.kernels[choice.cls], c))
                    {
                        continue;
                    }
                    c.preferred  = out.size() == first;
                    c.staticRank = static_cast<int>(out.size());
                    out.push_back(std::move(c));
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }
        } // namespace

        // ---------------------------------------------------------------------- contraction

        namespace
        {
            // A group of contraction modes (M, N, K or L) in a chosen order (innermost first):
            // per mode its extent and its stride in A, B, C, D.
            struct Group
            {
                std::vector<int32_t>                modes;
                std::vector<int64_t>                extent;
                std::vector<std::array<int64_t, 4>> stride; // A, B, C, D (0 when absent)
                int64_t                             size() const
                {
                    int64_t p = 1;
                    for(auto e : extent)
                    {
                        p *= e;
                    }
                    return p;
                }
                // Orders the modes by the strides of tensor `t`.
                void orderBy(int t)
                {
                    std::vector<size_t> idx(modes.size());
                    std::iota(idx.begin(), idx.end(), 0);
                    std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
                        return stride[x][t] < stride[y][t];
                    });
                    Group g;
                    for(auto i : idx)
                    {
                        g.modes.push_back(modes[i]);
                        g.extent.push_back(extent[i]);
                        g.stride.push_back(stride[i]);
                    }
                    *this = g;
                }
                // Packed strides of the modes in this order, times `unit`.
                std::vector<int64_t> packed(int64_t unit) const
                {
                    std::vector<int64_t> out;
                    int64_t              s = unit;
                    for(auto e : extent)
                    {
                        out.push_back(s);
                        s *= e;
                    }
                    return out;
                }
                // The stride of tensor `t` if the modes form one contiguous run in this order
                // (0 for an empty group), else -1.
                int64_t mergedStride(int t) const
                {
                    if(modes.empty())
                    {
                        return 0;
                    }
                    for(size_t i = 1; i < modes.size(); i++)
                    {
                        if(stride[i][t] != stride[i - 1][t] * extent[i - 1])
                        {
                            return -1;
                        }
                    }
                    return stride[0][t];
                }
            };

            struct Classified
            {
                Group M, N, K, L;
            };

            hiptensorStatus_t classify(Problem const& p, Classified& out, std::string& reason)
            {
                auto const&          A = p.t[0];
                auto const&          B = p.t[1];
                auto const&          C = p.t[2];
                auto const&          D = p.t[3];
                std::vector<int32_t> all;
                for(auto const* t : {&A, &B, &D})
                {
                    for(auto m : t->modes)
                    {
                        if(std::find(all.begin(), all.end(), m) == all.end())
                        {
                            all.push_back(m);
                        }
                    }
                }
                for(auto mode : all)
                {
                    std::array<int64_t, 4> s{};
                    int64_t                ea = extentOf(A, mode, &s[0]);
                    int64_t                eb = extentOf(B, mode, &s[1]);
                    int64_t                ec = C.present ? extentOf(C, mode, &s[2]) : 0;
                    int64_t                ed = extentOf(D, mode, &s[3]);
                    int64_t                e  = std::max({ea, eb, ed});
                    for(auto x : {ea, eb, ed})
                    {
                        if(x != 0 && x != e)
                        {
                            reason = "the extents of a mode differ between tensors";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                    }
                    if(C.present && ec != ed)
                    {
                        reason = "C and D modes differ";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    if(e == 1)
                    {
                        continue;
                    }
                    Group* g = nullptr;
                    if(ea && eb && ed)
                    {
                        g = &out.L;
                    }
                    else if(ea && ed)
                    {
                        g = &out.M;
                    }
                    else if(eb && ed)
                    {
                        g = &out.N;
                    }
                    else if(ea && eb)
                    {
                        g = &out.K;
                    }
                    else
                    {
                        reason = "a mode appears in only one tensor";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    g->modes.push_back(mode);
                    g->extent.push_back(e);
                    g->stride.push_back(s);
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // Adds the launch that packs operand `which` (0 = A, 1 = B) into the workspace as
            // [L][rows padded to rowsP][K padded to kP] elements of `toType`, with zero padding.
            hiptensorStatus_t packOperand(Pack const&        pack,
                                          Problem const&     p,
                                          Classified const&  c,
                                          int                which,
                                          std::string const& toType,
                                          bool               cplx,
                                          int64_t            rowsP,
                                          int64_t            kP,
                                          uint64_t           offset,
                                          Program&           program,
                                          std::string&       reason)
            {
                auto const&      X    = p.t[which];
                Group const&     R    = which == 0 ? c.M : c.N;
                int64_t const    rows = R.size();
                int64_t const    k    = c.K.size();
                uint64_t const   es   = elementBytes(toType, cplx);
                std::vector<Dim> dims;
                auto             add = [&](Group const& g, std::vector<int64_t> const& outStrides) {
                    for(size_t i = 0; i < g.modes.size(); i++)
                    {
                        Dim d;
                        d.extent    = g.extent[i];
                        d.stride[0] = g.stride[i][which];
                        d.stride[1] = outStrides[i];
                        dims.push_back(d);
                    }
                };
                add(c.K, c.K.packed(1));
                add(R, R.packed(kP));
                add(c.L, c.L.packed(rowsP * kP));
                canonicalize(dims, 2, 1);
                EwOperand in;
                in.name  = "A";
                in.role  = which == 0 ? BufferRole::A : BufferRole::B;
                in.scale = ScalarRole::One;
                in.op    = static_cast<int>(X.unaryOp);
                EwOperand out;
                out.name  = "D";
                out.role  = BufferRole::Workspace;
                out.bytes = offset;
                size_t const                                           first = program.steps.size();
                std::vector<std::pair<std::string, std::string>> const strs{
                    {"a", dtypeName(X.type)}, {"d", toType}};
                auto status = ewSteps(pack,
                                      {{"arity", 1}, {"cplx", cplx ? 1 : 0}},
                                      strs,
                                      dims,
                                      {in, out},
                                      {0, 0},
                                      cplx,
                                      program.steps,
                                      reason);
                if(status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                tileLaunch(pack, strs, dims, {in, out}, {0, 0}, cplx, program.steps, first);
                int64_t const batches = c.L.size();
                if(kP > k)
                {
                    program.memsets.push_back({offset + uint64_t(k) * es,
                                               uint64_t(kP) * es,
                                               uint64_t(kP - k) * es,
                                               uint64_t(batches * rowsP)});
                }
                if(rowsP > rows)
                {
                    program.memsets.push_back({offset + uint64_t(rows * kP) * es,
                                               uint64_t(rowsP * kP) * es,
                                               uint64_t((rowsP - rows) * kP) * es,
                                               uint64_t(batches)});
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // Adds the epilogue D = alpha * T + beta * opC(C) from a packed [L][M][N] result T
            // (row strides tM, tN) in the workspace.
            hiptensorStatus_t epilogue(Pack const&        pack,
                                       Problem const&     p,
                                       Classified const&  c,
                                       std::string const& tType,
                                       bool               cplx,
                                       int64_t            rowStride,
                                       int64_t            batchStride,
                                       uint64_t           offset,
                                       bool               alphaInT,
                                       Program&           program,
                                       std::string&       reason)
            {
                auto const&      C = p.t[2];
                auto const&      D = p.t[3];
                std::vector<Dim> dims;
                auto             add = [&](Group const& g, std::vector<int64_t> const& tStrides) {
                    for(size_t i = 0; i < g.modes.size(); i++)
                    {
                        Dim d;
                        d.extent    = g.extent[i];
                        d.stride[0] = tStrides[i];
                        d.stride[1] = C.present ? g.stride[i][2] : 0;
                        d.stride[2] = g.stride[i][3];
                        dims.push_back(d);
                    }
                };
                add(c.M, c.M.packed(rowStride));
                add(c.N, c.N.packed(1));
                add(c.L, c.L.packed(batchStride));
                canonicalize(dims, 3, 2);
                EwOperand t;
                t.name  = "A";
                t.role  = BufferRole::Workspace;
                t.bytes = offset;
                t.scale = alphaInT ? ScalarRole::One : ScalarRole::Alpha;
                EwOperand cin;
                cin.name  = "C";
                cin.role  = C.present ? BufferRole::C : BufferRole::None;
                cin.scale = C.present ? ScalarRole::Beta : ScalarRole::Zero;
                cin.op    = C.present ? static_cast<int>(C.unaryOp) : HIPTENSOR_OP_IDENTITY;
                EwOperand out;
                out.name = "D";
                out.role = BufferRole::Out;
                auto dt  = dtypeName(D.type);
                // C may have another data type than D (the scale form reads no C: D's type)
                auto         ct    = C.present ? dtypeName(C.type) : dt;
                size_t const first = program.steps.size();
                std::vector<std::pair<std::string, std::string>> const strs{
                    {"a", tType}, {"c", ct}, {"d", dt}};
                auto status = ewSteps(pack,
                                      {{"arity", 2}, {"cplx", cplx ? 1 : 0}},
                                      strs,
                                      dims,
                                      {t, cin, out},
                                      {HIPTENSOR_OP_ADD, 0},
                                      cplx,
                                      program.steps,
                                      reason);
                if(status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                tileLaunch(pack,
                           strs,
                           dims,
                           {t, cin, out},
                           {HIPTENSOR_OP_ADD, 0},
                           cplx,
                           program.steps,
                           first);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // rocKE's body rule of the emulated-f32 routes (rocke.dispatch.gemm.cf32), with the
            // thresholds from the pack's route table.
            std::string gemmBody(Pack const& pack, int64_t M, int64_t Nk, int cus, bool cplx)
            {
                double const smallMax = pack.routeDouble("gemm.cf32_small_max_waves", 0.5);
                double const skAlways = pack.routeDouble("gemm.cf32_sk_always_waves", 4.0);
                double const skMax    = pack.routeDouble("gemm.cf32_sk_max_waves", 32.0);
                double const skTail   = pack.routeDouble("gemm.cf32_sk_max_tail", 0.65);
                double const waves = double(ceilDiv(M, 256) * ceilDiv(Nk, 256)) / std::max(1, cus);
                if(waves < smallMax)
                {
                    return cplx ? "Q3_SK" : "C3";
                }
                if(waves < skAlways)
                {
                    return "RIW_SK";
                }
                double tail = waves - std::floor(waves);
                if(waves < skMax && tail > 0 && tail <= skTail)
                {
                    return "RIW_SK";
                }
                return "RIW";
            }

            // The bodies of a GEMM route in the static order: the rule's body first.
            std::vector<std::string>
                gemmBodies(Pack const& pack, int64_t M, int64_t Nk, int cus, bool cplx)
            {
                std::vector<std::string> bodies{gemmBody(pack, M, Nk, cus, cplx)};
                for(auto const* b : {"RIW", "C3", "Q3_SK", "RIW_SK"})
                {
                    if(std::find(bodies.begin(), bodies.end(), b) == bodies.end())
                    {
                        bodies.push_back(b);
                    }
                }
                return bodies;
            }

            int gemmBodyId(std::string const& body)
            {
                return body == "RIW"      ? 0
                       : body == "RIW_SK" ? 1
                       : body == "C3"     ? 2
                       : body == "Q3_SK"  ? 3
                                          : 9;
            }

            // The group of tensor t's stride-1 mode: 0 none, 1 M, 2 N, 3 K, 4 L.
            int unitGroup(Classified const& c, int t)
            {
                Group const* groups[] = {&c.M, &c.N, &c.K, &c.L};
                for(int g = 0; g < 4; g++)
                {
                    for(size_t i = 0; i < groups[g]->modes.size(); i++)
                    {
                        if(groups[g]->stride[i][t] == 1 && groups[g]->extent[i] > 1)
                        {
                            return g + 1;
                        }
                    }
                }
                return 0;
            }

            // The facts of a contraction every candidate shares.
            CandidateFacts contractionFacts(Problem const& p, Classified const& c, bool cplx)
            {
                CandidateFacts f;
                f.M         = c.M.size();
                f.N         = c.N.size();
                f.K         = c.K.size();
                f.L         = c.L.size();
                f.modesM    = static_cast<int>(c.M.modes.size());
                f.modesN    = static_cast<int>(c.N.modes.size());
                f.modesK    = static_cast<int>(c.K.modes.size());
                f.modesL    = static_cast<int>(c.L.modes.size());
                f.unitA     = unitGroup(c, 0);
                f.unitB     = unitGroup(c, 1);
                f.unitD     = unitGroup(c, 3);
                f.bilinear  = p.t[2].present;
                f.operandOp = p.t[0].unaryOp != HIPTENSOR_OP_IDENTITY
                              || p.t[1].unaryOp != HIPTENSOR_OP_IDENTITY;
                f.flops = 2ull * uint64_t(f.M) * uint64_t(f.N) * uint64_t(f.K) * uint64_t(f.L)
                          * (cplx ? 4u : 1u);
                return f;
            }

            // Why a GEMM body declines a problem whose operands or result are past its 32-bit
            // offsets (the 64-bit-address body may then serve it).
            constexpr char kPastGemmOffsets[]
                = "operands past the 32-bit offsets of the GEMM bodies";
            // The body of the 64-bit-address GEMM kernels in the pack (pack/rocke_pack_builder.cpp
            // kGemmWideBody).
            constexpr char kGemmWideBody[] = "RIW";

            // f16 / bf16 (and f32 with a 16-bit compute type), f32 and complex f32 contractions
            // on one universal GEMM body with an f32 output: pack A and B, GEMM, epilogue.
            // `wide`: the 64-bit-address body (rocKE's wide_addr RIW form), for operands past the
            // DMA buffer range or a result past 2^31 elements; one launch per batch and per
            // 65535 rows of tiles.
            hiptensorStatus_t planGemmBody(Problem const&     p,
                                           Pack const&        pack,
                                           Classified const&  c,
                                           std::string const& route,
                                           std::string const& storage,
                                           std::string const& b,
                                           bool               wide,
                                           uint64_t           workspaceLimit,
                                           Program&           program,
                                           CandidateFacts&    facts,
                                           std::string&       reason)
            {
                // routes: h16, the real f32 recipes x3 / x4 / x6 / x9, and cf32 (x6) / cf32x3 /
                // cf32x4 / cf32x9 for complex f32
                bool const    cplx = route.compare(0, 4, "cf32") == 0;
                int64_t const M = c.M.size(), N = c.N.size(), K = c.K.size(), L = c.L.size();
                // kernel extents per problem element: K in 16-bit halves (f32 pairs for the real
                // f32 recipes, complex quadruples for cf32), N doubled for complex
                int64_t const kUnit = route == "h16" ? 1 : cplx ? 4 : 2;
                int64_t const nUnit = cplx ? 2 : 1;
                int const     cus   = std::max(1, p.device.cuCount);
                auto const*   kernel
                    = pack.find(kGemm,
                                {{"batched", !wide && L > 1 ? 1 : 0}, {"wide", wide ? 1 : 0}},
                                {{"route", route}, {"dtype", storage}, {"body", b}});
                if(kernel == nullptr)
                {
                    reason = "no " + route + " GEMM body " + b + (wide ? " (64-bit addresses)" : "")
                             + (!wide && L > 1 ? " (batched)" : "");
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const P = kernel->attr("persistent_ctas", 0);
                if(P > cus)
                {
                    reason = "the device has fewer CUs than the persistent body " + b + " needs";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const tm = kernel->attr("tile_m"), tn = kernel->attr("tile_n");
                int64_t const tk = kernel->attr("tile_k");
                // padded problem extents (problem elements)
                int64_t const     Mp = roundUp(M, tm);
                int64_t const     Np = roundUp(N * nUnit, tn) / nUnit;
                int64_t const     Kp = roundUp(K * kUnit, tk) / kUnit;
                std::string const pType
                    = route == "h16" ? (storage == "fp16" ? "f16" : "bf16") : "f32";
                uint64_t const es     = elementBytes(pType, cplx);
                uint64_t const aBytes = uint64_t(L) * uint64_t(Mp) * uint64_t(Kp) * es;
                uint64_t const bBytes = uint64_t(L) * uint64_t(Np) * uint64_t(Kp) * es;
                uint64_t const tElems = uint64_t(L) * uint64_t(Mp) * uint64_t(Np) * uint64_t(nUnit);
                int64_t const  kM = Mp, kN = Np * nUnit, kK = Kp * kUnit;
                // the kernel's capability limits (pack attributes)
                auto const gridLimit = kernel->attr("limit_grid_yz", kGridYZMax);
                // rows of tiles per launch (grid y)
                int64_t rowsPerLaunch = Mp;
                if(wide)
                {
                    // only one tile's operand rows and one tile's result rows need the 32-bit
                    // offsets (a complex B tile spans tile_n / 2 rows)
                    auto const tileLimit = static_cast<uint64_t>(
                        kernel->attr("limit_tile_operand_bytes", int64_t(kDtlBufferRange)));
                    auto const cTileLimit = static_cast<uint64_t>(
                        kernel->attr("limit_c_tile_elements", int64_t(1) << 31));
                    uint64_t const rowBytes = uint64_t(kK) * 2;
                    if(uint64_t(tm) * rowBytes > tileLimit
                       || uint64_t(cplx ? tn / 2 : tn) * rowBytes > tileLimit
                       || uint64_t(tm) * uint64_t(kN) > cTileLimit)
                    {
                        reason = "a tile of the 64-bit-address GEMM body is past its 32-bit "
                                 "offsets";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    rowsPerLaunch          = std::min<int64_t>(Mp, gridLimit * tm);
                    int64_t const launches = L * ceilDiv(Mp, rowsPerLaunch);
                    if(launches > kMaxLaunches)
                    {
                        reason = "the 64-bit-address GEMM body needs more than "
                                 + std::to_string(kMaxLaunches) + " launches";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                else
                {
                    auto const opLimit = static_cast<uint64_t>(
                        kernel->attr("limit_operand_bytes", int64_t(kDtlBufferRange)));
                    auto const cLimit
                        = static_cast<uint64_t>(kernel->attr("limit_c_elements", kI32Max));
                    if(aBytes >= opLimit || bBytes >= opLimit || tElems > cLimit
                       || ceilDiv(Mp, tm) > gridLimit || L > gridLimit)
                    {
                        reason = kPastGemmOffsets;
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                uint64_t const offA     = 0;
                uint64_t const offB     = alignUp(offA + aBytes, kWorkspaceAlign);
                uint64_t const offT     = alignUp(offB + bBytes, kWorkspaceAlign);
                uint64_t const offFlags = alignUp(offT + tElems * 4, kWorkspaceAlign);
                uint64_t const flags    = kernel->attr("sk_flags_bytes", 0);
                uint64_t const parts    = kernel->attr("sk_part_bytes", 0);
                uint64_t const offPart  = alignUp(offFlags + flags, kWorkspaceAlign);
                uint64_t const total
                    = alignUp(parts > 0 ? offPart + parts : offFlags, kWorkspaceAlign);
                if(total > workspaceLimit)
                {
                    reason = "the " + b + " body needs more workspace than the limit";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                Program prog;
                prog.pack           = &pack;
                prog.scalarType     = p.scalarType;
                prog.complexScalars = cplx;
                prog.workspaceBytes = total;
                auto status = packOperand(pack, p, c, 0, pType, cplx, Mp, Kp, offA, prog, reason);
                if(status == HIPTENSOR_STATUS_SUCCESS)
                {
                    status = packOperand(pack, p, c, 1, pType, cplx, Np, Kp, offB, prog, reason);
                }
                if(status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                // the GEMM launches: one, or (64-bit addresses) one per batch and row range
                int64_t const batchesPerLaunch = wide ? 1 : L;
                int64_t       ctas             = 0;
                bool          firstLaunch      = true;
                for(int64_t l = 0; l < L; l += batchesPerLaunch)
                {
                    for(int64_t m0 = 0; m0 < Mp; m0 += rowsPerLaunch)
                    {
                        int64_t const  rows = std::min(rowsPerLaunch, Mp - m0);
                        uint64_t const aOff
                            = offA
                              + (uint64_t(l) * uint64_t(Mp) + uint64_t(m0)) * uint64_t(Kp) * es;
                        uint64_t const bOff = offB + uint64_t(l) * uint64_t(Np) * uint64_t(Kp) * es;
                        uint64_t const tOff
                            = offT + (uint64_t(l) * uint64_t(Mp) + uint64_t(m0)) * uint64_t(kN) * 4;
                        LaunchStep g;
                        g.kernel = kernel;
                        ArgWriter w(g);
                        bool      ok = w.pointer("A", BufferRole::Workspace, aOff)
                                  && w.pointer("B", BufferRole::Workspace, bOff)
                                  && w.pointer("C", BufferRole::Workspace, tOff)
                                  && w.set("M", wide ? rows : kM) && w.set("N", kN)
                                  && w.set("K", kK);
                        if(parts > 0)
                        {
                            ok = ok && w.pointer("SkFlags", BufferRole::Workspace, offFlags)
                                 && w.pointer("SkPart", BufferRole::Workspace, offPart);
                        }
                        if(!wide && L > 1)
                        {
                            // per-batch strides in elements of each kernel pointer (16-bit
                            // halves for A and B, f32 for C)
                            int64_t const halves = int64_t(es / 2);
                            ok                   = ok && w.set("stride_a", Mp * Kp * halves)
                                 && w.set("stride_b", Np * Kp * halves)
                                 && w.set("stride_c", kM * kN);
                        }
                        int64_t const tilesM = ceilDiv(wide ? rows : kM, tm),
                                      tilesN = ceilDiv(kN, tn);
                        if(kernel->attr("streamk", 0) == 1)
                        {
                            g.grid = dim3(static_cast<uint32_t>(P), 1, 1);
                        }
                        else if(P > 0)
                        {
                            g.grid = dim3(static_cast<uint32_t>(std::max<int64_t>(
                                              1, std::min<int64_t>(P, tilesM * tilesN))),
                                          1,
                                          1);
                        }
                        else
                        {
                            g.grid = dim3(static_cast<uint32_t>(tilesN),
                                          static_cast<uint32_t>(tilesM),
                                          static_cast<uint32_t>(!wide && L > 1 ? L : 1));
                        }
                        g.block = dim3(static_cast<uint32_t>(kernel->attr("block", 256)), 1, 1);
                        if(!ok)
                        {
                            reason = "GEMM arguments do not fit the kernel arguments";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                        if(auto s = finishStep(g, w, reason); s != HIPTENSOR_STATUS_SUCCESS)
                        {
                            return s;
                        }
                        if(firstLaunch)
                        {
                            ctas = int64_t(g.grid.x) * g.grid.y * g.grid.z;
                            if(parts > 0)
                            {
                                // stream-K flags zeroed before the GEMM launch
                                prog.memsets.push_back({offFlags,
                                                        flags,
                                                        flags,
                                                        1,
                                                        static_cast<uint32_t>(prog.steps.size())});
                            }
                        }
                        firstLaunch = false;
                        prog.steps.push_back(std::move(g));
                    }
                }
                prog.description = kernel->name;
                if(wide && prog.steps.size() > 3)
                {
                    prog.description += "x" + std::to_string(L * ceilDiv(Mp, rowsPerLaunch));
                }
                // T is [L][Mp][Np] (complex elements for cf32)
                status = epilogue(pack, p, c, "f32", cplx, Np, Mp * Np, offT, false, prog, reason);
                if(status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                auto const&    D      = p.t[3];
                uint64_t const esA    = elementBytes(dtypeName(p.t[0].type), cplx);
                uint64_t const esB    = elementBytes(dtypeName(p.t[1].type), cplx);
                uint64_t const esOut  = elementBytes(dtypeName(D.type), cplx);
                uint64_t const tBytes = tElems * 4;
                facts                 = contractionFacts(p, c, cplx);
                facts.route        = route == "h16" ? Route::H16 : cplx ? Route::CF32 : Route::X6;
                facts.body         = gemmBodyId(b);
                facts.tileM        = tm;
                facts.tileN        = tn;
                facts.tileK        = tk;
                facts.block        = kernel->attr("block", 256);
                facts.persistent   = P;
                facts.streamK      = kernel->attr("streamk", 0) == 1;
                facts.ctas         = ctas;
                facts.paddedVolume = Mp * Np * Kp * L;
                facts.main         = kernel;
                facts.bytes        = uint64_t(M * K * L) * esA + uint64_t(N * K * L) * esB
                              + 2 * (aBytes + bBytes) + 2 * tBytes
                              + uint64_t(M * N * L) * esOut * (p.t[2].present ? 2 : 1);
                program = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // rocke.instances.common.gett_f64.gett_f64_shape_class for packed operands (both
            // read along K).
            std::string gett64Class(std::string const& arch,
                                    bool               cplx,
                                    int64_t            M,
                                    int64_t            N,
                                    int64_t            K,
                                    int64_t            L,
                                    int                cus)
            {
                if(cplx)
                {
                    int64_t tiles = ceilDiv(M, 64) * ceilDiv(N, 64) * L;
                    int     perCu = arch == "gfx942" ? 2 : 4;
                    if(tiles < perCu * cus)
                    {
                        return "small";
                    }
                    if(K >= 2048)
                    {
                        return "large";
                    }
                    if(tiles >= 64 * cus)
                    {
                        return "wide";
                    }
                    return "medium";
                }
                int64_t tiles = ceilDiv(M, 128) * ceilDiv(N, 128) * L;
                if(tiles < (arch == "gfx90a" ? 2 : 1) * cus)
                {
                    return "small";
                }
                if(arch == "gfx90a")
                {
                    return "kk";
                }
                if(arch == "gfx950" && (K >= 4096 || tiles >= 8 * cus))
                {
                    return "large";
                }
                return "medium";
            }

            int gett64ClassId(std::string const& cls)
            {
                return cls == "small"    ? 10
                       : cls == "medium" ? 11
                       : cls == "large"  ? 12
                       : cls == "wide"   ? 13
                       : cls == "kk"     ? 14
                                         : 15;
            }

            // How the f64 strided GEMM writes a contraction: D (and C) through their own
            // strides when each group is one run of D's modes ("direct"), else into a packed T
            // [L][M][N] followed by an epilogue. `c` is ordered by D.
            struct Gett64Layout
            {
                bool        direct   = false;
                bool        bilinear = false;
                std::string dclass   = "n";
                int64_t     dsm = 0, dsn = 0, dsl = 0, csm = 0, csn = 0, csl = 0;
            };

            // `f32`: the f32 strided GEMM (f32 / complex f32 words), else the f64 one.
            Gett64Layout gett64Layout(Problem const& p, Classified const& c, bool cplx, bool f32)
            {
                auto const&   C = p.t[2];
                auto const&   D = p.t[3];
                int64_t const M = c.M.size(), N = c.N.size();
                Gett64Layout  g;
                // D has the kernel's word type
                bool const realOut = f32 ? D.type == HIPTENSOR_R_32F || D.type == HIPTENSOR_C_32F
                                         : D.type == HIPTENSOR_R_64F || D.type == HIPTENSOR_C_64F;
                g.dsm              = c.M.mergedStride(3);
                g.dsn              = c.N.mergedStride(3);
                g.dsl              = c.L.mergedStride(3);
                g.csm              = C.present ? c.M.mergedStride(2) : g.dsm;
                g.csn              = C.present ? c.N.mergedStride(2) : g.dsn;
                g.csl              = C.present ? c.L.mergedStride(2) : g.dsl;
                auto unit
                    = [](int64_t stride, int64_t extent) { return stride == 1 && extent > 1; };
                // the kernel reads C with D's data type
                bool direct
                    = realOut && g.dsm >= 0 && g.dsn >= 0 && g.dsl >= 0 && g.csm >= 0 && g.csn >= 0
                      && g.csl >= 0
                      && (!C.present || (C.unaryOp == HIPTENSOR_OP_IDENTITY && C.type == D.type))
                      && unit(g.dsm, M) == unit(g.csm, M) && unit(g.dsn, N) == unit(g.csn, N);
                uint64_t const es0  = (f32 ? 4 : 8) * (cplx ? 2 : 1);
                auto           span = [&](int64_t sm, int64_t sn) {
                    return ((M - 1) * sm + (N - 1) * sn + 1) * int64_t(es0);
                };
                direct = direct && span(g.dsm, g.dsn) <= kGett64SpanLimit
                         && span(g.csm, g.csn) <= kGett64SpanLimit && g.dsl <= INT64_MAX / 2;
                g.direct   = direct;
                g.dclass   = direct ? (unit(g.dsm, M) ? "m" : unit(g.dsn, N) ? "n" : "x") : "n";
                g.bilinear = direct && C.present;
                return g;
            }

            // The f64 (f32: f32) GEMM kernels for a layout in the static order: the rule's class
            // (with K blocks for a long K when the pack has them), the default configuration,
            // then the other classes.
            std::vector<PackKernel const*> gett64Kernels(Pack const&         pack,
                                                         Gett64Layout const& g,
                                                         bool                cplx,
                                                         std::string const&  ruleClass,
                                                         bool                longK,
                                                         bool                f32)
            {
                std::vector<PackKernel const*> out;
                auto                           matches = [&](PackKernel const& k) {
                    return k.family == (f32 ? kGett32 : kGett64) && k.attr("cplx") == (cplx ? 1 : 0)
                           && k.attr("bilinear") == (g.bilinear ? 1 : 0)
                           && k.attrStr("dclass") == g.dclass;
                };
                auto add = [&](PackKernel const* k) {
                    if(std::find(out.begin(), out.end(), k) == out.end())
                    {
                        out.push_back(k);
                    }
                };
                for(auto const& cl : {ruleClass, std::string("default")})
                {
                    for(bool kb : {longK, !longK})
                    {
                        for(auto const& k : pack.kernels())
                        {
                            if(matches(k) && k.attrStr("cls") == cl
                               && (k.attr("kblock", 0) > 0) == kb)
                            {
                                add(&k);
                            }
                        }
                    }
                }
                for(auto const& k : pack.kernels())
                {
                    if(matches(k))
                    {
                        add(&k);
                    }
                }
                return out;
            }

            // A contraction on one f64 strided GEMM kernel (f64 / complex f64 data, and every
            // data type on the F64 route), or (`f32`) on one f32 strided GEMM kernel (the native
            // route of f32 / complex f32 data: operands converted to f32 exactly, f32
            // accumulation, the K-range partials f32). `c` is ordered by D.
            //
            // One launch addresses at most 2^31 bytes per operand buffer (packed A and B rows,
            // the result's rows) and 65535 batches. Past that the GEMM runs as host-iterated
            // launches over ranges of M, N and L, each with its operands offset to its range.
            // A row of A or B past 2^31 bytes splits K: the K ranges are the batches of one
            // launch per row range (a last, shorter range its own launch), each writing an f64
            // partial result, and a reduction adds the partials into T (the epilogue then writes
            // D as without K ranges).
            hiptensorStatus_t planGett64(Problem const&      p,
                                         Pack const&         pack,
                                         Classified const&   c,
                                         bool                cplx,
                                         Gett64Layout const& g,
                                         PackKernel const&   kernel,
                                         uint64_t            workspaceLimit,
                                         Program&            program,
                                         CandidateFacts&     facts,
                                         std::string&        reason,
                                         bool                f32 = false)
            {
                int64_t const M = c.M.size(), N = c.N.size(), K = c.K.size(), L = c.L.size();
                // one layout class: even K (real f32: a multiple of 4, the 16-byte loads)
                int64_t const  Kp         = roundUp(K, f32 && !cplx ? 4 : 2);
                uint64_t const es         = (f32 ? 4 : 8) * (cplx ? 2 : 1);
                std::string    pt         = f32 ? "f32" : "f64";
                std::string    gemmName   = f32 ? "the f32 GEMM" : "the f64 GEMM";
                int64_t const  spanLimit  = kernel.attr("limit_span_bytes", kGett64SpanLimit);
                int64_t const  batchLimit = kernel.attr("limit_batch", kGridYZMax);
                int64_t const  tm = kernel.attr("tile_m"), tn = kernel.attr("tile_n");
                int const      cus = std::max(1, p.device.cuCount);
                // elements one buffer resource addresses
                int64_t const span = spanLimit / int64_t(es);

                // K ranges: when a row of A or B is past one buffer, as batches of one launch
                // (enough of them to fill the device, each at least kMinKRange long)
                constexpr int64_t kMinKRange = int64_t(1) << 16;
                bool const        splitK     = Kp > span;
                int64_t           kc = Kp, nk = 1;
                if(splitK)
                {
                    int64_t const align = std::max<int64_t>(2, kernel.attr("tile_k", 2));
                    int64_t const most  = std::max<int64_t>(2, span / align * align);
                    nk = std::max(ceilDiv(Kp, most), std::min<int64_t>(4 * cus, Kp / kMinKRange));
                    nk = std::min(nk, batchLimit);
                    kc = roundUp(ceilDiv(Kp, nk), align);
                    if(kc > most)
                    {
                        kc = most;
                    }
                    nk = ceilDiv(Kp, kc);
                    if(nk > batchLimit + 1)
                    {
                        reason = gemmName + " needs more than " + std::to_string(batchLimit + 1)
                                 + " K ranges";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                bool const direct  = g.direct && !splitK;
                auto       rowsFor = [&](int64_t rows) {
                    // (rows - 1) * Kp + kc elements
                    return std::min<int64_t>(rows, std::max<int64_t>(1, (span - kc) / Kp + 1));
                };
                int64_t mc = rowsFor(M), nc = rowsFor(N);
                // the result's rows: D (and C) through their strides, or T [L][M][N]
                int64_t const dsm = direct ? g.dsm : N, dsn = direct ? g.dsn : 1;
                auto          outSpan
                    = [&](int64_t sm, int64_t sn) { return (mc - 1) * sm + (nc - 1) * sn + 1; };
                while(outSpan(dsm, dsn) > span
                      || (direct && g.bilinear && outSpan(g.csm, g.csn) > span))
                {
                    if((mc - 1) * std::max(dsm, g.bilinear ? g.csm : 0)
                       >= (nc - 1) * std::max(dsn, g.bilinear ? g.csn : 0))
                    {
                        mc = std::max<int64_t>(1, mc / 2);
                    }
                    else
                    {
                        nc = std::max<int64_t>(1, nc / 2);
                    }
                }
                // whole tiles per range
                if(mc < M && mc >= tm)
                {
                    mc = mc / tm * tm;
                }
                if(nc < N && nc >= tn)
                {
                    nc = nc / tn * tn;
                }
                // batches per launch (K ranges: one batch dimension, so one launch per batch)
                int64_t const lc       = splitK ? 1 : std::min<int64_t>(L, batchLimit);
                int64_t const kLaunch  = splitK ? (Kp % kc != 0 ? 2 : 1) : 1;
                int64_t const launches = ceilDiv(M, mc) * ceilDiv(N, nc) * ceilDiv(L, lc) * kLaunch;
                if(launches > kMaxLaunches)
                {
                    reason = gemmName + " needs more than " + std::to_string(kMaxLaunches)
                             + " launches for operands of this size";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }

                // partial results of the K ranges: P[l][range][..] in the layout the kernel's
                // D class writes (m or n contiguous, or neither: every second element)
                auto const     dclass = kernel.attrStr("dclass");
                int64_t const  psm    = dclass == "m" ? 1 : dclass == "n" ? N : 2;
                int64_t const  psn    = dclass == "m" ? M : dclass == "n" ? 1 : 2 * M;
                int64_t const  pRange = dclass == "x" ? 2 * M * N : M * N;
                uint64_t const aBytes = uint64_t(L) * uint64_t(M) * uint64_t(Kp) * es;
                uint64_t const bBytes = uint64_t(L) * uint64_t(N) * uint64_t(Kp) * es;
                uint64_t const tBytes = direct ? 0 : uint64_t(L) * uint64_t(M) * uint64_t(N) * es;
                uint64_t const pBytes
                    = splitK ? uint64_t(L) * uint64_t(nk) * uint64_t(pRange) * es : 0;
                uint64_t const offA = 0;
                uint64_t const offB = alignUp(aBytes, kWorkspaceAlign);
                uint64_t const offT = alignUp(offB + bBytes, kWorkspaceAlign);
                uint64_t const offP = alignUp(offT + tBytes, kWorkspaceAlign);
                uint64_t const offW = alignUp(offP + pBytes, kWorkspaceAlign);

                // the reduction of the partials into T: a reduction of P over the ranges
                // (complex: of its components, one more kept mode), alpha 1 and no C
                Program sum;
                if(splitK)
                {
                    int64_t const    unit = cplx ? 2 : 1;
                    backend::Problem r;
                    r.kind       = OpKind::Reduction;
                    r.opReduce   = HIPTENSOR_OP_ADD;
                    r.compute    = f32 ? HIPTENSOR_COMPUTE_DESC_32F : HIPTENSOR_COMPUTE_DESC_64F;
                    r.scalarType = f32 ? HIPTENSOR_R_32F : HIPTENSOR_R_64F;
                    r.device     = p.device;
                    TensorArg P, T;
                    P.present = T.present = true;
                    P.type = T.type = f32 ? HIPTENSOR_R_32F : HIPTENSOR_R_64F;
                    auto add = [](TensorArg& t, int32_t mode, int64_t extent, int64_t stride) {
                        t.modes.push_back(mode);
                        t.extents.push_back(extent);
                        t.strides.push_back(stride);
                    };
                    if(cplx)
                    {
                        add(P, 0, 2, 1);
                        add(T, 0, 2, 1);
                    }
                    add(P, 1, M, psm * unit);
                    add(P, 2, N, psn * unit);
                    add(P, 3, nk, pRange * unit);
                    add(P, 4, L, nk * pRange * unit);
                    add(T, 1, M, N * unit);
                    add(T, 2, N, unit);
                    add(T, 4, L, M * N * unit);
                    r.t[0] = P;
                    r.t[2] = T; // C: the API's reductions have one; beta 0 does not read it
                    r.t[3] = T;
                    CandidateFacts sumFacts;
                    std::string    why;
                    uint64_t const left = workspaceLimit > offW ? workspaceLimit - offW : 0;
                    auto status = planReduction(r, pack, left, kRuleMode, 1, sum, sumFacts, why);
                    if(status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        reason = "the reduction of the K ranges: " + why;
                        return status;
                    }
                }
                uint64_t const total = alignUp(offW + sum.workspaceBytes, kWorkspaceAlign);
                if(total > workspaceLimit)
                {
                    reason = std::string(f32 ? "the native f32 route" : "the f64 route")
                             + " needs more workspace than the limit";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                Program prog;
                prog.pack           = &pack;
                prog.scalarType     = p.scalarType;
                prog.complexScalars = cplx;
                prog.workspaceBytes = total;
                auto status = packOperand(pack, p, c, 0, pt, cplx, M, Kp, offA, prog, reason);
                if(status == HIPTENSOR_STATUS_SUCCESS)
                {
                    status = packOperand(pack, p, c, 1, pt, cplx, N, Kp, offB, prog, reason);
                }
                if(status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                // what the GEMM writes: D (direct), T, or the partials P
                auto const     outRole = direct ? BufferRole::Out : BufferRole::Workspace;
                uint64_t const outBase = direct ? 0 : splitK ? offP : offT;
                int64_t const  osm     = direct ? g.dsm : splitK ? psm : N;
                int64_t const  osn     = direct ? g.dsn : splitK ? psn : 1;
                int64_t const  osl     = direct ? g.dsl : splitK ? nk * pRange : M * N;
                int64_t        ctas    = 0;
                // K ranges of one launch: [k0, k0 + kc * batches), batch stride kc
                std::vector<std::pair<int64_t, int64_t>> kLaunches; // first range, ranges
                if(splitK)
                {
                    int64_t const full = Kp / kc;
                    if(full > 0)
                    {
                        kLaunches.emplace_back(0, full);
                    }
                    if(Kp % kc != 0)
                    {
                        kLaunches.emplace_back(full, 1);
                    }
                }
                else
                {
                    kLaunches.emplace_back(0, 1);
                }
                for(auto const& [range0, ranges] : kLaunches)
                {
                    int64_t const k0 = range0 * kc;
                    int64_t const kk = std::min(kc, Kp - k0);
                    for(int64_t l0 = 0; l0 < L; l0 += lc)
                    {
                        for(int64_t m0 = 0; m0 < M; m0 += mc)
                        {
                            for(int64_t n0 = 0; n0 < N; n0 += nc)
                            {
                                int64_t const km = std::min(mc, M - m0), kn = std::min(nc, N - n0);
                                // batches: the K ranges, or the problem's L
                                int64_t const kl = splitK ? ranges : std::min(lc, L - l0);
                                LaunchStep    step;
                                step.kernel = &kernel;
                                ArgWriter w(step);
                                // a stride of a range of extent 1 is not used: 0 (it may not fit
                                // the i32 argument)
                                auto s = [](int64_t extent, int64_t stride) {
                                    return extent > 1 ? stride : 0;
                                };
                                uint64_t const aOff
                                    = offA
                                      + ((uint64_t(l0) * uint64_t(M) + uint64_t(m0)) * uint64_t(Kp)
                                         + uint64_t(k0))
                                            * es;
                                uint64_t const bOff
                                    = offB
                                      + ((uint64_t(l0) * uint64_t(N) + uint64_t(n0)) * uint64_t(Kp)
                                         + uint64_t(k0))
                                            * es;
                                int64_t const outIndex = splitK ? l0 * nk * pRange + range0 * pRange
                                                                      + m0 * osm + n0 * osn
                                                                : l0 * osl + m0 * osm + n0 * osn;
                                uint64_t const dOff    = outBase + uint64_t(outIndex) * es;
                                // batch strides: K ranges move A and B by kc, the result by one
                                // range of partials
                                int64_t const aSl = splitK ? kc : M * Kp;
                                int64_t const bSl = splitK ? kc : N * Kp;
                                int64_t const dSl = splitK ? pRange : osl;
                                bool          ok  = w.pointer("A", BufferRole::Workspace, aOff)
                                          && w.pointer("B", BufferRole::Workspace, bOff)
                                          && w.pointer("D", outRole, dOff)
                                          && w.set("d_sm", s(km, osm)) && w.set("d_sn", s(kn, osn))
                                          && w.set("d_sl", s(kl, dSl));
                                if(g.bilinear)
                                {
                                    // direct: the problem's C; K ranges: the kernel's C is not
                                    // read (beta 0)
                                    uint64_t const cOff
                                        = direct
                                              ? uint64_t(l0 * g.csl + m0 * g.csm + n0 * g.csn) * es
                                              : 0;
                                    ok = ok
                                         && w.pointer(
                                             "C", direct ? BufferRole::C : BufferRole::None, cOff)
                                         && w.set("c_sm", direct ? s(km, g.csm) : 0)
                                         && w.set("c_sn", direct ? s(kn, g.csn) : 0)
                                         && w.set("c_sl", direct ? s(kl, g.csl) : 0);
                                }
                                auto const alpha = direct ? ScalarRole::Alpha : ScalarRole::One;
                                auto const beta  = direct ? ScalarRole::Beta : ScalarRole::Zero;
                                if(cplx)
                                {
                                    ok = ok && w.scalar("alpha_re", alpha, false)
                                         && w.scalar("alpha_im", alpha, true);
                                    if(g.bilinear)
                                    {
                                        ok = ok && w.scalar("beta_re", beta, false)
                                             && w.scalar("beta_im", beta, true);
                                    }
                                }
                                else
                                {
                                    ok = ok && w.scalar("alpha", alpha);
                                    if(g.bilinear)
                                    {
                                        ok = ok && w.scalar("beta", beta);
                                    }
                                }
                                ok = ok && w.set("M", km) && w.set("N", kn) && w.set("K", kk)
                                     && w.set("L", kl) && w.set("a_sm", s(km, Kp))
                                     && w.set("a_sk", 1) && w.set("b_sn", s(kn, Kp))
                                     && w.set("b_sk", 1) && w.set("a_sl", s(kl, aSl))
                                     && w.set("b_sl", s(kl, bSl));
                                if(!ok)
                                {
                                    reason = std::string(f32 ? "f32" : "f64")
                                             + " GEMM arguments do not fit the kernel arguments";
                                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                                }
                                step.grid
                                    = dim3(static_cast<uint32_t>(ceilDiv(km, tm) * ceilDiv(kn, tn)),
                                           static_cast<uint32_t>(kl),
                                           1);
                                step.block
                                    = dim3(static_cast<uint32_t>(kernel.attr("block", 256)), 1, 1);
                                if(auto st = finishStep(step, w, reason);
                                   st != HIPTENSOR_STATUS_SUCCESS)
                                {
                                    return st;
                                }
                                if(ctas == 0)
                                {
                                    ctas = int64_t(step.grid.x) * step.grid.y * step.grid.z;
                                }
                                prog.steps.push_back(std::move(step));
                            }
                        }
                    }
                }
                prog.description = kernel.name;
                if(launches > 1)
                {
                    prog.description += "x" + std::to_string(launches);
                }
                if(splitK)
                {
                    // the partials' sum into T: P (A) and T (D) in the workspace, alpha 1, no C
                    auto const base = static_cast<uint32_t>(prog.steps.size());
                    for(auto m : sum.memsets)
                    {
                        m.offset += offW;
                        m.beforeStep += base;
                        prog.memsets.push_back(m);
                    }
                    auto toWorkspace = [offP, offT, offW](LaunchStep& launch) {
                        for(auto& ptr : launch.pointers)
                        {
                            switch(ptr.role)
                            {
                            case BufferRole::A:
                                ptr.role = BufferRole::Workspace;
                                ptr.bytes += offP;
                                break;
                            case BufferRole::Out:
                                ptr.role = BufferRole::Workspace;
                                ptr.bytes += offT;
                                break;
                            case BufferRole::Workspace:
                                ptr.bytes += offW;
                                break;
                            default:
                                ptr.role = BufferRole::None;
                                break;
                            }
                        }
                        for(auto& sc : launch.scalars)
                        {
                            sc.role = sc.role == ScalarRole::Alpha || sc.role == ScalarRole::One
                                          ? ScalarRole::One
                                          : ScalarRole::Zero;
                        }
                    };
                    for(auto step : sum.steps)
                    {
                        remapLaunch(step, toWorkspace);
                        prog.steps.push_back(std::move(step));
                    }
                    prog.description += "+k" + std::to_string(nk);
                }
                if(!direct)
                {
                    status = epilogue(pack, p, c, pt, cplx, N, M * N, offT, false, prog, reason);
                    if(status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                }
                auto const&    D     = p.t[3];
                uint64_t const esA   = elementBytes(dtypeName(p.t[0].type), cplx);
                uint64_t const esB   = elementBytes(dtypeName(p.t[1].type), cplx);
                uint64_t const esOut = elementBytes(dtypeName(D.type), cplx);
                facts                = contractionFacts(p, c, cplx);
                facts.route          = f32 ? Route::Gett32 : Route::Gett64;
                facts.body           = gett64ClassId(kernel.attrStr("cls"));
                facts.f64Route
                    = !f32 && p.t[0].type != HIPTENSOR_R_64F && p.t[0].type != HIPTENSOR_C_64F;
                facts.direct         = direct;
                facts.kBlock         = kernel.attr("kblock", 0) > 0;
                facts.tileM          = tm;
                facts.tileN          = tn;
                facts.tileK          = kernel.attr("tile_k", 0);
                facts.block          = kernel.attr("block", 256);
                facts.ctas           = ctas;
                facts.split          = nk;
                facts.paddedVolume   = M * N * Kp * L;
                facts.main           = &kernel;
                uint64_t const outIo = uint64_t(M * N * L) * esOut * (p.t[2].present ? 2 : 1);
                facts.bytes          = uint64_t(M * K * L) * esA + uint64_t(N * K * L) * esB
                              + 2 * (aBytes + bBytes) + 2 * tBytes + 2 * pBytes + outIo;
                program = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // The canonical key of a contraction for rocKE's GETT route tables, as the pack
            // builder writes it (pack/rocke_pack_builder.cpp gettProblemKey): D, A, B with
            // extent-1 modes dropped, modes ordered by stride and named by first appearance.
            std::string gettProblemKey(Problem const& p, std::string const& dtype, bool bilinear)
            {
                std::map<int32_t, std::string> names;
                std::string                    key = dtype + (bilinear ? "|bilinear" : "|scale");
                char const                     labels[] = {'D', 'A', 'B'};
                int const                      slots[]  = {3, 0, 1};
                for(int t = 0; t < 3; t++)
                {
                    auto const&                                        x = p.t[slots[t]];
                    std::vector<std::tuple<int64_t, int32_t, int64_t>> ms; // stride, mode, extent
                    for(int i = 0; i < x.numModes(); i++)
                    {
                        if(x.extents[i] != 1)
                        {
                            ms.emplace_back(x.strides[i], x.modes[i], x.extents[i]);
                        }
                    }
                    std::sort(ms.begin(), ms.end());
                    key += std::string("|") + labels[t] + ":";
                    for(size_t i = 0; i < ms.size(); i++)
                    {
                        auto [stride, mode, extent] = ms[i];
                        if(names.find(mode) == names.end())
                        {
                            auto n      = names.size();
                            names[mode] = "x" + std::to_string(n);
                        }
                        key += (i ? "," : "") + names[mode] + ":" + std::to_string(extent) + ":"
                               + std::to_string(stride);
                    }
                }
                return key;
            }

            std::vector<std::string> split(std::string const& text, char sep)
            {
                std::vector<std::string> out;
                size_t                   start = 0;
                while(true)
                {
                    auto end = text.find(sep, start);
                    out.push_back(text.substr(start, end - start));
                    if(end == std::string::npos)
                    {
                        return out;
                    }
                    start = end + 1;
                }
            }

            // A contraction of rocKE's tuned or AOT GETT route table: the launches of its record
            // (see pack/rocke_pack_builder.cpp gettPlanDescriptors). False when the problem has no
            // record or the record does not apply.
            bool planTunedGett(Problem const& p,
                               Pack const&    pack,
                               uint64_t       workspaceLimit,
                               Program&       program,
                               std::string&   reason)
            {
                auto const& C  = p.t[2];
                auto const& D  = p.t[3];
                auto        dt = dtypeName(D.type);
                if((dt != "f16" && dt != "bf16") || p.compute != HIPTENSOR_COMPUTE_DESC_32F
                   || p.t[0].type != D.type || p.t[1].type != D.type
                   || (C.present && C.type != D.type) || p.t[0].unaryOp != HIPTENSOR_OP_IDENTITY
                   || p.t[1].unaryOp != HIPTENSOR_OP_IDENTITY
                   || (C.present
                       && (C.unaryOp != HIPTENSOR_OP_IDENTITY || C.strides != D.strides
                           || C.modes != D.modes)))
                {
                    return false;
                }
                auto key = gettProblemKey(p, dt, C.present);
                char hex[17];
                snprintf(
                    hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(fnv1a64(key)));
                std::string route = std::string("gett.tuned.") + hex;
                if(pack.route(route + ".key") != key)
                {
                    return false;
                }
                auto steps = split(pack.route(route), '|');
                if(steps.size() < 2 || steps[0].rfind("v1;", 0) != 0)
                {
                    reason = "malformed tuned GETT record " + route;
                    return false;
                }
                uint64_t ws = 0;
                for(auto const& field : split(steps[0], ';'))
                {
                    if(field.rfind("ws=", 0) == 0)
                    {
                        ws = std::strtoull(field.c_str() + 3, nullptr, 10);
                    }
                }
                if(ws > workspaceLimit)
                {
                    reason = "the tuned GETT route needs more workspace than the limit";
                    return false;
                }
                Program prog;
                prog.pack           = &pack;
                prog.scalarType     = HIPTENSOR_R_32F;
                prog.workspaceBytes = alignUp(ws, kWorkspaceAlign);
                for(size_t s = 1; s < steps.size(); s++)
                {
                    LaunchStep  step;
                    std::string args;
                    for(auto const& field : split(steps[s], ';'))
                    {
                        if(field.rfind("k=", 0) == 0)
                        {
                            step.kernel = pack.byName(field.substr(2));
                        }
                        else if(field.rfind("g=", 0) == 0)
                        {
                            auto g    = split(field.substr(2), ',');
                            step.grid = dim3(static_cast<uint32_t>(std::stoul(g.at(0))),
                                             static_cast<uint32_t>(std::stoul(g.at(1))),
                                             static_cast<uint32_t>(std::stoul(g.at(2))));
                        }
                        else if(field.rfind("b=", 0) == 0)
                        {
                            step.block
                                = dim3(static_cast<uint32_t>(std::stoul(field.substr(2))), 1, 1);
                        }
                        else if(field.rfind("a=", 0) == 0)
                        {
                            args = field.substr(2);
                        }
                    }
                    if(step.kernel == nullptr)
                    {
                        reason = "tuned GETT record " + route + " names a kernel the pack lacks";
                        return false;
                    }
                    ArgWriter w(step);
                    bool      ok = true;
                    for(auto const& binding : split(args, ','))
                    {
                        auto eq = binding.find('=');
                        if(eq == std::string::npos || eq + 1 >= binding.size())
                        {
                            ok = false;
                            break;
                        }
                        auto name  = binding.substr(0, eq);
                        auto value = binding.substr(eq + 1);
                        if(value[0] == '#')
                        {
                            ok = ok && w.set(name, std::stoll(value.substr(1)));
                        }
                        else if(value == "@a" || value == "@b")
                        {
                            ok = ok
                                 && w.scalar(name,
                                             value == "@a" ? ScalarRole::Alpha : ScalarRole::Beta);
                        }
                        else
                        {
                            auto       plus = value.find('+');
                            auto       buf  = value.substr(0, plus);
                            BufferRole role = buf == "A"   ? BufferRole::A
                                              : buf == "B" ? BufferRole::B
                                              : buf == "C" ? BufferRole::C
                                              : buf == "O" ? BufferRole::Out
                                              : buf == "W" ? BufferRole::Workspace
                                                           : BufferRole::None;
                            uint64_t   off  = plus == std::string::npos
                                                  ? 0
                                                  : std::stoull(value.substr(plus + 1));
                            ok = ok && role != BufferRole::None && w.pointer(name, role, off);
                        }
                    }
                    std::string missing;
                    if(!ok || !w.complete(missing))
                    {
                        reason = "tuned GETT record " + route + " does not bind kernel "
                                 + step.kernel->name;
                        return false;
                    }
                    prog.steps.push_back(std::move(step));
                }
                prog.description
                    = "tuned-gett:" + std::string(hex) + ":" + prog.steps.back().kernel->name;
                program = std::move(prog);
                return true;
            }
        } // namespace

        namespace
        {
            // The routes a contraction's data and compute types admit.
            struct ContractionRoutes
            {
                // a universal GEMM route: h16 (operands rounded to f16 / bf16, `storage` fp16
                // or bf16), x3 / x4 / x6 / x9 (operands as f32, split into bf16 parts) or cf32 /
                // cf32x3 / cf32x4 / cf32x9 (complex f32, the x6 / x3 / x4 / x9 recipes)
                bool        gemm = false;
                std::string route, storage;
                // the f32 strided GEMM (operands converted to f32 exactly, f32 products and
                // accumulation): the native route of f32 data
                bool gett32 = false;
                // the f64 strided GEMM (operands converted to f64 exactly)
                bool f64 = false;
                // only the f64 route (the f64 recipe): no tuned GETT route either
                bool f64Only = false;
                // the f32 recipe that decided the routes, when not x6 (CandidateFacts::selectTag)
                std::string tag;
                // why a route is missing
                std::string why;
            };

            // The compute type sets the precision of the products: 64F / C64F the f64 route
            // only; 16F / 16BF every operand rounded once (nearest even) to that type, so the
            // h16 GEMM route, and the f64 route only when the operands already have that type
            // (it would skip the rounding); 32F / C32F f32 products: operands of f64 data take
            // the f64 route (more precise, and the only route that reads them without a first
            // rounding), f16 x f16 or bf16 x bf16 without an operand operator the h16 route
            // (exact products), the other real operands and complex f32 the route of the f32
            // recipe (their f32 values, every f16 / bf16 value is one): native, the f32 strided
            // GEMM (f32 products and accumulation); x3 / x4 / x6 / x9, the universal GEMM with
            // the operands split into bf16 parts (cf32 routes for complex data); f64, the f64
            // route only (also for 16-bit operands). The f64 route is a candidate of every
            // recipe: no recipe admits a route less accurate than the one it names.
            ContractionRoutes contractionRoutes(Problem const& p, F32Recipe recipe)
            {
                ContractionRoutes r;
                auto const&       A       = p.t[0];
                auto const&       B       = p.t[1];
                bool const        cplx    = isComplex(p.t[3].type);
                auto const        f64Data = [](hiptensorDataType_t t) {
                    return t == HIPTENSOR_R_64F || t == HIPTENSOR_C_64F;
                };
                switch(p.compute)
                {
                case HIPTENSOR_COMPUTE_DESC_64F:
                case HIPTENSOR_COMPUTE_DESC_C64F:
                    r.f64 = true;
                    r.why = "the f64 compute type takes the f64 route only";
                    return r;
                case HIPTENSOR_COMPUTE_DESC_16F:
                case HIPTENSOR_COMPUTE_DESC_16BF:
                {
                    if(cplx)
                    {
                        r.why = "a 16-bit compute type is not supported for complex data";
                        return r;
                    }
                    bool const f16  = p.compute == HIPTENSOR_COMPUTE_DESC_16F;
                    auto const type = f16 ? HIPTENSOR_R_16F : HIPTENSOR_R_16BF;
                    r.gemm          = true;
                    r.route         = "h16";
                    r.storage       = f16 ? "fp16" : "bf16";
                    r.f64           = A.type == type && B.type == type;
                    if(!r.f64)
                    {
                        r.why = "the f64 route does not round the operands to the 16-bit compute "
                                "type";
                    }
                    return r;
                }
                case HIPTENSOR_COMPUTE_DESC_32F:
                case HIPTENSOR_COMPUTE_DESC_C32F:
                {
                    r.f64 = true;
                    if(f64Data(A.type) || f64Data(B.type))
                    {
                        r.why = "operands of f64 data take the f64 route";
                        return r;
                    }
                    if(recipe == F32Recipe::F64)
                    {
                        r.f64Only = true;
                        r.tag     = f32RecipeName(recipe);
                        r.why     = "HIPTENSOR_ROCKE_F32_RECIPE=f64 takes the f64 route only";
                        return r;
                    }
                    bool const ops
                        = A.unaryOp != HIPTENSOR_OP_IDENTITY || B.unaryOp != HIPTENSOR_OP_IDENTITY;
                    if(!cplx && A.type == B.type && !ops
                       && (A.type == HIPTENSOR_R_16F || A.type == HIPTENSOR_R_16BF))
                    {
                        r.gemm    = true;
                        r.route   = "h16";
                        r.storage = A.type == HIPTENSOR_R_16F ? "fp16" : "bf16";
                        return r;
                    }
                    // the operators of 16-bit operands run in f32 and their results stay f32
                    // (no rounding back to the 16-bit type), as do mixed 16-bit operands
                    if(recipe != F32Recipe::X6)
                    {
                        r.tag = f32RecipeName(recipe);
                    }
                    if(recipe == F32Recipe::Native)
                    {
                        r.gett32 = true;
                        return r;
                    }
                    r.gemm    = true;
                    r.storage = "bf16";
                    if(cplx)
                    {
                        r.route = recipe == F32Recipe::X6
                                      ? std::string("cf32")
                                      : "cf32" + std::string(f32RecipeName(recipe));
                    }
                    else
                    {
                        r.route = f32RecipeName(recipe);
                    }
                    return r;
                }
                default:
                    r.why = "compute type not supported by the rocKE kernels";
                    return r;
                }
            }

            // The API's execution checks of a contraction (the same as the other backend's).
            void contractionChecks(Problem const& problem, Program& program)
            {
                program.nullStatus      = HIPTENSOR_STATUS_INVALID_VALUE;
                program.requiredScalars = {0};
                program.requiredBuffers
                    = {{BufferRole::A, -1}, {BufferRole::B, -1}, {BufferRole::Out, -1}};
                if(problem.t[2].present)
                {
                    program.requiredScalars.push_back(1);
                    program.requiredBuffers.emplace_back(BufferRole::C, 1);
                }
            }

            bool isDecline(hiptensorStatus_t status)
            {
                return status == HIPTENSOR_STATUS_NOT_SUPPORTED
                       || status == HIPTENSOR_STATUS_ARCH_MISMATCH;
            }

            // ------------------------------------------------- contractions read and written in place

            constexpr char kDgett[] = "dgett";
            constexpr char kRgett[] = "rgett";

            // A contraction mode: its extent and its strides in the kernel's A, B, C and D (0
            // where the tensor does not have it).
            struct GMode
            {
                int64_t                extent = 1;
                std::array<int64_t, 4> stride{};
            };

            std::vector<GMode> modesOf(Group const& g)
            {
                std::vector<GMode> out;
                for(size_t i = 0; i < g.modes.size(); i++)
                {
                    if(g.extent[i] != 1)
                    {
                        out.push_back({g.extent[i], g.stride[i]});
                    }
                }
                return out;
            }

            // Orders modes by tensor t's strides (innermost first, stable).
            void sortBy(std::vector<GMode>& modes, int t)
            {
                std::stable_sort(modes.begin(), modes.end(), [t](GMode const& a, GMode const& b) {
                    return std::llabs(a.stride[t]) < std::llabs(b.stride[t]);
                });
            }

            // Consecutive modes (innermost first) contiguous in every tensor of `tensors` merge
            // into one digit while the extent stays below 2^31 (gett_host.merge_modes).
            std::vector<GMode> mergeModes(std::vector<GMode> const& modes,
                                          std::vector<int> const&   tensors)
            {
                std::vector<GMode> out;
                for(auto const& m : modes)
                {
                    if(!out.empty())
                    {
                        auto& last = out.back();
                        bool  ok   = last.extent * m.extent <= kI32Max;
                        for(int t : tensors)
                        {
                            ok = ok && m.stride[t] == last.stride[t] * last.extent;
                        }
                        if(ok)
                        {
                            last.extent *= m.extent;
                            continue;
                        }
                    }
                    out.push_back(m);
                }
                return out;
            }

            int64_t spanElems(std::vector<GMode> const& modes, int t)
            {
                int64_t span = 1;
                for(auto const& m : modes)
                {
                    span += (m.extent - 1) * std::llabs(m.stride[t]);
                }
                return span;
            }

            // A digit of a K plan: radix (extent), mode index, multiplier.
            struct KTriple
            {
                int64_t radix = 1;
                int     mode  = 0;
                int64_t mult  = 1;
            };

            // gett_host.k_split: the in-tile and tile digits of K extents `exts` (innermost
            // first) at the K tile `tk`, or false when the K space does not split there.
            bool kSplit(std::vector<int64_t> const& exts,
                        int64_t                     tk,
                        std::vector<KTriple>&       local,
                        std::vector<KTriple>&       tile)
            {
                local.clear();
                tile.clear();
                int64_t rem = tk;
                for(int i = 0; i < int(exts.size()); i++)
                {
                    int64_t const e = exts[i];
                    if(rem == 1)
                    {
                        tile.push_back({e, i, 1});
                    }
                    else if(e <= rem)
                    {
                        if(rem % e != 0)
                        {
                            return false;
                        }
                        local.push_back({e, i, 1});
                        rem /= e;
                    }
                    else
                    {
                        if(e % rem != 0)
                        {
                            return false;
                        }
                        local.push_back({rem, i, 1});
                        tile.push_back({e / rem, i, rem});
                        rem = 1;
                    }
                }
                return rem == 1;
            }

            // gett_host.k_split_padded: one mode padded to the smallest extent that splits, the
            // padded mode an in-tile digit; the least padded K.
            bool kSplitPadded(std::vector<int64_t> const& exts,
                              int64_t                     tk,
                              std::vector<KTriple>&       local,
                              std::vector<KTriple>&       tile,
                              int&                        padMode,
                              int64_t&                    padReal,
                              int64_t&                    padVolume)
            {
                bool found = false;
                for(int j = 0; j < int(exts.size()); j++)
                {
                    int64_t rem = tk;
                    for(int x = 0; x < j; x++)
                    {
                        if(exts[x] > rem || rem % exts[x] != 0)
                        {
                            rem = 0;
                            break;
                        }
                        rem /= exts[x];
                    }
                    if(rem == 0 || exts[j] >= rem)
                    {
                        continue;
                    }
                    int64_t d = exts[j] + 1;
                    while(rem % d != 0)
                    {
                        d++;
                    }
                    auto padded = exts;
                    padded[j]   = d;
                    std::vector<KTriple> lo, ti;
                    if(!kSplit(padded, tk, lo, ti))
                    {
                        continue;
                    }
                    bool inTile = false;
                    for(auto const& x : lo)
                    {
                        inTile = inTile || (x.mode == j && x.radix == d);
                    }
                    if(!inTile)
                    {
                        continue;
                    }
                    int64_t vol = 1;
                    for(auto e : padded)
                    {
                        vol *= e;
                    }
                    if(!found || vol < padVolume)
                    {
                        found     = true;
                        padVolume = vol;
                        local     = lo;
                        tile      = ti;
                        padMode   = j;
                        padReal   = exts[j];
                    }
                }
                return found;
            }

            struct KDigit
            {
                int64_t extent = 1;
                int64_t sA = 0, sB = 0;
            };

            // How a kernel walks the K modes with its K tile (gett_host.k_plan).
            struct KPlan
            {
                std::vector<KDigit> local, tile;
                int                 padSlot  = -1; // in-tile digit that must stay below padLimit
                int64_t             padLimit = 0;
                int64_t             ragged   = -1; // kl + tk * (first tile digit) < ragged
                int64_t             tiles    = 1;
            };

            constexpr int kKOrderSearch = 6;

            // `kmodes`: strides of the kernel's A in slot 0 and B in slot 1. Orders tried: every
            // order of at most kKOrderSearch merged modes (else A's and B's memory orders); an
            // exact split first (smallest strides innermost), then one padded in-tile mode (least
            // padded K), then (`ragged`) the innermost mode of A's order in tiles of tk, the
            // last one cut at its extent.
            bool kPlan(std::vector<GMode> kmodes,
                       int64_t            tk,
                       size_t             slotsLocal,
                       size_t             slotsTile,
                       bool               ragged,
                       KPlan&             out,
                       std::string&       reason)
            {
                kmodes.erase(std::remove_if(kmodes.begin(),
                                            kmodes.end(),
                                            [](GMode const& m) { return m.extent == 1; }),
                             kmodes.end());
                if(kmodes.empty())
                {
                    kmodes.push_back(GMode{});
                }
                sortBy(kmodes, 0);
                auto const                      byA = mergeModes(kmodes, {0, 1});
                std::vector<std::vector<GMode>> orders;
                if(byA.size() <= size_t(kKOrderSearch))
                {
                    std::vector<int> idx(byA.size());
                    std::iota(idx.begin(), idx.end(), 0);
                    do
                    {
                        std::vector<GMode> o;
                        for(int i : idx)
                        {
                            o.push_back(byA[i]);
                        }
                        orders.push_back(std::move(o));
                    } while(std::next_permutation(idx.begin(), idx.end()));
                }
                else
                {
                    orders.push_back(byA);
                    auto byB = kmodes;
                    sortBy(byB, 1);
                    orders.push_back(mergeModes(byB, {0, 1}));
                }
                auto key = [](std::vector<GMode> const& o) {
                    std::vector<int64_t> k;
                    for(auto const& m : o)
                    {
                        k.push_back(std::min(std::llabs(m.stride[0]), std::llabs(m.stride[1])));
                    }
                    return k;
                };
                auto extsOf = [](std::vector<GMode> const& o) {
                    std::vector<int64_t> e;
                    for(auto const& m : o)
                    {
                        e.push_back(m.extent);
                    }
                    return e;
                };
                int                  best = -1;
                std::vector<int64_t> bestKey;
                std::vector<KTriple> bestLocal, bestTile;
                int                  padMode = -1;
                int64_t              padReal = 0;
                for(int o = 0; o < int(orders.size()); o++)
                {
                    std::vector<KTriple> lo, ti;
                    if(!kSplit(extsOf(orders[o]), tk, lo, ti) || lo.size() > slotsLocal
                       || ti.size() > slotsTile)
                    {
                        continue;
                    }
                    auto k = key(orders[o]);
                    if(best < 0 || k < bestKey)
                    {
                        best      = o;
                        bestKey   = k;
                        bestLocal = lo;
                        bestTile  = ti;
                    }
                }
                if(best < 0)
                {
                    for(int o = 0; o < int(orders.size()); o++)
                    {
                        std::vector<KTriple> lo, ti;
                        int                  j    = -1;
                        int64_t              real = 0, vol = 0;
                        if(!kSplitPadded(extsOf(orders[o]), tk, lo, ti, j, real, vol)
                           || lo.size() > slotsLocal || ti.size() > slotsTile)
                        {
                            continue;
                        }
                        auto k = key(orders[o]);
                        k.insert(k.begin(), vol);
                        if(best < 0 || k < bestKey)
                        {
                            best      = o;
                            bestKey   = k;
                            bestLocal = lo;
                            bestTile  = ti;
                            padMode   = j;
                            padReal   = real;
                        }
                    }
                }
                out = KPlan{};
                if(best >= 0)
                {
                    auto const& order = orders[best];
                    auto        digit = [&order](KTriple const& x) {
                        return KDigit{x.radix,
                                      x.mult * order[x.mode].stride[0],
                                      x.mult * order[x.mode].stride[1]};
                    };
                    for(size_t s = 0; s < bestLocal.size(); s++)
                    {
                        out.local.push_back(digit(bestLocal[s]));
                        if(padMode >= 0 && bestLocal[s].mode == padMode && out.padSlot < 0)
                        {
                            out.padSlot  = static_cast<int>(s);
                            out.padLimit = padReal;
                        }
                    }
                    for(auto const& x : bestTile)
                    {
                        out.tile.push_back(digit(x));
                        out.tiles *= x.radix;
                    }
                    return true;
                }
                if(!ragged)
                {
                    reason = "the K extents do not split at the K tile";
                    return false;
                }
                auto const& m0 = byA[0];
                out.local.push_back({tk, m0.stride[0], m0.stride[1]});
                out.tile.push_back({ceilDiv(m0.extent, tk), tk * m0.stride[0], tk * m0.stride[1]});
                for(size_t i = 1; i < byA.size(); i++)
                {
                    out.tile.push_back({byA[i].extent, byA[i].stride[0], byA[i].stride[1]});
                }
                if(out.tile.size() > slotsTile)
                {
                    reason = "too many K digits";
                    return false;
                }
                for(auto const& x : out.tile)
                {
                    out.tiles *= x.extent;
                }
                out.ragged = m0.extent;
                return true;
            }

            // The kernel arguments of a digit space: <prefix>e<i>, the magic pair <prefix>m<i>,
            // <prefix>s<i> of every slot but the last, the strides <prefix><name><i>.
            bool setDigits(ArgWriter&                                                   w,
                           std::string const&                                           prefix,
                           std::vector<std::pair<int64_t, std::vector<int64_t>>> const& digits,
                           std::vector<std::string> const&                              names,
                           int                                                          slots)
            {
                if(int(digits.size()) > slots)
                {
                    return false;
                }
                bool ok = true;
                for(int i = 0; ok && i < slots; i++)
                {
                    int64_t const e  = i < int(digits.size()) ? digits[i].first : 1;
                    auto const    is = std::to_string(i);
                    ok               = ok && w.set(prefix + "e" + is, e);
                    if(i < slots - 1)
                    {
                        ok = ok
                             && setMagic(
                                 w, (prefix + "m" + is).c_str(), (prefix + "s" + is).c_str(), e);
                    }
                    for(size_t n = 0; ok && n < names.size(); n++)
                    {
                        int64_t const s = e != 1 ? digits[i].second[n] : 0;
                        ok              = w.set(prefix + names[n] + is, s);
                    }
                }
                return ok;
            }

            std::vector<std::pair<int64_t, std::vector<int64_t>>>
                digitsOf(std::vector<GMode> const& modes, std::vector<int> const& tensors)
            {
                std::vector<std::pair<int64_t, std::vector<int64_t>>> out;
                for(auto const& m : modes)
                {
                    std::vector<int64_t> s;
                    for(int t : tensors)
                    {
                        s.push_back(m.stride[t]);
                    }
                    out.emplace_back(m.extent, s);
                }
                return out;
            }

            std::vector<std::pair<int64_t, std::vector<int64_t>>>
                digitsOf(std::vector<KDigit> const& ks)
            {
                std::vector<std::pair<int64_t, std::vector<int64_t>>> out;
                for(auto const& k : ks)
                {
                    out.emplace_back(k.extent, std::vector<int64_t>{k.sA, k.sB});
                }
                return out;
            }

            bool f64Type(hiptensorDataType_t t)
            {
                return t == HIPTENSOR_R_64F || t == HIPTENSOR_C_64F;
            }

            int dtCode(hiptensorDataType_t t)
            {
                switch(t)
                {
                case HIPTENSOR_R_16F:
                    return 0;
                case HIPTENSOR_R_16BF:
                    return 1;
                case HIPTENSOR_R_32F:
                case HIPTENSOR_C_32F:
                    return 2;
                default:
                    return 3;
                }
            }

            // ---------------------------------------------------------------- direct GETT

            // The direct GETT kernel's compute class (kernels.common.hiptensor_aot.dgett): the
            // MFMA operand type and the rounding of f64 operands to a 16-bit compute type.
            struct DgettClass
            {
                std::string comp;
                bool        cplx = false;
                int         rnd  = 0; // 1 f16, 2 bf16
                bool        math = false;
            };

            bool dgettClass(Problem const& p, DgettClass& out, std::string& reason)
            {
                auto const& A    = p.t[0];
                auto const& B    = p.t[1];
                auto const& C    = p.t[2];
                auto const& D    = p.t[3];
                out              = DgettClass{};
                out.cplx         = isComplex(D.type);
                bool const f64In = f64Type(A.type) || f64Type(B.type);
                for(auto const* t : {&A, &B, &C})
                {
                    out.math = out.math || (t->present && mathUnary(t->unaryOp));
                }
                switch(p.compute)
                {
                case HIPTENSOR_COMPUTE_DESC_64F:
                case HIPTENSOR_COMPUTE_DESC_C64F:
                    out.comp = "f64";
                    return true;
                case HIPTENSOR_COMPUTE_DESC_16F:
                case HIPTENSOR_COMPUTE_DESC_16BF:
                {
                    if(out.cplx)
                    {
                        reason = "a 16-bit compute type is not supported for complex data";
                        return false;
                    }
                    bool const f16 = p.compute == HIPTENSOR_COMPUTE_DESC_16F;
                    if(f64In)
                    {
                        // f64 operands rounded once to the compute type, products in f64
                        out.comp = "f64";
                        out.rnd  = f16 ? 1 : 2;
                    }
                    else
                    {
                        out.comp = f16 ? "f16" : "bf16";
                    }
                    return true;
                }
                case HIPTENSOR_COMPUTE_DESC_32F:
                case HIPTENSOR_COMPUTE_DESC_C32F:
                {
                    bool const anyF64 = f64In || (out.cplx && (f64Type(C.type) || f64Type(D.type)));
                    if(anyF64)
                    {
                        out.comp = "f64";
                        return true;
                    }
                    bool const ops
                        = A.unaryOp != HIPTENSOR_OP_IDENTITY || B.unaryOp != HIPTENSOR_OP_IDENTITY;
                    if(!out.cplx && A.type == B.type && !ops
                       && (A.type == HIPTENSOR_R_16F || A.type == HIPTENSOR_R_16BF))
                    {
                        // exact products of the 16-bit values, accumulated in f32
                        out.comp = A.type == HIPTENSOR_R_16F ? "f16" : "bf16";
                        return true;
                    }
                    out.comp = "f32";
                    return true;
                }
                default:
                    reason = "compute type not supported by the rocKE kernels";
                    return false;
                }
            }

            // The arrangement of a contraction on the direct GETT (gett_host.dgett_plan).
            struct DgettLayout
            {
                bool                   swapped = false;
                char                   aFast = 'k', bFast = 'k';
                std::vector<GMode>     M, N, L; // merged, innermost first; strides of A, B, C, D
                KPlan                  k;
                int64_t                m = 1, n = 1, l = 1;
                bool                   idx64 = false;
                std::array<int64_t, 4> spanBytes{};
            };

            bool dgettLayout(Problem const&         p,
                             Classified const&      c,
                             int64_t                tk,
                             std::array<int64_t, 4> es,
                             PackKernel const&      anyKernel,
                             DgettLayout&           out,
                             std::string&           reason)
            {
                bool const bilinear = p.t[2].present;
                auto       grab     = [bilinear](Group const& g) {
                    auto ms = modesOf(g);
                    if(!bilinear)
                    {
                        for(auto& m : ms)
                        {
                            m.stride[2] = m.stride[3];
                        }
                    }
                    return ms;
                };
                auto M = grab(c.M), N = grab(c.N), K = grab(c.K), L = grab(c.L);
                out = DgettLayout{};
                for(auto const& m : M)
                {
                    out.swapped = out.swapped || m.stride[3] == 1;
                }
                if(out.swapped)
                {
                    for(auto* g : {&M, &N, &K, &L})
                    {
                        for(auto& m : *g)
                        {
                            std::swap(m.stride[0], m.stride[1]);
                        }
                    }
                    std::swap(M, N);
                    std::swap(es[0], es[1]);
                }
                auto fast = [&K](std::vector<GMode> const& rows, int t) {
                    for(auto const& m : K)
                    {
                        if(m.stride[t] == 1)
                        {
                            return 'k';
                        }
                    }
                    for(auto const& m : rows)
                    {
                        if(m.stride[t] == 1)
                        {
                            return 'r';
                        }
                    }
                    return 'k';
                };
                out.aFast = fast(M, 0);
                out.bFast = fast(N, 1);
                auto all  = M;
                all.insert(all.end(), N.begin(), N.end());
                all.insert(all.end(), K.begin(), K.end());
                all.insert(all.end(), L.begin(), L.end());
                for(int t = 0; t < 4; t++)
                {
                    for(auto const& m : all)
                    {
                        if(m.stride[t] < 0)
                        {
                            reason = "a negative stride";
                            return false;
                        }
                    }
                    out.spanBytes[t] = spanElems(all, t) * es[t];
                    out.idx64        = out.idx64 || out.spanBytes[t] > kI32Max;
                }
                sortBy(M, out.aFast == 'r' ? 0 : 3);
                sortBy(N, out.bFast == 'r' ? 1 : 3);
                sortBy(L, 3);
                out.M = mergeModes(M, {0, 2, 3});
                out.N = mergeModes(N, {1, 2, 3});
                out.L = mergeModes(L, {0, 1, 2, 3});
                if(out.M.size() > size_t(anyKernel.attr("slots_mn", 0))
                   || out.N.size() > size_t(anyKernel.attr("slots_mn", 0))
                   || out.L.size() > size_t(anyKernel.attr("slots_l", 0)))
                {
                    reason = "more index digits than the direct GETT's slots";
                    return false;
                }
                if(!kPlan(K,
                          tk,
                          size_t(anyKernel.attr("slots_kl", 0)),
                          size_t(anyKernel.attr("slots_kt", 0)),
                          true,
                          out.k,
                          reason))
                {
                    return false;
                }
                for(auto const& g : out.M)
                {
                    out.m *= g.extent;
                }
                for(auto const& g : out.N)
                {
                    out.n *= g.extent;
                }
                for(auto const& g : out.L)
                {
                    out.l *= g.extent;
                }
                if(out.m > kI32Max || out.n > kI32Max || out.k.tiles * tk > kI32Max)
                {
                    reason = "an index space past 2^31";
                    return false;
                }
                return true;
            }

            // A contraction on the direct GETT: one launch, the operands read and D written
            // through their own strides, no workspace. `f64Compute`: products and accumulation
            // in f64 whatever the compute class (HIPTENSOR_ROCKE_F32_RECIPE=f64).
            hiptensorStatus_t planDgett(Problem const&    p,
                                        Pack const&       pack,
                                        Classified const& c,
                                        Program&          program,
                                        CandidateFacts&   facts,
                                        std::string&      reason,
                                        bool              f64Compute = false)
            {
                DgettClass cls;
                if(!dgettClass(p, cls, reason))
                {
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(f64Compute)
                {
                    // the operands' values exactly (a 32F / C32F compute type rounds nothing)
                    cls.comp = "f64";
                    cls.rnd  = 0;
                }
                auto const& A    = p.t[0];
                auto const& C    = p.t[2];
                auto const& D    = p.t[3];
                auto        find = [&](std::string const& comp, int idx64, char af, char bf) {
                    return pack.find(
                        kDgett,
                        {{"cplx", cls.cplx ? 1 : 0}, {"math", cls.math ? 1 : 0}, {"idx64", idx64}},
                        {{"comp", comp},
                         {"a_fast", std::string(1, af)},
                         {"b_fast", std::string(1, bf)}});
                };
                // a target without the f32 atom (gfx90a) computes f32 in f64
                if(cls.comp == "f32" && find("f32", 0, 'k', 'k') == nullptr)
                {
                    cls.comp = "f64";
                }
                auto const* any = find(cls.comp, 0, 'k', 'k');
                if(any == nullptr)
                {
                    reason = "the pack has no direct GETT kernel for this compute class";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const tk = any->attr("tile_k"), tm = any->attr("tile_m"),
                              tn = any->attr("tile_n");
                std::array<int64_t, 4> es{};
                for(int t = 0; t < 4; t++)
                {
                    auto const& x = p.t[t == 2 && !C.present ? 3 : t];
                    es[t]         = int64_t(elementBytes(dtypeName(x.type), cls.cplx));
                }
                DgettLayout lay;
                if(!dgettLayout(p, c, tk, es, *any, lay, reason))
                {
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const* kernel = find(cls.comp, lay.idx64 ? 1 : 0, lay.aFast, lay.bFast);
                if(kernel == nullptr)
                {
                    reason = "the pack has no direct GETT kernel for this layout";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const tilesM = ceilDiv(lay.m, tm), tilesN = ceilDiv(lay.n, tn);
                if(tilesM * tilesN > kI32Max || lay.l > kernel->attr("limit_grid_yz", kGridYZMax))
                {
                    reason = "the direct GETT's grid is past its limits";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                Program prog;
                prog.pack           = &pack;
                prog.scalarType     = p.scalarType;
                prog.complexScalars = cls.cplx;
                prog.workspaceBytes = 0;
                LaunchStep step;
                step.kernel = kernel;
                ArgWriter  w(step);
                auto const ka = lay.swapped ? 1 : 0, kb = lay.swapped ? 0 : 1;
                bool       ok = w.pointer("A", ka == 0 ? BufferRole::A : BufferRole::B)
                          && w.pointer("B", kb == 0 ? BufferRole::A : BufferRole::B)
                          && w.pointer("C", C.present ? BufferRole::C : BufferRole::Out)
                          && w.pointer("D", BufferRole::Out);
                if(cls.cplx)
                {
                    ok = ok && w.scalar("alpha_re", ScalarRole::Alpha, false)
                         && w.scalar("alpha_im", ScalarRole::Alpha, true)
                         && w.scalar("beta_re", C.present ? ScalarRole::Beta : ScalarRole::Zero)
                         && w.scalar(
                             "beta_im", C.present ? ScalarRole::Beta : ScalarRole::Zero, true);
                }
                else
                {
                    ok = ok && w.scalar("alpha", ScalarRole::Alpha)
                         && w.scalar("beta", C.present ? ScalarRole::Beta : ScalarRole::Zero);
                }
                auto const& kA    = p.t[ka];
                auto const& kB    = p.t[kb];
                auto const  cType = C.present ? C.type : D.type;
                ok = ok && w.set("dtA", dtCode(kA.type)) && w.set("dtB", dtCode(kB.type))
                     && w.set("dtC", dtCode(cType)) && w.set("dtD", dtCode(D.type))
                     && w.set("opA", kA.unaryOp) && w.set("opB", kB.unaryOp)
                     && w.set("opC", C.present ? C.unaryOp : HIPTENSOR_OP_IDENTITY)
                     && w.set("rnd", cls.rnd) && w.set("M", lay.m) && w.set("N", lay.n)
                     && w.set("nk", lay.k.tiles) && w.set("tiles_n", tilesN)
                     && setMagic(w, "tn_m", "tn_s", tilesN);
                for(int t = 0; t < 4; t++)
                {
                    char const* names[] = {"a_bytes", "b_bytes", "c_bytes", "d_bytes"};
                    ok = ok && w.set(names[t], std::min<int64_t>(lay.spanBytes[t], kI32Max));
                }
                int const mn = int(kernel->attr("slots_mn")), kt = int(kernel->attr("slots_kt")),
                          kl = int(kernel->attr("slots_kl")), sl = int(kernel->attr("slots_l"));
                ok = ok && setDigits(w, "m", digitsOf(lay.M, {0, 2, 3}), {"A", "C", "D"}, mn)
                     && setDigits(w, "n", digitsOf(lay.N, {1, 2, 3}), {"B", "C", "D"}, mn)
                     && setDigits(w, "kl", digitsOf(lay.k.local), {"A", "B"}, kl)
                     && setDigits(w, "kt", digitsOf(lay.k.tile), {"A", "B"}, kt)
                     && setDigits(w, "l", digitsOf(lay.L, {0, 1, 2, 3}), {"A", "B", "C", "D"}, sl)
                     && w.set("kps", lay.k.padSlot) && w.set("kpl", lay.k.padLimit)
                     && w.set("krl", lay.k.ragged >= 0 ? lay.k.ragged : (int64_t(1) << 30));
                if(!ok)
                {
                    reason = "direct GETT arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                step.grid
                    = dim3(static_cast<uint32_t>(tilesM * tilesN), 1, static_cast<uint32_t>(lay.l));
                step.block = dim3(static_cast<uint32_t>(kernel->attr("block", 256)), 1, 1);
                if(auto s = finishStep(step, w, reason); s != HIPTENSOR_STATUS_SUCCESS)
                {
                    return s;
                }
                prog.steps.push_back(std::move(step));
                prog.description = kernel->name;
                contractionChecks(p, prog);
                facts              = contractionFacts(p, c, cls.cplx);
                facts.route        = Route::DirectGett;
                facts.direct       = true;
                facts.f64Route     = cls.comp == "f64" && !f64Type(A.type);
                facts.tileM        = tm;
                facts.tileN        = tn;
                facts.tileK        = tk;
                facts.block        = kernel->attr("block", 256);
                facts.ctas         = tilesM * tilesN * lay.l;
                facts.paddedVolume = tilesM * tm * tilesN * tn * lay.k.tiles * tk * lay.l;
                facts.main         = kernel;
                facts.bytes        = uint64_t(lay.spanBytes[0] + lay.spanBytes[1]
                                       + lay.spanBytes[3] * (C.present ? 2 : 1));
                program            = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // ---------------------------------------------------------------- ring GETT

            // A contraction on rocKE's ring GETT body with runtime index maps
            // (gett_host.rgett_plan): f16 / bf16 data and output of one type, f32 accumulation,
            // no operators, no batch modes, the operands' fastest free runs even, K splitting at
            // the K tile (one padded in-tile mode allowed), C laid out as D. The direct GETT is
            // the program's fallback for zero scales (the ring body's epilogue reads every term)
            // and for buffers not aligned to its vector accesses.
            //
            // `packing`: the operands whose fastest free run is not a multiple of 8 elements (or
            // not even, or whose fastest mode is a K mode) are first copied into the workspace
            // with their free modes as one run (padded to a multiple of 8) followed by their K
            // modes, so that the ring body reads them with 16-byte lanes (rocKE's pack-hybrid
            // plan); the other operands are read in place.
            hiptensorStatus_t planRgett(Problem const&    p,
                                        Pack const&       pack,
                                        Classified const& c,
                                        bool              packing,
                                        uint64_t          workspaceLimit,
                                        Program&          program,
                                        CandidateFacts&   facts,
                                        std::string&      reason)
            {
                auto const& A = p.t[0];
                auto const& B = p.t[1];
                auto const& C = p.t[2];
                auto const& D = p.t[3];
                auto const  T = D.type;
                if(T != HIPTENSOR_R_16F && T != HIPTENSOR_R_16BF)
                {
                    reason = "the ring GETT serves f16 and bf16 data";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                bool const computeOk
                    = p.compute == HIPTENSOR_COMPUTE_DESC_32F
                      || (p.compute == HIPTENSOR_COMPUTE_DESC_16F && T == HIPTENSOR_R_16F)
                      || (p.compute == HIPTENSOR_COMPUTE_DESC_16BF && T == HIPTENSOR_R_16BF);
                if(!computeOk || A.type != T || B.type != T || (C.present && C.type != T)
                   || A.unaryOp != HIPTENSOR_OP_IDENTITY || B.unaryOp != HIPTENSOR_OP_IDENTITY
                   || (C.present && C.unaryOp != HIPTENSOR_OP_IDENTITY))
                {
                    reason = "the ring GETT needs one 16-bit type, f32 accumulation and no "
                             "operators";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto M = modesOf(c.M), N = modesOf(c.N), K = modesOf(c.K);
                if(!modesOf(c.L).empty() || M.empty() || N.empty() || K.empty())
                {
                    reason = "the ring GETT needs M, N and K modes and no batch modes";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                bool const bilinear = C.present;
                for(auto const* g : {&M, &N, &K})
                {
                    for(auto const& m : *g)
                    {
                        for(int t = 0; t < 4; t++)
                        {
                            if(m.stride[t] < 0)
                            {
                                reason = "a negative stride";
                                return HIPTENSOR_STATUS_NOT_SUPPORTED;
                            }
                        }
                        if(g != &K && bilinear && m.stride[2] != m.stride[3])
                        {
                            reason = "the ring GETT reads C with D's strides";
                            return HIPTENSOR_STATUS_NOT_SUPPORTED;
                        }
                    }
                }
                // the kernel's A owns D's fastest mode
                bool    swapped = false;
                int64_t dmin    = INT64_MAX;
                for(auto const* g : {&M, &N})
                {
                    for(auto const& m : *g)
                    {
                        if(m.stride[3] < dmin)
                        {
                            dmin    = m.stride[3];
                            swapped = g == &N;
                        }
                    }
                }
                int const ga = swapped ? 1 : 0, gb = swapped ? 0 : 1;
                auto      rowsA = swapped ? N : M;
                auto      rowsB = swapped ? M : N;
                // the operand's DTL lane width: 8 (16 bytes) or 2 (4 bytes) elements
                auto vecOf = [&K](std::vector<GMode> const& dig, int t) {
                    if(dig[0].stride[t] != 1)
                    {
                        return int64_t(1);
                    }
                    for(int64_t v = 8; v > 1; v /= 2)
                    {
                        bool ok = dig[0].extent % v == 0;
                        for(size_t i = 1; ok && i < dig.size(); i++)
                        {
                            ok = dig[i].stride[t] % v == 0;
                        }
                        for(auto const& m : K)
                        {
                            ok = ok && m.stride[t] % v == 0;
                        }
                        if(ok)
                        {
                            return v;
                        }
                    }
                    return int64_t(1);
                };
                std::stable_sort(rowsA.begin(), rowsA.end(), [ga](GMode const& a, GMode const& b) {
                    return a.stride[ga] < b.stride[ga];
                });
                std::stable_sort(rowsB.begin(), rowsB.end(), [gb](GMode const& a, GMode const& b) {
                    return a.stride[gb] < b.stride[gb];
                });
                auto       digA = mergeModes(rowsA, {ga});
                auto       digB = mergeModes(rowsB, {gb});
                int64_t    vecA = vecOf(digA, ga), vecB = vecOf(digB, gb);
                bool const packA = packing && vecA < 8, packB = packing && vecB < 8;
                if(packing && !packA && !packB)
                {
                    reason = "both operands are read in place";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if((!packA && vecA < 2) || (!packB && vecB < 2))
                {
                    reason = "an operand's fastest free run is not a multiple of 2";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                vecA = packA || vecA >= 8 ? 8 : 2;
                vecB = packB || vecB >= 8 ? 8 : 2;
                // the packed copy of an operand: its free modes (in the ring body's row order)
                // as one run padded to a multiple of 8, then its K modes in its memory order
                struct Packed
                {
                    int64_t              rows = 1, rowsPadded = 1;
                    std::vector<int64_t> kStride; // per K mode (K's order)
                    uint64_t             offset = 0, bytes = 0;
                };
                Packed     pk[2];
                uint64_t   workspace = 0;
                auto const packOf    = [&](int role, std::vector<GMode> const& rows, int t) {
                    Packed out;
                    for(auto const& x : rows)
                    {
                        out.rows *= x.extent;
                    }
                    out.rowsPadded = roundUp(out.rows, 8);
                    std::vector<size_t> order(K.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                        return K[a].stride[t] < K[b].stride[t];
                    });
                    out.kStride.assign(K.size(), 0);
                    int64_t s = out.rowsPadded;
                    for(auto i : order)
                    {
                        out.kStride[i] = s;
                        s *= K[i].extent;
                    }
                    out.offset = workspace;
                    out.bytes  = uint64_t(s) * 2;
                    workspace  = alignUp(workspace + out.bytes, kWorkspaceAlign);
                    static_cast<void>(role);
                    return out;
                };
                if(packA)
                {
                    pk[0] = packOf(0, rowsA, ga);
                    digA  = {GMode{pk[0].rowsPadded, {1, 1, 1, 1}}};
                }
                if(packB)
                {
                    pk[1] = packOf(1, rowsB, gb);
                    digB  = {GMode{pk[1].rowsPadded, {1, 1, 1, 1}}};
                }
                if(workspace > workspaceLimit)
                {
                    reason = "the packed operands of the ring GETT need more workspace than the "
                             "limit";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                std::vector<GMode> kmodes;
                for(size_t i = 0; i < K.size(); i++)
                {
                    GMode k;
                    k.extent    = K[i].extent;
                    k.stride[0] = packA ? pk[0].kStride[i] : K[i].stride[ga];
                    k.stride[1] = packB ? pk[1].kStride[i] : K[i].stride[gb];
                    kmodes.push_back(k);
                }
                // any class of the pack: its slot counts and K tile
                auto const* any = pack.find(kRgett, {{"wide", 0}}, {{"dtype", dtypeName(T)}});
                if(any == nullptr)
                {
                    reason = "the pack has no ring GETT kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const tk = any->attr("tile_k");
                KPlan         kp;
                if(!kPlan(kmodes,
                          tk,
                          size_t(any->attr("slots_kl")),
                          size_t(any->attr("slots_kt")),
                          false,
                          kp,
                          reason))
                {
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const dRow = mergeModes(rowsA, {3});
                auto const dCol = mergeModes(rowsB, {3});
                auto       dvec = [&](int64_t v) {
                    if(dRow[0].stride[3] != 1 || dRow[0].extent % v != 0)
                    {
                        return false;
                    }
                    for(size_t i = 1; i < dRow.size(); i++)
                    {
                        if(dRow[i].stride[3] % v != 0)
                        {
                            return false;
                        }
                    }
                    for(auto const& m : dCol)
                    {
                        if(m.stride[3] % v != 0)
                        {
                            return false;
                        }
                    }
                    return true;
                };
                int64_t const vecD = dvec(4) ? 4 : dvec(2) ? 2 : 1;
                bool const    pair = vecD == 4 && dvec(8);
                size_t const  rows = size_t(any->attr("slots_row"));
                if(digA.size() > rows || digB.size() > rows || dRow.size() > rows
                   || dCol.size() > rows)
                {
                    reason = "more row digits than the ring GETT's slots";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                std::vector<GMode> all = M;
                all.insert(all.end(), N.begin(), N.end());
                all.insert(all.end(), K.begin(), K.end());
                int64_t const es = 2;
                for(int t : {0, 1})
                {
                    bool const packed = t == ga ? packA : packB;
                    uint64_t   bytes  = packed ? pk[t == ga ? 0 : 1].bytes : spanElems(all, t) * es;
                    if(bytes > uint64_t(any->attr("limit_operand_bytes")))
                    {
                        reason = "an operand past the ring GETT's buffer range";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                // D (and C) past the narrow classes' offsets: the wide classes (masked offsets)
                bool const wideD
                    = spanElems(all, 3) * es > any->attr("limit_d_bytes")
                      || (bilinear && spanElems(all, 2) * es > any->attr("limit_d_bytes"));
                auto const* anyWide = pack.find(kRgett, {{"wide", 1}}, {{"dtype", dtypeName(T)}});
                if(wideD
                   && (anyWide == nullptr || spanElems(all, 3) * es > anyWide->attr("limit_d_bytes")
                       || (bilinear && spanElems(all, 2) * es > anyWide->attr("limit_d_bytes"))))
                {
                    reason = "D past the ring GETT's 32-bit offsets";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t m = 1, n = 1;
                for(auto const& x : rowsA)
                {
                    m *= x.extent;
                }
                for(auto const& x : rowsB)
                {
                    n *= x.extent;
                }
                int64_t const kPadded = kp.tiles * tk;
                if(m > kI32Max || n > kI32Max || kPadded > kI32Max
                   || ceilDiv(m, any->attr("tile_m")) > any->attr("limit_grid_yz"))
                {
                    reason = "the ring GETT's index spaces or grid are past their limits";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                // the class of the smallest digit-slot bucket that holds every digit space (its
                // kernel skips the unused slots), else the class of every slot; bilinear classes
                // of every slot store 8 bytes per lane
                size_t const      digits = std::max({digA.size(),
                                                     digB.size(),
                                                     dRow.size(),
                                                     dCol.size(),
                                                     kp.local.size(),
                                                     kp.tile.size()});
                PackKernel const* kernel = nullptr;
                for(int64_t slots : {int64_t(2), int64_t(4), int64_t(0)})
                {
                    if(kernel != nullptr || (slots != 0 && digits > size_t(slots)))
                    {
                        continue;
                    }
                    bool const classPair = pair && !(bilinear && slots == 0);
                    kernel               = pack.find(kRgett,
                                                     {{"bilinear", bilinear ? 1 : 0},
                                                      {"vec_a", vecA},
                                                      {"vec_b", vecB},
                                                      {"vec_d", vecD},
                                                      {"pair", classPair ? 1 : 0},
                                                      {"wide", wideD ? 1 : 0},
                                                      {"slots", slots}},
                                                     {{"dtype", dtypeName(T)}});
                }
                if(kernel == nullptr)
                {
                    reason = "the pack has no ring GETT class for this layout";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                // the fallback: the direct GETT (zero scales, unaligned buffers)
                Program        fallback;
                CandidateFacts fbFacts;
                std::string    why;
                if(planDgett(p, pack, c, fallback, fbFacts, why) != HIPTENSOR_STATUS_SUCCESS)
                {
                    reason = "the ring GETT has no direct GETT fallback: " + why;
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                Program prog;
                prog.pack           = &pack;
                prog.scalarType     = p.scalarType;
                prog.workspaceBytes = workspace;
                // the pack launches (the generic elementwise kernel): the free modes, then the K
                // modes, into the workspace
                for(int r = 0; r < 2; r++)
                {
                    if(!(r == 0 ? packA : packB))
                    {
                        continue;
                    }
                    int const        t    = r == 0 ? ga : gb;
                    auto const&      rows = r == 0 ? rowsA : rowsB;
                    std::vector<Dim> dims;
                    int64_t          s = 1;
                    for(auto const& x : rows)
                    {
                        Dim d;
                        d.extent    = x.extent;
                        d.stride[0] = x.stride[t];
                        d.stride[1] = s;
                        s *= x.extent;
                        dims.push_back(d);
                    }
                    for(size_t i = 0; i < K.size(); i++)
                    {
                        Dim d;
                        d.extent    = K[i].extent;
                        d.stride[0] = K[i].stride[t];
                        d.stride[1] = pk[r].kStride[i];
                        dims.push_back(d);
                    }
                    canonicalize(dims, 2, 1);
                    EwOperand in;
                    in.name  = "A";
                    in.role  = t == 0 ? BufferRole::A : BufferRole::B;
                    in.scale = ScalarRole::One;
                    EwOperand out;
                    out.name           = "D";
                    out.role           = BufferRole::Workspace;
                    out.bytes          = pk[r].offset;
                    size_t const first = prog.steps.size();
                    std::vector<std::pair<std::string, std::string>> const strs{
                        {"a", dtypeName(T)}, {"d", dtypeName(T)}};
                    auto status = ewSteps(pack,
                                          {{"arity", 1}, {"cplx", 0}},
                                          strs,
                                          dims,
                                          {in, out},
                                          {0, 0},
                                          false,
                                          prog.steps,
                                          reason);
                    if(status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                    tileLaunch(pack, strs, dims, {in, out}, {0, 0}, false, prog.steps, first);
                }
                LaunchStep step;
                step.kernel = kernel;
                ArgWriter  w(step);
                auto const roleOf = [](int t) { return t == 0 ? BufferRole::A : BufferRole::B; };
                bool       ok     = (packA ? w.pointer("A", BufferRole::Workspace, pk[0].offset)
                                           : w.pointer("A", roleOf(ga)))
                          && (packB ? w.pointer("B", BufferRole::Workspace, pk[1].offset)
                                    : w.pointer("B", roleOf(gb)))
                          && w.pointer("C", BufferRole::Out) && w.set("M", m) && w.set("N", n)
                          && w.set("K", kPadded) && w.scalar("alpha", ScalarRole::Alpha);
                if(bilinear)
                {
                    ok = ok && w.pointer("Cin", BufferRole::C)
                         && w.scalar("beta", ScalarRole::Beta);
                }
                auto rowDigits = [](std::vector<GMode> const& g, int t) {
                    std::vector<std::pair<int64_t, std::vector<int64_t>>> out;
                    for(auto const& x : g)
                    {
                        out.emplace_back(x.extent, std::vector<int64_t>{x.stride[t]});
                    }
                    return out;
                };
                int const kl = int(kernel->attr("slots_kl")), kt = int(kernel->attr("slots_kt"));
                ok = ok && setDigits(w, "ra", rowDigits(digA, ga), {"S"}, int(rows))
                     && setDigits(w, "rb", rowDigits(digB, gb), {"S"}, int(rows))
                     && setDigits(w, "dr", rowDigits(dRow, 3), {"S"}, int(rows))
                     && setDigits(w, "dc", rowDigits(dCol, 3), {"S"}, int(rows))
                     && setDigits(w, "kl", digitsOf(kp.local), {"A", "B"}, kl)
                     && setDigits(w, "kt", digitsOf(kp.tile), {"A", "B"}, kt)
                     && w.set("kps", kp.padSlot) && w.set("kpl", kp.padLimit);
                if(!ok)
                {
                    reason = "ring GETT arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const tilesM = ceilDiv(m, kernel->attr("tile_m"));
                int64_t const tilesN = ceilDiv(n, kernel->attr("tile_n"));
                step.grid  = dim3(static_cast<uint32_t>(tilesN), static_cast<uint32_t>(tilesM), 1);
                step.block = dim3(static_cast<uint32_t>(kernel->attr("block", 256)), 1, 1);
                if(auto s = finishStep(step, w, reason); s != HIPTENSOR_STATUS_SUCCESS)
                {
                    return s;
                }
                prog.steps.push_back(std::move(step));
                prog.description = (packA || packB ? "pk:" : "") + kernel->name;
                contractionChecks(p, prog);
                prog.fallback       = std::make_shared<Program>(std::move(fallback));
                prog.fallbackOnZero = {ScalarRole::Alpha};
                if(bilinear)
                {
                    prog.fallbackOnZero.push_back(ScalarRole::Beta);
                }
                uint32_t const outAlign = pair ? 16u : uint32_t(2 * vecD);
                prog.fallbackAlign      = {{BufferRole::Out, outAlign}};
                if(!packA)
                {
                    prog.fallbackAlign.emplace_back(roleOf(ga), uint32_t(2 * vecA));
                }
                if(!packB)
                {
                    prog.fallbackAlign.emplace_back(roleOf(gb), uint32_t(2 * vecB));
                }
                if(bilinear)
                {
                    prog.fallbackAlign.emplace_back(BufferRole::C, outAlign);
                }
                facts              = contractionFacts(p, c, false);
                facts.route        = Route::RingGett;
                facts.direct       = true;
                facts.tileM        = kernel->attr("tile_m");
                facts.tileN        = kernel->attr("tile_n");
                facts.tileK        = tk;
                facts.block        = kernel->attr("block", 256);
                facts.ctas         = tilesM * tilesN;
                facts.paddedVolume = tilesM * facts.tileM * tilesN * facts.tileN * kPadded;
                facts.main         = kernel;
                facts.bytes        = uint64_t(spanElems(all, 0) + spanElems(all, 1)
                                       + spanElems(all, 3) * (bilinear ? 2 : 1))
                              * 2;
                program = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }

            hiptensorStatus_t enumerateContraction(Problem const&          problem,
                                                   Pack const&             pack,
                                                   CandidateOptions const& options,
                                                   uint64_t                workspaceLimit,
                                                   std::vector<Candidate>& out,
                                                   std::string&            reason)
            {
                if(problem.kind != OpKind::Contraction)
                {
                    reason = "not a contraction";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const& A = problem.t[0];
                auto const& B = problem.t[1];
                auto const& C = problem.t[2];
                auto const& D = problem.t[3];
                if(!A.present || !B.present || !D.present)
                {
                    reason = "contraction tensors missing";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                // Data types: A, B, C and D each of any real type (f16, bf16, f32, f64), or each
                // of any complex type (complex f32, complex f64). The compute type decides the
                // route and the precision of the operands; mixed types are converted when the
                // operands are packed (contractionRoutes).
                bool const cplx = isComplex(D.type);
                for(auto const* t : {&A, &B, &C, &D})
                {
                    if(!t->present)
                    {
                        continue;
                    }
                    if(dtypeName(t->type).empty())
                    {
                        reason = "data type not supported by the rocKE kernels";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    if(isComplex(t->type) != cplx)
                    {
                        reason = "contractions of real and complex tensors are not supported by "
                                 "the rocKE kernels";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                if(isComplex(problem.scalarType) != cplx)
                {
                    reason = "the compute type of a contraction of "
                             + std::string(cplx ? "complex" : "real")
                             + " tensors is not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const routes = contractionRoutes(problem, options.f32Recipe);
                if(!routes.gemm && !routes.gett32 && !routes.f64)
                {
                    reason = routes.why;
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                bool const f64Only = options.f64Only || routes.f64Only;
                for(auto const* t : {&A, &B})
                {
                    bool okOp = t->unaryOp == HIPTENSOR_OP_IDENTITY
                                || (cplx && t->unaryOp == HIPTENSOR_OP_CONJ)
                                || (!cplx && supportedUnary(t->unaryOp)
                                    && t->unaryOp != HIPTENSOR_OP_CONJ);
                    if(!okOp)
                    {
                        reason = "operand operator not supported by the rocKE kernels";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                if(C.present
                   && !(C.unaryOp == HIPTENSOR_OP_IDENTITY
                        || (!cplx && supportedUnary(C.unaryOp) && C.unaryOp != HIPTENSOR_OP_CONJ)
                        || (cplx && C.unaryOp == HIPTENSOR_OP_CONJ)))
                {
                    reason = "operator on C not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(D.unaryOp != HIPTENSOR_OP_IDENTITY && D.unaryOp != HIPTENSOR_OP_UNKNOWN)
                {
                    reason = "operator on D not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                Classified c;
                if(auto status = classify(problem, c, reason); status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                // K order: A's memory order; M by A, N by B (packing reads), L by D.
                c.K.orderBy(0);
                c.M.orderBy(0);
                c.N.orderBy(1);
                c.L.orderBy(3);

                std::string firstReason;
                auto        note = [&firstReason](std::string const& why) {
                    if(firstReason.empty())
                    {
                        firstReason = why;
                    }
                };
                auto add = [&](std::string key, Program&& program, CandidateFacts const& facts) {
                    contractionChecks(problem, program);
                    Candidate candidate;
                    candidate.key             = std::move(key);
                    candidate.program         = std::move(program);
                    candidate.facts           = facts;
                    candidate.facts.selectTag = routes.tag;
                    candidate.staticRank      = static_cast<int>(out.size());
                    candidate.tuned
                        = facts.route == Route::TunedGett || facts.route == Route::JitGett;
                    out.push_back(std::move(candidate));
                };

                // rocKE's tuned GETT route for the problems it measured
                if(!f64Only && !options.disableTuned)
                {
                    Program     program;
                    std::string why;
                    if(planTunedGett(problem, pack, workspaceLimit, program, why))
                    {
                        auto facts  = contractionFacts(problem, c, cplx);
                        facts.route = Route::TunedGett;
                        facts.main  = program.steps.back().kernel;
                        facts.ctas  = int64_t(program.steps.back().grid.x)
                                     * program.steps.back().grid.y * program.steps.back().grid.z;
                        facts.block        = program.steps.back().block.x;
                        facts.paddedVolume = facts.M * facts.N * facts.K * facts.L;
                        auto const& d      = program.description;
                        auto        hex    = d.substr(0, d.rfind(':'));
                        // the direct GETT for zero scales (the tuned epilogue reads every term)
                        // and buffers not aligned to the tuned kernels' 16-byte accesses
                        Program        fallback;
                        CandidateFacts fbFacts;
                        std::string    fbWhy;
                        if(planDgett(problem, pack, c, fallback, fbFacts, fbWhy)
                           == HIPTENSOR_STATUS_SUCCESS)
                        {
                            program.fallback       = std::make_shared<Program>(std::move(fallback));
                            program.fallbackOnZero = {ScalarRole::Alpha};
                            program.fallbackAlign  = {
                                {BufferRole::A, 16u}, {BufferRole::B, 16u}, {BufferRole::Out, 16u}};
                            if(C.present)
                            {
                                program.fallbackOnZero.push_back(ScalarRole::Beta);
                                program.fallbackAlign.emplace_back(BufferRole::C, 16u);
                            }
                            // the pack steps' outputs (16-byte stores, read by the GETT's 16-byte
                            // DMA loads) sit at 256-byte offsets of the workspace
                            if(program.workspaceBytes > 0)
                            {
                                program.fallbackAlign.emplace_back(BufferRole::Workspace, 16u);
                            }
                        }
                        add("tuned:" + hex.substr(hex.find(':') + 1), std::move(program), facts);
                    }
                    else if(!why.empty())
                    {
                        note(why);
                    }
                }

#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
                // a problem the route tables do not hold: the route rocKE's tensor GETT planner
                // chooses, built at plan creation (HIPTENSOR_JIT_MODE_DEFAULT). The direct GETT
                // is its fallback for zero scales (the GETT epilogue reads every term), buffers
                // not aligned to its 16-byte accesses and another device than the plan's (the
                // JIT kernels are loaded there only); without one it is not a candidate.
                if(options.jitGett && !f64Only && !options.disableTuned
                   && (out.empty() || !out.back().tuned))
                {
                    std::shared_ptr<GettJitPlan const> jit;
                    GettJitPlanInfo                    info;
                    std::string                        why;
                    Program                            fallback;
                    CandidateFacts                     fbFacts;
                    std::string                        fbWhy;
                    if(!planGettJit(problem, pack, workspaceLimit, jit, info, why))
                    {
                        note("GETT JIT: " + why);
                    }
                    else if(planDgett(problem, pack, c, fallback, fbFacts, fbWhy)
                            != HIPTENSOR_STATUS_SUCCESS)
                    {
                        note("GETT JIT: no direct GETT fallback: " + fbWhy);
                    }
                    else
                    {
                        contractionChecks(problem, fallback);
                        Program program;
                        program.pack           = &pack;
                        program.pending        = true;
                        program.deviceBound    = true;
                        program.scalarType     = HIPTENSOR_R_32F;
                        program.workspaceBytes = alignUp(info.workspaceBytes, kWorkspaceAlign);
                        program.description    = "jit-gett:" + info.route + ":" + info.kernel;
                        program.fallback       = std::make_shared<Program>(std::move(fallback));
                        program.fallbackOnZero = {ScalarRole::Alpha};
                        program.fallbackAlign
                            = {{BufferRole::A, 16u}, {BufferRole::B, 16u}, {BufferRole::Out, 16u}};
                        if(C.present)
                        {
                            program.fallbackOnZero.push_back(ScalarRole::Beta);
                            program.fallbackAlign.emplace_back(BufferRole::C, 16u);
                        }
                        if(program.workspaceBytes > 0)
                        {
                            program.fallbackAlign.emplace_back(BufferRole::Workspace, 16u);
                        }
                        auto facts         = contractionFacts(problem, c, cplx);
                        facts.route        = Route::JitGett;
                        facts.tileM        = info.tileM;
                        facts.tileN        = info.tileN;
                        facts.tileK        = info.tileK;
                        facts.block        = info.block;
                        facts.ctas         = info.ctas;
                        facts.paddedVolume = facts.M * facts.N * facts.K * facts.L;
                        facts.launches     = info.launches;
                        facts.gettKind     = info.kind;
                        facts.predictedNs  = info.predictedNs;
                        add("jit:" + info.kernel, std::move(program), facts);
                        out.back().jitGett = std::move(jit);
                    }
                }
#endif

                // rocKE's ring GETT body with runtime index maps: the operands read and D
                // written in place, preferred when the cell's selection model does not score it
                // (f32 accumulation: not under the f64 recipe)
                if(!f64Only)
                {
                    // the pack-hybrid form first (the operands with short runs packed), then the
                    // operands read in place; the first one that applies is preferred
                    bool preferred = false;
                    for(bool packing : {true, false})
                    {
                        Program        program;
                        CandidateFacts facts;
                        std::string    why;
                        auto           status = planRgett(
                            problem, pack, c, packing, workspaceLimit, program, facts, why);
                        if(status == HIPTENSOR_STATUS_SUCCESS)
                        {
                            add(std::string(packing ? "rgett:pk:" : "rgett:") + facts.main->name,
                                std::move(program),
                                facts);
                            out.back().preferred = !preferred;
                            preferred            = true;
                        }
                        else if(isDecline(status))
                        {
                            note(why);
                        }
                        else
                        {
                            reason = why;
                            return status;
                        }
                    }
                }

                // the GEMM route of the data and compute types
                if(!f64Only && routes.gemm)
                {
                    std::string const& route   = routes.route;
                    std::string const& storage = routes.storage;
                    int64_t const      nk      = c.N.size() * (cplx ? 2 : 1);
                    int const          cus     = std::max(1, problem.device.cuCount);
                    // the bodies of the route, then (only for operands or a result past their
                    // 32-bit offsets) the 64-bit-address body
                    bool pastOffsets = false;
                    auto bodies      = gemmBodies(pack, c.M.size(), nk, cus, cplx);
                    for(size_t i = 0; i <= bodies.size(); i++)
                    {
                        bool const wide = i == bodies.size();
                        if(wide && !pastOffsets)
                        {
                            break;
                        }
                        Program        program;
                        CandidateFacts facts;
                        std::string    why;
                        auto           status = planGemmBody(problem,
                                                   pack,
                                                   c,
                                                   route,
                                                   storage,
                                                   wide ? std::string(kGemmWideBody) : bodies[i],
                                                   wide,
                                                   workspaceLimit,
                                                   program,
                                                   facts,
                                                   why);
                        if(status == HIPTENSOR_STATUS_SUCCESS)
                        {
                            add("gemm:" + facts.main->name, std::move(program), facts);
                        }
                        else if(isDecline(status))
                        {
                            pastOffsets = pastOffsets || why == kPastGemmOffsets;
                            note(why);
                        }
                        else
                        {
                            reason = why;
                            return status;
                        }
                    }
                }

                // the strided GEMM with f32 words (`f32`: the native route of f32 data) or f64
                // words (the route of f64 / complex f64 data, and the F64 route of the other
                // types: operands converted to f64, the result rounded once); a status other
                // than success ends the enumeration
                auto stridedGemm = [&](bool f32) -> hiptensorStatus_t {
                    Classified cd = c;
                    cd.M.orderBy(3);
                    cd.N.orderBy(3);
                    cd.L.orderBy(3);
                    auto const    layout = gett64Layout(problem, cd, cplx, f32);
                    int64_t const Kp     = roundUp(cd.K.size(), f32 && !cplx ? 4 : 2);
                    int const     cus    = std::max(1, problem.device.cuCount);
                    int64_t const kbMin
                        = static_cast<int64_t>(pack.routeDouble("gett64.k_block_min_k", 4096));
                    auto const cls = gett64Class(
                        pack.arch(), cplx, cd.M.size(), cd.N.size(), Kp, cd.L.size(), cus);
                    auto const kernels = gett64Kernels(pack, layout, cplx, cls, Kp >= kbMin, f32);
                    if(kernels.empty())
                    {
                        note(f32 ? "the pack has no f32 GEMM kernel for this layout class"
                                 : "the pack has no f64 GEMM kernel for this layout class");
                    }
                    for(auto const* kernel : kernels)
                    {
                        Program        program;
                        CandidateFacts facts;
                        std::string    why;
                        auto           status = planGett64(problem,
                                                 pack,
                                                 cd,
                                                 cplx,
                                                 layout,
                                                 *kernel,
                                                 workspaceLimit,
                                                 program,
                                                 facts,
                                                 why,
                                                 f32);
                        if(status == HIPTENSOR_STATUS_SUCCESS)
                        {
                            add((f32              ? "gett32:"
                                 : facts.f64Route ? "f64:"
                                                  : "gett64:")
                                    + kernel->name,
                                std::move(program),
                                facts);
                        }
                        else if(isDecline(status))
                        {
                            note(why);
                        }
                        else
                        {
                            reason = why;
                            return status;
                        }
                    }
                    return HIPTENSOR_STATUS_SUCCESS;
                };
                if(!f64Only && routes.gett32)
                {
                    if(auto status = stridedGemm(true); status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                }
                if(!routes.f64)
                {
                    note(routes.why);
                }
                else if(auto status = stridedGemm(false); status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                // the direct GETT: every contraction, in place, no workspace (after the others
                // in the static order)
                if(!options.f64Only)
                {
                    Program        program;
                    CandidateFacts facts;
                    std::string    why;
                    auto status = planDgett(problem, pack, c, program, facts, why, routes.f64Only);
                    if(status == HIPTENSOR_STATUS_SUCCESS)
                    {
                        // on a target without the GEMM routes (gfx90a, gfx942: no GEMM kernel in
                        // the pack) the direct GETT of a 16-bit or f32 compute class is preferred
                        // to the f64 route. A property of the target, not of the candidates
                        // within the workspace limit, so that a plan with the workspace of the
                        // estimate selects the estimate's kernel.
                        // The native f32 route (gett32) of such a target (gfx942) is the
                        // recipe's own route and stays ahead: the direct GETT is preferred to
                        // the f64 route only (also a property of the target and the recipe).
                        bool const gemmTarget = pack.find(kGemm, {}, {}) != nullptr;
                        bool const ownRoute
                            = routes.gett32 && pack.find(kGett32, {}, {}) != nullptr;
                        bool const prefer
                            = !gemmTarget && !ownRoute && facts.main->attrStr("comp") != "f64";
                        add("dgett:" + facts.main->name, std::move(program), facts);
                        out.back().preferred = prefer;
                    }
                    else if(isDecline(status))
                    {
                        note(why);
                    }
                    else
                    {
                        reason = why;
                        return status;
                    }
                }
                if(out.empty())
                {
                    reason = firstReason.empty() ? "no rocKE kernel serves this contraction"
                                                 : firstReason;
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // ------------------------------------------------------------- trinary contraction

            // The intermediate T = opA(A) * opB(B) of a trinary contraction as hipTensor defines
            // it (hiptensor_contraction.cpp: computeTrinaryContractionIntermediateModes,
            // buildTrinaryContractionIntermediateDescriptor): the modes of A that the first step
            // does not contract, in A's order, then the modes of B that A does not have; E's
            // data type; packed column-major strides. A mode of A and B that neither C nor E
            // has is contracted by the first step.
            TensorArg intermediateOf(Problem const& p)
            {
                auto const& A   = p.t[0];
                auto const& B   = p.t[1];
                auto const& C   = p.t[2];
                auto const& E   = p.t[4];
                auto        has = [](TensorArg const& t, int32_t m) {
                    return std::find(t.modes.begin(), t.modes.end(), m) != t.modes.end();
                };
                auto contracted
                    = [&](int32_t m) { return has(A, m) && has(B, m) && !has(C, m) && !has(E, m); };
                TensorArg T;
                T.present        = true;
                T.type           = E.type;
                T.unaryOp        = HIPTENSOR_OP_IDENTITY;
                T.alignmentBytes = static_cast<uint32_t>(kWorkspaceAlign);
                int64_t stride   = 1;
                auto    add      = [&](int32_t mode, int64_t extent) {
                    T.modes.push_back(mode);
                    T.extents.push_back(extent);
                    T.strides.push_back(stride);
                    stride
                        = extent > 0 && stride > INT64_MAX / extent ? INT64_MAX : stride * extent;
                };
                for(int i = 0; i < A.numModes(); i++)
                {
                    if(!contracted(A.modes[i]))
                    {
                        add(A.modes[i], A.extents[i]);
                    }
                }
                for(int i = 0; i < B.numModes(); i++)
                {
                    if(!has(A, B.modes[i]) && !contracted(B.modes[i]))
                    {
                        add(B.modes[i], B.extents[i]);
                    }
                }
                return T;
            }

            // The program of a trinary contraction from a program of each step: the first writes
            // T at the start of the workspace (alpha 1, no C), the second reads T as its A, C as
            // its B and D as its C. Both steps use the workspace after T; each step's zero fills
            // run right before its launches.
            Program combineTrinary(Problem const& problem,
                                   Program const& first,
                                   Program const& second,
                                   uint64_t       tBytes)
            {
                Program out;
                out.pack           = second.pack;
                out.scalarType     = second.scalarType;
                out.complexScalars = second.complexScalars;
                out.workspaceBytes = tBytes + std::max(first.workspaceBytes, second.workspaceBytes);
                out.description = "trinary[" + first.description + "][" + second.description + "]";
                auto append     = [&](Program const& from, bool isFirst) {
                    auto const base = static_cast<uint32_t>(out.steps.size());
                    for(auto m : from.memsets)
                    {
                        m.offset += tBytes;
                        m.beforeStep += base;
                        out.memsets.push_back(m);
                    }
                    // the step's buffers and scalars in the trinary's roles, for its launches
                    // and their fallbacks (taken at execution for misaligned buffers)
                    auto toTrinary = [isFirst, tBytes](LaunchStep& launch) {
                        for(auto& ptr : launch.pointers)
                        {
                            switch(ptr.role)
                            {
                            case BufferRole::Workspace:
                                ptr.bytes += tBytes;
                                break;
                            case BufferRole::Out:
                                // the first step's output is T, at offset 0
                                ptr.role = isFirst ? BufferRole::Workspace : BufferRole::Out;
                                break;
                            case BufferRole::A:
                                ptr.role = isFirst ? BufferRole::A : BufferRole::Workspace;
                                break;
                            case BufferRole::B:
                                ptr.role = isFirst ? BufferRole::B : BufferRole::C;
                                break;
                            case BufferRole::C:
                                ptr.role = isFirst ? BufferRole::None : BufferRole::D;
                                break;
                            default:
                                break;
                            }
                        }
                        if(isFirst)
                        {
                            for(auto& s : launch.scalars)
                            {
                                s.role = s.role == ScalarRole::Alpha ? ScalarRole::One
                                             : s.role == ScalarRole::One ? ScalarRole::One
                                                                         : ScalarRole::Zero;
                            }
                        }
                    };
                    for(auto step : from.steps)
                    {
                        remapLaunch(step, toTrinary);
                        out.steps.push_back(std::move(step));
                    }
                };
                append(first, true);
                append(second, false);
                // the API's execution checks (the same as the other backend's)
                out.nullStatus      = HIPTENSOR_STATUS_INVALID_VALUE;
                out.requiredScalars = {0};
                out.requiredBuffers = {{BufferRole::A, -1},
                                       {BufferRole::B, -1},
                                       {BufferRole::C, -1},
                                       {BufferRole::Out, -1}};
                if(problem.t[3].present)
                {
                    out.requiredScalars.push_back(1);
                    out.requiredBuffers.emplace_back(BufferRole::D, 1);
                }
                // a step with a fallback program: the trinary program falls back to the pair of
                // fallbacks (the conditions in the trinary's roles; T, at the workspace's start,
                // is aligned; the first step's scalars are one and zero)
                if(first.fallback != nullptr || second.fallback != nullptr)
                {
                    auto const& f1 = first.fallback != nullptr ? *first.fallback : first;
                    auto const& f2 = second.fallback != nullptr ? *second.fallback : second;
                    out.fallback
                        = std::make_shared<Program>(combineTrinary(problem, f1, f2, tBytes));
                    for(auto const& [role, align] : first.fallbackAlign)
                    {
                        if(role == BufferRole::A || role == BufferRole::B)
                        {
                            out.fallbackAlign.emplace_back(role, align);
                        }
                    }
                    for(auto role : second.fallbackOnZero)
                    {
                        out.fallbackOnZero.push_back(role);
                    }
                    for(auto const& [role, align] : second.fallbackAlign)
                    {
                        auto r = role == BufferRole::B     ? BufferRole::C
                                 : role == BufferRole::C   ? BufferRole::D
                                 : role == BufferRole::Out ? BufferRole::Out
                                                           : BufferRole::None;
                        if(r != BufferRole::None)
                        {
                            out.fallbackAlign.emplace_back(r, align);
                        }
                    }
                }
                return out;
            }

            // The candidates of a trinary contraction: pairs of a candidate of each step, ordered
            // by the sum of their ranks in their step's selection (options.rankStep). Every
            // candidate of a step is paired with the best one of the other step; the first three
            // of each step are paired with each other.
            hiptensorStatus_t enumerateTrinary(Problem const&          problem,
                                               Pack const&             pack,
                                               CandidateOptions const& options,
                                               uint64_t                workspaceLimit,
                                               std::vector<Candidate>& out,
                                               std::string&            reason)
            {
                auto const& A = problem.t[0];
                auto const& B = problem.t[1];
                auto const& C = problem.t[2];
                auto const& E = problem.t[4];
                if(!A.present || !B.present || !C.present || !E.present)
                {
                    reason = "trinary contraction tensors missing";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const tType = dtypeName(E.type);
                if(tType.empty())
                {
                    reason = "data type not supported by the rocKE kernels";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                auto const     T        = intermediateOf(problem);
                uint64_t const es       = elementBytes(tType, isComplex(E.type));
                uint64_t       elements = 1;
                for(auto e : T.extents)
                {
                    if(e > 1 && elements > std::numeric_limits<uint64_t>::max() / 2 / es / e)
                    {
                        reason = "the intermediate T = A * B of the trinary contraction is too "
                                 "large";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    elements *= uint64_t(std::max<int64_t>(e, 0));
                }
                uint64_t const tBytes = alignUp(elements * es, kWorkspaceAlign);
                if(tBytes > workspaceLimit)
                {
                    reason = "the intermediate T = A * B of the trinary contraction needs more "
                             "workspace than the limit";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                uint64_t const subLimit = workspaceLimit - tBytes;

                Problem first  = problem;
                first.kind     = OpKind::Contraction;
                first.t        = {A, B, TensorArg{}, T, TensorArg{}};
                Problem second = problem;
                second.kind    = OpKind::Contraction;
                second.t       = {T, C, problem.t[3], E, TensorArg{}};

                std::vector<Candidate> c1, c2;
                std::string            why;
                // the steps take the pack's kernels (no JIT route)
                CandidateOptions stepOptions = options;
                stepOptions.jitGett          = false;
                if(auto status = enumerateContraction(first, pack, stepOptions, subLimit, c1, why);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    reason = "step 1 (T = A * B) of the trinary contraction: " + why;
                    return status;
                }
                if(auto status = enumerateContraction(second, pack, stepOptions, subLimit, c2, why);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    reason = "step 2 (E = alpha * T * C + beta * D) of the trinary contraction: "
                             + why;
                    return status;
                }
                auto orderOf = [&options](Problem const& p, std::vector<Candidate> const& c) {
                    std::vector<int> order;
                    if(options.rankStep)
                    {
                        order = options.rankStep(p, c);
                    }
                    if(order.size() != c.size())
                    {
                        order.resize(c.size());
                        std::iota(order.begin(), order.end(), 0);
                        std::stable_sort(order.begin(), order.end(), [&c](int x, int y) {
                            return c[x].staticRank < c[y].staticRank;
                        });
                    }
                    return order;
                };
                auto const o1 = orderOf(first, c1);
                auto const o2 = orderOf(second, c2);

                constexpr int                    kCross = 3;
                std::vector<std::pair<int, int>> pairs;
                for(int i = 0; i < int(o1.size()); i++)
                {
                    for(int j = 0; j < int(o2.size()); j++)
                    {
                        if(i == 0 || j == 0 || (i < kCross && j < kCross))
                        {
                            pairs.emplace_back(i, j);
                        }
                    }
                }
                std::stable_sort(pairs.begin(), pairs.end(), [](auto const& x, auto const& y) {
                    return x.first + x.second != y.first + y.second
                               ? x.first + x.second < y.first + y.second
                               : x.first < y.first;
                });
                for(auto const& [i, j] : pairs)
                {
                    auto const& x = c1[o1[i]];
                    auto const& y = c2[o2[j]];
                    Candidate   candidate;
                    candidate.key     = "tri:" + x.key + "+" + y.key;
                    candidate.program = combineTrinary(problem, x.program, y.program, tBytes);
                    candidate.facts   = y.facts;
                    candidate.facts.bytes += x.facts.bytes + 2 * elements * es;
                    candidate.facts.flops += x.facts.flops;
                    candidate.staticRank = static_cast<int>(out.size());
                    out.push_back(std::move(candidate));
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }
        } // namespace

        hiptensorStatus_t enumerateCandidates(Problem const&          original,
                                              Pack const&             pack,
                                              CandidateOptions const& options,
                                              uint64_t                workspaceLimit,
                                              std::vector<Candidate>& out,
                                              std::string&            reason)
        {
            out.clear();
            // CONJ of a real tensor is the identity (as for the other backend's kernels): plan
            // the problem without it, so every route of identity operands applies.
            Problem problem = original;
            for(auto& t : problem.t)
            {
                if(t.present && t.unaryOp == HIPTENSOR_OP_CONJ && !isComplex(t.type))
                {
                    t.unaryOp = HIPTENSOR_OP_IDENTITY;
                }
            }
            switch(problem.kind)
            {
            case OpKind::Permute:
            case OpKind::ElementwiseBinary:
            case OpKind::ElementwiseTrinary:
            {
                Candidate candidate;
                EwLayout  layout;
                auto      status = planElementwise(
                    problem, pack, candidate.program, candidate.facts, reason, &layout);
                if(status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                candidate.key = "ew:" + candidate.program.description;
                std::string why;
                // the flat kernel first when every tensor is one contiguous run in the same order
                // (not with HIPTENSOR_ROCKE_DISABLE_TUNED)
                Candidate flat;
                if(!options.disableTuned
                   && planEwf(layout, pack, candidate.program, flat.program, flat.facts, why)
                          == HIPTENSOR_STATUS_SUCCESS)
                {
                    flat.key        = "ewf:" + flat.program.description;
                    flat.staticRank = static_cast<int>(out.size());
                    out.push_back(std::move(flat));
                }
                // the tiled kernel when an input is fastest along another mode than the output
                // (the generic program is its fallback)
                Candidate tiled;
                if(planEwt(layout, pack, candidate.program, tiled.program, tiled.facts, why)
                   == HIPTENSOR_STATUS_SUCCESS)
                {
                    tiled.key        = "ewt:" + tiled.program.description;
                    tiled.staticRank = static_cast<int>(out.size());
                    out.push_back(std::move(tiled));
                }
                // then the vector class of the generic kernel when an input is read with vector
                // loads along D's fastest mode
                Candidate vector;
                if(planEwv(layout, pack, candidate.program, vector.program, vector.facts, why)
                   == HIPTENSOR_STATUS_SUCCESS)
                {
                    vector.key        = "ewv:" + vector.program.description;
                    vector.staticRank = static_cast<int>(out.size());
                    out.push_back(std::move(vector));
                }
                candidate.staticRank = static_cast<int>(out.size());
                out.push_back(std::move(candidate));
                return status;
            }
            case OpKind::Reduction:
            {
                // rocKE's tuned reduction classes first (the tuned route of a table problem, or
                // the classes of the nearest measured prototypes), then the generic kernels: the
                // rule's mode and split first, then fewer and more partial ranges, then the
                // other mode. Two choices that plan the same launches are one candidate.
                // HIPTENSOR_ROCKE_DISABLE_TUNED: the generic kernels only.
                if(!options.disableTuned)
                {
                    if(auto status = enumerateTunedReduction(problem, pack, workspaceLimit, out);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                }
                struct Choice
                {
                    int mode;
                    int splitWaves;
                };
                Choice const choices[] = {{kRuleMode, 1},
                                          {kRuleMode, 0},
                                          {kRuleMode, 4},
                                          {kRowMode, 1},
                                          {kRowMode, 0},
                                          {kRowMode, 4},
                                          {kColMode, 1},
                                          {kColMode, 0},
                                          {kColMode, 4}};
                std::string  firstReason;
                for(auto const& choice : choices)
                {
                    Candidate   candidate;
                    std::string why;
                    auto        status = planReduction(problem,
                                                pack,
                                                workspaceLimit,
                                                choice.mode,
                                                choice.splitWaves,
                                                candidate.program,
                                                candidate.facts,
                                                why);
                    if(status == HIPTENSOR_STATUS_SUCCESS)
                    {
                        candidate.key = std::string("reduce:")
                                        + (candidate.facts.body == kRowMode ? "row" : "col")
                                        + ":split" + std::to_string(candidate.facts.split);
                        bool const seen
                            = std::any_of(out.begin(), out.end(), [&candidate](auto const& x) {
                                  return x.key == candidate.key;
                              });
                        if(!seen)
                        {
                            candidate.staticRank = static_cast<int>(out.size());
                            out.push_back(std::move(candidate));
                        }
                    }
                    else if(isDecline(status))
                    {
                        if(firstReason.empty())
                        {
                            firstReason = why;
                        }
                    }
                    else
                    {
                        reason = why;
                        return status;
                    }
                }
                if(out.empty())
                {
                    reason = firstReason;
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                return HIPTENSOR_STATUS_SUCCESS;
            }
            case OpKind::Contraction:
                return enumerateContraction(problem, pack, options, workspaceLimit, out, reason);
            case OpKind::ContractionTrinary:
                return enumerateTrinary(problem, pack, options, workspaceLimit, out, reason);
            default:
                reason = "the rocKE backend has no kernels for this operation";
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            }
        }

        hiptensorStatus_t buildJitCandidate(Candidate&                candidate,
                                            backend::DeviceKey const& device,
                                            double                    budgetMs,
                                            double&                   spentMs,
                                            std::string&              note,
                                            std::string&              reason)
        {
            auto& program = candidate.program;
            if(!program.pending)
            {
                return HIPTENSOR_STATUS_SUCCESS;
            }
#ifdef HIPTENSOR_ENABLE_ROCKE_JIT
            if(candidate.jitGett != nullptr)
            {
                GettJitReport report;
                Program       built = program;
                built.steps.clear();
                // the budget left (a remainder of 0 still serves cache hits, compiles nothing)
                double const left = budgetMs <= 0.0 ? 0.0 : std::max(budgetMs - spentMs, 1e-9);
                auto const   t0   = std::chrono::steady_clock::now();
                bool const   ok
                    = buildGettJitProgram(*candidate.jitGett, device, left, built, report, reason);
                spentMs += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - t0)
                               .count();
                if(!ok)
                {
                    note = "rocKE GETT JIT: " + candidate.key + " not built: " + reason;
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                built.pending = false;
                if(hipGetDevice(&built.device) != hipSuccess)
                {
                    reason = "hipGetDevice failed";
                    return HIPTENSOR_STATUS_HIP_ERROR;
                }
                if(built.fallback != nullptr)
                {
                    if(auto status = loadFunctions(*built.fallback);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        reason = "the fallback's kernels do not load";
                        return status;
                    }
                }
                program              = std::move(built);
                candidate.facts.main = program.steps.back().kernel;
                char msg[256];
                snprintf(msg,
                         sizeof(msg),
                         "rocKE GETT JIT: %d kernels (compiled %d, kernel cache hits %d, of which "
                         "from a file %d) in %.3f ms",
                         report.kernels,
                         report.compiled,
                         report.memoryHits,
                         report.fileHits,
                         report.milliseconds);
                note = msg;
                return HIPTENSOR_STATUS_SUCCESS;
            }
#else
            static_cast<void>(device);
            static_cast<void>(budgetMs);
            static_cast<void>(spentMs);
            static_cast<void>(note);
#endif
            reason = "the candidate has no launches";
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }

        hiptensorStatus_t loadFunctions(Program& program)
        {
            if(program.pending)
            {
                return HIPTENSOR_STATUS_INTERNAL_ERROR;
            }
            if(hipGetDevice(&program.device) != hipSuccess)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            for(auto& step : program.steps)
            {
                if(step.deviceBound && step.function != nullptr)
                {
                    continue; // a JIT kernel, loaded when it was built
                }
                if(auto status = program.pack->function(*step.kernel, step.function);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                // a launch's own fallback (the generic launches of a tiled pack or epilogue)
                if(step.fallback != nullptr
                   && std::any_of(step.fallback->begin(),
                                  step.fallback->end(),
                                  [](LaunchStep const& f) { return f.function == nullptr; }))
                {
                    auto fallback = std::make_shared<std::vector<LaunchStep>>(*step.fallback);
                    for(auto& f : *fallback)
                    {
                        if(auto status = program.pack->function(*f.kernel, f.function);
                           status != HIPTENSOR_STATUS_SUCCESS)
                        {
                            return status;
                        }
                    }
                    step.fallback = std::move(fallback);
                }
            }
            if(program.fallback != nullptr)
            {
                return loadFunctions(*program.fallback);
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
    } // namespace rockebackend
} // namespace hiptensor
