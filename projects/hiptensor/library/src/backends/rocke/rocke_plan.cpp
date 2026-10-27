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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <tuple>

#include "rocke_plan.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
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
                    return false;
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
                bool idx64 = false;
                for(int t = 0; t < tensors; t++)
                {
                    idx64 = idx64 || spanOf(dims, t) > kI32Max;
                }
                ints.emplace_back("idx64", idx64 ? 1 : 0);
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
                                       std::vector<EwOperand> const&  operands,
                                       std::vector<uint64_t> const&   bytes,
                                       std::array<int, 2> const&      binaryOps,
                                       std::vector<LaunchStep>&       steps,
                                       int64_t&                       budget,
                                       std::string&                   reason)
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
                        auto const           sub = restrictMode(
                            dims, i, first, std::min(piece, dims[i].extent - first), tensors, offsets);
                        auto moved = operands;
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
            }
            return "?";
        }

        namespace
        {
            // ---------------------------------------------------------------------- elementwise

            hiptensorStatus_t planElementwise(Problem const&  problem,
                                              Pack const&     pack,
                                              Program&        program,
                                              CandidateFacts& facts,
                                              std::string&    reason)
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
                return HIPTENSOR_STATUS_SUCCESS;
            }

            // ------------------------------------------------------------------------ reduction

            // Reduction modes of the generic kernels: a wave per output reading the innermost
            // reduced mode (row), or a thread per output (col).
            constexpr int kRuleMode = -1;
            constexpr int kRowMode  = 0;
            constexpr int kColMode  = 1;

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
                auto a = dtypeName(A.type);
                auto d = dtypeName(D.type);
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
                std::vector<Dim> kept, red;
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
                if(kept.size() > size_t(kMaxRank) || red.size() > size_t(kMaxRank))
                {
                    reason = "more than 8 kept or reduced modes after merging";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                kept.resize(kMaxRank);
                red.resize(kMaxRank);
                int64_t nOut = product(kept, 0, kMaxRank);
                int64_t nRed = product(red, 0, kMaxRank);
                if(nOut > kI32Max || nRed > kI32Max)
                {
                    reason = "more than 2^31 outputs or reduced elements";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                bool const ruleRow
                    = nRed > 1
                      && (nOut == 1
                          || std::llabs(red[0].stride[0]) < std::llabs(kept[0].stride[0]));
                bool const row = mode == kRuleMode ? ruleRow : mode == kRowMode;
                if(row && nRed <= 1)
                {
                    reason = "the row reduction needs a reduced mode";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                bool idx64 = spanOf(kept, 0) + spanOf(red, 0) > kI32Max || spanOf(kept, 1) > kI32Max
                             || spanOf(kept, 2) > kI32Max;
                std::vector<std::pair<std::string, int64_t>>     ints{{"idx64", idx64 ? 1 : 0}};
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
                int64_t const  cus    = std::max(1, problem.device.cuCount);
                uint64_t const wbytes = d == "f64" || a == "f64" ? 8 : 4;

                // Work units and the split that fills the device.
                int64_t split = 1, chunk = 0, nchunk = 0, gridX = 0, gridY = 1, units = 0;
                if(row)
                {
                    nchunk         = ceilDiv(red[0].extent, kWave);
                    int64_t trips  = nchunk * (nRed / red[0].extent);
                    int64_t waves  = nOut;
                    int64_t target = int64_t(splitWaves) * 8 * cus * 4; // waves
                    split          = std::clamp<int64_t>(ceilDiv(target, waves), 1, trips);
                    chunk          = ceilDiv(trips, split);
                    split          = ceilDiv(trips, chunk);
                    gridX          = ceilDiv(nOut, block / kWave);
                    units          = gridX;
                }
                else
                {
                    int64_t outer  = nOut / kept[0].extent;
                    gridX          = ceilDiv(kept[0].extent, block);
                    gridY          = std::min<int64_t>(outer, kGridYZMax);
                    int64_t blocks = gridX * gridY;
                    int64_t target = int64_t(splitWaves) * 4 * cus;
                    split          = std::clamp<int64_t>(ceilDiv(target, blocks), 1, nRed);
                    split          = std::min<int64_t>(split, kGridYZMax);
                    chunk          = ceilDiv(nRed, split);
                    split          = ceilDiv(nRed, chunk);
                    units          = blocks;
                }
                static_cast<void>(units);
                auto workspaceFor
                    = [&](int64_t s) { return s > 1 ? uint64_t(s) * nOut * wbytes : 0; };
                while(split > 1
                      && (workspaceFor(split) > workspaceLimit || split * nOut > kI32Max
                          || (row && split > kGridYZMax)))
                {
                    // Fewer, longer ranges until the partials fit.
                    if(row)
                    {
                        int64_t trips = nchunk * (nRed / red[0].extent);
                        chunk         = ceilDiv(trips, split / 2);
                        split         = ceilDiv(trips, chunk);
                    }
                    else
                    {
                        chunk = ceilDiv(nRed, split / 2);
                        split = ceilDiv(nRed, chunk);
                    }
                }
                if(split == 1)
                {
                    chunk = row ? nchunk * (nRed / red[0].extent) : nRed;
                }

                program                 = Program{};
                program.pack            = &pack;
                program.scalarType      = problem.scalarType;
                program.description     = kernel->name;
                program.workspaceBytes  = alignUp(workspaceFor(split), kWorkspaceAlign);
                program.nullStatus      = HIPTENSOR_STATUS_NOT_INITIALIZED;
                program.requiredScalars = {0, 1};
                program.requiredBuffers
                    = {{BufferRole::A, 0}, {BufferRole::C, 1}, {BufferRole::Out, -1}};

                auto geometry = [&](ArgWriter& w, bool withRed) {
                    bool ok = true;
                    for(int i = 0; i < kMaxRank; i++)
                    {
                        auto n = std::to_string(i);
                        ok     = ok && w.set("ke" + n, kept[i].extent)
                             && w.set("kC" + n, kept[i].extent == 1 ? 0 : kept[i].stride[1])
                             && w.set("kD" + n, kept[i].extent == 1 ? 0 : kept[i].stride[2]);
                        if(withRed)
                        {
                            ok = ok && w.set("kA" + n, kept[i].extent == 1 ? 0 : kept[i].stride[0])
                                 && w.set("re" + n, red[i].extent)
                                 && w.set("rA" + n, red[i].extent == 1 ? 0 : red[i].stride[0]);
                        }
                        if(i < kMaxRank - 1)
                        {
                            ok = ok
                                 && setMagic(
                                     w, ("km" + n).c_str(), ("ks" + n).c_str(), kept[i].extent);
                            if(withRed)
                            {
                                ok = ok
                                     && setMagic(
                                         w, ("rm" + n).c_str(), ("rs" + n).c_str(), red[i].extent);
                            }
                        }
                    }
                    return ok;
                };
                auto common = [&](ArgWriter& w, bool withA) {
                    bool ok = w.pointer("C", C.present ? BufferRole::C : BufferRole::None)
                              && w.pointer("D", BufferRole::Out)
                              && w.pointer("W", BufferRole::Workspace)
                              && w.scalar("alpha", ScalarRole::Alpha)
                              && w.scalar("beta", C.present ? ScalarRole::Beta : ScalarRole::Zero)
                              && w.set("opC", static_cast<int>(opC))
                              && w.set("opR", static_cast<int>(problem.opReduce))
                              && w.set("n_out", nOut) && w.set("split", split);
                    if(withA)
                    {
                        ok = ok && w.pointer("A", BufferRole::A)
                             && w.set("opA", static_cast<int>(A.unaryOp));
                    }
                    return ok;
                };

                LaunchStep step;
                step.kernel = kernel;
                ArgWriter w(step);
                bool      ok = common(w, true) && geometry(w, true);
                if(row)
                {
                    auto magic = magicU31(static_cast<uint32_t>(nchunk));
                    ok         = ok && w.set("nchunk", nchunk) && w.set("jchunk", chunk)
                         && w.setU32("jm", magic.first) && w.set("js", magic.second);
                    step.grid = dim3(static_cast<uint32_t>(gridX), static_cast<uint32_t>(split), 1);
                }
                else
                {
                    ok = ok && w.set("n_red", nRed) && w.set("rchunk", chunk)
                         && w.set("ystep", gridY) && w.set("n_outer", nOut / kept[0].extent);
                    step.grid = dim3(static_cast<uint32_t>(gridX),
                                     static_cast<uint32_t>(gridY),
                                     static_cast<uint32_t>(split));
                }
                step.block = dim3(static_cast<uint32_t>(block), 1, 1);
                if(!ok)
                {
                    reason = "reduction arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(auto status = finishStep(step, w, reason); status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
                program.steps.push_back(std::move(step));
                if(split > 1)
                {
                    LaunchStep fin;
                    fin.kernel = final;
                    ArgWriter fw(fin);
                    if(!(common(fw, false) && geometry(fw, false)))
                    {
                        reason = "reduction arguments do not fit the kernel arguments";
                        return HIPTENSOR_STATUS_NOT_SUPPORTED;
                    }
                    int64_t const fblock = final->attr("block", 256);
                    fin.grid             = dim3(static_cast<uint32_t>(ceilDiv(nOut, fblock)), 1, 1);
                    fin.block            = dim3(static_cast<uint32_t>(fblock), 1, 1);
                    if(auto status = finishStep(fin, fw, reason);
                       status != HIPTENSOR_STATUS_SUCCESS)
                    {
                        return status;
                    }
                    program.steps.push_back(std::move(fin));
                    program.description += "+" + final->name;
                }
                facts           = CandidateFacts{};
                facts.route     = Route::Reduction;
                facts.body      = row ? kRowMode : kColMode;
                facts.M         = nOut;
                facts.K         = nRed;
                facts.modesM    = static_cast<int>(std::count_if(
                    kept.begin(), kept.end(), [](Dim const& x) { return x.extent > 1; }));
                facts.modesK    = static_cast<int>(std::count_if(
                    red.begin(), red.end(), [](Dim const& x) { return x.extent > 1; }));
                facts.inner     = red[0].extent;
                facts.innerKept = kept[0].extent;
                facts.unitA     = red[0].extent > 1 && std::llabs(red[0].stride[0]) == 1     ? 3
                                  : kept[0].extent > 1 && std::llabs(kept[0].stride[0]) == 1 ? 1
                                                                                             : 0;
                facts.split     = split;
                facts.chunk     = chunk;
                facts.block     = block;
                facts.ctas      = int64_t(program.steps[0].grid.x) * program.steps[0].grid.y
                             * program.steps[0].grid.z;
                facts.main      = kernel;
                facts.operandOp = A.unaryOp != HIPTENSOR_OP_IDENTITY;
                facts.bilinear  = C.present;
                facts.bytes     = uint64_t(nOut) * uint64_t(nRed) * elementBytes(a)
                              + uint64_t(nOut) * elementBytes(d) * (C.present ? 2 : 1)
                              + 2 * workspaceFor(split);
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
                out.name    = "D";
                out.role    = BufferRole::Workspace;
                out.bytes   = offset;
                auto status = ewSteps(pack,
                                      {{"arity", 1}, {"cplx", cplx ? 1 : 0}},
                                      {{"a", dtypeName(X.type)}, {"d", toType}},
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
                out.name    = "D";
                out.role    = BufferRole::Out;
                auto dt = dtypeName(D.type);
                // C may have another data type than D (the scale form reads no C: D's type)
                auto ct     = C.present ? dtypeName(C.type) : dt;
                auto status = ewSteps(pack,
                                      {{"arity", 2}, {"cplx", cplx ? 1 : 0}},
                                      {{"a", tType}, {"c", ct}, {"d", dt}},
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

            // f16 / bf16 (and f32 with a 16-bit compute type), f32 and complex f32 contractions
            // on one universal GEMM body with an f32 output: pack A and B, GEMM, epilogue.
            hiptensorStatus_t planGemmBody(Problem const&     p,
                                           Pack const&        pack,
                                           Classified const&  c,
                                           std::string const& route,
                                           std::string const& storage,
                                           std::string const& b,
                                           uint64_t           workspaceLimit,
                                           Program&           program,
                                           CandidateFacts&    facts,
                                           std::string&       reason)
            {
                bool const    cplx = route == "cf32";
                int64_t const M = c.M.size(), N = c.N.size(), K = c.K.size(), L = c.L.size();
                // kernel extents per problem element: K in 16-bit halves (f32 pairs for x6,
                // complex quadruples for cf32), N doubled for complex
                int64_t const kUnit  = route == "h16" ? 1 : route == "x6" ? 2 : 4;
                int64_t const nUnit  = cplx ? 2 : 1;
                int const     cus    = std::max(1, p.device.cuCount);
                auto const*   kernel = pack.find(kGemm,
                                                 {{"batched", L > 1 ? 1 : 0}},
                                                 {{"route", route}, {"dtype", storage}, {"body", b}});
                if(kernel == nullptr)
                {
                    reason = "no " + route + " GEMM body " + b + (L > 1 ? " (batched)" : "");
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
                uint64_t const aBytes = uint64_t(L * Mp * Kp) * es;
                uint64_t const bBytes = uint64_t(L * Np * Kp) * es;
                uint64_t const tElems = uint64_t(L * Mp * Np * nUnit);
                // the kernel's capability limits (pack attributes)
                auto const opLimit = static_cast<uint64_t>(
                    kernel->attr("limit_operand_bytes", int64_t(kDtlBufferRange)));
                auto const cLimit
                    = static_cast<uint64_t>(kernel->attr("limit_c_elements", kI32Max));
                auto const gridLimit = kernel->attr("limit_grid_yz", kGridYZMax);
                if(aBytes >= opLimit || bBytes >= opLimit || tElems > cLimit
                   || ceilDiv(Mp, tm) > gridLimit || L > gridLimit)
                {
                    reason = "operands past the 32-bit offsets of the GEMM bodies";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
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
                LaunchStep g;
                g.kernel = kernel;
                ArgWriter     w(g);
                int64_t const kM = Mp, kN = Np * nUnit, kK = Kp * kUnit;
                bool          ok = w.pointer("A", BufferRole::Workspace, offA)
                          && w.pointer("B", BufferRole::Workspace, offB)
                          && w.pointer("C", BufferRole::Workspace, offT) && w.set("M", kM)
                          && w.set("N", kN) && w.set("K", kK);
                if(parts > 0)
                {
                    ok = ok && w.pointer("SkFlags", BufferRole::Workspace, offFlags)
                         && w.pointer("SkPart", BufferRole::Workspace, offPart);
                    prog.memsets.push_back({offFlags, flags, flags, 1});
                }
                if(L > 1)
                {
                    // per-batch strides in elements of each kernel pointer (16-bit halves for A
                    // and B, f32 for C)
                    int64_t const halves = int64_t(es / 2);
                    ok                   = ok && w.set("stride_a", Mp * Kp * halves)
                         && w.set("stride_b", Np * Kp * halves) && w.set("stride_c", kM * kN);
                }
                int64_t const tilesM = ceilDiv(kM, tm), tilesN = ceilDiv(kN, tn);
                if(kernel->attr("streamk", 0) == 1)
                {
                    g.grid = dim3(static_cast<uint32_t>(P), 1, 1);
                }
                else if(P > 0)
                {
                    g.grid = dim3(static_cast<uint32_t>(
                                      std::max<int64_t>(1, std::min<int64_t>(P, tilesM * tilesN))),
                                  1,
                                  1);
                }
                else
                {
                    g.grid = dim3(static_cast<uint32_t>(tilesN),
                                  static_cast<uint32_t>(tilesM),
                                  static_cast<uint32_t>(L > 1 ? L : 1));
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
                int64_t const ctas = int64_t(g.grid.x) * g.grid.y * g.grid.z;
                prog.description   = kernel->name;
                prog.steps.push_back(std::move(g));
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
                facts.route = route == "h16" ? Route::H16 : route == "x6" ? Route::X6 : Route::CF32;
                facts.body  = gemmBodyId(b);
                facts.tileM = tm;
                facts.tileN = tn;
                facts.tileK = tk;
                facts.block = kernel->attr("block", 256);
                facts.persistent   = P;
                facts.streamK      = kernel->attr("streamk", 0) == 1;
                facts.ctas         = ctas;
                facts.paddedVolume = Mp * Np * Kp * L;
                facts.main         = kernel;
                facts.bytes        = uint64_t(M * K * L) * esA + uint64_t(N * K * L) * esB + 2 * (aBytes + bBytes)
                              + 2 * tBytes + uint64_t(M * N * L) * esOut * (p.t[2].present ? 2 : 1);
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

            Gett64Layout gett64Layout(Problem const& p, Classified const& c, bool cplx)
            {
                auto const&   C = p.t[2];
                auto const&   D = p.t[3];
                int64_t const M = c.M.size(), N = c.N.size();
                Gett64Layout  g;
                bool const    realOut = D.type == HIPTENSOR_R_64F || D.type == HIPTENSOR_C_64F;
                g.dsm                 = c.M.mergedStride(3);
                g.dsn                 = c.N.mergedStride(3);
                g.dsl                 = c.L.mergedStride(3);
                g.csm                 = C.present ? c.M.mergedStride(2) : g.dsm;
                g.csn                 = C.present ? c.N.mergedStride(2) : g.dsn;
                g.csl                 = C.present ? c.L.mergedStride(2) : g.dsl;
                auto unit
                    = [](int64_t stride, int64_t extent) { return stride == 1 && extent > 1; };
                // the kernel reads C with D's data type
                bool direct
                    = realOut && g.dsm >= 0 && g.dsn >= 0 && g.dsl >= 0 && g.csm >= 0 && g.csn >= 0
                      && g.csl >= 0
                      && (!C.present || (C.unaryOp == HIPTENSOR_OP_IDENTITY && C.type == D.type))
                      && unit(g.dsm, M) == unit(g.csm, M) && unit(g.dsn, N) == unit(g.csn, N);
                uint64_t const es0  = cplx ? 16 : 8;
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

            // The f64 GEMM kernels for a layout in the static order: the rule's class (with K
            // blocks for a long K when the pack has them), the default configuration, then the
            // other classes.
            std::vector<PackKernel const*> gett64Kernels(Pack const&         pack,
                                                         Gett64Layout const& g,
                                                         bool                cplx,
                                                         std::string const&  ruleClass,
                                                         bool                longK)
            {
                std::vector<PackKernel const*> out;
                auto                           matches = [&](PackKernel const& k) {
                    return k.family == kGett64 && k.attr("cplx") == (cplx ? 1 : 0)
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
            // data type on the F64 route). `c` is ordered by D.
            hiptensorStatus_t planGett64(Problem const&      p,
                                         Pack const&         pack,
                                         Classified const&   c,
                                         bool                cplx,
                                         Gett64Layout const& g,
                                         PackKernel const&   kernel,
                                         uint64_t            workspaceLimit,
                                         Program&            program,
                                         CandidateFacts&     facts,
                                         std::string&        reason)
            {
                int64_t const  M = c.M.size(), N = c.N.size(), K = c.K.size(), L = c.L.size();
                int64_t const  Kp     = roundUp(K, 2); // even K: one layout class
                bool const     direct = g.direct;
                uint64_t const es0    = cplx ? 16 : 8;
                auto           span   = [&](int64_t sm, int64_t sn) {
                    return ((M - 1) * sm + (N - 1) * sn + 1) * int64_t(es0);
                };
                uint64_t const es        = cplx ? 16 : 8;
                std::string    pt        = "f64";
                int64_t const  spanLimit = kernel.attr("limit_span_bytes", kGett64SpanLimit);
                if(L > kernel.attr("limit_batch", kGridYZMax) || M * Kp > spanLimit / int64_t(es)
                   || N * Kp > spanLimit / int64_t(es)
                   || (direct
                       && (span(g.dsm, g.dsn) > spanLimit || span(g.csm, g.csn) > spanLimit)))
                {
                    reason = "operands past the 32-bit offsets of the f64 GEMM";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                uint64_t const aBytes = uint64_t(L * M * Kp) * es;
                uint64_t const bBytes = uint64_t(L * N * Kp) * es;
                uint64_t const offA   = 0;
                uint64_t const offB   = alignUp(aBytes, kWorkspaceAlign);
                uint64_t const offT   = alignUp(offB + bBytes, kWorkspaceAlign);
                uint64_t const tBytes = direct ? 0 : uint64_t(L * M * N) * es;
                uint64_t const total  = alignUp(offT + tBytes, kWorkspaceAlign);
                if(total > workspaceLimit)
                {
                    reason = "the f64 route needs more workspace than the limit";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                if(!direct && M * N > spanLimit / int64_t(es))
                {
                    reason = "the f64 GEMM result is past its 32-bit offsets";
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
                LaunchStep step;
                step.kernel = &kernel;
                ArgWriter w(step);
                bool      ok = w.pointer("A", BufferRole::Workspace, offA)
                          && w.pointer("B", BufferRole::Workspace, offB);
                if(direct)
                {
                    ok = ok && w.pointer("D", BufferRole::Out) && w.set("d_sm", g.dsm)
                         && w.set("d_sn", g.dsn) && w.set("d_sl", g.dsl);
                    if(g.bilinear)
                    {
                        ok = ok && w.pointer("C", BufferRole::C) && w.set("c_sm", g.csm)
                             && w.set("c_sn", g.csn) && w.set("c_sl", g.csl);
                    }
                }
                else
                {
                    ok = ok && w.pointer("D", BufferRole::Workspace, offT) && w.set("d_sm", N)
                         && w.set("d_sn", 1) && w.set("d_sl", M * N);
                }
                auto alpha = direct ? ScalarRole::Alpha : ScalarRole::One;
                if(cplx)
                {
                    ok = ok && w.scalar("alpha_re", alpha, false)
                         && w.scalar("alpha_im", alpha, true);
                    if(g.bilinear)
                    {
                        ok = ok && w.scalar("beta_re", ScalarRole::Beta, false)
                             && w.scalar("beta_im", ScalarRole::Beta, true);
                    }
                }
                else
                {
                    ok = ok && w.scalar("alpha", alpha);
                    if(g.bilinear)
                    {
                        ok = ok && w.scalar("beta", ScalarRole::Beta);
                    }
                }
                ok = ok && w.set("M", M) && w.set("N", N) && w.set("K", Kp) && w.set("L", L)
                     && w.set("a_sm", Kp) && w.set("a_sk", 1) && w.set("b_sn", Kp)
                     && w.set("b_sk", 1) && w.set("a_sl", M * Kp) && w.set("b_sl", N * Kp);
                if(!ok)
                {
                    reason = "f64 GEMM arguments do not fit the kernel arguments";
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
                int64_t const tm = kernel.attr("tile_m"), tn = kernel.attr("tile_n");
                step.grid  = dim3(static_cast<uint32_t>(ceilDiv(M, tm) * ceilDiv(N, tn)),
                                 static_cast<uint32_t>(L),
                                 1);
                step.block = dim3(static_cast<uint32_t>(kernel.attr("block", 256)), 1, 1);
                if(auto s = finishStep(step, w, reason); s != HIPTENSOR_STATUS_SUCCESS)
                {
                    return s;
                }
                int64_t const ctas = int64_t(step.grid.x) * step.grid.y * step.grid.z;
                prog.description   = kernel.name;
                prog.steps.push_back(std::move(step));
                if(!direct)
                {
                    status = epilogue(pack, p, c, "f64", cplx, N, M * N, offT, false, prog, reason);
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
                facts.route          = Route::Gett64;
                facts.body           = gett64ClassId(kernel.attrStr("cls"));
                facts.f64Route       = dtypeName(p.t[0].type) != "f64";
                facts.direct         = direct;
                facts.kBlock         = kernel.attr("kblock", 0) > 0;
                facts.tileM          = tm;
                facts.tileN          = tn;
                facts.tileK          = kernel.attr("tile_k", 0);
                facts.block          = kernel.attr("block", 256);
                facts.ctas           = ctas;
                facts.paddedVolume   = M * N * Kp * L;
                facts.main           = &kernel;
                uint64_t const outIo = uint64_t(M * N * L) * esOut * (p.t[2].present ? 2 : 1);
                facts.bytes = uint64_t(M * K * L) * esA + uint64_t(N * K * L) * esB + 2 * (aBytes + bBytes)
                              + 2 * tBytes + outIo;
                program = std::move(prog);
                return HIPTENSOR_STATUS_SUCCESS;
            }
            uint64_t fnv1a64(std::string const& text)
            {
                uint64_t hash = 0xCBF29CE484222325ull;
                for(unsigned char c : text)
                {
                    hash = (hash ^ c) * 0x100000001B3ull;
                }
                return hash;
            }

            // The canonical key of a contraction for rocKE's tuned GETT route table, as
            // make_rocke_pack.gett_problem_key writes it: D, A, B with extent-1 modes dropped,
            // modes ordered by stride and named by first appearance.
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

            // A contraction of rocKE's tuned GETT route table: the launches of its record
            // (see make_rocke_pack.build_gett_plan). False when the problem has no record or the
            // record does not apply.
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
                // or bf16), x6 (operands as f32, split into three bf16 parts) or cf32
                bool        gemm = false;
                std::string route, storage;
                // the f64 strided GEMM (operands converted to f64 exactly)
                bool f64 = false;
                // why a route is missing
                std::string why;
            };

            // The compute type sets the precision of the products: 64F / C64F the f64 route
            // only; 16F / 16BF every operand rounded once (nearest even) to that type, so the
            // h16 GEMM route, and the f64 route only when the operands already have that type
            // (it would skip the rounding); 32F / C32F f32 products: operands of f64 data take
            // the f64 route (more precise, and the only route that reads them without a first
            // rounding), f16 x f16 or bf16 x bf16 without an operand operator the h16 route
            // (exact products), the other real operands the x6 route (their f32 values, every
            // f16 / bf16 value is one), complex f32 the cf32 route; the f64 route as well.
            ContractionRoutes contractionRoutes(Problem const& p)
            {
                ContractionRoutes r;
                auto const&       A    = p.t[0];
                auto const&       B    = p.t[1];
                bool const        cplx = isComplex(p.t[3].type);
                auto const        f64Data
                    = [](hiptensorDataType_t t) { return t == HIPTENSOR_R_64F || t == HIPTENSOR_C_64F; };
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
                    r.gemm = true;
                    if(cplx)
                    {
                        r.route   = "cf32";
                        r.storage = "bf16";
                        return r;
                    }
                    bool const ops
                        = A.unaryOp != HIPTENSOR_OP_IDENTITY || B.unaryOp != HIPTENSOR_OP_IDENTITY;
                    if(A.type == B.type && !ops
                       && (A.type == HIPTENSOR_R_16F || A.type == HIPTENSOR_R_16BF))
                    {
                        r.route   = "h16";
                        r.storage = A.type == HIPTENSOR_R_16F ? "fp16" : "bf16";
                        return r;
                    }
                    // the operators of 16-bit operands run in f32 and their results stay f32
                    // (no rounding back to the 16-bit type), as do mixed 16-bit operands
                    r.route   = "x6";
                    r.storage = "bf16";
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
                auto const routes = contractionRoutes(problem);
                if(!routes.gemm && !routes.f64)
                {
                    reason = routes.why;
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                }
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
                    candidate.key        = std::move(key);
                    candidate.program    = std::move(program);
                    candidate.facts      = facts;
                    candidate.staticRank = static_cast<int>(out.size());
                    candidate.tuned      = facts.route == Route::TunedGett;
                    out.push_back(std::move(candidate));
                };

                // rocKE's tuned GETT route for the problems it measured
                if(!options.f64Only && !options.disableTuned)
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
                        add("tuned:" + hex.substr(hex.find(':') + 1), std::move(program), facts);
                    }
                    else if(!why.empty())
                    {
                        note(why);
                    }
                }

                // the GEMM route of the data and compute types
                if(!options.f64Only && routes.gemm)
                {
                    std::string const& route   = routes.route;
                    std::string const& storage = routes.storage;
                    int64_t const      nk      = c.N.size() * (cplx ? 2 : 1);
                    int const     cus = std::max(1, problem.device.cuCount);
                    for(auto const& body : gemmBodies(pack, c.M.size(), nk, cus, cplx))
                    {
                        Program        program;
                        CandidateFacts facts;
                        std::string    why;
                        auto           status = planGemmBody(problem,
                                                   pack,
                                                   c,
                                                   route,
                                                   storage,
                                                   body,
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
                            note(why);
                        }
                        else
                        {
                            reason = why;
                            return status;
                        }
                    }
                }

                // the f64 strided GEMM: the route of f64 / complex f64 data, and the F64 route of
                // the other types (operands converted to f64, the result rounded once)
                if(!routes.f64)
                {
                    note(routes.why);
                }
                else
                {
                    Classified cd = c;
                    cd.M.orderBy(3);
                    cd.N.orderBy(3);
                    cd.L.orderBy(3);
                    auto const    layout = gett64Layout(problem, cd, cplx);
                    int64_t const Kp     = roundUp(cd.K.size(), 2);
                    int const     cus    = std::max(1, problem.device.cuCount);
                    int64_t const kbMin
                        = static_cast<int64_t>(pack.routeDouble("gett64.k_block_min_k", 4096));
                    auto const cls = gett64Class(
                        pack.arch(), cplx, cd.M.size(), cd.N.size(), Kp, cd.L.size(), cus);
                    auto const kernels = gett64Kernels(pack, layout, cplx, cls, Kp >= kbMin);
                    if(kernels.empty())
                    {
                        note("the pack has no f64 GEMM kernel for this layout class");
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
                                                 why);
                        if(status == HIPTENSOR_STATUS_SUCCESS)
                        {
                            add((facts.f64Route ? "f64:" : "gett64:") + kernel->name,
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
                    stride = extent > 0 && stride > INT64_MAX / extent ? INT64_MAX : stride * extent;
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
                    for(auto step : from.steps)
                    {
                        for(auto& ptr : step.pointers)
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
                            for(auto& s : step.scalars)
                            {
                                s.role = s.role == ScalarRole::Alpha ? ScalarRole::One
                                         : s.role == ScalarRole::One ? ScalarRole::One
                                                                     : ScalarRole::Zero;
                            }
                        }
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
                auto const     T  = intermediateOf(problem);
                uint64_t const es = elementBytes(tType, isComplex(E.type));
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
                if(auto status = enumerateContraction(first, pack, options, subLimit, c1, why);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    reason = "step 1 (T = A * B) of the trinary contraction: " + why;
                    return status;
                }
                if(auto status = enumerateContraction(second, pack, options, subLimit, c2, why);
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

        hiptensorStatus_t enumerateCandidates(Problem const&          problem,
                                              Pack const&             pack,
                                              CandidateOptions const& options,
                                              uint64_t                workspaceLimit,
                                              std::vector<Candidate>& out,
                                              std::string&            reason)
        {
            out.clear();
            switch(problem.kind)
            {
            case OpKind::Permute:
            case OpKind::ElementwiseBinary:
            case OpKind::ElementwiseTrinary:
            {
                Candidate candidate;
                auto      status
                    = planElementwise(problem, pack, candidate.program, candidate.facts, reason);
                if(status == HIPTENSOR_STATUS_SUCCESS)
                {
                    candidate.key = "ew:" + candidate.program.description;
                    out.push_back(std::move(candidate));
                }
                return status;
            }
            case OpKind::Reduction:
            {
                // The rule's mode and split first, then fewer and more partial ranges, then the
                // other mode. Two choices that plan the same launches are one candidate.
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

        hiptensorStatus_t loadFunctions(Program& program)
        {
            if(hipGetDevice(&program.device) != hipSuccess)
            {
                return HIPTENSOR_STATUS_HIP_ERROR;
            }
            for(auto& step : program.steps)
            {
                if(auto status = program.pack->function(*step.kernel, step.function);
                   status != HIPTENSOR_STATUS_SUCCESS)
                {
                    return status;
                }
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
    } // namespace rockebackend
} // namespace hiptensor
