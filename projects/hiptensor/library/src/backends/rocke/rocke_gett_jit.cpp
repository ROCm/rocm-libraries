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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <vector>

#include "rocke/gemm_dispatch.h"
#include "rocke/instance_tensor_gett.h"
#include "rocke/rocke_build_id.h"
#include "rocke/tensor_gett_planner.h"

#include "logger.hpp"
#include "platform.hpp"
#include "rocke_gett_jit.hpp"
#include "rocke_jit.hpp"

namespace hiptensor
{
    namespace rockebackend
    {
        namespace
        {
            namespace gd = rocke::gemm_dispatch;
            namespace gp = rocke::gett::planner;
            namespace rg = rocke::gett;

            // The pack steps' outputs sit in the workspace at multiples of this many bytes, in
            // step order (as the launch records of the pack's route tables).
            constexpr uint64_t kPackOffsetAlign = 256;
            // The selection models of the planner next to a pack, and their index
            // ("<sha256 hex> <file name>" per line, written by the pack builder).
            constexpr char kModelDir[]   = "gemm_models";
            constexpr char kModelIndex[] = "index.txt";
            // Planning decisions kept per process (all of them are dropped when one more would
            // exceed this; the planner then decides again).
            constexpr size_t kMaxPlans = 4096;

            void logGett(std::string const& msg)
            {
                Logger::instance()->logHeuristics("hiptensorCreatePlan", msg.c_str());
            }

            std::string hexOf(std::array<uint8_t, 32> const& d)
            {
                static constexpr char digits[] = "0123456789abcdef";
                std::string           out;
                for(auto b : d)
                {
                    out += digits[b >> 4];
                    out += digits[b & 15];
                }
                return out;
            }

            bool readWhole(std::string const& path, std::string& out)
            {
                std::ifstream f(path, std::ios::binary);
                if(!f)
                {
                    return false;
                }
                std::ostringstream ss;
                ss << f.rdbuf();
                out = ss.str();
                return f.good() || f.eof();
            }

            std::string directoryOf(std::string const& path)
            {
                auto slash = path.find_last_of("/\\");
                return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
            }

            // ------------------------------------------------------------ selection models

            // The model files of a pack's gemm_models directory: only the files its index
            // names, each checked against its SHA-256.
            struct ModelReader
            {
                std::string                        dir;
                std::map<std::string, std::string> digests; // file -> sha256 hex
                std::string                        damaged; // first file that failed its check
            };

            bool readModelFile(void* ctx, std::string const& name, std::string* text)
            {
                auto* r  = static_cast<ModelReader*>(ctx);
                auto  it = r->digests.find(name);
                if(it == r->digests.end())
                {
                    // the pack build's loader did not find it either
                    return false;
                }
                std::string body;
                if(!readWhole(r->dir + "/" + name, body)
                   || hexOf(sha256(reinterpret_cast<uint8_t const*>(body.data()), body.size()))
                          != it->second)
                {
                    if(r->damaged.empty())
                    {
                        r->damaged = name;
                    }
                    return false;
                }
                *text = std::move(body);
                return true;
            }

            struct LoadedModels
            {
                std::once_flag              once;
                std::unique_ptr<gd::Models> models;
                std::string                 reason;
                std::string                 digest; // sha256 hex of the index (models loaded)
            };

            // The planner's selection models of a pack's target, loaded once per process. The
            // load (a text parse of the model files) runs outside the lock of the table of
            // packs: only the plans of the same pack wait for it.
            LoadedModels const& loadedModelsFor(Pack const& pack)
            {
                static std::mutex                                           mutex;
                static std::map<std::string, std::shared_ptr<LoadedModels>> loaded;
                std::shared_ptr<LoadedModels>                               slot;
                {
                    std::scoped_lock lock(mutex);
                    auto&            s = loaded[pack.path() + "|" + pack.arch()];
                    if(s == nullptr)
                    {
                        s = std::make_shared<LoadedModels>();
                    }
                    slot = s;
                }
                std::call_once(slot->once, [&pack, &lm = *slot] {
                    using Clock    = std::chrono::steady_clock;
                    auto const  t0 = Clock::now();
                    ModelReader r;
                    r.dir = directoryOf(pack.path()) + "/" + kModelDir;
                    std::string index;
                    if(!readWhole(r.dir + "/" + kModelIndex, index))
                    {
                        lm.reason = "the pack has no GEMM selection models (" + r.dir + "/"
                                    + kModelIndex + ")";
                    }
                    else
                    {
                        std::istringstream lines(index);
                        std::string        line;
                        while(std::getline(lines, line))
                        {
                            auto sp = line.find(' ');
                            if(sp == 64 && line.size() > 65
                               && line.find_first_of("/\\") == std::string::npos)
                            {
                                r.digests[line.substr(65)] = line.substr(0, 64);
                            }
                        }
                        auto        models = std::make_unique<gd::Models>();
                        std::string why;
                        if(r.digests.empty())
                        {
                            lm.reason = "the GEMM selection model index " + r.dir + "/"
                                        + kModelIndex + " names no file";
                        }
                        else if(!models->load(readModelFile, &r, pack.arch(), &why))
                        {
                            lm.reason = "the GEMM selection models do not load: " + why;
                        }
                        else if(!r.damaged.empty())
                        {
                            lm.reason = "the GEMM selection model " + r.dir + "/" + r.damaged
                                        + " does not match its digest";
                        }
                        else
                        {
                            lm.models = std::move(models);
                            lm.digest = hexOf(sha256(reinterpret_cast<uint8_t const*>(index.data()),
                                                     index.size()));
                        }
                    }
                    double const ms
                        = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    char msg[96];
                    snprintf(msg, sizeof(msg), " (%.1f ms)", ms);
                    logGett(lm.models != nullptr
                                ? "rocKE GETT JIT: loaded the GEMM selection models of "
                                      + pack.arch() + " from " + r.dir + msg
                                : "rocKE GETT JIT: " + lm.reason);
                });
                return *slot;
            }

            gd::Models const* modelsFor(Pack const& pack, std::string& reason)
            {
                auto const& lm = loadedModelsFor(pack);
                reason         = lm.reason;
                return lm.models.get();
            }

            // ------------------------------------------------------------ the request

            // The planner's view of a contraction: mode lists row-major (last mode fastest) by
            // decreasing stride, extent-1 modes dropped, modes named x0, x1, ... by first
            // appearance in D, A, B (each in increasing stride order: the canonical key of the
            // pack's route tables), strides given only where a tensor is not packed in that
            // order. `key` names the request (the per-process planning memo).
            bool toPlannerProblem(backend::Problem const& p,
                                  gp::Problem&            out,
                                  std::string&            key,
                                  std::string&            reason)
            {
                auto const& A  = p.t[0];
                auto const& B  = p.t[1];
                auto const& C  = p.t[2];
                auto const& D  = p.t[3];
                auto const  dt = D.type == HIPTENSOR_R_16F    ? std::string("f16")
                                 : D.type == HIPTENSOR_R_16BF ? std::string("bf16")
                                                              : std::string();
                if(dt.empty() || p.compute != HIPTENSOR_COMPUTE_DESC_32F || A.type != D.type
                   || B.type != D.type || (C.present && C.type != D.type))
                {
                    reason = "the GETT JIT serves f16 / bf16 data of one type with the 32F "
                             "compute type";
                    return false;
                }
                if(A.unaryOp != HIPTENSOR_OP_IDENTITY || B.unaryOp != HIPTENSOR_OP_IDENTITY
                   || (C.present && C.unaryOp != HIPTENSOR_OP_IDENTITY))
                {
                    reason = "the GETT JIT has no operand operators";
                    return false;
                }
                if(C.present && (C.strides != D.strides || C.modes != D.modes))
                {
                    reason = "the GETT JIT reads C with D's layout";
                    return false;
                }
                std::map<int32_t, std::string> names;
                key                               = dt + (C.present ? "|bilinear" : "|scale");
                std::vector<std::string>* lists[] = {&out.modes_d, &out.modes_a, &out.modes_b};
                std::map<std::string, int64_t>* strides[]
                    = {&out.strides_d, &out.strides_a, &out.strides_b};
                bool*      has[]    = {&out.has_strides_d, &out.has_strides_a, &out.has_strides_b};
                int const  slots[]  = {3, 0, 1};
                char const labels[] = {'D', 'A', 'B'};
                for(int t = 0; t < 3; t++)
                {
                    auto const&                                        x = p.t[slots[t]];
                    std::vector<std::tuple<int64_t, int32_t, int64_t>> ms; // stride, mode, extent
                    for(int i = 0; i < x.numModes(); i++)
                    {
                        if(x.extents[i] < 1)
                        {
                            reason = "an extent below 1";
                            return false;
                        }
                        if(x.extents[i] == 1)
                        {
                            continue;
                        }
                        if(x.strides[i] < 1)
                        {
                            reason = "a stride below 1 on a mode of extent above 1";
                            return false;
                        }
                        ms.emplace_back(x.strides[i], x.modes[i], x.extents[i]);
                    }
                    std::sort(ms.begin(), ms.end());
                    for(size_t i = 1; i < ms.size(); i++)
                    {
                        if(std::get<0>(ms[i]) == std::get<0>(ms[i - 1]))
                        {
                            reason = "two modes of a tensor share a stride";
                            return false;
                        }
                    }
                    key += std::string("|") + labels[t] + ":";
                    for(size_t i = 0; i < ms.size(); i++)
                    {
                        auto [stride, mode, extent] = ms[i];
                        auto found                  = names.find(mode);
                        if(found == names.end())
                        {
                            found = names.emplace(mode, "x" + std::to_string(names.size())).first;
                        }
                        auto const& name = found->second;
                        if(auto e = out.extents.find(name);
                           e != out.extents.end() && e->second != extent)
                        {
                            reason = "a mode with two extents";
                            return false;
                        }
                        out.extents[name] = extent;
                        key += (i ? "," : "") + name + ":" + std::to_string(extent) + ":"
                               + std::to_string(stride);
                    }
                    // row-major: decreasing stride
                    int64_t packed   = 1;
                    bool    isPacked = true;
                    for(auto it = ms.begin(); it != ms.end(); ++it)
                    {
                        auto const& name = names.at(std::get<1>(*it));
                        lists[t]->insert(lists[t]->begin(), name);
                        (*strides[t])[name] = std::get<0>(*it);
                        isPacked            = isPacked && std::get<0>(*it) == packed;
                        packed *= std::get<2>(*it);
                    }
                    *has[t] = !isPacked;
                    if(isPacked)
                    {
                        strides[t]->clear();
                    }
                }
                out.dtype    = dt;
                out.bilinear = C.present;
                return true;
            }

            // A pack step of a planned route.
            struct PackLaunch
            {
                rg::TensorPackSpec spec;
                std::string        operand; // the user operand copied: "A" or "B"
                std::string        name;
                std::string        specKey;
                uint64_t           offset  = 0; // of its output in the workspace
                int                grid[3] = {1, 1, 1};
                int                block   = 256;
            };

            std::string specKeyOf(std::string const& arch,
                                  std::string const& name,
                                  std::string const& jsonKey)
            {
                return std::string(kGettJitSpecKeyPrefix) + "|" + arch + "|" + name + "|"
                       + rocke_build_id() + "|" + rg::sha256_hex(jsonKey);
            }
        } // namespace

        struct GettJitPlan
        {
            std::string             arch;
            std::vector<PackLaunch> packs;
            rg::TensorGettSpec      gett;
            std::string             gettName;
            std::string             gettSpecKey;
            int                     grid[3] = {1, 1, 1};
            int                     block   = 256;
            int64_t                 M = 0, N = 0, K = 0;
            bool                    bilinear = false;
            // per GETT operand param (A, B): the pack step it reads (>= 0) or the user
            // operand `user[i]` ("A" / "B")
            int             src[2] = {-1, -1};
            std::string     user[2];
            uint64_t        workspace = 0;
            GettJitPlanInfo info;
            // the launches in the format of the pack's GETT route records (logged per plan)
            std::string record;
        };

        namespace
        {
            // The plan of a planner route (the pack builder's gettPlanDescriptor of a table
            // entry, for any route). False with `reason` when the route uses a lever only rocKE's
            // Python engine builds, or is not a valid spec of the C++ builders.
            bool makePlan(gp::RoutePlan const& route,
                          std::string const&   arch,
                          GettJitPlan&         plan,
                          std::string&         reason)
            {
                plan.arch = arch;
                if(!gp::to_builder_spec(route.gett, &plan.gett, &reason))
                {
                    reason = "the planned GETT uses a lever of rocKE's Python engine only ("
                             + reason + ")";
                    return false;
                }
                if(!rg::is_valid_spec(plan.gett, arch.c_str(), &reason))
                {
                    reason = "the planned GETT is not a valid spec of the C++ builders: " + reason;
                    return false;
                }
                rg::Plan pl;
                if(!plan.gett.plan(&pl, &reason))
                {
                    return false;
                }
                plan.M        = pl.M;
                plan.N        = pl.N;
                plan.K        = pl.K;
                plan.bilinear = plan.gett.knobs.bilinear;
                plan.gettName = plan.gett.kernel_name();
                if(plan.gettName != route.gett.kernel_name())
                {
                    reason = "the C++ builders' GETT spec is not the planner's";
                    return false;
                }
                plan.gettSpecKey = specKeyOf(arch, plan.gettName, plan.gett.json_key());
                rg::tensor_gett_grid(plan.gett, plan.grid);
                plan.block = rg::tensor_gett_block(plan.gett);
                if(plan.block != route.block || plan.grid[0] != route.grid[0]
                   || plan.grid[1] != route.grid[1] || plan.grid[2] != route.grid[2])
                {
                    reason = "the GETT launch shape is not the planner's";
                    return false;
                }
                uint64_t off = 0;
                for(auto const& s : route.packs)
                {
                    std::string why;
                    if(!rg::pack_is_valid_spec(s.spec, arch.c_str(), &why))
                    {
                        reason = "the planned pack step is not valid: " + why;
                        return false;
                    }
                    PackLaunch pk;
                    pk.spec    = s.spec;
                    pk.operand = s.operand;
                    pk.name    = s.spec.kernel_name();
                    pk.specKey = specKeyOf(arch, pk.name, s.spec.json_key());
                    pk.offset  = off;
                    std::copy(s.grid, s.grid + 3, pk.grid);
                    pk.block = s.block;
                    plan.packs.push_back(std::move(pk));
                    off += (uint64_t(s.workspace_bytes) + kPackOffsetAlign - 1) / kPackOffsetAlign
                           * kPackOffsetAlign;
                }
                plan.workspace = off;
                rg::TablePlan tp;
                tp.packs = route.packs;
                tp.gett  = plan.gett;
                rg::gett_operand_sources(tp, plan.src);
                auto ops     = rg::launch_operands(plan.gett);
                plan.user[0] = ops.first;
                plan.user[1] = ops.second;
                for(int i = 0; i < 2; i++)
                {
                    if(plan.src[i] < 0 && plan.user[i] != "A" && plan.user[i] != "B")
                    {
                        reason = "the GETT reads an operand that is not A or B";
                        return false;
                    }
                }
                // the record of the pack's GETT route tables (rocke_pack_builder.cpp
                // gettPlanDescriptor): per launch the kernel, grid, block and the bound arguments
                // (X / Y: a pack step's source operand and its copy at a workspace offset; the
                // GETT's operands, D (O), C, the GEMM view and the scales)
                auto launch = [](std::string const&                        kernel,
                                 int const*                                grid,
                                 int                                       block,
                                 std::map<std::string, std::string> const& binds) {
                    std::string out = "|k=" + kernel + ";g=" + std::to_string(grid[0]) + ","
                                      + std::to_string(grid[1]) + "," + std::to_string(grid[2])
                                      + ";b=" + std::to_string(block) + ";a=";
                    bool first = true;
                    for(auto const& [k, v] : binds)
                    {
                        out += (first ? "" : ",") + k + "=" + v;
                        first = false;
                    }
                    return out;
                };
                plan.record = "v1;ws=" + std::to_string(plan.workspace)
                              + ";steps=" + std::to_string(plan.packs.size() + 1);
                for(auto const& pk : plan.packs)
                {
                    plan.record += launch(
                        pk.name,
                        pk.grid,
                        pk.block,
                        {{"X", pk.operand + "+0"}, {"Y", "W+" + std::to_string(pk.offset)}});
                }
                std::map<std::string, std::string> binds = {
                    {"C", "O+0"},
                    {"M", "#" + std::to_string(plan.M)},
                    {"N", "#" + std::to_string(plan.N)},
                    {"K", "#" + std::to_string(plan.K)},
                    {"alpha", "@a"},
                };
                if(plan.bilinear)
                {
                    binds["Cin"]  = "C+0";
                    binds["beta"] = "@b";
                }
                for(int i = 0; i < 2; i++)
                {
                    binds[i == 0 ? "A" : "B"]
                        = plan.src[i] >= 0
                              ? "W+" + std::to_string(plan.packs[size_t(plan.src[i])].offset)
                              : plan.user[i] + "+0";
                }
                plan.record += launch(plan.gettName, plan.grid, plan.block, binds);

                auto& info          = plan.info;
                info.workspaceBytes = plan.workspace;
                info.launches       = static_cast<int>(plan.packs.size()) + 1;
                info.M              = plan.M;
                info.N              = plan.N;
                info.K              = plan.K;
                info.block          = plan.block;
                info.ctas   = int64_t(plan.grid[0]) * int64_t(plan.grid[1]) * int64_t(plan.grid[2]);
                auto body   = route.gett.body();
                info.tileM  = static_cast<int>(body.tile.tile_m);
                info.tileN  = static_cast<int>(body.tile.tile_n);
                info.tileK  = static_cast<int>(body.tile.tile_k);
                info.route  = route.kind + "/" + route.candidate;
                info.kernel = plan.gettName;
                bool const packed = !route.packs.empty();
                info.kind         = route.candidate == "tensor_gett_t32s3" ? 4
                                    : route.candidate == "tensor_gett_kc"  ? (packed ? 3 : 5)
                                    : packed                               ? 2
                                                                           : 1;
                // bytes moved: each pack step reads its operand and writes the copy (its
                // workspace bytes each way), the GETT reads both operands of its GEMM view, writes
                // D and, bilinear, reads C (16-bit elements)
                uint64_t const es = 2;
                info.bytes        = es * uint64_t(plan.M + plan.N) * uint64_t(plan.K)
                             + es * uint64_t(plan.M) * uint64_t(plan.N) * (plan.bilinear ? 2 : 1);
                for(auto const& s : route.packs)
                {
                    info.bytes += 2 * uint64_t(s.workspace_bytes);
                }
                return true;
            }

            struct Planned
            {
                std::shared_ptr<GettJitPlan const> plan;
                std::string                        reason;
            };

            BufferRole operandRole(std::string const& operand)
            {
                return operand == "A" ? BufferRole::A : BufferRole::B;
            }
        } // namespace

        bool planGettJit(backend::Problem const&             problem,
                         Pack const&                         pack,
                         uint64_t                            workspaceLimit,
                         std::shared_ptr<GettJitPlan const>& out,
                         GettJitPlanInfo&                    info,
                         std::string&                        reason)
        {
            out.reset();
            if(problem.kind != backend::OpKind::Contraction)
            {
                reason = "not a contraction";
                return false;
            }
            if(auto mode = getEnvironmentVariable("HIPTENSOR_ROCKE_JIT");
               mode && (*mode == "0" || *mode == "off"))
            {
                reason = "JIT disabled by HIPTENSOR_ROCKE_JIT";
                return false;
            }
            auto const& arch = problem.device.baseArch;
            if(!gp::plans_arch(arch))
            {
                reason = "rocKE's tensor GETT planner does not serve " + arch;
                return false;
            }
            gp::Problem request;
            std::string key;
            if(!toPlannerProblem(problem, request, key, reason))
            {
                return false;
            }
            int64_t const cus = std::max<int64_t>(1, problem.device.cuCount);
            key += "|" + arch + "|" + std::to_string(cus) + "|" + pack.path();

            static std::mutex                     mutex;
            static std::map<std::string, Planned> memo;
            Planned                               planned;
            bool                                  hit = false;
            {
                std::scoped_lock lock(mutex);
                if(auto it = memo.find(key); it != memo.end())
                {
                    planned = it->second;
                    hit     = true;
                }
            }
            if(!hit)
            {
                gd::Models const* models = modelsFor(pack, planned.reason);
                if(models != nullptr)
                {
                    gd::Context ctx;
                    ctx.cus    = cus;
                    ctx.models = models;
                    gp::RoutePlan route;
                    std::string   why;
                    if(!gp::plan_tensor_gett(request, arch, ctx, &route, &why))
                    {
                        planned.reason = "rocKE's tensor GETT planner has no route: " + why;
                    }
                    else
                    {
                        auto plan = std::make_shared<GettJitPlan>();
                        if(makePlan(route, arch, *plan, why))
                        {
                            // the planner's predicted time of its route (a selection feature)
                            double us = 0.0;
                            if(gp::plan_cost(route, arch, ctx, &us) && std::isfinite(us)
                               && us > 0.0)
                            {
                                plan->info.predictedNs = std::llround(
                                    std::min(us * 1000.0, 9.0e15)); // well inside int64
                            }
                            planned.plan = std::move(plan);
                        }
                        else
                        {
                            planned.reason = why;
                        }
                    }
                }
                std::scoped_lock lock(mutex);
                if(memo.size() >= kMaxPlans)
                {
                    memo.clear();
                }
                memo.emplace(key, planned);
            }
            if(planned.plan == nullptr)
            {
                reason = planned.reason;
                return false;
            }
            logGett("rocKE GETT JIT plan " + planned.plan->info.route + ": "
                    + planned.plan->record);
            if(planned.plan->workspace > workspaceLimit)
            {
                reason = "the planned GETT route needs more workspace than the limit";
                return false;
            }
            out  = planned.plan;
            info = planned.plan->info;
            return true;
        }

        std::string gettJitModelsDigest(Pack const& pack)
        {
            return loadedModelsFor(pack).digest;
        }

        bool buildGettJitProgram(GettJitPlan const&        plan,
                                 backend::DeviceKey const& device,
                                 double                    budgetMs,
                                 Program&                  program,
                                 GettJitReport&            report,
                                 std::string&              reason)
        {
            using Clock   = std::chrono::steady_clock;
            auto const t0 = Clock::now();
            if(device.baseArch != plan.arch)
            {
                reason = "the route was planned for " + plan.arch;
                return false;
            }
            auto allow = [&] {
                return budgetMs <= 0.0
                       || std::chrono::duration<double, std::milli>(Clock::now() - t0).count()
                              < budgetMs;
            };
            auto obtain = [&](JitKernelSource const& source, LaunchStep& step) {
                JitKernel k;
                if(!jitKernel(source, device, allow(), k, reason))
                {
                    reason = source.label + ": " + reason;
                    return false;
                }
                report.kernels++;
                report.compiled += k.compiled ? 1 : 0;
                report.memoryHits += k.compiled ? 0 : 1;
                report.fileHits += k.fileHit ? 1 : 0;
                step.kernel      = k.kernel;
                step.function    = k.function;
                step.deviceBound = true;
                if(k.kernel->maxWorkgroup != 0 && step.block.x > k.kernel->maxWorkgroup)
                {
                    reason = source.label + " takes smaller workgroups than its launch";
                    return false;
                }
                return true;
            };
            std::vector<LaunchStep> steps;
            for(auto const& pk : plan.packs)
            {
                LaunchStep step;
                step.grid  = dim3(static_cast<uint32_t>(pk.grid[0]),
                                 static_cast<uint32_t>(pk.grid[1]),
                                 static_cast<uint32_t>(pk.grid[2]));
                step.block = dim3(static_cast<uint32_t>(pk.block), 1, 1);
                JitKernelSource source;
                source.specKey = pk.specKey;
                source.family  = "gett";
                source.label   = "gett:" + pk.name;
                source.build   = [&pk, &plan](rocke_ir_builder_t* b) {
                    return rg::build_tensor_pack(b, pk.spec, plan.arch.c_str());
                };
                if(!obtain(source, step))
                {
                    return false;
                }
                ArgWriter   w(step);
                std::string missing;
                if(!w.pointer("X", operandRole(pk.operand))
                   || !w.pointer("Y", BufferRole::Workspace, pk.offset) || !w.complete(missing))
                {
                    reason = "the tensor pack kernel " + pk.name + " does not take its arguments";
                    return false;
                }
                steps.push_back(std::move(step));
            }
            LaunchStep step;
            step.grid  = dim3(static_cast<uint32_t>(plan.grid[0]),
                             static_cast<uint32_t>(plan.grid[1]),
                             static_cast<uint32_t>(plan.grid[2]));
            step.block = dim3(static_cast<uint32_t>(plan.block), 1, 1);
            JitKernelSource source;
            source.specKey = plan.gettSpecKey;
            source.family  = "gett";
            source.label   = "gett:" + plan.gettName;
            source.build   = [&plan](rocke_ir_builder_t* b) {
                return rg::build_tensor_gett(b, plan.gett, plan.arch.c_str());
            };
            if(!obtain(source, step))
            {
                return false;
            }
            ArgWriter  w(step);
            bool       ok         = true;
            char const operands[] = {'A', 'B'};
            for(int i = 0; i < 2; i++)
            {
                std::string const param(1, operands[i]);
                ok = ok
                     && (plan.src[i] >= 0 ? w.pointer(param,
                                                      BufferRole::Workspace,
                                                      plan.packs[size_t(plan.src[i])].offset)
                                          : w.pointer(param, operandRole(plan.user[i])));
            }
            ok = ok && w.pointer("C", BufferRole::Out) && w.set("M", plan.M) && w.set("N", plan.N)
                 && w.set("K", plan.K) && w.scalar("alpha", ScalarRole::Alpha);
            if(plan.bilinear)
            {
                ok = ok && w.pointer("Cin", BufferRole::C) && w.scalar("beta", ScalarRole::Beta);
            }
            std::string missing;
            if(!ok || !w.complete(missing))
            {
                reason = "the GETT kernel " + plan.gettName + " does not take its arguments"
                         + (missing.empty() ? std::string() : " (" + missing + ")");
                return false;
            }
            steps.push_back(std::move(step));
            for(auto& s : steps)
            {
                program.steps.push_back(std::move(s));
            }
            report.milliseconds
                = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return true;
        }
    } // namespace rockebackend
} // namespace hiptensor
