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

// rocke_pack_builder: builds the rocKE kernel pack of hipTensor's rocKE backend in C++.
//
// It builds the shape-generic kernels the backend launches with rocKE's C++ engine (rocke_core),
// lowers them to LLVM IR with the flavor of the linked COMGR library, compiles them with that
// library (rocke_comgr.cpp, the JIT's compile path), reads every kernel's argument layout from
// its code object metadata and writes one pack per GPU target:
//
//     <out>/<arch>/rocke.htpk          binary pack (read by rocke_runtime.cpp)
//     <out>/<arch>/rocke.index.json    the same index as JSON (tooling only)
//
// The pack is the one rocKE's Python reference writer (library/builders/hiptensor/
// make_rocke_pack.py of the rocKE tree, "the Python reference" below) writes for the same rocKE
// sources: same kernels, code objects, kernel records, routes and selection models; only the two
// header strings (compiler id, rocKE version) name this builder's toolchain unless --compiler-id /
// --rocke-version override them. rocKE's tests assert that equality
// (library/tests/test_hiptensor_pack_cross_check.py); hipTensor's build never runs Python.
//
// Every kernel family is built with rocKE's C++ builders (the descriptor tables of the Python
// reference): ew, reduce, rreduce (the tuned reduction classes of rocKE's reduction route
// table), ewt, ewv, dgett and rgett (rocke/instance_hiptensor_aot.h),
// gemm (the emulated-f32 GEMM bodies with their operand prologue: rocke_gemm_cf32_body,
// rocke_build_universal_gemm), gett (the launches of the tuned and AOT GETT route tables the
// engine carries: the tensor pack and GETT kernels of rocke/instance_tensor_gett.h) and gett64 /
// gett32 (rocke_build_gett_f64 through the gett_f64 spec policy, f64 and f32 words), and ewf
// (the flat elementwise kernel, rocke_build_htaot_ewf; gfx942 / gfx950).
//
// hipTensor's build compiles this tool against rocke_core and the COMGR library and runs it to
// write the packs it ships (cmake/Functions/hiptensorRockePack.cmake). The packs carry no JIT
// input: the backend's JIT builds its kernels with the same C++ builders (rocke_jit.cpp).
//
// --gemm-models DIR (rocKE's dispatch/gemm/selector_models) writes, next to the pack of every
// target rocKE's tensor GETT planner serves, <out>/<arch>/gemm_models: the selection model files
// the planner's loader reads for that target (rocke::gemm_dispatch::Models::load, exactly those)
// and index.txt ("<sha256> <file>" per file), which the backend's contraction JIT
// (rocke_gett_jit.cpp) loads and checks at run time. They are data, not part of the pack.
//
// --list writes, instead of the packs, each target's manifest <out>/<arch>.manifest: the
// kernels with their attributes and argument names, the routes (no selection models) and the
// JIT's kernel classes (rocke_ew_rows.hpp), without lowering or compiling anything (no COMGR
// needed). --check-manifests DIR compares the manifests with DIR/<arch>.manifest, the ones the
// Python reference writes (make_rocke_pack.py --list), which rocKE keeps current in
// library/builders/hiptensor/manifests (hipTensor's test rocke_pack_manifest_test).
//
// Usage:
//     rocke_pack_builder --arch gfx950 [--arch gfx942 ...] --out DIR [--jobs N]
//                        [--cache-dir DIR] [--compile-timeout S] [--num-cus N]
//                        [--select-models DIR] [--only REGEX] [--gemm-models DIR]
//                        [--compiler-id STR] [--rocke-version STR]
//     rocke_pack_builder --list --arch gfx950 [--arch ...] [--out DIR] [--check-manifests DIR]
//                        [--num-cus N]

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>

#if !defined(_WIN32)
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../rocke_comgr.hpp"
#include "../rocke_ew_rows.hpp"
#include "../rocke_rreduce_rule.hpp"
#include "rocke_pack_format.hpp"

#include "rocke/arch_num_cus.h"
#include "rocke/compile_options.h"
#include "rocke/gemm_dispatch.h"
#include "rocke/instance_gemm_cf32.h"
#include "rocke/instance_gemm_prologue.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/instance_gett_f64.h"
#include "rocke/instance_hiptensor_aot.h"
#include "rocke/instance_tensor_gett.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/rocke_build_id.h"
#include "rocke/tensor_gett_planner.h"

namespace fs = std::filesystem;

using hiptensor::rockepack::Attrs;
using hiptensor::rockepack::AttrValue;
using hiptensor::rockepack::KernelMd;
using hiptensor::rockepack::PackEntry;

namespace
{
    // ---------------------------------------------------------------------------------------------
    // Descriptor tables (the Python reference)
    // ---------------------------------------------------------------------------------------------

    // The arches with GEMM bodies (GEMM_ARCHES) and f64 GETT kernels (GETT64_CLASS_ARCHES).
    char const* const kGemmArches[]   = {"gfx950"};
    char const* const kGett64Arches[] = {"gfx90a", "gfx942", "gfx950"};
    // the native f32 strided GEMM (the Python reference's GETT32_CLASS_ARCHES)
    char const* const kGett32Arches[] = {"gfx942", "gfx950"};

    bool contains(char const* const* begin, char const* const* end, std::string const& arch)
    {
        return std::any_of(begin, end, [&](char const* a) { return arch == a; });
    }

    template <size_t N>
    bool contains(char const* const (&list)[N], std::string const& arch)
    {
        return contains(list, list + N, arch);
    }

    // Python repr(float(x)) of the route constants.
    std::string pyFloat(double v)
    {
        char buf[64];
        for(int prec = 1; prec <= 17; ++prec)
        {
            std::snprintf(buf, sizeof buf, "%.*g", prec, v);
            if(std::strtod(buf, nullptr) == v)
            {
                break;
            }
        }
        std::string s = buf;
        if(s.find_first_of(".en") == std::string::npos)
        {
            s += ".0";
        }
        return s;
    }

    // ---------------------------------------------------------------------------------------------
    // Kernels
    // ---------------------------------------------------------------------------------------------

    // One kernel to compile: name, pack attributes and its lowered LLVM IR.
    struct Kernel
    {
        std::string              name;
        Attrs                    attrs;
        std::vector<std::string> params; // the kernel's argument names (its IR parameters)
        std::string              ll;
        std::vector<std::string> options;
        std::string              key; // content address of the compile
        std::vector<uint8_t>     co;
        KernelMd                 md;
    };

    // One descriptor: the kernels it contributes (none when it does not apply) and its routes.
    struct Descriptor
    {
        std::string                        family;
        std::string                        label;
        bool                               applies = true;
        std::string                        reason; // why not
        std::vector<Kernel>                kernels;
        std::map<std::string, std::string> routes;
    };

    struct Context
    {
        std::string         arch;
        std::string         isa;
        rocke_llvm_flavor_t flavor = ROCKE_LLVM_FLAVOR_AUTO;
        std::string         compilerId; // part of the compile cache key
        bool                listOnly = false; // --list: build the kernels, lower nothing
    };

    // Lowers a built kernel and fills in its compile inputs. False with `why` set on failure.
    bool lower(Context const& ctx, rocke_kernel_def_t const* kd, Kernel& k, std::string& why)
    {
        if(rocke_attr_get_bool(&kd->attrs, ROCKE_COMGR_FRESH_PROCESS_ATTR, false))
        {
            why = k.name + " needs a fresh compile process (history-dependent codegen)";
            return false;
        }
        for(int i = 0; i < kd->num_params; ++i)
        {
            k.params.emplace_back(kd->params[i]->name);
        }
        if(ctx.listOnly)
        {
            return true;
        }
        char  err[ROCKE_ERR_MSG_CAP] = {0};
        char* text                   = nullptr;
        if(rocke_lower_kernel_to_llvm_ex(kd, ctx.flavor, ctx.arch.c_str(), &text, err, sizeof err)
               != ROCKE_OK
           || text == nullptr)
        {
            why = k.name + ": lowering failed: " + err;
            return false;
        }
        k.ll = text;
        std::free(text);
        char const* opts[64];
        int         n = rocke_comgr_options_for_kernel(kd, ctx.flavor, opts, 64, err, sizeof err);
        if(n < 0)
        {
            why = k.name + ": " + err;
            return false;
        }
        k.options.assign(opts, opts + n);
        std::string joined;
        for(size_t i = 0; i < k.options.size(); ++i)
        {
            joined += (i ? " " : "") + k.options[i];
        }
        auto h = hiptensor::rockepack::sha256(
            (ctx.compilerId + "\n" + ctx.isa + "\n" + joined + "\n" + k.ll).data(),
            ctx.compilerId.size() + ctx.isa.size() + joined.size() + k.ll.size() + 3);
        k.key = hiptensor::rockepack::hex(h.data(), h.size());
        return true;
    }

    // RAII owner of an IR builder.
    struct Builder
    {
        rocke_ir_builder_t b{};
        ~Builder()
        {
            rocke_ir_builder_free(&b);
        }
    };

    // ---- gett64 / gett32: rocke_build_gett_f64 through the gett_f64 spec policy
    // (the Python reference's _gett64_spec; f32: f32 / complex f32 words) ----

    bool gett64Spec(std::string const&     arch,
                    bool                   cplx,
                    std::string const&     cls,
                    char                   dclass,
                    bool                   bilinear,
                    bool                   kblock,
                    bool                   f32,
                    rocke_gett_f64_spec_t& out,
                    std::string&           why)
    {
        const int64_t M = 64, N = 64, K = 64, L = 2;
        // A [L][M][K] and B [L][N][K] packed with K contiguous (K padded to even by the pack
        // step); D (and C) [L][M][N] with M contiguous ("m"), N contiguous ("n") or neither ("x").
        int64_t aStrides[2] = {K, 1};
        int64_t bStrides[2] = {K, 1};
        int64_t dStrides[2] = {1, M};
        if(dclass == 'n')
        {
            dStrides[0] = N;
            dStrides[1] = 1;
        }
        else if(dclass == 'x')
        {
            dStrides[0] = 2;
            dStrides[1] = 2 * M;
        }
        int64_t                  batchStrides[3] = {M * K, N * K, 4 * M * N};
        rocke_gett_f64_problem_t prob            = rocke_gett_f64_gemm_problem(M,
                                                                    N,
                                                                    K,
                                                                    aStrides,
                                                                    bStrides,
                                                                    dStrides,
                                                                    nullptr,
                                                                    L,
                                                                    batchStrides,
                                                                    nullptr,
                                                                    cplx,
                                                                    false,
                                                                    false);
        prob.f32                                 = f32;
        rocke_gett_f64_knobs_t knobs{};
        if(cls != "default"
           && !rocke_gett_f64_tuned_config(
               arch.c_str(), cplx ? "complex" : "real", cls.c_str(), &knobs))
        {
            why = "no tuned configuration " + cls;
            return false;
        }
        knobs.set |= ROCKE_GETT_F64_KNOB_BILINEAR | ROCKE_GETT_F64_KNOB_RUNTIME_SIZES;
        knobs.bilinear             = bilinear;
        knobs.runtime_sizes        = true;
        rocke_gett_f64_spec_t spec = rocke_gett_f64_arch_spec(&prob, arch.c_str(), &knobs);
        spec.split_k               = 1;
        if(f32)
        {
            spec.name = "rocke_gett";
        }
        if(!rocke_gett_f64_runtime_template(&prob, &spec.problem))
        {
            why = "no runtime template";
            return false;
        }
        if(kblock)
        {
            rocke_gett_f64_spec_t kb = rocke_gett_f64_with_k_block(&spec, arch.c_str());
            if(kb.k_block == spec.k_block)
            {
                why = "no k_block variant";
                return false;
            }
            spec = kb;
        }
        char reason[512] = {0};
        if(!rocke_gett_f64_is_valid_spec(&spec, arch.c_str(), reason, sizeof reason))
        {
            why = reason;
            return false;
        }
        out = spec;
        return true;
    }

    void gett64Descriptors(Context const& ctx, std::vector<Descriptor>& out, bool f32 = false)
    {
        std::string const family = f32 ? "gett32" : "gett64";
        if(!(f32 ? contains(kGett32Arches, ctx.arch) : contains(kGett64Arches, ctx.arch)))
        {
            return;
        }
        for(bool cplx : {false, true})
        {
            char const*              kind = cplx ? "complex" : "real";
            std::vector<std::string> classes;
            for(int i = 0;; ++i)
            {
                char const* c = rocke_gett_f64_tuned_class_at(ctx.arch.c_str(), kind, i);
                if(c == nullptr)
                {
                    break;
                }
                classes.push_back(c);
            }
            std::sort(classes.begin(), classes.end());
            classes.push_back("default");
            for(auto const& cls : classes)
            {
                for(char dclass : {'m', 'n', 'x'})
                {
                    for(bool bilinear : {false, true})
                    {
                        for(bool kblock : {false, true})
                        {
                            Descriptor d;
                            d.family = family;
                            d.label  = std::string("cplx=") + (cplx ? "1" : "0") + " cls=" + cls
                                      + " dclass=" + dclass + " bilinear=" + (bilinear ? "1" : "0")
                                      + " kblock=" + (kblock ? "1" : "0");
                            rocke_gett_f64_spec_t spec;
                            if(!gett64Spec(ctx.arch,
                                           cplx,
                                           cls,
                                           dclass,
                                           bilinear,
                                           kblock,
                                           f32,
                                           spec,
                                           d.reason))
                            {
                                d.applies = false;
                                out.push_back(std::move(d));
                                continue;
                            }
                            Builder             b;
                            rocke_kernel_def_t* kd
                                = rocke_build_gett_f64(&b.b, &spec, ctx.arch.c_str());
                            if(kd == nullptr)
                            {
                                throw std::runtime_error(family + " " + d.label + ": "
                                                         + rocke_ir_builder_error(&b.b));
                            }
                            Kernel k;
                            k.name  = kd->name;
                            k.attrs = {
                                {"family", family},
                                {"abi",
                                 std::string(f32 ? "rocke-gett-f32/v1" : "rocke-gett-f64/v2")},
                                {"cplx", int64_t(cplx)},
                                {"cls", cls},
                                {"dclass", std::string(1, dclass)},
                                {"bilinear", int64_t(bilinear)},
                                {"kblock", int64_t(spec.k_block)},
                                {"tile_m", int64_t(spec.tile_m)},
                                {"tile_n", int64_t(spec.tile_n)},
                                {"tile_k", int64_t(spec.tile_k)},
                                {"block", int64_t(64 * spec.warps_m * spec.warps_n)},
                                // capability: bytes one buffer resource addresses, batch on grid y
                                {"limit_span_bytes", int64_t(ROCKE_GETT_F64_MAX_SPAN)},
                                {"limit_batch", int64_t(ROCKE_GETT_F64_MAX_GRID_Y)},
                            };
                            std::string why;
                            if(!lower(ctx, kd, k, why))
                            {
                                throw std::runtime_error(family + " " + d.label + ": " + why);
                            }
                            d.kernels.push_back(std::move(k));
                            out.push_back(std::move(d));
                        }
                    }
                }
            }
        }
    }

    // ---- gemm: the universal GEMM bodies of the emulated-f32 routes (rocKE dispatch/gemm/cf32
    // cf32_body) with their fused operand prologue (the Python reference's _gemm_spec) ----

    // route: bodies (GEMM_BODIES); the 64-bit-address body (GEMM_WIDE_BODY)
    struct GemmRoute
    {
        char const*              route;
        std::vector<char const*> bodies;
        std::vector<char const*> dtypes;
    };

    std::vector<GemmRoute> const& gemmRoutes()
    {
        static std::vector<GemmRoute> const routes = {
            {"h16", {"RIW", "RIW_SK", "C3", "Q3_SK"}, {"fp16", "bf16"}},
            {"x6", {"RIW", "RIW_SK", "C3"}, {"bf16"}},
            {"cf32", {"RIW", "RIW_SK", "Q3_SK"}, {"bf16"}},
            // the other f32 recipes (HIPTENSOR_ROCKE_F32_RECIPE), on the bodies of x6 / cf32
            {"x3", {"RIW", "RIW_SK", "C3"}, {"bf16"}},
            {"x4", {"RIW", "RIW_SK", "C3"}, {"bf16"}},
            {"x9", {"RIW", "RIW_SK", "C3"}, {"bf16"}},
            {"cf32x3", {"RIW", "RIW_SK", "Q3_SK"}, {"bf16"}},
            {"cf32x4", {"RIW", "RIW_SK", "Q3_SK"}, {"bf16"}},
            {"cf32x9", {"RIW", "RIW_SK", "Q3_SK"}, {"bf16"}},
        };
        return routes;
    }
    char const* const kGemmWideBody = "RIW";

    std::string lowercase(std::string s)
    {
        for(auto& c : s)
        {
            c = char(std::tolower(static_cast<unsigned char>(c)));
        }
        return s;
    }

    void gemmDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        if(!contains(kGemmArches, ctx.arch))
        {
            return;
        }
        for(auto const& r : gemmRoutes())
        {
            std::string route = r.route;
            for(char const* dt : r.dtypes)
            {
                struct Item
                {
                    char const* body;
                    bool        batched;
                    bool        wide;
                };
                std::vector<Item> items;
                for(char const* body : r.bodies)
                {
                    items.push_back({body, false, false});
                    items.push_back({body, true, false});
                }
                // the 64-bit-address body for operands past the DMA buffer range or a result
                // past 2^31 elements (one batch per launch)
                items.push_back({kGemmWideBody, false, true});
                for(auto const& it : items)
                {
                    Descriptor d;
                    d.family = "gemm";
                    d.label  = "route=" + route + " dtype=" + dt + " body=" + it.body
                              + " batched=" + (it.batched ? "1" : "0") + (it.wide ? " wide=1" : "");
                    std::string name
                        = "hiptensor_" + route + "_" + lowercase(it.body) + (it.wide ? "_w64" : "");
                    rocke_gemm_universal_spec_t spec = rocke_gemm_universal_spec_default();
                    spec.name                        = name.c_str();
                    if(!rocke_gemm_cf32_body(it.body, ctx.arch.c_str(), &spec))
                    {
                        throw std::runtime_error("gemm: unknown body " + std::string(it.body));
                    }
                    spec.data.dtype_a = dt;
                    spec.data.dtype_b = dt;
                    spec.data.dtype_c = dt;
                    spec.batched      = it.batched;
                    if(it.wide
                       && !rocke_gemm_cf32_adjust_k(
                           &spec, spec.tile.tile_m, spec.tile.tile_n, true))
                    {
                        throw std::runtime_error("gemm " + d.label + ": no 64-bit-address form");
                    }
                    char reason[512] = {0};
                    if(!rocke_gemm_universal_is_valid_spec(
                           &spec, ctx.arch.c_str(), reason, sizeof reason))
                    {
                        d.applies = false;
                        d.reason  = reason;
                        out.push_back(std::move(d));
                        continue;
                    }
                    rocke_gemm_prologue_t pro = rocke_gemm_prologue_default();
                    pro.out_f32               = true;
                    // real routes name their recipe; complex routes are cf32 (x6) or
                    // cf32<recipe>
                    bool const  cplx   = route.compare(0, 4, "cf32") == 0;
                    std::string recipe = cplx ? route.substr(4) : route;
                    if(recipe.empty())
                    {
                        recipe = "x6";
                    }
                    if(route != "h16")
                    {
                        pro.src      = "f32";
                        pro.recipe   = recipe.c_str();
                        pro.split_at = "use";
                        pro.cplx_b   = cplx;
                        pro.psub     = true;
                    }
                    if(!rocke_gemm_prologue_check(&pro, reason, sizeof reason))
                    {
                        throw std::runtime_error("gemm " + d.label + ": " + reason);
                    }
                    spec._prologue = &pro;
                    Builder             b;
                    rocke_kernel_def_t* kd
                        = rocke_build_universal_gemm_new(&b.b, &spec, ctx.arch.c_str());
                    if(kd == nullptr)
                    {
                        throw std::runtime_error("gemm " + d.label + ": "
                                                 + rocke_ir_builder_error(&b.b));
                    }
                    long long fb = 0, pb = 0;
                    if(!rocke_gemm_universal_streamk_workspace_bytes(&spec, &fb, &pb))
                    {
                        throw std::runtime_error("gemm " + d.label + ": stream-K workspace");
                    }
                    Kernel k;
                    k.name  = kd->name;
                    k.attrs = {
                        {"family", std::string("gemm")},
                        {"abi", std::string("hiptensor-aot-gemm/v1")},
                        {"route", route},
                        {"dtype", std::string(dt)},
                        {"body", std::string(it.body)},
                        {"batched", int64_t(it.batched)},
                        {"tile_m", int64_t(spec.tile.tile_m)},
                        {"tile_n", int64_t(spec.tile.tile_n)},
                        {"tile_k", int64_t(spec.tile.tile_k)},
                        {"block", int64_t(spec.block_size)},
                        {"persistent_ctas", int64_t(spec.trait.persistent_ctas)},
                        {"streamk", int64_t(spec.trait.streamk)},
                        {"sk_flags_bytes", int64_t(fb)},
                        {"sk_part_bytes", int64_t(pb)},
                        {"wide", int64_t(it.wide)},
                        // capability: launched on extents padded to tile multiples; the operands
                        // within the DMA buffer range, C within the i32 element offsets, rows of
                        // tiles on grid y. The 64-bit-address body: only one tile's operand rows
                        // within the DMA buffer range and one tile's C rows within the i32
                        // element offsets.
                        {"limit_operand_bytes",
                         int64_t(ROCKE_GEMM_UNIVERSAL_DTL_BUFFER_NUM_RECORDS)},
                        {"limit_c_elements", int64_t((int64_t(1) << 31) - 1)},
                        {"limit_tile_operand_bytes",
                         int64_t(ROCKE_GEMM_UNIVERSAL_DTL_BUFFER_NUM_RECORDS)},
                        {"limit_c_tile_elements", int64_t(1) << 31},
                        {"limit_grid_yz", int64_t(65535)},
                    };
                    std::string why;
                    if(!lower(ctx, kd, k, why))
                    {
                        throw std::runtime_error("gemm " + d.label + ": " + why);
                    }
                    d.kernels.push_back(std::move(k));
                    out.push_back(std::move(d));
                }
            }
        }
    }

    // ---- ew / reduce / ewt / ewv / dgett / rgett: the shape-generic hipTensor kernels of rocKE's
    // library/kernels/common/hiptensor_aot (rocke/instance_hiptensor_aot.h) ----

    // the Python reference's EW_ROWS_*: unary (A, D), binary (A, C, D), trinary (A, B, C, D),
    // complex (component types) unary / binary, and the mixed-type rows of contractions
    char const* const kReal[4] = {"f16", "bf16", "f32", "f64"};

    using Row2 = hiptensor::rockebackend::EwRow2;
    using Row3 = hiptensor::rockebackend::EwRow3;
    using Row4 = hiptensor::rockebackend::EwRow4;

    // the rows of the API's operations (rocke_ew_rows.hpp, shared with the JIT)
    std::vector<Row2> const kEwRows1(hiptensor::rockebackend::kEwRows1.begin(),
                                     hiptensor::rockebackend::kEwRows1.end());
    std::vector<Row3> const kEwRows2(hiptensor::rockebackend::kEwRows2.begin(),
                                     hiptensor::rockebackend::kEwRows2.end());
    std::vector<Row4> const kEwRows3(hiptensor::rockebackend::kEwRows3.begin(),
                                     hiptensor::rockebackend::kEwRows3.end());
    std::vector<Row2> const kEwRowsC1(hiptensor::rockebackend::kEwRowsC1.begin(),
                                      hiptensor::rockebackend::kEwRowsC1.end());
    std::vector<Row3> const kEwRowsC2(hiptensor::rockebackend::kEwRowsC2.begin(),
                                      hiptensor::rockebackend::kEwRowsC2.end());
    std::vector<Row2> const kReduceRows
        = {{"f16", "f16"}, {"bf16", "bf16"}, {"f32", "f32"}, {"f64", "f64"}};

    // EW_ROWS_1_MIXED / EW_ROWS_2_MIXED / EW_ROWS_C2_MIXED (rocke_ew_rows.hpp, shared with the JIT)
    using hiptensor::rockebackend::ewRows1Mixed;
    using hiptensor::rockebackend::ewRows2Mixed;

    // the Python reference's ew_params(idx64, mixed): the rows of the generic elementwise kernel
    std::vector<rocke_htaot_ew_spec_t> ewParams(bool idx64, bool mixed)
    {
        std::vector<Row2> rows1  = kEwRows1;
        std::vector<Row3> rows2  = kEwRows2;
        std::vector<Row3> rowsc2 = kEwRowsC2;
        if(mixed)
        {
            auto m1 = ewRows1Mixed();
            auto m2 = ewRows2Mixed(false);
            auto mc = ewRows2Mixed(true);
            rows1.insert(rows1.end(), m1.begin(), m1.end());
            rows2.insert(rows2.end(), m2.begin(), m2.end());
            rowsc2.insert(rowsc2.end(), mc.begin(), mc.end());
        }
        std::vector<rocke_htaot_ew_spec_t> out;
        auto                               row =
            [&](int arity, char const* a, char const* b, char const* c, char const* d, bool cplx) {
                rocke_htaot_ew_spec_t s = rocke_htaot_ew_spec_default();
                s.arity                 = arity;
                s.a                     = a;
                s.b                     = b;
                s.c                     = c;
                s.d                     = d;
                s.cplx                  = cplx;
                s.idx64                 = idx64;
                out.push_back(s);
            };
        for(auto const& r : rows1)
            row(1, r[0], nullptr, nullptr, r[1], false);
        for(auto const& r : rows2)
            row(2, r[0], nullptr, r[1], r[2], false);
        for(auto const& r : kEwRows3)
            row(3, r[0], r[1], r[2], r[3], false);
        for(auto const& r : kEwRowsC1)
            row(1, r[0], nullptr, nullptr, r[1], true);
        for(auto const& r : rowsc2)
            row(2, r[0], nullptr, r[1], r[2], true);
        // the same real rows with the math operators: separate kernels
        size_t n = out.size();
        for(size_t i = 0; i < n; ++i)
        {
            if(!out[i].cplx)
            {
                rocke_htaot_ew_spec_t s = out[i];
                s.math                  = true;
                out.push_back(s);
            }
        }
        return out;
    }

    std::string str(char const* s)
    {
        return s == nullptr ? std::string() : std::string(s);
    }

    int64_t const kLimitIndex = (int64_t(1) << 31) - 1;

    // Builds `kd` (NULL: the builder's error), lowers it and adds it to `d`.
    void addKernel(
        Context const& ctx, Descriptor& d, Builder& b, rocke_kernel_def_t* kd, Attrs attrs)
    {
        if(kd == nullptr)
        {
            throw std::runtime_error(d.family + " " + d.label + ": "
                                     + rocke_ir_builder_error(&b.b));
        }
        Kernel k;
        k.name  = kd->name;
        k.attrs = std::move(attrs);
        std::string why;
        if(!lower(ctx, kd, k, why))
        {
            throw std::runtime_error(d.family + " " + d.label + ": " + why);
        }
        d.kernels.push_back(std::move(k));
    }

    std::string ewLabel(rocke_htaot_ew_spec_t const& s)
    {
        return "arity=" + std::to_string(s.arity) + " a=" + str(s.a) + " b=" + str(s.b)
               + " c=" + str(s.c) + " d=" + str(s.d) + " cplx=" + (s.cplx ? "1" : "0")
               + " idx64=" + (s.idx64 ? "1" : "0") + " math=" + (s.math ? "1" : "0");
    }

    void ewDescriptor(Context const&               ctx,
                      rocke_htaot_ew_spec_t const& s,
                      std::vector<Descriptor>&     out)
    {
        Descriptor d;
        d.family = "ew";
        d.label  = ewLabel(s);
        Builder b;
        addKernel(ctx,
                  d,
                  b,
                  rocke_build_htaot_ew(&b.b, &s),
                  {
                      {"family", std::string("ew")},
                      {"abi", std::string(ROCKE_HTAOT_EW_ABI)},
                      {"arity", int64_t(s.arity)},
                      {"cplx", int64_t(s.cplx)},
                      {"idx64", int64_t(s.idx64)},
                      {"math", int64_t(s.math)},
                      {"a", str(s.a)},
                      {"b", str(s.b)},
                      {"c", str(s.c)},
                      {"d", str(s.d)},
                      {"block", int64_t(s.block)},
                      // capability: rank bucket R8, i32 coordinates
                      {"max_rank", int64_t(ROCKE_HTAOT_MAX_RANK)},
                      {"limit_index", kLimitIndex},
                  });
        out.push_back(std::move(d));
    }

    void reduceDescriptor(Context const&                   ctx,
                          rocke_htaot_reduce_spec_t const& s,
                          std::vector<Descriptor>&         out)
    {
        Descriptor d;
        d.family = "reduce";
        d.label  = std::string("mode=") + s.mode + " a=" + s.a + " d=" + s.d
                  + " idx64=" + (s.idx64 ? "1" : "0") + " math=" + (s.math ? "1" : "0");
        Builder b;
        addKernel(ctx,
                  d,
                  b,
                  rocke_build_htaot_reduce(&b.b, &s),
                  {
                      {"family", std::string("reduce")},
                      {"abi", std::string(ROCKE_HTAOT_REDUCE_ABI)},
                      {"mode", str(s.mode)},
                      {"a", str(s.a)},
                      {"d", str(s.d)},
                      {"idx64", int64_t(s.idx64)},
                      {"math", int64_t(s.math)},
                      {"block", int64_t(s.block)},
                      {"max_rank", int64_t(ROCKE_HTAOT_MAX_RANK)},
                      {"limit_index", kLimitIndex},
                  });
        out.push_back(std::move(d));
    }

    // ew and reduce: for each offset width the elementwise rows, then the reduction rows
    void ewReduceDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        for(bool idx64 : {false, true})
        {
            for(auto const& s : ewParams(idx64, true))
            {
                ewDescriptor(ctx, s, out);
            }
            for(bool math : {false, true})
            {
                for(auto const& r : kReduceRows)
                {
                    for(char const* mode : {"row", "col", "final"})
                    {
                        rocke_htaot_reduce_spec_t s = rocke_htaot_reduce_spec_default();
                        s.mode                      = mode;
                        s.a                         = r[0];
                        s.d                         = r[1];
                        s.idx64                     = idx64;
                        s.math                      = math;
                        reduceDescriptor(ctx, s, out);
                    }
                }
            }
        }
    }

    // ---- rreduce / rreduce_final: the tuned reduction classes of the target (its palette, then
    // the classes of its tuned problems: rocKE's reduction route table) and the finalize classes
    // (rocke/instance_hiptensor_aot.h rocke_htaot_rreduce_pack_classes / _final_classes) ----
    void rreduceDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        char err[512] = {0};
        int  n = rocke_htaot_rreduce_pack_classes(ctx.arch.c_str(), nullptr, 0, err, sizeof err);
        if(n < 0)
        {
            throw std::runtime_error(std::string("rreduce: ") + err);
        }
        std::vector<rocke_htaot_rreduce_spec_t> classes(std::max(n, 1));
        rocke_htaot_rreduce_pack_classes(ctx.arch.c_str(), classes.data(), n, err, sizeof err);
        for(int i = 0; i < n; ++i)
        {
            auto const& s = classes[i];
            Descriptor  d;
            d.family       = "rreduce";
            char name[256] = {0};
            rocke_htaot_rreduce_kernel_name(&s, name, sizeof name);
            d.label = name;
            Builder b;
            addKernel(ctx,
                      d,
                      b,
                      rocke_build_htaot_rreduce(&b.b, &s),
                      {
                          {"family", std::string("rreduce")},
                          {"abi", std::string(ROCKE_HTAOT_RREDUCE_ABI)},
                          {"mode", str(s.mode)},
                          {"a", str(s.a)},
                          {"d", str(s.d)},
                          {"op", str(s.op)},
                          {"nr", int64_t(s.nr)},
                          {"block", int64_t(s.block)},
                          {"vec", int64_t(s.vec)},
                          {"lanes", int64_t(s.lanes)},
                          {"unroll", int64_t(s.unroll)},
                          {"rows", int64_t(s.rows)},
                          {"nt", int64_t(s.nt)},
                          {"una", int64_t(s.una)},
                          {"idx64", int64_t(s.idx64)},
                          {"max_rank", int64_t(ROCKE_HTAOT_MAX_RANK)},
                      });
            out.push_back(std::move(d));
        }
        int const nf = rocke_htaot_rreduce_final_classes(nullptr, 0);
        std::vector<rocke_htaot_rreduce_final_spec_t> finals(std::max(nf, 1));
        rocke_htaot_rreduce_final_classes(finals.data(), nf);
        for(int i = 0; i < nf; ++i)
        {
            auto const& f = finals[i];
            Descriptor  d;
            d.family       = "rreduce_final";
            char name[256] = {0};
            rocke_htaot_rreduce_final_kernel_name(&f, name, sizeof name);
            d.label = name;
            Builder b;
            addKernel(ctx,
                      d,
                      b,
                      rocke_build_htaot_rreduce_final(&b.b, &f),
                      {
                          {"family", std::string("rreduce_final")},
                          {"abi", std::string(ROCKE_HTAOT_RREDUCE_ABI)},
                          {"a", str(f.a)},
                          {"d", str(f.d)},
                          {"op", str(f.op)},
                          {"lanes", int64_t(f.lanes)},
                          {"idx64", int64_t(f.idx64)},
                          {"block", int64_t(f.block)},
                          {"max_rank", int64_t(ROCKE_HTAOT_MAX_RANK)},
                      });
            out.push_back(std::move(d));
        }
    }

    // the real rows of the API's elementwise operations (32-bit offsets, no math operators)
    std::vector<rocke_htaot_ew_spec_t> plainRows()
    {
        std::vector<rocke_htaot_ew_spec_t> out;
        for(auto const& s : ewParams(false, false))
        {
            if(!s.cplx && !s.math)
            {
                out.push_back(s);
            }
        }
        return out;
    }

    void ewtDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        // every plain row and its folded class (several small planes per tile)
        for(auto const& p : plainRows())
        {
            for(bool fold : {false, true})
            {
                rocke_htaot_ewt_spec_t s{};
                s.arity = p.arity;
                s.a     = p.a;
                s.b     = p.b;
                s.c     = p.c;
                s.d     = p.d;
                s.fold  = fold;
                Descriptor d;
                d.family = "ewt";
                d.label  = "arity=" + std::to_string(s.arity) + " a=" + str(s.a) + " b=" + str(s.b)
                          + " c=" + str(s.c) + " d=" + str(s.d) + " fold=" + (fold ? "1" : "0");
                Builder b;
                addKernel(ctx,
                          d,
                          b,
                          rocke_build_htaot_ewt(&b.b, &s),
                          {
                              {"family", std::string("ewt")},
                              {"abi", std::string(ROCKE_HTAOT_EWT_ABI)},
                              {"arity", int64_t(s.arity)},
                              {"a", str(s.a)},
                              {"b", str(s.b)},
                              {"c", str(s.c)},
                              {"d", str(s.d)},
                              {"block", int64_t(ROCKE_HTAOT_EWT_THREADS)},
                              {"tile_o", int64_t(ROCKE_HTAOT_EWT_TILE_O)},
                              {"tile_t", int64_t(rocke_htaot_ewt_tile_t(&s))},
                              {"fold", int64_t(fold)},
                              // capability: batch digit slots, i32 element offsets; folded: planes
                              // per tile (the LDS table of batch offsets)
                              {"slots_b", int64_t(ROCKE_HTAOT_EWT_SLOTS_B)},
                              {"max_planes", int64_t(ROCKE_HTAOT_EWT_THREADS)},
                              {"limit_index", kLimitIndex},
                          });
                out.push_back(std::move(d));
            }
        }
    }

    // the vector classes of the plain rows (jit_ew.jit_classes with vec > 1: the output's
    // vector width and every combination of vector / gathered inputs)
    void ewvDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        for(auto const& p : plainRows())
        {
            int nIn = p.arity; // A, then C (arity 2) or B and C (arity 3)
            for(int mask = 0; mask < (1 << nIn); ++mask)
            {
                rocke_htaot_ew_jit_spec_t s{};
                s.base   = p;
                s.vec    = rocke_htaot_ew_jit_vec(p.d);
                s.n_unit = nIn;
                std::string unit;
                for(int i = 0; i < nIn; ++i)
                {
                    s.unit[i] = (mask >> i & 1) != 0;
                    unit += s.unit[i] ? "u" : "g";
                }
                if(s.vec <= 1)
                {
                    continue;
                }
                Descriptor d;
                d.family = "ewv";
                d.label  = ewLabel(p) + " vec=" + std::to_string(s.vec) + " unit=" + unit;
                Builder b;
                addKernel(ctx,
                          d,
                          b,
                          rocke_build_htaot_ew_jit(&b.b, &s, nullptr, 0),
                          {
                              {"family", std::string("ewv")},
                              {"abi", std::string(ROCKE_HTAOT_EWV_ABI)},
                              {"arity", int64_t(p.arity)},
                              {"cplx", int64_t(0)},
                              {"idx64", int64_t(0)},
                              {"math", int64_t(0)},
                              {"a", str(p.a)},
                              {"b", str(p.b)},
                              {"c", str(p.c)},
                              {"d", str(p.d)},
                              {"block", int64_t(p.block)},
                              {"vec", int64_t(s.vec)},
                              {"unit", unit},
                              {"max_rank", int64_t(ROCKE_HTAOT_MAX_RANK)},
                              {"limit_index", kLimitIndex},
                          });
                out.push_back(std::move(d));
            }
        }
    }

    // the flat classes of the plain rows whose tensors are all f16 or bf16 (ewf.ewf_classes:
    // streaming accesses, gfx942 / gfx950)
    void ewfDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        if(ctx.arch != "gfx942" && ctx.arch != "gfx950")
        {
            return;
        }
        auto sixteen = [](char const* t) {
            return t == nullptr || std::string(t) == "f16" || std::string(t) == "bf16";
        };
        for(auto const& p : plainRows())
        {
            if(!sixteen(p.a) || !sixteen(p.b) || !sixteen(p.c) || !sixteen(p.d))
            {
                continue;
            }
            rocke_htaot_ewf_spec_t s = rocke_htaot_ewf_spec_default();
            s.arity                  = p.arity;
            s.a                      = p.a;
            s.b                      = p.b;
            s.c                      = p.c;
            s.d                      = p.d;
            s.nt                     = true;
            Descriptor d;
            d.family = "ewf";
            d.label  = ewLabel(p) + " nt=1";
            Builder b;
            addKernel(ctx,
                      d,
                      b,
                      rocke_build_htaot_ewf(&b.b, &s),
                      {
                          {"family", std::string("ewf")},
                          {"abi", std::string(ROCKE_HTAOT_EWF_ABI)},
                          {"arity", int64_t(s.arity)},
                          {"a", str(s.a)},
                          {"b", str(s.b)},
                          {"c", str(s.c)},
                          {"d", str(s.d)},
                          {"block", int64_t(s.block)},
                          {"vec", int64_t(rocke_htaot_ewf_vec(&s))},
                          {"unroll", int64_t(ROCKE_HTAOT_EWF_UNROLL)},
                          {"nt", int64_t(s.nt)},
                          {"limit_index", kLimitIndex},
                      });
            out.push_back(std::move(d));
        }
    }

    // The direct GETT (every target with MFMA) and the ring GETT classes (gfx950).
    char const* const kDgettArches[] = {"gfx90a", "gfx942", "gfx950"};
    char const* const kRgettArches[] = {"gfx950"};

    void dgettDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        if(!contains(kDgettArches, ctx.arch))
        {
            return;
        }
        for(bool cplx : {false, true})
        {
            std::vector<char const*> comps = cplx ? std::vector<char const*>{"f32", "f64"}
                                                  : std::vector<char const*>(kReal, kReal + 4);
            for(char const* comp : comps)
            {
                for(char const* aFast : {"k", "r"})
                {
                    for(char const* bFast : {"k", "r"})
                    {
                        for(bool math :
                            cplx ? std::vector<bool>{false} : std::vector<bool>{false, true})
                        {
                            for(bool idx64 : {false, true})
                            {
                                rocke_htaot_dgett_spec_t s = rocke_htaot_dgett_spec_default();
                                s.comp                     = comp;
                                s.cplx                     = cplx;
                                s.a_fast                   = aFast;
                                s.b_fast                   = bFast;
                                s.math                     = math;
                                s.idx64                    = idx64;
                                Descriptor d;
                                d.family = "dgett";
                                d.label  = std::string("comp=") + comp
                                          + " cplx=" + (cplx ? "1" : "0") + " a_fast=" + aFast
                                          + " b_fast=" + bFast + " math=" + (math ? "1" : "0")
                                          + " idx64=" + (idx64 ? "1" : "0");
                                char reason[512] = {0};
                                if(!rocke_htaot_dgett_check(
                                       &s, ctx.arch.c_str(), reason, sizeof reason))
                                {
                                    d.applies = false;
                                    d.reason  = reason;
                                    out.push_back(std::move(d));
                                    continue;
                                }
                                rocke_htaot_dgett_config_t cfg = rocke_htaot_dgett_config(&s);
                                Builder                    b;
                                addKernel(
                                    ctx,
                                    d,
                                    b,
                                    rocke_build_htaot_dgett(&b.b, &s, ctx.arch.c_str()),
                                    {
                                        {"family", std::string("dgett")},
                                        {"abi", std::string(ROCKE_HTAOT_DGETT_ABI)},
                                        {"comp", std::string(comp)},
                                        {"cplx", int64_t(cplx)},
                                        {"a_fast", std::string(aFast)},
                                        {"b_fast", std::string(bFast)},
                                        {"math", int64_t(math)},
                                        {"idx64", int64_t(idx64)},
                                        {"tile_m", int64_t(cfg.tm)},
                                        {"tile_n", int64_t(cfg.tn)},
                                        {"tile_k", int64_t(cfg.tk)},
                                        {"block", int64_t(ROCKE_HTAOT_DGETT_THREADS)},
                                        // capability: digit slots of the M / N spaces, the K
                                        // tiles, the in-tile K digits and the batch space; i32
                                        // coordinates; 32-bit offsets unless idx64
                                        {"slots_mn", int64_t(ROCKE_HTAOT_DGETT_SLOTS_MN)},
                                        {"slots_kt", int64_t(ROCKE_HTAOT_DGETT_SLOTS_KT)},
                                        {"slots_kl", int64_t(ROCKE_HTAOT_DGETT_SLOTS_KL)},
                                        {"slots_l", int64_t(ROCKE_HTAOT_DGETT_SLOTS_L)},
                                        {"limit_index", kLimitIndex},
                                        {"limit_grid_yz", int64_t(65535)},
                                    });
                                out.push_back(std::move(d));
                            }
                        }
                    }
                }
            }
        }
    }

    void rgettDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        if(!contains(kRgettArches, ctx.arch))
        {
            return;
        }
        int                                   n = rocke_htaot_rgett_classes(nullptr, nullptr, 0);
        std::vector<rocke_htaot_rgett_spec_t> classes(static_cast<size_t>(n));
        rocke_htaot_rgett_classes(nullptr, classes.data(), n);
        for(auto const& s : classes)
        {
            Descriptor d;
            d.family = "rgett";
            char name[256];
            if(rocke_htaot_rgett_kernel_name(&s, name, sizeof name) != ROCKE_OK)
            {
                throw std::runtime_error("rgett: kernel name");
            }
            d.label = name;
            Builder b;
            addKernel(ctx,
                      d,
                      b,
                      rocke_build_htaot_rgett(&b.b, &s, ctx.arch.c_str(), nullptr),
                      {
                          {"family", std::string("rgett")},
                          {"abi", std::string(ROCKE_HTAOT_RGETT_ABI)},
                          {"dtype", str(s.dtype)},
                          {"bilinear", int64_t(s.bilinear)},
                          {"vec_a", int64_t(s.vec_a)},
                          {"vec_b", int64_t(s.vec_b)},
                          {"vec_d", int64_t(s.vec_d)},
                          {"pair", int64_t(s.epi_pair)},
                          {"wide", int64_t(s.wide_d)},
                          // digits per digit space the class decomposes (0: every slot), C
                          // prefetch distance
                          {"slots", int64_t(s.slots)},
                          {"cp", int64_t(s.c_prefetch)},
                          {"tile_m", int64_t(256)},
                          {"tile_n", int64_t(256)},
                          {"tile_k", int64_t(s.tile_k)},
                          {"block", int64_t(256)},
                          {"slots_row", int64_t(ROCKE_HTAOT_RGETT_SLOTS_ROW)},
                          {"slots_kt", int64_t(ROCKE_HTAOT_RGETT_SLOTS_KT)},
                          {"slots_kl", int64_t(ROCKE_HTAOT_RGETT_SLOTS_KL)},
                          // capability: operands within the DMA buffer range, D and C within the
                          // epilogue's 32-bit offsets (the wide classes: masked summed offsets),
                          // rows of tiles on grid y
                          {"limit_operand_bytes", int64_t(0x7FFF0000)},
                          {"limit_d_bytes",
                           s.wide_d ? int64_t(rocke::gett::MAX_D_BYTES) : int64_t(0x7FFFFFF0)},
                          {"limit_grid_yz", int64_t(65535)},
                      });
            out.push_back(std::move(d));
        }
    }

    // ---- gett_plan: the launches of rocKE's GETT route tables (the Python reference's
    // build_gett_plan): per table problem its pack and GETT kernels and its launch record ----

    std::string fnv1a64(std::string const& text)
    {
        uint64_t h = 0xCBF29CE484222325ull;
        for(unsigned char c : text)
        {
            h = (h ^ c) * 0x100000001B3ull;
        }
        char buf[17];
        std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
        return buf;
    }

    // the Python reference's gett_problem_key (hipTensor's rocke_plan.cpp builds the same
    // string):
    // tensors D, A, B as (mode, extent, stride) lists
    using ModeExt = std::tuple<std::string, int64_t, int64_t>;

    std::string gettProblemKey(std::string const&                       dtype,
                               bool                                     bilinear,
                               std::vector<std::vector<ModeExt>> const& tensors)
    {
        std::map<std::string, std::string> names;
        std::string                        key    = dtype + "|" + (bilinear ? "bilinear" : "scale");
        char const*                        labels = "DAB";
        for(size_t t = 0; t < tensors.size(); ++t)
        {
            std::vector<ModeExt> ms;
            for(auto const& m : tensors[t])
            {
                if(std::get<1>(m) != 1)
                {
                    ms.push_back(m);
                }
            }
            std::stable_sort(ms.begin(), ms.end(), [](ModeExt const& x, ModeExt const& y) {
                return std::get<2>(x) < std::get<2>(y);
            });
            std::string part = std::string(1, labels[t]) + ":";
            for(size_t i = 0; i < ms.size(); ++i)
            {
                auto const& m = std::get<0>(ms[i]);
                if(!names.count(m))
                {
                    std::string x = "x" + std::to_string(names.size());
                    names[m]      = x;
                }
                part += (i ? "," : "") + names[m] + ":" + std::to_string(std::get<1>(ms[i])) + ":"
                        + std::to_string(std::get<2>(ms[i]));
            }
            key += "|" + part;
        }
        return key;
    }

    // One route table entry's launches (the Python reference's build_gett_plan): the pack kernels
    // and the GETT kernel of rocke::gett::table_plan, and the launch record
    // "v1;ws=<bytes>;steps=<n>" then per launch "|k=<kernel>;g=<x>,<y>,<z>;b=<threads>;a=<name>=
    // <value>,..." (values: <buffer>+<byte offset> with buffers A, B, C, O (the output) and W (the
    // workspace), #<int>, or @a / @b (alpha / beta)). Pack outputs sit in the workspace at
    // 256-byte multiples, in step order.
    void gettPlanDescriptor(Context const&                 ctx,
                            rocke::gett::TunedEntry const& e,
                            char const*                    table,
                            std::vector<Descriptor>&       out)
    {
        Descriptor d;
        d.family                         = "gett_plan";
        d.label                          = std::string(table) + " key=" + e.key;
        rocke::gett::TensorGettSpec prob = rocke::gett::tuned_problem(e);
        if(rocke::gett::route_key(prob) != e.key)
        {
            d.applies = false;
            d.reason  = "the route table entry's key does not reproduce";
            out.push_back(std::move(d));
            return;
        }
        rocke::gett::TablePlan plan;
        std::string            why;
        if(!rocke::gett::table_plan(e, ctx.arch, &plan, &why))
        {
            // the Python planner would fall back to its cost-model routes for a stale entry: not
            // mirrored, so the pack would differ
            throw std::runtime_error("gett_plan " + d.label + ": " + why);
        }
        struct Step
        {
            std::string                        kernel;
            int                                grid[3];
            int                                block;
            std::map<std::string, std::string> binds;
        };
        std::vector<Step>              steps;
        std::map<std::string, int64_t> wsOff;
        int64_t                        off = 0;
        auto                           addKernel
            = [&](rocke_kernel_def_t* kd, Builder& b, std::set<std::string> const& want) {
                  if(kd == nullptr)
                  {
                      throw std::runtime_error("gett_plan " + d.label + ": "
                                               + rocke_ir_builder_error(&b.b));
                  }
                  std::set<std::string> params;
                  for(int i = 0; i < kd->num_params; ++i)
                  {
                      params.insert(kd->params[i]->name);
                  }
                  if(params != want)
                  {
                      throw std::runtime_error("gett_plan " + d.label + ": kernel " + kd->name
                                               + " params are not the bound ones");
                  }
                  Kernel k;
                  k.name  = kd->name;
                  k.attrs = {
                      {"family", std::string("gett")},
                      {"abi", std::string("rocke-tensor-gett/v1")},
                  };
                  if(!lower(ctx, kd, k, why))
                  {
                      throw std::runtime_error("gett_plan " + d.label + ": " + why);
                  }
                  d.kernels.push_back(std::move(k));
              };
        for(auto const& st : plan.packs)
        {
            Builder             b;
            rocke_kernel_def_t* kd
                = rocke::gett::build_tensor_pack(&b.b, st.spec, ctx.arch.c_str());
            addKernel(kd, b, {"X", "Y"});
            wsOff[st.operand] = off;
            Step s;
            s.kernel = kd->name;
            std::copy(st.grid, st.grid + 3, s.grid);
            s.block = st.block;
            s.binds = {{"X", st.operand + "+0"}, {"Y", "W+" + std::to_string(off)}};
            steps.push_back(std::move(s));
            off += (st.workspace_bytes + 255) / 256 * 256;
        }
        Builder             b;
        rocke_kernel_def_t* kd = rocke::gett::build_tensor_gett(&b.b, plan.gett, ctx.arch.c_str());
        rocke::gett::Plan   pl;
        if(kd != nullptr && !plan.gett.plan(&pl, &why))
        {
            throw std::runtime_error("gett_plan " + d.label + ": " + why);
        }
        std::map<std::string, std::string> binds;
        if(kd != nullptr)
        {
            binds = {
                {"C", "O+0"},
                {"M", "#" + std::to_string(pl.M)},
                {"N", "#" + std::to_string(pl.N)},
                {"K", "#" + std::to_string(pl.K)},
                {"alpha", "@a"},
            };
            if(plan.gett.knobs.bilinear)
            {
                binds["Cin"]  = "C+0";
                binds["beta"] = "@b";
            }
            int src[2];
            rocke::gett::gett_operand_sources(plan, src);
            auto ops   = rocke::gett::launch_operands(plan.gett);
            binds["A"] = src[0] >= 0 ? "W+" + std::to_string(wsOff.at(plan.packs[src[0]].operand))
                                     : ops.first + "+0";
            binds["B"] = src[1] >= 0 ? "W+" + std::to_string(wsOff.at(plan.packs[src[1]].operand))
                                     : ops.second + "+0";
        }
        std::set<std::string> want;
        for(auto const& kv : binds)
        {
            want.insert(kv.first);
        }
        addKernel(kd, b, want);
        Step g;
        g.kernel = kd->name;
        rocke::gett::tensor_gett_grid(plan.gett, g.grid);
        g.block = rocke::gett::tensor_gett_block(plan.gett);
        g.binds = binds;
        steps.push_back(std::move(g));
        // the canonical key with the packed row-major strides the entry's problem has (in
        // hipTensor: the same memory layout, mode lists reversed and column-major strides)
        std::vector<std::vector<ModeExt>> tensors;
        for(auto const* modes : {&e.modes_d, &e.modes_a, &e.modes_b})
        {
            auto                 st = rocke::gett::packed_strides(*modes, e.extents);
            std::vector<ModeExt> t;
            for(auto const& m : *modes)
            {
                t.emplace_back(m, e.extents.at(m), st.at(m));
            }
            tensors.push_back(std::move(t));
        }
        std::string pkey = gettProblemKey(
            plan.gett.dtype == "fp16" ? "f16" : plan.gett.dtype, plan.gett.knobs.bilinear, tensors);
        std::string record
            = "v1;ws=" + std::to_string(off) + ";steps=" + std::to_string(steps.size());
        for(auto const& s : steps)
        {
            record += "|k=" + s.kernel + ";g=" + std::to_string(s.grid[0]) + ","
                      + std::to_string(s.grid[1]) + "," + std::to_string(s.grid[2])
                      + ";b=" + std::to_string(s.block) + ";a=";
            bool first = true;
            for(auto const& kv : s.binds)
            {
                record += (first ? "" : ",") + kv.first + "=" + kv.second;
                first = false;
            }
        }
        std::string route                        = fnv1a64(pkey);
        d.routes["gett.tuned." + route]          = record;
        d.routes["gett.tuned." + route + ".key"] = pkey;
        out.push_back(std::move(d));
    }

    // ---- gett_plan: per entry of rocKE's tuned GETT route table, then of its AOT route table, the
    // launches of the entry's route ----
    void gettPlanDescriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        struct Table
        {
            char const* name;
            char const* json;
        };
        for(Table t : {Table{"tuned", rocke_tensor_gett_tuned_json()},
                       Table{"aot", rocke_tensor_gett_aot_json()}})
        {
            std::vector<rocke::gett::TunedEntry> entries;
            std::string                          why;
            if(!rocke::gett::tuned_entries(t.json, ctx.arch, &entries, &why))
            {
                throw std::runtime_error(std::string("gett_plan ") + t.name + ": " + why);
            }
            for(auto const& e : entries)
            {
                gettPlanDescriptor(ctx, e, t.name, out);
            }
        }
    }

    std::string slurp(fs::path const& p)
    {
        std::ifstream f(p, std::ios::binary);
        if(!f)
        {
            throw std::runtime_error("cannot read " + p.string());
        }
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    // ---------------------------------------------------------------------------------------------
    // Compile: a pool of worker processes (one COMGR per process, as the Python builder's
    // process pool), each compiling the next uncached kernel into the content-addressed cache.
    // ---------------------------------------------------------------------------------------------

    bool readFile(fs::path const& p, std::vector<uint8_t>& out)
    {
        std::ifstream f(p, std::ios::binary);
        if(!f)
        {
            return false;
        }
        out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        return true;
    }

    void writeFile(fs::path const& p, void const* data, size_t size)
    {
        fs::path tmp = p;
        tmp += ".tmp"
               + std::to_string(
#if defined(_WIN32)
                   0
#else
                   getpid()
#endif
               );
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f.write(static_cast<char const*>(data), std::streamsize(size));
            if(!f)
            {
                throw std::runtime_error("cannot write " + tmp.string());
            }
        }
        fs::rename(tmp, p);
    }

    // Compiles one kernel into <cache>/<key>.co, or writes <cache>/<key>.err.
    void compileOne(Context const& ctx, Kernel const& k, fs::path const& cache)
    {
        std::vector<uint8_t> co;
        std::string          log;
        if(hiptensor::rockebackend::compileLlvmIr(k.ll, ctx.isa, k.options, co, log))
        {
            writeFile(cache / (k.key + ".co"), co.data(), co.size());
        }
        else
        {
            log = k.name + ": " + log;
            writeFile(cache / (k.key + ".err"), log.data(), log.size());
        }
    }

    // Compiles every kernel of `work` that is not in the cache. False when a worker timed out or
    // died.
    bool compileAll(Context const&              ctx,
                    std::vector<Kernel const*>& work,
                    fs::path const&             cache,
                    int                         jobs,
                    double                      timeout)
    {
        if(work.empty())
        {
            return true;
        }
#if defined(_WIN32)
        (void)jobs;
        (void)timeout;
        for(auto const* k : work)
        {
            compileOne(ctx, *k, cache);
        }
        return true;
#else
        auto* next = static_cast<std::atomic<size_t>*>(mmap(nullptr,
                                                            sizeof(std::atomic<size_t>),
                                                            PROT_READ | PROT_WRITE,
                                                            MAP_SHARED | MAP_ANONYMOUS,
                                                            -1,
                                                            0));
        if(next == MAP_FAILED)
        {
            throw std::runtime_error("mmap failed");
        }
        new(next) std::atomic<size_t>(0);
        std::fflush(stdout);
        std::fflush(stderr);
        int                workers = std::max(1, std::min<int>(jobs, int(work.size())));
        std::vector<pid_t> pids;
        for(int w = 0; w < workers; ++w)
        {
            pid_t pid = fork();
            if(pid < 0)
            {
                throw std::runtime_error("fork failed");
            }
            if(pid == 0)
            {
                for(;;)
                {
                    size_t i = next->fetch_add(1);
                    if(i >= work.size())
                    {
                        break;
                    }
                    // a compile that runs past the limit ends the worker (SIGALRM)
                    alarm(unsigned(std::max(1.0, timeout)));
                    try
                    {
                        compileOne(ctx, *work[i], cache);
                    }
                    catch(std::exception const& e)
                    {
                        std::fprintf(stderr, "rocKE pack ERROR %s\n", e.what());
                        _exit(3);
                    }
                    alarm(0);
                }
                _exit(0);
            }
            pids.push_back(pid);
        }
        bool ok = true;
        for(pid_t pid : pids)
        {
            int status = 0;
            waitpid(pid, &status, 0);
            if(WIFSIGNALED(status))
            {
                std::fprintf(stderr,
                             "rocKE pack: compile worker %s\n",
                             WTERMSIG(status) == SIGALRM ? "timed out" : "died");
                ok = false;
            }
            else if(WEXITSTATUS(status) != 0)
            {
                ok = false;
            }
        }
        munmap(next, sizeof(std::atomic<size_t>));
        return ok;
#endif
    }

    // ---------------------------------------------------------------------------------------------
    // Routes and selection models
    // ---------------------------------------------------------------------------------------------

    std::map<std::string, std::string> routesFor(std::string const& arch, int numCus)
    {
        std::map<std::string, std::string> r;
        r["num_cus"] = std::to_string(numCus);
        if(contains(kGemmArches, arch))
        {
            // the thresholds of rocKE's emulated-f32 body rule (rocke.dispatch.gemm.cf32)
            r["gemm.cf32_sk_always_waves"] = pyFloat(ROCKE_GEMM_CF32_SK_ALWAYS_WAVES);
            r["gemm.cf32_sk_max_waves"]    = pyFloat(ROCKE_GEMM_CF32_SK_MAX_WAVES);
            r["gemm.cf32_sk_max_tail"]     = pyFloat(ROCKE_GEMM_CF32_SK_MAX_TAIL);
            r["gemm.cf32_small_max_waves"] = pyFloat(ROCKE_GEMM_CF32_SMALL_MAX_WAVES);
        }
        if(contains(kGett64Arches, arch))
        {
            r["gett64.k_block_min_k"] = std::to_string(ROCKE_GETT_F64_K_BLOCK_MIN_K);
        }
        // the tuned reduction problems of the target: reduce.tuned.<hash> (knobs) and .key
        char      err[512] = {0};
        int const n        = rocke_htaot_rreduce_route_count(arch.c_str(), err, sizeof err);
        if(n < 0)
        {
            throw std::runtime_error(std::string("rreduce routes: ") + err);
        }
        for(int i = 0; i < n; ++i)
        {
            std::vector<char> name(256), value(4096);
            if(!rocke_htaot_rreduce_route(
                   arch.c_str(), i, name.data(), name.size(), value.data(), value.size()))
            {
                throw std::runtime_error("rreduce route " + std::to_string(i) + " of " + arch);
            }
            r[name.data()] = value.data();
        }
        // the records the planner parses (rocke_rreduce_rule.hpp): a malformed one fails the
        // build instead of silently changing the reduction choice
        if(auto const why = hiptensor::rockebackend::rrule::rCheckRoutes(r); !why.empty())
        {
            throw std::runtime_error("rocKE pack " + arch + ": reduction routes: " + why);
        }
        return r;
    }

    std::map<std::string, std::string> selectModels(std::string const& modelDir,
                                                    std::string const& arch)
    {
        std::map<std::string, std::string> out;
        if(modelDir.empty())
        {
            return out;
        }
        fs::path d = fs::path(modelDir) / arch;
        if(!fs::is_directory(d))
        {
            return out;
        }
        std::vector<std::string> names;
        for(auto const& e : fs::directory_iterator(d))
        {
            std::string fn = e.path().filename().string();
            if(fn.size() > 6 && fn.compare(fn.size() - 6, 6, ".model") == 0)
            {
                names.push_back(fn);
            }
        }
        std::sort(names.begin(), names.end());
        for(auto const& fn : names)
        {
            std::string text = slurp(d / fn);
            // str.strip(): ASCII whitespace at both ends
            auto ws = [](char c) {
                return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'
                       || c == '\x1c' || c == '\x1d' || c == '\x1e' || c == '\x1f';
            };
            while(!text.empty() && ws(text.back()))
            {
                text.pop_back();
            }
            size_t b = 0;
            while(b < text.size() && ws(text[b]))
            {
                ++b;
            }
            text       = text.substr(b);
            bool ascii = std::all_of(text.begin(), text.end(), [](char c) {
                return static_cast<unsigned char>(c) < 0x80;
            });
            bool const format
                = text.rfind("v1;schema=", 0) == 0 || text.rfind("v2;schema=", 0) == 0;
            if(!ascii || !format || text.find('\n') != std::string::npos
               || text.find('\0') != std::string::npos)
            {
                throw std::runtime_error("rocKE pack: " + (d / fn).string()
                                         + " is not a selection model");
            }
            out["select.model." + fn.substr(0, fn.size() - 6)] = text;
        }
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // Manifest (--list)
    // ---------------------------------------------------------------------------------------------

    // Every descriptor of `ctx.arch`, in the Python reference's order.
    void collectDescriptors(Context const& ctx, std::vector<Descriptor>& descs)
    {
        ewReduceDescriptors(ctx, descs);
        rreduceDescriptors(ctx, descs);
        ewtDescriptors(ctx, descs);
        ewvDescriptors(ctx, descs);
        ewfDescriptors(ctx, descs);
        gemmDescriptors(ctx, descs);
        gettPlanDescriptors(ctx, descs);
        dgettDescriptors(ctx, descs);
        rgettDescriptors(ctx, descs);
        gett64Descriptors(ctx, descs);
        gett64Descriptors(ctx, descs, true);
    }

    // The kernel classes the JIT specializes (rocke_ew_rows.hpp jitClassAdmitted): every class of
    // every elementwise row of the pack (vec 1, and the output's vector width with every
    // combination of vector / gathered inputs) the JIT admits, by kernel name.
    std::set<std::string> jitClassNames()
    {
        std::set<std::string> names;
        for(bool idx64 : {false, true})
        {
            for(auto const& p : ewParams(idx64, true))
            {
                std::vector<rocke_htaot_ew_jit_spec_t> classes(1);
                classes[0].base = p;
                classes[0].vec  = 1;
                for(int mask = 0; mask < (1 << p.arity); ++mask)
                {
                    rocke_htaot_ew_jit_spec_t s{};
                    s.base   = p;
                    s.vec    = rocke_htaot_ew_jit_vec(p.d);
                    s.n_unit = p.arity;
                    for(int i = 0; i < p.arity; ++i)
                    {
                        s.unit[i] = (mask >> i & 1) != 0;
                    }
                    classes.push_back(s);
                }
                for(auto const& s : classes)
                {
                    char name[256] = {0};
                    if(hiptensor::rockebackend::jitClassAdmitted(s)
                       && rocke_htaot_ew_jit_kernel_name(&s, name, sizeof(name)) == ROCKE_OK)
                    {
                        names.insert(name);
                    }
                }
            }
        }
        return names;
    }

    // A manifest field: backslashes and line breaks escaped.
    std::string escaped(std::string const& s)
    {
        std::string out;
        for(char c : s)
        {
            out += c == '\\' ? "\\\\" : c == '\n' ? "\\n" : std::string(1, c);
        }
        return out;
    }

    // The manifest of the pack of `arch`: a comment line, then one line per kernel in pack order
    // ("kernel <name> <attribute>=<value>... params=<count>:<digest>", strings quoted; the digest:
    // the first 16 hex digits of the SHA-256 of the comma-joined argument names), one per route
    // ("route <key>=<value>", sorted, no selection models) and one per JIT kernel class
    // ("jit <kernel name>", sorted). The Python reference writes the same text.
    std::string
        manifestText(std::string const& arch, int numCus, std::vector<Descriptor> const& descs)
    {
        std::map<std::string, std::string> kernels;
        auto                               routes = routesFor(arch, numCus);
        for(auto const& d : descs)
        {
            if(!d.applies)
            {
                continue;
            }
            for(auto const& k : d.kernels)
            {
                std::string line = "kernel " + k.name;
                for(auto const& kv : k.attrs)
                {
                    line += " " + kv.first + "=";
                    if(auto const* i = std::get_if<int64_t>(&kv.second))
                    {
                        line += std::to_string(*i);
                    }
                    else
                    {
                        line += "\"" + escaped(std::get<std::string>(kv.second)) + "\"";
                    }
                }
                std::string names;
                for(size_t i = 0; i < k.params.size(); ++i)
                {
                    names += (i ? "," : "") + k.params[i];
                }
                auto const digest = hiptensor::rockepack::sha256(names.data(), names.size());
                line += " params=" + std::to_string(k.params.size()) + ":"
                        + hiptensor::rockepack::hex(digest.data(), digest.size()).substr(0, 16);
                // a kernel several descriptors build keeps the first one's record, as in the pack
                kernels.emplace(k.name, line);
            }
            for(auto const& kv : d.routes)
            {
                routes[kv.first] = kv.second;
            }
        }
        std::string out = "# rocKE kernel pack manifest " + arch + "\n";
        for(auto const& kv : kernels)
        {
            out += kv.second + "\n";
        }
        for(auto const& kv : routes)
        {
            out += "route " + escaped(kv.first) + "=" + escaped(kv.second) + "\n";
        }
        for(auto const& name : jitClassNames())
        {
            out += "jit " + name + "\n";
        }
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // main
    // ---------------------------------------------------------------------------------------------

    struct Options
    {
        std::vector<std::string> arches;
        std::string              out;
        int                      jobs = 0;
        std::string              cacheDir;
        double                   compileTimeout = 600.0;
        int                      numCus         = 0;
        std::string              only;
        std::string              selectModels;
        std::string              gemmModels; // --gemm-models DIR
        std::string              compilerId;
        std::string              rockeVersion;
        bool                     list = false; // --list
        std::string              checkManifests;
        std::string              rreducePicks; // --rreduce-picks FILE
    };

    [[noreturn]] void usage(char const* msg)
    {
        std::fprintf(
            stderr,
            "%s\nusage: rocke_pack_builder --arch GFX [--arch GFX ...] --out DIR [--jobs N]\n"
            "       [--cache-dir DIR] [--compile-timeout S] [--num-cus N]\n"
            "       [--select-models DIR] [--only REGEX] [--gemm-models DIR]\n"
            "       [--compiler-id STR] [--rocke-version STR]\n"
            "       rocke_pack_builder --list --arch GFX [--arch GFX ...] [--out DIR]\n"
            "       [--check-manifests DIR] [--num-cus N]\n"
            "       rocke_pack_builder --rreduce-picks FILE --arch GFX\n",
            msg);
        std::exit(2);
    }

    Options parse(int argc, char** argv)
    {
        Options o;
        for(int i = 1; i < argc; ++i)
        {
            std::string a   = argv[i];
            auto        val = [&]() -> std::string {
                if(i + 1 >= argc)
                {
                    usage(("missing value of " + a).c_str());
                }
                return argv[++i];
            };
            if(a == "--arch")
                o.arches.push_back(val());
            else if(a == "--out")
                o.out = val();
            else if(a == "--jobs")
                o.jobs = std::atoi(val().c_str());
            else if(a == "--cache-dir")
                o.cacheDir = val();
            else if(a == "--compile-timeout")
                o.compileTimeout = std::atof(val().c_str());
            else if(a == "--num-cus")
                o.numCus = std::atoi(val().c_str());
            else if(a == "--only")
                o.only = val();
            else if(a == "--select-models")
                o.selectModels = val();
            else if(a == "--gemm-models")
                o.gemmModels = val();
            else if(a == "--compiler-id")
                o.compilerId = val();
            else if(a == "--rocke-version")
                o.rockeVersion = val();
            else if(a == "--list")
                o.list = true;
            else if(a == "--check-manifests")
                o.checkManifests = val();
            else if(a == "--rreduce-picks")
                o.rreducePicks = val();
            else
                usage(("unknown argument " + a).c_str());
        }
        if(!o.checkManifests.empty() && !o.list)
        {
            usage("--check-manifests needs --list");
        }
        if(!o.rreducePicks.empty())
        {
            if(o.arches.size() != 1 || o.list)
            {
                usage("--rreduce-picks takes one --arch and no --list");
            }
            return o;
        }
        if(o.arches.empty() || (o.out.empty() && o.checkManifests.empty()))
        {
            usage(o.list ? "--arch and --out or --check-manifests are required"
                         : "--arch and --out are required");
        }
        return o;
    }

    // --gemm-models: the files Models::load reads, recorded.
    struct RecordingReader
    {
        fs::path                           dir;
        std::map<std::string, std::string> files;
    };

    bool recordModelFile(void* ctx, std::string const& name, std::string* text)
    {
        auto* r = static_cast<RecordingReader*>(ctx);
        if(name.empty() || name.find_first_of("/\\") != std::string::npos || name == "index.txt")
        {
            return false;
        }
        std::ifstream f(r->dir / name, std::ios::binary);
        if(!f)
        {
            return false;
        }
        std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        r->files[name] = body;
        *text          = std::move(body);
        return true;
    }

    // <out>/<arch>/gemm_models for a target the tensor GETT planner serves (removed for the
    // others). Returns the number of files written; throws on errors.
    size_t writeGemmModels(std::string const& modelDir,
                           fs::path const&    packDir,
                           std::string const& arch)
    {
        fs::path const  dst = packDir / "gemm_models";
        std::error_code ec;
        fs::remove_all(dst, ec);
        if(!rocke::gett::planner::plans_arch(arch))
        {
            return 0;
        }
        RecordingReader r;
        r.dir = modelDir;
        rocke::gemm_dispatch::Models models;
        std::string                  why;
        if(!models.load(recordModelFile, &r, arch, &why))
        {
            throw std::runtime_error("gemm models of " + arch + ": " + why);
        }
        if(r.files.empty())
        {
            throw std::runtime_error("gemm models of " + arch + ": no model file in " + modelDir);
        }
        fs::create_directories(dst);
        std::string index;
        for(auto const& [name, body] : r.files)
        {
            writeFile(dst / name, body.data(), body.size());
            auto d = hiptensor::rockepack::sha256(body.data(), body.size());
            index += hiptensor::rockepack::hex(d.data(), d.size()) + " " + name + "\n";
        }
        writeFile(dst / "index.txt", index.data(), index.size());
        return r.files.size();
    }

    // --list: writes and / or checks the manifests. 0, 1 when a manifest differs from the
    // expected one, 2 on errors.
    int runList(Options const& o)
    {
        int status = 0;
        for(auto const& arch : o.arches)
        {
            Context ctx;
            ctx.arch           = arch;
            ctx.isa            = hiptensor::rockebackend::isaName(arch);
            ctx.listOnly       = true;
            int const   numCus = o.numCus > 0 ? o.numCus : rocke_arch_num_cus(arch.c_str());
            std::string text;
            try
            {
                std::vector<Descriptor> descs;
                collectDescriptors(ctx, descs);
                text = manifestText(arch, numCus, descs);
            }
            catch(std::exception const& e)
            {
                std::fprintf(stderr, "rocKE pack ERROR %s: %s\n", arch.c_str(), e.what());
                status = 2;
                continue;
            }
            if(!o.out.empty())
            {
                fs::create_directories(o.out);
                writeFile(fs::path(o.out) / (arch + ".manifest"), text.data(), text.size());
            }
            if(o.checkManifests.empty())
            {
                continue;
            }
            fs::path    expectedPath = fs::path(o.checkManifests) / (arch + ".manifest");
            std::string expected;
            try
            {
                expected = slurp(expectedPath);
            }
            catch(std::exception const& e)
            {
                std::fprintf(stderr, "rocKE pack ERROR %s: %s\n", arch.c_str(), e.what());
                status = 2;
                continue;
            }
            if(expected == text)
            {
                std::printf("rocKE pack %s: manifest equals %s\n",
                            arch.c_str(),
                            expectedPath.string().c_str());
                continue;
            }
            auto lines = [](std::string const& s) {
                std::vector<std::string> out;
                size_t                   b = 0;
                for(size_t e = s.find('\n'); e != std::string::npos; e = s.find('\n', b))
                {
                    out.push_back(s.substr(b, e - b));
                    b = e + 1;
                }
                return out;
            };
            auto const                  have = lines(text);
            auto const                  want = lines(expected);
            std::set<std::string> const haveSet(have.begin(), have.end());
            std::set<std::string> const wantSet(want.begin(), want.end());
            std::fprintf(stderr,
                         "rocKE pack %s: the manifest differs from %s (rocKE's make_rocke_pack.py "
                         "--list writes that one)\n",
                         arch.c_str(),
                         expectedPath.string().c_str());
            int shown = 0;
            for(auto const& l : want)
            {
                if(!haveSet.count(l) && shown++ < 20)
                {
                    std::fprintf(stderr, "  only in the expected manifest: %s\n", l.c_str());
                }
            }
            for(auto const& l : have)
            {
                if(!wantSet.count(l) && shown++ < 40)
                {
                    std::fprintf(stderr, "  only in this builder's manifest: %s\n", l.c_str());
                }
            }
            if(shown == 0)
            {
                std::fprintf(stderr, "  the same lines in another order\n");
            }
            status = std::max(status, 1);
        }
        return status;
    }

    // --rreduce-picks FILE: the untuned reduction choice of the planner (rocke_rreduce_rule.hpp
    // untunedChoices, with the target's palette and prototypes) for each problem of FILE, one
    // per line: "cus=N;limit=BYTES;una=0|1;a=T;d=T;op=add|mul|max|min;k=e:sa:sd,...;r=e:sa,..."
    // (canonical kept / reduced modes, innermost first; limit -1: none). Prints one line per
    // problem: the choices as "<class>:split<s>" separated by spaces, "-" for none. rocKE's
    // cross-check test compares them with rreduce_host.untuned_choices.
    int runRreducePicks(Options const& o)
    {
        namespace rr         = hiptensor::rockebackend::rrule;
        auto const& arch     = o.arches[0];
        char        err[512] = {0};
        int const   n = rocke_htaot_rreduce_pack_classes(arch.c_str(), nullptr, 0, err, sizeof err);
        if(n < 0)
        {
            throw std::runtime_error(std::string("rreduce: ") + err);
        }
        std::vector<rocke_htaot_rreduce_spec_t> specs(std::max(n, 1));
        rocke_htaot_rreduce_pack_classes(arch.c_str(), specs.data(), n, err, sizeof err);
        std::vector<rr::RClass> classes;
        for(int i = 0; i < n; ++i)
        {
            auto const& s         = specs[i];
            char        name[256] = {0};
            rocke_htaot_rreduce_kernel_name(&s, name, sizeof name);
            rr::RClass c;
            c.name   = name;
            c.mode   = str(s.mode);
            c.a      = str(s.a);
            c.d      = str(s.d);
            c.op     = str(s.op);
            c.nr     = s.nr;
            c.block  = s.block;
            c.vec    = s.vec;
            c.lanes  = s.lanes;
            c.unroll = s.unroll;
            c.rows   = s.rows;
            c.nt     = s.nt;
            c.una    = s.una;
            c.idx64  = s.idx64;
            classes.push_back(c);
        }
        auto const              routes = routesFor(arch, o.numCus);
        std::vector<rr::RProto> protos;
        auto const              count  = routes.find("reduce.proto.count");
        int64_t const           nproto = count == routes.end() ? 0 : std::stoll(count->second);
        for(int64_t i = 0; i < nproto; ++i)
        {
            rr::RProto r;
            if(!rr::rParseProto(routes.at("reduce.proto." + std::to_string(i)), r))
            {
                throw std::runtime_error("reduce.proto." + std::to_string(i));
            }
            protos.push_back(r);
        }
        bool const    streaming = arch == "gfx942" || arch == "gfx950";
        std::ifstream in(o.rreducePicks);
        if(!in)
        {
            throw std::runtime_error("cannot read " + o.rreducePicks);
        }
        auto split = [](std::string const& t, char sep) {
            std::vector<std::string> out;
            size_t                   at = 0;
            while(at <= t.size())
            {
                size_t const e = t.find(sep, at);
                out.push_back(t.substr(at, e == std::string::npos ? std::string::npos : e - at));
                if(e == std::string::npos)
                {
                    break;
                }
                at = e + 1;
            }
            return out;
        };
        std::string line;
        while(std::getline(in, line))
        {
            if(line.empty())
            {
                continue;
            }
            rr::RProblem p;
            int64_t      cus   = 0;
            uint64_t     limit = UINT64_MAX;
            bool         una   = false;
            for(auto const& field : split(line, ';'))
            {
                auto const eq = field.find('=');
                if(eq == std::string::npos)
                {
                    throw std::runtime_error("bad field " + field + " in " + line);
                }
                auto const key = field.substr(0, eq), value = field.substr(eq + 1);
                if(key == "cus")
                    cus = std::stoll(value);
                else if(key == "limit")
                    limit = value == "-1" ? UINT64_MAX : std::stoull(value);
                else if(key == "una")
                    una = value == "1";
                else if(key == "a")
                    p.a = value;
                else if(key == "d")
                    p.d = value;
                else if(key == "op")
                    p.op = value;
                else if(key == "k" || key == "r")
                {
                    for(auto const& dim :
                        value.empty() ? std::vector<std::string>{} : split(value, ','))
                    {
                        auto const v = split(dim, ':');
                        if(key == "k" && v.size() == 3)
                            p.kept.push_back(
                                {std::stoll(v[0]), std::stoll(v[1]), std::stoll(v[2])});
                        else if(key == "r" && v.size() == 2)
                            p.red.push_back({std::stoll(v[0]), std::stoll(v[1])});
                        else
                            throw std::runtime_error("bad mode " + dim + " in " + line);
                    }
                }
                else
                    throw std::runtime_error("bad field " + field + " in " + line);
            }
            if(cus <= 0 || p.a.empty() || p.d.empty() || p.op.empty())
            {
                throw std::runtime_error("incomplete problem " + line);
            }
            p.eb = p.a == "f64" ? 8 : p.a == "f32" ? 4 : 2;
            std::string text;
            for(auto const& c : rr::untunedChoices(p, classes, protos, una, cus, streaming, limit))
            {
                text += (text.empty() ? "" : " ") + classes[c.cls].name + ":split"
                        + std::to_string(c.knobs.split);
            }
            std::printf("%s\n", text.empty() ? "-" : text.c_str());
        }
        return 0;
    }

    int run(Options const& o)
    {
        if(!o.rreducePicks.empty())
        {
            return runRreducePicks(o);
        }
        if(o.list)
        {
            return runList(o);
        }
        auto const& info = hiptensor::rockebackend::compilerInfo();
        if(!info.available)
        {
            std::fprintf(stderr,
                         "rocKE pack: no COMGR library to compile the pack: %s\n",
                         info.reason.c_str());
            return 2;
        }
        rocke_llvm_flavor_t flavor = rocke_llvm_flavor_from_name(info.flavor.c_str());
        if(!rocke_llvm_flavor_is_known(flavor))
        {
            std::fprintf(stderr, "rocKE pack: unknown LLVM flavor %s\n", info.flavor.c_str());
            return 2;
        }
        std::string compilerId = o.compilerId.empty() ? info.id : o.compilerId;
        std::string rockeVersion
            = o.rockeVersion.empty()
                  ? std::string("rocke ") + rocke_engine_version() + "; build " + rocke_build_id()
                  : o.rockeVersion;
        int jobs = o.jobs;
        if(jobs <= 0)
        {
            // the CPUs this process may run on (a batch job's cgroup can be far below the
            // host's count), as the Python reference does
            int cpus = int(std::thread::hardware_concurrency());
#if !defined(_WIN32)
            cpu_set_t set;
            if(sched_getaffinity(0, sizeof set, &set) == 0)
            {
                cpus = CPU_COUNT(&set);
            }
#endif
            jobs = std::max(1, cpus - 1);
        }
        fs::path cache
            = o.cacheDir.empty() ? fs::path(o.out) / ".rocke_pack_cache" : fs::path(o.cacheDir);
        bool ownCache = o.cacheDir.empty();
        fs::create_directories(cache);
        std::printf("rocKE pack: compiler=%s flavor=%s jobs=%d\n",
                    info.id.c_str(),
                    info.flavor.c_str(),
                    jobs);
        std::fflush(stdout);

        int status = 0;
        for(auto const& arch : o.arches)
        {
            auto    t0 = std::chrono::steady_clock::now();
            Context ctx;
            ctx.arch       = arch;
            ctx.isa        = hiptensor::rockebackend::isaName(arch);
            ctx.flavor     = flavor;
            ctx.compilerId = info.id;
            int numCus     = o.numCus > 0 ? o.numCus : rocke_arch_num_cus(arch.c_str());

            std::vector<Descriptor> descs;
            try
            {
                collectDescriptors(ctx, descs);
            }
            catch(std::exception const& e)
            {
                std::fprintf(stderr, "rocKE pack ERROR %s: %s\n", arch.c_str(), e.what());
                status = 2;
                continue;
            }
            if(!o.only.empty())
            {
                std::regex re(o.only);
                descs.erase(std::remove_if(descs.begin(),
                                           descs.end(),
                                           [&](Descriptor const& d) {
                                               return !std::regex_search(d.family + " " + d.label,
                                                                         re);
                                           }),
                            descs.end());
            }

            // compile every distinct kernel that is not cached yet
            std::vector<Kernel const*> work;
            std::set<std::string>      keys;
            for(auto const& d : descs)
            {
                for(auto const& k : d.kernels)
                {
                    if(keys.insert(k.key).second && !fs::exists(cache / (k.key + ".co")))
                    {
                        fs::remove(cache / (k.key + ".err"));
                        work.push_back(&k);
                    }
                }
            }
            if(!compileAll(ctx, work, cache, jobs, o.compileTimeout))
            {
                std::fprintf(stderr, "rocKE pack: compile failed\n");
                return 3;
            }

            bool                               errors = false;
            std::map<std::string, std::string> extraRoutes;
            std::map<std::string, PackEntry>   byName;
            size_t                             skipped = 0;
            for(auto& d : descs)
            {
                if(!d.applies)
                {
                    ++skipped;
                    continue;
                }
                for(auto& k : d.kernels)
                {
                    std::vector<uint8_t> co;
                    if(!readFile(cache / (k.key + ".co"), co))
                    {
                        std::vector<uint8_t> log;
                        readFile(cache / (k.key + ".err"), log);
                        std::fprintf(stderr,
                                     "rocKE pack ERROR %s %s %s: %.*s\n",
                                     arch.c_str(),
                                     d.family.c_str(),
                                     d.label.c_str(),
                                     int(log.size()),
                                     reinterpret_cast<char const*>(log.data()));
                        errors = true;
                        continue;
                    }
                    PackEntry   e;
                    std::string why;
                    if(!hiptensor::rockepack::readKernelMd(co, k.name, e.md, why))
                    {
                        std::fprintf(stderr,
                                     "rocKE pack ERROR %s %s: %s\n",
                                     arch.c_str(),
                                     k.name.c_str(),
                                     why.c_str());
                        errors = true;
                        continue;
                    }
                    e.name  = k.name;
                    e.attrs = k.attrs;
                    e.co    = std::move(co);
                    auto it = byName.find(e.name);
                    if(it != byName.end())
                    {
                        if(it->second.co != e.co)
                        {
                            std::fprintf(stderr,
                                         "rocKE pack ERROR %s: two kernels named %s\n",
                                         arch.c_str(),
                                         e.name.c_str());
                            status = 2;
                        }
                        continue;
                    }
                    byName.emplace(e.name, std::move(e));
                }
                extraRoutes.insert(d.routes.begin(), d.routes.end());
            }
            if(errors)
            {
                status = 2;
                continue;
            }
            std::vector<PackEntry> kernels;
            for(auto& kv : byName)
            {
                kernels.push_back(std::move(kv.second));
            }
            auto routes = routesFor(arch, numCus);
            for(auto const& kv : extraRoutes)
            {
                routes[kv.first] = kv.second;
            }
            std::map<std::string, std::string> models;
            try
            {
                models = selectModels(o.selectModels, arch);
            }
            catch(std::exception const& e)
            {
                std::fprintf(stderr, "%s\n", e.what());
                return 2;
            }
            for(auto const& kv : models)
            {
                routes[kv.first] = kv.second;
            }
            fs::path dir = fs::path(o.out) / arch;
            fs::create_directories(dir);
            std::string packId, why;
            size_t      size = 0;
            if(!hiptensor::rockepack::writePack(dir.string(),
                                                arch,
                                                kernels,
                                                routes,
                                                compilerId,
                                                rockeVersion,
                                                packId,
                                                size,
                                                why))
            {
                std::fprintf(stderr, "rocKE pack ERROR %s: %s\n", arch.c_str(), why.c_str());
                status = 2;
                continue;
            }
            size_t gemmModelFiles = 0;
            if(!o.gemmModels.empty())
            {
                try
                {
                    gemmModelFiles = writeGemmModels(o.gemmModels, dir, arch);
                }
                catch(std::exception const& e)
                {
                    std::fprintf(stderr, "rocKE pack ERROR %s: %s\n", arch.c_str(), e.what());
                    status = 2;
                    continue;
                }
            }
            std::string modelNames;
            for(auto const& kv : models)
            {
                modelNames += (modelNames.empty() ? "" : ", ") + kv.first.substr(13);
            }
            double wall
                = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf(
                "rocKE pack %s: %zu kernels (%zu descriptors do not apply), %zu bytes, pack id "
                "%.16s, %zu compiled, wall %.0f s, selection models: %s, GETT planner models: "
                "%zu files\n",
                arch.c_str(),
                kernels.size(),
                skipped,
                size,
                packId.c_str(),
                work.size(),
                wall,
                modelNames.empty() ? "none" : modelNames.c_str(),
                gemmModelFiles);
            for(auto const& d : descs)
            {
                if(!d.applies)
                {
                    std::printf("  not applicable: %s %s: %s\n",
                                d.family.c_str(),
                                d.label.c_str(),
                                d.reason.c_str());
                }
            }
            std::fflush(stdout);
        }
        if(ownCache)
        {
            std::error_code ec;
            fs::remove_all(cache, ec);
        }
        return status;
    }
} // namespace

int main(int argc, char** argv)
{
    uint16_t probe = 1;
    if(*reinterpret_cast<uint8_t*>(&probe) != 1)
    {
        std::fprintf(stderr, "rocKE pack: the pack format is little-endian\n");
        return 2;
    }
    Options o = parse(argc, argv);
    try
    {
        return run(o);
    }
    catch(std::exception const& e)
    {
        std::fprintf(stderr, "rocKE pack ERROR: %s\n", e.what());
        return 2;
    }
}
