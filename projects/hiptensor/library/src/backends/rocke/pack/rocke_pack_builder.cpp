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
// The pack is the one pack/make_rocke_pack.py writes for the same rocKE sources: same kernels,
// code objects, kernel records, routes and selection models; only the two header strings
// (compiler id, rocKE version) name this builder's toolchain unless --compiler-id /
// --rocke-version override them.
//
// Kernel families built natively: gemm (the emulated-f32 GEMM bodies with their operand
// prologue: rocke_gemm_cf32_body, rocke_build_universal_gemm) and gett64 (rocke_build_gett_f64
// through the gett_f64 spec policy). The other families (ew, reduce, gett_plan) are read from a
// kernel manifest of rocKE portable IR (--ir-manifest) until their builders have C++ twins; the
// kernels still go through this builder's lowering, compile and pack writer.
//
// Usage:
//     rocke_pack_builder --arch gfx950 [--arch gfx942 ...] --out DIR [--jobs N]
//                        [--cache-dir DIR] [--compile-timeout S] [--num-cus N]
//                        [--select-models DIR] [--ir-manifest DIR] [--only REGEX]
//                        [--compiler-id STR] [--rocke-version STR]

#include <algorithm>
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
#include <vector>

#if !defined(_WIN32)
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../rocke_comgr.hpp"
#include "rocke_pack_format.hpp"

#include "rocke/arch_num_cus.h"
#include "rocke/compile_options.h"
#include "rocke/instance_gemm_cf32.h"
#include "rocke/instance_gemm_prologue.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/instance_gett_f64.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/json_dom.h"
#include "rocke/lower_llvm.h"
#include "rocke/rocke_build_id.h"

namespace fs = std::filesystem;

using hiptensor::rockepack::Attrs;
using hiptensor::rockepack::AttrValue;
using hiptensor::rockepack::KernelMd;
using hiptensor::rockepack::PackEntry;

namespace
{
    // ---------------------------------------------------------------------------------------------
    // Descriptor tables (make_rocke_pack.py)
    // ---------------------------------------------------------------------------------------------

    // The arches with GEMM bodies (GEMM_ARCHES) and f64 GETT kernels (GETT64_CLASS_ARCHES).
    char const* const kGemmArches[]   = {"gfx950"};
    char const* const kGett64Arches[] = {"gfx90a", "gfx942", "gfx950"};

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
    };

    // Lowers a built kernel and fills in its compile inputs. False with `why` set on failure.
    bool lower(Context const& ctx, rocke_kernel_def_t const* kd, Kernel& k, std::string& why)
    {
        if(rocke_attr_get_bool(&kd->attrs, ROCKE_COMGR_FRESH_PROCESS_ATTR, false))
        {
            why = k.name + " needs a fresh compile process (history-dependent codegen)";
            return false;
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

    // ---- gett64: rocke_build_gett_f64 through the gett_f64 spec policy (make_rocke_pack.py
    // _gett64_spec) ----

    bool gett64Spec(std::string const&     arch,
                    bool                   cplx,
                    std::string const&     cls,
                    char                   dclass,
                    bool                   bilinear,
                    bool                   kblock,
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
        rocke_gett_f64_knobs_t   knobs{};
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

    void gett64Descriptors(Context const& ctx, std::vector<Descriptor>& out)
    {
        if(!contains(kGett64Arches, ctx.arch))
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
                            d.family = "gett64";
                            d.label  = std::string("cplx=") + (cplx ? "1" : "0") + " cls=" + cls
                                      + " dclass=" + dclass + " bilinear=" + (bilinear ? "1" : "0")
                                      + " kblock=" + (kblock ? "1" : "0");
                            rocke_gett_f64_spec_t spec;
                            if(!gett64Spec(
                                   ctx.arch, cplx, cls, dclass, bilinear, kblock, spec, d.reason))
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
                                throw std::runtime_error("gett64 " + d.label + ": "
                                                         + rocke_ir_builder_error(&b.b));
                            }
                            Kernel k;
                            k.name  = kd->name;
                            k.attrs = {
                                {"family", std::string("gett64")},
                                {"abi", std::string("rocke-gett-f64/v2")},
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
                                throw std::runtime_error("gett64 " + d.label + ": " + why);
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
    // cf32_body) with their fused operand prologue (make_rocke_pack.py _gemm_spec) ----

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
                    if(route != "h16")
                    {
                        pro.src      = "f32";
                        pro.recipe   = "x6";
                        pro.split_at = "use";
                        pro.cplx_b   = route == "cf32";
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

    // ---- manifest families: kernels exported as rocKE portable IR ----

    std::string slurp(fs::path const& p)
    {
        std::ifstream f(p, std::ios::binary);
        if(!f)
        {
            throw std::runtime_error("cannot read " + p.string());
        }
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }

    struct Arena
    {
        rocke_arena_t a{};
        Arena()
        {
            rocke_arena_init(&a, size_t(1) << 20);
        }
        ~Arena()
        {
            rocke_arena_destroy(&a);
        }
    };

    std::string jstr(jd_val_t const* o, char const* key)
    {
        char const* s = rocke_jstr(rocke_jget(o, key));
        if(s == nullptr)
        {
            throw std::runtime_error(std::string("manifest: missing string ") + key);
        }
        return s;
    }

    // Reads <dir>/<arch>.json: the descriptors of the families in `families`.
    void manifestDescriptors(Context const&               ctx,
                             fs::path const&              dir,
                             std::set<std::string> const& families,
                             std::vector<Descriptor>&     out)
    {
        fs::path    path = dir / (ctx.arch + ".json");
        std::string text = slurp(path);
        Arena       arena;
        char        err[512] = {0};
        jd_val_t*   root     = rocke_json_parse(text.c_str(), &arena.a, err, sizeof err);
        if(root == nullptr)
        {
            throw std::runtime_error(path.string() + ": " + err);
        }
        if(jstr(root, "arch") != ctx.arch)
        {
            throw std::runtime_error(path.string() + ": manifest of another arch");
        }
        jd_val_t const* descs = rocke_jget(root, "descriptors");
        if(descs == nullptr || descs->kind != JD_ARR)
        {
            throw std::runtime_error(path.string() + ": no descriptors");
        }
        // the families the manifest holds every descriptor of
        std::set<std::string> listed;
        jd_val_t const*       jf = rocke_jget(root, "families");
        for(int i = 0; jf != nullptr && jf->kind == JD_ARR && i < jf->arr_len; ++i)
        {
            if(rocke_jstr(jf->arr[i]) != nullptr)
            {
                listed.insert(rocke_jstr(jf->arr[i]));
            }
        }
        for(auto const& f : families)
        {
            if(!listed.count(f))
            {
                throw std::runtime_error(path.string() + ": the manifest has no " + f + " kernels");
            }
        }
        for(int i = 0; i < descs->arr_len; ++i)
        {
            jd_val_t const* jd     = descs->arr[i];
            std::string     family = jstr(jd, "family");
            if(!families.count(family))
            {
                continue;
            }
            Descriptor d;
            d.family  = family;
            d.label   = jstr(jd, "label");
            d.applies = jstr(jd, "status") == "ok";
            if(!d.applies)
            {
                d.reason = jstr(jd, "reason");
                out.push_back(std::move(d));
                continue;
            }
            jd_val_t const* ks = rocke_jget(jd, "kernels");
            for(int j = 0; ks != nullptr && ks->kind == JD_ARR && j < ks->arr_len; ++j)
            {
                jd_val_t const* jk = ks->arr[j];
                Kernel          k;
                k.name             = jstr(jk, "name");
                jd_val_t const* ja = rocke_jget(jk, "attrs");
                if(ja == nullptr || ja->kind != JD_OBJ)
                {
                    throw std::runtime_error(path.string() + ": " + k.name + " has no attrs");
                }
                for(int m = 0; m < ja->obj_len; ++m)
                {
                    jd_val_t const* v = ja->obj[m].val;
                    if(v->kind == JD_STR)
                    {
                        k.attrs[ja->obj[m].key] = std::string(v->str);
                    }
                    else if(v->kind == JD_NUM && v->num == double(int64_t(v->num)))
                    {
                        k.attrs[ja->obj[m].key] = int64_t(v->num);
                    }
                    else
                    {
                        throw std::runtime_error(path.string() + ": " + k.name + " attr "
                                                 + ja->obj[m].key + " is not an int or a string");
                    }
                }
                std::string ir = slurp(dir / jstr(jk, "ir"));
                Builder     b;
                if(rocke_ir_builder_init(&b.b, k.name.c_str()) != ROCKE_OK)
                {
                    throw std::runtime_error("builder init failed");
                }
                rocke_kernel_def_t* kd = nullptr;
                if(rocke_ir_parse(ir.c_str(), &b.b, &kd) != ROCKE_OK || kd == nullptr)
                {
                    throw std::runtime_error(path.string() + ": " + k.name + ": IR parse failed: "
                                             + rocke_ir_builder_error(&b.b));
                }
                if(k.name != kd->name)
                {
                    throw std::runtime_error(path.string() + ": " + k.name
                                             + ": the IR names another kernel");
                }
                std::string why;
                if(!lower(ctx, kd, k, why))
                {
                    throw std::runtime_error(why);
                }
                d.kernels.push_back(std::move(k));
            }
            jd_val_t const* jr = rocke_jget(jd, "routes");
            for(int m = 0; jr != nullptr && jr->kind == JD_OBJ && m < jr->obj_len; ++m)
            {
                char const* v = rocke_jstr(jr->obj[m].val);
                if(v == nullptr)
                {
                    throw std::runtime_error(path.string() + ": route values are strings");
                }
                d.routes[jr->obj[m].key] = v;
            }
            out.push_back(std::move(d));
        }
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
            if(!ascii || text.rfind("v1;schema=", 0) != 0 || text.find('\n') != std::string::npos
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
        std::string              irManifest;
        std::string              compilerId;
        std::string              rockeVersion;
    };

    [[noreturn]] void usage(char const* msg)
    {
        std::fprintf(
            stderr,
            "%s\nusage: rocke_pack_builder --arch GFX [--arch GFX ...] --out DIR [--jobs N]\n"
            "       [--cache-dir DIR] [--compile-timeout S] [--num-cus N]\n"
            "       [--select-models DIR] [--ir-manifest DIR] [--only REGEX]\n"
            "       [--compiler-id STR] [--rocke-version STR]\n",
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
            else if(a == "--ir-manifest")
                o.irManifest = val();
            else if(a == "--compiler-id")
                o.compilerId = val();
            else if(a == "--rocke-version")
                o.rockeVersion = val();
            else
                usage(("unknown argument " + a).c_str());
        }
        if(o.arches.empty() || o.out.empty())
        {
            usage("--arch and --out are required");
        }
        return o;
    }

    int run(Options const& o)
    {
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
            // host's count), as make_rocke_pack.py does
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

        // Families without a C++ builder yet: from the kernel manifest.
        std::set<std::string> manifestFamilies = {"ew", "reduce", "gett_plan"};

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
                if(o.irManifest.empty())
                {
                    throw std::runtime_error("families ew, reduce and gett_plan need --ir-manifest "
                                             "(their builders have no C++ twin yet)");
                }
                // make_rocke_pack.py's descriptor order: ew / reduce, gemm, gett_plan, gett64
                std::vector<Descriptor> man;
                manifestDescriptors(ctx, o.irManifest, manifestFamilies, man);
                for(auto& m : man)
                {
                    if(m.family != "gett_plan")
                    {
                        descs.push_back(std::move(m));
                    }
                }
                gemmDescriptors(ctx, descs);
                for(auto& m : man)
                {
                    if(m.family == "gett_plan")
                    {
                        descs.push_back(std::move(m));
                    }
                }
                gett64Descriptors(ctx, descs);
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
            std::string modelNames;
            for(auto const& kv : models)
            {
                modelNames += (modelNames.empty() ? "" : ", ") + kv.first.substr(13);
            }
            double wall
                = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf(
                "rocKE pack %s: %zu kernels (%zu descriptors do not apply), %zu bytes, pack id "
                "%.16s, %zu compiled, wall %.0f s, selection models: %s\n",
                arch.c_str(),
                kernels.size(),
                skipped,
                size,
                packId.c_str(),
                work.size(),
                wall,
                modelNames.empty() ? "none" : modelNames.c_str());
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
