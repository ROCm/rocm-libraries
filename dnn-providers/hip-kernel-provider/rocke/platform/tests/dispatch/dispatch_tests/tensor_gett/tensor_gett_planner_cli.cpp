/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tensor_gett_planner_cli <selector_models dir> <problems.jsonl> [--gemm | --valid | --trees | --ir]
 *
 * The C++ tensor GETT planner (rocke/tensor_gett_planner.h) over a list of contraction
 * problems, one JSON object per line ({"a", "b", "d": mode lists, "ext": {mode: extent},
 * "sa" / "sb" / "sd": {mode: stride} or null, "dtype", "bilinear", "cus", "arch"}); prints one
 * line per problem in the format test_tensor_gett_planner_cpp.py builds from the Python planner.
 * With --gemm each line is a GEMM request ({"M", "N", "K", "dtype", "layout", "cus", "arch"}) and
 * the output is the GEMM dispatcher's pick and its predicted time. With --valid each line is a
 * gemm_universal spec ({"cfg": selector config string, "dtype", "layout", "M", "N", "K", "arch"})
 * and the output is "valid=<0|1> launch=<0|1|->" (is_valid_spec; for a valid spec whether
 * launch_shape_error is none).
 * With --trees each line is a feature row of one shipped model ({"arch", "model": "rcr" /
 * "native" / "skinny", "bag": -1 for the model or the bag index, "x": [...]}, or of a model file
 * {"file": path, "packed": bool}) and the output is its score (TreeEnsemble.predict; skinny /
 * packed: the packed predictor's pairwise sum), "-" when the model does not exist.
 * With --ir each line is a contraction problem and the output is the SHA-256 of the LLVM IR
 * (ROCKE_LLVM_FLAVOR) of every kernel of the planned route as the C++ kernel builders build it
 * (what hipTensor's contraction JIT compiles): "<kind>|<pack sha>;...|<GETT sha>", "none" when
 * there is no route, "pyonly" when the route uses a lever only rocKE's Python engine builds.
 */
#include "rocke/gemm_dispatch.h"
#include "rocke/json_dom.h"
#include "rocke/lower_llvm.h"
#include "rocke/tensor_gett_planner.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace gd = rocke::gemm_dispatch;
namespace gp = rocke::gett::planner;

namespace
{

std::string g17(double v)
{
    char b[64];
    snprintf(b, sizeof b, "%.17g", v);
    return b;
}

bool modes(const jd_val_t* v, std::vector<std::string>* out)
{
    out->clear();
    if(v == nullptr || v->kind != JD_ARR)
        return false;
    for(int i = 0; i < v->arr_len; ++i)
        out->push_back(v->arr[i]->str);
    return true;
}

bool modemap(const jd_val_t* v, std::map<std::string, int64_t>* out)
{
    out->clear();
    if(v == nullptr || v->kind != JD_OBJ)
        return false;
    for(int i = 0; i < v->obj_len; ++i)
        (*out)[v->obj[i].key] = (int64_t)v->obj[i].val->num;
    return true;
}

/* a string member of a JSON object ("" when absent) */
std::string jstr(const jd_val_t* obj, const char* key)
{
    const char* s = rocke_jstr(rocke_jget(obj, key));
    return s != nullptr ? std::string(s) : std::string();
}

std::string grid(const int g[3])
{
    return std::to_string(g[0]) + "," + std::to_string(g[1]) + "," + std::to_string(g[2]);
}

/* the SHA-256 of a built kernel's LLVM IR ("ERR:<why>" when it does not build or lower) */
std::string ir_sha(rocke_ir_builder_t* b, rocke_kernel_def_t* kd, const std::string& arch)
{
    if(kd == nullptr)
    {
        const char* e = rocke_ir_builder_error(b);
        return std::string("ERR:build:") + (e ? e : "");
    }
    char* text = nullptr;
    char err[512] = {0};
    if(rocke_lower_kernel_to_llvm_ex(
           kd, ROCKE_LLVM_FLAVOR_AUTO, arch.c_str(), &text, err, sizeof err)
           != ROCKE_OK
       || text == nullptr)
    {
        std::free(text);
        return std::string("ERR:lower:") + err;
    }
    std::string out = rocke::gett::sha256_hex(text);
    std::free(text);
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        std::fprintf(
            stderr,
            "usage: %s <selector_models dir> <problems.jsonl> [--gemm | --valid | --trees]\n",
            argv[0]);
        return 2;
    }
    bool gemm = argc > 3 && std::strcmp(argv[3], "--gemm") == 0;
    bool valid = argc > 3 && std::strcmp(argv[3], "--valid") == 0;
    bool trees = argc > 3 && std::strcmp(argv[3], "--trees") == 0;
    bool ir = argc > 3 && std::strcmp(argv[3], "--ir") == 0;
    std::ifstream in(argv[2]);
    if(!in)
    {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 2;
    }
    std::map<std::string, gd::Models> models;
    std::string line;
    int64_t idx = 0;
    while(std::getline(in, line))
    {
        if(line.empty())
            continue;
        rocke_arena_t arena{};
        rocke_arena_init(&arena, 1 << 16);
        char err[256] = {0};
        jd_val_t* root = rocke_json_parse(line.c_str(), &arena, err, sizeof err);
        if(root == nullptr)
        {
            std::fprintf(stderr, "line %lld: %s\n", (long long)idx, err);
            return 2;
        }
        std::string arch = jstr(root, "arch");
        double cus = 256;
        rocke_jnum(rocke_jget(root, "cus"), &cus);
        if(!valid && models.find(arch) == models.end())
        {
            std::string why;
            if(!models[arch].load_dir(argv[1], arch, &why))
            {
                std::fprintf(stderr, "models of %s: %s\n", arch.c_str(), why.c_str());
                return 2;
            }
        }
        gd::Context ctx;
        ctx.cus = (int64_t)cus;
        ctx.models = &models[arch];
        std::string out = std::to_string(idx) + " ";
        if(trees)
        {
            std::string kind = jstr(root, "model");
            double bag = -1;
            rocke_jnum(rocke_jget(root, "bag"), &bag);
            std::vector<double> x;
            const jd_val_t* xs = rocke_jget(root, "x");
            for(int i = 0; xs != nullptr && i < xs->arr_len; ++i)
                x.push_back(xs->arr[i]->num);
            const gd::Models& md = models[arch];
            const gd::TreeEnsemble* e = nullptr;
            const std::vector<gd::TreeEnsemble>* bags = nullptr;
            std::string file = jstr(root, "file");
            const jd_val_t* pk = rocke_jget(root, "packed");
            bool packed = kind == "skinny" || (pk != nullptr && pk->kind == JD_BOOL && pk->b);
            static std::map<std::string, gd::TreeEnsemble> files;
            if(!file.empty())
            {
                if(!files.count(file))
                {
                    std::ifstream f(file);
                    std::stringstream ss;
                    ss << f.rdbuf();
                    std::string why;
                    if(!gd::TreeEnsemble::from_text(ss.str(), &files[file], &why))
                    {
                        std::fprintf(stderr, "%s: %s\n", file.c_str(), why.c_str());
                        return 2;
                    }
                }
                e = &files[file];
                bag = -1;
            }
            else if(kind == "skinny" && md.skinny.count(arch))
                e = &md.skinny.at(arch).model, bags = &md.skinny.at(arch).bags;
            else if(kind == "rcr" && md.rcr.count(arch))
                e = &md.rcr.at(arch).ensemble, bags = &md.rcr.at(arch).bags;
            else if(kind == "native" && md.native.count(arch))
                e = &md.native.at(arch).ensemble, bags = &md.native.at(arch).bags;
            if(e != nullptr && bag >= 0)
                e = (size_t)bag < bags->size() ? &(*bags)[(size_t)bag] : nullptr;
            if(e == nullptr)
                out += "-";
            else
                out += g17(packed ? e->predict_packed(x) : e->predict(x));
        }
        else if(valid)
        {
            gd::UGemm u;
            std::string why;
            if(!gd::parse_config(jstr(root, "cfg"), &u.tile, &u.trait, &why))
            {
                std::fprintf(stderr, "line %lld: %s\n", (long long)idx, why.c_str());
                return 2;
            }
            std::string dt = jstr(root, "dtype");
            u.name = "selector_library";
            u.data.dtype_a = u.data.dtype_b = u.data.dtype_c = dt;
            u.data.layout = jstr(root, "layout");
            u.post_init();
            double M = 0, N = 0, K = 0;
            rocke_jnum(rocke_jget(root, "M"), &M);
            rocke_jnum(rocke_jget(root, "N"), &N);
            rocke_jnum(rocke_jget(root, "K"), &K);
            /* launch_shape_error is defined on valid specs only */
            if(!gd::is_valid_spec(u, arch, nullptr))
                out += "valid=0 launch=-";
            else
                out += std::string("valid=1 launch=")
                       + (gd::launch_shape_error(u, (int64_t)M, (int64_t)N, (int64_t)K).empty()
                              ? "1"
                              : "0");
        }
        else if(gemm)
        {
            gd::Request r;
            double v;
            rocke_jnum(rocke_jget(root, "M"), &v);
            r.M = (int64_t)v;
            rocke_jnum(rocke_jget(root, "N"), &v);
            r.N = (int64_t)v;
            rocke_jnum(rocke_jget(root, "K"), &v);
            r.K = (int64_t)v;
            r.arch = arch;
            r.dtype = jstr(root, "dtype");
            r.layout = jstr(root, "layout");
            gd::Pick pick;
            std::string why;
            if(!gd::dispatch_gemm(r, ctx, &pick, &why))
                out += "none";
            else
            {
                out += pick.candidate + " " + gd::config_string(pick.spec);
                if(pick.spec.skinny)
                    out += " L" + pick.spec.s.layout + " d" + pick.spec.s.dtype;
                else
                {
                    out += " L" + pick.spec.u.data.layout + " bs"
                           + std::to_string(pick.spec.u.block_size);
                    double t;
                    out += " t="
                           + (gd::predicted_us(r, pick.spec.u, ctx, &t) ? g17(t)
                                                                        : std::string("-"));
                }
            }
        }
        else
        {
            gp::Problem p;
            modes(rocke_jget(root, "a"), &p.modes_a);
            modes(rocke_jget(root, "b"), &p.modes_b);
            modes(rocke_jget(root, "d"), &p.modes_d);
            modemap(rocke_jget(root, "ext"), &p.extents);
            p.has_strides_a = modemap(rocke_jget(root, "sa"), &p.strides_a);
            p.has_strides_b = modemap(rocke_jget(root, "sb"), &p.strides_b);
            p.has_strides_d = modemap(rocke_jget(root, "sd"), &p.strides_d);
            p.dtype = jstr(root, "dtype");
            const jd_val_t* bl = rocke_jget(root, "bilinear");
            p.bilinear = bl != nullptr && bl->kind == JD_BOOL && bl->b;
            gp::RoutePlan plan;
            std::string why;
            rocke::gett::TensorGettSpec c;
            if(!gp::plan_tensor_gett(p, arch, ctx, &plan, &why))
                out += "none";
            else if(ir && !gp::to_builder_spec(plan.gett, &c, &why))
                out += "pyonly";
            else if(ir)
            {
                out += plan.kind + "|";
                for(size_t i = 0; i < plan.packs.size(); ++i)
                {
                    rocke_ir_builder_t b;
                    std::memset(&b, 0, sizeof b);
                    out += (i ? ";" : "")
                           + ir_sha(
                               &b,
                               rocke::gett::build_tensor_pack(&b, plan.packs[i].spec, arch.c_str()),
                               arch);
                    rocke_ir_builder_free(&b);
                }
                rocke_ir_builder_t b;
                std::memset(&b, 0, sizeof b);
                out += "|" + ir_sha(&b, rocke::gett::build_tensor_gett(&b, c, arch.c_str()), arch);
                rocke_ir_builder_free(&b);
            }
            else
            {
                out += plan.kind + "|" + plan.candidate + "|" + plan.gett.kernel_name() + "|"
                       + grid(plan.grid) + "|" + std::to_string(plan.block) + "|"
                       + std::to_string(plan.workspace_bytes) + "|";
                for(size_t i = 0; i < plan.packs.size(); ++i)
                {
                    const auto& s = plan.packs[i];
                    out += (i ? ";" : "") + s.operand + ":" + s.spec.kernel_name() + ":"
                           + grid(s.grid) + ":" + std::to_string(s.block) + ":"
                           + std::to_string(s.workspace_bytes);
                }
                out += "|" + (plan.has_t_mn ? g17(plan.t_mn) : std::string("-"));
                out += "|" + (plan.has_t_kc ? g17(plan.t_kc) : std::string("-"));
                /* the builder spec: the same kernel name, or the Python-only levers */
                rocke::gett::TensorGettSpec c;
                std::string bw;
                if(!gp::to_builder_spec(plan.gett, &c, &bw))
                    out += " builder=pyonly";
                else if(c.kernel_name() != plan.gett.kernel_name())
                    out += " builder=DIFF:" + c.kernel_name();
                else
                    out += " builder=same";
                out += " src=" + plan.source;
            }
        }
        std::cout << out << "\n";
        rocke_arena_destroy(&arena);
        ++idx;
    }
    return 0;
}
