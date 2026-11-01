/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C++ GEMM dispatcher (rocke/gemm_dispatch.h): the learned selector of
 * dispatch/gemm/selector.py (LightGBM text models walked by TreeEnsemble, the feature vector,
 * the gated candidate sets of RCR and native requests, the margin / guard decision and
 * dispatch_override), the model loaders and dispatch/gemm/plan.py predicted_us.
 */
#include "gemm_dispatch_internal.h"

#include "rocke/json_dom.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

namespace rocke
{
namespace gemm_dispatch
{

/* ------------------------------------------------------------- TreeEnsemble */

namespace
{

std::string strip(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\n\r\f\v");
    size_t b = s.find_last_not_of(" \t\n\r\f\v");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

std::vector<std::string> split_ws(const std::string& s)
{
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string t;
    while(is >> t)
        out.push_back(t);
    return out;
}

bool parse_ints(const std::string& s, std::vector<int>* out)
{
    out->clear();
    for(const auto& t : split_ws(s))
    {
        char* end = nullptr;
        long v = std::strtol(t.c_str(), &end, 10);
        if(end == t.c_str() || *end != '\0')
            return false;
        out->push_back((int)v);
    }
    return true;
}

bool parse_floats(const std::string& s, std::vector<double>* out)
{
    out->clear();
    for(const auto& t : split_ws(s))
    {
        char* end = nullptr;
        double v = std::strtod(t.c_str(), &end);
        if(end == t.c_str() || *end != '\0')
            return false;
        out->push_back(v);
    }
    return true;
}

/* numpy's pairwise summation of float64 (DOUBLE_pairwise_sum, blocks of 128, 8 lanes) */
double pairwise_sum(const double* a, size_t n)
{
    if(n < 8)
    {
        double res = 0.0;
        for(size_t i = 0; i < n; ++i)
            res += a[i];
        return res;
    }
    if(n <= 128)
    {
        double r[8];
        for(int j = 0; j < 8; ++j)
            r[j] = a[j];
        size_t i = 8;
        for(; i < n - (n % 8); i += 8)
            for(int j = 0; j < 8; ++j)
                r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for(; i < n; ++i)
            res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

} // namespace

bool TreeEnsemble::from_text(const std::string& text, TreeEnsemble* out, std::string* why)
{
    TreeEnsemble e;
    std::map<std::string, std::string> cur;
    bool open = false;
    auto flush = [&]() -> bool {
        if(!open)
            return true;
        Tree t;
        auto get = [&](const char* k) {
            auto it = cur.find(k);
            return it == cur.end() ? std::string() : it->second;
        };
        std::string nl = get("num_leaves");
        t.num_leaves = nl.empty() ? 1 : std::atoi(nl.c_str());
        std::vector<int> split_feature, decision_type, left, right;
        std::vector<double> threshold;
        if(!parse_ints(get("split_feature"), &split_feature)
           || !parse_floats(get("threshold"), &threshold)
           || !parse_ints(get("decision_type"), &decision_type)
           || !parse_ints(get("left_child"), &left) || !parse_ints(get("right_child"), &right)
           || !parse_floats(get("leaf_value"), &t.leaf_value))
            return detail::set_why(why, "malformed tree");
        if(t.num_leaves > 1)
        {
            size_t n = split_feature.size();
            if(threshold.size() != n || decision_type.size() != n || left.size() != n
               || right.size() != n || n == 0)
                return detail::set_why(why, "malformed tree");
            for(size_t i = 0; i < n; ++i)
            {
                int d = decision_type[i];
                if(d & 1)
                    return detail::set_why(why, "categorical splits are not supported");
                if(split_feature[i] < 0 || left[i] >= (int)n || right[i] >= (int)n
                   || (left[i] < 0 && ~left[i] >= (int)t.leaf_value.size())
                   || (right[i] < 0 && ~right[i] >= (int)t.leaf_value.size()))
                    return detail::set_why(why, "malformed tree");
                Node nd;
                nd.threshold = threshold[i];
                nd.feature = split_feature[i];
                nd.left = left[i];
                nd.right = right[i];
                nd.missing_zero = ((d >> 2) & 3) == 1;
                nd.default_left = (d & 2) != 0;
                t.nodes.push_back(nd);
                e.num_features_ = std::max(e.num_features_, (size_t)split_feature[i] + 1);
            }
        }
        e.trees_.push_back(std::move(t));
        cur.clear();
        open = false;
        return true;
    };
    bool in_tree = false;
    bool have_names = false;
    std::istringstream is(text);
    std::string raw;
    while(std::getline(is, raw))
    {
        std::string line = strip(raw);
        if(line.rfind("feature_names=", 0) == 0 && !have_names)
        {
            e.feature_names = split_ws(line.substr(line.find('=') + 1));
            have_names = !e.feature_names.empty();
        }
        else if(line.rfind("Tree=", 0) == 0)
        {
            if(!flush())
                return false;
            in_tree = true;
            open = true;
        }
        else if(line.rfind("end of trees", 0) == 0)
        {
            if(!flush())
                return false;
            in_tree = false;
            break;
        }
        else if(in_tree && line.find('=') != std::string::npos)
        {
            size_t p = line.find('=');
            cur[line.substr(0, p)] = line.substr(p + 1);
        }
    }
    if(!flush())
        return false;
    if(e.trees_.empty())
        return detail::set_why(why, "model has no trees");
    *out = std::move(e);
    return true;
}

double TreeEnsemble::leaf(const Tree& t, const double* x) const
{
    const Node* nodes = t.nodes.data();
    int n = 0;
    while(n >= 0)
    {
        const Node& nd = nodes[n];
        double v = x[nd.feature];
        if(nd.missing_zero && std::fabs(v) <= 1e-35)
            n = nd.default_left ? nd.left : nd.right;
        else
            n = v <= nd.threshold ? nd.left : nd.right;
    }
    return t.leaf_value[~n];
}

double TreeEnsemble::predict(const std::vector<double>& x) const
{
    if(x.size() < num_features_)
        return std::numeric_limits<double>::quiet_NaN();
    double s = 0.0;
    for(const Tree& t : trees_)
    {
        if(t.num_leaves <= 1)
        {
            s += t.leaf_value.empty() ? 0.0 : t.leaf_value[0];
            continue;
        }
        s += leaf(t, x.data());
    }
    return s;
}

double TreeEnsemble::predict_packed(const std::vector<double>& x) const
{
    if(x.size() < num_features_)
        return std::numeric_limits<double>::quiet_NaN();
    double cst = 0.0;
    thread_local std::vector<double> leaves;
    leaves.clear();
    leaves.reserve(trees_.size());
    for(const Tree& t : trees_)
    {
        if(t.num_leaves <= 1)
        {
            cst += t.leaf_value.empty() ? 0.0 : t.leaf_value[0];
            continue;
        }
        leaves.push_back(leaf(t, x.data()));
    }
    return pairwise_sum(leaves.data(), leaves.size()) + cst;
}

/* ----------------------------------------------------------- SelectorModel */

namespace
{
/* selector.SELECTOR_MIN_MARGIN (generated) */
const std::map<std::string, double> k_min_margin = {
#define ROCKE_GD_MARGIN(cls, value) {cls, value},
#include "gemm_dispatch_consts.inc"
#undef ROCKE_GD_MARGIN
};

std::string origin_class(const std::string& origin)
{
    auto starts = [&](const char* p) { return origin.rfind(p, 0) == 0; };
    if(starts("library"))
        return "library";
    if(starts("neighbour"))
        return "neighbour";
    if(starts("form"))
        return "form";
    if(starts("skinny"))
        return "skinny";
    return "registered";
}
} // namespace

bool SelectorModel::knows_skinny() const
{
    return feature_version == "r2c-v2" && families_skinny;
}

double SelectorModel::margin_for(const std::string& cls) const
{
    double fitted;
    auto it = margins.find(cls);
    if(it != margins.end())
        fitted = it->second;
    else
    {
        auto d = margins.find("default");
        fitted = d != margins.end() ? d->second : 0.03;
    }
    auto f = k_min_margin.find(cls);
    auto d = k_min_margin.find("default");
    double floor
        = f != k_min_margin.end() ? f->second : (d != k_min_margin.end() ? d->second : 0.0);
    return std::max(fitted, floor);
}

double SelectorModel::origin_margin(const std::string& origin) const
{
    auto it = margins.find("origin:" + origin_class(origin));
    return it == margins.end() ? 0.0 : it->second;
}

bool SelectorModel::guard_enabled() const
{
    return !bags.empty() && guard_enabled_flag;
}

/* ------------------------------------------------------------------ loading */

namespace
{

struct JsonDoc
{
    rocke_arena_t a{};
    jd_val_t* root = nullptr;
    JsonDoc()
    {
        rocke_arena_init(&a, (size_t)1 << 16);
    }
    ~JsonDoc()
    {
        rocke_arena_destroy(&a);
    }
    bool parse(const std::string& text, std::string* why)
    {
        char err[512] = {0};
        root = rocke_json_parse(text.c_str(), &a, err, sizeof err);
        if(root == nullptr)
            return detail::set_why(why, std::string("JSON: ") + err);
        return true;
    }
};

std::vector<std::string> jstrs(const jd_val_t* v)
{
    std::vector<std::string> out;
    if(v == nullptr || v->kind != JD_ARR)
        return out;
    for(int i = 0; i < v->arr_len; ++i)
        if(v->arr[i]->kind == JD_STR)
            out.push_back(v->arr[i]->str);
    return out;
}

FieldValue jfield(const jd_val_t* v)
{
    FieldValue f;
    if(v == nullptr || v->kind == JD_NULL)
        return f;
    if(v->kind == JD_BOOL)
    {
        f.kind = FieldValue::BOOL;
        f.b = v->b;
    }
    else if(v->kind == JD_NUM)
    {
        if(std::floor(v->num) == v->num && std::fabs(v->num) < 9.0e15)
        {
            f.kind = FieldValue::INT;
            f.i = (int64_t)v->num;
        }
        else
        {
            f.kind = FieldValue::FLOAT;
            f.f = v->num;
        }
    }
    else if(v->kind == JD_STR)
    {
        f.kind = FieldValue::STR;
        f.s = v->str;
    }
    return f;
}

bool load_selector(Models::Reader read,
                   void* rctx,
                   const std::string& arch,
                   const std::string& kind,
                   SelectorModel* out,
                   bool* present,
                   std::string* why)
{
    *present = false;
    std::string stem = (kind == "rcr" ? "gemm_rcr_" : "gemm_native_") + arch;
    std::string txt, meta;
    if(!read(rctx, stem + ".model.txt", &txt) || !read(rctx, stem + ".meta.json", &meta))
        return true;
    JsonDoc doc;
    if(!doc.parse(meta, why))
        return false;
    const char* fv = rocke_jstr(rocke_jget(doc.root, "feature_version"));
    const char* ma = rocke_jstr(rocke_jget(doc.root, "arch"));
    std::string fvs = fv ? fv : "";
    bool ok_version = kind == "rcr" ? (fvs == "g4-v1" || fvs == "r2c-v2") : fvs == "r2n-v1";
    if(!ok_version || ma == nullptr || arch != ma)
        return true;
    SelectorModel m;
    m.arch = arch;
    m.kind = kind;
    m.feature_version = fvs;
    m.features = jstrs(rocke_jget(doc.root, "features"));
    if(!TreeEnsemble::from_text(txt, &m.ensemble, why))
        return false;
    if(!m.ensemble.feature_names.empty() && m.ensemble.feature_names != m.features)
        return detail::set_why(why, stem + ".model.txt: feature names disagree with the sidecar");
    const jd_val_t* don = rocke_jget(doc.root, "default_on");
    m.default_on = don == nullptr || don->kind != JD_BOOL ? true : don->b;
    const jd_val_t* fam = rocke_jget(doc.root, "families");
    const jd_val_t* sk = fam ? rocke_jget(fam, "skinny") : nullptr;
    m.families_skinny
        = sk != nullptr && ((sk->kind == JD_BOOL && sk->b) || (sk->kind == JD_NUM && sk->num != 0));
    const jd_val_t* mg = rocke_jget(doc.root, "margins");
    if(mg != nullptr && mg->kind == JD_OBJ)
        for(int i = 0; i < mg->obj_len; ++i)
        {
            double d;
            if(rocke_jnum(mg->obj[i].val, &d))
                m.margins[mg->obj[i].key] = d;
        }
    m.library = jstrs(rocke_jget(doc.root, "library"));
    const jd_val_t* nl = rocke_jget(doc.root, "neighbour_levers");
    m.has_neighbour_levers = nl != nullptr && nl->kind != JD_NULL;
    m.neighbour_levers = jstrs(nl);
    const jd_val_t* td = rocke_jget(doc.root, "trait_defaults");
    if(td != nullptr && td->kind == JD_OBJ)
        for(int i = 0; i < td->obj_len; ++i)
            m.meta_trait_defaults[td->obj[i].key] = jfield(td->obj[i].val);
    if(!detail::make_feature_plan(&m, why))
        return false;
    const jd_val_t* g = rocke_jget(doc.root, "guard");
    if(g != nullptr && g->kind == JD_OBJ)
    {
        m.guard_bags = jstrs(rocke_jget(g, "bags"));
        const jd_val_t* en = rocke_jget(g, "enabled");
        m.guard_enabled_flag = en == nullptr || en->kind != JD_BOOL ? true : en->b;
        double mgn = 0.0;
        if(rocke_jnum(rocke_jget(g, "min_gain"), &mgn))
            m.guard_min_gain = mgn;
    }
    for(const auto& b : m.guard_bags)
    {
        std::string bt;
        if(!read(rctx, b, &bt))
            return detail::set_why(why, b + ": guard bag named by the sidecar is missing");
        TreeEnsemble be;
        if(!TreeEnsemble::from_text(bt, &be, why))
            return false;
        if(!be.feature_names.empty() && be.feature_names != m.features)
            return detail::set_why(why, b + ": feature names disagree with the sidecar");
        m.bags.push_back(std::move(be));
    }
    *out = std::move(m);
    *present = true;
    return true;
}

const std::vector<std::string> k_skinny_features
    = {"a942",     "bf16",   "a_mn",  "b_mn",   "ld_odd", "k_odd",  "lM",     "lN",
       "lK",       "lsmall", "lbig",  "m_le_n", "ma",     "na",     "wm",     "wn",
       "wk",       "kc",     "pf",    "lsplit", "tile_m", "tile_n", "ltiles", "lwaves",
       "waves_cu", "lsteps", "pad_m", "pad_n",  "lload",  "lws",    "regs",   "lds_kb",
       "u16",      "mnp",    "ru8",   "lflop",  "lbytes"};

bool load_skinny(Models::Reader read,
                 void* rctx,
                 const std::string& arch,
                 SkinnyModel* out,
                 bool* present,
                 std::string* why)
{
    *present = false;
    std::string meta;
    if(!read(rctx, "skinny_cfg_" + arch + ".meta.json", &meta))
        return true;
    JsonDoc doc;
    if(!doc.parse(meta, why))
        return false;
    const char* fv = rocke_jstr(rocke_jget(doc.root, "feature_version"));
    const char* ma = rocke_jstr(rocke_jget(doc.root, "arch"));
    if(fv == nullptr || std::string(fv) != "r3sk-v1" || ma == nullptr || arch != ma)
        return true;
    if(jstrs(rocke_jget(doc.root, "features")) != k_skinny_features)
        return detail::set_why(why, "skinny_cfg sidecar: feature list disagrees with this tree");
    std::vector<std::string> names;
    const char* mn = rocke_jstr(rocke_jget(doc.root, "model"));
    names.push_back(mn ? mn : "");
    for(const auto& b : jstrs(rocke_jget(doc.root, "bags")))
        names.push_back(b);
    std::vector<TreeEnsemble> models;
    for(const auto& n : names)
    {
        std::string t;
        if(n.empty() || !read(rctx, n, &t))
            return detail::set_why(why, n + ": model file named by the sidecar is missing");
        TreeEnsemble e;
        if(!TreeEnsemble::from_text(t, &e, why))
            return false;
        if(!e.feature_names.empty() && e.feature_names != k_skinny_features)
            return detail::set_why(why, n + ": feature names disagree with this tree");
        models.push_back(std::move(e));
    }
    if(models.size() < 2)
        return true;
    SkinnyModel m;
    m.arch = arch;
    m.model = std::move(models[0]);
    for(size_t i = 1; i < models.size(); ++i)
        m.bags.push_back(std::move(models[i]));
    const jd_val_t* menu = rocke_jget(doc.root, "menu");
    if(menu != nullptr && menu->kind == JD_ARR)
        for(int i = 0; i < menu->arr_len; ++i)
        {
            std::vector<int64_t> row;
            const jd_val_t* r = menu->arr[i];
            if(r->kind == JD_ARR)
                for(int j = 0; j < r->arr_len; ++j)
                    row.push_back((int64_t)r->arr[j]->num);
            m.menu.push_back(row);
        }
    *out = std::move(m);
    *present = true;
    return true;
}

bool file_reader(void* ctx, const std::string& name, std::string* text)
{
    std::string path = *(const std::string*)ctx + "/" + name;
    std::ifstream f(path, std::ios::binary);
    if(!f)
        return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *text = ss.str();
    return true;
}

} // namespace

bool Models::load(Reader read, void* ctx, const std::string& arch, std::string* why)
{
    static std::atomic<uint64_t> next_serial{1};
    serial = next_serial.fetch_add(1);
    bool present = false;
    SelectorModel m;
    if(!load_selector(read, ctx, arch, "rcr", &m, &present, why))
        return false;
    if(present)
        rcr[arch] = std::move(m);
    if(!load_selector(read, ctx, arch, "native", &m, &present, why))
        return false;
    if(present)
        native[arch] = std::move(m);
    SkinnyModel s;
    if(!load_skinny(read, ctx, arch, &s, &present, why))
        return false;
    if(present)
        skinny[arch] = std::move(s);
    return true;
}

bool Models::load_dir(const std::string& dir, const std::string& arch, std::string* why)
{
    std::string d = dir;
    return load(file_reader, &d, arch, why);
}

namespace detail
{

/* ---------------------------------------------------------------- features */

double roofline_cycles(int64_t M, int64_t N, int64_t K)
{
    const double cus = 256.0, flops = 4096.0, bytes = 3300.0;
    double tc = 2.0 * (double)M * (double)N * (double)K / (cus * flops);
    double tm = 2.0 * (double)(M * K + N * K + M * N) / bytes;
    return std::max(tc, tm);
}

namespace
{

double ln(double x)
{
    return std::log(std::max(x, 1e-30));
}

double enc(const FieldValue& v)
{
    switch(v.kind)
    {
    case FieldValue::NONE:
        return -1.0;
    case FieldValue::BOOL:
        return v.b ? 1.0 : 0.0;
    case FieldValue::INT:
        return (double)v.i;
    case FieldValue::FLOAT:
        return v.f;
    case FieldValue::STR:
        if(v.s == "mem")
            return 0.0;
        if(v.s == "compv3")
            return 1.0;
        if(v.s == "compv4")
            return 2.0;
        if(v.s == "intrawave")
            return 0.0;
        if(v.s == "interwave")
            return 1.0;
        if(v.s == "default")
            return 0.0;
        if(v.s == "cshuffle")
            return 1.0;
        return v.s.empty() ? 0.0 : 1.0;
    }
    return 0.0;
}

/* the "full" dict of feature_dict: tile geometry + every trait field (+ layout) */
struct Full
{
    Tile tile;
    std::vector<FieldValue> trait; /* TraitSpec order */
    std::string layout;
    bool has_layout = false;
};

FieldValue full_trait(const Full& f, const char* name)
{
    int i = trait_field_index(name);
    return i < 0 ? FieldValue{} : f.trait[i];
}

struct FeatureCtx
{
    enum Kind
    {
        PROBLEM,
        DERIVED,
        LAYOUT,
        SKINNY,
        TRAIT,
        META_DEFAULT,
        ZERO
    };
};

const std::vector<std::string> k_problem
    = {"p_ln_m", "p_ln_n",  "p_ln_k",   "p_ln_mn",   "p_ln_min_mn",  "p_ln_max_mn", "p_ln_aspect",
       "p_bf16", "p_m_a8",  "p_m_a64",  "p_m_a256",  "p_n_a8",       "p_n_a64",     "p_n_a256",
       "p_k_a8", "p_k_a64", "p_k_a256", "p_ln_roof", "p_comp_share", "p_cold"};
const std::vector<std::string> k_derived = {"d_tm",           "d_tn",
                                            "d_tk",           "d_wm",
                                            "d_wn",           "d_wk",
                                            "d_atm",          "d_atn",
                                            "d_atk",          "d_nwave",
                                            "d_tiles_m",      "d_tiles_n",
                                            "d_ln_tiles",     "d_split",
                                            "d_ln_nwg",       "d_occ",
                                            "d_waves",        "d_last_fill",
                                            "d_wave_eff",     "d_fill_m",
                                            "d_fill_n",       "d_fill",
                                            "d_ln_kiters",    "d_ktail",
                                            "d_kslice_tail",  "d_ai",
                                            "d_lds_kb",       "d_acc_regs",
                                            "d_ln_tcomp",     "d_ln_tmem",
                                            "d_ln_tq",        "d_ln_tq_over_roof",
                                            "d_tiles_per_cu", "d_persist_ratio",
                                            "d_pad_mn"};
const std::vector<std::string> k_layout = {"x_a_mn", "x_b_mn", "x_n_mn", "x_mn_dtl", "x_mn_vgpr"};
const std::vector<std::string> k_skinny
    = {"s_fam", "s_ma",    "s_na",     "s_wm",    "s_wn",     "s_wk",       "s_kc",
       "s_pf",  "s_wpe",   "s_ntws",   "s_bload", "s_kml",    "s_kalign",   "s_rsc1",
       "s_ru",  "s_split", "s_ln_kps", "s_regs",  "s_ln_nwg", "s_wave_fill"};

} // namespace

/* the feature plan of a model (false: a feature name this tree does not compute) */
bool make_feature_plan(SelectorModel* m, std::string* why)
{
    auto pos = [](const std::vector<std::string>& v, const std::string& n) {
        auto i = std::find(v.begin(), v.end(), n);
        return i == v.end() ? -1 : (int)(i - v.begin());
    };
    m->feature_plan.clear();
    m->feature_meta.clear();
    for(const auto& n : m->features)
    {
        int p;
        if((p = pos(k_problem, n)) >= 0)
            m->feature_plan.push_back({FeatureCtx::PROBLEM, p});
        else if((p = pos(k_derived, n)) >= 0)
            m->feature_plan.push_back({FeatureCtx::DERIVED, p});
        else if((p = pos(k_layout, n)) >= 0)
            m->feature_plan.push_back({FeatureCtx::LAYOUT, p});
        else if((p = pos(k_skinny, n)) >= 0)
            m->feature_plan.push_back({FeatureCtx::SKINNY, p});
        else if(n.rfind("tr_", 0) == 0)
        {
            std::string f = n.substr(3);
            int fi = trait_field_index(f);
            if(fi >= 0)
                m->feature_plan.push_back({FeatureCtx::TRAIT, fi});
            else
            {
                auto md = m->meta_trait_defaults.find(f);
                if(md != m->meta_trait_defaults.end())
                {
                    m->feature_plan.push_back(
                        {FeatureCtx::META_DEFAULT, (int)m->feature_meta.size()});
                    m->feature_meta.push_back(md->second);
                }
                else
                    m->feature_plan.push_back({FeatureCtx::ZERO, 0});
            }
        }
        else
            return set_why(why, "unknown selector feature '" + n + "'");
    }
    return true;
}

namespace
{

void problem_features(const std::string& dtype, int64_t M, int64_t N, int64_t K, double* p)
{
    const double cus = 256.0, flops = 4096.0, bytes = 3300.0;
    double tc = 2.0 * (double)M * (double)N * (double)K / (cus * flops);
    double tmm = 2.0 * (double)(M * K + N * K + M * N) / bytes;
    p[0] = ln((double)M);
    p[1] = ln((double)N);
    p[2] = ln((double)K);
    p[3] = ln((double)M * (double)N);
    p[4] = ln((double)std::min(M, N));
    p[5] = ln((double)std::max(M, N));
    p[6] = ln((double)M) - ln((double)N);
    p[7] = (dtype == "bf16" || dtype == "bfloat16") ? 1.0 : 0.0;
    p[8] = (double)(pmod(M, 8) == 0);
    p[9] = (double)(pmod(M, 64) == 0);
    p[10] = (double)(pmod(M, 256) == 0);
    p[11] = (double)(pmod(N, 8) == 0);
    p[12] = (double)(pmod(N, 64) == 0);
    p[13] = (double)(pmod(N, 256) == 0);
    p[14] = (double)(pmod(K, 8) == 0);
    p[15] = (double)(pmod(K, 64) == 0);
    p[16] = (double)(pmod(K, 256) == 0);
    p[17] = ln(std::max(tc, tmm));
    p[18] = tc / (tc + tmm);
    p[19] = 0.0;
}

void derived_features(int64_t M, int64_t N, int64_t K, const Full& full, double* d)
{
    const double cus = 256.0, lds = 160.0 * 1024, flops = 4096.0, bytes_clk = 3300.0;
    double tm = (double)full.tile.tile_m;
    double tn = (double)full.tile.tile_n;
    double tk = (double)full.tile.tile_k;
    double wm = (double)full.tile.warp_m;
    double wn = (double)full.tile.warp_n;
    double wk = (double)full.tile.warp_k;
    double nwave = std::max(1.0, wm * wn * wk);
    double tiles_m = std::ceil((double)M / tm);
    double tiles_n = std::ceil((double)N / tn);
    double tiles = tiles_m * tiles_n;
    double split = (double)std::max((int64_t)1, (int64_t)enc(full_trait(full, "split_k")));
    double persist
        = (double)std::max((int64_t)0, (int64_t)enc(full_trait(full, "persistent_ctas")));
    int64_t stages = (int64_t)enc(full_trait(full, "dtl_stages"));
    FieldValue pipe = full_trait(full, "pipeline");
    double nbuf = stages >= 2 ? (double)stages
                              : ((pipe.kind == FieldValue::STR && pipe.s == "mem") ? 1.0 : 2.0);
    double lds_bytes = (tm + tn) * tk * 2.0 * nbuf;
    double acc = tm * tn / (nwave * 64.0);
    double vgpr = acc + (tm + tn) * tk * 2.0 / (nwave * 64.0 * 4.0) + 32.0;
    double wps = std::min(8.0, std::max(0.0, std::floor(512.0 / std::max(vgpr, 1.0))));
    double occ = std::max(
        1.0, std::min(std::floor(wps * 4.0 / nwave), std::floor(lds / std::max(lds_bytes, 1.0))));
    double nwg = tiles * split;
    double slots = cus * occ;
    double waves, last, wave_eff;
    if(persist > 0)
    {
        double pool = std::max(1.0, std::min(persist, slots));
        waves = std::ceil(nwg / pool);
        last = (nwg - (waves - 1) * pool) / pool;
        wave_eff = nwg / (waves * pool) * std::min(1.0, pool / slots);
    }
    else
    {
        waves = std::ceil(nwg / slots);
        last = (nwg - (waves - 1) * slots) / slots;
        wave_eff = nwg / (waves * slots);
    }
    double fill_m = (double)M / (tiles_m * tm);
    double fill_n = (double)N / (tiles_n * tn);
    double kslice = std::ceil((double)K / split);
    double kiters = std::ceil(kslice / tk);
    double t_comp = waves * occ * 2.0 * tm * tn * kiters * tk / flops;
    double bytes_ = 2.0 * (double)(M * K + N * K + M * N)
                    + (split > 1 ? 4.0 * (double)M * (double)N * split : 0.0);
    double t_mem = bytes_ / bytes_clk;
    double t_q = std::max(t_comp, t_mem);
    double roof = roofline_cycles(M, N, K);
    FieldValue pm = full_trait(full, "pad_m"), pn = full_trait(full, "pad_n");
    auto truthy = [](const FieldValue& v) {
        switch(v.kind)
        {
        case FieldValue::NONE:
            return false;
        case FieldValue::BOOL:
            return v.b;
        case FieldValue::INT:
            return v.i != 0;
        case FieldValue::FLOAT:
            return v.f != 0.0;
        case FieldValue::STR:
            return !v.s.empty();
        }
        return false;
    };
    double pad_mn = (truthy(pm) ? 1.0 : 0.0) + (truthy(pn) ? 1.0 : 0.0);
    int64_t itk = (int64_t)tk;
    int64_t isk = (int64_t)(split * tk);
    d[0] = tm;
    d[1] = tn;
    d[2] = tk;
    d[3] = wm;
    d[4] = wn;
    d[5] = wk;
    d[6] = (double)full.tile.warp_tile_m;
    d[7] = (double)full.tile.warp_tile_n;
    d[8] = (double)full.tile.warp_tile_k;
    d[9] = nwave;
    d[10] = tiles_m;
    d[11] = tiles_n;
    d[12] = ln(tiles);
    d[13] = split;
    d[14] = ln(nwg);
    d[15] = occ;
    d[16] = waves;
    d[17] = last;
    d[18] = wave_eff;
    d[19] = fill_m;
    d[20] = fill_n;
    d[21] = fill_m * fill_n;
    d[22] = ln(kiters);
    d[23] = (itk != 0 && pmod(K, itk)) ? 1.0 : 0.0;
    d[24] = (isk != 0 && pmod(K, isk)) ? 1.0 : 0.0;
    d[25] = tm * tn / (tm + tn);
    d[26] = lds_bytes / 1024.0;
    d[27] = acc;
    d[28] = ln(t_comp);
    d[29] = ln(t_mem);
    d[30] = ln(t_q);
    d[31] = ln(t_q) - ln(roof);
    d[32] = nwg / cus;
    d[33] = persist > 0 ? persist / tiles : 0.0;
    d[34] = pad_mn;
}

void layout_features(const Full& full, double* x)
{
    std::string lay = full.has_layout ? full.layout : "RCR";
    for(auto& c : lay)
        c = (char)toupper((unsigned char)c);
    if(lay.empty())
        lay = "RCR";
    bool a_mn = lay.substr(0, 1) == "C";
    bool b_mn = lay.size() > 1 && lay.substr(1, 1) == "R";
    bool mn = a_mn || b_mn;
    auto truthy = [](const FieldValue& v) {
        return (v.kind == FieldValue::BOOL && v.b) || (v.kind == FieldValue::INT && v.i != 0)
               || (v.kind == FieldValue::STR && !v.s.empty())
               || (v.kind == FieldValue::FLOAT && v.f != 0.0);
    };
    bool ring = truthy(full_trait(full, "direct_to_lds"))
                && truthy(full_trait(full, "dtl_prefetch"))
                && (int64_t)enc(full_trait(full, "dtl_stages")) >= 2;
    x[0] = a_mn ? 1.0 : 0.0;
    x[1] = b_mn ? 1.0 : 0.0;
    x[2] = (a_mn ? 1.0 : 0.0) + (b_mn ? 1.0 : 0.0);
    x[3] = (mn && ring) ? 1.0 : 0.0;
    x[4] = (mn && !ring) ? 1.0 : 0.0;
}

void skinny_family_features(int64_t M, int64_t N, int64_t K, const Skinny* s, double* f)
{
    if(s == nullptr)
    {
        for(int i = 0; i < 20; ++i)
            f[i] = -1.0;
        f[0] = 0.0;
        return;
    }
    const double cus = 256.0;
    int64_t ma = s->wave_m_atoms, na = s->wave_n_atoms;
    int64_t wm = s->warp_m, wn = s->warp_n, wk = s->warp_k;
    int64_t kc = s->kc, pf = s->prefetch;
    int64_t split = std::max((int64_t)1, s->split);
    int64_t tm = 16 * ma * wm, tn = 16 * na * wn;
    int64_t steps = cdiv(K, 4 * kc);
    int64_t kps = cdiv(steps, split);
    int64_t nwg = cdiv(M, tm) * cdiv(N, tn) * split;
    int64_t waves = nwg * wm * wn * wk;
    int64_t acc = ma * na * 4;
    int64_t frag = (ma + na) * fdiv(kc, 2);
    double slots = cus * 32.0;
    f[0] = 1.0;
    f[1] = (double)ma;
    f[2] = (double)na;
    f[3] = (double)wm;
    f[4] = (double)wn;
    f[5] = (double)wk;
    f[6] = (double)kc;
    f[7] = (double)pf;
    f[8] = (double)s->waves_per_eu;
    f[9] = s->nt_ws ? 1.0 : 0.0;
    f[10] = s->b_load == "global" ? 1.0 : (s->b_load == "global_nt" ? 2.0 : 0.0);
    f[11] = s->kmap == "lane" ? 1.0 : 0.0;
    f[12] = (double)s->kalign;
    f[13] = s->red_sync == "sc1" ? 1.0 : 0.0;
    f[14] = (double)s->red_unroll;
    f[15] = (double)split;
    f[16] = ln((double)kps);
    f[17] = (double)(acc + (pf + 1) * frag);
    f[18] = ln((double)nwg);
    f[19] = std::min(1.0, (double)waves / slots);
}

} // namespace

std::vector<double> feature_row(const SelectorModel& m,
                                const std::string& dtype,
                                int64_t M,
                                int64_t N,
                                int64_t K,
                                const Spec& spec)
{
    Full full;
    if(spec.skinny)
    {
        const Skinny& s = spec.s;
        Trait t; /* the TraitSpec defaults */
        full.trait = trait_values(t);
        int64_t ma = s.wave_m_atoms, na = s.wave_n_atoms;
        int64_t wm = s.warp_m, wn = s.warp_n, wk = s.warp_k;
        int64_t tm = 16 * ma * wm, tn = 16 * na * wn;
        full.tile.tile_m = tm;
        full.tile.tile_n = tn;
        full.tile.tile_k = 4 * s.kc * wk;
        full.tile.warp_m = wm;
        full.tile.warp_n = wn;
        full.tile.warp_k = wk;
        full.tile.warp_tile_m = 16;
        full.tile.warp_tile_n = 16;
        full.tile.warp_tile_k = 32;
        FieldValue sk;
        sk.kind = FieldValue::INT;
        sk.i = std::max((int64_t)1, s.split);
        full.trait[trait_field_index("split_k")] = sk;
        FieldValue pm, pn;
        pm.kind = pn.kind = FieldValue::BOOL;
        pm.b = pmod(M, tm) != 0;
        pn.b = pmod(N, tn) != 0;
        full.trait[trait_field_index("pad_m")] = pm;
        full.trait[trait_field_index("pad_n")] = pn;
        full.has_layout = false;
    }
    else
    {
        full.tile = spec.u.tile;
        full.trait = trait_values(spec.u.trait);
        std::string lay = spec.u.data.layout.empty() ? "RCR" : spec.u.data.layout;
        for(auto& c : lay)
            c = (char)toupper((unsigned char)c);
        full.layout = lay;
        full.has_layout = true;
    }
    double p[20], d[35], x[5], s[20];
    problem_features(dtype, M, N, K, p);
    derived_features(M, N, K, full, d);
    layout_features(full, x);
    skinny_family_features(M, N, K, spec.skinny ? &spec.s : nullptr, s);
    std::vector<double> row;
    row.reserve(m.feature_plan.size());
    for(const auto& e : m.feature_plan)
    {
        switch(e.first)
        {
        case FeatureCtx::PROBLEM:
            row.push_back(p[e.second]);
            break;
        case FeatureCtx::DERIVED:
            row.push_back(d[e.second]);
            break;
        case FeatureCtx::LAYOUT:
            row.push_back(x[e.second]);
            break;
        case FeatureCtx::SKINNY:
            row.push_back(s[e.second]);
            break;
        case FeatureCtx::TRAIT:
            row.push_back(enc(full.trait[e.second]));
            break;
        case FeatureCtx::META_DEFAULT:
            row.push_back(enc(m.feature_meta[e.second]));
            break;
        case FeatureCtx::ZERO:
            row.push_back(0.0);
            break;
        }
    }
    return row;
}

const SelectorModel*
    load_model(const Context& ctx, const std::string& arch, const std::string& kind)
{
    if(ctx.models == nullptr)
        return nullptr;
    const auto& m = kind == "rcr" ? ctx.models->rcr : ctx.models->native;
    auto it = m.find(arch);
    return it == m.end() ? nullptr : &it->second;
}

/* ---------------------------------------------------------- the selector */

namespace
{

std::string upper(std::string s)
{
    for(auto& c : s)
        c = (char)toupper((unsigned char)c);
    return s;
}

struct SCand
{
    std::string origin;
    std::string cname;
    Spec spec;
    bool is_rule_pick = false;
};

bool spec_key_eq(const Spec& a, const Spec& b)
{
    if(a.skinny != b.skinny)
        return false;
    if(a.skinny)
        return a.s == b.s;
    return a.u.tile == b.u.tile && a.u.trait == b.u.trait;
}

bool rcr(const Request& r)
{
    return upper(r.layout) == "RCR";
}

/* selector._gate ("" = None) */
bool gate_ok(const Request& req, const UGemm& spec)
{
    if(!universal_spec_supported(spec, req.arch, nullptr))
        return false;
    if(!request_shape_supported(req, spec, nullptr))
        return false;
    return launch_shape_error(spec, req.M, req.N, req.K).empty();
}

UGemm fix_pads(UGemm s, int64_t M, int64_t N, int64_t K)
{
    s.trait.pad_m = s.trait.pad_m || pmod(M, s.tile.tile_m) != 0;
    s.trait.pad_n = s.trait.pad_n || pmod(N, s.tile.tile_n) != 0;
    s.trait.pad_k = s.trait.pad_k || pmod(K, s.tile.tile_k) != 0;
    return s;
}

bool ld_any_generated_ok(const Request& req, const UGemm& spec)
{
    if(req.arch != "gfx950")
        return false;
    const Trait& tr = spec.trait;
    if(!tr.mn_ld_any)
        return false;
    return tr.direct_to_lds && tr.dtl_tight_rsrc;
}

/* selector.generated_spec_error (true: an error) */
bool generated_spec_error(const Request& req, const UGemm& spec, const UGemm* parent)
{
    int64_t M = req.M, N = req.N, K = req.K;
    const Tile& t = spec.tile;
    const Trait& tr = spec.trait;
    if(pmod(K, 8) || ((pmod(M, 8) || pmod(N, 8)) && !ld_any_generated_ok(req, spec)))
        return true;
    const Tile* ptile = parent ? &parent->tile : nullptr;
    const Trait* ptr = parent ? &parent->trait : nullptr;
    struct D
    {
        int64_t dim, tile;
        bool pad;
        char p;
    } dims[2] = {{M, t.tile_m, tr.pad_m, 'm'}, {N, t.tile_n, tr.pad_n, 'n'}};
    for(const auto& d : dims)
    {
        if(pmod(d.dim, d.tile) || d.pad)
        {
            bool bounded = ptr != nullptr ? (ptr->direct_to_lds && ptr->dtl_tight_rsrc
                                             && tr.direct_to_lds && tr.dtl_tight_rsrc)
                                          : false;
            bounded = bounded
                      || (ptr != nullptr && ptr->pad_row_clamp && tr.pad_row_clamp
                          && !tr.direct_to_lds);
            bool same = ptile != nullptr && (d.p == 'm' ? ptile->tile_m : ptile->tile_n) == d.tile
                        && (d.p == 'm' ? ptr->pad_m : ptr->pad_n) == d.pad && bounded;
            if(!same)
                return true;
        }
    }
    if(tr.direct_to_lds && (pmod(K, t.tile_k) || tr.pad_k))
        if(!(ptile != nullptr && ptile->tile_k == t.tile_k && ptr->pad_k == tr.pad_k))
            return true;
    if(tr.split_k > 1)
    {
        if(tr.direct_to_lds || tr.persistent_ctas || tr.streamk)
            return true;
        bool aligned = tr.split_k_aligned && ptr != nullptr && ptr->split_k_aligned;
        if(!aligned && (tr.pad_k || pmod(K, tr.split_k * t.tile_k)))
            return true;
    }
    if(tr.streamk && !(ptr != nullptr && ptr->streamk))
        return true;
    if(tr.persistent_ctas && !(ptr != nullptr && ptr->persistent_ctas == tr.persistent_ctas))
        return true;
    return false;
}

/* SELECTOR_NEIGHBOUR_LEVERS / SELECTOR_NATIVE_NEIGHBOUR_LEVERS */
using Lever = std::pair<std::string, std::vector<FieldValue>>;

FieldValue I(int64_t v)
{
    FieldValue f;
    f.kind = FieldValue::INT;
    f.i = v;
    return f;
}
FieldValue B(bool v)
{
    FieldValue f;
    f.kind = FieldValue::BOOL;
    f.b = v;
    return f;
}

const std::vector<Lever>& neighbour_levers()
{
    static const std::vector<Lever> v = {
        {"waves_per_eu", {I(1), I(2)}},
        {"chiplet_chunk_size", {I(4), I(8), I(16), I(32)}},
        {"chiplet_wgm", {I(4), I(8), I(16)}},
        {"split_k", {I(2), I(4), I(8)}},
        {"tile_k", {I(32), I(64), I(128)}},
        {"kloop_unroll", {I(2), I(4)}},
        {"pad_fast_tile", {B(true)}},
        {"dtl_stages", {I(2), I(3)}},
    };
    return v;
}

const std::vector<Lever>& native_neighbour_levers()
{
    static const std::vector<Lever> v = {
        {"mn_chunk", {I(64), I(128), I(256)}},
        {"mn_pad", {I(64)}},
        {"persistent_ctas", {I(0)}},
        {"dtl_stages", {I(2), I(3)}},
        {"lds_swizzle", {B(true), B(false)}},
        {"kloop_unroll", {I(1)}},
    };
    return v;
}

/* selector._mutations */
std::vector<std::pair<std::string, UGemm>> mutations(const UGemm& spec,
                                                     const std::vector<Lever>& levers)
{
    std::vector<std::pair<std::string, UGemm>> out;
    for(const auto& lv : levers)
    {
        const std::string& name = lv.first;
        for(const auto& v : lv.second)
        {
            std::string lab = name + "=" + py_str(v);
            if(name == "tile_m" || name == "tile_n" || name == "tile_k" || name == "warp_m"
               || name == "warp_n" || name == "warp_k" || name == "warp_tile_m"
               || name == "warp_tile_n" || name == "warp_tile_k")
            {
                UGemm s = spec;
                int64_t cur = name == "tile_m"        ? s.tile.tile_m
                              : name == "tile_n"      ? s.tile.tile_n
                              : name == "tile_k"      ? s.tile.tile_k
                              : name == "warp_m"      ? s.tile.warp_m
                              : name == "warp_n"      ? s.tile.warp_n
                              : name == "warp_k"      ? s.tile.warp_k
                              : name == "warp_tile_m" ? s.tile.warp_tile_m
                              : name == "warp_tile_n" ? s.tile.warp_tile_n
                                                      : s.tile.warp_tile_k;
                if(field_eq(I(cur), v))
                    continue;
                int64_t nv = v.kind == FieldValue::BOOL ? (v.b ? 1 : 0) : v.i;
                if(name == "tile_m")
                    s.tile.tile_m = nv;
                else if(name == "tile_n")
                    s.tile.tile_n = nv;
                else if(name == "tile_k")
                    s.tile.tile_k = nv;
                else if(name == "warp_m")
                    s.tile.warp_m = nv;
                else if(name == "warp_n")
                    s.tile.warp_n = nv;
                else if(name == "warp_k")
                    s.tile.warp_k = nv;
                else if(name == "warp_tile_m")
                    s.tile.warp_tile_m = nv;
                else if(name == "warp_tile_n")
                    s.tile.warp_tile_n = nv;
                else
                    s.tile.warp_tile_k = nv;
                out.push_back({lab, s});
                continue;
            }
            if(trait_field_index(name) < 0 || field_eq(get_trait_field(spec.trait, name), v))
                continue;
            if(name == "dtl_stages" && !spec.trait.direct_to_lds)
                continue;
            UGemm s = spec;
            set_trait_field(&s.trait, name, v);
            if(name == "chiplet_wgm" || name == "chiplet_chunk_size")
                s.trait.chiplet_swizzle = true;
            out.push_back({lab, s});
        }
    }
    return out;
}

/* selector.skinny_variant_error (true: an error) */
bool skinny_variant_error(const Request& req, const Skinny& spec)
{
    if(!skinny_is_valid(spec, req.arch, nullptr))
        return true;
    if(!skinny_launch_error(spec, req.M, req.N, req.K, spec.split).empty())
        return true;
    int64_t steps = skinny_steps(spec, req.K);
    int64_t kps = skinny_kps(spec, req.K, spec.split);
    return (spec.split - 1) * kps >= steps;
}

/* selector.library_spec (false: None or a parse failure) */
bool library_spec(const Request& req, const std::string& cfg, UGemm* out)
{
    Tile tile;
    Trait trait;
    if(!parse_config(cfg, &tile, &trait, nullptr))
        return false;
    std::string dt = (req.dtype == "fp16" || req.dtype == "f16") ? "fp16" : "bf16";
    UGemm s;
    s.name = "selector_library";
    s.tile = tile;
    s.trait = trait;
    s.data.dtype_a = dt;
    s.data.dtype_b = dt;
    s.data.dtype_c = dt;
    s.data.layout = "RCR";
    s.wave_size = 64;
    s.block_size = 0;
    s.post_init();
    *out = fix_pads(s, req.M, req.N, req.K);
    return true;
}

std::vector<SCand> selector_candidates_rcr(const Request& req,
                                           const Context& ctx,
                                           const Registry& reg,
                                           const SelectorModel& model,
                                           const std::pair<const Candidate*, Spec>* rule)
{
    std::vector<SCand> out;
    auto add = [&](const std::string& origin, const std::string& cname, const Spec& spec) {
        for(const auto& c : out)
            if(spec_key_eq(c.spec, spec))
                return;
        SCand c;
        c.origin = origin;
        c.cname = cname;
        c.spec = spec;
        c.is_rule_pick = rule != nullptr && spec_key_eq(spec, rule->second);
        out.push_back(c);
    };
    std::vector<std::pair<const Candidate*, Spec>> registered;
    if(rule != nullptr)
        registered.push_back(*rule);
    for(auto& c : registry_combos(reg, req))
        registered.push_back(c);
    bool skinny_on = model.knows_skinny();
    for(const auto& r : registered)
    {
        if(!r.second.skinny)
        {
            if(gate_ok(req, r.second.u))
                add("registered", r.first->name, r.second);
        }
        else if(skinny_on && !skinny_variant_error(req, r.second.s))
            add("registered", r.first->name, r.second);
    }
    if(skinny_on)
    {
        std::vector<SCand> bases;
        for(const auto& c : out)
            if(c.spec.skinny)
                bases.push_back(c);
        for(const auto& base : bases)
        {
            const Skinny& b = base.spec.s;
            std::vector<std::pair<std::string, Skinny>> vars;
            for(int64_t s : {1, 2, 4, 8, 16, 32, 64, 128})
                if(s != b.split)
                {
                    Skinny v = b;
                    v.split = s;
                    v.red_unroll = s >= 16 ? 8 : 4;
                    vars.push_back({"split=" + std::to_string(s), v});
                }
            for(int64_t v : {8, 16, 32})
                if(b.kc != v)
                {
                    Skinny x = b;
                    x.kc = v;
                    vars.push_back({"kc=" + std::to_string(v), x});
                }
            for(int64_t v : {1, 2, 4})
                if(b.prefetch != v)
                {
                    Skinny x = b;
                    x.prefetch = v;
                    vars.push_back({"prefetch=" + std::to_string(v), x});
                }
            for(int64_t v : {1, 2, 4, 8})
                if(b.warp_k != v)
                {
                    Skinny x = b;
                    x.warp_k = v;
                    vars.push_back({"warp_k=" + std::to_string(v), x});
                }
            for(const auto& v : vars)
                if(!skinny_variant_error(req, v.second))
                {
                    Spec sp;
                    sp.skinny = true;
                    sp.s = v.second;
                    add("skinny:" + v.first, base.cname, sp);
                }
        }
    }
    std::vector<SCand> bases;
    for(const auto& c : out)
        if(!c.spec.skinny)
            bases.push_back(c);
    std::vector<Lever> levers;
    for(const auto& l : neighbour_levers())
        if(!model.has_neighbour_levers
           || std::find(model.neighbour_levers.begin(), model.neighbour_levers.end(), l.first)
                  != model.neighbour_levers.end())
            levers.push_back(l);
    for(const auto& base : bases)
        for(auto& m : mutations(base.spec.u, levers))
        {
            UGemm s = fix_pads(m.second, req.M, req.N, req.K);
            if(generated_spec_error(req, s, &base.spec.u) || !gate_ok(req, s))
                continue;
            Spec sp;
            sp.u = s;
            add("neighbour:" + m.first, base.cname, sp);
        }
    for(const auto& cfg : model.library)
    {
        UGemm s;
        if(!library_spec(req, cfg, &s))
            continue;
        if(generated_spec_error(req, s, nullptr) || !gate_ok(req, s))
            continue;
        Spec sp;
        sp.u = s;
        add("library", "", sp);
    }
    (void)ctx;
    return out;
}

bool mn_ring(const UGemm& s)
{
    bool a, b;
    if(!mn_layout_letters(s.data.layout, &a, &b))
        return false;
    return (a || b) && uses_mn_dtl(s);
}

std::vector<SCand> native_candidates(const Request& req,
                                     const Context& ctx,
                                     const Registry& reg,
                                     const SelectorModel& model,
                                     const std::pair<const Candidate*, Spec>* rule)
{
    std::string lay = upper(req.layout);
    std::vector<SCand> out;
    auto add = [&](const std::string& origin, const std::string& cname, const Spec& spec) {
        if(spec.skinny || upper(spec.u.data.layout) != lay)
            return;
        for(const auto& c : out)
            if(spec_key_eq(c.spec, spec))
                return;
        if(!gate_ok(req, spec.u))
            return;
        SCand c;
        c.origin = origin;
        c.cname = cname;
        c.spec = spec;
        c.is_rule_pick = rule != nullptr && spec_key_eq(spec, rule->second);
        out.push_back(c);
    };
    std::vector<std::pair<const Candidate*, Spec>> registered;
    if(rule != nullptr)
        registered.push_back(*rule);
    for(auto& c : registry_combos(reg, req))
        registered.push_back(c);
    for(const auto& r : registered)
        if(!r.second.skinny)
            add("registered", r.first->name, r.second);
    /* "form": every native form of every registered RCR spec */
    Request rreq = req;
    rreq.layout = "RCR";
    for(const auto& rc : registry_combos(reg, rreq))
    {
        if(rc.second.skinny)
            continue;
        std::vector<std::pair<std::string, UGemm>> alts;
        std::string w;
        native_layout_alternatives(req, ctx, rc.second.u, &alts, &w);
        for(const auto& a : alts)
        {
            Spec sp;
            sp.u = a.second;
            add("form:" + a.first, rc.first->name, sp);
        }
    }
    std::vector<SCand> bases = out;
    const size_t nb = 8;
    if(bases.size() > nb)
    {
        std::vector<double> sc;
        for(const auto& b : bases)
            sc.push_back(
                model.ensemble.predict(feature_row(model, req.dtype, req.M, req.N, req.K, b.spec)));
        std::vector<size_t> order(bases.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(
            order.begin(), order.end(), [&](size_t a, size_t b) { return sc[a] < sc[b]; });
        std::set<size_t> keep(order.begin(), order.begin() + nb);
        std::vector<SCand> kept;
        for(size_t i = 0; i < bases.size(); ++i)
            if(keep.count(i) || bases[i].is_rule_pick)
                kept.push_back(bases[i]);
        bases = kept;
    }
    std::vector<Lever> levers;
    for(const auto* list : {&neighbour_levers(), &native_neighbour_levers()})
        for(const auto& l : *list)
            if(!model.has_neighbour_levers
               || std::find(model.neighbour_levers.begin(), model.neighbour_levers.end(), l.first)
                      != model.neighbour_levers.end())
                levers.push_back(l);
    for(const auto& base : bases)
    {
        bool ring = mn_ring(base.spec.u);
        std::vector<Lever> lv;
        for(const auto& l : levers)
            if(ring || (l.first != "mn_chunk" && l.first != "mn_pad"))
                lv.push_back(l);
        for(auto& m : mutations(base.spec.u, lv))
        {
            UGemm s = fix_pads(m.second, req.M, req.N, req.K);
            if(generated_spec_error(req, s, &base.spec.u))
                continue;
            Spec sp;
            sp.u = s;
            add("neighbour:" + m.first, base.cname, sp);
        }
    }
    return out;
}

std::string spec_class(const Spec& spec)
{
    if(spec.skinny)
        return "skinny";
    const Trait& t = spec.u.trait;
    if(t.direct_to_lds)
    {
        if(t.hb_sched || t.sia_sched)
            return "dtl:beat";
        if(t.persistent_ctas)
            return "dtl:persist";
        return "dtl";
    }
    if(t.split_k > 1)
        return "global:splitk";
    if(t.pipeline == "mem")
        return "global:mem";
    return "global";
}

/* selector.select: the chosen candidate (index into cands) */
int select_pick(const Request& req, const SelectorModel& model, const std::vector<SCand>& cands)
{
    if(cands.empty())
        return -1;
    std::vector<double> scores;
    for(const auto& c : cands)
        scores.push_back(
            model.ensemble.predict(feature_row(model, req.dtype, req.M, req.N, req.K, c.spec)));
    std::vector<size_t> order(cands.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] < scores[b]; });
    size_t best = order[0];
    double best_s = scores[best];
    int rp = -1;
    for(size_t i : order)
        if(cands[i].is_rule_pick)
        {
            rp = (int)i;
            break;
        }
    if(rp < 0)
        return (int)best;
    double rule_s = scores[rp];
    std::string cls = spec_class(cands[rp].spec);
    double margin = std::max(model.margin_for(cls), model.origin_margin(cands[best].origin));
    if((int)best == rp || rule_s - best_s <= margin)
        return rp;
    if(model.guard_enabled())
    {
        std::vector<double> r0 = feature_row(model, req.dtype, req.M, req.N, req.K, cands[rp].spec);
        std::vector<double> r1
            = feature_row(model, req.dtype, req.M, req.N, req.K, cands[best].spec);
        bool first = true;
        double mn = 0.0;
        for(const auto& b : model.bags)
        {
            double g = b.predict(r0) - b.predict(r1);
            if(first || g < mn)
                mn = g;
            first = false;
        }
        if(!model.bags.empty() && mn <= model.guard_min_gain)
            return rp;
    }
    return (int)best;
}

} // namespace

void dispatch_override(
    const Request& req, const Context& ctx, const Registry& reg, const Candidate*& cand, Spec& spec)
{
    std::string kind = rcr(req) ? "rcr" : "native";
    const SelectorModel* model = load_model(ctx, req.arch, kind);
    if(model == nullptr)
        return;
    /* selector_on_for: the model's default_on (RCR: selector_enabled; native:
     * native_selector_enabled, whose sidecar check is the same flag) */
    if(!model->default_on)
        return;
    const Candidate* rc = nullptr;
    Spec rs;
    if(!registry_select(reg, req, &rc, nullptr) || !rc->select(req, &rs, nullptr))
        return;
    bool known;
    if(rc->spec_id == "cdna_ring_split")
        known = false;
    else if(kind == "native")
        known = !rs.skinny;
    else
        known = !rs.skinny || model->knows_skinny();
    if(!known)
        return;
    std::pair<const Candidate*, Spec> rule = {rc, rs};
    std::vector<SCand> cands = kind == "rcr" ? selector_candidates_rcr(req, ctx, reg, *model, &rule)
                                             : native_candidates(req, ctx, reg, *model, &rule);
    int pick = select_pick(req, *model, cands);
    if(pick < 0)
        return;
    const SCand& p = cands[pick];
    if(p.is_rule_pick)
        return;
    for(const auto& c : reg.candidates)
    {
        if(c.name == p.cname && p.origin == "registered")
        {
            Spec own;
            if(!c.select(req, &own, nullptr))
                break;
            if(spec_key_eq(own, p.spec))
            {
                cand = &c;
                spec = own;
                return;
            }
            break;
        }
    }
    std::string target = "universal_gemm_" + reg.dtype
                         + (p.spec.skinny ? "_cdna_selected_skinny" : "_cdna_selected");
    const Candidate* t = reg.get(target);
    if(t == nullptr)
        return;
    cand = t;
    spec = p.spec;
}

} // namespace detail

using namespace detail;

bool predicted_us(const Request& req, const UGemm& spec, const Context& ctx, double* out)
{
    const SelectorModel* m = load_model(ctx, req.arch, "native");
    if(m == nullptr)
        return false;
    std::string dtype = (req.dtype == "bf16" || req.dtype == "bfloat16") ? "bf16" : "fp16";
    Spec sp;
    sp.u = spec;
    double s = m->ensemble.predict(feature_row(*m, dtype, req.M, req.N, req.K, sp));
    *out = std::exp(s + ln(roofline_cycles(req.M, req.N, req.K)));
    return true;
}

} // namespace gemm_dispatch
} // namespace rocke
