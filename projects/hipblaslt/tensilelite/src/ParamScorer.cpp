// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/ParamScorer.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>

#if defined(TENSILE_MSGPACK)
#include <msgpack.hpp>
#endif

namespace TensileLite
{
    namespace ParamScorer
    {
        using FeatureFn = std::function<double(uint64_t, uint64_t, uint64_t, uint64_t)>;

        struct Layer
        {
            size_t              in = 0, out = 0;
            std::vector<double> W; // out x in, row-major
            std::vector<double> b;
            bool                relu = false;
        };

        struct Head
        {
            std::string name;
            size_t      offset = 0, size = 0;
        };

        struct TestVector
        {
            uint64_t            m = 0, n = 0, k = 0, b = 0;
            std::vector<double> x;
            std::vector<int>    order;
        };

        struct Model
        {
            std::string              path;
            int                      version = 0;
            std::string              problemKey;
            std::vector<std::string> featureNames;
            std::vector<FeatureFn>   featureFns;
            std::vector<double>      mu, sd;
            std::vector<Layer>       layers;
            std::vector<Head>        heads;
            std::vector<double>      headWeight;
            double                   unkLogp = -20.0;
            std::vector<std::vector<int>> solutions;
            std::vector<std::string>      solutionNames;
            std::vector<std::string>      solutionNamesBuilt;
            int                           topkDefault = 1;
            std::vector<TestVector>       testVectors;

            std::vector<double> features(uint64_t m, uint64_t n, uint64_t k, uint64_t b) const
            {
                std::vector<double> x(featureFns.size());
                for(size_t i = 0; i < featureFns.size(); ++i)
                {
                    double s = sd[i] != 0.0 ? sd[i] : 1.0;
                    x[i]     = (featureFns[i](m, n, k, b) - mu[i]) / s;
                }
                return x;
            }

            std::vector<double> scoresFromFeatures(std::vector<double> x) const
            {
                for(auto const& L : layers)
                {
                    std::vector<double> y(L.out);
                    for(size_t o = 0; o < L.out; ++o)
                    {
                        double        acc = L.b[o];
                        double const* w   = L.W.data() + o * L.in;
                        for(size_t i = 0; i < L.in; ++i)
                            acc += w[i] * x[i];
                        y[o] = (L.relu && acc < 0.0) ? 0.0 : acc;
                    }
                    x.swap(y);
                }
                // log-softmax per head
                std::vector<std::vector<double>> logp(heads.size());
                for(size_t h = 0; h < heads.size(); ++h)
                {
                    auto const& H  = heads[h];
                    double      mx = -std::numeric_limits<double>::infinity();
                    for(size_t j = 0; j < H.size; ++j)
                        mx = std::max(mx, x[H.offset + j]);
                    double se = 0.0;
                    for(size_t j = 0; j < H.size; ++j)
                        se += std::exp(x[H.offset + j] - mx);
                    double lse = mx + std::log(se);
                    logp[h].resize(H.size);
                    for(size_t j = 0; j < H.size; ++j)
                        logp[h][j] = x[H.offset + j] - lse;
                }
                std::vector<double> scores(solutions.size(), 0.0);
                for(size_t r = 0; r < solutions.size(); ++r)
                {
                    double s = 0.0;
                    for(size_t h = 0; h < heads.size(); ++h)
                    {
                        int idx = solutions[r][h];
                        s += headWeight[h] * (idx >= 0 ? logp[h][idx] : unkLogp);
                    }
                    scores[r] = s;
                }
                return scores;
            }

            std::vector<int> order(std::vector<double> const& scores) const
            {
                std::vector<int> rows(scores.size());
                std::iota(rows.begin(), rows.end(), 0);
                std::stable_sort(rows.begin(), rows.end(), [&](int a, int b) {
                    return scores[a] > scores[b];
                });
                return rows;
            }
        };

        namespace
        {
            constexpr char const* kTag = "[TENSILE_PARAM] ";

            bool envFlag(char const* name)
            {
                char const* v = std::getenv(name);
                return v && *v && std::string(v) != "0";
            }

            uint64_t ceilDiv(uint64_t a, uint64_t c)
            {
                return c == 0 ? 0 : (a + c - 1) / c;
            }

            // Variable accessor: m, n, k, b.
            std::function<uint64_t(uint64_t, uint64_t, uint64_t, uint64_t)> var(char c)
            {
                switch(c)
                {
                case 'm':
                    return [](uint64_t m, uint64_t, uint64_t, uint64_t) { return m; };
                case 'n':
                    return [](uint64_t, uint64_t n, uint64_t, uint64_t) { return n; };
                case 'k':
                    return [](uint64_t, uint64_t, uint64_t k, uint64_t) { return k; };
                default:
                    return [](uint64_t, uint64_t, uint64_t, uint64_t b) { return b; };
                }
            }

            // Feature-name grammar (see MODEL_FORMAT.md of the campaign):
            //   v | log2<vs> | log2_<v>_div_<w> | ceil_<v>_div_<C> | log2_ceil_<v>_div_<C>
            //   <v>mod<C> | <v>mod<C>_is0 | <v>_(gt|ge|lt|le|eq)<C>
            //   tiles_<A>x<B> | log2_tiles_<A>x<B> | waveeff_<A>x<B>_cu<C> | waves_<A>x<B>_cu<C>
            bool compileFeature(std::string const& name, FeatureFn& fn)
            {
                std::smatch mt;
                auto        u64 = [](std::string const& s) { return std::stoull(s); };
                if(std::regex_match(name, mt, std::regex("^([mnkb])$")))
                {
                    auto v = var(mt[1].str()[0]);
                    fn     = [v](uint64_t m, uint64_t n, uint64_t k, uint64_t b) {
                        return double(v(m, n, k, b));
                    };
                    return true;
                }
                if(std::regex_match(name, mt, std::regex("^log2([mnkb]+)$")))
                {
                    std::string vs = mt[1].str();
                    fn             = [vs](uint64_t m, uint64_t n, uint64_t k, uint64_t b) {
                        double p = 1.0;
                        for(char c : vs)
                            p *= double(var(c)(m, n, k, b));
                        return std::log2(p);
                    };
                    return true;
                }
                if(std::regex_match(name, mt, std::regex("^log2_([mnkb])_div_([mnkb])$")))
                {
                    auto v = var(mt[1].str()[0]);
                    auto w = var(mt[2].str()[0]);
                    fn     = [v, w](uint64_t m, uint64_t n, uint64_t k, uint64_t b) {
                        return std::log2(double(v(m, n, k, b)) / double(w(m, n, k, b)));
                    };
                    return true;
                }
                if(std::regex_match(name, mt, std::regex("^(log2_)?ceil_([mnkb])_div_([0-9]+)$")))
                {
                    bool     lg = mt[1].matched;
                    auto     v  = var(mt[2].str()[0]);
                    uint64_t C  = u64(mt[3].str());
                    fn          = [lg, v, C](uint64_t m, uint64_t n, uint64_t k, uint64_t b) {
                        double c = double(ceilDiv(v(m, n, k, b), C));
                        return lg ? std::log2(c) : c;
                    };
                    return C > 0;
                }
                if(std::regex_match(name, mt, std::regex("^([mnkb])mod([0-9]+)(_is0)?$")))
                {
                    auto     v   = var(mt[1].str()[0]);
                    uint64_t C   = u64(mt[2].str());
                    bool     is0 = mt[3].matched;
                    fn           = [v, C, is0](uint64_t m, uint64_t n, uint64_t k, uint64_t b) {
                        uint64_t r = v(m, n, k, b) % C;
                        return is0 ? (r == 0 ? 1.0 : 0.0) : double(r);
                    };
                    return C > 0;
                }
                if(std::regex_match(name, mt, std::regex("^([mnkb])_(gt|ge|lt|le|eq)([0-9]+)$")))
                {
                    auto        v  = var(mt[1].str()[0]);
                    std::string op = mt[2].str();
                    uint64_t    C  = u64(mt[3].str());
                    fn = [v, op, C](uint64_t m, uint64_t n, uint64_t k, uint64_t b) {
                        uint64_t x = v(m, n, k, b);
                        bool     r = op == "gt"   ? x > C
                                     : op == "ge" ? x >= C
                                     : op == "lt" ? x < C
                                     : op == "le" ? x <= C
                                                  : x == C;
                        return r ? 1.0 : 0.0;
                    };
                    return true;
                }
                if(std::regex_match(name, mt, std::regex("^(log2_)?tiles_([0-9]+)x([0-9]+)$")))
                {
                    bool     lg = mt[1].matched;
                    uint64_t A = u64(mt[2].str()), B = u64(mt[3].str());
                    fn = [lg, A, B](uint64_t m, uint64_t n, uint64_t, uint64_t b) {
                        double t = double(ceilDiv(m, A) * ceilDiv(n, B) * b);
                        return lg ? std::log2(t) : t;
                    };
                    return A > 0 && B > 0;
                }
                if(std::regex_match(
                       name, mt, std::regex("^(waveeff|waves)_([0-9]+)x([0-9]+)_cu([0-9]+)$")))
                {
                    bool     eff = mt[1].str() == "waveeff";
                    uint64_t A = u64(mt[2].str()), B = u64(mt[3].str()), C = u64(mt[4].str());
                    fn = [eff, A, B, C](uint64_t m, uint64_t n, uint64_t, uint64_t b) {
                        uint64_t t = ceilDiv(m, A) * ceilDiv(n, B) * b;
                        if(eff)
                            return double(t) / double(ceilDiv(t, C) * C);
                        return double(t) / double(C);
                    };
                    return A > 0 && B > 0 && C > 0;
                }
                return false;
            }

#if defined(TENSILE_MSGPACK)
            using Obj = msgpack::object;

            Obj const* find(Obj const& o, char const* key)
            {
                if(o.type != msgpack::type::MAP)
                    throw std::runtime_error("expected a map");
                for(uint32_t i = 0; i < o.via.map.size; ++i)
                {
                    auto const& kv = o.via.map.ptr[i];
                    if(kv.key.type == msgpack::type::STR
                       && std::string(kv.key.via.str.ptr, kv.key.via.str.size) == key)
                        return &kv.val;
                    if(kv.key.type == msgpack::type::BIN
                       && std::string(kv.key.via.bin.ptr, kv.key.via.bin.size) == key)
                        return &kv.val;
                }
                return nullptr;
            }

            Obj const& get(Obj const& o, char const* key)
            {
                auto p = find(o, key);
                if(!p)
                    throw std::runtime_error(std::string("missing key '") + key + "'");
                return *p;
            }

            double num(Obj const& o)
            {
                switch(o.type)
                {
                case msgpack::type::FLOAT32:
                case msgpack::type::FLOAT64:
                    return o.via.f64;
                case msgpack::type::POSITIVE_INTEGER:
                    return double(o.via.u64);
                case msgpack::type::NEGATIVE_INTEGER:
                    return double(o.via.i64);
                default:
                    throw std::runtime_error("expected a number");
                }
            }

            int64_t integer(Obj const& o)
            {
                double d = num(o);
                if(d != std::floor(d))
                    throw std::runtime_error("expected an integer");
                return int64_t(d);
            }

            std::string str(Obj const& o)
            {
                if(o.type == msgpack::type::STR)
                    return std::string(o.via.str.ptr, o.via.str.size);
                if(o.type == msgpack::type::BIN)
                    return std::string(o.via.bin.ptr, o.via.bin.size);
                throw std::runtime_error("expected a string");
            }

            std::vector<Obj const*> arr(Obj const& o)
            {
                if(o.type != msgpack::type::ARRAY)
                    throw std::runtime_error("expected an array");
                std::vector<Obj const*> v;
                for(uint32_t i = 0; i < o.via.array.size; ++i)
                    v.push_back(&o.via.array.ptr[i]);
                return v;
            }

            std::vector<double> nums(Obj const& o)
            {
                std::vector<double> v;
                for(auto p : arr(o))
                    v.push_back(num(*p));
                return v;
            }

            std::vector<std::string> strs(Obj const& o)
            {
                std::vector<std::string> v;
                for(auto p : arr(o))
                    v.push_back(str(*p));
                return v;
            }

            std::unique_ptr<Model> parse(std::string const& path)
            {
                std::ifstream f(path, std::ios::binary);
                if(!f)
                    throw std::runtime_error("cannot open file");
                std::string data((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
                auto        handle = msgpack::unpack(data.data(), data.size());
                Obj const&  root   = handle.get();

                auto M        = std::make_unique<Model>();
                M->path       = path;
                M->version    = int(integer(get(root, "version")));
                M->problemKey = str(get(root, "problem_key"));
                if(M->version != 1)
                    throw std::runtime_error("unsupported version " + std::to_string(M->version));

                auto const& feats = get(root, "features");
                M->featureNames   = strs(get(feats, "names"));
                M->mu             = nums(get(feats, "mu"));
                M->sd             = nums(get(feats, "sd"));
                size_t F          = M->featureNames.size();
                if(M->mu.size() != F || M->sd.size() != F)
                    throw std::runtime_error("features: names/mu/sd size mismatch");
                for(auto const& nm : M->featureNames)
                {
                    FeatureFn fn;
                    if(!compileFeature(nm, fn))
                        throw std::runtime_error("unsupported feature name '" + nm + "'");
                    M->featureFns.push_back(fn);
                }

                size_t width = F;
                for(auto lp : arr(get(root, "layers")))
                {
                    Layer L;
                    L.in        = size_t(integer(get(*lp, "in")));
                    L.out       = size_t(integer(get(*lp, "out")));
                    L.W         = nums(get(*lp, "W"));
                    L.b         = nums(get(*lp, "b"));
                    std::string act = str(get(*lp, "act"));
                    if(act != "relu" && act != "none")
                        throw std::runtime_error("unsupported activation '" + act + "'");
                    L.relu = act == "relu";
                    if(L.in != width || L.W.size() != L.in * L.out || L.b.size() != L.out)
                        throw std::runtime_error("layer " + std::to_string(M->layers.size())
                                                 + ": shape mismatch");
                    width = L.out;
                    M->layers.push_back(std::move(L));
                }
                if(M->layers.empty())
                    throw std::runtime_error("no layers");

                for(auto hp : arr(get(root, "heads")))
                {
                    Head H;
                    H.name   = str(get(*hp, "name"));
                    H.offset = size_t(integer(get(*hp, "offset")));
                    H.size   = size_t(integer(get(*hp, "size")));
                    if(H.size == 0 || H.offset + H.size > width)
                        throw std::runtime_error("head '" + H.name + "' out of range");
                    M->heads.push_back(H);
                }
                M->headWeight = nums(get(root, "head_weight"));
                if(M->heads.empty() || M->headWeight.size() != M->heads.size())
                    throw std::runtime_error("heads/head_weight size mismatch");
                if(auto p = find(root, "unk_logp"))
                    M->unkLogp = num(*p);

                for(auto rp : arr(get(root, "solutions")))
                {
                    std::vector<int> row;
                    for(auto ip : arr(*rp))
                        row.push_back(int(integer(*ip)));
                    if(row.size() != M->heads.size())
                        throw std::runtime_error("solutions row " + std::to_string(M->solutions.size())
                                                 + ": wrong number of head indices");
                    for(size_t h = 0; h < row.size(); ++h)
                        if(row[h] < -1 || row[h] >= int(M->heads[h].size))
                            throw std::runtime_error(
                                "solutions row " + std::to_string(M->solutions.size())
                                + ": head index out of range");
                    M->solutions.push_back(std::move(row));
                }
                if(M->solutions.empty())
                    throw std::runtime_error("no solutions");
                M->solutionNames = strs(get(root, "solution_names"));
                if(M->solutionNames.size() != M->solutions.size())
                    throw std::runtime_error("solution_names size != solutions size");
                if(auto p = find(root, "solution_names_built"))
                {
                    M->solutionNamesBuilt = strs(*p);
                    if(M->solutionNamesBuilt.size() != M->solutions.size())
                        throw std::runtime_error("solution_names_built size != solutions size");
                }
                if(auto p = find(root, "topk_default"))
                    M->topkDefault = int(integer(*p));

                if(auto p = find(root, "test_vectors"))
                {
                    for(auto tp : arr(*p))
                    {
                        TestVector t;
                        t.m = uint64_t(integer(get(*tp, "m")));
                        t.n = uint64_t(integer(get(*tp, "n")));
                        t.k = uint64_t(integer(get(*tp, "k")));
                        t.b = uint64_t(integer(get(*tp, "b")));
                        t.x = nums(get(*tp, "x"));
                        for(auto ip : arr(get(*tp, "order")))
                            t.order.push_back(int(integer(*ip)));
                        M->testVectors.push_back(std::move(t));
                    }
                }
                return M;
            }
#endif

            // Abort on mismatch: features within 1e-4, first 10 rows in the same
            // order except where the two rows' scores tie.
            void selfTest(Model const& M)
            {
                if(M.testVectors.empty())
                {
                    std::cerr << kTag << "SELFTEST FAILED: model has no test_vectors" << std::endl;
                    std::abort();
                }
                size_t ties = 0;
                for(size_t t = 0; t < M.testVectors.size(); ++t)
                {
                    auto const& T = M.testVectors[t];
                    auto        x = M.features(T.m, T.n, T.k, T.b);
                    if(x.size() != T.x.size())
                    {
                        std::cerr << kTag << "SELFTEST FAILED: vector " << t << " has "
                                  << T.x.size() << " features, model computes " << x.size()
                                  << std::endl;
                        std::abort();
                    }
                    for(size_t i = 0; i < x.size(); ++i)
                    {
                        if(!(std::abs(x[i] - T.x[i]) <= 1e-4))
                        {
                            std::cerr << kTag << "SELFTEST FAILED: vector " << t << " (m=" << T.m
                                      << " n=" << T.n << " k=" << T.k << " b=" << T.b
                                      << ") feature " << i << " '" << M.featureNames[i]
                                      << "': C++ " << x[i] << " vs ref " << T.x[i] << std::endl;
                            std::abort();
                        }
                    }
                    auto   scores = M.scoresFromFeatures(M.features(T.m, T.n, T.k, T.b));
                    auto   order  = M.order(scores);
                    size_t n      = std::min<size_t>(10, std::min(order.size(), T.order.size()));
                    for(size_t i = 0; i < n; ++i)
                    {
                        if(order[i] == T.order[i])
                            continue;
                        int    r = T.order[i];
                        bool   inRange = r >= 0 && size_t(r) < scores.size();
                        double a = scores[order[i]], b = inRange ? scores[r] : 0.0;
                        if(inRange && std::abs(a - b) <= 1e-6 * (1.0 + std::abs(a)))
                        {
                            ++ties;
                            continue;
                        }
                        std::cerr << kTag << "SELFTEST FAILED: vector " << t << " (m=" << T.m
                                  << " n=" << T.n << " k=" << T.k << " b=" << T.b
                                  << ") order position " << i << ": C++ row " << order[i]
                                  << " (score " << a << ") vs ref row " << r << " (score "
                                  << (inRange ? b : std::nan("")) << ")" << std::endl;
                        std::abort();
                    }
                }
                std::cerr << kTag << "SELFTEST PASSED: " << M.testVectors.size()
                          << " test vectors (features tol 1e-4, top-10 order; " << ties
                          << " tie swaps)" << std::endl;
            }

            Model const* loadOne(std::string const& pathStr)
            {
                char const* path = pathStr.c_str();
#if defined(TENSILE_MSGPACK)
                try
                {
                    auto M = parse(path);
                    std::cerr << kTag << "loaded model " << path << ": key=" << M->problemKey
                              << " features=" << M->featureNames.size()
                              << " layers=" << M->layers.size() << " heads=" << M->heads.size()
                              << " rows=" << M->solutions.size()
                              << " topk=" << topK(*M) << " test_vectors=" << M->testVectors.size()
                              << std::endl;
                    if(envFlag("TENSILE_PARAM_SELFTEST"))
                        selfTest(*M);
                    return M.release(); // intentionally leaked: lives for the process
                }
                catch(std::exception const& e)
                {
                    std::cerr << kTag << "ERROR loading model " << path << ": " << e.what()
                              << "; param scorer DISABLED" << std::endl;
                    return nullptr;
                }
#else
                std::cerr << kTag << "model " << path
                          << " ignored: this build has no msgpack support; param scorer DISABLED"
                          << std::endl;
                return nullptr;
#endif
            }

            // "<token><int>" right after an '_' (e.g. "_WGM8"), followed by '_' or end.
            bool nameInt(std::string const& name, std::string const& tok, long& v)
            {
                std::string pat = "_" + tok;
                for(size_t p = name.find(pat); p != std::string::npos; p = name.find(pat, p + 1))
                {
                    size_t q = p + pat.size(), e = q;
                    while(e < name.size() && std::isdigit(static_cast<unsigned char>(name[e])))
                        ++e;
                    if(e > q && (e == name.size() || name[e] == '_' || name[e] == 'x'))
                    {
                        v = std::stol(name.substr(q, e - q));
                        return true;
                    }
                }
                return false;
            }

            // "<token>AxBxC" / "<token>A_B_C" -> up to 3 ints.
            bool nameInts(std::string const& name, std::string const& tok, char sep, long out[3])
            {
                std::string const sp = "[" + std::string(1, sep) + "]";
                std::regex        re("_" + tok + "([0-9]+)" + sp + "([0-9]+)" + sp + "([0-9]+)(_|$)");
                std::smatch mt;
                if(!std::regex_search(name, mt, re))
                    return false;
                for(int i = 0; i < 3; ++i)
                    out[i] = std::stol(mt[i + 1].str());
                return true;
            }

            // Structural mismatch description, or "" when the logic-file name agrees
            // with the runtime solution on every token it carries.
            std::string structuralMismatch(std::string const& nm, SolutionInfo const& s)
            {
                std::ostringstream why;
                long               a[3];
                if(!nameInts(nm, "MT", 'x', a))
                    return "no MT token in model name";
                if(size_t(a[0]) != s.mt0 || size_t(a[1]) != s.mt1 || size_t(a[2]) != s.depthU)
                    why << "MT " << a[0] << "x" << a[1] << "x" << a[2] << " vs " << s.mt0 << "x"
                        << s.mt1 << "x" << s.depthU << "; ";
                if(nameInts(nm, "MI", 'x', a) && (a[0] != s.mi0 || a[1] != s.mi1))
                    why << "MI " << a[0] << "x" << a[1] << " vs " << s.mi0 << "x" << s.mi1 << "; ";
                if(nameInts(nm, "WG", '_', a)
                   && (size_t(a[0]) != s.wg0 || size_t(a[1]) != s.wg1 || size_t(a[2]) != s.wg2))
                    why << "WG " << a[0] << "_" << a[1] << "_" << a[2] << " vs " << s.wg0 << "_"
                        << s.wg1 << "_" << s.wg2 << "; ";
                long v;
                if(nameInt(nm, "WGM", v) && v != s.wgm)
                    why << "WGM " << v << " vs " << s.wgm << "; ";
                if(nameInt(nm, "LDSB", v) && v != s.ldsb)
                    why << "LDSB " << v << " vs " << s.ldsb << "; ";
                if(nameInt(nm, "PGR", v) && v != s.pgr)
                    why << "PGR " << v << " vs " << s.pgr << "; ";
                return why.str();
            }
        } // namespace

        namespace
        {
            std::mutex&  dirMutex()
            {
                static std::mutex mu;
                return mu;
            }
            std::string& defaultDirRef()
            {
                static std::string dir;
                return dir;
            }
            std::string defaultDir()
            {
                std::lock_guard<std::mutex> lock(dirMutex());
                return defaultDirRef();
            }

            // *.msgpack in dir, sorted (deterministic load order).
            std::vector<std::string> modelsInDir(std::string const& dir)
            {
                std::vector<std::string> out;
                std::error_code          ec;
                for(auto const& e : std::filesystem::directory_iterator(dir, ec))
                    if(e.is_regular_file(ec) && e.path().extension() == ".msgpack")
                        out.push_back(e.path().string());
                std::sort(out.begin(), out.end());
                return out;
            }

            std::vector<Model const*> loadAll()
            {
                std::vector<std::string> files;
                char const*              env = std::getenv("TENSILE_PARAM_MODEL");
                if(env && *env)
                {
                    std::string v = env;
                    if(v == "none" || v == "off" || v == "0")
                    {
                        std::cerr << kTag << "TENSILE_PARAM_MODEL=" << v
                                  << ": no param model loaded" << std::endl;
                        return {};
                    }
                    std::stringstream ss(v);
                    for(std::string item; std::getline(ss, item, ':');)
                    {
                        std::error_code ec;
                        if(item.empty())
                            continue;
                        if(std::filesystem::is_directory(item, ec))
                        {
                            auto d = modelsInDir(item);
                            files.insert(files.end(), d.begin(), d.end());
                        }
                        else
                            files.push_back(item);
                    }
                }
                else
                {
                    std::string dir = defaultDir();
                    std::error_code ec;
                    if(!dir.empty() && std::filesystem::is_directory(dir, ec))
                        files = modelsInDir(dir);
                }
                std::vector<Model const*> out;
                for(auto const& f : files)
                    if(auto const* M = loadOne(f))
                        out.push_back(M);
                return out;
            }

            // Built solution names of a library start with this (the logic-file
            // basename minus its "<arch>_" prefix, plus '_').
            std::string keyPrefix(Model const& M)
            {
                std::string prefix = M.problemKey;
                if(prefix.rfind("gfx", 0) == 0 && prefix.find('_') != std::string::npos)
                    prefix = prefix.substr(prefix.find('_') + 1);
                return prefix + "_";
            }
        } // namespace

        void setDefaultModelDir(std::string const& dir)
        {
            std::lock_guard<std::mutex> lock(dirMutex());
            if(defaultDirRef().empty())
                defaultDirRef() = dir;
        }

        std::vector<Model const*> const& models()
        {
            static std::vector<Model const*> const all = loadAll();
            return all;
        }

        bool paramOnly()
        {
            static bool const only = [] {
                std::string     dir = defaultDir();
                std::error_code ec;
                return !dir.empty() && std::filesystem::is_directory(dir, ec);
            }();
            return only;
        }

        Model const* modelFor(std::vector<SolutionInfo> const& sols)
        {
            if(sols.empty())
                return nullptr;
            for(auto const* M : models())
            {
                std::string prefix = keyPrefix(*M);
                bool        mine   = std::all_of(sols.begin(), sols.end(), [&](auto const& s) {
                    return s.name.rfind(prefix, 0) == 0;
                });
                if(mine)
                    return M; // count and structure are checked (loudly) by bindLibrary
            }
            return nullptr;
        }

        std::vector<int> bindLibrary(Model const& M, std::vector<SolutionInfo> const& sols)
        {
            std::ostringstream head;
            head << kTag << "Prediction library (" << sols.size() << " solutions"
                 << (sols.empty() ? "" : ", first '" + sols[0].name.substr(0, 60) + "...'")
                 << "): param model ";
            auto inactive = [&](std::string const& why) {
                std::cerr << head.str() << "INACTIVE: " << why << std::endl;
                return std::vector<int>{};
            };

            if(sols.size() != M.solutions.size())
                return inactive("solution count " + std::to_string(sols.size()) + " != model rows "
                                + std::to_string(M.solutions.size()));

            // Problem type: the built solution names start with the logic-file
            // basename minus its "<arch>_" prefix.
            std::string const prefix = keyPrefix(M);

            std::vector<int>  rowOfPos(sols.size(), -1);
            std::vector<char> seen(sols.size(), 0);
            size_t            builtNameMismatch = 0;
            std::string       firstBuiltMismatch;
            for(size_t pos = 0; pos < sols.size(); ++pos)
            {
                auto const& s   = sols[pos];
                int         row = s.libraryLogicIndex;
                if(row < 0 || size_t(row) >= sols.size() || seen[row])
                    return inactive("position " + std::to_string(pos)
                                    + ": libraryLogicIndex " + std::to_string(row)
                                    + " is out of range or duplicated");
                seen[row] = 1;
                if(s.name.rfind(prefix, 0) != 0)
                    return inactive("solution '" + s.name.substr(0, 80)
                                    + "' does not start with problem key '" + prefix + "'");
                std::string why = structuralMismatch(M.solutionNames[row], s);
                if(!why.empty())
                    return inactive("row " + std::to_string(row) + " structural mismatch: " + why
                                    + " (model name '" + M.solutionNames[row].substr(0, 80)
                                    + "...')");
                if(!M.solutionNamesBuilt.empty() && M.solutionNamesBuilt[row] != s.name)
                {
                    if(builtNameMismatch++ == 0)
                        firstBuiltMismatch = std::to_string(row);
                }
                rowOfPos[pos] = row;
            }
            std::cerr << head.str() << "ACTIVE (model " << M.path << ", key " << M.problemKey
                      << ", rows by libraryLogicIndex, structural check MT/DU/MI/WG/WGM/LDSB/PGR "
                         "passed for all rows, built-name check: "
                      << (M.solutionNamesBuilt.empty()
                              ? std::string("n/a")
                              : (builtNameMismatch == 0
                                     ? std::string("all equal")
                                     : std::to_string(builtNameMismatch)
                                           + " differ (first row " + firstBuiltMismatch + ")"))
                      << ", topk=" << topK(M) << ")" << std::endl;
            return rowOfPos;
        }

        std::vector<int> rankRows(Model const&         M,
                                  size_t               m,
                                  size_t               n,
                                  size_t               k,
                                  size_t               b,
                                  std::vector<double>* scores)
        {
            auto s     = M.scoresFromFeatures(M.features(m, n, k, b));
            auto order = M.order(s);
            if(scores)
                scores->swap(s);
            return order;
        }

        int topK(Model const& M)
        {
            static int const env = [] {
                char const* v = std::getenv("TENSILE_PARAM_TOPK");
                return (v && *v) ? std::atoi(v) : -1;
            }();
            return env >= 0 ? env : M.topkDefault;
        }

        bool debug()
        {
            static bool const d = envFlag("TENSILE_PARAM_DEBUG");
            return d;
        }
    } // namespace ParamScorer
} // namespace TensileLite
