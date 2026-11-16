// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/param/param_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#if ORIGAMI_HAS_MSGPACK
#include <msgpack.hpp>
#endif

namespace origami::nn::param {

using feature_fn_t = std::function<double(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t)>;

struct layer_t {
  std::size_t in = 0, out = 0;
  std::vector<double> Wt;  // in x out: W transposed, so a layer is a sum of x[i] * column i
  std::vector<double> b;
  bool relu = false;
};

// The candidate parameter a head predicts: TensileLite sizeMapping fields carried in config_t,
// the same values the logic-file solution names encode.
enum class field_t { mt0, mt1, du, wgm, gsu, su, sum, sus, pgr, tlds, ldsb, dtvb };

/// One softmax head: its slice of the output layer and the parameter value of each class.
struct head_t {
  std::string name;
  std::size_t offset = 0, size = 0;
  field_t field = field_t::mt0;    // the candidate parameter the head predicts
  std::vector<long long> values;   // values[class]; a handful per head, so a linear scan
};

struct test_vector_t {
  std::uint64_t m = 0, n = 0, k = 0, b = 0;
  std::vector<double> x;
  std::vector<int> order;
};

struct model_t {
  std::string path;
  std::string problem_key;
  std::vector<std::string> feature_names;
  std::vector<feature_fn_t> feature_fns;
  std::vector<double> mu, sd;
  std::vector<layer_t> layers;
  std::vector<head_t> heads;
  std::vector<double> head_weight;
  double unk_logp = -20.0;
  /// The training catalog's head classes per row: only for the embedded test vectors.
  std::vector<std::vector<int>> rows;
  std::vector<test_vector_t> test_vectors;
};

namespace {

std::uint64_t ceil_div(std::uint64_t a, std::uint64_t c) { return c == 0 ? 0 : (a + c - 1) / c; }

std::function<std::uint64_t(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t)> var(char c) {
  switch (c) {
    case 'm':
      return [](std::uint64_t m, std::uint64_t, std::uint64_t, std::uint64_t) { return m; };
    case 'n':
      return [](std::uint64_t, std::uint64_t n, std::uint64_t, std::uint64_t) { return n; };
    case 'k':
      return [](std::uint64_t, std::uint64_t, std::uint64_t k, std::uint64_t) { return k; };
    default:
      return [](std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t b) { return b; };
  }
}

// Feature-name grammar (MODEL_FORMAT.md of the nn_selector param_predictor tools):
//   v | log2<vs> | log2_<v>_div_<w> | ceil_<v>_div_<C> | log2_ceil_<v>_div_<C>
//   <v>mod<C> | <v>mod<C>_is0 | <v>_(gt|ge|lt|le|eq)<C>
//   tiles_<A>x<B> | log2_tiles_<A>x<B> | waveeff_<A>x<B>_cu<C> | waves_<A>x<B>_cu<C>
bool compile_feature(const std::string& name, feature_fn_t& fn) {
  std::smatch mt;
  auto u64 = [](const std::string& s) { return std::stoull(s); };
  if (std::regex_match(name, mt, std::regex("^([mnkb])$"))) {
    auto v = var(mt[1].str()[0]);
    fn = [v](std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t b) {
      return double(v(m, n, k, b));
    };
    return true;
  }
  if (std::regex_match(name, mt, std::regex("^log2([mnkb]+)$"))) {
    std::string vs = mt[1].str();
    fn = [vs](std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t b) {
      double p = 1.0;
      for (char c : vs) p *= double(var(c)(m, n, k, b));
      return std::log2(p);
    };
    return true;
  }
  if (std::regex_match(name, mt, std::regex("^log2_([mnkb])_div_([mnkb])$"))) {
    auto v = var(mt[1].str()[0]);
    auto w = var(mt[2].str()[0]);
    fn = [v, w](std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t b) {
      return std::log2(double(v(m, n, k, b)) / double(w(m, n, k, b)));
    };
    return true;
  }
  if (std::regex_match(name, mt, std::regex("^(log2_)?ceil_([mnkb])_div_([0-9]+)$"))) {
    bool lg = mt[1].matched;
    auto v = var(mt[2].str()[0]);
    std::uint64_t C = u64(mt[3].str());
    fn = [lg, v, C](std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t b) {
      double c = double(ceil_div(v(m, n, k, b), C));
      return lg ? std::log2(c) : c;
    };
    return C > 0;
  }
  if (std::regex_match(name, mt, std::regex("^([mnkb])mod([0-9]+)(_is0)?$"))) {
    auto v = var(mt[1].str()[0]);
    std::uint64_t C = u64(mt[2].str());
    bool is0 = mt[3].matched;
    fn = [v, C, is0](std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t b) {
      std::uint64_t r = v(m, n, k, b) % C;
      return is0 ? (r == 0 ? 1.0 : 0.0) : double(r);
    };
    return C > 0;
  }
  if (std::regex_match(name, mt, std::regex("^([mnkb])_(gt|ge|lt|le|eq)([0-9]+)$"))) {
    auto v = var(mt[1].str()[0]);
    std::string op = mt[2].str();
    std::uint64_t C = u64(mt[3].str());
    fn = [v, op, C](std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t b) {
      std::uint64_t x = v(m, n, k, b);
      bool r = op == "gt" ? x > C : op == "ge" ? x >= C : op == "lt" ? x < C : op == "le" ? x <= C : x == C;
      return r ? 1.0 : 0.0;
    };
    return true;
  }
  if (std::regex_match(name, mt, std::regex("^(log2_)?tiles_([0-9]+)x([0-9]+)$"))) {
    bool lg = mt[1].matched;
    std::uint64_t A = u64(mt[2].str()), B = u64(mt[3].str());
    fn = [lg, A, B](std::uint64_t m, std::uint64_t n, std::uint64_t, std::uint64_t b) {
      double t = double(ceil_div(m, A) * ceil_div(n, B) * b);
      return lg ? std::log2(t) : t;
    };
    return A > 0 && B > 0;
  }
  if (std::regex_match(name, mt, std::regex("^(waveeff|waves)_([0-9]+)x([0-9]+)_cu([0-9]+)$"))) {
    bool eff = mt[1].str() == "waveeff";
    std::uint64_t A = u64(mt[2].str()), B = u64(mt[3].str()), C = u64(mt[4].str());
    fn = [eff, A, B, C](std::uint64_t m, std::uint64_t n, std::uint64_t, std::uint64_t b) {
      std::uint64_t t = ceil_div(m, A) * ceil_div(n, B) * b;
      if (eff) return double(t) / double(ceil_div(t, C) * C);
      return double(t) / double(C);
    };
    return A > 0 && B > 0 && C > 0;
  }
  return false;
}

// The head named `head` reads `*field`; false for a head with no run-time value.
bool field_of(const std::string& head, field_t* field) {
  static const std::pair<const char*, field_t> names[] = {
      {"MT0", field_t::mt0}, {"MT1", field_t::mt1}, {"DU", field_t::du},     {"WGM", field_t::wgm},
      {"GSU", field_t::gsu}, {"SU", field_t::su},   {"SUM", field_t::sum},   {"SUS", field_t::sus},
      {"PGR", field_t::pgr}, {"TLDS", field_t::tlds}, {"LDSB", field_t::ldsb}, {"DTVB", field_t::dtvb}};
  for (const auto& [name, f] : names)
    if (head == name) {
      *field = f;
      return true;
    }
  return false;
}

// The candidate's value of `field`; false when it needs Tensile params the config does not carry.
inline bool field_value(const config_t& c, const tensile_params_t* t, field_t field, long long& v) {
  switch (field) {
    case field_t::mt0: v = (long long)c.mt.m; return true;
    case field_t::mt1: v = (long long)c.mt.n; return true;
    case field_t::du: v = (long long)c.mt.k; return true;
    case field_t::wgm: v = c.workgroup_mapping; return true;
    default: break;
  }
  if (t == nullptr) return false;
  switch (field) {
    case field_t::gsu: v = t->global_split_u; break;
    case field_t::su: v = t->stagger_u; break;
    case field_t::sum: v = t->stagger_u_mapping; break;
    case field_t::sus: v = t->stagger_stride_shift; break;
    case field_t::pgr: v = t->prefetch_global_read; break;
    // TransposeLDS as the kernel name carries it: UnrollMajorLDSB (TLDS1 -> 1 for B; TLDS0 -> 0).
    case field_t::tlds: v = t->unroll_major_lds_b; break;
    case field_t::ldsb: v = t->one_lds_buffer ? 1 : 0; break;
    case field_t::dtvb: v = t->direct_to_vgpr_b ? 1 : 0; break;
    default: return false;
  }
  return true;
}

std::vector<double> features(const model_t& M, std::uint64_t m, std::uint64_t n, std::uint64_t k, std::uint64_t b) {
  std::vector<double> x(M.feature_fns.size());
  for (std::size_t i = 0; i < M.feature_fns.size(); ++i) {
    double s = M.sd[i] != 0.0 ? M.sd[i] : 1.0;
    x[i] = (M.feature_fns[i](m, n, k, b) - M.mu[i]) / s;
  }
  return x;
}

std::vector<std::vector<double>> logp_from_features(const model_t& M, std::vector<double> x) {
  for (const auto& L : M.layers) {
    // y[o] = b[o] + sum_i W[o][i] * x[i], accumulated in ascending i for every o (the row-wise
    // order), one column at a time so the inner loop vectorizes; zero inputs (ReLU) add nothing.
    std::vector<double> y(L.b);
    for (std::size_t i = 0; i < L.in; ++i) {
      const double xi = x[i];
      if (xi == 0.0) continue;
      const double* w = L.Wt.data() + i * L.out;
      for (std::size_t o = 0; o < L.out; ++o) y[o] += w[o] * xi;
    }
    if (L.relu)
      for (auto& v : y) v = v < 0.0 ? 0.0 : v;
    x.swap(y);
  }
  std::vector<std::vector<double>> logp(M.heads.size());
  for (std::size_t h = 0; h < M.heads.size(); ++h) {
    const auto& H = M.heads[h];
    double mx = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < H.size; ++j) mx = std::max(mx, x[H.offset + j]);
    double se = 0.0;
    for (std::size_t j = 0; j < H.size; ++j) se += std::exp(x[H.offset + j] - mx);
    const double lse = mx + std::log(se);
    logp[h].resize(H.size);
    for (std::size_t j = 0; j < H.size; ++j) logp[h][j] = x[H.offset + j] - lse;
  }
  return logp;
}

#if ORIGAMI_HAS_MSGPACK
using obj_t = msgpack::object;

const obj_t* find(const obj_t& o, const char* key) {
  if (o.type != msgpack::type::MAP) throw std::runtime_error("expected a map");
  for (std::uint32_t i = 0; i < o.via.map.size; ++i) {
    const auto& kv = o.via.map.ptr[i];
    if (kv.key.type == msgpack::type::STR && std::string(kv.key.via.str.ptr, kv.key.via.str.size) == key)
      return &kv.val;
    if (kv.key.type == msgpack::type::BIN && std::string(kv.key.via.bin.ptr, kv.key.via.bin.size) == key)
      return &kv.val;
  }
  return nullptr;
}

const obj_t& get(const obj_t& o, const char* key) {
  auto p = find(o, key);
  if (!p) throw std::runtime_error(std::string("missing key '") + key + "'");
  return *p;
}

double num(const obj_t& o) {
  switch (o.type) {
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

long long integer(const obj_t& o) {
  double d = num(o);
  if (d != std::floor(d)) throw std::runtime_error("expected an integer");
  return (long long)d;
}

std::string str(const obj_t& o) {
  if (o.type == msgpack::type::STR) return std::string(o.via.str.ptr, o.via.str.size);
  if (o.type == msgpack::type::BIN) return std::string(o.via.bin.ptr, o.via.bin.size);
  throw std::runtime_error("expected a string");
}

std::vector<const obj_t*> arr(const obj_t& o) {
  if (o.type != msgpack::type::ARRAY) throw std::runtime_error("expected an array");
  std::vector<const obj_t*> v;
  for (std::uint32_t i = 0; i < o.via.array.size; ++i) v.push_back(&o.via.array.ptr[i]);
  return v;
}

std::vector<double> nums(const obj_t& o) {
  std::vector<double> v;
  for (auto p : arr(o)) v.push_back(num(*p));
  return v;
}

std::vector<std::string> strs(const obj_t& o) {
  std::vector<std::string> v;
  for (auto p : arr(o)) v.push_back(str(*p));
  return v;
}

std::shared_ptr<model_t> parse(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open file");
  std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  auto handle = msgpack::unpack(data.data(), data.size());
  const obj_t& root = handle.get();

  auto M = std::make_shared<model_t>();
  M->path = path;
  const long long version = integer(get(root, "version"));
  if (version != 1) throw std::runtime_error("unsupported version " + std::to_string(version));
  M->problem_key = str(get(root, "problem_key"));

  const auto& feats = get(root, "features");
  M->feature_names = strs(get(feats, "names"));
  M->mu = nums(get(feats, "mu"));
  M->sd = nums(get(feats, "sd"));
  const std::size_t F = M->feature_names.size();
  if (M->mu.size() != F || M->sd.size() != F) throw std::runtime_error("features: names/mu/sd size mismatch");
  for (const auto& nm : M->feature_names) {
    feature_fn_t fn;
    if (!compile_feature(nm, fn)) throw std::runtime_error("unsupported feature name '" + nm + "'");
    M->feature_fns.push_back(fn);
  }

  std::size_t width = F;
  for (auto lp : arr(get(root, "layers"))) {
    layer_t L;
    L.in = std::size_t(integer(get(*lp, "in")));
    L.out = std::size_t(integer(get(*lp, "out")));
    const auto W = nums(get(*lp, "W"));  // out x in, row-major
    L.b = nums(get(*lp, "b"));
    const std::string act = str(get(*lp, "act"));
    if (act != "relu" && act != "none") throw std::runtime_error("unsupported activation '" + act + "'");
    L.relu = act == "relu";
    if (L.in != width || W.size() != L.in * L.out || L.b.size() != L.out)
      throw std::runtime_error("layer " + std::to_string(M->layers.size()) + ": shape mismatch");
    L.Wt.resize(W.size());
    for (std::size_t o = 0; o < L.out; ++o)
      for (std::size_t i = 0; i < L.in; ++i) L.Wt[i * L.out + o] = W[o * L.in + i];
    width = L.out;
    M->layers.push_back(std::move(L));
  }
  if (M->layers.empty()) throw std::runtime_error("no layers");

  const auto& values = get(root, "head_values");
  for (auto hp : arr(get(root, "heads"))) {
    head_t H;
    H.name = str(get(*hp, "name"));
    H.offset = std::size_t(integer(get(*hp, "offset")));
    H.size = std::size_t(integer(get(*hp, "size")));
    if (H.size == 0 || H.offset + H.size > width) throw std::runtime_error("head '" + H.name + "' out of range");
    if (!field_of(H.name, &H.field))
      throw std::runtime_error("head '" + H.name + "' has no run-time value in origami::config_t");
    const auto vals = arr(get(values, H.name.c_str()));
    if (vals.size() != H.size) throw std::runtime_error("head '" + H.name + "': head_values size != head size");
    for (const auto* v : vals) H.values.push_back(integer(*v));
    M->heads.push_back(std::move(H));
  }
  M->head_weight = nums(get(root, "head_weight"));
  if (M->heads.empty() || M->head_weight.size() != M->heads.size())
    throw std::runtime_error("heads/head_weight size mismatch");
  if (auto p = find(root, "unk_logp")) M->unk_logp = num(*p);

  for (auto rp : arr(get(root, "solutions"))) {
    std::vector<int> row;
    for (auto ip : arr(*rp)) row.push_back(int(integer(*ip)));
    if (row.size() != M->heads.size())
      throw std::runtime_error("solutions row " + std::to_string(M->rows.size()) + ": wrong number of head indices");
    M->rows.push_back(std::move(row));
  }

  if (auto p = find(root, "test_vectors")) {
    for (auto tp : arr(*p)) {
      test_vector_t t;
      t.m = std::uint64_t(integer(get(*tp, "m")));
      t.n = std::uint64_t(integer(get(*tp, "n")));
      t.k = std::uint64_t(integer(get(*tp, "k")));
      t.b = std::uint64_t(integer(get(*tp, "b")));
      t.x = nums(get(*tp, "x"));
      for (auto ip : arr(get(*tp, "order"))) t.order.push_back(int(integer(*ip)));
      M->test_vectors.push_back(std::move(t));
    }
  }
  return M;
}
#endif

}  // namespace

bool supported() { return ORIGAMI_HAS_MSGPACK != 0; }

std::shared_ptr<const model_t> load(const std::string& path, std::string* error) {
#if ORIGAMI_HAS_MSGPACK
  try {
    return parse(path);
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
#else
  (void)path;
  if (error) *error = "this Origami build has no msgpack support";
  return nullptr;
#endif
}

const std::string& problem_key(const model_t& model) { return model.problem_key; }

std::string summary(const model_t& M) {
  std::ostringstream os;
  os << "key=" << M.problem_key << " features=" << M.feature_names.size() << " layers=" << M.layers.size()
     << " heads=" << M.heads.size() << " test_vectors=" << M.test_vectors.size();
  return os.str();
}

std::vector<std::vector<double>> head_logp(const model_t& M,
                                           std::uint64_t m,
                                           std::uint64_t n,
                                           std::uint64_t k,
                                           std::uint64_t batch) {
  return logp_from_features(M, features(M, m, n, k, batch));
}

bool score(const model_t& M, const std::vector<std::vector<double>>& logp, const config_t& config, double* out) {
  const tensile_params_t* t = config.has_tensile_params() ? &config.tensile() : nullptr;
  double s = 0.0;
  for (std::size_t h = 0; h < M.heads.size(); ++h) {
    const auto& H = M.heads[h];
    long long v = 0;
    if (!field_value(config, t, H.field, v)) return false;
    const auto it = std::find(H.values.begin(), H.values.end(), v);
    s += M.head_weight[h] * (it != H.values.end() ? logp[h][std::size_t(it - H.values.begin())] : M.unk_logp);
  }
  *out = s;
  return true;
}

std::vector<prediction_result_t> rank(const model_t& M, const problem_t& problem, const std::vector<config_t>& configs) {
  const auto logp = head_logp(M, problem.size.m, problem.size.n, problem.size.k, problem.batch);
  std::vector<double> scores(configs.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < configs.size(); ++i) {
    double s = 0.0;
    if (score(M, logp, configs[i], &s)) scores[i] = s;
  }
  // Best score first, ties and unscored configs in input order: sort (-score, position) pairs,
  // with the unscored ones keyed +inf so they come last.
  std::vector<std::pair<double, std::size_t>> order(configs.size());
  for (std::size_t i = 0; i < configs.size(); ++i)
    order[i] = {std::isfinite(scores[i]) ? -scores[i] : std::numeric_limits<double>::infinity(), i};
  std::sort(order.begin(), order.end());
  std::vector<prediction_result_t> out;
  out.reserve(configs.size());
  for (const auto& [key, i] : order) out.push_back(prediction_result_t{-scores[i], configs[i]});
  return out;
}

bool self_test(const model_t& M, std::string* error) {
  auto fail = [&](const std::string& why) {
    if (error) *error = why;
    return false;
  };
  if (M.test_vectors.empty()) return fail("model has no test_vectors");
  if (M.rows.empty()) return fail("model has no solutions table for its test vectors");
  for (std::size_t t = 0; t < M.test_vectors.size(); ++t) {
    const auto& T = M.test_vectors[t];
    auto x = features(M, T.m, T.n, T.k, T.b);
    if (x.size() != T.x.size()) return fail("vector " + std::to_string(t) + ": feature count mismatch");
    for (std::size_t i = 0; i < x.size(); ++i)
      if (!(std::abs(x[i] - T.x[i]) <= 1e-4))
        return fail("vector " + std::to_string(t) + " feature '" + M.feature_names[i] + "' differs");
    const auto logp = logp_from_features(M, x);
    std::vector<double> scores(M.rows.size(), 0.0);
    for (std::size_t r = 0; r < M.rows.size(); ++r)
      for (std::size_t h = 0; h < M.heads.size(); ++h) {
        const int idx = M.rows[r][h];
        scores[r] += M.head_weight[h] * (idx >= 0 ? logp[h][idx] : M.unk_logp);
      }
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    const std::size_t n = std::min<std::size_t>(10, std::min(order.size(), T.order.size()));
    for (std::size_t i = 0; i < n; ++i) {
      if (order[i] == T.order[i]) continue;
      const int r = T.order[i];
      const bool in_range = r >= 0 && std::size_t(r) < scores.size();
      const double a = scores[order[i]], b = in_range ? scores[r] : 0.0;
      if (in_range && std::abs(a - b) <= 1e-6 * (1.0 + std::abs(a))) continue;  // score tie
      return fail("vector " + std::to_string(t) + " order position " + std::to_string(i) + " differs");
    }
  }
  return true;
}

}  // namespace origami::nn::param
