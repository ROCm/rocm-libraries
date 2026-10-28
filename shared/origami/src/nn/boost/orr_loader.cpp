// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ORIBOOST (.orr) loader of Origami Boost: the section parser (format, checksum, HEAD, FEAT,
// RNGE, TREE, MANI), the breadth-first tree layout, the quantized forest built at load and the
// batched evaluator. Compatibility with the build and the deployment is load_validate.cpp's.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "origami/nn/boost/library_stamp.hpp"
#include "origami/nn/boost/load_limits.hpp"
#include "origami/nn/boost/loaded_model.hpp"
#include "origami/nn/boost/sha256.hpp"

namespace origami::nn::boost {

namespace {

using detail::kDecCategorical;
using detail::kMissingNaN;
using detail::kMissingZero;

struct reader_t {
  const std::uint8_t* p;
  std::size_t         n;
  std::size_t         o  = 0;
  bool                ok = true;

  bool need(std::size_t k) {
    if (!ok || o + k > n || o + k < o) ok = false;
    return ok;
  }
  template <typename T>
  T get() {
    T v{};
    if (need(sizeof(T))) {
      std::memcpy(&v, p + o, sizeof(T));  // little-endian hosts only (x86/AMD64, AArch64 LE)
      o += sizeof(T);
    }
    return v;
  }
  std::string str() {
    const auto len = get<std::uint32_t>();
    if (!need(len)) return {};
    std::string s(reinterpret_cast<const char*>(p + o), len);
    o += len;
    return s;
  }
};

// A tree node after the breadth-first layout: siblings are adjacent, so a node stores only its
// left child (right = left + 1). A leaf points to itself.
struct node_t {
  double        threshold;
  std::int32_t  left;
  std::uint16_t feature;
  std::uint8_t  decision;
  std::uint8_t  pad;
};

}  // namespace

// With finite inputs (prepare()) and no zero-as-missing splits every decision is
// x <= threshold. With T_f the sorted distinct split
// thresholds of feature f and bin(x) = #{t in T_f : t < x}, x <= T_f[k] exactly when
// bin(x) <= k, so a node keeps only its threshold's index k and the walk compares bytes. Rows
// are binned once per call. NaN bins to 255 and goes right, as !(NaN <= t) does.
struct detail::forest_t {
  struct node_t {
    std::uint8_t  feature;  // kLeafFeature for a leaf
    std::uint8_t  bin;      // index of the threshold in T_feature (0 for a leaf)
    std::uint16_t left;     // left child relative to the tree root; a leaf holds itself - 1
  };
  static constexpr std::uint8_t kLeafFeature = 255;  // its bin is always 255: a leaf goes "right"
  std::vector<node_t>        nodes;       // same positions as ensemble_t::nodes_
  std::vector<double>        thresholds;  // every T_f, concatenated
  std::vector<std::uint32_t> begin;       // T_f = thresholds[begin[f], begin[f + 1])
  std::vector<std::uint8_t>  used;        // features with at least one split
};

namespace {

// nullptr when the model does not fit the byte encoding (the loader then refuses it).
std::shared_ptr<const detail::forest_t> build_forest(const std::vector<node_t>& nodes,
                                                     const std::vector<std::int32_t>& roots,
                                                     std::size_t                      num_features) {
  using forest_t = detail::forest_t;
  if (num_features > detail::kMaxQuantizedFeatures) return nullptr;
  auto is_leaf = [&](std::size_t p) { return nodes[p].left == static_cast<std::int32_t>(p); };
  std::vector<std::vector<double>> thr(num_features);
  for (std::size_t p = 0; p < nodes.size(); ++p) {
    if (is_leaf(p)) continue;
    if (std::isnan(nodes[p].threshold)) return nullptr;
    thr[nodes[p].feature].push_back(nodes[p].threshold);
  }
  auto f = std::make_shared<forest_t>();
  f->begin.push_back(0);
  for (std::size_t j = 0; j < num_features; ++j) {
    auto& t = thr[j];
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    if (t.size() > detail::kMaxThresholdsPerFeature) return nullptr;  // bins 0..254, 255 for NaN
    if (!t.empty()) f->used.push_back(static_cast<std::uint8_t>(j));
    f->thresholds.insert(f->thresholds.end(), t.begin(), t.end());
    f->begin.push_back(static_cast<std::uint32_t>(f->thresholds.size()));
  }
  f->nodes.resize(nodes.size());
  for (std::size_t t = 0; t < roots.size(); ++t) {
    const std::size_t b = static_cast<std::size_t>(roots[t]);
    const std::size_t e = t + 1 < roots.size() ? static_cast<std::size_t>(roots[t + 1]) : nodes.size();
    if (e - b > detail::kMaxNodesPerTree) return nullptr;
    for (std::size_t p = b; p < e; ++p) {
      const auto& nd = nodes[p];
      if (is_leaf(p)) {
        f->nodes[p] = {forest_t::kLeafFeature, 0, static_cast<std::uint16_t>(p - b - 1)};
        continue;
      }
      const double* T   = f->thresholds.data() + f->begin[nd.feature];
      const double* end = f->thresholds.data() + f->begin[nd.feature + 1u];
      f->nodes[p]       = {static_cast<std::uint8_t>(nd.feature),
                           static_cast<std::uint8_t>(std::lower_bound(T, end, nd.threshold) - T),
                           static_cast<std::uint16_t>(static_cast<std::size_t>(nd.left) - b)};
    }
  }
  return f;
}

// All R rows through one tree at a time; leaves are added in tree order.
template <int R>
void forest_walk(const detail::forest_t& f, const double* values, const std::int32_t* roots,
                 const std::uint32_t* depths, std::size_t num_trees, const double* rows,
                 std::size_t nf, double init, double* out) {
  alignas(64) std::uint8_t xb[R][256];
  for (const std::uint8_t j : f.used) {
    const double*     T = f.thresholds.data() + f.begin[j];
    const std::size_t K = f.begin[j + 1u] - f.begin[j];
    std::size_t       lo[R];
    double            x[R];
    for (int i = 0; i < R; ++i) {
      lo[i] = 0;
      x[i]  = rows[static_cast<std::size_t>(i) * nf + j];
    }
    for (std::size_t len = K; len > 1;) {
      const std::size_t half = len / 2;
      for (int i = 0; i < R; ++i)
        lo[i] += half & (std::size_t{0} - static_cast<std::size_t>(T[lo[i] + half - 1] < x[i]));
      len -= half;
    }
    for (int i = 0; i < R; ++i)
      xb[i][j] = std::isnan(x[i]) ? std::uint8_t{255}
                                  : static_cast<std::uint8_t>(lo[i] + (T[lo[i]] < x[i]));
  }
  for (int i = 0; i < R; ++i) xb[i][detail::forest_t::kLeafFeature] = 255;

  double sum[R];
  for (int i = 0; i < R; ++i) sum[i] = 0.0;
  for (std::size_t t = 0; t < num_trees; ++t) {
    const auto*   nb = f.nodes.data() + roots[t];
    std::uint32_t cur[R];
    for (int i = 0; i < R; ++i) cur[i] = 0;
    for (std::uint32_t d = 0; d < depths[t]; ++d)
      for (int i = 0; i < R; ++i) {
        const auto& nd = nb[cur[i]];
        cur[i]         = nd.left + static_cast<std::uint32_t>(nd.bin < xb[i][nd.feature]);
      }
    const double* v = values + roots[t];
    for (int i = 0; i < R; ++i) sum[i] += v[cur[i]];
  }
  for (int i = 0; i < R; ++i) out[i] = sum[i] + init;
}

}  // namespace

struct model_parser {
  static std::shared_ptr<const ensemble_t> parse(const std::uint8_t* data, std::size_t size,
                                                 status_t& status, std::string& detail) {
    status = status_t::bad_file;
    if (size < detail::kHeaderBytes + detail::kChecksumBytes ||
        std::memcmp(data, detail::kMagic, sizeof(detail::kMagic)) != 0) {
      detail = "bad magic";
      return nullptr;
    }
    reader_t   pre{data, detail::kHeaderBytes, sizeof(detail::kMagic)};
    const auto major = pre.get<std::uint16_t>();
    (void)pre.get<std::uint16_t>();  // minor: additions old readers may ignore
    (void)pre.get<std::uint32_t>();
    const auto body_len = pre.get<std::uint64_t>();
    if (major != detail::kFormatMajor) {
      status = status_t::unsupported_format;
      detail = "format major " + std::to_string(major) + " (reader supports " +
               std::to_string(detail::kFormatMajor) + ")";
      return nullptr;
    }
    if (body_len > size || detail::kHeaderBytes + body_len + detail::kChecksumBytes != size) {
      detail = "length mismatch (truncated or padded file)";
      return nullptr;
    }
    const std::uint8_t* body = data + detail::kHeaderBytes;
    if (sha256_hex(body, body_len) != to_hex(body + body_len, detail::kChecksumBytes)) {
      detail = "checksum mismatch";
      return nullptr;
    }

    std::shared_ptr<ensemble_t> m(new ensemble_t());
    std::vector<node_t>         nodes;
    std::string                 mani_dtype;
    reader_t                    r{body, static_cast<std::size_t>(body_len)};
    std::set<std::string>       seen;
    while (r.ok && r.o < r.n) {
      if (!r.need(12)) break;
      std::string tag(reinterpret_cast<const char*>(r.p + r.o), 4);
      r.o += 4;
      const auto len = r.get<std::uint64_t>();
      if (!r.need(len)) break;
      reader_t s{r.p + r.o, static_cast<std::size_t>(len)};
      r.o += len;
      if (!detail::known_section(tag.c_str())) continue;  // unknown section: skip
      if (!seen.insert(tag).second) {
        detail = "duplicate section " + tag;
        return nullptr;
      }
      if (tag == "MANI") {
        const auto res = detail::parse_manifest_section(
            std::string(reinterpret_cast<const char*>(s.p), s.n), mani_dtype, m->manifest_,
            m->seen_values_, detail);
        if (res == detail::manifest_parse_t::ok) continue;
        if (res == detail::manifest_parse_t::unsupported) status = status_t::unsupported_format;
        return nullptr;
      }
      const bool sec_ok = tag == "HEAD"   ? parse_head(s, *m, detail)
                          : tag == "FEAT" ? parse_feat(s, *m, detail)
                          : tag == "RNGE" ? parse_rnge(s, *m)
                                          : parse_tree(s, *m, nodes, status, detail);
      if (!sec_ok || !s.ok || s.o != s.n) {
        if (detail.empty()) detail = "malformed section " + tag;
        return nullptr;
      }
    }
    if (!r.ok) {
      detail = "malformed section table";
      return nullptr;
    }
    if (seen.size() != detail::kNumSections) {
      detail = "missing required section";
      return nullptr;
    }
    for (const char* key : {"arch", "dtype", "feature_schema_hash", "origami_source_hash",
                            "origami_commit", "objective"}) {
      if (m->head_string(key).empty()) {
        detail = std::string("HEAD lacks ") + key;
        return nullptr;
      }
    }
    if (m->head_string("objective") != "regression") {
      detail = "objective " + m->head_string("objective") + " is not regression";
      return nullptr;
    }
    if (!check_ranges(*m, detail)) return nullptr;
    if (mani_dtype != m->head_string("dtype")) {
      detail = "MANI dtype " + mani_dtype + " differs from HEAD dtype " + m->head_string("dtype");
      return nullptr;
    }
    m->forest_ = build_forest(nodes, m->tree_root_, m->features_.size());
    if (!m->forest_) {
      status = status_t::unsupported_format;
      detail = "trees do not fit the quantized evaluator (more than 254 features, 254 thresholds "
               "per feature or 65536 nodes per tree)";
      return nullptr;
    }
    status = status_t::ok;
    detail.clear();
    return m;
  }

  static std::string to_hex(const std::uint8_t* p, std::size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string        s;
    for (std::size_t i = 0; i < n; ++i) {
      s.push_back(digits[p[i] >> 4]);
      s.push_back(digits[p[i] & 0xF]);
    }
    return s;
  }

  static bool parse_head(reader_t& s, ensemble_t& m, std::string& detail) {
    const auto count = s.get<std::uint32_t>();
    for (std::uint32_t i = 0; i < count && s.ok; ++i) {
      const std::string key  = s.str();
      const auto        type = s.get<std::uint8_t>();
      if (type == 1) {
        m.head_strings_.emplace_back(key, s.str());
      } else if (type == 2) {
        (void)s.get<double>();
      } else if (type == 3) {
        const auto v = s.get<std::int64_t>();
        if (key == "top_n") m.top_n_ = static_cast<std::size_t>(std::max<std::int64_t>(v, 1));
      } else {
        detail = "HEAD entry " + key + " has unknown type";
        return false;
      }
    }
    return true;
  }

  static bool parse_feat(reader_t& s, ensemble_t& m, std::string& detail) {
    const auto n = s.get<std::uint32_t>();
    if (n == 0 || n > detail::kMaxFeatures) {
      detail = "bad feature count";
      return false;
    }
    for (std::uint32_t i = 0; i < n && s.ok; ++i) {
      m.features_.push_back({s.str()});
    }
    return true;
  }

  static bool parse_rnge(reader_t& s, ensemble_t& m) {
    const auto n = s.get<std::uint32_t>();
    for (std::uint32_t i = 0; i < n && s.ok; ++i) {
      ensemble_t::range_t rg;
      rg.name = s.str();
      rg.min  = s.get<double>();
      rg.max  = s.get<double>();
      m.ranges_.push_back(std::move(rg));
    }
    return true;
  }

  // RNGE holds the training ranges of exactly M, N, K and batch, each once. A CU budget is not
  // a range (the features see it), so a file with num_cus or any other entry is refused.
  static bool check_ranges(const ensemble_t& m, std::string& detail) {
    static const char* const kNames[] = {"M", "N", "K", "batch"};
    bool seen[std::size(kNames)]      = {};
    for (const auto& rg : m.ranges_) {
      const auto it = std::find(std::begin(kNames), std::end(kNames), rg.name);
      if (it == std::end(kNames)) {
        detail = "RNGE entry " + rg.name + ": only M, N, K, batch";
        return false;
      }
      bool& s = seen[it - std::begin(kNames)];
      if (s) {
        detail = "RNGE entry " + rg.name + " appears twice";
        return false;
      }
      s = true;
    }
    for (std::size_t i = 0; i < std::size(kNames); ++i) {
      if (!seen[i]) {
        detail = std::string("RNGE lacks ") + kNames[i];
        return false;
      }
    }
    return true;
  }

  static bool parse_tree(reader_t& s, ensemble_t& m, std::vector<node_t>& nodes, status_t& status,
                         std::string& detail) {
    const auto num_trees = s.get<std::uint32_t>();
    const auto nf        = static_cast<std::int32_t>(m.features_.size());
    for (std::uint32_t t = 0; t < num_trees && s.ok; ++t) {
      const auto num_leaves   = s.get<std::uint32_t>();
      const auto num_internal = s.get<std::uint32_t>();
      if (!s.ok || num_leaves != num_internal + 1 ||
          !s.need(std::size_t(num_internal) * 24 + std::size_t(num_leaves) * 8)) {
        detail = "tree " + std::to_string(t) + ": bad shape";
        return false;
      }
      if (nodes.size() + num_leaves + num_internal >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        detail = "too many nodes";
        return false;
      }
      const auto base = static_cast<std::int32_t>(nodes.size());
      const auto I    = static_cast<std::int32_t>(num_internal);
      const auto L    = static_cast<std::int32_t>(num_leaves);
      struct raw_t {
        std::int32_t feature, left, right;
        std::uint8_t decision;
        double       threshold;
      };
      std::vector<raw_t>         raw(static_cast<std::size_t>(I));
      std::vector<std::uint32_t> depth(num_internal, 0);
      std::uint32_t              max_depth = 0;
      for (std::int32_t i = 0; i < I; ++i) {
        raw_t& nd   = raw[static_cast<std::size_t>(i)];
        nd.feature  = s.get<std::int32_t>();
        nd.decision = s.get<std::uint8_t>();
        s.o += 3;
        nd.threshold = s.get<double>();
        nd.left      = s.get<std::int32_t>();
        nd.right     = s.get<std::int32_t>();
        // Children follow their parent, so depths fill in one forward pass.
        auto child_ok = [&](std::int32_t c) { return c >= 0 ? (c > i && c < I) : (~c < L); };
        const int missing = (nd.decision >> 2) & 3;
        if (nd.decision & kDecCategorical) {
          detail = "tree " + std::to_string(t) + ": categorical split";
          return false;
        }
        if (nd.feature < 0 || nd.feature >= nf || !child_ok(nd.left) || !child_ok(nd.right) ||
            missing > kMissingNaN) {
          detail = "tree " + std::to_string(t) + ": bad node " + std::to_string(i);
          return false;
        }
        // Inputs are finite after prepare(), where only zero-as-missing departs from
        // x <= threshold.
        if (missing == kMissingZero) {
          status = status_t::unsupported_format;
          detail = "tree " + std::to_string(t) + ": zero-as-missing split";
          return false;
        }
        const std::uint32_t d = depth[static_cast<std::size_t>(i)] + 1;
        max_depth             = std::max(max_depth, d);
        for (std::int32_t c : {nd.left, nd.right})
          if (c >= 0) depth[static_cast<std::size_t>(c)] = d;
      }
      std::vector<double> leaf(static_cast<std::size_t>(L));
      for (auto& v : leaf) v = s.get<double>();
      if (!s.ok) return true;  // the caller reports the truncated section

      // Breadth-first layout with siblings adjacent. A file child code c >= 0 is internal
      // node c and c < 0 is leaf ~c; the slot vectors map them to new absolute indices.
      std::vector<std::int32_t> slot_internal(static_cast<std::size_t>(I), -1);
      std::vector<std::int32_t> slot_leaf(static_cast<std::size_t>(L), -1);
      const std::size_t total = static_cast<std::size_t>(I) + static_cast<std::size_t>(L);
      nodes.resize(nodes.size() + total);
      m.values_.resize(m.values_.size() + total, 0.0);
      std::int32_t next = base;
      if (I == 0) {
        slot_leaf[0] = next++;
      } else {
        slot_internal[0] = next++;
        std::vector<std::int32_t> queue{0};
        for (std::size_t q = 0; q < queue.size(); ++q) {
          const raw_t& nd = raw[static_cast<std::size_t>(queue[q])];
          for (const std::int32_t c : {nd.left, nd.right}) {
            const std::int32_t pos = next++;
            if (c >= 0) {
              slot_internal[static_cast<std::size_t>(c)] = pos;
              queue.push_back(c);
            } else {
              slot_leaf[static_cast<std::size_t>(~c)] = pos;
            }
          }
        }
      }
      if (static_cast<std::size_t>(next - base) != total ||
          std::find(slot_leaf.begin(), slot_leaf.end(), -1) != slot_leaf.end()) {
        detail = "tree " + std::to_string(t) + ": nodes not reachable from the root";
        return false;
      }
      auto child = [&](std::int32_t c) {
        return c >= 0 ? slot_internal[static_cast<std::size_t>(c)]
                      : slot_leaf[static_cast<std::size_t>(~c)];
      };
      for (std::int32_t i = 0; i < I; ++i) {
        const raw_t&       nd  = raw[static_cast<std::size_t>(i)];
        const std::int32_t pos = slot_internal[static_cast<std::size_t>(i)];
        nodes[static_cast<std::size_t>(pos)] = {nd.threshold, child(nd.left),
                                                static_cast<std::uint16_t>(nd.feature),
                                                nd.decision, 0};
      }
      for (std::int32_t j = 0; j < L; ++j) {
        const std::int32_t pos = slot_leaf[static_cast<std::size_t>(j)];
        nodes[static_cast<std::size_t>(pos)] = {std::numeric_limits<double>::infinity(), pos, 0, 0,
                                                0};
        m.values_[static_cast<std::size_t>(pos)] = leaf[static_cast<std::size_t>(j)];
      }
      m.tree_root_.push_back(base);
      m.tree_depth_.push_back(max_depth);
    }
    return true;
  }
};

std::shared_ptr<const ensemble_t> ensemble_t::load_bytes(const void* data, std::size_t size,
                                                         status_t& status, std::string& detail) {
  return model_parser::parse(static_cast<const std::uint8_t*>(data), size, status, detail);
}

std::shared_ptr<const ensemble_t> ensemble_t::load_file(const std::string& path, status_t& status,
                                                        std::string& detail) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    status = status_t::model_missing;
    detail = "no model file " + path;
    return nullptr;
  }
  std::ifstream             f(path, std::ios::binary);
  std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (!f && !f.eof()) {
    status = status_t::bad_file;
    detail = "cannot read " + path;
    return nullptr;
  }
  return load_bytes(buf.data(), buf.size(), status, detail);
}

std::string ensemble_t::head_string(const std::string& key) const {
  for (const auto& kv : head_strings_)
    if (kv.first == key) return kv.second;
  return {};
}

void ensemble_t::prepare(double* row) const {
  for (std::size_t j = 0; j < features_.size(); ++j)
    if (!std::isfinite(row[j])) row[j] = 0.0;
}

double ensemble_t::evaluate(const double* row) const {
  double out;
  evaluate_batch(row, 1, &out);
  return out;
}

void ensemble_t::evaluate_batch(const double* rows, std::size_t n, double* out) const {
  const std::size_t nf  = features_.size();
  auto              run = [&](auto r, const double* x, double* y) {
    forest_walk<decltype(r)::value>(*forest_, values_.data(), tree_root_.data(),
                                    tree_depth_.data(), tree_root_.size(), x, nf, 0.0, y);
  };
  constexpr std::size_t kMax = 16;
  for (; n >= kMax; n -= kMax, rows += kMax * nf, out += kMax)
    run(std::integral_constant<int, kMax>{}, rows, out);
  switch (n) {
#define ORIGAMI_BOOST_FOREST_CASE(r) \
  case r: run(std::integral_constant<int, r>{}, rows, out); break;
    ORIGAMI_BOOST_FOREST_CASE(1) ORIGAMI_BOOST_FOREST_CASE(2) ORIGAMI_BOOST_FOREST_CASE(3)
    ORIGAMI_BOOST_FOREST_CASE(4) ORIGAMI_BOOST_FOREST_CASE(5) ORIGAMI_BOOST_FOREST_CASE(6)
    ORIGAMI_BOOST_FOREST_CASE(7) ORIGAMI_BOOST_FOREST_CASE(8) ORIGAMI_BOOST_FOREST_CASE(9)
    ORIGAMI_BOOST_FOREST_CASE(10) ORIGAMI_BOOST_FOREST_CASE(11) ORIGAMI_BOOST_FOREST_CASE(12)
    ORIGAMI_BOOST_FOREST_CASE(13) ORIGAMI_BOOST_FOREST_CASE(14) ORIGAMI_BOOST_FOREST_CASE(15)
#undef ORIGAMI_BOOST_FOREST_CASE
    default: break;
  }
}

}  // namespace origami::nn::boost
