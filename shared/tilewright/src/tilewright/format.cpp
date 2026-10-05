// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "model_impl.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tilewright {
namespace detail {
namespace {

constexpr std::size_t kFixedHeaderBytes = 52;
constexpr std::uint32_t kEndianMarker   = 0x01020304u;
constexpr char kMagicV1[]               = "MLREC_v1";
constexpr char kMagicV2[]               = "MLREC_v2";
constexpr char kTrailer[]               = "MLRECEND";
constexpr char kLfsPointerPrefix[]      = "version https://git-lfs.github.com/spec/";

// Smallest possible split record: three one-byte labels, axis pair, threshold.
constexpr std::size_t kMinSplitBytes = 3 * 3 + 2 + 4;
// Smallest possible cell record before its weights: a one-byte label, dims,
// temperature, the whitening vectors and the signature count.
constexpr std::size_t kMinCellBytes  = 3 + 12 + 4 + 4 * 2 * (kQueryDim + kItemDim + kInterDim) + 4;
constexpr std::uint32_t kMaxLayerDim = 1u << 16;
// The table is searched linearly for every scored kernel.
constexpr std::uint32_t kMaxMiEntries = 1024;

std::uint16_t le16(const unsigned char* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t le32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t le64(const unsigned char* p) {
  return static_cast<std::uint64_t>(le32(p)) | (static_cast<std::uint64_t>(le32(p + 4)) << 32);
}

float f32_of(std::uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

double f64_of(std::uint64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof d);
  return d;
}

// Exponent tests on the raw bits rather than std::isfinite, so the check holds
// even if the translation unit is built with finite-math assumptions.
bool finite32(std::uint32_t bits) { return (bits & 0x7F800000u) != 0x7F800000u; }
bool finite64(std::uint64_t bits) {
  return (bits & 0x7FF0000000000000ull) != 0x7FF0000000000000ull;
}

bool graphic_ascii(const unsigned char* p, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    if (p[i] < 0x21 || p[i] > 0x7E) return false;
  return true;
}

bool starts_with(const unsigned char* data, std::size_t size, const char* prefix) {
  const std::size_t n = std::strlen(prefix);
  return size >= n && std::memcmp(data, prefix, n) == 0;
}

bool is_fnuz(std::int32_t dt) {
  return dt == static_cast<std::int32_t>(DataType::Float8_fnuz) ||
         dt == static_cast<std::int32_t>(DataType::BFloat8_fnuz) ||
         dt == static_cast<std::int32_t>(DataType::Float8BFloat8_fnuz) ||
         dt == static_cast<std::int32_t>(DataType::BFloat8Float8_fnuz);
}

std::uint64_t weight_bytes(WeightType wt, std::uint64_t n) {
  switch (wt) {
    case WeightType::Fp32: return 4 * n;
    case WeightType::Bf16: return 2 * n;
    case WeightType::Int8: return 4 + n;
    case WeightType::Int4: return 4 + (n + 1) / 2;
  }
  return 0;
}

const char* base_tier_m(std::size_t i) {
  static const char* const kTiers[] = {"Tiny", "Small", "Mid", "Large"};
  return kTiers[i];
}

const char* base_tier_k(std::size_t i) {
  static const char* const kTiers[] = {"TinyK", "MidK", "LargeK"};
  return kTiers[i];
}

// Bounds-checked cursor over one region of the image. Every failure records
// the first error, naming the field and its file offset.
class Reader {
 public:
  Reader(const unsigned char* image,
         std::size_t begin,
         std::size_t end,
         std::string* error,
         const char* region)
      : image_(image), pos_(begin), end_(end), error_(error), region_(region) {}

  std::size_t remaining() const { return end_ - pos_; }
  std::size_t pos() const { return pos_; }

  bool fail(const std::string& what) {
    if (error_ != nullptr && error_->empty()) *error_ = what;
    return false;
  }

  bool fail_at(const std::string& what, std::size_t offset) {
    return fail(what + " at offset " + std::to_string(offset));
  }

  const unsigned char* take(std::size_t n, const char* field) {
    if (n > remaining()) {
      fail_at(std::string("truncated ") + region_ + ": " + field, pos_);
      return nullptr;
    }
    const unsigned char* p = image_ + pos_;
    pos_ += n;
    return p;
  }

  bool u8(std::uint8_t* out, const char* field) {
    const unsigned char* p = take(1, field);
    if (p == nullptr) return false;
    *out = p[0];
    return true;
  }

  bool u16(std::uint16_t* out, const char* field) {
    const unsigned char* p = take(2, field);
    if (p == nullptr) return false;
    *out = le16(p);
    return true;
  }

  bool u32(std::uint32_t* out, const char* field) {
    const unsigned char* p = take(4, field);
    if (p == nullptr) return false;
    *out = le32(p);
    return true;
  }

  bool i32(std::int32_t* out, const char* field) {
    std::uint32_t u = 0;
    if (!u32(&u, field)) return false;
    std::memcpy(out, &u, sizeof u);
    return true;
  }

  bool u64(std::uint64_t* out, const char* field) {
    const unsigned char* p = take(8, field);
    if (p == nullptr) return false;
    *out = le64(p);
    return true;
  }

  bool finite_f32(float* out, const char* field) {
    const std::size_t at = pos_;
    std::uint32_t bits   = 0;
    if (!u32(&bits, field)) return false;
    if (!finite32(bits)) return fail_at(std::string("non-finite ") + field, at);
    *out = f32_of(bits);
    return true;
  }

  bool finite_f64(double* out, const char* field) {
    const std::size_t at = pos_;
    std::uint64_t bits   = 0;
    if (!u64(&bits, field)) return false;
    if (!finite64(bits)) return fail_at(std::string("non-finite ") + field, at);
    *out = f64_of(bits);
    return true;
  }

  // u16 length + graphic ASCII bytes, at least one byte long.
  bool label(std::string* out, const char* field) {
    const std::size_t at = pos_;
    std::uint16_t len    = 0;
    if (!u16(&len, field)) return false;
    if (len == 0) return fail_at(std::string("empty ") + field, at);
    const unsigned char* p = take(len, field);
    if (p == nullptr) return false;
    if (!graphic_ascii(p, len)) return fail_at(std::string("non-printable ") + field, at);
    out->assign(reinterpret_cast<const char*>(p), len);
    return true;
  }

  // `count` f32 values that must all be finite.
  bool f32_array(std::size_t count, TensorRef* out, const char* field) {
    const std::size_t at   = pos_;
    const unsigned char* p = take(4 * count, field);
    if (p == nullptr) return false;
    for (std::size_t i = 0; i < count; ++i)
      if (!finite32(le32(p + 4 * i))) return fail_at(std::string("non-finite ") + field, at);
    out->offset = at;
    out->count  = count;
    return true;
  }

  bool weight_array(WeightType wt, std::uint64_t count, TensorRef* out, const char* field) {
    const std::size_t at       = pos_;
    const std::uint64_t nbytes = weight_bytes(wt, count);
    if (nbytes > remaining())
      return fail_at(std::string("truncated ") + region_ + ": " + field, at);
    const unsigned char* p = take(static_cast<std::size_t>(nbytes), field);
    if (wt == WeightType::Fp32) {
      for (std::uint64_t i = 0; i < count; ++i)
        if (!finite32(le32(p + 4 * i))) return fail_at(std::string("non-finite ") + field, at);
    } else if (wt == WeightType::Bf16) {
      for (std::uint64_t i = 0; i < count; ++i)
        if ((le16(p + 2 * i) & 0x7F80u) == 0x7F80u)
          return fail_at(std::string("non-finite ") + field, at);
    } else {
      const std::uint32_t bits = le32(p);
      const double max_level   = wt == WeightType::Int8 ? 128.0 : 8.0;
      if (!finite32(bits) ||
          std::fabs(static_cast<double>(f32_of(bits))) * max_level > static_cast<double>(FLT_MAX))
        return fail_at(std::string("invalid quantization scale of ") + field, at);
    }
    out->offset = at;
    out->count  = static_cast<std::size_t>(count);
    return true;
  }

 private:
  const unsigned char* image_;
  std::size_t pos_;
  std::size_t end_;
  std::string* error_;
  const char* region_;
};

struct RawSplit {
  std::string parent;
  char axis              = 'M';
  std::int32_t threshold = 0;
  std::string lo;
  std::string hi;
};

class Parser {
 public:
  Parser(const unsigned char* data, std::size_t size, std::string* error)
      : data_(data), size_(size), error_(error) {}

  std::unique_ptr<Model> run() {
    if (!check_magic() || !parse_fixed_header() || !parse_header_tail() || !check_payload_crc() ||
        !parse_splits() || !parse_cells() || !build_tree())
      return nullptr;
    model_->payload.assign(data_ + header_size_, data_ + header_size_ + payload_size_);
    for (detail::Cell& cell : model_->cells) rebase(cell);
    model_->init_lazy_state();
    return std::move(model_);
  }

 private:
  bool fail(const std::string& what) {
    if (error_ != nullptr && error_->empty()) *error_ = what;
    return false;
  }

  bool check_magic() {
    if (starts_with(data_, size_, kMagicV1))
      return fail("MLREC_v1 model files are not supported; convert the file to MLREC_v2 with "
                  "training_pipeline/scripts/convert_mlrec_v1_to_v2.py");
    if (starts_with(data_, size_, kLfsPointerPrefix))
      return fail("file is a Git LFS pointer, not a model file");
    if (!starts_with(data_, size_, kMagicV2))
      return fail("not a tilewright model file (bad magic)");
    return true;
  }

  bool parse_fixed_header() {
    Reader r(data_, 8, std::min(size_, kFixedHeaderBytes), error_, "header");
    std::uint32_t endian = 0, header_size = 0, crc = 0;
    std::uint64_t payload_size = 0;
    std::uint8_t wdt = 0, reserved[3] = {0, 0, 0};
    std::uint32_t dims[3] = {0, 0, 0};
    if (!r.u32(&endian, "endian marker") || !r.u32(&header_size, "header_size") ||
        !r.u64(&payload_size, "payload_size") || !r.u32(&crc, "payload_crc32") ||
        !r.u8(&wdt, "weight_dtype") || !r.u8(&reserved[0], "reserved") ||
        !r.u8(&reserved[1], "reserved") || !r.u8(&reserved[2], "reserved") ||
        !r.u32(&dims[0], "q_dim") || !r.u32(&dims[1], "i_dim") || !r.u32(&dims[2], "x_dim") ||
        !r.u32(&n_cells_, "n_cells") || !r.u32(&n_splits_, "n_splits"))
      return false;
    if (endian != kEndianMarker) return fail("bad endian marker");
    if (header_size < kFixedHeaderBytes || header_size > size_)
      return fail("header_size " + std::to_string(header_size) + " out of range");
    if (payload_size > size_ - header_size || size_ - header_size - payload_size != 8)
      return fail("file size " + std::to_string(size_) +
                  " does not match header_size + payload_size + 8 (" + std::to_string(header_size) +
                  " + " + std::to_string(payload_size) + " + 8)");
    if (std::memcmp(data_ + size_ - 8, kTrailer, 8) != 0) return fail("missing MLRECEND trailer");
    if (wdt > static_cast<std::uint8_t>(WeightType::Int4))
      return fail("unknown weight_dtype " + std::to_string(wdt));
    if (reserved[0] != 0 || reserved[1] != 0 || reserved[2] != 0)
      return fail("reserved header bytes are not zero");
    if (dims[0] != kQueryDim || dims[1] != kItemDim || dims[2] != kInterDim)
      return fail("feature dims (" + std::to_string(dims[0]) + ", " + std::to_string(dims[1]) +
                  ", " + std::to_string(dims[2]) + ") do not match the engine (" +
                  std::to_string(kQueryDim) + ", " + std::to_string(kItemDim) + ", " +
                  std::to_string(kInterDim) + ")");
    if (n_cells_ == 0) return fail("model has no cells");
    header_size_        = header_size;
    payload_size_       = static_cast<std::size_t>(payload_size);
    crc_                = crc;
    model_              = std::make_unique<Model>();
    model_->weight_type = static_cast<WeightType>(wdt);
    return true;
  }

  bool parse_header_tail() {
    Reader r(data_, kFixedHeaderBytes, header_size_, error_, "header");
    std::uint16_t len = 0;
    if (!r.u16(&len, "feature_catalog_hash")) return false;
    const unsigned char* hash = r.take(len, "feature_catalog_hash");
    if (hash == nullptr) return false;
    model_->feature_hash.assign(reinterpret_cast<const char*>(hash), len);
    if (model_->feature_hash != kFeatureCatalogHash)
      return fail("feature catalog hash '" + printable(model_->feature_hash) +
                  "' does not match the engine's '" + kFeatureCatalogHash + "'");
    if (!r.label(&model_->arch, "arch")) return false;

    double* bw = model_->bw_coef;
    if (!r.finite_f64(&model_->parallel_mi_cu, "parallel_mi_cu") ||
        !r.finite_f64(&bw[0], "bw_c0") || !r.finite_f64(&bw[1], "bw_c1") ||
        !r.finite_f64(&bw[2], "bw_c2") ||
        !r.finite_f64(&model_->mi_default_cycles, "mi_default_cycles"))
      return false;
    if (!(model_->parallel_mi_cu > 0.0)) return fail("parallel_mi_cu must be positive");
    if (!(model_->mi_default_cycles > 0.0)) return fail("mi_default_cycles must be positive");

    std::uint32_t n_mi = 0;
    if (!r.u32(&n_mi, "n_mi")) return false;
    if (n_mi > r.remaining() / 24) return fail("n_mi " + std::to_string(n_mi) + " exceeds header");
    if (n_mi > kMaxMiEntries)
      return fail("n_mi " + std::to_string(n_mi) + " exceeds " + std::to_string(kMaxMiEntries));
    model_->mi_table.reserve(n_mi);
    for (std::uint32_t i = 0; i < n_mi; ++i) {
      MiEntry e;
      const std::size_t at = r.pos();
      if (!r.u32(&e.m, "mi_m") || !r.u32(&e.n, "mi_n") || !r.u32(&e.k, "mi_k") ||
          !r.i32(&e.dtype, "mi_dtype") || !r.finite_f64(&e.cycles, "mi cycles"))
        return false;
      if (e.dtype < 0 || e.dtype >= static_cast<std::int32_t>(DataType::Count) || is_fnuz(e.dtype))
        return r.fail_at("invalid MI table dtype " + std::to_string(e.dtype), at);
      if (!(e.cycles > 0.0)) return r.fail_at("MI table cycles must be positive", at);
      for (const MiEntry& o : model_->mi_table)
        if (o.m == e.m && o.n == e.n && o.k == e.k && o.dtype == e.dtype)
          return r.fail_at("duplicate MI table entry", at);
      model_->mi_table.push_back(e);
    }
    if (r.remaining() != 0) return fail("header_size does not match the header contents");
    return true;
  }

  bool check_payload_crc() {
    if (crc32(data_ + header_size_, payload_size_) != crc_) return fail("payload CRC mismatch");
    return true;
  }

  bool parse_splits() {
    Reader r(data_, header_size_, header_size_ + payload_size_, error_, "split record");
    if (n_splits_ > payload_size_ / kMinSplitBytes)
      return fail("n_splits " + std::to_string(n_splits_) + " exceeds the payload");
    raw_splits_.reserve(n_splits_);
    std::unordered_map<std::string, std::size_t> parents;
    for (std::uint32_t s = 0; s < n_splits_; ++s) {
      RawSplit split;
      const std::size_t at = r.pos();
      if (!r.label(&split.parent, "split parent label")) return false;
      const unsigned char* axis = r.take(2, "split axis");
      if (axis == nullptr) return false;
      if ((axis[0] != 'M' && axis[0] != 'N' && axis[0] != 'K' && axis[0] != 'B') || axis[1] != 0)
        return r.fail_at("invalid split axis", at);
      split.axis = static_cast<char>(axis[0]);
      if (!r.i32(&split.threshold, "split threshold") || !r.label(&split.lo, "split lo label") ||
          !r.label(&split.hi, "split hi label"))
        return false;
      if (!parents.emplace(split.parent, s).second)
        return r.fail_at("duplicate split parent '" + split.parent + "'", at);
      raw_splits_.push_back(std::move(split));
    }
    payload_pos_ = r.pos();
    return true;
  }

  bool parse_cells() {
    Reader r(data_, payload_pos_, header_size_ + payload_size_, error_, "cell record");
    if (n_cells_ > r.remaining() / kMinCellBytes)
      return fail("n_cells " + std::to_string(n_cells_) + " exceeds the payload");
    const WeightType wt = model_->weight_type;
    model_->cells.reserve(n_cells_);
    for (std::uint32_t c = 0; c < n_cells_; ++c) {
      detail::Cell cell;
      const std::size_t at = r.pos();
      if (!r.label(&cell.label, "cell label") || !r.u32(&cell.embed_dim, "embed_dim") ||
          !r.u32(&cell.hidden_dim, "hidden_dim") || !r.u32(&cell.inter_hidden, "inter_hidden"))
        return false;
      for (std::uint32_t d : {cell.embed_dim, cell.hidden_dim, cell.inter_hidden})
        if (d == 0 || d > kMaxLayerDim)
          return r.fail_at("cell '" + cell.label + "' has an invalid layer width", at);
      if (!r.finite_f32(&cell.temperature, "temperature") ||
          !r.f32_array(kQueryDim, &cell.q_mean, "q_mean") ||
          !r.f32_array(kQueryDim, &cell.q_std, "q_std") ||
          !r.f32_array(kItemDim, &cell.i_mean, "i_mean") ||
          !r.f32_array(kItemDim, &cell.i_std, "i_std") ||
          !r.f32_array(kInterDim, &cell.x_mean, "x_mean") ||
          !r.f32_array(kInterDim, &cell.x_std, "x_std"))
        return false;

      std::uint32_t n_sk = 0;
      if (!r.u32(&n_sk, "signature count")) return false;
      if (n_sk > r.remaining() / 32)
        return r.fail_at("signature count " + std::to_string(n_sk) + " exceeds the payload", at);
      const unsigned char* sigs = r.take(32 * static_cast<std::size_t>(n_sk), "signatures");
      cell.signatures.resize(n_sk);
      for (std::uint32_t s = 0; s < n_sk; ++s)
        for (std::size_t t = 0; t < 8; ++t) {
          const std::uint32_t u = le32(sigs + 32 * s + 4 * t);
          std::memcpy(&cell.signatures[s][t], &u, sizeof u);
        }

      const std::uint64_t q = kQueryDim, i = kItemDim, x = kInterDim;
      const std::uint64_t h = cell.hidden_dim, e = cell.embed_dim, ih = cell.inter_hidden;
      if (!r.weight_array(wt, h * q, &cell.q_w0, "q_w0") ||
          !r.f32_array(cell.hidden_dim, &cell.q_b0, "q_b0") ||
          !r.weight_array(wt, h * h, &cell.q_w2, "q_w2") ||
          !r.f32_array(cell.hidden_dim, &cell.q_b2, "q_b2") ||
          !r.weight_array(wt, e * h, &cell.q_w4, "q_w4") ||
          !r.f32_array(cell.embed_dim, &cell.q_b4, "q_b4") ||
          !r.weight_array(wt, h * i, &cell.i_w0, "i_w0") ||
          !r.f32_array(cell.hidden_dim, &cell.i_b0, "i_b0") ||
          !r.weight_array(wt, e * h, &cell.i_w2, "i_w2") ||
          !r.f32_array(cell.embed_dim, &cell.i_b2, "i_b2") ||
          !r.weight_array(wt, ih * x, &cell.x_w0, "x_w0") ||
          !r.f32_array(cell.inter_hidden, &cell.x_b0, "x_b0") ||
          !r.weight_array(wt, ih, &cell.x_w2, "x_w2") || !r.f32_array(1, &cell.x_b2, "x_b2"))
        return false;

      if (!cell_index_.emplace(cell.label, model_->cells.size()).second)
        return r.fail_at("duplicate cell label '" + cell.label + "'", at);
      model_->cells.push_back(std::move(cell));
    }
    if (r.remaining() != 0) return fail("payload has trailing bytes after the last cell");
    return true;
  }

  int node_of(const std::string& label) {
    auto it = node_index_.find(label);
    if (it != node_index_.end()) return it->second;
    const int id = static_cast<int>(node_labels_.size());
    node_index_.emplace(label, id);
    node_labels_.push_back(label);
    return id;
  }

  bool build_tree() {
    for (const detail::Cell& cell : model_->cells) node_of(cell.label);
    std::vector<detail::Split>& splits = model_->splits;
    splits.reserve(raw_splits_.size());
    std::vector<int> parent_nodes;
    parent_nodes.reserve(raw_splits_.size());
    for (const RawSplit& raw : raw_splits_) {
      detail::Split s;
      s.axis      = raw.axis;
      s.threshold = raw.threshold;
      parent_nodes.push_back(node_of(raw.parent));
      s.lo = node_of(raw.lo);
      s.hi = node_of(raw.hi);
      splits.push_back(s);
    }

    std::vector<detail::Node>& nodes = model_->nodes;
    nodes.assign(node_labels_.size(), detail::Node{});
    for (std::size_t s = 0; s < splits.size(); ++s)
      nodes[static_cast<std::size_t>(parent_nodes[s])].split = static_cast<int>(s);
    if (!acyclic()) return false;

    for (std::size_t n = 0; n < nodes.size(); ++n) {
      std::string cur = node_labels_[n];
      for (;;) {
        auto it = cell_index_.find(cur);
        if (it != cell_index_.end()) {
          nodes[n].cell = static_cast<int>(it->second);
          break;
        }
        const std::size_t hash = cur.rfind('#');
        if (hash == std::string::npos) break;
        cur.resize(hash);
      }
    }

    std::size_t b = 0;
    for (std::size_t mt = 0; mt < 4; ++mt)
      for (std::size_t nt = 0; nt < 4; ++nt)
        for (std::size_t kt = 0; kt < 3; ++kt)
          for (std::size_t bt = 0; bt < 2; ++bt) {
            const std::string label = std::string(base_tier_m(mt)) + "|" + base_tier_m(nt) + "|" +
                                      base_tier_k(kt) + "|" + (bt == 0 ? "Bnone" : "Bany");
            auto it                 = node_index_.find(label);
            model_->base_nodes[b++] = it == node_index_.end() ? -1 : it->second;
          }
    return true;
  }

  // Iterative DFS over parent -> {lo, hi} edges; a back edge is a cycle.
  bool acyclic() {
    const std::vector<detail::Node>& nodes   = model_->nodes;
    const std::vector<detail::Split>& splits = model_->splits;
    std::vector<unsigned char> state(nodes.size(), 0);
    std::vector<std::pair<int, int>> stack;
    for (std::size_t root = 0; root < nodes.size(); ++root) {
      if (state[root] != 0) continue;
      stack.emplace_back(static_cast<int>(root), 0);
      state[root] = 1;
      while (!stack.empty()) {
        auto& top    = stack.back();
        const int sp = nodes[static_cast<std::size_t>(top.first)].split;
        if (sp < 0 || top.second == 2) {
          state[static_cast<std::size_t>(top.first)] = 2;
          stack.pop_back();
          continue;
        }
        const detail::Split& s = splits[static_cast<std::size_t>(sp)];
        const int child        = top.second++ == 0 ? s.lo : s.hi;
        if (state[static_cast<std::size_t>(child)] == 1)
          return fail("split tree has a cycle through '" +
                      node_labels_[static_cast<std::size_t>(child)] + "'");
        if (state[static_cast<std::size_t>(child)] == 0) {
          state[static_cast<std::size_t>(child)] = 1;
          stack.emplace_back(child, 0);
        }
      }
    }
    return true;
  }

  void rebase(detail::Cell& cell) const {
    for (TensorRef* t : {&cell.q_mean, &cell.q_std, &cell.i_mean, &cell.i_std, &cell.x_mean,
                         &cell.x_std,  &cell.q_w0,  &cell.q_b0,   &cell.q_w2,  &cell.q_b2,
                         &cell.q_w4,   &cell.q_b4,  &cell.i_w0,   &cell.i_b0,  &cell.i_w2,
                         &cell.i_b2,   &cell.x_w0,  &cell.x_b0,   &cell.x_w2,  &cell.x_b2})
      t->offset -= header_size_;
  }

  static std::string printable(const std::string& s) {
    std::string out;
    for (char c : s.substr(0, 64)) out += (c >= 0x20 && c <= 0x7E) ? c : '?';
    return out;
  }

  const unsigned char* data_;
  std::size_t size_;
  std::string* error_;
  std::unique_ptr<Model> model_;
  std::size_t header_size_  = 0;
  std::size_t payload_size_ = 0;
  std::size_t payload_pos_  = 0;
  std::uint32_t crc_        = 0;
  std::uint32_t n_cells_    = 0;
  std::uint32_t n_splits_   = 0;
  std::vector<RawSplit> raw_splits_;
  std::unordered_map<std::string, std::size_t> cell_index_;
  std::unordered_map<std::string, int> node_index_;
  std::vector<std::string> node_labels_;
};

struct CrcTables {
  std::uint32_t t[8][256];
  CrcTables() {
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
      t[0][i] = c;
    }
    for (std::size_t s = 1; s < 8; ++s)
      for (std::size_t i = 0; i < 256; ++i)
        t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFFu];
  }
};

void read_f32(const std::vector<unsigned char>& payload,
              const TensorRef& t,
              std::vector<float>* out) {
  out->resize(t.count);
  for (std::size_t i = 0; i < t.count; ++i) (*out)[i] = f32_of(le32(&payload[t.offset + 4 * i]));
}

void read_scale(const std::vector<unsigned char>& payload,
                const TensorRef& t,
                std::vector<float>* out) {
  read_f32(payload, t, out);
  for (float& s : *out) s = (s < 1e-6f) ? 1.0f : s;
}

void read_weights(const std::vector<unsigned char>& payload,
                  WeightType wt,
                  const TensorRef& t,
                  std::vector<float>* out) {
  const unsigned char* p = &payload[t.offset];
  const std::size_t n    = t.count;
  out->resize(n);
  float* w = out->data();
  switch (wt) {
    case WeightType::Fp32:
      for (std::size_t j = 0; j < n; ++j) w[j] = f32_of(le32(p + 4 * j));
      break;
    case WeightType::Bf16:
      for (std::size_t j = 0; j < n; ++j)
        w[j] = f32_of(static_cast<std::uint32_t>(le16(p + 2 * j)) << 16);
      break;
    case WeightType::Int8: {
      const float scale = f32_of(le32(p));
      for (std::size_t j = 0; j < n; ++j)
        w[j] = scale * static_cast<float>(static_cast<std::int8_t>(p[4 + j]));
      break;
    }
    case WeightType::Int4: {
      const float scale = f32_of(le32(p));
      for (std::size_t j = 0; j < n; ++j) {
        const unsigned char byte = p[4 + (j >> 1)];
        const int nib            = (j & 1) ? (byte >> 4) : (byte & 0x0F);
        w[j]                     = scale * static_cast<float>(nib - 8);
      }
      break;
    }
  }
}

std::unique_ptr<const CellWeights> dequantize(const Model& model, const Cell& cell) {
  const std::vector<unsigned char>& p = model.payload;
  const WeightType wt                 = model.weight_type;
  auto w                              = std::make_unique<CellWeights>();
  read_f32(p, cell.q_mean, &w->q_mean);
  read_scale(p, cell.q_std, &w->q_scale);
  read_f32(p, cell.i_mean, &w->i_mean);
  read_scale(p, cell.i_std, &w->i_scale);
  read_f32(p, cell.x_mean, &w->x_mean);
  read_scale(p, cell.x_std, &w->x_scale);
  read_weights(p, wt, cell.q_w0, &w->q_w0);
  read_f32(p, cell.q_b0, &w->q_b0);
  read_weights(p, wt, cell.q_w2, &w->q_w2);
  read_f32(p, cell.q_b2, &w->q_b2);
  read_weights(p, wt, cell.q_w4, &w->q_w4);
  read_f32(p, cell.q_b4, &w->q_b4);
  read_weights(p, wt, cell.i_w0, &w->i_w0);
  read_f32(p, cell.i_b0, &w->i_b0);
  read_weights(p, wt, cell.i_w2, &w->i_w2);
  read_f32(p, cell.i_b2, &w->i_b2);
  read_weights(p, wt, cell.x_w0, &w->x_w0);
  read_f32(p, cell.x_b0, &w->x_b0);
  read_weights(p, wt, cell.x_w2, &w->x_w2);
  w->x_b2        = f32_of(le32(&p[cell.x_b2.offset]));
  w->temperature = std::max(std::fabs(cell.temperature), 0.1f);
  return w;
}

}  // namespace

std::uint32_t crc32(const unsigned char* data, std::size_t size) noexcept {
  static const CrcTables tables;
  const auto& t     = tables.t;
  std::uint32_t crc = 0xFFFFFFFFu;
  while (size >= 8) {
    const std::uint32_t lo = le32(data) ^ crc;
    const std::uint32_t hi = le32(data + 4);
    crc = t[7][lo & 0xFFu] ^ t[6][(lo >> 8) & 0xFFu] ^ t[5][(lo >> 16) & 0xFFu] ^ t[4][lo >> 24] ^
          t[3][hi & 0xFFu] ^ t[2][(hi >> 8) & 0xFFu] ^ t[1][(hi >> 16) & 0xFFu] ^ t[0][hi >> 24];
    data += 8;
    size -= 8;
  }
  while (size-- > 0) crc = t[0][(crc ^ *data++) & 0xFFu] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

std::unique_ptr<Model> parse_model(const unsigned char* data,
                                   std::size_t size,
                                   std::string* error) {
  return Parser(data, size, error).run();
}

}  // namespace detail

void Model::init_lazy_state() { lazy_ = std::make_unique<LazyCell[]>(cells.size()); }

const detail::CellWeights& Model::weights(std::size_t cell) const {
  LazyCell& lazy = lazy_[cell];
  std::call_once(lazy.once, [&] { lazy.weights = detail::dequantize(*this, cells[cell]); });
  return *lazy.weights;
}

}  // namespace tilewright
