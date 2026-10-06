// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "model_writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace tilewright_test {
namespace {

constexpr std::size_t kQ = 55;
constexpr std::size_t kI = 12;
constexpr std::size_t kX = 37;

struct Rng {
  std::uint64_t state;
  std::uint64_t next() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform in [-1, 1).
  float uniform() {
    return static_cast<float>(static_cast<double>(next() >> 40) / 8388608.0 - 1.0);
  }
};

std::vector<float> fill(Rng& rng, std::size_t n, float scale) {
  std::vector<float> v(n);
  for (float& x : v) x = scale * rng.uniform();
  return v;
}

std::vector<float> fill_std(Rng& rng, std::size_t n) {
  std::vector<float> v(n);
  for (float& x : v) x = 0.5f + std::fabs(rng.uniform());
  return v;
}

std::uint32_t crc32(const unsigned char* p, std::size_t n) {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    crc ^= p[i];
    for (int b = 0; b < 8; ++b) crc = (crc & 1u) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
  }
  return crc ^ 0xFFFFFFFFu;
}

float quant_scale(const std::vector<float>& v, float levels) {
  float amax = 0.0f;
  for (float x : v) amax = std::max(amax, std::fabs(x));
  return amax > 0.0f ? amax / levels : 1.0f;
}

int quant_level(float v, float scale, int limit) {
  const long q = std::lround(v / scale);
  return static_cast<int>(std::max<long>(-limit, std::min<long>(limit, q)));
}

class Writer {
 public:
  explicit Writer(Image& image) : image_(image) {}

  void mark(const std::string& name) { image_.at.emplace(name, image_.bytes.size()); }

  void raw(const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    image_.bytes.insert(image_.bytes.end(), b, b + n);
  }
  void u8(std::uint8_t v) { image_.bytes.push_back(v); }
  void u16(std::uint16_t v) {
    for (int i = 0; i < 2; ++i) u8(static_cast<std::uint8_t>(v >> (8 * i)));
  }
  void u32(std::uint32_t v) {
    for (int i = 0; i < 4; ++i) u8(static_cast<std::uint8_t>(v >> (8 * i)));
  }
  void u64(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) u8(static_cast<std::uint8_t>(v >> (8 * i)));
  }
  void i32(std::int32_t v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    u32(u);
  }
  void f32(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    u32(u);
  }
  void f64(double v) {
    std::uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    u64(u);
  }
  void lenstr(const std::string& s) {
    u16(static_cast<std::uint16_t>(s.size()));
    raw(s.data(), s.size());
  }
  void f32s(const std::vector<float>& v) {
    for (float x : v) f32(x);
  }

  void weights(tilewright::WeightType wt, const std::vector<float>& v) {
    switch (wt) {
      case tilewright::WeightType::Fp32: f32s(v); break;
      case tilewright::WeightType::Bf16:
        for (float x : v) {
          std::uint32_t u;
          std::memcpy(&u, &x, sizeof u);
          u16(static_cast<std::uint16_t>(u >> 16));
        }
        break;
      case tilewright::WeightType::Int8: {
        const float scale = quant_scale(v, 127.0f);
        f32(scale);
        for (float x : v)
          u8(static_cast<std::uint8_t>(static_cast<std::int8_t>(quant_level(x, scale, 127))));
        break;
      }
      case tilewright::WeightType::Int4: {
        const float scale = quant_scale(v, 7.0f);
        f32(scale);
        for (std::size_t j = 0; j < v.size(); j += 2) {
          const int lo = quant_level(v[j], scale, 7) + 8;
          const int hi = j + 1 < v.size() ? quant_level(v[j + 1], scale, 7) + 8 : 0;
          u8(static_cast<std::uint8_t>(lo | (hi << 4)));
        }
        break;
      }
    }
  }

  std::size_t size() const { return image_.bytes.size(); }

 private:
  Image& image_;
};

std::vector<float> roundtrip(tilewright::WeightType wt, const std::vector<float>& v) {
  std::vector<float> out(v.size());
  switch (wt) {
    case tilewright::WeightType::Fp32: out = v; break;
    case tilewright::WeightType::Bf16:
      for (std::size_t j = 0; j < v.size(); ++j) {
        std::uint32_t u;
        std::memcpy(&u, &v[j], sizeof u);
        u &= 0xFFFF0000u;
        std::memcpy(&out[j], &u, sizeof u);
      }
      break;
    case tilewright::WeightType::Int8: {
      const float scale = quant_scale(v, 127.0f);
      for (std::size_t j = 0; j < v.size(); ++j)
        out[j] = scale * static_cast<float>(quant_level(v[j], scale, 127));
      break;
    }
    case tilewright::WeightType::Int4: {
      const float scale = quant_scale(v, 7.0f);
      for (std::size_t j = 0; j < v.size(); ++j)
        out[j] = scale * static_cast<float>((quant_level(v[j], scale, 7) + 8) - 8);
      break;
    }
  }
  return out;
}

}  // namespace

CellTensors tensors_of(const CellSpec& cell) {
  Rng rng{cell.seed * 0x2545F4914F6CDD1Dull + 17};
  const std::size_t e = cell.embed_dim, h = cell.hidden_dim, ih = cell.inter_hidden;
  CellTensors t;
  t.temperature = 0.75f + 0.25f * rng.uniform();
  t.q_mean      = fill(rng, kQ, 2.0f);
  t.q_std       = fill_std(rng, kQ);
  t.i_mean      = fill(rng, kI, 0.5f);
  t.i_std       = fill_std(rng, kI);
  t.x_mean      = fill(rng, kX, 2.0f);
  t.x_std       = fill_std(rng, kX);
  // Exercise the std < 1e-6 -> 1 whitening rule; an item feature with a std
  // below 1e-3 is constant instead (the constant-feature tests set one).
  t.q_std[3] = 1e-7f;
  t.i_std[1] = 1.0f;
  t.x_std[5] = 1e-8f;
  t.q_w0     = fill(rng, h * kQ, 0.2f);
  t.q_b0     = fill(rng, h, 0.1f);
  t.q_w2     = fill(rng, h * h, 0.3f);
  t.q_b2     = fill(rng, h, 0.1f);
  t.q_w4     = fill(rng, e * h, 0.3f);
  t.q_b4     = fill(rng, e, 0.1f);
  t.i_w0     = fill(rng, h * kI, 0.5f);
  t.i_b0     = fill(rng, h, 0.1f);
  t.i_w2     = fill(rng, e * h, 0.3f);
  t.i_b2     = fill(rng, e, 0.1f);
  t.x_w0     = fill(rng, ih * kX, 0.2f);
  t.x_b0     = fill(rng, ih, 0.1f);
  t.x_w2     = fill(rng, ih, 0.3f);
  t.x_b2     = fill(rng, 1, 0.1f);
  if (cell.edit) cell.edit(t);
  return t;
}

Image write_model(const ModelSpec& spec) {
  Image image;
  Writer w(image);
  w.mark("magic");
  w.raw("MLREC_v2", 8);
  w.mark("endian");
  w.u32(0x01020304u);
  w.mark("header_size");
  w.u32(0);
  w.mark("payload_size");
  w.u64(0);
  w.mark("crc");
  w.u32(0);
  w.mark("weight_dtype");
  w.u8(static_cast<std::uint8_t>(spec.weight_type));
  w.mark("reserved");
  w.u8(0);
  w.u8(0);
  w.u8(0);
  w.mark("q_dim");
  w.u32(spec.q_dim);
  w.mark("i_dim");
  w.u32(spec.i_dim);
  w.mark("x_dim");
  w.u32(spec.x_dim);
  w.mark("n_cells");
  w.u32(static_cast<std::uint32_t>(spec.cells.size()));
  w.mark("n_splits");
  w.u32(static_cast<std::uint32_t>(spec.splits.size()));
  w.mark("hash");
  w.lenstr(spec.feature_hash);
  w.mark("arch");
  w.lenstr(spec.arch);
  w.mark("parallel_mi_cu");
  w.f64(spec.parallel_mi_cu);
  w.mark("bw_c0");
  w.f64(spec.bw[0]);
  w.mark("bw_c1");
  w.f64(spec.bw[1]);
  w.mark("bw_c2");
  w.f64(spec.bw[2]);
  w.mark("mi_default");
  w.f64(spec.mi_default_cycles);
  w.mark("n_mi");
  w.u32(static_cast<std::uint32_t>(spec.mi_table.size()));
  for (std::size_t i = 0; i < spec.mi_table.size(); ++i) {
    const std::string p = "mi" + std::to_string(i) + ".";
    const MiRow& r      = spec.mi_table[i];
    w.mark(p + "m");
    w.u32(r.m);
    w.mark(p + "n");
    w.u32(r.n);
    w.mark(p + "k");
    w.u32(r.k);
    w.mark(p + "dtype");
    w.i32(r.dtype);
    w.mark(p + "cycles");
    w.f64(r.cycles);
  }
  image.header_size = w.size();

  w.mark("payload");
  for (std::size_t s = 0; s < spec.splits.size(); ++s) {
    const std::string p = "split" + std::to_string(s) + ".";
    const SplitSpec& sp = spec.splits[s];
    w.mark(p + "parent");
    w.lenstr(sp.parent);
    w.mark(p + "axis");
    w.u8(static_cast<std::uint8_t>(sp.axis));
    w.u8(0);
    w.mark(p + "threshold");
    w.i32(sp.threshold);
    w.mark(p + "lo");
    w.lenstr(sp.lo);
    w.mark(p + "hi");
    w.lenstr(sp.hi);
  }
  for (std::size_t c = 0; c < spec.cells.size(); ++c) {
    const std::string p  = "cell" + std::to_string(c) + ".";
    const CellSpec& cell = spec.cells[c];
    const CellTensors t  = tensors_of(cell);
    w.mark(p + "label");
    w.lenstr(cell.label);
    w.mark(p + "embed_dim");
    w.u32(cell.embed_dim);
    w.mark(p + "hidden_dim");
    w.u32(cell.hidden_dim);
    w.mark(p + "inter_hidden");
    w.u32(cell.inter_hidden);
    w.mark(p + "temperature");
    w.f32(t.temperature);
    const std::pair<const char*, const std::vector<float>*> vectors[] = {{"q_mean", &t.q_mean},
                                                                         {"q_std", &t.q_std},
                                                                         {"i_mean", &t.i_mean},
                                                                         {"i_std", &t.i_std},
                                                                         {"x_mean", &t.x_mean},
                                                                         {"x_std", &t.x_std}};
    for (const auto& v : vectors) {
      w.mark(p + v.first);
      w.f32s(*v.second);
    }
    w.mark(p + "n_sk");
    w.u32(static_cast<std::uint32_t>(cell.signatures.size()));
    w.mark(p + "signatures");
    for (const Signature& s : cell.signatures)
      for (int x : s) w.i32(x);
    const std::pair<const char*, const std::vector<float>*> tensors[] = {{"q_w0", &t.q_w0},
                                                                         {"q_b0", &t.q_b0},
                                                                         {"q_w2", &t.q_w2},
                                                                         {"q_b2", &t.q_b2},
                                                                         {"q_w4", &t.q_w4},
                                                                         {"q_b4", &t.q_b4},
                                                                         {"i_w0", &t.i_w0},
                                                                         {"i_b0", &t.i_b0},
                                                                         {"i_w2", &t.i_w2},
                                                                         {"i_b2", &t.i_b2},
                                                                         {"x_w0", &t.x_w0},
                                                                         {"x_b0", &t.x_b0},
                                                                         {"x_w2", &t.x_w2},
                                                                         {"x_b2", &t.x_b2}};
    for (const auto& v : tensors) {
      w.mark(p + v.first);
      const bool is_bias = v.first[2] == 'b';
      if (is_bias)
        w.f32s(*v.second);
      else
        w.weights(spec.weight_type, *v.second);
    }
  }
  const std::size_t payload_size = w.size() - image.header_size;
  w.mark("trailer");
  w.raw("MLRECEND", 8);

  put_u32(image, image.at.at("header_size"), static_cast<std::uint32_t>(image.header_size));
  put_u64(image, image.at.at("payload_size"), payload_size);
  refresh_crc(image);
  return image;
}

void refresh_crc(Image& image) {
  std::uint64_t payload_size = 0;
  std::memcpy(&payload_size, &image.bytes[image.at.at("payload_size")], sizeof payload_size);
  const std::size_t begin = image.header_size;
  const std::size_t n     = std::min<std::size_t>(payload_size, image.bytes.size() - begin);
  put_u32(image, image.at.at("crc"), crc32(image.bytes.data() + begin, n));
}

void put_u16(Image& image, std::size_t offset, std::uint16_t v) {
  for (int i = 0; i < 2; ++i) image.bytes[offset + i] = static_cast<unsigned char>(v >> (8 * i));
}

void put_u32(Image& image, std::size_t offset, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) image.bytes[offset + i] = static_cast<unsigned char>(v >> (8 * i));
}

void put_u64(Image& image, std::size_t offset, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) image.bytes[offset + i] = static_cast<unsigned char>(v >> (8 * i));
}

void put_f32(Image& image, std::size_t offset, float v) {
  std::uint32_t u;
  std::memcpy(&u, &v, sizeof u);
  put_u32(image, offset, u);
}

void put_f64(Image& image, std::size_t offset, double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  put_u64(image, offset, u);
}

ModelSpec dequantized(const ModelSpec& spec) {
  ModelSpec out                   = spec;
  out.weight_type                 = tilewright::WeightType::Fp32;
  const tilewright::WeightType wt = spec.weight_type;
  for (CellSpec& cell : out.cells) {
    std::function<void(CellTensors&)> original = cell.edit;
    cell.edit                                  = [original, wt](CellTensors& t) {
      if (original) original(t);
      for (std::vector<float>* m : {&t.q_w0, &t.q_w2, &t.q_w4, &t.i_w0, &t.i_w2, &t.x_w0, &t.x_w2})
        *m = roundtrip(wt, *m);
    };
  }
  return out;
}

std::string base_label(std::size_t m, std::size_t n, std::size_t k, std::size_t batch) {
  const auto mn = [](std::size_t v) -> std::string {
    return v <= 32 ? "Tiny" : v <= 128 ? "Small" : v <= 512 ? "Mid" : "Large";
  };
  const std::string kt = k <= 32 ? "TinyK" : k <= 512 ? "MidK" : "LargeK";
  return mn(m) + "|" + mn(n) + "|" + kt + "|" + (batch == 1 ? "Bnone" : "Bany");
}

ModelSpec grid_model(tilewright::WeightType wt) {
  ModelSpec spec;
  spec.weight_type   = wt;
  const char* mn[]   = {"Tiny", "Small", "Mid", "Large"};
  const char* kt[]   = {"TinyK", "MidK", "LargeK"};
  std::uint64_t seed = 1;
  for (const char* m : mn)
    for (const char* n : mn)
      for (const char* k : kt)
        for (const char* b : {"Bnone", "Bany"}) {
          CellSpec cell;
          cell.label        = std::string(m) + "|" + n + "|" + k + "|" + b;
          cell.embed_dim    = 4;
          cell.hidden_dim   = 8;
          cell.inter_hidden = 4;
          cell.seed         = seed++;
          spec.cells.push_back(cell);
        }
  return spec;
}

}  // namespace tilewright_test
