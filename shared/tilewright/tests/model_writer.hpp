// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Writes synthetic MLREC_v2 and MLREC_v3 model images so the tests never
// depend on the shipped weights.

#pragma once

#include "model_impl.hpp"
#include "tilewright/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tilewright_test {

using Signature = std::array<int, 8>;

// fp32 values of one cell before they are stored at the model's weight type.
struct CellTensors {
  float temperature = 1.0f;
  std::vector<float> q_mean, q_std, i_mean, i_std, x_mean, x_std;
  std::vector<float> q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  std::vector<float> i_w0, i_b0, i_w2, i_b2;
  std::vector<float> x_w0, x_b0, x_w2, x_b2;
};

struct CellSpec {
  std::string label;
  std::uint32_t embed_dim    = 8;
  std::uint32_t hidden_dim   = 16;
  std::uint32_t inter_hidden = 8;
  std::vector<Signature> signatures;
  std::vector<std::uint64_t> view_key_hashes;  // MLREC_v3
  // MLREC_v3 with clip: raw feature bounds; empty means unbounded.
  std::vector<float> q_lo, q_hi, i_lo, i_hi, x_lo, x_hi;
  std::uint64_t seed = 1;
  // Applied to the seeded random tensors before they are written.
  std::function<void(CellTensors&)> edit;
};

struct SplitSpec {
  std::string parent;
  char axis              = 'M';
  std::int32_t threshold = 0;
  std::string lo;
  std::string hi;
};

struct MiRow {
  std::uint32_t m, n, k;
  std::int32_t dtype;
  double cycles;
};

// One MLREC_v3 item-attribute spec entry; transform 3 is one-hot over values.
struct AttrSpec {
  std::string name;
  std::int64_t default_value = 0;
  std::uint8_t transform     = 0;
  std::vector<std::int64_t> values;
};

struct ModelSpec {
  int version                        = 2;
  tilewright::WeightType weight_type = tilewright::WeightType::Fp32;
  std::string arch                   = "gfxtest";
  std::string feature_hash           = "e7fe4b524851e895";
  std::uint32_t q_dim                = 55;
  std::uint32_t i_dim                = 12;
  std::uint32_t x_dim                = 37;
  double parallel_mi_cu              = 4.0;
  double bw[3]                       = {0.0, 0.01, 0.0};
  double mi_default_cycles           = 32.0;
  std::uint8_t serving_flags         = 0;
  std::uint16_t cascade_keys         = 0;
  std::vector<MiRow> mi_table;
  std::vector<SplitSpec> splits;
  std::vector<CellSpec> cells;

  // MLREC_v3 header fields.
  std::uint32_t v3_serving_flags = 0;
  std::uint32_t v3_cascade_keys  = 0;
  double min_cu_fraction         = 1.0;
  double max_cu_fraction         = 1.0;
  std::uint8_t schedules         = 1;
  std::uint8_t clip              = 0;
  std::vector<AttrSpec> attributes;
  // Written instead of the spec's hash when nonzero.
  std::uint64_t spec_hash_override = 0;
  // MLREC_v3 provenance entries (key, value), written when not empty, then
  // `header_extra` at the end of the header.
  std::vector<std::pair<std::string, std::string>> provenance;
  std::vector<unsigned char> header_extra;
};

// The MLREC_v3 catalog hash and dims before attribute features.
constexpr const char* kV3Hash = "b8b9f4a518add96f";
constexpr std::uint32_t kV3Q  = 59;
constexpr std::uint32_t kV3I  = 8;
constexpr std::uint32_t kV3X  = 44;

// `attributes` as the engine's attribute spec.
std::vector<tilewright::detail::AttributeFeature> attribute_features(
    const std::vector<AttrSpec>& attributes);

// The feature_spec_hash MLREC_v3 stores for `attributes`, as this engine
// computes it.
std::uint64_t spec_hash(const std::vector<AttrSpec>& attributes);

// Item features `attributes` add.
std::uint32_t attribute_dims(const std::vector<AttrSpec>& attributes);

// `spec` as an MLREC_v3 model with `attributes` (dims set accordingly).
ModelSpec as_v3(ModelSpec spec, std::vector<AttrSpec> attributes = {});

struct Image {
  std::vector<unsigned char> bytes;
  // Byte offset of every written field, e.g. "n_cells", "cell0.q_w0", "split1.axis".
  std::map<std::string, std::size_t> at;
  std::size_t header_size = 0;
};

// The seeded tensors of `cell` (dims from the cell, feature dims from the
// engine) with `cell.edit` applied.
CellTensors tensors_of(const CellSpec& cell);
CellTensors tensors_of(const CellSpec& cell, std::size_t q, std::size_t i, std::size_t x);

Image write_model(const ModelSpec& spec);

// Recomputes the payload CRC after the payload bytes were patched.
void refresh_crc(Image& image);

void put_u16(Image& image, std::size_t offset, std::uint16_t v);
void put_u32(Image& image, std::size_t offset, std::uint32_t v);
void put_u64(Image& image, std::size_t offset, std::uint64_t v);
void put_f32(Image& image, std::size_t offset, float v);
void put_f64(Image& image, std::size_t offset, double v);

// `spec` with fp32 weights equal to what the engine dequantizes from `spec`'s
// stored weights, so both score identically.
ModelSpec dequantized(const ModelSpec& spec);

// A model with one small cell per base grid label (96 cells, grid order).
ModelSpec grid_model(tilewright::WeightType wt = tilewright::WeightType::Fp32);

std::string base_label(std::size_t m, std::size_t n, std::size_t k, std::size_t batch);

}  // namespace tilewright_test
