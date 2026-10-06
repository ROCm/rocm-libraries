// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Writes synthetic MLREC_v2 model images so the tests never depend on the
// shipped weights.

#pragma once

#include "tilewright/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
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

struct ModelSpec {
  tilewright::WeightType weight_type = tilewright::WeightType::Fp32;
  std::string arch                   = "gfxtest";
  std::string feature_hash           = "e7fe4b524851e895";
  std::uint32_t q_dim                = 55;
  std::uint32_t i_dim                = 12;
  std::uint32_t x_dim                = 37;
  double parallel_mi_cu              = 4.0;
  double bw[3]                       = {0.0, 0.01, 0.0};
  double mi_default_cycles           = 32.0;
  std::vector<MiRow> mi_table;
  std::vector<SplitSpec> splits;
  std::vector<CellSpec> cells;
};

struct Image {
  std::vector<unsigned char> bytes;
  // Byte offset of every written field, e.g. "n_cells", "cell0.q_w0", "split1.axis".
  std::map<std::string, std::size_t> at;
  std::size_t header_size = 0;
};

// The seeded tensors of `cell` (dims from the cell, feature dims from the
// engine) with `cell.edit` applied.
CellTensors tensors_of(const CellSpec& cell);

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
