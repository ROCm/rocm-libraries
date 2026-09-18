// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/twrec/twrec_loader.hpp"

#include "origami/nn/twrec/b64_decode.hpp"
#include "origami/nn/twrec/load_limits.hpp"
#include "origami/nn/twrec/load_validate.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <unordered_map>

namespace origami::nn::twrec {
namespace {

std::string parent_dir(const std::string& path) {
  const std::filesystem::path p(path);
  const auto parent = p.parent_path();
  if (parent.empty()) return ".";
  return parent.string();
}

std::string join_path(const std::string& dir, const std::string& file) {
  return (std::filesystem::path(dir) / file).string();
}

bool read_float_vector(const YAML::Node& node, std::size_t expected, std::vector<float>* out) {
  if (!node || !node.IsSequence()) return false;
  if (node.size() != expected) return false;
  out->resize(expected);
  for (std::size_t i = 0; i < expected; ++i) {
    const float v = node[i].as<float>();
    if (!std::isfinite(v)) return false;
    (*out)[i] = v;
  }
  return true;
}

bool read_smart_k(const YAML::Node& node, std::vector<std::array<int, 8>>* out) {
  if (!node || !node.IsSequence()) return false;
  if (node.size() > detail::kMaxSmartK) return false;
  out->clear();
  out->reserve(node.size());
  for (const auto& row : node) {
    if (!row.IsSequence() || row.size() != 8) return false;
    std::array<int, 8> sig{};
    for (int t = 0; t < 8; ++t) sig[t] = row[t].as<int>();
    out->push_back(sig);
  }
  return true;
}

bool load_tensor_b64(const YAML::Node& tensor,
                     const char* expected_dtype,
                     std::size_t expected_count,
                     std::vector<float>* out) {
  if (!tensor || !tensor.IsMap()) return false;
  const std::string dtype = tensor["dtype"].as<std::string>();
  if (dtype != expected_dtype) return false;
  const std::string b64 = tensor["data_b64"].as<std::string>();
  if (b64.size() > detail::kMaxB64Decoded * 2) return false;
  std::vector<std::uint8_t> raw;
  if (!detail::b64_decode(b64, &raw)) return false;
  if (dtype == "int4") return detail::decode_int4_tensor(raw, expected_count, out);
  if (dtype == "fp32") return detail::decode_fp32_tensor(raw, expected_count, out);
  return false;
}

bool load_cell_weights(const YAML::Node& tensors,
                       const detail::WeightCounts& wc,
                       detail::CellModel* cm) {
  if (!tensors || !tensors.IsMap()) return false;
  return load_tensor_b64(tensors["q_w0"], "int4", wc.q_w0, &cm->q_w0) &&
         load_tensor_b64(tensors["q_b0"], "fp32", wc.q_b0, &cm->q_b0) &&
         load_tensor_b64(tensors["q_w2"], "int4", wc.q_w2, &cm->q_w2) &&
         load_tensor_b64(tensors["q_b2"], "fp32", wc.q_b2, &cm->q_b2) &&
         load_tensor_b64(tensors["q_w4"], "int4", wc.q_w4, &cm->q_w4) &&
         load_tensor_b64(tensors["q_b4"], "fp32", wc.q_b4, &cm->q_b4) &&
         load_tensor_b64(tensors["i_w0"], "int4", wc.i_w0, &cm->i_w0) &&
         load_tensor_b64(tensors["i_b0"], "fp32", wc.i_b0, &cm->i_b0) &&
         load_tensor_b64(tensors["i_w2"], "int4", wc.i_w2, &cm->i_w2) &&
         load_tensor_b64(tensors["i_b2"], "fp32", wc.i_b2, &cm->i_b2) &&
         load_tensor_b64(tensors["x_w0"], "int4", wc.x_w0, &cm->x_w0) &&
         load_tensor_b64(tensors["x_b0"], "fp32", wc.x_b0, &cm->x_b0) &&
         load_tensor_b64(tensors["x_w2"], "int4", wc.x_w2, &cm->x_w2) &&
         load_tensor_b64(tensors["x_b2"], "fp32", wc.x_b2, &cm->x_b2);
}

bool parse_manifest(const YAML::Node& root,
                    const std::string& manifest_dir,
                    detail::LoadedModel* out) {
  if (!root.IsMap()) return false;
  if (root["schema_version"].as<std::uint32_t>() != detail::kTwrecSchemaVersion) return false;
  if (root["format"].as<std::string>() != "TWREC_v1") return false;

  const YAML::Node meta = root["metadata"];
  if (!meta) return false;
  out->arch               = meta["arch"].as<std::string>();
  out->feature_names_hash = meta["feature_names_hash"].as<std::string>();
  if (!detail::validate_string_len(out->arch)) return false;
  if (!detail::validate_string_len(out->feature_names_hash)) return false;

  const YAML::Node feat = root["features"];
  if (!feat) return false;
  out->q_dim = feat["query_dim"].as<std::uint32_t>();
  out->i_dim = feat["item_dim"].as<std::uint32_t>();
  out->x_dim = feat["interaction_dim"].as<std::uint32_t>();
  if (!detail::validate_feature_dims(out->q_dim, out->i_dim, out->x_dim)) return false;

  out->splits.clear();
  const YAML::Node splits = root["routing"]["splits"];
  if (!splits || !splits.IsSequence()) return false;
  if (splits.size() > detail::kMaxSplits) return false;
  for (const auto& s : splits) {
    detail::SplitRule r;
    r.cell        = s["parent"].as<std::string>();
    const auto ax = s["axis"].as<std::string>();
    if (ax.size() != 1) return false;
    r.axis      = ax[0];
    r.threshold = s["threshold"].as<int>();
    r.lo_label  = s["lo"].as<std::string>();
    r.hi_label  = s["hi"].as<std::string>();
    if (!detail::validate_split_rule(r)) return false;
    auto [it, inserted] = out->splits.emplace(r.cell, std::move(r));
    if (!inserted) return false;
  }

  const std::string sidecar_name = root["weights_sidecar"].as<std::string>();
  if (!detail::validate_string_len(sidecar_name)) return false;
  const std::string sidecar_path = join_path(manifest_dir, sidecar_name);

  YAML::Node wts;
  try {
    wts = YAML::LoadFile(sidecar_path);
  } catch (...) { return false; }
  if (!wts.IsMap()) return false;
  if (wts["schema_version"].as<std::uint32_t>() != detail::kTwrecSchemaVersion) return false;
  if (wts["format"].as<std::string>() != "TWREC_WTS_v1") return false;
  if (wts["encoding"].as<std::string>() != "int4_b64_fp32_b64") return false;

  std::unordered_map<std::string, YAML::Node> wts_by_label;
  const YAML::Node wcells = wts["cells"];
  if (!wcells || !wcells.IsSequence()) return false;
  for (const auto& wc : wcells) {
    const std::string label = wc["label"].as<std::string>();
    if (!detail::validate_string_len(label)) return false;
    auto [it, inserted] = wts_by_label.emplace(label, wc["tensors"]);
    if (!inserted) return false;
  }

  out->cells.clear();
  const YAML::Node cells = root["cells"];
  if (!cells || !cells.IsSequence()) return false;
  if (cells.size() > detail::kMaxCells) return false;

  for (const auto& c : cells) {
    detail::CellModel cm;
    cm.label        = c["label"].as<std::string>();
    cm.embed_dim    = c["embed_dim"].as<std::uint32_t>();
    cm.hidden_dim   = c["hidden_dim"].as<std::uint32_t>();
    cm.inter_hidden = c["inter_hidden"].as<std::uint32_t>();
    cm.temperature  = c["temperature"].as<float>();
    if (!detail::validate_cell_hyperparams(cm)) return false;

    const YAML::Node wh = c["whitening"];
    if (!read_float_vector(wh["query"]["mean"], out->q_dim, &cm.q_mean)) return false;
    if (!read_float_vector(wh["query"]["std"], out->q_dim, &cm.q_std)) return false;
    if (!read_float_vector(wh["item"]["mean"], out->i_dim, &cm.i_mean)) return false;
    if (!read_float_vector(wh["item"]["std"], out->i_dim, &cm.i_std)) return false;
    if (!read_float_vector(wh["interaction"]["mean"], out->x_dim, &cm.x_mean)) return false;
    if (!read_float_vector(wh["interaction"]["std"], out->x_dim, &cm.x_std)) return false;
    if (!read_smart_k(c["smart_k"], &cm.smart_k_signatures)) return false;

    auto wit = wts_by_label.find(cm.label);
    if (wit == wts_by_label.end()) return false;
    const auto wc = detail::cell_weight_counts(
        out->q_dim, out->i_dim, out->x_dim, cm.hidden_dim, cm.embed_dim, cm.inter_hidden);
    if (!load_cell_weights(wit->second, wc, &cm)) return false;

    out->cells.push_back(std::move(cm));
  }

  return true;
}

}  // namespace

bool load_twrec_yaml(const std::string& manifest_path, detail::LoadedModel* out) {
  if (!out) return false;
  out->cells.clear();
  out->splits.clear();
  out->cell_index.clear();

  YAML::Node root;
  try {
    root = YAML::LoadFile(manifest_path);
  } catch (...) { return false; }

  const std::string dir = parent_dir(manifest_path);
  try {
    if (!parse_manifest(root, dir, out)) return false;
    return detail::finalize_loaded_model(out);
  } catch (const YAML::Exception& e) {
    std::cerr << "Failed to load YAML config from " << manifest_path << ": " << e.what() << '\n';
    return false;
  }
}

}  // namespace origami::nn::twrec
