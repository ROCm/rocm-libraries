// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/esrec/esrec_loader.hpp"

#include "origami/nn/esrec/weights_hash.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <unordered_map>
#include <vector>

namespace origami::nn::esrec {
namespace {

constexpr std::uint32_t kSchemaVersion = 1;

std::string parent_dir(const std::string& path) {
  const std::filesystem::path p(path);
  const auto parent = p.parent_path();
  if (parent.empty()) return ".";
  return parent.string();
}

std::string join_path(const std::string& dir, const std::string& file) {
  return (std::filesystem::path(dir) / file).string();
}

bool read_float_vector(const YAML::Node& node, std::vector<float>* out) {
  if (!node || !node.IsSequence()) return false;
  out->resize(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    (*out)[i] = node[i].as<float>();
  }
  return true;
}

bool read_matrix(const YAML::Node& node, std::vector<std::vector<float>>* out) {
  if (!node || !node.IsSequence()) return false;
  out->resize(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    if (!read_float_vector(node[i], &(*out)[i])) return false;
  }
  return true;
}

bool read_int_matrix(const YAML::Node& node, std::vector<std::vector<int>>* out) {
  if (!node || !node.IsSequence()) return false;
  out->resize(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    const YAML::Node row = node[i];
    if (!row || !row.IsSequence()) return false;
    (*out)[i].resize(row.size());
    for (std::size_t j = 0; j < row.size(); ++j) {
      (*out)[i][j] = row[j].as<int>();
    }
  }
  return true;
}

bool read_embedding_tensor(const YAML::Node& node,
                           std::vector<std::vector<std::vector<float>>>* out) {
  if (!node || !node.IsSequence()) return false;
  out->resize(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    if (!read_matrix(node[i], &(*out)[i])) return false;
  }
  return true;
}

bool read_hardware_constants(const YAML::Node& node, detail::HardwareConstants* hw) {
  if (!node || !node.IsMap()) return false;
  hw->n_cu       = node["n_cu"].as<int>();
  hw->peak_flops = node["peak_flops"].as<float>();
  hw->mem_bw     = node["mem_bw"].as<float>();
  hw->l1_size    = node["l1_size"].as<float>();
  hw->l2_size    = node["l2_size"].as<float>();
  hw->l3_size    = node["l3_size"].as<float>();
  hw->wave_size  = node["wave_size"].as<float>();
  hw->dtype_size = node["dtype_size"].as<float>();
  hw->acc_size   = node["acc_size"].as<float>();
  return hw->n_cu > 0 && hw->peak_flops > 0.0f && hw->mem_bw > 0.0f;
}

bool read_encoder(const YAML::Node& node, detail::EncoderModel* encoder) {
  if (!node || !node.IsMap()) return false;
  const YAML::Node scaler = node["scaler"];
  if (!read_float_vector(scaler["mean"], &encoder->scaler_mean)) return false;
  if (!read_float_vector(scaler["scale"], &encoder->scaler_scale)) return false;

  const YAML::Node state = node["state_dict"];
  if (!read_matrix(state["weights"], &encoder->weights)) return false;
  if (!read_matrix(state["bias"], &encoder->bias)) return false;
  if (!read_float_vector(state["proj_weights"], &encoder->proj_weights)) return false;
  if (!read_float_vector(state["proj_bias"], &encoder->proj_bias)) return false;
  return !encoder->proj_bias.empty();
}

bool read_hidden_layers(const YAML::Node& model_node, std::vector<std::uint32_t>* hidden_layers) {
  if (!model_node || !model_node.IsMap()) return false;
  const YAML::Node layers = model_node["hidden_layers"];
  if (!layers || !layers.IsSequence() || layers.size() == 0) return false;
  hidden_layers->clear();
  hidden_layers->reserve(layers.size());
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const int dim = layers[i].as<int>();
    if (dim <= 0) return false;
    hidden_layers->push_back(static_cast<std::uint32_t>(dim));
  }
  return true;
}

bool validate_encoder(const detail::EncoderModel& encoder,
                      std::uint32_t input_dim,
                      std::uint32_t embed_dim,
                      const std::vector<std::uint32_t>& hidden_layers) {
  if (encoder.scaler_mean.size() != input_dim) return false;
  if (encoder.scaler_scale.size() != input_dim) return false;
  if (encoder.weights.size() != hidden_layers.size()) return false;
  if (encoder.bias.size() != hidden_layers.size()) return false;

  std::size_t prev_dim = input_dim;
  for (std::size_t layer = 0; layer < hidden_layers.size(); ++layer) {
    const std::size_t out_dim = hidden_layers[layer];
    if (encoder.bias[layer].size() != out_dim) return false;
    if (encoder.weights[layer].size() != out_dim * prev_dim) return false;
    prev_dim = out_dim;
  }

  if (encoder.proj_weights.size() != static_cast<std::size_t>(embed_dim) * prev_dim) return false;
  if (encoder.proj_bias.size() != embed_dim) return false;
  return true;
}

bool finalize_solution_map(detail::LoadedModel* model) {
  model->solution_by_index.clear();
  for (std::size_t c = 0; c < model->cluster_indices.size(); ++c) {
    if (c >= model->embeddings.size()) return false;
    const auto& indices = model->cluster_indices[c];
    const auto& vectors = model->embeddings[c];
    if (indices.size() != vectors.size()) return false;
    for (std::size_t i = 0; i < indices.size(); ++i) {
      model->solution_by_index.emplace(indices[i], vectors[i]);
    }
  }
  return !model->solution_by_index.empty();
}

}  // namespace

bool load_esrec_yaml(const std::string& manifest_path, detail::LoadedModel* out) {
  if (out == nullptr) return false;

  YAML::Node root;
  try {
    root = YAML::LoadFile(manifest_path);
  } catch (...) {
    return false;
  }

  if (!root.IsMap()) return false;
  if (root["schema_version"].as<std::uint32_t>() != kSchemaVersion) return false;
  if (root["format"].as<std::string>() != "ESREC_v1") return false;

  const YAML::Node meta = root["metadata"];
  if (!meta) return false;
  out->arch          = meta["arch"].as<std::string>();
  out->problem_stem  = meta["problem_stem"].as<std::string>();
  out->weights_hash  = meta["weights_hash"].as<std::string>();

  const YAML::Node feat = root["features"];
  if (!feat) return false;
  out->input_dim = feat["input_dim"].as<std::uint32_t>();
  out->embed_dim = feat["embed_dim"].as<std::uint32_t>();
  const std::string layout = feat["layout"].as<std::string>();
  out->is_nt               = (layout == "NT");

  if (!read_hardware_constants(root["hardware_constants"], &out->hw)) return false;
  if (!out->fallback.load_from_yaml(root["fallback"])) return false;

  const std::string sidecar_name = root["weights_sidecar"].as<std::string>();
  const std::string sidecar_path = join_path(parent_dir(manifest_path), sidecar_name);

  YAML::Node wts;
  try {
    wts = YAML::LoadFile(sidecar_path);
  } catch (...) {
    return false;
  }

  if (!wts.IsMap()) return false;
  if (wts["schema_version"].as<std::uint32_t>() != kSchemaVersion) return false;
  if (wts["format"].as<std::string>() != "ESREC_WTS_v1") return false;

  const std::string sidecar_hash = detail::compute_weights_hash(wts);
  if (sidecar_hash.empty() || sidecar_hash != out->weights_hash) return false;

  std::vector<std::uint32_t> hidden_layers;
  if (!read_hidden_layers(root["model"], &hidden_layers)) return false;

  if (!read_encoder(wts["encoder"], &out->encoder)) return false;

  const YAML::Node solution_embeddings = wts["solution_embeddings"];
  if (!read_matrix(solution_embeddings["centroids"], &out->centroids)) return false;
  if (!read_embedding_tensor(solution_embeddings["embeddings"], &out->embeddings)) return false;
  if (!read_int_matrix(solution_embeddings["cluster_indices"], &out->cluster_indices)) return false;

  if (!validate_encoder(out->encoder, out->input_dim, out->embed_dim, hidden_layers)) return false;

  return finalize_solution_map(out);
}

}  // namespace origami::nn::esrec
