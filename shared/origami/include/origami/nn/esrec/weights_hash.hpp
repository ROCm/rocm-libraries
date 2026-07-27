// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace YAML {
class Node;
}

namespace origami::nn::esrec::detail {

/// Canonical JSON + SHA256 (first 16 hex chars), matching split_embedding_yaml.py.
std::string compute_weights_hash(const YAML::Node& sidecar_root);

}  // namespace origami::nn::esrec::detail
