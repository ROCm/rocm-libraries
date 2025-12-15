// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/log.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace origami {

void logger_t::print() const {
  if (!metrics_ || metrics_->empty()) {
    std::cout << "{}\n";
    return;
  }
  std::cout << "{\n";
  bool first = true;
  for (const auto& [key, val] : *metrics_) {
    if (!first) std::cout << ",\n";
    std::cout << "  \"" << key << "\": " << val;
    first = false;
  }
  std::cout << "\n}\n";
}

void logger_t::export_json(const std::string& filename) const {
  if (!metrics_ || metrics_->empty()) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << filename << " for writing\n";
      return;
    }
    file << "{}\n";
    file.close();
    std::cout << "Analytical metrics exported to JSON: " << filename << "\n";
    return;
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing\n";
    return;
  }

  file << std::fixed << std::setprecision(6);
  file << "{\n";
  bool first = true;
  for (const auto& [key, val] : *metrics_) {
    if (!first) file << ",\n";
    file << "  \"" << key << "\": " << val;
    first = false;
  }
  file << "\n}\n";

  file.close();
  std::cout << "Analytical metrics exported to JSON: " << filename << "\n";
}

}  // namespace origami

