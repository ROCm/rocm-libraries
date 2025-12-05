/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace origami {

/**
 * @brief Logger for collecting and exporting analytical metrics in JSON format.
 *
 * Provides a single templated logging function that stores key-value pairs
 * and can export them as JSON. The caller is responsible for checking if
 * debug is enabled before calling log().
 */
class logger_t {
 public:
  /**
   * @brief Default constructor (constexpr-compatible).
   * The map is lazily allocated on first use.
   */
  constexpr logger_t() : metrics_(nullptr) {}

  /**
   * @brief Destructor
   */
  ~logger_t() = default;

  /**
   * @brief Copy constructor.
   */
  logger_t(const logger_t& other) {
    if (other.metrics_) {
      metrics_ = std::make_unique<std::unordered_map<std::string, std::string>>(*other.metrics_);
    } else {
      metrics_ = nullptr;
    }
  }

  /**
   * @brief Move constructor.
   */
  logger_t(logger_t&& other) noexcept = default;

  /**
   * @brief Copy assignment operator.
   */
  logger_t& operator=(const logger_t& other) {
    if (this != &other) {
      if (other.metrics_) {
        metrics_ = std::make_unique<std::unordered_map<std::string, std::string>>(*other.metrics_);
      } else {
        metrics_ = nullptr;
      }
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   */
  logger_t& operator=(logger_t&& other) noexcept = default;

  /**
   * @brief Ensure metrics map is allocated.
   */
  void ensure_metrics() const {
    if (!metrics_) { metrics_ = std::make_unique<std::unordered_map<std::string, std::string>>(); }
  }

  /**
   * @brief Log a key-value pair.
   *
   * @tparam T Type of the value (must be convertible to JSON-compatible string)
   * @param key The metric key
   * @param value The metric value
   */
  template <typename T>
  void log(const std::string& key, const T& value) {
    ensure_metrics();
    (*metrics_)[key] = to_json_string(value);
  }

  /**
   * @brief Clear all logged metrics.
   */
  void clear() {
    if (metrics_) { metrics_->clear(); }
  }

  /**
   * @brief Print all metrics as JSON to stdout.
   */
  void print() const;

  /**
   * @brief Export metrics to a JSON file.
   *
   * @param filename Output filename
   */
  void export_json(const std::string& filename) const;

  /**
   * @brief Get all metrics as a map.
   *
   * @return Map of metric key-value pairs
   */
  std::unordered_map<std::string, std::string> get_metrics() const {
    if (!metrics_) { return std::unordered_map<std::string, std::string>(); }
    return *metrics_;
  }

  /**
   * @brief Check if logger has any metrics.
   *
   * @return true if metrics map is not empty
   */
  bool empty() const { return !metrics_ || metrics_->empty(); }

 private:
  mutable std::unique_ptr<std::unordered_map<std::string, std::string>> metrics_;

  // Convert value to JSON-compatible string
  template <typename T>
  std::string to_json_string(const T& value) {
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::string>) {
      return "\"" + value + "\"";
    } else if constexpr (std::is_convertible_v<T, const char*> ||
                         std::is_array_v<std::remove_reference_t<T>>) {
      return "\"" + std::string(value) + "\"";
    } else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, bool>) {
      return value ? "true" : "false";
    } else if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      return std::to_string(value);
    } else {
      // For other types, try to_string (this may fail for some types)
      static_assert(std::is_arithmetic_v<T>, "Type must be convertible to string or arithmetic");
      return std::to_string(value);
    }
  }
};

}  // namespace origami
