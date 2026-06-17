/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include "common.hpp"
#include "origami/logger.hpp"

static std::string unique_temp_path(const std::string& prefix, const std::string& ext) {
  static std::atomic<int> counter{0};
  auto dir  = std::filesystem::temp_directory_path();
#ifdef _WIN32
  auto pid = _getpid();
#else
  auto pid = getpid();
#endif
  auto name = prefix + "_" + std::to_string(pid) + "_" + std::to_string(counter++) + ext;
  return (dir / name).string();
}

static std::string read_file(const std::string& path) {
  std::ifstream ifs(path);
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

static size_t count_occurrences(const std::string& text, const std::string& sub) {
  size_t count = 0;
  size_t pos   = 0;
  while ((pos = text.find(sub, pos)) != std::string::npos) {
    ++count;
    pos += sub.size();
  }
  return count;
}

static void log_debug(const std::string& msg) {
  origami::Logger::instance().log(origami::LogLevel::DEBUG, msg, "test", 0);
}

static void disable_logger() {
  portable_unsetenv("ORIGAMI_LOG_FILE");
  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::Logger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();
}

// ---------------------------------------------------------------------------
// Text mode tests
// ---------------------------------------------------------------------------

TEST_CASE("Logger: text log writes debug messages when enabled", "[logger]") {
  const std::string log_path = unique_temp_path("origami_log", ".log");
  std::remove(log_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", log_path.c_str(), 1);
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::Logger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();

  REQUIRE(origami::Logger::instance().format() == origami::LogFormat::TEXT);

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      std::remove(log_path.c_str());
      origami::Logger::instance().update_from_env();

      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 1024);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      origami::compute_total_latency(problem, hardware, config, hardware.N_CU);
      origami::Logger::instance().flush();

      std::string contents = read_file(log_path);
      REQUIRE_FALSE(contents.empty());
      REQUIRE(contents.find("[DEBUG]") != std::string::npos);
      REQUIRE(contents.find("Origami Debug Info") != std::string::npos);
      REQUIRE(contents.find("total_latency") != std::string::npos);
    }
  }

  std::remove(log_path.c_str());
  disable_logger();
}

TEST_CASE("Logger: text log is not written when debug is disabled", "[logger]") {
  const std::string log_path = unique_temp_path("origami_log_disabled", ".log");
  std::remove(log_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", log_path.c_str(), 1);
  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::Logger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();

  auto hardware = make_hardware(942);
  auto problem  = make_problem(2048, 2048, 1024);
  auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

  origami::compute_total_latency(problem, hardware, config, hardware.N_CU);
  origami::Logger::instance().flush();

  std::string contents = read_file(log_path);
  REQUIRE(contents.find("[DEBUG]") == std::string::npos);

  std::remove(log_path.c_str());
  disable_logger();
}

// ---------------------------------------------------------------------------
// CSV mode tests
// ---------------------------------------------------------------------------

TEST_CASE("Logger: CSV mode parses key-value pairs", "[logger][csv]") {
  const std::string csv_path = unique_temp_path("origami_csv_parse", ".csv");
  std::remove(csv_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", csv_path.c_str(), 1);
  origami::Logger::instance().update_from_env();

  REQUIRE(origami::Logger::instance().is_enabled());
  REQUIRE(origami::Logger::instance().format() == origami::LogFormat::CSV);

  log_debug("======== Origami Debug Info ========");
  log_debug("Alpha: 42");
  log_debug("Beta: 3.14");
  log_debug("Gamma: hello");
  log_debug("=================================");

  origami::Logger::instance().flush();

  std::string contents = read_file(csv_path);
  REQUIRE_FALSE(contents.empty());

  REQUIRE(contents.find("Alpha") != std::string::npos);
  REQUIRE(contents.find("Beta") != std::string::npos);
  REQUIRE(contents.find("Gamma") != std::string::npos);

  REQUIRE(contents.find("42") != std::string::npos);
  REQUIRE(contents.find("3.14") != std::string::npos);
  REQUIRE(contents.find("hello") != std::string::npos);

  REQUIRE(count_occurrences(contents, "\n") == 2);

  std::remove(csv_path.c_str());
  disable_logger();
}

TEST_CASE("Logger: CSV mode ignores messages outside a row", "[logger][csv]") {
  const std::string csv_path = unique_temp_path("origami_csv_outside", ".csv");
  std::remove(csv_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", csv_path.c_str(), 1);
  origami::Logger::instance().update_from_env();
  REQUIRE(origami::Logger::instance().is_enabled());

  log_debug("Stray: should_not_appear");
  log_debug("Hand-optimized kernel gfx950_BF16_TN, efficiency: 0.95");

  log_debug("======== Origami Debug Info ========");
  log_debug("ValidKey: 99");
  log_debug("=================================");

  origami::Logger::instance().flush();

  std::string contents = read_file(csv_path);
  REQUIRE_FALSE(contents.empty());

  REQUIRE(contents.find("should_not_appear") == std::string::npos);
  REQUIRE(contents.find("ValidKey") != std::string::npos);
  REQUIRE(contents.find("99") != std::string::npos);

  std::remove(csv_path.c_str());
  disable_logger();
}

TEST_CASE("Logger: CSV output from GEMM evaluation contains expected columns", "[logger][csv]") {
  const std::string csv_path = unique_temp_path("origami_csv_gemm", ".csv");
  std::remove(csv_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", csv_path.c_str(), 1);
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::Logger::instance().update_from_env();
  origami::runtime_options::get().update_from_env();

  REQUIRE(origami::Logger::instance().is_enabled());

  auto hardware = make_hardware(942);
  auto problem  = make_problem(4096, 4096, 2048);
  auto config   = make_config(256, 256, 64, 16, 16, 16, false, 1);

  origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

  auto problem2 = make_problem(256, 256, 16384);
  origami::compute_total_latency(problem2, hardware, config, hardware.N_CU);

  origami::Logger::instance().flush();

  std::string contents = read_file(csv_path);
  REQUIRE_FALSE(contents.empty());
  REQUIRE(contents.find("total_latency") != std::string::npos);
  REQUIRE(contents.find("L_mem") != std::string::npos);
  REQUIRE(contents.find("L_compute") != std::string::npos);
  REQUIRE(contents.find("L_parallel_reduce") != std::string::npos);
  REQUIRE(count_occurrences(contents, "\n") >= 3);

  std::remove(csv_path.c_str());
  disable_logger();
}

TEST_CASE("Logger: CSV escape handles special characters", "[logger][csv]") {
  const std::string csv_path = unique_temp_path("origami_csv_escape", ".csv");
  std::remove(csv_path.c_str());

  portable_setenv("ORIGAMI_LOG_FILE", csv_path.c_str(), 1);
  origami::Logger::instance().update_from_env();

  log_debug("======== Origami Debug Info ========");
  log_debug("CommaVal: 1,2,3");
  log_debug("QuoteVal: he said \"hi\"");
  log_debug("PlainVal: 42");
  log_debug("=================================");

  origami::Logger::instance().flush();

  std::string contents = read_file(csv_path);
  REQUIRE_FALSE(contents.empty());

  REQUIRE(contents.find("\"1,2,3\"") != std::string::npos);
  REQUIRE(contents.find("\"he said \"\"hi\"\"\"") != std::string::npos);
  REQUIRE(contents.find("42") != std::string::npos);

  std::remove(csv_path.c_str());
  disable_logger();
}

// ---------------------------------------------------------------------------
// Latency consistency tests
// ---------------------------------------------------------------------------

TEST_CASE("Logger: debug logging produces consistent latency values", "[logger]") {
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::runtime_options::get().update_from_env();

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 1024);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      double latency1 = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);
      double latency2 = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

      REQUIRE(latency1 == latency2);
      REQUIRE(latency1 > 0.0);
    }
  }

  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::runtime_options::get().update_from_env();
}

TEST_CASE("Logger: latency matches between debug enabled and disabled", "[logger]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 2048);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
      origami::runtime_options::get().update_from_env();
      double latency_no_debug = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

      portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
      origami::runtime_options::get().update_from_env();
      double latency_with_debug = origami::compute_total_latency(problem, hardware, config, hardware.N_CU);

      REQUIRE(latency_no_debug == latency_with_debug);
    }
  }

  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::runtime_options::get().update_from_env();
}

TEST_CASE("Logger: ANALYTICAL_GEMM_DEBUG env var controls debug_enabled flag", "[logger]") {
  portable_setenv("ANALYTICAL_GEMM_DEBUG", "1", 1);
  origami::runtime_options::get().update_from_env();
  REQUIRE(origami::runtime_options::get().debug_enabled == true);

  portable_setenv("ANALYTICAL_GEMM_DEBUG", "0", 1);
  origami::runtime_options::get().update_from_env();
  REQUIRE(origami::runtime_options::get().debug_enabled == false);

  portable_unsetenv("ANALYTICAL_GEMM_DEBUG");
  origami::runtime_options::get().update_from_env();
  REQUIRE(origami::runtime_options::get().debug_enabled == false);
}
