// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Run by the tilewright-env-* ctest entries, which set the environment knobs
// before the process starts (the engine reads them once).

#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#define TILEWRIGHT_TEST_CAPTURE 1
#else
#define TILEWRIGHT_TEST_CAPTURE 0
#endif

namespace tw = tilewright;
using namespace tilewright_test;

namespace {

#if TILEWRIGHT_TEST_CAPTURE
// Redirects stderr (file descriptor 2) into a file until finish().
class StderrCapture {
 public:
  explicit StderrCapture(const std::string& path) : path_(path) {
    std::fflush(stderr);
    saved_         = dup(2);
    FILE* f        = std::fopen(path.c_str(), "w");
    const int f_fd = fileno(f);
    dup2(f_fd, 2);
    std::fclose(f);
  }
  std::string finish() {
    std::fflush(stderr);
    dup2(saved_, 2);
    close(saved_);
    std::ifstream in(path_);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
  }

 private:
  std::string path_;
  int saved_ = -1;
};
#endif

// Mirrors the engine: a knob is on when set to anything but "" or "0".
bool knob_on(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

// Mirrors the engine: a whole non-negative decimal number, else -1.
long long forced_cell(const char* v) {
  char* end           = nullptr;
  const long long val = std::strtoll(v, &end, 10);
  return (end == v || *end != '\0' || val < 0) ? -1 : val;
}

}  // namespace

TEST_CASE("the cache limit starts at TILEWRIGHT_CACHE_MB", "[env]") {
  const char* v       = std::getenv("TILEWRIGHT_CACHE_MB");
  const long long mib = v == nullptr || v[0] == '\0' ? -1 : forced_cell(v);
  CHECK(tw::cache_limit() ==
        (mib >= 0 ? static_cast<std::size_t>(mib) << 20 : std::size_t{1} << 30));
}

TEST_CASE("environment knobs", "[env]") {
#if !TILEWRIGHT_TEST_CAPTURE
  SKIP("stderr capture is not available on this platform");
#else
  const char* force = std::getenv("TILEWRIGHT_FORCE_CELL");
  if (force == nullptr) SKIP("run through the tilewright-env-* ctest entries");
  const long long forced = forced_cell(force);

  TempDir dir("env");
  StderrCapture capture((dir.path() / "stderr.txt").string());
  const ModelSpec spec     = grid_model();
  const tw::ModelPtr model = load_spec(spec);
  const tw::Problem p      = make_problem(4096, 4096, 4096);
  int natural              = -1;
  for (std::size_t c = 0; c < spec.cells.size(); ++c)
    if (spec.cells[c].label == base_label(4096, 4096, 4096, 1)) natural = static_cast<int>(c);
  const int routed             = tw::route(*model, p);
  std::vector<tw::Config> pool = {
      make_config(128, 128, 64), make_config(64, 256, 64), make_config(128, 128, 64)};
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = 10 + i;
  const std::vector<tw::Result> r = tw::rank_configs(*model, p, test_hardware(), pool);
  const std::string text          = capture.finish();

  REQUIRE(tw::cell_label(*model, natural) == base_label(4096, 4096, 4096, 1));
  if (forced >= 0 && forced < static_cast<long long>(spec.cells.size()))
    CHECK(routed == static_cast<int>(forced));
  else
    CHECK(routed == natural);
  REQUIRE(r.size() == 3);
  CHECK(r[0].scored);

  const std::regex diag(
      R"(\[TILEWRIGHT_DIAG\s+\S+\].*?\barch=(\S+)\s+qhash=(\S+)\s+qdim=(\d+)\s+idim=(\d+)\s+xdim=(\d+)\s+n_cells=(\d+)\s+n_splits=(\d+))");
  const std::regex pick(
      R"(\[TILEWRIGHT_PICK\] m=4096 n=4096 k=4096 b=1 tA=T tB=N leaf=(\S+) top1_sig=\(mt_m=(\d+),mt_n=(\d+),mt_k=64,mi_m=16,mi_n=16,mi_k=32,cha=0,chb=0\) top1_score=-?[0-9]+\.[0-9]{6} top1_index=(\d+) n_configs=3)");
  std::smatch m;
  if (knob_on("TILEWRIGHT_DIAG")) {
    REQUIRE(std::regex_search(text, m, diag));
    CHECK(m[1] == "gfxtest");
    CHECK(m[2] == "e7fe4b524851e895");
    CHECK(m[3] == "55");
    CHECK(m[6] == "96");
  } else {
    CHECK_FALSE(std::regex_search(text, m, diag));
  }
  if (knob_on("TILEWRIGHT_PICK_LOG")) {
    REQUIRE(std::regex_search(text, m, pick));
    CHECK(m[1] == tw::cell_label(*model, routed));
    const tw::Config& top = pool[r[0].config_index];
    CHECK(m[2] == std::to_string(top.mt.m));
    CHECK(m[3] == std::to_string(top.mt.n));
    CHECK(m[4] == std::to_string(r[0].config_index));
  } else {
    CHECK(text.find("TILEWRIGHT_PICK") == std::string::npos);
  }
#endif
}

TEST_CASE("TILEWRIGHT_DIAG reports the cells a candidate set serves degraded", "[env][drift]") {
#if !TILEWRIGHT_TEST_CAPTURE
  SKIP("stderr capture is not available on this platform");
#else
  std::vector<tw::Config> pool;
  for (std::size_t mt : {16, 32, 64, 128, 256})
    for (std::size_t nt : {16, 32, 64, 128, 256})
      for (std::size_t kt : {32, 64}) pool.push_back(make_config(mt, nt, kt));
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  // FNV-1a 64 over each signature field as 8 little-endian bytes: the view-key hash.
  std::vector<std::uint64_t> hashes;
  for (const tw::Config& c : pool) {
    std::uint64_t h         = 0xCBF29CE484222325ull;
    const std::int64_t v[8] = {static_cast<std::int64_t>(c.mt.m),
                               static_cast<std::int64_t>(c.mt.n),
                               static_cast<std::int64_t>(c.mt.k),
                               static_cast<std::int64_t>(c.mi.m),
                               static_cast<std::int64_t>(c.mi.n),
                               static_cast<std::int64_t>(c.mi.k),
                               c.cache_hints_a,
                               c.cache_hints_b};
    for (std::int64_t x : v)
      for (int b = 0; b < 8; ++b)
        h = (h ^ static_cast<unsigned char>(static_cast<std::uint64_t>(x) >> (8 * b))) *
            0x100000001B3ull;
    hashes.push_back(h);
  }
  // The gate lists every key but the first five as trained; every third cell lists no key,
  // the others some trained keys and the first (untrained) one, which counts in neither number.
  ModelSpec spec        = as_v3(grid_model());
  spec.v3_serving_flags = 0x80 | 0x400;
  spec.drift_keep_ppm   = 900000;
  spec.trained_keys.assign(hashes.begin() + 5, hashes.end());
  std::sort(spec.trained_keys.begin(), spec.trained_keys.end());
  for (std::size_t c = 0; c < spec.cells.size(); ++c) {
    if (c % 3 == 0) continue;
    std::vector<std::uint64_t> listed = {hashes[0]};
    for (std::size_t i = 5; i < hashes.size(); ++i)
      if ((i + c) % 4 == 0) listed.push_back(hashes[i]);
    std::sort(listed.begin(), listed.end());
    spec.cells[c].view_key_hashes = listed;
  }
  // Without every seventh kernel: the cells that list enough of them are degraded.
  std::vector<tw::Config> lacking;
  std::set<std::uint64_t> held;
  for (std::size_t i = 0; i < pool.size(); ++i)
    if (i % 7 != 3) {
      lacking.push_back(pool[i]);
      lacking.back().index = lacking.size() - 1;
      held.insert(hashes[i]);
    }
  std::size_t with_whitelist = 0, degraded = 0;
  std::uint64_t missing = 0;
  for (const CellSpec& cell : spec.cells) {
    if (cell.view_key_hashes.empty()) continue;
    ++with_whitelist;
    std::uint64_t total = 0, present = 0;
    for (const std::uint64_t h : cell.view_key_hashes) {
      if (!std::binary_search(spec.trained_keys.begin(), spec.trained_keys.end(), h)) continue;
      ++total;
      present += held.count(h);
    }
    if (total > 0 && present * 1000000 < std::uint64_t{spec.drift_keep_ppm} * total) {
      ++degraded;
      missing += total - present;
    }
  }
  REQUIRE(degraded > 0);
  REQUIRE(degraded < with_whitelist);

  TempDir dir("drift-diag");
  StderrCapture capture((dir.path() / "stderr.txt").string());
  const tw::ModelPtr model = load_spec(spec);
  const tw::CandidateSet whole(model, pool), served(model, lacking);
  const std::string text = capture.finish();

  CHECK(whole.degraded_cells().empty());
  CHECK(served.degraded_cells().size() == degraded);
  const std::regex line(
      R"(\[TILEWRIGHT_DIAG DRIFT\] cells=(\d+)/(\d+) keep_ppm=(\d+) missing=(\d+)\n)");
  const auto lines =
      std::distance(std::sregex_iterator(text.begin(), text.end(), line), std::sregex_iterator());
  if (knob_on("TILEWRIGHT_DIAG")) {
    // One line, for the candidate set with degraded cells only.
    REQUIRE(lines == 1);
    CHECK(text.find("TILEWRIGHT_DIAG DRIFT") == text.rfind("TILEWRIGHT_DIAG DRIFT"));
    std::smatch m;
    REQUIRE(std::regex_search(text, m, line));
    CHECK(m[1] == std::to_string(degraded));
    CHECK(m[2] == std::to_string(with_whitelist));
    CHECK(m[3] == "900000");
    CHECK(m[4] == std::to_string(missing));
  } else {
    CHECK(text.find("TILEWRIGHT_DIAG DRIFT") == std::string::npos);
  }
#endif
}
