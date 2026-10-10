// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// errmap: an exact correctness check of hipBLASLt GEMM solutions picked by
// solution index (lib/correctness_gate.py runs it).
//
// A and B hold integers in [-2, 2] and C integers in [-2, 2], so every product
// and every partial sum of the f32 accumulation is exact and a correct
// solution writes the bits of the CPU reference into every element of D; any
// other element is wrong. Elements a solution must write but does not keep a
// NaN pattern (beta 0 and C == D, or a separate D).
//
// usage: errmap M N K BATCH --ids I1,I2,... [--reps R] [--alpha A] [--beta B]
//          [--trans-a T|N] [--trans-b T|N] [--a-type bf16|f32] [--b-type bf16|f32]
//          [--c-type bf16|f32] [--d-type bf16|f32] [--compute f32|xf32]
//          [--lda X] [--ldb X] [--ldc X] [--ldd X] [--sep 0|1] [--ws BYTES]
//          [--seed S] [--maxlist N] [--mt MTMxMTN] [--du DEPTHU]
//
// On stdout, {"ready": true, "algos": N, "sids": [...]} once the device holds
// the operands: the N solutions of --ids that were found, in the order of
// --ids, which they then run in. Then one JSON line per (solution,
// repetition):
//   {"sid", "rep", "supported", "ws_need", "nbad", "nnan", "maxabs",
//    "bad": [[i, j, batch, got, expected, diff, "K-chunk explanation"], ...],
//    "per_tile": {...}, "name"}
// and {"sid", "supported": false, "ws_need", "name"} once for a solution that
// does not support the problem. With --mt the wrong elements are counted per
// output tile; with --du the listed ones name the contiguous range of DepthU
// chunks of K whose partial sum the difference equals, when one does.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define HIP_CHECK(x)                                                                      \
  do {                                                                                    \
    hipError_t e_ = (x);                                                                  \
    if (e_ != hipSuccess) {                                                               \
      std::fprintf(stderr, "HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__); \
      std::exit(2);                                                                       \
    }                                                                                     \
  } while (0)
#define LT_CHECK(x)                                                                \
  do {                                                                             \
    hipblasStatus_t s_ = (x);                                                      \
    if (s_ != HIPBLAS_STATUS_SUCCESS) {                                            \
      std::fprintf(stderr, "hipBLASLt status %d at line %d\n", int(s_), __LINE__); \
      std::exit(3);                                                                \
    }                                                                              \
  } while (0)

namespace {

enum class Type { BF16, F32 };

struct Options {
  int64_t m = 0, n = 0, k = 0, batch = 1;
  std::vector<int> ids;
  int reps         = 1;
  float alpha      = 1.f;
  float beta       = 0.f;
  bool trans_a     = true;
  bool trans_b     = false;
  Type a_type      = Type::BF16;
  Type b_type      = Type::BF16;
  Type c_type      = Type::BF16;
  Type d_type      = Type::BF16;
  bool xf32        = false;
  int64_t lda      = 0;
  int64_t ldb      = 0;
  int64_t ldc      = 0;
  int64_t ldd      = 0;
  bool separate_d  = false;
  size_t workspace = 128ull << 20;
  uint64_t seed    = 12345;
  int max_list     = 40;
  int tile_m       = 0;
  int tile_n       = 0;
  int depth_u      = 0;
};

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

float from_bf16(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A deterministic value in [-2, 2] for element `x` of a matrix.
int value_of(uint64_t x, uint64_t seed) {
  uint64_t z = x * 0x9E3779B97F4A7C15ull + seed;
  z ^= z >> 31;
  z *= 0xBF58476D1CE4E5B9ull;
  z ^= z >> 29;
  return static_cast<int>(z % 5) - 2;
}

size_t type_size(Type t) { return t == Type::BF16 ? 2 : 4; }

hipDataType hip_type(Type t) { return t == Type::BF16 ? HIP_R_16BF : HIP_R_32F; }

Type parse_type(const std::string& s) {
  if (s == "bf16") return Type::BF16;
  if (s == "f32") return Type::F32;
  std::fprintf(stderr, "unknown type %s (bf16 or f32)\n", s.c_str());
  std::exit(1);
}

void put(std::vector<uint8_t>& buf, Type t, int64_t at, float v) {
  if (t == Type::BF16) {
    uint16_t b = to_bf16(v);
    std::memcpy(&buf[at * 2], &b, 2);
  } else {
    std::memcpy(&buf[at * 4], &v, 4);
  }
}

float get(const std::vector<uint8_t>& buf, Type t, int64_t at) {
  if (t == Type::BF16) {
    uint16_t b;
    std::memcpy(&b, &buf[at * 2], 2);
    return from_bf16(b);
  }
  float f;
  std::memcpy(&f, &buf[at * 4], 4);
  return f;
}

bool same_bits(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, Type t, int64_t at) {
  size_t s = type_size(t);
  return std::memcmp(&a[at * s], &b[at * s], s) == 0;
}

Options parse(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: errmap M N K BATCH --ids I1,I2,... [options]\n");
    std::exit(1);
  }
  Options o;
  o.m     = std::atoll(argv[1]);
  o.n     = std::atoll(argv[2]);
  o.k     = std::atoll(argv[3]);
  o.batch = std::atoll(argv[4]);
  for (int i = 5; i < argc; i++) {
    std::string a = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s needs a value\n", a.c_str());
      std::exit(1);
    }
    std::string v = argv[++i];
    if (a == "--ids") {
      std::stringstream ss(v);
      std::string t;
      while (std::getline(ss, t, ',')) o.ids.push_back(std::atoi(t.c_str()));
    } else if (a == "--reps") {
      o.reps = std::atoi(v.c_str());
    } else if (a == "--alpha") {
      o.alpha = std::strtof(v.c_str(), nullptr);
    } else if (a == "--beta") {
      o.beta = std::strtof(v.c_str(), nullptr);
    } else if (a == "--trans-a") {
      o.trans_a = v == "T";
    } else if (a == "--trans-b") {
      o.trans_b = v == "T";
    } else if (a == "--a-type") {
      o.a_type = parse_type(v);
    } else if (a == "--b-type") {
      o.b_type = parse_type(v);
    } else if (a == "--c-type") {
      o.c_type = parse_type(v);
    } else if (a == "--d-type") {
      o.d_type = parse_type(v);
    } else if (a == "--compute") {
      o.xf32 = v == "xf32";
    } else if (a == "--lda") {
      o.lda = std::atoll(v.c_str());
    } else if (a == "--ldb") {
      o.ldb = std::atoll(v.c_str());
    } else if (a == "--ldc") {
      o.ldc = std::atoll(v.c_str());
    } else if (a == "--ldd") {
      o.ldd = std::atoll(v.c_str());
    } else if (a == "--sep") {
      o.separate_d = std::atoi(v.c_str()) != 0;
    } else if (a == "--ws") {
      o.workspace = std::strtoull(v.c_str(), nullptr, 10);
    } else if (a == "--seed") {
      o.seed = std::strtoull(v.c_str(), nullptr, 10);
    } else if (a == "--maxlist") {
      o.max_list = std::atoi(v.c_str());
    } else if (a == "--mt") {
      std::sscanf(v.c_str(), "%dx%d", &o.tile_m, &o.tile_n);
    } else if (a == "--du") {
      o.depth_u = std::atoi(v.c_str());
    } else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      std::exit(1);
    }
  }
  if (o.m < 1 || o.n < 1 || o.k < 1 || o.batch < 1 || o.ids.empty() || o.reps < 1) {
    std::fprintf(stderr, "M, N, K, BATCH, --reps must be positive and --ids non-empty\n");
    std::exit(1);
  }
  // The stored A is K x M when transposed, else M x K; B is N x K when
  // transposed, else K x N (column major).
  int64_t rows_a = o.trans_a ? o.k : o.m;
  int64_t rows_b = o.trans_b ? o.n : o.k;
  o.lda          = o.lda > 0 ? o.lda : rows_a;
  o.ldb          = o.ldb > 0 ? o.ldb : rows_b;
  o.ldc          = o.ldc > 0 ? o.ldc : o.m;
  o.ldd          = o.ldd > 0 ? o.ldd : (o.separate_d ? o.m : o.ldc);
  if (o.lda < rows_a || o.ldb < rows_b || o.ldc < o.m || o.ldd < o.m) {
    std::fprintf(stderr, "a leading dimension is below its matrix's rows\n");
    std::exit(1);
  }
  if (!o.separate_d && (o.ldd != o.ldc || o.d_type != o.c_type)) {
    std::fprintf(stderr, "C == D needs equal leading dimensions and types (or --sep 1)\n");
    std::exit(1);
  }
  return o;
}

struct Problem {
  Options o;
  int64_t stride_a = 0, stride_b = 0, stride_c = 0, stride_d = 0;
  std::vector<uint8_t> a, b, c, d, ref;
  // op(A) as [batch][m][k], op(B) as [batch][n][k], C as [batch][n][m].
  std::vector<int8_t> int_a, int_b, int_c;
  std::vector<int32_t> int_ref;

  explicit Problem(const Options& opt) : o(opt) {
    stride_a = o.lda * (o.trans_a ? o.m : o.k);
    stride_b = o.ldb * (o.trans_b ? o.k : o.n);
    stride_c = o.ldc * o.n;
    stride_d = o.ldd * o.n;
    a.assign(stride_a * o.batch * type_size(o.a_type), 0);
    b.assign(stride_b * o.batch * type_size(o.b_type), 0);
    c.assign(stride_c * o.batch * type_size(o.c_type), 0);
    d.assign(stride_d * o.batch * type_size(o.d_type), 0);
    ref.assign(d.size(), 0xff);
    int_a.resize(o.batch * o.m * o.k);
    int_b.resize(o.batch * o.n * o.k);
    int_c.resize(o.batch * o.n * o.m);
    int_ref.resize(o.batch * o.n * o.m);
    fill();
    reference();
  }

  int64_t a_at(int64_t bt, int64_t i, int64_t kk) const {
    return bt * stride_a + (o.trans_a ? i * o.lda + kk : kk * o.lda + i);
  }
  int64_t b_at(int64_t bt, int64_t kk, int64_t j) const {
    return bt * stride_b + (o.trans_b ? kk * o.ldb + j : j * o.ldb + kk);
  }

  void fill() {
#pragma omp parallel for
    for (int64_t bt = 0; bt < o.batch; bt++) {
      for (int64_t i = 0; i < o.m; i++)
        for (int64_t kk = 0; kk < o.k; kk++) {
          int v = value_of((uint64_t(bt) * o.m + i) * o.k + kk, o.seed);
          put(a, o.a_type, a_at(bt, i, kk), float(v));
          int_a[(bt * o.m + i) * o.k + kk] = int8_t(v);
        }
      for (int64_t j = 0; j < o.n; j++)
        for (int64_t kk = 0; kk < o.k; kk++) {
          int v = value_of((uint64_t(bt) * o.n + j) * o.k + kk, o.seed ^ 0x5555);
          put(b, o.b_type, b_at(bt, kk, j), float(v));
          int_b[(bt * o.n + j) * o.k + kk] = int8_t(v);
        }
      for (int64_t j = 0; j < o.n; j++)
        for (int64_t i = 0; i < o.m; i++) {
          int v = value_of((uint64_t(bt) * o.n + j) * o.m + i, o.seed ^ 0xAAAA);
          put(c, o.c_type, bt * stride_c + j * o.ldc + i, float(v));
          int_c[(bt * o.n + j) * o.m + i] = int8_t(v);
        }
    }
  }

  void reference() {
#pragma omp parallel for collapse(2)
    for (int64_t bt = 0; bt < o.batch; bt++)
      for (int64_t j = 0; j < o.n; j++) {
        const int8_t* bb = &int_b[(bt * o.n + j) * o.k];
        for (int64_t i = 0; i < o.m; i++) {
          const int8_t* aa = &int_a[(bt * o.m + i) * o.k];
          int32_t s        = 0;
#pragma omp simd reduction(+ : s)
          for (int64_t kk = 0; kk < o.k; kk++) s += int32_t(aa[kk]) * int32_t(bb[kk]);
          float total = o.alpha * float(s) +
                        (o.beta != 0.f ? o.beta * float(int_c[(bt * o.n + j) * o.m + i]) : 0.f);
          int_ref[(bt * o.n + j) * o.m + i] = int32_t(std::lrint(total));
          put(ref, o.d_type, bt * stride_d + j * o.ldd + i, total);
        }
      }
  }

  // The contiguous range of DepthU chunks of K whose partial sum equals the
  // difference of element (i, j) of batch `bt`, or "".
  std::string explain(int64_t bt, int64_t i, int64_t j, int32_t diff) const {
    if (o.depth_u <= 0 || diff == 0) return "";
    int64_t chunks = (o.k + o.depth_u - 1) / o.depth_u;
    std::vector<int32_t> part(chunks, 0);
    const int8_t* aa = &int_a[(bt * o.m + i) * o.k];
    const int8_t* bb = &int_b[(bt * o.n + j) * o.k];
    for (int64_t kk = 0; kk < o.k; kk++) part[kk / o.depth_u] += aa[kk] * bb[kk];
    for (int64_t c0 = 0; c0 < chunks; c0++) {
      int32_t acc = 0;
      for (int64_t c1 = c0; c1 < chunks; c1++) {
        acc += part[c1];
        if (acc != 0 && (acc == diff || acc == -diff)) {
          char text[96];
          std::snprintf(text,
                        sizeof text,
                        "%s chunks [%lld,%lld]",
                        acc == diff ? "extra" : "missing",
                        static_cast<long long>(c0),
                        static_cast<long long>(c1));
          return text;
        }
      }
    }
    return "";
  }
};

void dump_map(const char* name, const std::map<int64_t, int64_t>& counts) {
  std::printf(", \"%s\": {", name);
  int shown = 0;
  for (const auto& kv : counts) {
    if (shown == 64) break;
    std::printf("%s\"%lld\": %lld",
                shown ? ", " : "",
                static_cast<long long>(kv.first),
                static_cast<long long>(kv.second));
    shown++;
  }
  std::printf("}");
}

// Compares D with the reference and prints the repetition's line.
void report(const Problem& p, int sid, int rep, size_t need, const std::string& name) {
  const Options& o = p.o;
  int64_t bad = 0, nan = 0;
  double max_abs  = 0;
  int64_t tiles_m = o.tile_m > 0 ? (o.m + o.tile_m - 1) / o.tile_m : 0;
  int64_t tiles_n = o.tile_n > 0 ? (o.n + o.tile_n - 1) / o.tile_n : 0;
  std::map<int64_t, int64_t> per_tile;
  std::vector<std::string> listed;
  for (int64_t bt = 0; bt < o.batch; bt++)
    for (int64_t j = 0; j < o.n; j++)
      for (int64_t i = 0; i < o.m; i++) {
        int64_t at = bt * p.stride_d + j * o.ldd + i;
        if (same_bits(p.d, p.ref, o.d_type, at)) continue;
        bad++;
        float got      = get(p.d, o.d_type, at);
        float expected = get(p.ref, o.d_type, at);
        if (std::isnan(got)) {
          nan++;
        } else {
          max_abs = std::max(max_abs, std::fabs(double(got) - double(expected)));
        }
        if (tiles_m > 0 && tiles_n > 0)
          per_tile[(bt * tiles_n + j / o.tile_n) * tiles_m + i / o.tile_m]++;
        if (int(listed.size()) < o.max_list) {
          int32_t diff = 0;
          if (!std::isnan(got) && std::fabs(got) < 16777216.f)
            diff = int32_t(std::lrint(got)) - p.int_ref[(bt * o.n + j) * o.m + i];
          char text[256];
          std::snprintf(text,
                        sizeof text,
                        "[%lld,%lld,%lld,%g,%g,%d,\"%s\"]",
                        static_cast<long long>(i),
                        static_cast<long long>(j),
                        static_cast<long long>(bt),
                        double(got),
                        double(expected),
                        diff,
                        p.explain(bt, i, j, diff).c_str());
          listed.push_back(text);
        }
      }
  std::printf("{\"sid\": %d, \"rep\": %d, \"supported\": true, \"ws_need\": %zu, \"nbad\": %lld, "
              "\"nnan\": %lld, \"maxabs\": %g",
              sid,
              rep,
              need,
              static_cast<long long>(bad),
              static_cast<long long>(nan),
              max_abs);
  if (tiles_m > 0 && tiles_n > 0) dump_map("per_tile", per_tile);
  std::printf(", \"bad\": [");
  for (size_t q = 0; q < listed.size(); q++) std::printf("%s%s", q ? ", " : "", listed[q].c_str());
  std::printf("], \"name\": \"%s\"}\n", name.c_str());
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  Problem p(o);

  hipblasLtHandle_t handle;
  LT_CHECK(hipblasLtCreate(&handle));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  void *dev_a, *dev_b, *dev_c, *dev_d, *dev_ws;
  HIP_CHECK(hipMalloc(&dev_a, p.a.size()));
  HIP_CHECK(hipMalloc(&dev_b, p.b.size()));
  HIP_CHECK(hipMalloc(&dev_c, p.c.size()));
  HIP_CHECK(hipMalloc(&dev_ws, o.workspace));
  if (o.separate_d) {
    HIP_CHECK(hipMalloc(&dev_d, p.d.size()));
  } else {
    dev_d = dev_c;
  }
  HIP_CHECK(hipMemcpy(dev_a, p.a.data(), p.a.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_b, p.b.data(), p.b.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_c, p.c.data(), p.c.size(), hipMemcpyHostToDevice));

  hipblasLtMatmulDesc_t desc;
  LT_CHECK(hipblasLtMatmulDescCreate(
      &desc, o.xf32 ? HIPBLAS_COMPUTE_32F_FAST_TF32 : HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t op_a = o.trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t op_b = o.trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof op_a));
  LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof op_b));
  hipblasLtMatrixLayout_t lay_a, lay_b, lay_c, lay_d;
  LT_CHECK(hipblasLtMatrixLayoutCreate(
      &lay_a, hip_type(o.a_type), o.trans_a ? o.k : o.m, o.trans_a ? o.m : o.k, o.lda));
  LT_CHECK(hipblasLtMatrixLayoutCreate(
      &lay_b, hip_type(o.b_type), o.trans_b ? o.n : o.k, o.trans_b ? o.k : o.n, o.ldb));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&lay_c, hip_type(o.c_type), o.m, o.n, o.ldc));
  LT_CHECK(hipblasLtMatrixLayoutCreate(&lay_d, hip_type(o.d_type), o.m, o.n, o.ldd));
  int32_t batch = int32_t(o.batch);
  for (auto [layout, stride] : {std::pair{lay_a, p.stride_a},
                                std::pair{lay_b, p.stride_b},
                                std::pair{lay_c, p.stride_c},
                                std::pair{lay_d, p.stride_d}}) {
    LT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch));
    int64_t s = stride;
    LT_CHECK(hipblasLtMatrixLayoutSetAttribute(
        layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s, sizeof s));
  }

  std::vector<hipblasLtMatmulHeuristicResult_t> found;
  hipblaslt_ext::getAlgosFromIndex(handle, o.ids, found);
  std::printf("{\"ready\": true, \"algos\": %zu, \"sids\": [", found.size());
  for (size_t q = 0; q < found.size(); q++)
    std::printf("%s%d", q ? ", " : "", hipblaslt_ext::getIndexFromAlgo(found[q].algo));
  std::printf("]}\n");
  std::fflush(stdout);
  for (auto& r : found) {
    int sid                    = hipblaslt_ext::getIndexFromAlgo(r.algo);
    std::string name           = hipblaslt_ext::getSolutionNameFromAlgo(handle, r.algo);
    r.algo.max_workspace_bytes = o.workspace;
    size_t need                = 0;
    auto status                = hipblaslt_ext::matmulIsAlgoSupported(
        handle, desc, &o.alpha, lay_a, lay_b, &o.beta, lay_c, lay_d, r.algo, need);
    if (status != HIPBLAS_STATUS_SUCCESS || need > o.workspace) {
      std::printf("{\"sid\": %d, \"supported\": false, \"ws_need\": %zu, \"name\": \"%s\"}\n",
                  sid,
                  need,
                  name.c_str());
      std::fflush(stdout);
      continue;
    }
    for (int rep = 0; rep < o.reps; rep++) {
      // Elements a solution must write keep a NaN pattern (all bits set) until
      // it does; with beta != 0 and C == D, D starts as C.
      if (o.separate_d) {
        HIP_CHECK(hipMemcpyAsync(dev_c, p.c.data(), p.c.size(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemsetAsync(dev_d, 0xff, p.d.size(), stream));
      } else if (o.beta != 0.f) {
        HIP_CHECK(hipMemcpyAsync(dev_c, p.c.data(), p.c.size(), hipMemcpyHostToDevice, stream));
      } else {
        HIP_CHECK(hipMemsetAsync(dev_c, 0xff, p.c.size(), stream));
      }
      HIP_CHECK(hipStreamSynchronize(stream));
      LT_CHECK(hipblasLtMatmul(handle,
                               desc,
                               &o.alpha,
                               dev_a,
                               lay_a,
                               dev_b,
                               lay_b,
                               &o.beta,
                               dev_c,
                               lay_c,
                               dev_d,
                               lay_d,
                               &r.algo,
                               dev_ws,
                               o.workspace,
                               stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipMemcpy(p.d.data(), dev_d, p.d.size(), hipMemcpyDeviceToHost));
      report(p, sid, rep, need, name);
    }
  }
  HIP_CHECK(hipFree(dev_a));
  HIP_CHECK(hipFree(dev_b));
  HIP_CHECK(hipFree(dev_c));
  HIP_CHECK(hipFree(dev_ws));
  if (o.separate_d) HIP_CHECK(hipFree(dev_d));
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_a));
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_b));
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_c));
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_d));
  LT_CHECK(hipblasLtMatmulDescDestroy(desc));
  LT_CHECK(hipblasLtDestroy(handle));
  return 0;
}
