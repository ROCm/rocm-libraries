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
// 8-bit (f8: E4M3, bf8: E5M2) and 4-bit (f4: E2M1, two to a byte, the first
// in the low nibble) A and B hold the same integers exactly, with block-32
// E8M0 scales (--scale-a 3 and --scale-b 3, hipblaslt-bench's scaling format
// 3): every scale of A is 2 and every scale of B is 4, so every product stays
// exact, and a solution that leaves out a scale or applies one operand's
// scale to both is wrong wherever a dot product is not 0. All scales of an
// operand being equal, the harness need not know their layout, which differs
// between devices and hipBLASLt versions, but it does not check how a
// solution indexes them. It writes as many scales as the largest padding
// hipBLASLt gives a scale tensor holds: K blocks to a multiple of 8, rows to a
// multiple of 32. hipBLASLt's own data holds zero scales in that padding, and
// a solution may rely on them to cancel what it reads past K; so a
// block-scaled operand is stored with K rounded up to a multiple of 256 and
// zeros past K (its leading dimension holds that many rows where K is its
// stored rows, its batch stride that many columns where K counts them).
//
// 8-bit A and B also take one f32 scale per tensor (--scale-a 1 and
// --scale-b 1, hipblaslt-bench's scaling format 1): a device pointer to one
// float each, which hipBLASLt passes to the solution as it is and the solution
// multiplies into alpha. A's scale is 2 and B's 4 here too, and such an
// operand is stored with K as it is.
//
// With --bias 1 the epilogue adds a vector of M integers in [-2, 2] to every
// column of D in every batch.
//
// With --fresh 1 every buffer of the problem (A, B, C, a separate D, the
// scales and the bias) starts an allocation of its own, right after a
// reserved range that is never mapped (DeviceMemory), so that a read before
// a buffer's first byte faults. hipMalloc places small buffers next to each
// other in one block, where such a read lands in another buffer, returns
// bytes the solution then ignores, and goes unseen. This takes the device's
// virtual memory management; without it, or when one of its calls fails,
// the buffers come from hipMalloc, and the ready line says which ("fresh").
//
// usage: errmap M N K BATCH --ids I1,I2,... [--reps R] [--alpha A] [--beta B]
//          [--trans-a T|N] [--trans-b T|N] [--a-type bf16|f32] [--b-type bf16|f32]
//          [--c-type bf16|f32] [--d-type bf16|f32] [--compute f32|xf32]
//          [--lda X] [--ldb X] [--ldc X] [--ldd X] [--sep 0|1] [--ws BYTES]
//          [--seed S] [--maxlist N] [--mt MTMxMTN] [--du DEPTHU]
//          [--scale-a 0|1|3] [--scale-b 0|1|3] [--bias 0|1] [--bias-type bf16|f32]
//          [--corrupt-ref I,J,BATCH] [--drop-scale A|B|AB] [--fresh 0|1]
//        errmap --caps
// --caps prints {"caps": [...]}, the options of this harness that earlier
// builds did not take, and exits without calling HIP.
// --a-type and --b-type also take f8, bf8 and f4, which need scales: f8 and
// bf8 one f32 scale (1) or block-32 ones (3), f4 block-32 ones; A and B scaled
// in one format, and K a multiple of 32 with block-32 scales. An f4 operand's
// leading dimension is even (by default its stored rows rounded up to even).
// The bias type defaults to hipBLASLt's: f32 with xf32 compute, bf16 when A and
// B are 8-bit, else D's.
// --corrupt-ref changes the last bit of one element of the reference, so that
// a correct solution fails there: a check that the harness sees a wrong
// element.
// --drop-scale gives the solution scales of 1 for A, B or both while the
// reference keeps 2 and 4, so that a correct solution computes what one that
// leaves those scales out would and fails wherever a dot product is not 0: a
// check that the harness sees a scale a solution ignores.
//
// On stdout, {"ready": true, "algos": N, "sids": [...], "fresh": F} once the
// device holds the operands: the N solutions of --ids that were found, in the
// order of --ids, which they then run in, and whether every buffer of the
// problem starts an allocation of its own behind an unmapped range (--fresh).
// Then one JSON line per (solution, repetition):
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
#include <tuple>
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

enum class Type { BF16, F32, F8, BF8, F4 };

// hipblaslt-bench's scaling formats of one f32 scale per tensor and of
// block-32 E8M0 scales, the multiple of K the K blocks of a scale tensor are
// padded to, and the power of 2 every scale of A and of B holds.
constexpr int kScalar      = 1;
constexpr int kBlock32     = 3;
constexpr int64_t kMxBlock = 32;
constexpr int64_t kScaledK = 256;
constexpr int kScaleExpA   = 1;
constexpr int kScaleExpB   = 2;

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

  // Scaling formats of A and B (0: none), the K A and B are stored with, the
  // bias epilogue, the element (i, j, batch) whose reference --corrupt-ref
  // changes (corrupt_i < 0: none), and the operands whose scales --drop-scale
  // gives the solution as 1.
  int scale_a           = 0;
  int scale_b           = 0;
  int64_t k_a           = 0;
  int64_t k_b           = 0;
  bool bias             = false;
  bool bias_type_given  = false;
  Type bias_type        = Type::BF16;
  int64_t corrupt_i     = -1;
  int64_t corrupt_j     = 0;
  int64_t corrupt_batch = 0;
  bool drop_a           = false;
  bool drop_b           = false;
  // Every buffer of the problem at the start of an allocation of its own,
  // behind an unmapped range (DeviceMemory).
  bool fresh = false;
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

bool narrow(Type t) { return t == Type::F8 || t == Type::BF8 || t == Type::F4; }

size_t type_size(Type t) { return t == Type::BF16 ? 2 : 4; }

// The bytes `count` elements of type `t` take; a count of f4 elements is even.
size_t type_bytes(Type t, int64_t count) {
  switch (t) {
    case Type::F8:
    case Type::BF8: return size_t(count);
    case Type::F4: return size_t(count / 2);
    default: return size_t(count) * type_size(t);
  }
}

hipDataType hip_type(Type t) {
  switch (t) {
    case Type::F8: return HIP_R_8F_E4M3;
    case Type::BF8: return HIP_R_8F_E5M2;
    case Type::F4: return static_cast<hipDataType>(HIP_R_4F_E2M1_EXT);
    default: return t == Type::BF16 ? HIP_R_16BF : HIP_R_32F;
  }
}

Type parse_type(const std::string& s) {
  if (s == "bf16") return Type::BF16;
  if (s == "f32") return Type::F32;
  if (s == "f8") return Type::F8;
  if (s == "bf8") return Type::BF8;
  if (s == "f4") return Type::F4;
  std::fprintf(stderr, "unknown type %s (bf16, f32, f8, bf8 or f4)\n", s.c_str());
  std::exit(1);
}

// The code of integer `v` in [-2, 2] in an 8- or 4-bit type.
uint8_t narrow_code(Type t, int v) {
  static const uint8_t f8[5]  = {0xc0, 0xb8, 0x00, 0x38, 0x40};
  static const uint8_t bf8[5] = {0xc0, 0xbc, 0x00, 0x3c, 0x40};
  static const uint8_t f4[5]  = {0xc, 0xa, 0x0, 0x2, 0x4};
  const uint8_t* codes        = t == Type::F8 ? f8 : t == Type::BF8 ? bf8 : f4;
  return codes[v + 2];
}

void put(std::vector<uint8_t>& buf, Type t, int64_t at, float v) {
  if (t == Type::BF16) {
    uint16_t b = to_bf16(v);
    std::memcpy(&buf[at * 2], &b, 2);
  } else if (t == Type::F32) {
    std::memcpy(&buf[at * 4], &v, 4);
  } else if (t == Type::F4) {
    uint8_t code = narrow_code(t, int(v));
    uint8_t& b   = buf[at / 2];
    b            = at % 2 ? uint8_t((b & 0x0f) | (code << 4)) : uint8_t((b & 0xf0) | code);
  } else {
    buf[at] = narrow_code(t, int(v));
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

// Bytes of the block-32 E8M0 scales of one batch of an operand of `outer` (M
// or N) x `k`: the blocks of K rounded up to a multiple of 8 times `outer`
// rounded up to a multiple of 32, the largest padding hipBLASLt gives the
// scale tensor.
int64_t scale_bytes(int64_t outer, int64_t k) {
  int64_t blocks = (k + kMxBlock - 1) / kMxBlock;
  return (blocks + 7) / 8 * 8 * ((outer + 31) / 32 * 32);
}

// The scales of an operand of `outer` (M or N) x `k` in `batch` batches, each
// 2^`exp`: one f32 in scaling format kScalar, else E8M0 bytes (scale_bytes per
// batch).
std::vector<uint8_t> scales_of(int format, int64_t outer, int64_t k, int64_t batch, int exp) {
  if (format == kScalar) {
    float s = std::ldexp(1.f, exp);
    std::vector<uint8_t> out(sizeof s);
    std::memcpy(out.data(), &s, sizeof s);
    return out;
  }
  return std::vector<uint8_t>(scale_bytes(outer, k) * batch, uint8_t(127 + exp));
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
    } else if (a == "--scale-a") {
      o.scale_a = std::atoi(v.c_str());
    } else if (a == "--scale-b") {
      o.scale_b = std::atoi(v.c_str());
    } else if (a == "--bias") {
      o.bias = std::atoi(v.c_str()) != 0;
    } else if (a == "--bias-type") {
      o.bias_type       = parse_type(v);
      o.bias_type_given = true;
    } else if (a == "--corrupt-ref") {
      long long ci = -1, cj = -1, cb = -1;
      if (std::sscanf(v.c_str(), "%lld,%lld,%lld", &ci, &cj, &cb) != 3) ci = -1;
      if (ci < 0 || ci >= o.m || cj < 0 || cj >= o.n || cb < 0 || cb >= o.batch) {
        std::fprintf(stderr, "--corrupt-ref needs I,J,BATCH of an element of D\n");
        std::exit(1);
      }
      o.corrupt_i     = ci;
      o.corrupt_j     = cj;
      o.corrupt_batch = cb;
    } else if (a == "--drop-scale") {
      if (v != "A" && v != "B" && v != "AB") {
        std::fprintf(stderr, "--drop-scale takes A, B or AB\n");
        std::exit(1);
      }
      o.drop_a = v != "B";
      o.drop_b = v != "A";
    } else if (a == "--fresh") {
      o.fresh = std::atoi(v.c_str()) != 0;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      std::exit(1);
    }
  }
  if (o.m < 1 || o.n < 1 || o.k < 1 || o.batch < 1 || o.ids.empty() || o.reps < 1) {
    std::fprintf(stderr, "M, N, K, BATCH, --reps must be positive and --ids non-empty\n");
    std::exit(1);
  }
  for (auto [name, t, scale] :
       {std::tuple{"A", o.a_type, o.scale_a}, std::tuple{"B", o.b_type, o.scale_b}}) {
    if (scale != 0 && scale != kScalar && scale != kBlock32) {
      std::fprintf(stderr,
                   "scaling format %d of %s is not run (0, %d or %d)\n",
                   scale,
                   name,
                   kScalar,
                   kBlock32);
      std::exit(1);
    }
    bool fits = scale == kBlock32  ? narrow(t)
                : scale == kScalar ? narrow(t) && t != Type::F4
                                   : !narrow(t);
    if (!fits) {
      std::fprintf(stderr,
                   "%s takes scales exactly when it is 8- or 4-bit, one f32 scale only when it "
                   "is 8-bit\n",
                   name);
      std::exit(1);
    }
  }
  if (o.scale_a && o.scale_b && o.scale_a != o.scale_b) {
    std::fprintf(stderr, "A and B take scales of one format\n");
    std::exit(1);
  }
  if ((o.drop_a && !o.scale_a) || (o.drop_b && !o.scale_b)) {
    std::fprintf(stderr, "--drop-scale names an operand without scales\n");
    std::exit(1);
  }
  if ((o.scale_a == kBlock32 || o.scale_b == kBlock32) && o.k % kMxBlock) {
    std::fprintf(stderr, "block-32 scales need K to be a multiple of 32\n");
    std::exit(1);
  }
  if (narrow(o.c_type) || narrow(o.d_type) || narrow(o.bias_type)) {
    std::fprintf(stderr, "C, D and the bias are bf16 or f32\n");
    std::exit(1);
  }
  // The stored A is K x M when transposed, else M x K; B is N x K when
  // transposed, else K x N (column major); a block-scaled one with K rounded
  // up to a multiple of kScaledK.
  o.k_a          = o.scale_a == kBlock32 ? (o.k + kScaledK - 1) / kScaledK * kScaledK : o.k;
  o.k_b          = o.scale_b == kBlock32 ? (o.k + kScaledK - 1) / kScaledK * kScaledK : o.k;
  int64_t rows_a = o.trans_a ? o.k_a : o.m;
  int64_t rows_b = o.trans_b ? o.n : o.k_b;
  o.lda          = o.lda > 0 ? o.lda : rows_a + (o.a_type == Type::F4 ? rows_a % 2 : 0);
  o.ldb          = o.ldb > 0 ? o.ldb : rows_b + (o.b_type == Type::F4 ? rows_b % 2 : 0);
  o.ldc          = o.ldc > 0 ? o.ldc : o.m;
  o.ldd          = o.ldd > 0 ? o.ldd : (o.separate_d ? o.m : o.ldc);
  if (o.lda < rows_a || o.ldb < rows_b || o.ldc < o.m || o.ldd < o.m) {
    std::fprintf(stderr, "a leading dimension is below its matrix's rows\n");
    std::exit(1);
  }
  if ((o.a_type == Type::F4 && o.lda % 2) || (o.b_type == Type::F4 && o.ldb % 2)) {
    std::fprintf(stderr, "an f4 operand needs an even leading dimension\n");
    std::exit(1);
  }
  if (!o.separate_d && (o.ldd != o.ldc || o.d_type != o.c_type)) {
    std::fprintf(stderr, "C == D needs equal leading dimensions and types (or --sep 1)\n");
    std::exit(1);
  }
  if (!o.bias_type_given) {
    bool f8_ab =
        narrow(o.a_type) && narrow(o.b_type) && o.a_type != Type::F4 && o.b_type != Type::F4;
    o.bias_type = o.xf32 ? Type::F32 : f8_ab ? Type::BF16 : o.d_type;
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
  // The scales of A and B the solution is given (empty when the operand has
  // none), the bias vector and its integers, and the product of the scales of
  // A and B the reference applies.
  std::vector<uint8_t> scale_a, scale_b, bias;
  std::vector<int8_t> int_bias;
  int32_t unit = 1;

  explicit Problem(const Options& opt) : o(opt) {
    stride_a = o.lda * (o.trans_a ? o.m : o.k_a);
    stride_b = o.ldb * (o.trans_b ? o.k_b : o.n);
    stride_c = o.ldc * o.n;
    stride_d = o.ldd * o.n;
    a.assign(type_bytes(o.a_type, stride_a * o.batch), 0);
    b.assign(type_bytes(o.b_type, stride_b * o.batch), 0);
    c.assign(stride_c * o.batch * type_size(o.c_type), 0);
    d.assign(stride_d * o.batch * type_size(o.d_type), 0);
    ref.assign(d.size(), 0xff);
    int_a.resize(o.batch * o.m * o.k);
    int_b.resize(o.batch * o.n * o.k);
    int_c.resize(o.batch * o.n * o.m);
    int_ref.resize(o.batch * o.n * o.m);
    if (o.scale_a) {
      scale_a = scales_of(o.scale_a, o.m, o.k, o.batch, o.drop_a ? 0 : kScaleExpA);
      unit <<= kScaleExpA;
    }
    if (o.scale_b) {
      scale_b = scales_of(o.scale_b, o.n, o.k, o.batch, o.drop_b ? 0 : kScaleExpB);
      unit <<= kScaleExpB;
    }
    if (o.bias) {
      bias.assign(o.m * type_size(o.bias_type), 0);
      int_bias.resize(o.m);
    }
    fill();
    reference();
    if (o.corrupt_i >= 0) {
      int64_t at = o.corrupt_batch * stride_d + o.corrupt_j * o.ldd + o.corrupt_i;
      ref[at * type_size(o.d_type)] ^= 1;
    }
  }

  // Fills operand A (`outer` = M) or B (N) of type `t`, stored column major
  // with leading dimension `ld` and batch stride `stride`, K as its rows when
  // `k_rows`: element (x, kk) of op(A) or op(B) in batch `bt` holds
  // value_of((bt * outer + x) * K + kk, seed), which `ints` keeps at that
  // index. Columns run in parallel: no two share a byte, the leading
  // dimension and batch stride of an f4 operand being even.
  void fill_operand(std::vector<uint8_t>& buf,
                    Type t,
                    std::vector<int8_t>& ints,
                    int64_t outer,
                    bool k_rows,
                    int64_t ld,
                    int64_t stride,
                    uint64_t seed) {
    int64_t rows = k_rows ? o.k : outer;
    int64_t cols = k_rows ? outer : o.k;
#pragma omp parallel for collapse(2)
    for (int64_t bt = 0; bt < o.batch; bt++)
      for (int64_t col = 0; col < cols; col++)
        for (int64_t row = 0; row < rows; row++) {
          int64_t idx = (bt * outer + (k_rows ? col : row)) * o.k + (k_rows ? row : col);
          int v       = value_of(uint64_t(idx), seed);
          put(buf, t, bt * stride + col * ld + row, float(v));
          ints[idx] = int8_t(v);
        }
  }

  void fill() {
    fill_operand(a, o.a_type, int_a, o.m, o.trans_a, o.lda, stride_a, o.seed);
    fill_operand(b, o.b_type, int_b, o.n, !o.trans_b, o.ldb, stride_b, o.seed ^ 0x5555);
#pragma omp parallel for collapse(2)
    for (int64_t bt = 0; bt < o.batch; bt++)
      for (int64_t j = 0; j < o.n; j++)
        for (int64_t i = 0; i < o.m; i++) {
          int v = value_of((uint64_t(bt) * o.n + j) * o.m + i, o.seed ^ 0xAAAA);
          put(c, o.c_type, bt * stride_c + j * o.ldc + i, float(v));
          int_c[(bt * o.n + j) * o.m + i] = int8_t(v);
        }
    for (int64_t i = 0; i < int64_t(int_bias.size()); i++) {
      int v = value_of(uint64_t(i), o.seed ^ 0x3333);
      put(bias, o.bias_type, i, float(v));
      int_bias[i] = int8_t(v);
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
          float total = o.alpha * float(s * unit) +
                        (o.beta != 0.f ? o.beta * float(int_c[(bt * o.n + j) * o.m + i]) : 0.f);
          if (o.bias) total += float(int_bias[i]);
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
    for (int64_t kk = 0; kk < o.k; kk++) part[kk / o.depth_u] += aa[kk] * bb[kk] * unit;
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

// The unmapped range reserved before every fresh buffer: at least the block
// hipMalloc places small buffers in, in whole allocation granules.
constexpr size_t kGuardBytes = size_t(2) << 20;

// The device buffers of one process. A fresh buffer (`fresh` and the
// device's virtual memory management) is a physical allocation of its own
// (hipMemCreate) mapped at the start of what remains of a reserved virtual
// range after its first kGuardBytes, which stay unmapped: the buffer's first
// byte is its allocation's, and a read before it faults. A buffer that cannot
// be mapped so, and every buffer after it, comes from hipMalloc, and `fresh()`
// is false from then on.
class DeviceMemory {
 public:
  explicit DeviceMemory(bool fresh) {
    if (!fresh) return;
    int device = 0, vmm = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, device) !=
            hipSuccess ||
        vmm == 0)
      return;
    prop_.type                = hipMemAllocationTypePinned;
    prop_.requestedHandleType = hipMemHandleTypeNone;
    prop_.location.type       = hipMemLocationTypeDevice;
    prop_.location.id         = device;
    access_.location          = prop_.location;
    access_.flags             = hipMemAccessFlagsProtReadWrite;
    if (hipMemGetAllocationGranularity(&granule_, &prop_, hipMemAllocationGranularityMinimum) !=
            hipSuccess ||
        granule_ == 0)
      return;
    fresh_ = true;
  }

  bool fresh() const { return fresh_; }

  // A buffer of `bytes`; a fresh one is zeroed first, which checks that its
  // mapping takes writes.
  void* alloc(size_t bytes) {
    if (fresh_) {
      void* p = map(bytes);
      if (p != nullptr && hipMemset(p, 0, bytes) == hipSuccess) return p;
      if (p != nullptr) unmap_last();
      fresh_ = false;
    }
    return plain(bytes);
  }

  // A device copy of `host`, or nullptr when it is empty.
  void* copy(const std::vector<uint8_t>& host) {
    if (host.empty()) return nullptr;
    if (fresh_) {
      void* p = map(host.size());
      if (p != nullptr &&
          hipMemcpy(p, host.data(), host.size(), hipMemcpyHostToDevice) == hipSuccess)
        return p;
      if (p != nullptr) unmap_last();
      fresh_ = false;
    }
    void* dev = plain(host.size());
    HIP_CHECK(hipMemcpy(dev, host.data(), host.size(), hipMemcpyHostToDevice));
    return dev;
  }

  // A buffer from hipMalloc, also with --fresh (the workspace).
  void* plain(size_t bytes) {
    void* dev = nullptr;
    HIP_CHECK(hipMalloc(&dev, bytes));
    plain_.push_back(dev);
    return dev;
  }

  void release() {
    while (!mappings_.empty()) unmap_last();
    for (void* dev : plain_) HIP_CHECK(hipFree(dev));
    plain_.clear();
  }

 private:
  struct Mapping {
    void* base      = nullptr;
    size_t reserved = 0;
    void* buffer    = nullptr;
    size_t mapped   = 0;
    hipMemGenericAllocationHandle_t handle{};
  };

  size_t round_up(size_t bytes) const { return (bytes + granule_ - 1) / granule_ * granule_; }

  // The buffer of a new mapping of `bytes`, or nullptr (nothing left behind).
  void* map(size_t bytes) {
    Mapping m;
    size_t guard = round_up(kGuardBytes);
    m.mapped     = round_up(std::max<size_t>(bytes, 1));
    m.reserved   = guard + m.mapped;
    if (hipMemAddressReserve(&m.base, m.reserved, granule_, nullptr, 0) != hipSuccess)
      return nullptr;
    if (hipMemCreate(&m.handle, m.mapped, &prop_, 0) != hipSuccess) {
      (void)hipMemAddressFree(m.base, m.reserved);
      return nullptr;
    }
    m.buffer = static_cast<char*>(m.base) + guard;
    if (hipMemMap(m.buffer, m.mapped, 0, m.handle, 0) != hipSuccess) {
      (void)hipMemRelease(m.handle);
      (void)hipMemAddressFree(m.base, m.reserved);
      return nullptr;
    }
    mappings_.push_back(m);
    if (hipMemSetAccess(m.buffer, m.mapped, &access_, 1) != hipSuccess) {
      unmap_last();
      return nullptr;
    }
    return m.buffer;
  }

  void unmap_last() {
    Mapping m = mappings_.back();
    mappings_.pop_back();
    (void)hipMemUnmap(m.buffer, m.mapped);
    (void)hipMemRelease(m.handle);
    (void)hipMemAddressFree(m.base, m.reserved);
  }

  bool fresh_     = false;
  size_t granule_ = 0;
  hipMemAllocationProp prop_{};
  hipMemAccessDesc access_{};
  std::vector<Mapping> mappings_;
  std::vector<void*> plain_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--caps") {
    std::printf("{\"caps\": [\"fresh\"]}\n");
    return 0;
  }
  Options o = parse(argc, argv);
  Problem p(o);

  hipblasLtHandle_t handle;
  LT_CHECK(hipblasLtCreate(&handle));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  DeviceMemory memory(o.fresh);
  void* dev_a       = memory.copy(p.a);
  void* dev_b       = memory.copy(p.b);
  void* dev_c       = memory.copy(p.c);
  void* dev_d       = o.separate_d ? memory.alloc(p.d.size()) : dev_c;
  void* dev_scale_a = memory.copy(p.scale_a);
  void* dev_scale_b = memory.copy(p.scale_b);
  void* dev_bias    = memory.copy(p.bias);
  void* dev_ws      = memory.plain(o.workspace);

  hipblasLtMatmulDesc_t desc;
  LT_CHECK(hipblasLtMatmulDescCreate(
      &desc, o.xf32 ? HIPBLAS_COMPUTE_32F_FAST_TF32 : HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t op_a = o.trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t op_b = o.trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof op_a));
  LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof op_b));
  for (auto [scale, format, pointer, mode] : {std::tuple{dev_scale_a,
                                                         o.scale_a,
                                                         HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                         HIPBLASLT_MATMUL_DESC_A_SCALE_MODE},
                                              std::tuple{dev_scale_b,
                                                         o.scale_b,
                                                         HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                         HIPBLASLT_MATMUL_DESC_B_SCALE_MODE}}) {
    if (scale == nullptr) continue;
    hipblasLtMatmulMatrixScale_t scaling = format == kScalar
                                               ? HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F
                                               : HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, pointer, &scale, sizeof scale));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(desc, mode, &scaling, sizeof scaling));
  }
  if (dev_bias != nullptr) {
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
    int32_t bias_type            = hip_type(o.bias_type);
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof epilogue));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dev_bias, sizeof dev_bias));
    LT_CHECK(hipblasLtMatmulDescSetAttribute(
        desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof bias_type));
  }
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
  std::printf("], \"fresh\": %s}\n", memory.fresh() ? "true" : "false");
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
  memory.release();
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_a));
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_b));
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_c));
  LT_CHECK(hipblasLtMatrixLayoutDestroy(lay_d));
  LT_CHECK(hipblasLtMatmulDescDestroy(desc));
  LT_CHECK(hipblasLtDestroy(handle));
  return 0;
}
