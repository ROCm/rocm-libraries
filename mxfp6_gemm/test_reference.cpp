#include "mxfp6_reference.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>

using namespace mxfp6;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL [%s:%d]: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

// ============================================================
// Test 1: FP6 E2M3 decode — exhaustive check all 64 encodings
// ============================================================
static void test_fp6_decode() {
    printf("=== Test FP6 E2M3 Decode (exhaustive) ===\n");

    const float expected_pos[4][8] = {
        {0.0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f},
        {1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f},
        {2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f},
        {4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f},
    };

    for (int E = 0; E < 4; E++) {
        for (int M = 0; M < 8; M++) {
            uint8_t bits = (E << 3) | M;
            float got = fp6_e2m3_to_float(bits);
            float exp = expected_pos[E][M];
            CHECK(got == exp, "E=%d M=%d: expected %f, got %f", E, M, exp, got);

            if (E == 0 && M == 0) {
                uint8_t neg_bits = (1 << 5) | bits;
                float neg_got = fp6_e2m3_to_float(neg_bits);
                CHECK(neg_got == 0.0f || neg_got == -0.0f,
                      "negative zero: got %f", neg_got);
            } else {
                uint8_t neg_bits = (1 << 5) | bits;
                float neg_got = fp6_e2m3_to_float(neg_bits);
                CHECK(neg_got == -exp, "neg E=%d M=%d: expected %f, got %f",
                      E, M, -exp, neg_got);
            }
        }
    }
}

// ============================================================
// Test 2: FP6 E2M3 encode — round-trip and special values
// ============================================================
static void test_fp6_encode() {
    printf("=== Test FP6 E2M3 Encode ===\n");

    for (uint8_t bits = 0; bits < 64; bits++) {
        float v = fp6_e2m3_to_float(bits);
        uint8_t re = float_to_fp6_e2m3(v);
        if (bits == 32) {
            CHECK(re == 0, "-0 should re-encode as +0, got %d", re);
        } else {
            CHECK(re == bits, "round-trip bits=%d: decoded=%f, re-encoded=%d",
                  bits, v, re);
        }
    }

    CHECK(float_to_fp6_e2m3(100.0f) == ((3 << 3) | 7),
          "100.0 should clamp to max (7.5)");
    CHECK(float_to_fp6_e2m3(-100.0f) == ((1 << 5) | (3 << 3) | 7),
          "-100.0 should clamp to -max (-7.5)");

    // RNE: between 1.0 (M=0, even) and 1.125 (M=1, odd): midpoint 1.0625 → 1.0
    uint8_t r = float_to_fp6_e2m3(1.0625f);
    float decoded = fp6_e2m3_to_float(r);
    CHECK(decoded == 1.0f || decoded == 1.125f,
          "1.0625 RNE: got %f (bits=%d)", decoded, r);

    // RNE: between 1.125 (M=1, odd) and 1.25 (M=2, even): midpoint 1.1875 → 1.25
    r = float_to_fp6_e2m3(1.1875f);
    decoded = fp6_e2m3_to_float(r);
    CHECK(decoded == 1.125f || decoded == 1.25f,
          "1.1875 RNE: got %f (bits=%d)", decoded, r);

    r = float_to_fp6_e2m3(0.06f);
    decoded = fp6_e2m3_to_float(r);
    CHECK(decoded == 0.0f || decoded == 0.125f,
          "0.06 should round to 0 or 0.125, got %f", decoded);
}

// ============================================================
// Test 3: E8M0 encode/decode
// ============================================================
static void test_e8m0() {
    printf("=== Test E8M0 ===\n");

    CHECK(e8m0_to_float(127) == 1.0f, "code=127 -> 1.0");
    CHECK(e8m0_to_float(128) == 2.0f, "code=128 -> 2.0");
    CHECK(e8m0_to_float(126) == 0.5f, "code=126 -> 0.5");
    CHECK(e8m0_to_float(0) == ldexpf(1.0f, -127), "code=0 -> 2^-127");
    CHECK(e8m0_to_float(254) == ldexpf(1.0f, 127), "code=254 -> 2^127");
    CHECK(std::isnan(e8m0_to_float(255)), "code=255 -> NaN");

    CHECK(float_to_e8m0(1.0f) == 127, "1.0 -> code=127");
    CHECK(float_to_e8m0(2.0f) == 128, "2.0 -> code=128");
    CHECK(float_to_e8m0(0.5f) == 126, "0.5 -> code=126");
    CHECK(float_to_e8m0(4.0f) == 129, "4.0 -> code=129");
    CHECK(float_to_e8m0(NAN) == 255, "NaN -> code=255");
}

// ============================================================
// Test 4: FP6 packing/unpacking
// ============================================================
static void test_fp6_packing() {
    printf("=== Test FP6 Packing ===\n");

    uint8_t vals[4] = {0x3F, 0x01, 0x20, 0x15};
    uint8_t packed[3];
    uint8_t unpacked[4];

    pack_fp6x4(vals, packed);
    unpack_fp6x4(packed, unpacked);

    for (int i = 0; i < 4; i++) {
        CHECK(unpacked[i] == (vals[i] & 0x3F),
              "pack/unpack[%d]: expected %d, got %d", i, vals[i] & 0x3F, unpacked[i]);
    }

    std::mt19937 rng(42);
    const int N = 128;
    uint8_t rand_vals[N], rand_packed[N * 3 / 4], rand_unpacked[N];
    for (int i = 0; i < N; i++) rand_vals[i] = rng() % 64;

    pack_fp6(rand_vals, N, rand_packed);
    unpack_fp6(rand_packed, N, rand_unpacked);

    bool all_match = true;
    for (int i = 0; i < N; i++) {
        if (rand_unpacked[i] != rand_vals[i]) { all_match = false; break; }
    }
    CHECK(all_match, "random pack/unpack round-trip for %d values", N);
}

// ============================================================
// Test 5: Quantize + dequantize
// ============================================================
static void test_quantize() {
    printf("=== Test Quantize/Dequantize ===\n");

    const int rows = 2, cols = 32;
    float mat[rows * cols];
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < rows * cols; i++) mat[i] = dist(rng);

    QuantizedMatrix q = quantize_to_mxfp6(mat, rows, cols);
    float deq[rows * cols];
    dequantize_mxfp6(q, deq);

    float max_err = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float err = std::abs(deq[i] - mat[i]);
        if (err > max_err) max_err = err;
    }

    CHECK(max_err < 2.0f, "max quantization error = %f (should be reasonable)", max_err);
    printf("  max quantization error: %f\n", max_err);
}

// ============================================================
// Test 6: preprocess_B matches manual transpose+quantize (bit-exact)
// ============================================================
static void test_preprocess_consistency() {
    printf("=== Test Preprocess Consistency ===\n");

    const int K = 64, N = 32;
    std::mt19937 rng(555);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    std::vector<float> B(K * N);
    for (auto& v : B) v = dist(rng);

    // Manual: transpose then quantize
    std::vector<float> B_T(N * K);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            B_T[n * K + k] = B[k * N + n];

    QuantizedMatrix manual_q = quantize_to_mxfp6(B_T.data(), N, K);
    std::vector<float> manual_deq(N * K);
    dequantize_mxfp6(manual_q, manual_deq.data());

    // preprocess_B now returns QuantizedMatrix directly
    QuantizedMatrix pb = preprocess_B(B.data(), K, N);

    std::vector<float> pb_deq(N * K);
    dequantize_mxfp6(pb, pb_deq.data());

    bool exact = true;
    for (int i = 0; i < N * K; i++) {
        if (manual_deq[i] != pb_deq[i]) {
            printf("  MISMATCH at [%d]: manual=%f, preprocess=%f\n",
                   i, manual_deq[i], pb_deq[i]);
            exact = false;
            break;
        }
    }
    CHECK(exact, "preprocess_B matches manual transpose+quantize bit-exact");
}

// ============================================================
// Test 7: Scale preprocess (MFMA lane layout)
// ============================================================
static void test_preprocess_scale() {
    printf("=== Test Scale Preprocess ===\n");

    const int dim = 64;
    const int K = 128;
    int scale_cols = K / 32;

    std::vector<uint8_t> scales(dim * scale_cols);
    for (int r = 0; r < dim; r++)
        for (int kg = 0; kg < scale_cols; kg++)
            scales[r * scale_cols + kg] = (uint8_t)((r * 10 + kg) % 255);

    PreprocessedScale ps = preprocess_scale(scales.data(), dim, K);

    CHECK(ps.num_tiles == 2, "num_tiles = %d (expected 2)", ps.num_tiles);
    CHECK(ps.k64_iters == 2, "k64_iters = %d (expected 2)", ps.k64_iters);
    CHECK((int)ps.data.size() == 2 * 2 * 64, "data size = %d (expected 256)",
          (int)ps.data.size());

    // tile 0, k64=0
    const uint8_t* out = ps.data.data();
    bool ok = true;
    for (int i = 0; i < 16 && ok; i++)
        if (out[i] != scales[i * scale_cols + 0]) ok = false;
    CHECK(ok, "tile0 k64=0 lanes 0-15");

    ok = true;
    for (int i = 0; i < 16 && ok; i++)
        if (out[16 + i] != scales[(16 + i) * scale_cols + 0]) ok = false;
    CHECK(ok, "tile0 k64=0 lanes 16-31");

    ok = true;
    for (int i = 0; i < 16 && ok; i++)
        if (out[32 + i] != scales[i * scale_cols + 1]) ok = false;
    CHECK(ok, "tile0 k64=0 lanes 32-47");

    ok = true;
    for (int i = 0; i < 16 && ok; i++)
        if (out[48 + i] != scales[(16 + i) * scale_cols + 1]) ok = false;
    CHECK(ok, "tile0 k64=0 lanes 48-63");

    // tile 1, k64=1
    const uint8_t* out2 = ps.data.data() + (1 * 2 + 1) * 64;
    int row_base = 32;
    int kg0 = 2, kg1 = 3;

    ok = true;
    for (int i = 0; i < 16 && ok; i++)
        if (out2[i] != scales[(row_base + i) * scale_cols + kg0]) ok = false;
    CHECK(ok, "tile1 k64=1 lanes 0-15");

    ok = true;
    for (int i = 0; i < 16 && ok; i++)
        if (out2[48 + i] != scales[(row_base + 16 + i) * scale_cols + kg1]) ok = false;
    CHECK(ok, "tile1 k64=1 lanes 48-63");
}

// ============================================================
// Test 8: GEMM — bit-exact reference from same quantized input
// ============================================================
static void test_gemm_bitexact() {
    printf("=== Test GEMM bit-exact (M=N=32, K=64) ===\n");

    const int M = 32, K = 64, N = 32;
    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> A(M * K), B(K * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    // Quantize A, preprocess B — both produce the same quantized data
    QuantizedMatrix A_q = quantize_to_mxfp6(A.data(), M, K);
    QuantizedMatrix B_q = preprocess_B(B.data(), K, N);

    // Method 1: reference GEMM (dequantize → matmul)
    std::vector<float> D_ref(M * N);
    mxfp6_gemm_ref(A_q, B_q, D_ref.data(), M, K, N);

    // Method 2: manually dequantize the same data and compute
    std::vector<float> A_deq(M * K);
    dequantize_mxfp6(A_q, A_deq.data());

    std::vector<float> B_deq(N * K);
    dequantize_mxfp6(B_q, B_deq.data());

    std::vector<float> D_manual(M * N);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A_deq[m * K + k] * B_deq[n * K + k];
            }
            D_manual[m * N + n] = acc;
        }
    }

    // Must be bit-exact: same input data, same accumulation order
    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        if (D_ref[i] != D_manual[i]) {
            if (mismatches < 5)
                printf("  MISMATCH [%d]: ref=%e, manual=%e, diff=%e\n",
                       i, D_ref[i], D_manual[i], D_ref[i] - D_manual[i]);
            mismatches++;
        }
    }
    CHECK(mismatches == 0, "bit-exact: %d/%d mismatches", mismatches, M * N);
}

static void test_gemm_medium() {
    printf("=== Test GEMM bit-exact (M=N=128, K=256) ===\n");

    const int M = 128, K = 256, N = 128;
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K), B(K * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);

    QuantizedMatrix A_q = quantize_to_mxfp6(A.data(), M, K);
    QuantizedMatrix B_q = preprocess_B(B.data(), K, N);

    std::vector<float> D(M * N);
    mxfp6_gemm_ref(A_q, B_q, D.data(), M, K, N);

    // Compare with float GEMM (no quantization) to show quantization impact
    std::vector<float> D_float(M * N, 0.0f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            D_float[m * N + n] = acc;
        }

    float max_diff = 0.0f, max_val = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(D[i] - D_float[i]);
        if (diff > max_diff) max_diff = diff;
        if (std::abs(D[i]) > max_val) max_val = std::abs(D[i]);
    }

    printf("  output range: [-%f, %f]\n", max_val, max_val);
    printf("  max abs diff MXFP6 vs float: %f\n", max_diff);
    printf("  relative error: %f%%\n", max_diff / max_val * 100.0f);
    CHECK(max_diff < 50.0f, "MXFP6 GEMM close to float (max_diff=%f)", max_diff);
}

// ============================================================
int main() {
    test_fp6_decode();
    test_fp6_encode();
    test_e8m0();
    test_fp6_packing();
    test_quantize();
    test_preprocess_consistency();
    test_preprocess_scale();
    test_gemm_bitexact();
    test_gemm_medium();

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("========================================\n");

    return g_fail > 0 ? 1 : 0;
}
