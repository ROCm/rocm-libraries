// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-only unit tests for the ck_dsl runtime's pure (GPU-free, comgr-free) host
// logic. NONE of these touch a GPU or libamd_comgr: they exercise only
//   * Manifest::parse / from_json    (manifest.hpp)
//   * Kernel::pack_args              (kernel.hpp -- kernarg byte packing/bounds)
//   * Kernel::gemm_grid             (kernel.hpp -- grid geometry)
//   * Dispatcher::rank / select      (dispatcher.hpp -- candidate ranking)
//   * ArtifactStore::add_bundle      (artifact_store.hpp -- manifest indexing)
// so the file runs on any host (CI, laptop) with no ROCm device present.
//
// It complements the GPU launch test (test_runtime.cpp) and the provider's
// parser layout tests. Build target: ck_dsl_runtime_hostonly_test.
//
// Coverage focus (the hardening surface):
//   - kernarg packing matches the AMDGPU alignment ABI for a mixed ptr/scalar
//     signature, and REJECTS malformed widths / missing args (no OOB).
//   - manifest parsing tolerates absent/optional fields and odd JSON.
//   - the dispatcher ranks by the documented per-op order and applies the
//     dtype / arch-family / shape gates.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "ck_dsl_runtime/artifact_store.hpp"
#include "ck_dsl_runtime/dispatcher.hpp"
#include "ck_dsl_runtime/kernel.hpp"
#include "ck_dsl_runtime/manifest.hpp"

namespace {
int g_fail = 0;
void check(bool cond, const char* what) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) ++g_fail;
}
template <typename Fn>
bool throws(Fn&& f) {
    try {
        f();
    } catch (...) {
        return true;
    }
    return false;
}

using ck_dsl::ArtifactStore;
using ck_dsl::Dispatcher;
using ck_dsl::Kernel;
using ck_dsl::Manifest;
using ck_dsl::Problem;

// ----- Manifest parsing ---------------------------------------------------

void test_manifest_parse() {
    std::printf("-- manifest parse --\n");
    const char* j = R"({
      "schema":"ck.dsl.example.manifest/v1",
      "kind":"gemm_fp16",
      "kernel_name":"gemm_k0",
      "hsaco":"gemm_k0.hsaco",
      "threads_per_block":256,
      "block_m":128,"block_n":128,"block_k":32,
      "grid_order":"NM",
      "priority":20,
      "args_signature":[
        {"name":"A","type":"ptr<f16, global>","size_bytes":8},
        {"name":"B","type":"ptr<f16, global>","size_bytes":8},
        {"name":"C","type":"ptr<f16, global>","size_bytes":8},
        {"name":"M","type":"i32","size_bytes":4},
        {"name":"N","type":"i32","size_bytes":4},
        {"name":"K","type":"i32","size_bytes":4}
      ]
    })";
    Manifest m = Manifest::parse(j);
    check(m.schema == "ck.dsl.example.manifest/v1", "schema");
    check(m.kind == "gemm_fp16", "kind");
    check(m.kernel_name == "gemm_k0", "kernel_name");
    check(m.block_m == 128 && m.block_n == 128 && m.block_k == 32, "block dims");
    check(m.grid_order == "NM", "grid_order NM");
    check(m.args_signature.size() == 6, "6 args parsed");
    check(m.args_signature[0].is_pointer() && m.args_signature[0].width() == 8, "ptr width 8");
    check(!m.args_signature[3].is_pointer() && m.args_signature[3].width() == 4, "i32 width 4");
    check(m.raw.get_int("priority") == 20, "raw priority readable");
    check(m.id() == "gemm_k0", "id falls back to kernel_name");

    // Missing optional fields -> defaults, no throw.
    Manifest m2 = Manifest::parse(R"({"schema":"s","kind":"k","kernel_name":"n"})");
    check(m2.threads_per_block == 256, "default tpb 256");
    check(m2.block_m == 0 && m2.grid_order == "MN", "defaults block/grid");
    check(m2.args_signature.empty(), "no args -> empty sig");

    // width() derivation when size_bytes is absent (attention-style sig).
    Manifest m3 = Manifest::parse(
        R"({"schema":"s","kind":"attention_unified","kernel_name":"a",
            "args_signature":[{"name":"q","type":"ptr<f16, global>"},
                              {"name":"s","type":"i64"},
                              {"name":"n","type":"i32"},
                              {"name":"f","type":"f32"}]})");
    check(m3.args_signature[0].width() == 8, "ptr width derived 8");
    check(m3.args_signature[1].width() == 8, "i64 width derived 8");
    check(m3.args_signature[2].width() == 4, "i32 width derived 4");
    check(m3.args_signature[3].width() == 4, "f32 width derived 4");

    // grid_explicit array.
    Manifest m4 =
        Manifest::parse(R"({"schema":"s","kind":"k","kernel_name":"n","grid_explicit":[7,3,1]})");
    check(m4.grid_explicit.has_value() && (*m4.grid_explicit)[0] == 7, "grid_explicit parsed");
}

// ----- kernarg packing (the hardened path) --------------------------------

Manifest sig_manifest(const std::string& args_json) {
    return Manifest::parse(R"({"schema":"s","kind":"gemm_fp16","kernel_name":"k",)" +
                           std::string("\"args_signature\":") + args_json + "}");
}

void test_pack_args() {
    std::printf("-- kernarg packing --\n");
    // 3 pointers (8B each) + 3 i32 (4B each). AMDGPU aligns each arg to its size:
    //   off 0  A(8) | 8  B(8) | 16 C(8) | 24 M(4) | 28 N(4) | 32 K(4) -> 36
    //   segment aligned to max member (8) -> total 40.
    Manifest m = sig_manifest(R"([
        {"name":"A","type":"ptr<f16, global>","size_bytes":8},
        {"name":"B","type":"ptr<f16, global>","size_bytes":8},
        {"name":"C","type":"ptr<f16, global>","size_bytes":8},
        {"name":"M","type":"i32","size_bytes":4},
        {"name":"N","type":"i32","size_bytes":4},
        {"name":"K","type":"i32","size_bytes":4}])");
    Kernel k = Kernel::from_hsaco({}, m);  // no GPU; pack_args is pure

    void* pA = reinterpret_cast<void*>(0x1000);
    void* pB = reinterpret_cast<void*>(0x2000);
    void* pC = reinterpret_cast<void*>(0x3000);
    auto buf =
        k.pack_args({{"A", pA}, {"B", pB}, {"C", pC}}, {{"M", 64u}, {"N", 128u}, {"K", 32u}});
    check(buf.size() == 40, "packed size = 40 (8*3 + 4*3, seg-aligned to 8)");
    void* gotA;
    std::memcpy(&gotA, buf.data() + 0, 8);
    void* gotC;
    std::memcpy(&gotC, buf.data() + 16, 8);
    uint32_t gotM, gotN, gotK;
    std::memcpy(&gotM, buf.data() + 24, 4);
    std::memcpy(&gotN, buf.data() + 28, 4);
    std::memcpy(&gotK, buf.data() + 32, 4);
    check(gotA == pA && gotC == pC, "pointers at aligned offsets");
    check(gotM == 64 && gotN == 128 && gotK == 32, "scalars packed little-endian");

    // Mixed alignment: i64 forces 8B alignment with a preceding i32 (pad).
    Manifest m2 = sig_manifest(R"([
        {"name":"a","type":"i32","size_bytes":4},
        {"name":"b","type":"i64","size_bytes":8}])");
    Kernel k2 = Kernel::from_hsaco({}, m2);
    auto buf2 = k2.pack_args({}, {{"a", 1u}, {"b", 0x1122334455667788ull}});
    // off 0 a(4) | pad to 8 | 8 b(8) -> 16, seg align 8 -> 16.
    check(buf2.size() == 16, "i32+i64 packs to 16 with 4B pad");
    uint64_t gotb;
    std::memcpy(&gotb, buf2.data() + 8, 8);
    check(gotb == 0x1122334455667788ull, "i64 at offset 8");

    // Missing args throw cleanly (no OOB / no crash).
    check(throws([&] { k.pack_args({{"A", pA}}, {}); }), "missing pointer arg throws");
    check(throws([&] { k2.pack_args({}, {{"a", 1u}}); }), "missing scalar arg throws");

    // Malformed widths (hardening): non-power-of-two and >8 are rejected, not
    // packed (would otherwise corrupt alignment math / read past the value).
    Manifest bad3 = sig_manifest(R"([{"name":"x","type":"i32","size_bytes":3}])");
    Kernel kb3 = Kernel::from_hsaco({}, bad3);
    check(throws([&] { kb3.pack_args({}, {{"x", 1u}}); }), "width 3 (non-pow2) rejected");

    Manifest bad16 = sig_manifest(R"([{"name":"x","type":"i32","size_bytes":16}])");
    Kernel kb16 = Kernel::from_hsaco({}, bad16);
    check(throws([&] { kb16.pack_args({}, {{"x", 1u}}); }), "width 16 (>8) rejected");

    // Empty signature -> empty buffer, no throw.
    Manifest e = sig_manifest("[]");
    Kernel ke = Kernel::from_hsaco({}, e);
    check(ke.pack_args({}, {}).empty(), "empty signature -> empty buffer");
}

void test_gemm_grid() {
    std::printf("-- gemm grid --\n");
    Manifest m = Manifest::parse(
        R"({"schema":"s","kind":"gemm_fp16","kernel_name":"k","block_m":128,"block_n":64,
            "grid_order":"MN"})");
    Kernel k = Kernel::from_hsaco({}, m);
    auto g = k.gemm_grid(256, 256);  // ceil(256/128)=2, ceil(256/64)=4
    check(g[0] == 2 && g[1] == 4 && g[2] == 1, "MN grid {2,4,1}");
    auto g2 = k.gemm_grid(300, 200);  // ceil(300/128)=3, ceil(200/64)=4
    check(g2[0] == 3 && g2[1] == 4, "MN ceil-div rounds up");

    Manifest mn = Manifest::parse(
        R"({"schema":"s","kind":"gemm_fp16","kernel_name":"k","block_m":128,"block_n":64,
            "grid_order":"NM"})");
    Kernel kn = Kernel::from_hsaco({}, mn);
    auto gn = kn.gemm_grid(256, 256);  // swapped
    check(gn[0] == 4 && gn[1] == 2, "NM grid swaps x/y");

    Manifest me =
        Manifest::parse(R"({"schema":"s","kind":"k","kernel_name":"k","grid_explicit":[5,6,7]})");
    Kernel ke = Kernel::from_hsaco({}, me);
    auto ge = ke.gemm_grid(999, 999);
    check(ge[0] == 5 && ge[1] == 6 && ge[2] == 7, "grid_explicit overrides tiling");
}

// ----- dispatcher ranking -------------------------------------------------

// Build an ArtifactStore in memory by writing tiny manifest files to a temp dir.
struct TmpBundle {
    std::string dir;
    explicit TmpBundle(const std::string& name) {
        dir = std::string("/tmp/ckdsl_hostonly_") + name;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
    }
    void add(const std::string& fname, const std::string& json) {
        std::ofstream(dir + "/" + fname) << json;
    }
    ~TmpBundle() {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
};

void test_dispatcher_gemm() {
    std::printf("-- dispatcher gemm ranking --\n");
    TmpBundle b("gemm");
    // Two fp16 candidates: a big 128x128 cshuffle and a small 64x128 mem tile.
    b.add("big.manifest.json",
          R"({"schema":"s","kind":"gemm_fp16","kernel_name":"big","cache_key":"big",
              "block_m":128,"block_n":128,"block_k":32,"arch":"gfx950"})");
    b.add("small.manifest.json",
          R"({"schema":"s","kind":"gemm_fp16","kernel_name":"small","cache_key":"small",
              "block_m":64,"block_n":128,"block_k":32,"arch":"gfx950"})");
    b.add("bf16.manifest.json",
          R"({"schema":"s","kind":"gemm_bf16","kernel_name":"bf","cache_key":"bf",
              "block_m":128,"block_n":128,"block_k":32,"arch":"gfx950"})");
    ArtifactStore store;
    check(store.add_bundle(b.dir) == 3, "indexed 3 manifests");

    Dispatcher d(store);
    Problem p;
    p.op = "gemm";
    p.dtype = "fp16";
    p.arch = "gfx950";
    p.M = 256;
    p.N = 256;
    p.K = 256;  // divisible by both tiles
    auto r = d.rank(p);
    check(r.size() == 2, "only fp16 candidates support (bf16 gated out)");
    check(!r.empty() && r.front().cache_key == "big", "largest CTA tile ranked first");

    // dtype gate: a bf16 request never picks an fp16 kernel.
    Problem pb = p;
    pb.dtype = "bf16";
    auto rb = d.rank(pb);
    check(rb.size() == 1 && rb.front().cache_key == "bf", "bf16 selects bf16 kernel only");

    // shape gate: a K not divisible by block_k -> no candidate.
    Problem pk = p;
    pk.K = 250;  // 250 % 32 != 0
    check(d.rank(pk).empty(), "indivisible K -> no candidate");

    // arch-family gate: an RDNA problem must not pick a CDNA kernel.
    Problem pr = p;
    pr.arch = "gfx1100";
    check(d.rank(pr).empty(), "rdna problem rejects cdna kernels");
}

void test_dispatcher_norm() {
    std::printf("-- dispatcher norm ranking --\n");
    TmpBundle b("norm");
    b.add("a.manifest.json",
          R"({"schema":"s","kind":"norm","kernel_name":"a","cache_key":"a","norm_kind":"rmsnorm",
              "block_m":256,"vec":8,"arch":"gfx950"})");
    b.add("c.manifest.json",
          R"({"schema":"s","kind":"norm","kernel_name":"c","cache_key":"c","norm_kind":"rmsnorm",
              "block_m":256,"vec":4,"arch":"gfx950"})");
    b.add(
        "ln.manifest.json",
        R"({"schema":"s","kind":"norm","kernel_name":"ln","cache_key":"ln","norm_kind":"layernorm",
              "block_m":256,"vec":8,"arch":"gfx950"})");
    ArtifactStore store;
    store.add_bundle(b.dir);
    Dispatcher d(store);
    Problem p;
    p.op = "norm";
    p.kind = "rmsnorm";
    p.arch = "gfx950";
    p.rows = 1024;
    p.cols = 4096;  // divisible by 256*8 and 256*4
    auto r = d.rank(p);
    check(r.size() == 2, "only rmsnorm candidates (layernorm gated by norm_kind)");
    check(!r.empty() && r.front().cache_key == "a", "widest vec ranked first (256/v8)");

    // cols not divisible by block_size*vec for the v8 chunk -> only v4 fits.
    Problem p2 = p;
    p2.cols = 256 * 4;  // 1024: divisible by 256*4 but not 256*8 (2048)
    auto r2 = d.rank(p2);
    check(r2.size() == 1 && r2.front().cache_key == "c", "v8 dropped when cols indivisible");
}

void test_dispatcher_attention() {
    std::printf("-- dispatcher attention path selection --\n");
    TmpBundle b("attn");
    // One 2d and one 3d unified-attention manifest, both hd128 bs16.
    b.add("a2d.manifest.json",
          R"({"schema":"s","kind":"attention_unified","kernel_name":"a2d","cache_key":"a2d",
              "path":"2d","block_m":128,"block_n":16,"arch":"gfx950"})");
    b.add("a3d.manifest.json",
          R"({"schema":"s","kind":"attention_unified","kernel_name":"a3d","cache_key":"a3d",
              "path":"3d","block_m":128,"block_n":16,"arch":"gfx950"})");
    ArtifactStore store;
    store.add_bundle(b.dir);
    Dispatcher d(store);

    // Mirror UnifiedAttentionProblem.select_path: is_2d when sliding_window>0, or
    // seqlen_k<=512, or the 2d q-block count exceeds num_sms*4.
    auto mkP = [](long batch, long nhq, long nhk, long sq, long sk) {
        Problem p;
        p.op = "attention";
        p.dtype = "fp16";
        p.arch = "gfx950";
        p.batch = batch;
        p.nhead_q = nhq;
        p.nhead_k = nhk;
        p.seqlen_q = sq;
        p.seqlen_k = sk;
        p.hdim_q = 128;
        p.hdim_v = 128;
        p.total_q = batch * sq;
        p.num_seqs = batch;
        p.block_kv = 16;
        p.num_sms = 120;
        return p;
    };
    {
        auto r = d.select(mkP(1, 16, 2, 256, 256));  // sk<=512 -> 2d
        check(r.valid() && r.cache_key == "a2d", "short seqlen routes 2d");
    }
    {
        // Long-seqlen, few q-blocks (MQA-like, large block_q, small total_q) -> 3d.
        // nqpkv=1 -> block_q=16; total_q=64 -> ~5 q-blocks*nhead_k(2)=10 << 480.
        auto r = d.select(mkP(1, 2, 2, 64, 4096));
        check(r.valid() && r.cache_key == "a3d", "long-seqlen few-blocks routes 3d");
    }
    {
        // Long-seqlen but large batch -> many 2d blocks exceed target -> 2d.
        // This is the B>1 path that the buildProblem total_q/num_seqs fix feeds:
        // with the old num_seqs=1/total_q=seqlen_q defaults the block count would
        // be undercounted; here total_q=16*2048 keeps the routing correct.
        auto r = d.select(mkP(16, 16, 2, 2048, 2048));
        check(r.valid() && r.cache_key == "a2d", "long-seqlen large-batch routes 2d (B>1)");
    }
    {
        // Unsupported head dim -> no candidate.
        auto p = mkP(1, 16, 2, 256, 256);
        p.hdim_q = 100;
        p.hdim_v = 100;
        check(!d.select(p).valid(), "unsupported head_dim 100 -> no candidate");
    }
    {
        // GQA divisibility: nhead_q not a multiple of nhead_k -> rejected.
        auto p = mkP(1, 15, 2, 256, 256);  // 15 % 2 != 0
        check(!d.select(p).valid(), "non-divisible GQA heads -> no candidate");
    }
}

void test_artifact_store_skips_malformed() {
    std::printf("-- artifact store robustness --\n");
    TmpBundle b("malformed");
    b.add("good.manifest.json",
          R"({"schema":"s","kind":"gemm_fp16","kernel_name":"g","cache_key":"g","block_m":1})");
    b.add("bad.manifest.json", "{ this is not valid json ");
    b.add("notes.txt", "ignored");  // not a manifest
    ArtifactStore store;
    size_t n = store.add_bundle(b.dir);
    check(n == 1, "malformed manifest skipped, good one indexed");
    check(store.has("g"), "good manifest present");

    // Non-existent dir -> 0, no throw.
    ArtifactStore s2;
    check(s2.add_bundle("/tmp/ckdsl_does_not_exist_xyz") == 0, "missing dir -> 0");
}

}  // namespace

int main() {
    std::printf("=== ck_dsl runtime host-only unit tests ===\n");
    test_manifest_parse();
    test_pack_args();
    test_gemm_grid();
    test_dispatcher_gemm();
    test_dispatcher_norm();
    test_dispatcher_attention();
    test_artifact_store_skips_malformed();
    std::printf(g_fail == 0 ? "ALL PASS\n" : "FAILURES (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
