// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone test for modules/flash/csrc/varlen.h -- the host-side
// translation from AOTriton's `VarlenType` enum into the `varlen_bits` word the
// GPU kernels decode. There is no C++ test harness anywhere in this project (no
// gtest, no CMake add_test target), so this is a plain main() rather than an
// invented framework. It is NOT part of the CMake build: v3src/CMakeLists.txt
// globs "modules/*/csrc/*.cc" recursively, and this file deliberately lives
// under modules/flash/tests/ (no "csrc" path component) so it is never swept
// into libaotriton_v2.so.
//
// It exercises the FOUR rows of the legacy varlen table plus the version-gated
// upgrade path. A dense-only test would be worthless here: it would pass three
// of the four wrong implementations, since dense is the one row where every
// input is zero and every plausible bug still produces 0x0000.
//
// It does NOT test varlen_to_wire() against the spec's hex on its own. That is
// not an omission: csrc/varlen.h carries a static_assert table pinning every
// legacy row through varlen_bits_of(), so the property most at risk of platform
// variation is checked by every compiler that reads the header rather than by
// whichever one happens to run this binary. The prior art on the flydsl branch needed a
// runtime `varlen_selfcheck()` only because it kept a SECOND representation (a
// bitfield decoder alongside a shift encoder) that clang would not let it
// check at compile time; under the single-representation design there is
// nothing left for such a call to compare.
//
// Build -- header-only, so no generated per-kernel header and no library link:
//
//   g++ -std=c++20 -D__HIP_PLATFORM_AMD__=1
//       -I . -I include -I <any build dir>/include -I <hip headers>
//       modules/flash/tests/test_varlen_translation.cc -o /tmp/varlen_test
//   /tmp/varlen_test
//
// The build dir is only needed for aotriton/config.h.

#include <aotriton/config.h>

#include "../csrc/varlen.h"
#include "../csrc/params_abi_compat.h"

#include <cstdio>

// util.h declares `extern template class TensorView<N>`, so its member
// definitions normally come from libaotriton_v2.so. Instantiate them here
// instead -- that is what keeps this a two-command test with no library build,
// and TensorView is a header-only class template, so nothing is lost.
template class AOTRITON_NS::TensorView<1>;
template class AOTRITON_NS::TensorView<2>;
template class AOTRITON_NS::TensorView<4>;

// Same reason, for the params constructors: they are defined (empty) in the
// shim translation units that end up in the library. Constructing a params
// object is new here -- the test grew one when the compatibility path became a
// struct translation rather than a scalar mapping.
namespace AOTRITON_NS::v3::flash {
attn_fwd_params::attn_fwd_params() {}
attn_bwd_params::attn_bwd_params() {}
}

namespace {

using AOTRITON_NS::v3::flash::VarlenBits;
using AOTRITON_NS::v3::flash::VarlenLength;
using AOTRITON_NS::v3::flash::VarlenPosition;
using AOTRITON_NS::v3::flash::VarlenStacked;
using AOTRITON_NS::v3::flash::VarlenMode;

// The named configurations this test needs, defined HERE rather than pulled in
// from a production header. Production carries no such constants -- the only
// place a configuration is spelled out is varlen_bits_of(), which is the thing
// under test -- and a test that reused them could not detect a change to them.
// Written out as fields so a wrong expectation looks wrong.
constexpr VarlenBits kDense = {};

constexpr VarlenBits kCompact = {
  .qmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::CUMULATIVE, .position = VarlenPosition::REUSE},
  .kmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::CUMULATIVE, .position = VarlenPosition::REUSE},
};

// torch.nn.attention.varlen's `seqused_k` on a packed KV cache: the K side
// takes its LENGTH from an individual array and its POSITION from a cumulative
// one. Two different tensors, which is why seqinfo_k0/k1 are named by role.
// No VarlenType can spell this, so it only ever arrives through the struct.
constexpr VarlenBits kSequsedKOnPacked = {
  .qmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::CUMULATIVE, .position = VarlenPosition::REUSE},
  .kmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::INDIVIDUAL, .position = VarlenPosition::ARRAY},
};

int g_failures = 0;

void
check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  } else {
    std::fprintf(stderr, "  ok: %s\n", what);
  }
}

void
check_eq(uint32_t got, uint32_t want, const char* what) {
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s -- got 0x%04x, want 0x%04x\n", what, got, want);
    ++g_failures;
  } else {
    std::fprintf(stderr, "  ok: %s == 0x%04x\n", what, got);
  }
}

using namespace AOTRITON_NS::v3::flash;
// varlen.h lives in a nested namespace so it stays out of the user API; a test
// of that header is exactly who should reach into it.
using namespace AOTRITON_NS::v3::flash::internal;

// A side, from plain numbers. The table-driven checks below are about the LOGIC,
// so they build one directly rather than through varlen_addressing_of(), which
// would need a TensorView per row.
constexpr VarlenAddressing
side_of(uint32_t wire, bool is_q, int32_t batch = 1, int32_t tokens = 0,
        int32_t info0_len = 0, int32_t info1_len = 0,
        int32_t caller_max = 0, int32_t tensor_max = 0) {
  const VarlenBits v = varlen_from_wire(wire);
  // `is_q` picks WHICH mode, and nothing more -- the object itself is
  // side-agnostic, because the two modes are symmetric.
  return VarlenAddressing(is_q ? v.qmode : v.kmode,
                          batch, tokens, info0_len, info1_len,
                          caller_max, tensor_max);
}


// One row of the legacy table, as the triple the pre-varlen_bits shim actually
// keyed on: the enum only supplied the padded SIGN, tensor presence supplied
// the rest.
void
check_row(const char* name, int8_t varlen_type, bool cu_seqlens_q, bool seq_strides_q,
          uint32_t want_wire) {
  const uint32_t got = varlen_to_wire(varlen_bits_of(varlen_type, cu_seqlens_q, seq_strides_q));
  std::fprintf(stderr, "-- %s: varlen_type=%d cu_seqlens_q=%d seq_strides_q=%d\n",
               name, int(varlen_type), int(cu_seqlens_q), int(seq_strides_q));
  check_eq(got, want_wire, name);
}

void
check_seq_count(const char* name, uint32_t wire, int32_t q0_len, int32_t q_batch,
                int32_t want, int32_t q_tokens = 0, int32_t max_seqlen_q = 0) {
  const int32_t got = side_of(wire, true, q_batch, q_tokens, q0_len, 0,
                              max_seqlen_q, max_seqlen_q).seq_count();
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s -- N got %d, want %d\n", name, got, want);
    ++g_failures;
  } else {
    std::fprintf(stderr, "  ok: %s N == %d\n", name, got);
  }
}

void
test_legacy_table() {
  // Row 1: dense. No arrays at all.
  check_row("dense", VarlenType::None, false, false, 0x0000u);

  // Row 2: compact varlen. cu_seqlens_q supplied, seq_strides_q is null --
  // which is the ONLY thing separating this row from strided.
  check_row("compact", VarlenType::CompactVarlen, true, false, 0x0B0Bu);

  // Row 3: padded varlen. BHSD with one sequence per batch slot; the enum is
  // load-bearing here and nowhere else, because it is the only input that
  // distinguishes this from compact.
  check_row("padded", VarlenType::PaddedVarlen, true, false, 0x0202u);

  // Row 4: strided varlen. Position read from a dedicated array.
  check_row("strided", VarlenType::StridedVarlen, true, true, 0x1313u);
}

void
test_legacy_quirks() {
  // These three are the reason varlen_bits_of() takes tensor presence and not
  // just the enum. Each reproduces what the OLD code did, which is what
  // backward compatibility means -- not what the enum's name suggests.

  // The enum was never set by callers that predate it; cu_seqlens_q alone made
  // Num_seqlens positive, and positive meant compact.
  check_row("None + cu_seqlens_q => compact",
            VarlenType::None, true, false, 0x0B0Bu);

  // The kernel picked strided by testing seq_strides_q for null, so the enum
  // saying "compact" did not override a supplied position array.
  check_row("CompactVarlen + seq_strides_q => strided",
            VarlenType::CompactVarlen, true, true, 0x1313u);

  // Negating a zero sequence count leaves it zero, so padded without the
  // length array was dense.
  check_row("PaddedVarlen without cu_seqlens_q => dense",
            VarlenType::PaddedVarlen, false, false, 0x0000u);
}

// A kVersion 3 object, as an old binary would have filled it, translated into
// the current one. This is the whole ABI story: the old layout is DESCRIBED,
// not reinterpreted field by field, so removing varlen_type from the current
// struct costs nothing.
void
test_version_translation() {
  auto fwd_wire = [](int8_t varlen_type, bool cu_q, bool ss_q) {
    versioned_attn_fwd_params<3> old{};
    old.varlen_type = varlen_type;
    // Presence is what the old shim keyed on, and a TensorView is truthy when
    // it has a base pointer, so a fake non-null one is enough here.
    int32_t storage[2] = {0, 0};
    if (cu_q) {
      old.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(storage), {2}, {1}, AOTRITON_NS::DType::kInt32);
    }
    if (ss_q) {
      old.seq_strides_q = T1(reinterpret_cast<intptr_t>(storage), {2}, {1}, AOTRITON_NS::DType::kInt32);
    }
    old.causal_type = CausalType::WindowedAttention;
    old.window_left = 17;
    old.window_right = 23;
    old.Max_seqlen_q = 128;
    const attn_fwd_params now =
        versioned_attn_fwd_params<3>::translate_to_current(old);
    // Everything else must survive the translation, not just varlen. A copy
    // that dropped a field would still pass a varlen-only assertion.
    check(now.causal_type == CausalType::WindowedAttention, "fwd v3: causal_type survives");
    check(now.window_left == 17 && now.window_right == 23, "fwd v3: window survives");
    check(now.Max_seqlen_q == 128, "fwd v3: Max_seqlen_q survives");
    return varlen_to_wire(now.varlen_bits);
  };

  check_eq(fwd_wire(VarlenType::None, false, false), 0x0000u, "fwd v3 -> dense");
  check_eq(fwd_wire(VarlenType::CompactVarlen, true, false), 0x0B0Bu, "fwd v3 -> compact");
  check_eq(fwd_wire(VarlenType::PaddedVarlen, true, false), 0x0202u, "fwd v3 -> padded");
  check_eq(fwd_wire(VarlenType::StridedVarlen, true, true), 0x1313u, "fwd v3 -> strided");
  // The quirks again, this time through the translation rather than the raw
  // mapping -- an old binary really does reach the library this way.
  check_eq(fwd_wire(VarlenType::None, true, false), 0x0B0Bu,
           "fwd v3 -> None + cu_seqlens_q is compact");

  // A translated object never carries an LSE layout: kVersion 3 had no way to
  // ask for one, so _TH can only arrive from a caller new enough to have the
  // struct. Anything else would be inventing a request the caller never made.
  versioned_attn_fwd_params<3> plain{};
  check(versioned_attn_fwd_params<3>::translate_to_current(plain).varlen_bits.lse_layout
            == VarlenLseLayout::HT,
        "fwd v3 translates to the default LSE layout");

  // Backward, same shape. DQ_ACC is checked because it is the one field that
  // sits AFTER the varlen fields in the v6 layout.
  versioned_attn_bwd_params<6> oldb{};
  oldb.varlen_type = VarlenType::CompactVarlen;
  int32_t storage[2] = {0, 0};
  oldb.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(storage), {2}, {1}, AOTRITON_NS::DType::kInt32);
  oldb.causal_type = CausalType::WindowedAttention;
  oldb.window_left = 5;
  const attn_bwd_params nowb =
      versioned_attn_bwd_params<6>::translate_to_current(oldb);
  check_eq(varlen_to_wire(nowb.varlen_bits), 0x0B0Bu, "bwd v6 -> compact");
  check(nowb.causal_type == CausalType::WindowedAttention, "bwd v6: causal_type survives");
  check(nowb.window_left == 5, "bwd v6: window survives");

  // Dispatch by the version actually passed, through the same entry point the
  // shims use. The rejection case is the one that matters: it is the signature
  // of the bug this replaced, where a range test was paired with a hard-coded
  // versioned_attn_fwd_params<3> and any other old version would have been
  // reinterpreted as a v3 rather than refused.
  {
    versioned_attn_fwd_params<3> v3{};
    v3.varlen_type = VarlenType::CompactVarlen;
    int32_t seq[2] = {0, 0};
    v3.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(seq), {2}, {1},
                         AOTRITON_NS::DType::kInt32);
    attn_fwd_params out;
    const auto& as_current = *reinterpret_cast<const attn_fwd_params*>(&v3);
    check(translate_fwd_params(as_current, 3, &out), "fwd: version 3 translates");
    check_eq(varlen_to_wire(out.varlen_bits), 0x0B0Bu, "fwd: version 3 -> compact");
    check(!translate_fwd_params(as_current, 2, &out),
          "fwd: version 2 is refused, not read as a v3");
    check(!translate_fwd_params(as_current, 0, &out), "fwd: version 0 is refused");

    versioned_attn_bwd_params<6> v6{};
    v6.varlen_type = VarlenType::PaddedVarlen;
    v6.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(seq), {2}, {1},
                         AOTRITON_NS::DType::kInt32);
    attn_bwd_params outb;
    const auto& b_as_current = *reinterpret_cast<const attn_bwd_params*>(&v6);
    check(translate_bwd_params(b_as_current, 6, &outb), "bwd: version 6 translates");
    check_eq(varlen_to_wire(outb.varlen_bits), 0x0202u, "bwd: version 6 -> padded");
    check(!translate_bwd_params(b_as_current, 5, &outb),
          "bwd: version 5 is refused, not read as a v6");
    // The current version is not a legacy row: the shims handle it by not
    // translating at all, so the dispatcher must not claim it either.
    check(!translate_fwd_params(as_current, attn_fwd_params::kVersion, &out),
          "fwd: current version is not in the legacy table");
  }

  // A current-version caller needs no translation at all: the struct IS the
  // answer, which is what removing the enum from the API bought.
  attn_fwd_params fresh;
  fresh.varlen_bits.qmode = {.stacked = VarlenStacked::THD,
                             .length = VarlenLength::CUMULATIVE,
                             .position = VarlenPosition::REUSE};
  fresh.varlen_bits.kmode = {.stacked = VarlenStacked::THD,
                             .length = VarlenLength::INDIVIDUAL,
                             .position = VarlenPosition::ARRAY};
  check_eq(varlen_to_wire(fresh.varlen_bits), 0x150Bu,
           "current caller: seqused_k on packed KV, which no VarlenType can spell");
  fresh.varlen_bits.lse_layout = VarlenLseLayout::TH;
  check_eq(varlen_to_wire(fresh.varlen_bits), 0x1'150Bu, "current caller: + _TH");
}

// PR #222 review: the shim keyed Max_seqlen off tensor presence, so a THD side
// with LENGTH == MAX -- which carries no length array -- kept the tensor's own
// extent. Under THD that extent is the total packed token count.
void
test_max_seqlen_source() {
  // Dense: no bits set, so the tensor extent is the per-sequence length and the
  // caller's Max_seqlen must NOT override it.
  check(!side_of(0x0000u, true).uses_caller_max_seqlen(), "dense Q trusts the tensor");
  check(!side_of(0x0000u, false).uses_caller_max_seqlen(), "dense K trusts the tensor");

  // The reported case: compact Q against a uniformly-stacked K (side 0x01 =
  // THD, MAX, IMPLIED). Q is already refused elsewhere for STACKED+MAX, so it
  // is the K side that slipped through.
  check(side_of(0x010Bu, false).uses_caller_max_seqlen(),
        "THD+MAX K takes Max_seqlen_k from the caller");
  check(side_of(0x010Bu, true).uses_caller_max_seqlen() == true,
        "compact Q takes Max_seqlen_q from the caller");

  // Every shipped mode already did the right thing, via presence; they must
  // keep doing it now that the predicate is the bits.
  for (uint32_t wire : {0x0B0Bu, 0x0202u, 0x1313u, 0x150Bu, 0x040Bu}) {
    check(side_of(wire, true).uses_caller_max_seqlen(), "non-dense Q uses caller max");
    check(side_of(wire, false).uses_caller_max_seqlen(), "non-dense K uses caller max");
  }
}

// PR #222 review: the shim validated only the Q-derived sequence count, so
// combinations that make the kernel tl.load from null reached the launch.
void
test_bits_validation() {
  // The shipped configurations, with exactly the arrays each mode needs.
  check(varlen_valid(varlen_from_wire(0x0000u),
                             side_of(0x0000u, true,  1, 0, 0, 0),
                             side_of(0x0000u, false, 1, 0, 0, 0)), "dense is valid");
  check(varlen_valid(varlen_from_wire(0x0B0Bu),
                             side_of(0x0B0Bu, true,  1, 0, 1, 0),
                             side_of(0x0B0Bu, false, 1, 0, 1, 0)), "compact is valid");
  check(varlen_valid(varlen_from_wire(0x0202u),
                             side_of(0x0202u, true,  1, 0, 1, 0),
                             side_of(0x0202u, false, 1, 0, 1, 0)), "padded is valid");
  check(varlen_valid(varlen_from_wire(0x1313u),
                             side_of(0x1313u, true,  1, 0, 1, 1),
                             side_of(0x1313u, false, 1, 0, 1, 1)), "strided is valid");
  check(varlen_valid(varlen_from_wire(0x150Bu),
                             side_of(0x150Bu, true,  1, 0, 1, 0),
                             side_of(0x150Bu, false, 1, 0, 1, 1)), "seqused on packed is valid");
  check(varlen_valid(varlen_from_wire(0x040Bu),
                             side_of(0x040Bu, true,  1, 0, 1, 0),
                             side_of(0x040Bu, false, 1, 0, 1, 0)), "seqused on BHSD is valid");

  // Missing arrays -- each of these would have been a load from null.
  check(!varlen_valid(varlen_from_wire(0x0B0Bu),
                             side_of(0x0B0Bu, true,  1, 0, 1, 0),
                             side_of(0x0B0Bu, false, 1, 0, 0, 0)), "compact without seqinfo_k0");
  check(!varlen_valid(varlen_from_wire(0x1313u),
                             side_of(0x1313u, true,  1, 0, 1, 0),
                             side_of(0x1313u, false, 1, 0, 1, 1)), "ARRAY Q without seqinfo_q1");
  check(!varlen_valid(varlen_from_wire(0x150Bu),
                             side_of(0x150Bu, true,  1, 0, 1, 0),
                             side_of(0x150Bu, false, 1, 0, 1, 0)), "seqused without seqinfo_k1");

  // A stray array the mode never reads: dense + seqinfo_q0 used to pass and
  // replace the dense max length with an unset Max_seqlen_q.
  check(!varlen_valid(varlen_from_wire(0x0000u),
                             side_of(0x0000u, true,  1, 0, 1, 0),
                             side_of(0x0000u, false, 1, 0, 0, 0)), "dense with a stray seqinfo_q0");

  // REUSE takes a POSITION out of the length array, which only holds positions
  // under CUMULATIVE. 0x09 = THD, MAX, REUSE; 0x0D = THD, INDIVIDUAL, REUSE.
  check(!varlen_valid(varlen_from_wire(0x0009u),
                             side_of(0x0009u, true,  1, 0, 1, 0),
                             side_of(0x0009u, false, 1, 0, 0, 0)), "REUSE with MAX length");
  check(!varlen_valid(varlen_from_wire(0x000Du),
                             side_of(0x000Du, true,  1, 0, 1, 0),
                             side_of(0x000Du, false, 1, 0, 0, 0)), "REUSE with INDIVIDUAL length");

  // Out-of-range fields and reserved bits.
  check(!varlen_valid(varlen_from_wire(0x0006u),
                             side_of(0x0006u, true,  1, 0, 1, 0),
                             side_of(0x0006u, false, 1, 0, 0, 0)), "LENGTH == 3 is not a value");
  check(!varlen_valid(varlen_from_wire(0x0018u),
                             side_of(0x0018u, true,  1, 0, 0, 1),
                             side_of(0x0018u, false, 1, 0, 0, 0)), "POSITION == 3 is not a value");
  check(!varlen_valid(varlen_from_wire(0x0020u),
                             side_of(0x0020u, true,  1, 0, 0, 0),
                             side_of(0x0020u, false, 1, 0, 0, 0)), "reserved bit 5 set");
  check(!varlen_valid(varlen_from_wire(0x00040000u),
                             side_of(0x00040000u, true,  1, 0, 0, 0),
                             side_of(0x00040000u, false, 1, 0, 0, 0)), "reserved bit 18 set");

  // LSE_LAYOUT is not addressing and must not be rejected.
  check(varlen_valid(varlen_from_wire(0x10000u),
                             side_of(0x10000u, true,  1, 0, 0, 0),
                             side_of(0x10000u, false, 1, 0, 0, 0)), "dense + TH is valid");
  check(varlen_valid(varlen_from_wire(0x1150Bu),
                             side_of(0x1150Bu, true,  1, 0, 1, 0),
                             side_of(0x1150Bu, false, 1, 0, 1, 1)), "seqused + TH is valid");

  // ... and only those two: the kernel maps every nonzero value to TH.
  check(!varlen_valid(varlen_from_wire(0x20000u),
                             side_of(0x20000u, true,  1, 0, 0, 0),
                             side_of(0x20000u, false, 1, 0, 0, 0)), "lse_layout == 2 is refused");
  check(!varlen_valid(varlen_from_wire(0x30000u),
                             side_of(0x30000u, true,  1, 0, 0, 0),
                             side_of(0x30000u, false, 1, 0, 0, 0)), "lse_layout == 3 is refused");
  check(!varlen_valid(varlen_from_wire(0x2150Bu),
                             side_of(0x2150Bu, true,  1, 0, 1, 0),
                             side_of(0x2150Bu, false, 1, 0, 1, 1)), "lse_layout == 2 on a valid mode");
}

// PR #222 review: presence was checked, extent was not, so an undersized array
// or a BHSD tensor with fewer batch slots than N was loaded past its end.
void
test_extent_validation() {
  const int32_t N = 7;
  // compact 0x0B0B: both sides CUMULATIVE, so both need N+1.
  check((side_of(0x0B0Bu, true,  1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x0B0Bu, false, 1, 0, N + 1, 0).extents_ok(N)),
        "compact with exactly N+1");
  check(!(side_of(0x0B0Bu, true,  1, 0, N, 0).extents_ok(N)
         && side_of(0x0B0Bu, false, 1, 0, N + 1, 0).extents_ok(N)),
        "compact with a short seqinfo_q0");
  check(!(side_of(0x0B0Bu, true,  1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x0B0Bu, false, 1, 0, N, 0).extents_ok(N)),
        "compact with a short seqinfo_k0");

  // LOWER bound, not equality: a bigger buffer than the mode needs is fine.
  check((side_of(0x0B0Bu, true,  1, 0, N + 64, 0).extents_ok(N)
         && side_of(0x0B0Bu, false, 1, 0, N + 64, 0).extents_ok(N)),
        "oversized arrays are accepted");

  // strided 0x1313: ARRAY on both. Stacked Q also reads [N], so it needs N+1
  // where K needs only N.
  check((side_of(0x1313u, true,  1, 0, N + 1, N + 1).extents_ok(N)
         && side_of(0x1313u, false, 1, 0, N + 1, N).extents_ok(N)),
        "strided at its minimum extents");
  // Symmetric now: N entries is enough for ADDRESSING on either side.
  check(side_of(0x1313u, true,  1, 0, N + 1, N).extents_ok(N)
        && side_of(0x1313u, false, 1, 0, N + 1, N).extents_ok(N),
        "strided addressing needs only N position entries");
  // The [N] slot is the logsumexp pitch's, and Q's alone.
  check(!side_of(0x1313u, true, 1, 0, N + 1, N).lse_pitch_ok(N),
        "strided Q position array missing its [N] slot");
  check(side_of(0x1313u, true, 1, 0, N + 1, N + 1).lse_pitch_ok(N),
        "strided Q with the [N] slot");
  check(side_of(0x0B0Bu, true, 1, 0, N + 1, 0).lse_pitch_ok(N),
        "compact Q reuses the length array's [N] slot");
  check(!side_of(0x0B0Bu, true, 1, 0, N, 0).lse_pitch_ok(N),
        "compact Q whose length array lacks [N]");
  check(side_of(0x0202u, true, N, 0, N + 1, 0).lse_pitch_ok(N),
        "batched Q needs no [N] slot at all");
  check(side_of(0x0001u, true, 1, 640, 0, 0, 128, 128).lse_pitch_ok(N),
        "uniform stacking derives the pitch without an array");

  // seqused 0x150B: K length INDIVIDUAL needs N, K position ARRAY needs N.
  check((side_of(0x150Bu, true,  1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x150Bu, false, 1, 0, N, N).extents_ok(N)),
        "seqused at its minimum extents");
  check(!(side_of(0x150Bu, true,  1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x150Bu, false, 1, 0, N - 1, N).extents_ok(N)),
        "seqused with a short seqused_k");

  // The reported case: mixed 0x000B derives N from a packed Q, so a dense K
  // with fewer batch slots than N is indexed with a z it never had.
  check((side_of(0x000Bu, true,  1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x000Bu, false, N, 0, 0, 0).extents_ok(N)),
        "mixed with a K batch of N");
  check(!(side_of(0x000Bu, true,  1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x000Bu, false, N - 1, 0, 0, 0).extents_ok(N)),
        "mixed with a K batch shorter than N");
  check((side_of(0x000Bu, true,  1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x000Bu, false, N + 3, 0, 0, 0).extents_ok(N)),
        "mixed with a K batch larger than N");

  // padded 0x0202: BHSD both sides, so both tensors need N slots.
  check((side_of(0x0202u, true,  N, 0, N + 1, 0).extents_ok(N)
         && side_of(0x0202u, false, N, 0, N + 1, 0).extents_ok(N)),
        "padded with N batch slots");
  check(!(side_of(0x0202u, true,  N - 1, 0, N + 1, 0).extents_ok(N)
         && side_of(0x0202u, false, N, 0, N + 1, 0).extents_ok(N)),
        "padded with a short Q batch");

  // The TOKEN axis. Reported on PR #222: only the batch extent was checked, so
  // a stacked K with LENGTH == MAX (side 0x01) against a short K.size(2) passed
  // and launched reads through N * max_seqlen_k.
  //
  // 0x010B is compact Q against uniformly-stacked K. Q supplies N; K's token
  // axis must hold N * max_seqlen_k rows.
  check(side_of(0x010Bu, false, 1, N * 32, 0, 0, 32, 32).extents_ok(N),
        "stacked MAX K with exactly N * max_seqlen tokens");
  check(!side_of(0x010Bu, false, 1, N * 32 - 1, 0, 0, 32, 32).extents_ok(N),
        "stacked MAX K one token short");
  check(side_of(0x010Bu, false, 1, N * 32 + 99, 0, 0, 32, 32).extents_ok(N),
        "a longer token axis is fine");

  // Q cannot trip it: seq_count() derives N by dividing that same extent.
  check(side_of(0x0001u, true, 1, N * 32, 0, 0, 32, 32).extents_ok(N),
        "stacked MAX Q is self-consistent by construction");

  // BHSD reads rows 0..seqlen of its own slice, so max_seqlen must fit.
  check(side_of(0x0202u, true, N, 128, N + 1, 0, 128, 128).extents_ok(N),
        "padded Q whose seq axis holds max_seqlen");
  check(!side_of(0x0202u, true, N, 64, N + 1, 0, 128, 128).extents_ok(N),
        "padded Q whose seq axis is shorter than Max_seqlen_q");

  // REUSE/ARRAY reach is decided by array CONTENTS, so the token axis is not
  // constrained here -- a documented precondition, not a check.
  check(side_of(0x0B0Bu, true, 1, 1, N + 1, 0).extents_ok(N),
        "packed positions are not bounded by the token axis");

  // Max_seqlen zero, where the mode reads the caller's. Zero passes every other
  // check -- it satisfies every lower bound, the bits are well-formed, and both
  // sequence counts agree -- and then launches cdiv(0, BLOCK_M) == 0
  // workgroups and returns success. Found by review on PR #222.
  check(!side_of(0x0202u, true, N, 128, N + 1, 0, /*caller_max*/0, 128).max_seqlen_ok(),
        "padded Q with Max_seqlen_q unset is refused");
  check(side_of(0x0202u, true, N, 128, N + 1, 0, 128, 128).max_seqlen_ok(),
        "padded Q with Max_seqlen_q set is accepted");
  check(!side_of(0x0B0Bu, false, 1, 0, N + 1, 0, 0, 0).max_seqlen_ok(),
        "compact K with Max_seqlen_k unset is refused");
  // A fully dense side reads the tensor, so an unset caller value is no error.
  check(side_of(0x0000u, true, N, 128, 0, 0, 0, 128).max_seqlen_ok(),
        "dense needs no caller Max_seqlen");
  // The trap it guards: everything else says yes.
  check(side_of(0x0202u, true, N, 128, N + 1, 0, 0, 128).extents_ok(N)
        && side_of(0x0202u, true, N, 128, N + 1, 0, 0, 128).lse_pitch_ok(N)
        && side_of(0x0202u, true, N, 128, N + 1, 0, 0, 128).seq_count() == N,
        "...which every other check accepts");

  // dense reads no array at all and is indexed by batch on both sides.
  check((side_of(0x0000u, true,  N, 0, 0, 0).extents_ok(N)
         && side_of(0x0000u, false, N, 0, 0, 0).extents_ok(N)), "dense with N batch slots");
  check(!(side_of(0x0000u, true,  N, 0, 0, 0).extents_ok(N)
         && side_of(0x0000u, false, N - 1, 0, 0, 0).extents_ok(N)), "dense with a short K batch");
}

void
test_seq_count() {
  // N off the Q side. cu_seqlens_q has N+1 entries under CUMULATIVE.
  check_seq_count("dense", 0x0000u, 0, 4, 4);
  check_seq_count("compact", 0x0B0Bu, 8, 1, 7);
  check_seq_count("strided", 0x1313u, 10, 1, 9);
  // Padded is BHSD on the Q side, so N rides in on Q's batch axis and the
  // array is not consulted -- the non-trivial row, and the one a "just read
  // the array" implementation gets wrong when the two disagree.
  check_seq_count("padded", 0x0202u, 6, 5, 5);
  // Q-side INDIVIDUAL: the array is (N,), not (N+1,).
  check_seq_count("q individual", 0x0004u | 0x0001u, 7, 1, 7);
  // STACKED + MAX is uniform stacking: every sequence is max_seqlen_q rows, so
  // N is the token axis divided by it. This used to report -1 and be refused,
  // though the configuration is perfectly well defined.
  check_seq_count("STACKED + MAX counts by division", 0x0001u, 0, 1, 5, 640, 128);
  check_seq_count("STACKED + MAX, one sequence", 0x0001u, 0, 1, 1, 128, 128);
  // Still undetermined where it genuinely is: a token axis that is not a whole
  // multiple is not a uniform stacking, and must not be rounded into one.
  check_seq_count("STACKED + MAX with a ragged token axis", 0x0001u, 0, 1, -1, 650, 128);
  check_seq_count("STACKED + MAX with max_seqlen 0", 0x0001u, 0, 1, -1, 640, 0);

  // And nothing to cross-check it against: Q's batch axis is 1 under THD, so an
  // independent count would disagree with every correct answer above one.
  check(side_of(0x0001u, true, 1, 0, 0).independent_seq_count() < 0,
        "STACKED + MAX has no independent count");

  // The independent derivation agrees on every shipped row, and is what
  // catches a padded caller whose cu_seqlens_q has the wrong length.
  check(side_of(0x0202u, true, 5, 0, 6).independent_seq_count() == 5, "independent: padded agrees");
  check(side_of(0x0202u, true, 5, 0, 4).independent_seq_count() == 3, "independent: padded disagrees");
  check(side_of(0x0B0Bu, true, 1, 0, 8).independent_seq_count() == 7, "independent: compact agrees");
  check(side_of(0x0000u, true, 4, 0, 0).independent_seq_count() == 4, "independent: dense agrees");
  check(side_of(0x0405u, true, 1, 0, 7).independent_seq_count() < 0,
        "independent: declines Q-side INDIVIDUAL");
}

// A stand-in for the generated OpAttnBwdParams: the fields
// varlen_bwd_seq_count() reads, and nothing else. Compiling against this is the
// point -- the real struct lives in a generated header, so without a stub the
// template body is only ever checked by a full AOT build.
struct FakeBwdParams {
  int32_t varlen_bits;
  const T1* seqinfo_q0;
  const T1* seqinfo_q1;
  const T4* Q;
  int32_t max_seqlen_q;   // read only under stacked MAX, where N is a division
};

void
test_bwd_grid_extent() {
  // None of these rows reads a position array; it is passed because the side
  // object is built from every operand, not only the ones N happens to need.
  const T1 no_array;

  // Padded: Q's batch axis IS N, and the length array is present but not
  // consulted -- the row where reading the array instead would look plausible
  // and be wrong whenever the two disagree.
  const T1 cu_seqlens{1, {6}, {1}, AOTRITON_NS::kInt32};
  const T4 q_padded{1, {5, 3, 64, 64}, {0, 0, 0, 1}, AOTRITON_NS::kFloat16};
  FakeBwdParams padded{0x0202, &cu_seqlens, &no_array, &q_padded};
  check(varlen_bwd_seq_count(&padded) == 5u, "bwd grid z extent, padded");

  // Compact: Q's batch axis is 1 under THD, so the array is the only source.
  const T4 q_packed{1, {1, 3, 512, 64}, {0, 0, 0, 1}, AOTRITON_NS::kFloat16};
  FakeBwdParams compact{0x0B0B, &cu_seqlens, &no_array, &q_packed};
  check(varlen_bwd_seq_count(&compact) == 5u, "bwd grid z extent, compact");

  // Dense: no array at all, and the null tensor's sizes are INDETERMINATE, so
  // this also checks that the BHSD path never reaches size(0).
  const T4 q_dense{1, {4, 3, 128, 64}, {0, 0, 0, 1}, AOTRITON_NS::kFloat16};
  FakeBwdParams dense{0x0000, &no_array, &no_array, &q_dense};
  check(varlen_bwd_seq_count(&dense) == 4u, "bwd grid z extent, dense");
}

void
test_persistent_mask() {
  // The addressing mask is what makes a dense call asking for _TH keep the
  // persistent path. attn_fwd.cc and fwd_kernel.py compute this predicate
  // SEPARATELY and must agree; disagreeing is a grid/indexing mismatch, so
  // wrong output rather than a slowdown.
  VarlenBits dense_th = kDense;
  dense_th.lse_layout = VarlenLseLayout::TH;
  const uint32_t wire = varlen_to_wire(dense_th);
  check(wire != 0u, "dense + TH is a non-zero word");
  check(varlen_addressing(wire) == 0u, "dense + TH has zero addressing bytes");
  check(varlen_addressing(varlen_to_wire(kCompact)) != 0u,
        "compact has non-zero addressing bytes");
}

} // anonymous namespace

int
main() {
  test_legacy_table();
  test_legacy_quirks();
  test_version_translation();
  test_max_seqlen_source();
  test_bits_validation();
  test_extent_validation();
  test_seq_count();
  test_bwd_grid_extent();
  test_persistent_mask();

  if (g_failures == 0) {
    std::fprintf(stderr, "ALL PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
