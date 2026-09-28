// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Thin, exception-safe C ABI bridge from ctypes to AOTriton 0.14.2b's v3
// flash-attention forward API (aotriton::v3::flash::attn_fwd). Forward-only:
// this harness phase is inference-only, so there is deliberately no dropout,
// no backward, and no philox/RNG plumbing beyond satisfying the ABI with
// null tensors that the compiled forward kernel never dereferences.
//
// Every exported symbol is a catch(...) trampoline: a C++ exception must
// never unwind across the ctypes boundary (undefined behavior). Failures are
// reported through an integer status plus aotriton_last_error(), never by
// throwing.
//
// Zero per-launch allocation/rebuild: aotriton_prepare() builds the
// attn_fwd_params once (TensorViews are pointer+shape+stride value types,
// no heap use) and, only for causal/windowed cases, allocates a 4-byte
// device scratch counter once. aotriton_launch() only resets that counter
// (hipMemsetAsync, not an allocation) and constructs a trivial on-stack
// attn_options before calling attn_fwd. This is required, not just
// stylistic: the parent runner replays these launches inside a captured
// HIP graph (see benchmark_sdpa.py's HipGraphs), and graph capture forbids
// host-side allocation inside the captured region.
//
// Primary sources (0.14.2b tag, see aotriton.py's module docstring for the
// full bibliography with URLs):
//   include/aotriton/flash.h   - attn_fwd_params, CausalType, WindowValue,
//                                VarlenBits/VarlenMode/VarlenStacked/...
//   include/aotriton/util.h    - TensorView<Rank>, LazyTensor, get_null_tensor
//   include/aotriton/dtypes.h  - DType enum values
//   include/aotriton/runtime.h - Stream
//   include/aotriton/cpp_tune.h- attn_options / KernelFineControl (tuning-only)
//   modules/flash/csrc/attn_fwd.cc   - host dispatcher; ground truth for what
//                                      the shipped functional space actually
//                                      wires up (USE_ALIBI, INT8*, BIAS_TYPE)
//   modules/flash/aot/attn_fwd.py    - compiled functional space (which
//                                      scalar options are ever compiled in)

#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>

namespace flash = AOTRITON_NS::v3::flash;
using AOTRITON_NS::DType;

namespace {

// ---------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------
// thread_local: the harness is expected to drive one GPU from one thread
// (see contract: "one exclusive GPU in bounded jobs"), but thread_local
// costs nothing here and removes any doubt if that ever changes.
thread_local std::string g_last_error;

void set_error(const std::string& msg) noexcept {
  try {
    g_last_error = msg;
  } catch (...) {
    // Out of memory formatting the message: leave whatever was there before
    // rather than throwing across the boundary we exist to protect.
  }
}

constexpr int32_t kStatusOk = 0;
constexpr int32_t kStatusInvalidArgument = 2;   // caller/programmer error; safe to fix and retry
constexpr int32_t kStatusRuntimeError = 3;      // HIP/AOTriton runtime failure; see out_hip_status

bool valid_dtype_code(int32_t code) {
  switch (code) {
    case DType::kFloat32:
    case DType::kFloat16:
    case DType::kBFloat16:
    case DType::kInt8:
    case DType::kInt16:
    case DType::kInt32:
    case DType::kInt64:
    case DType::kUInt8:
    case DType::kUInt16:
    case DType::kUInt32:
    case DType::kUInt64:
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------
// ABI-facing plain-old-data descriptors (mirrored exactly by aotriton.py's
// ctypes.Structure definitions -- field order, widths and natural alignment
// must match byte-for-byte; see the static_asserts below and
// aotriton_abi_sizeof_fwd_desc()/aotriton_abi_sizeof_tensor_desc() which let
// the Python side assert the two layouts agree at import time).
// ---------------------------------------------------------------------
struct AotritonTensorDesc {
  uint64_t ptr;          // device pointer; 0 == absent/null tensor
  int64_t shape[4];      // element units; only [0, rank) significant
  int64_t strides[4];    // element units; only [0, rank) significant
  int32_t rank;          // 0, 1, 2 or 4 -- the only ranks attn_fwd_params uses
  int32_t dtype;         // AOTRITON_NS::DType value
};
static_assert(sizeof(AotritonTensorDesc) == 80, "ABI layout drift: update aotriton.py in lockstep");
static_assert(alignof(AotritonTensorDesc) == 8, "ABI layout drift: update aotriton.py in lockstep");

struct AotritonFwdDesc {
  AotritonTensorDesc q;             // rank4, logical BHSD (contract: "AOT logical BHSD for
  AotritonTensorDesc k;             // rank4 (underlying storage may be BSHD)")
  AotritonTensorDesc v;             // rank4
  AotritonTensorDesc out;           // rank4
  AotritonTensorDesc bias;          // rank4; ptr==0 => attn_fwd_params.B left null (BIAS_TYPE=0)
  AotritonTensorDesc cu_seqlens_q;  // rank1 int32 (batch+1,); ptr==0 => dense
  AotritonTensorDesc cu_seqlens_k;  // rank1 int32 (batch+1,); ptr==0 => dense
  AotritonTensorDesc seqused_k;     // rank1 int32 (batch,);   ptr==0 => absent
  float scale;
  int32_t mask_kind;     // 0 none, 1 causal_topleft, 2 causal_bottomright (aotriton.py _MASK_CODES)
  int32_t window;        // 0 disables; otherwise width including the current key
  int32_t max_seqlen_q;
  int32_t max_seqlen_k;
  int32_t batch;         // cross-validation only; attn_fwd_params has no Batch field of its own --
                         // dense infers it from Q.size(0), ragged from seqinfo_q0.size(0)-1.
};
static_assert(sizeof(AotritonFwdDesc) == 8 * 80 + 24, "ABI layout drift: update aotriton.py in lockstep");

DType dtype_of(const AotritonTensorDesc& t) { return static_cast<DType>(t.dtype); }

flash::T4 to_t4(const AotritonTensorDesc& t) {
  std::array<uint64_t, 4> shape{}, strides{};
  for (int i = 0; i < 4; ++i) {
    shape[i] = static_cast<uint64_t>(t.shape[i]);
    strides[i] = static_cast<uint64_t>(t.strides[i]);
  }
  return flash::T4(static_cast<intptr_t>(t.ptr), shape, strides, dtype_of(t));
}

flash::T1 to_t1(const AotritonTensorDesc& t) {
  std::array<uint64_t, 1> shape{static_cast<uint64_t>(t.shape[0])};
  std::array<uint64_t, 1> strides{static_cast<uint64_t>(t.strides[0])};
  return flash::T1(static_cast<intptr_t>(t.ptr), shape, strides, dtype_of(t));
}

void require(bool cond, const char* msg) {
  if (!cond) throw std::invalid_argument(msg);
}

void require_tensor(const AotritonTensorDesc& t, const char* name, int32_t expect_rank) {
  if (t.ptr == 0) {
    throw std::invalid_argument(std::string(name) + " is required but ptr==0");
  }
  if (t.rank != expect_rank) {
    throw std::invalid_argument(std::string(name) + " rank mismatch: expected " +
                                 std::to_string(expect_rank) + ", got " + std::to_string(t.rank));
  }
  if (!valid_dtype_code(t.dtype)) {
    throw std::invalid_argument(std::string(name) + " has an unrecognized DType code");
  }
}

// The kernel keeps m - Window_left <= n <= m + Window_right.
// rocKE's causal window of width W keeps m + context - W + 1 <= n <= m + context.
// Thus the finite left diagonal is W - 1 - context, NOT a radius W.
// context is 0 for top-left and seqlen_k - seqlen_q for bottom-right.
// Source: 0.14.2b modules/flash/kernel/{masked_load_store,fwd_kernel_inner}.py.
// Infinite causal windows retain the upstream per-sequence alignment sentinels.
struct WindowEncoding {
  int8_t causal_type;
  int32_t window_left;
  int32_t window_right;
};

WindowEncoding translate_mask(int32_t mask_kind, int32_t window,
                              int32_t seqlen_q, int32_t seqlen_k) {
  if (mask_kind == 0) {
    require(window == 0, "finite window requires a causal mask");
    return {flash::CausalType::None, 0, 0};
  }
  const int32_t sentinel = (mask_kind == 1) ? flash::WindowValue::TopLeftAligned
                                           : flash::WindowValue::BottomRightAligned;
  if (window == 0) {
    return {flash::CausalType::WindowedAttention, sentinel, sentinel};
  }
  const int32_t context = mask_kind == 2 ? seqlen_k - seqlen_q : 0;
  return {flash::CausalType::WindowedAttention, window - 1 - context, sentinel};
}

// ---------------------------------------------------------------------
// Prepared call handle
// ---------------------------------------------------------------------
struct PreparedCall {
  flash::attn_fwd_params params;
  int32_t* counter_ptr = nullptr;  // owned iff has_counter; 4-byte device scratch, see below
  bool has_counter = false;
};

}  // namespace

extern "C" {

// ---- Version / build facts --------------------------------------------

void aotriton_get_version(int32_t* major, int32_t* minor, int32_t* patch) noexcept {
#if defined(AOTRITON_VERSION_MAJOR) && defined(AOTRITON_VERSION_MINOR) && defined(AOTRITON_VERSION_PATCH)
  if (major) *major = AOTRITON_VERSION_MAJOR;
  if (minor) *minor = AOTRITON_VERSION_MINOR;
  if (patch) *patch = AOTRITON_VERSION_PATCH;
#else
  if (major) *major = -1;
  if (minor) *minor = -1;
  if (patch) *patch = -1;
#endif
}

#define AOTRITON_BRIDGE_STRINGIFY_(x) #x
#define AOTRITON_BRIDGE_STRINGIFY(x) AOTRITON_BRIDGE_STRINGIFY_(x)

const char* aotriton_get_git_sha1() noexcept {
#if defined(AOTRITON_GIT_SHA1)
  return AOTRITON_BRIDGE_STRINGIFY(AOTRITON_GIT_SHA1);
#else
  return "";
#endif
}

const char* aotriton_get_name_suffix() noexcept {
#if defined(AOTRITON_ENABLE_SUFFIX) && AOTRITON_ENABLE_SUFFIX
  return AOTRITON_BRIDGE_STRINGIFY(AOTRITON_NAME_SUFFIX);
#else
  return "";
#endif
}

int32_t aotriton_fwd_params_version() noexcept { return flash::attn_fwd_params::kVersion; }

int32_t aotriton_abi_sizeof_tensor_desc() noexcept {
  return static_cast<int32_t>(sizeof(AotritonTensorDesc));
}

int32_t aotriton_abi_sizeof_fwd_desc() noexcept { return static_cast<int32_t>(sizeof(AotritonFwdDesc)); }

const char* aotriton_last_error() noexcept { return g_last_error.c_str(); }

// ---- Prepare / launch / destroy ---------------------------------------

int32_t aotriton_prepare(const AotritonFwdDesc* desc, void** out_handle) noexcept {
  try {
    set_error("");
    if (!desc || !out_handle) {
      set_error("aotriton_prepare: null desc/out_handle");
      return kStatusInvalidArgument;
    }
    *out_handle = nullptr;

    require_tensor(desc->q, "q", 4);
    require_tensor(desc->k, "k", 4);
    require_tensor(desc->v, "v", 4);
    require_tensor(desc->out, "out", 4);
    require(desc->q.shape[3] == desc->k.shape[3], "q/k head_dim mismatch");
    require(desc->k.shape[1] > 0 && desc->q.shape[1] % desc->k.shape[1] == 0,
            "num_head_q must be a multiple of num_head_k (MHA/GQA/MQA only)");
    require(desc->mask_kind >= 0 && desc->mask_kind <= 2, "mask_kind out of range");
    require(desc->window >= 0, "window must be >= 0");

    const bool ragged = desc->cu_seqlens_q.ptr != 0;
    require(ragged == (desc->cu_seqlens_k.ptr != 0),
            "cu_seqlens_q and cu_seqlens_k must both be present or both absent");
    require(!(desc->seqused_k.ptr != 0 && !ragged),
            "seqused_k requires cu_seqlens_q/cu_seqlens_k");
    require(!(ragged && desc->window > 0 && desc->mask_kind == 2),
            "finite bottom-right windows need per-sequence left diagonals; this bridge supports dense inputs only");

    if (ragged) {
      require_tensor(desc->cu_seqlens_q, "cu_seqlens_q", 1);
      require_tensor(desc->cu_seqlens_k, "cu_seqlens_k", 1);
      require(dtype_of(desc->cu_seqlens_q) == DType::kInt32 && dtype_of(desc->cu_seqlens_k) == DType::kInt32,
              "cu_seqlens_q/k must be int32 (AOTriton VarlenBits contract)");
      if (desc->batch > 0) {
        require(desc->cu_seqlens_q.shape[0] == desc->batch + 1, "cu_seqlens_q length must be batch+1");
      }
      if (desc->seqused_k.ptr != 0) {
        require_tensor(desc->seqused_k, "seqused_k", 1);
        require(dtype_of(desc->seqused_k) == DType::kInt32, "seqused_k must be int32");
      }
    } else if (desc->batch > 0) {
      require(desc->q.shape[0] == desc->batch, "dense q.shape[0] must equal batch");
    }

    auto handle = std::make_unique<PreparedCall>();
    flash::attn_fwd_params& p = handle->params;  // default-constructed: every T* member is
                                                  // already a null TensorView (base_==nullptr);
                                                  // varlen_bits is zero-initialized in-class.

    p.Q = to_t4(desc->q);
    p.K = to_t4(desc->k);
    p.V = to_t4(desc->v);
    p.Out = to_t4(desc->out);
    if (desc->bias.ptr != 0) {
      require_tensor(desc->bias, "bias", 4);
      require(dtype_of(desc->bias) == dtype_of(desc->q),
              "AOTriton 0.14.2b requires bias and Q to share their dtype");
      p.B = to_t4(desc->bias);
    }
    // p.A (ALiBi slopes) is intentionally never set. modules/flash/aot/attn_fwd.py
    // compiles `@ati.scalar('USE_ALIBI', options=[False])` -- False is the ONLY
    // functional ever built -- and modules/flash/csrc/attn_fwd.cc hardcodes
    // `.USE_ALIBI = false` in the host dispatcher regardless of whether A is
    // set. Wiring params.A here would silently do nothing (not "unsupported",
    // just ignored), which the contract forbids; ALiBi is rejected in
    // aotriton.py/benchmark_sdpa.py before this function is ever called.
    // params.A therefore stays a null TensorView<2>, matching its default.

    p.Sm_scale = desc->scale;
    p.L = flash::T2::get_null_tensor(DType::kFloat32);  // inference-only: no logsumexp consumer.
                                                         // Null L is an upstream-verified pattern:
                                                         // PyTorch's own mem_eff path passes exactly
                                                         // this (empty_t2 with base_ptr==0) when
                                                         // compute_logsumexp is false.
    p.Max_seqlen_q = desc->max_seqlen_q;
    p.Max_seqlen_k = desc->max_seqlen_k;
    p.dropout_p = 0.0f;  // ENABLE_DROPOUT compiles to false (attn_fwd.cc:
                         // `.ENABLE_DROPOUT = in.dropout_p > 0.0`); the compiled
                         // functional then never dereferences philox_*, so those
                         // fields stay their default-constructed null T0.

    const WindowEncoding enc = translate_mask(
        desc->mask_kind, desc->window, static_cast<int32_t>(desc->q.shape[2]),
        static_cast<int32_t>(desc->k.shape[2]));
    p.causal_type = enc.causal_type;
    p.window_left = enc.window_left;
    p.window_right = enc.window_right;

    if (ragged) {
      p.seqinfo_q0 = to_t1(desc->cu_seqlens_q);
      if (desc->seqused_k.ptr != 0) {
        // seqused_k pairing: Q stays packed/cumulative; K takes its length from
        // seqused_k (INDIVIDUAL) and its start offset from cu_seqlens_k (ARRAY).
        // This is the ONLY varlen_bits combination flash.h's own comment calls
        // out as exercised upstream ("Only the seqused_k pairing is exercised
        // through this API; the rest of the combination space is expected to
        // work but untested").
        p.seqinfo_k0 = to_t1(desc->seqused_k);
        p.seqinfo_k1 = to_t1(desc->cu_seqlens_k);
        flash::VarlenMode packed{};
        packed.stacked = flash::VarlenStacked::THD;
        packed.length = flash::VarlenLength::CUMULATIVE;
        packed.position = flash::VarlenPosition::REUSE;
        flash::VarlenMode k_indiv{};
        k_indiv.stacked = flash::VarlenStacked::THD;
        k_indiv.length = flash::VarlenLength::INDIVIDUAL;
        k_indiv.position = flash::VarlenPosition::ARRAY;
        p.varlen_bits.qmode = packed;
        p.varlen_bits.kmode = k_indiv;
        p.varlen_bits.lse_layout = flash::VarlenLseLayout::HT;
      } else {
        p.seqinfo_k0 = to_t1(desc->cu_seqlens_k);
        flash::VarlenMode packed{};
        packed.stacked = flash::VarlenStacked::THD;
        packed.length = flash::VarlenLength::CUMULATIVE;
        packed.position = flash::VarlenPosition::REUSE;
        p.varlen_bits.qmode = packed;
        p.varlen_bits.kmode = packed;
        p.varlen_bits.lse_layout = flash::VarlenLseLayout::HT;
      }
    }
    // dense: p.varlen_bits stays its in-class default `= {}` (BHSD/MAX/IMPLIED
    // on both sides, HT lse layout) -- Max_seqlen_q/k above are what the dense
    // kernel actually reads as each (uniform) sequence's length.

    // persistent_atomic_counter: only read by the compiled kernel when
    // CAUSAL_TYPE != 0 (attn_fwd.py: `@ati.derives('persistent_atomic_counter',
    // to=0, when=ati.eq('CAUSAL_TYPE', 0))`), where it backs the persistent-grid
    // scheduling scheme (attn_fwd.cc's grid_calculator / PERSISTENT_TYPE). It
    // must be a valid zeroed int32 device buffer for every windowed/causal
    // launch (PyTorch allocates+zeroes one fresh every forward call:
    // `atomic_counter = at::zeros({1}, ...)`). This bridge instead allocates it
    // ONCE here (the one piece of device memory this bridge owns, and only for
    // this internal kernel-scheduling scratch role -- never a user Q/K/V/Out/
    // bias tensor) and re-zeroes it with hipMemsetAsync at the start of every
    // launch, which is graph-capture-safe (a memset is not an allocation).
    if (enc.causal_type != flash::CausalType::None) {
      hipError_t alloc_err = hipMalloc(reinterpret_cast<void**>(&handle->counter_ptr), sizeof(int32_t));
      if (alloc_err != hipSuccess) {
        set_error(std::string("aotriton_prepare: hipMalloc(persistent_atomic_counter) failed: ") +
                  hipGetErrorString(alloc_err));
        return kStatusRuntimeError;
      }
      handle->has_counter = true;
      p.persistent_atomic_counter = flash::T0(reinterpret_cast<intptr_t>(handle->counter_ptr), DType::kInt32);
    }

    *out_handle = handle.release();
    return kStatusOk;
  } catch (const std::exception& e) {
    set_error(std::string("aotriton_prepare: ") + e.what());
    return kStatusInvalidArgument;
  } catch (...) {
    set_error("aotriton_prepare: unknown C++ exception");
    return kStatusRuntimeError;
  }
}

int32_t aotriton_launch(void* handle_ptr, void* stream, int32_t backend_index,
                         int32_t* out_hip_status) noexcept {
  try {
    set_error("");
    if (out_hip_status) *out_hip_status = static_cast<int32_t>(hipSuccess);
    if (!handle_ptr) {
      set_error("aotriton_launch: null handle");
      return kStatusInvalidArgument;
    }
    auto* handle = static_cast<PreparedCall*>(handle_ptr);
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    if (handle->has_counter) {
      hipError_t reset_err = hipMemsetAsync(handle->counter_ptr, 0, sizeof(int32_t), hip_stream);
      if (reset_err != hipSuccess) {
        if (out_hip_status) *out_hip_status = static_cast<int32_t>(reset_err);
        set_error(std::string("aotriton_launch: hipMemsetAsync(persistent_atomic_counter) failed: ") +
                  hipGetErrorString(reset_err));
        return kStatusRuntimeError;
      }
    }

    // Trivial on-stack struct (int32_t + bool in a release runtime build --
    // KernelFineControl only becomes a member under AOTRITON_BUILD_FOR_TUNING,
    // which a shipped runtime install does not define). No allocation here.
    flash::attn_options options;
    options.force_backend_index = backend_index;

    hipError_t err =
        flash::attn_fwd(handle->params, flash::attn_fwd_params::kVersion, AOTRITON_NS::Stream(hip_stream), &options);
    if (out_hip_status) *out_hip_status = static_cast<int32_t>(err);
    if (err != hipSuccess) {
      set_error(std::string("aotriton_launch: attn_fwd returned ") + hipGetErrorString(err));
      return kStatusRuntimeError;
    }
    return kStatusOk;
  } catch (const std::exception& e) {
    set_error(std::string("aotriton_launch: ") + e.what());
    if (out_hip_status) *out_hip_status = static_cast<int32_t>(hipErrorUnknown);
    return kStatusRuntimeError;
  } catch (...) {
    set_error("aotriton_launch: unknown C++ exception");
    if (out_hip_status) *out_hip_status = static_cast<int32_t>(hipErrorUnknown);
    return kStatusRuntimeError;
  }
}

void aotriton_destroy(void* handle_ptr) noexcept {
  try {
    if (!handle_ptr) return;
    auto* handle = static_cast<PreparedCall*>(handle_ptr);
    if (handle->has_counter && handle->counter_ptr) {
      hipFree(handle->counter_ptr);  // best-effort; nothing actionable if this fails during teardown
    }
    delete handle;
  } catch (...) {
    // Never throw across the ctypes boundary, even during cleanup.
  }
}

}  // extern "C"
