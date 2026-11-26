// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @file
/// Instruction-cache helpers for megakernels on gfx12 / gfx1250.
///
/// A megakernel inlines several heavily unrolled ops, so its code can outgrow what the hardware
/// brings in at launch. What gfx1250 does on its own:
///  - The instruction cache is 64 KB per WGP, with 128 B lines (CDNA 5 whitepaper; LLVM
///    FeatureInstCacheLineSize128).
///  - Before a wave starts, the hardware prefetches INST_PREF_SIZE lines from the entry point.
///    That field is in COMPUTE_PGM_RSRC3 and is 8 bits wide, so at most 255 lines (32,640 B).
///    LLVM sets it to min(ceil(code bytes / 128), 255). No attribute overrides it.
///  - Code past the first 32 KB, and any non-inlined callee, comes in on demand. A miss goes
///    through the L1 to the L2. That the I$ shares the data L2 is inferred, not documented.
///
/// What this header adds:
///  - code_range: the [begin, begin + bytes) of a kernel's code.
///    * Exact, from the linker: put the kernel in its own section with CK_TILE_ICACHE_SECTION,
///      then use CK_TILE_ICACHE_SECTION_RANGE. lld defines __start_<sec> and __stop_<sec> for any
///      section whose name is a C identifier, and the section lands in the code object's R+X
///      segment.
///    * Approximate: code_range_of(&kernel, bytes). The kernel's address comes from a PC-relative
///      relocation. The size has to be passed in; megakernel_code_size.py prints it.
///  - warm_code(range, policy): one wave from a chosen subset of workgroups touches every 128 B
///    line of the range, either with an L2 prefetch or with a load that carries a retain hint.
///    This plays the role of NVIDIA's "evict-last read of one word per code line".
///  - prefetch_inst / prefetch_inst_range / prefetch_inst_ahead: wrappers for s_prefetch_inst and
///    s_prefetch_inst_pc_rel. These fill this WGP's instruction cache directly, 1 to 32 lines
///    per instruction.
///
/// Hook (megakernel.hpp is not changed by this header). Use either:
///  (a) An entry point that warms the code and then runs the megakernel. Example 64 does this:
///      @code
///        __global__ __launch_bounds__(MK::kBlockSize) CK_TILE_ICACHE_SECTION(ck_mk_chain)
///        void mk_chain_entry(MK::Kargs k)
///        {
///            icache::warm_code(CK_TILE_ICACHE_SECTION_RANGE(ck_mk_chain), icache::warm_policy{});
///            MK{}(k);
///        }
///      @endcode
///  (b) Inside Megakernel::operator(), before InitBarriers(), call
///      warm_code(kargs.code, Policy{}). This needs a `code_range code` member in Kargs, filled by
///      the host, or the section range if Megakernel gets a section-name hook. Between stages, a
///      phase may also prefetch_inst_range() the next stage's code, given per-stage ranges.
///
/// Limits:
///  - Not run on hardware: the GPUs were down when this was written. Everything here compiles and
///    links for gfx1250, but nothing has been timed.
///  - The L2 warm assumes instruction fetch is served by the same L2 as data, which is inferred
///    rather than documented.
///  - MI455X has two L2 domains, one per FCD. A workgroup warms only its own domain, so the
///    warming workgroups must be spread so that both FCDs get one. Workgroups are dispatched
///    round-robin over XCDs, so a few consecutive workgroup ids usually reach both.
///  - s_prefetch_inst only runs when MODE.SCALAR_PREFETCH_EN (bit 24) is set; otherwise it is an
///    s_nop. That bit is documented for RDNA4, and assumed to be the same on gfx1250.
///    enable_scalar_prefetch() in ck_tile/core/arch/inst_prefetch.hpp sets it.
///  - A kernel in its own section is ordinary ELF (the loader maps segments, not sections).
///    Nobody has checked yet that ROCr and the HIP runtime launch it correctly.
namespace icache {

/// Instruction cache line size in bytes (gfx11+; gfx9/gfx10 use 64).
inline constexpr uint32_t kLineBytes = 128;
/// Lines covered by one s_prefetch_inst (length field is 5 bits: 1..32 lines).
inline constexpr uint32_t kMaxPrefetchLines = 32;

struct code_range
{
    uint64_t begin = 0; // byte address of the first instruction
    uint32_t bytes = 0; // 0 = empty

    CK_TILE_HOST_DEVICE constexpr bool empty() const { return bytes == 0; }
    CK_TILE_HOST_DEVICE constexpr uint32_t num_lines() const
    {
        return bytes == 0 ? 0
                          : static_cast<uint32_t>(((begin + bytes + kLineBytes - 1) / kLineBytes) -
                                                  begin / kLineBytes);
    }
};

/// The range starting at a device function's entry. In device code, `&kernel` is the address of
/// the code (a PC-relative relocation), not the host-side launch handle.
CK_TILE_DEVICE code_range code_range_of(const void* fn, uint32_t bytes)
{
    return code_range{reinterpret_cast<uint64_t>(fn), bytes};
}

/// The range between two addresses, e.g. two linker-defined section bounds.
CK_TILE_DEVICE code_range code_range_between(const void* begin, const void* end)
{
    const uint64_t b = reinterpret_cast<uint64_t>(begin);
    const uint64_t e = reinterpret_cast<uint64_t>(end);
    return code_range{b, e > b ? static_cast<uint32_t>(e - b) : 0u};
}

/// How warm_code touches the code.
enum struct warm_mode : uint8_t
{
    /// global_prefetch_b8 of one byte per line. Allocates in L2, doesn't wait, uses no VGPR
    /// result. Cheapest option. The hardware may drop a prefetch.
    l2_prefetch,
    /// global_load_b32 of one dword per line, then a wait. The lines are guaranteed to be in L2
    /// when it returns. Costs one VGPR per lane and the latency of the loads.
    l2_load,
    /// s_prefetch_inst of 32 lines at a time. Fills this WGP's instruction cache (and the L2 on
    /// the way). Only reaches the WGP that runs it, and only up to the 64 KB I$ capacity. Needs
    /// MODE.SCALAR_PREFETCH_EN.
    inst_prefetch,
};

/// Temporal hint (GFX12 CPol TH, bits [2:0] of the cache-policy immediate).
enum struct temporal_hint : uint8_t
{
    rt    = 0, // regular
    nt    = 1, // non-temporal (streaming)
    ht    = 2, // high-priority temporal: retained over RT, i.e. "evict last"
    lu    = 3, // last use (loads): drop after the hit
    nt_rt = 4, // NT in the near cache, RT in the far one
    rt_nt = 5,
    nt_ht = 6, // NT in the near cache, HT in the far one
};

/// Coherence scope (GFX12 CPol SCOPE, bits [4:3]).
enum struct cache_scope : uint8_t
{
    cu  = 0,
    se  = 1,
    dev = 2,
    sys = 3,
};

/// Cache-policy immediate for gfx12+ loads and prefetches (`aux` of the buffer intrinsics, `cpol`
/// of
/// __builtin_amdgcn_global_prefetch).
CK_TILE_HOST_DEVICE constexpr int32_t cpol(temporal_hint th, cache_scope scope)
{
    return static_cast<int32_t>(th) | (static_cast<int32_t>(scope) << 3);
}

struct warm_policy
{
    warm_mode mode = warm_mode::l2_prefetch;
    /// Workgroups [0, num_workgroups) warm. Each touches the whole range, so more than one only
    /// adds coverage of more L2 domains (or, with inst_prefetch, of more WGPs).
    index_t num_workgroups = 8;
    /// Wave (warp) within the workgroup that does the work.
    index_t wave = 0;
};

#if defined(__gfx12__) || defined(__gfx125__)
#define CK_TILE_ICACHE_HAS_GFX12_PREFETCH 1
#else
#define CK_TILE_ICACHE_HAS_GFX12_PREFETCH 0
#endif

/// s_prefetch_inst: bring @p lines (1..32) lines of code starting at @p addr (rounded down to
/// 128 B) into this WGP's instruction cache. Silently a no-op if MODE.SCALAR_PREFETCH_EN is 0.
/// @p addr must be wave-uniform (it goes in SGPRs).
template <uint32_t Lines = kMaxPrefetchLines>
CK_TILE_DEVICE void prefetch_inst(uint64_t addr)
{
    static_assert(Lines >= 1 && Lines <= kMaxPrefetchLines, "s_prefetch_inst covers 1..32 lines");
#if CK_TILE_ICACHE_HAS_GFX12_PREFETCH
    __builtin_amdgcn_s_prefetch_inst(reinterpret_cast<const void*>(addr), Lines - 1);
#else
    ignore = addr;
#endif
}

/// s_prefetch_inst_pc_rel: prefetch @p Lines lines starting @p OffsetBytes after the next
/// instruction. Use it at a phase boundary ahead of the code that follows. The offset is a 24-bit
/// signed immediate counted from the PC of the instruction after this one, so it reaches 8 MB.
/// Exact targets need the two-pass patcher in script/patch_prefetch_offset.py (INST_PREFETCH in
/// ck_tile/core/arch/inst_prefetch.hpp). This form only covers "the next N KB of straight-line
/// code".
template <int32_t OffsetBytes = 0, uint32_t Lines = kMaxPrefetchLines>
CK_TILE_DEVICE void prefetch_inst_ahead()
{
    static_assert(Lines >= 1 && Lines <= kMaxPrefetchLines, "s_prefetch_inst covers 1..32 lines");
    static_assert(OffsetBytes >= -(1 << 23) && OffsetBytes < (1 << 23), "24-bit offset");
#if CK_TILE_ICACHE_HAS_GFX12_PREFETCH
    asm volatile("s_prefetch_inst_pc_rel %0, null, %1" ::"n"(OffsetBytes), "n"(Lines - 1));
#endif
}

/// Prefetch a whole range into this WGP's instruction cache, 4 KB per instruction. Call it from
/// every wave or from one; the I$ is per WGP. Ranges over 64 KB evict their own beginning.
CK_TILE_DEVICE void prefetch_inst_range(const code_range& r)
{
    const uint64_t first = amd_wave_read_first_lane(r.begin) & ~uint64_t(kLineBytes - 1);
    const uint32_t lines = amd_wave_read_first_lane(r.num_lines());
    for(uint32_t l = 0; l < lines; l += kMaxPrefetchLines)
        prefetch_inst<kMaxPrefetchLines>(first + uint64_t(l) * kLineBytes);
}

/// Touch every line of @p r from the calling wave, one line per lane per step.
template <temporal_hint TH = temporal_hint::ht, cache_scope Scope = cache_scope::dev>
CK_TILE_DEVICE void warm_lines(const code_range& r, warm_mode mode)
{
    if(mode == warm_mode::inst_prefetch)
    {
        ::enable_scalar_prefetch(); // MODE.SCALAR_PREFETCH_EN, this wave only
        prefetch_inst_range(r);
        return;
    }
    const uint64_t first = r.begin & ~uint64_t(kLineBytes - 1);
    const uint32_t lines = r.num_lines();
    const uint32_t lane  = static_cast<uint32_t>(get_lane_id());
    const uint32_t step  = static_cast<uint32_t>(get_warp_size());
#if CK_TILE_ICACHE_HAS_GFX12_PREFETCH
    constexpr int32_t kCpol = cpol(TH, Scope);
    if(mode == warm_mode::l2_prefetch)
    {
        for(uint32_t l = lane; l < lines; l += step)
            __builtin_amdgcn_global_prefetch(
                reinterpret_cast<const void*>(first + uint64_t(l) * kLineBytes), kCpol);
        return;
    }
    // l2_load: a real load with the retain hint, then wait so the lines are in L2 on return.
    // The TH/SCOPE modifiers are spelled out because the clang load builtins take no cache policy
    // for flat/global pointers.
    static_assert(TH == temporal_hint::ht && Scope == cache_scope::dev,
                  "l2_load is spelled for TH_LOAD_HT scope:SCOPE_DEV only");
    uint32_t sink = 0;
    for(uint32_t l = lane; l < lines; l += step)
    {
        uint32_t v;
        const uint64_t a = first + uint64_t(l) * kLineBytes;
        asm volatile("global_load_b32 %0, %1, off th:TH_LOAD_HT scope:SCOPE_DEV"
                     : "=v"(v)
                     : "v"(a)
                     : "memory");
        sink ^= v;
    }
    asm volatile("s_wait_loadcnt 0" ::"v"(sink) : "memory");
#else
    // Pre-gfx12: plain loads (no per-instruction temporal hint in this header).
    ignore        = mode;
    uint32_t sink = 0;
    for(uint32_t l = lane; l < lines; l += step)
        sink ^= *reinterpret_cast<const volatile uint32_t*>(first + uint64_t(l) * kLineBytes);
    asm volatile("" ::"v"(sink));
#endif
}

/// The per-launch code-warm routine: wave @p p.wave of workgroups [0, p.num_workgroups) touches
/// every line of @p r. Every other wave returns at once (no barrier), so the warming wave runs
/// alongside the first work items. With inst_prefetch every workgroup should call it (each WGP
/// has its own I$), so set num_workgroups to the grid size.
template <temporal_hint TH = temporal_hint::ht, cache_scope Scope = cache_scope::dev>
CK_TILE_DEVICE void warm_code(const code_range& r, const warm_policy& p = {})
{
    if(r.empty() || get_block_id() >= p.num_workgroups ||
       get_warp_id(bool_constant<true>{}) != p.wave)
        return;
    warm_lines<TH, Scope>(r, p.mode);
}

} // namespace icache

/// Put a kernel (or device function) into its own ELF section, so that the linker defines
/// __start_<name> and __stop_<name> around it. @p name must be a C identifier. Device pass only:
/// the host-side launch stub stays in .text.
#if defined(__HIP_DEVICE_COMPILE__)
#define CK_TILE_ICACHE_SECTION(name) __attribute__((section(#name)))
#else
#define CK_TILE_ICACHE_SECTION(name)
#endif

/// Declares the linker-defined bounds of section @p name, at namespace scope, once per section.
#define CK_TILE_ICACHE_DECLARE_SECTION(name)                                                 \
    extern "C" __device__ __attribute__((visibility("hidden"))) const char __start_##name[]; \
    extern "C" __device__ __attribute__((visibility("hidden"))) const char __stop_##name[]

/// The code_range of section @p name (device code; needs CK_TILE_ICACHE_DECLARE_SECTION(name)).
#define CK_TILE_ICACHE_SECTION_RANGE(name) \
    ::ck_tile::icache::code_range_between(__start_##name, __stop_##name)

} // namespace ck_tile
