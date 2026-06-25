// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Stage-5 Kernel: the object the provider holds directly. Backed by either a
// prebuilt HSACO (path a) or LLVM IR that is comgr-compiled on demand (path b).
// Exposes load + signature-driven kernarg packing + launch, plus direct access
// to the HSACO bytes / entry / launch config / source IR for
// serialization, caching, graph capture, and timing.
#pragma once

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ck_dsl_runtime/comgr.hpp"
#include "ck_dsl_runtime/manifest.hpp"

namespace ck_dsl {

class HipError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

inline void hip_check(hipError_t e, const char* where) {
    if (e != hipSuccess) throw HipError(std::string(where) + ": " + hipGetErrorString(e));
}

class Kernel {
   public:
    // Path (a): prebuilt HSACO.
    static Kernel from_hsaco(std::vector<std::byte> hsaco, Manifest m) {
        Kernel k;
        k.manifest_ = std::move(m);
        k.hsaco_ = std::move(hsaco);
        return k;
    }
    // Path (b): LLVM IR, compiled lazily for `isa` on first ensure_compiled().
    static Kernel from_llvm_ir(std::string llvm_ir, Manifest m, std::string isa) {
        Kernel k;
        k.manifest_ = std::move(m);
        k.llvm_ir_ = std::move(llvm_ir);
        k.isa_ = std::move(isa);
        return k;
    }

    ~Kernel() {
        if (module_) hipModuleUnload(module_);
    }
    Kernel(Kernel&& o) noexcept {
        move_from(o);
    }
    Kernel& operator=(Kernel&& o) noexcept {
        if (this != &o) {
            if (module_) hipModuleUnload(module_);
            move_from(o);
        }
        return *this;
    }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    // AOT hook (ICompilablePlan::compile): compile .ll -> HSACO if needed, then
    // load the module + resolve the entry function. Idempotent.
    //
    // Lifetime: on a partial failure (module loads but the entry symbol does not
    // resolve) the freshly loaded module is unloaded before throwing, so a retry
    // never leaks a dangling hipModule_t. Without this, a second ensure_compiled()
    // would overwrite module_ and leak the first load.
    void ensure_compiled() {
        if (function_) return;
        if (hsaco_.empty()) {
            if (llvm_ir_.empty()) throw std::runtime_error("Kernel has neither HSACO nor LLVM IR");
            hsaco_ = Compiler::compile(llvm_ir_, isa_);
        }
        // module_ may be non-null from a prior attempt that loaded the module but
        // failed to resolve the function; unload it before reloading.
        if (module_) {
            (void)hipModuleUnload(module_);
            module_ = nullptr;
        }
        hip_check(hipModuleLoadData(&module_, hsaco_.data()), "hipModuleLoadData");
        hipError_t fe = hipModuleGetFunction(&function_, module_, manifest_.kernel_name.c_str());
        if (fe != hipSuccess) {
            // Don't strand the module if the entry symbol is missing/misnamed.
            (void)hipModuleUnload(module_);
            module_ = nullptr;
            function_ = nullptr;
            hip_check(fe, "hipModuleGetFunction");
        }
    }
    // Compile targeting a specific device's arch (path b), then load.
    void ensure_compiled(const hipDeviceProp_t& props) {
        if (!function_ && hsaco_.empty() && isa_.empty()) {
            std::string gcn = props.gcnArchName;  // e.g. "gfx950:sramecc+:xnack-"
            auto colon = gcn.find(':');
            if (colon != std::string::npos) gcn = gcn.substr(0, colon);
            isa_ = Compiler::isa_for(gcn);
        }
        ensure_compiled();
    }

    // Launch with explicit grid/block (the engine computes these). `ptr_args`
    // maps arg name -> device pointer; `scalar_args` maps arg name -> raw value
    // (zero-extended into its declared width).
    void launch(const std::unordered_map<std::string, void*>& ptr_args,
                const std::unordered_map<std::string, uint64_t>& scalar_args,
                std::array<unsigned, 3> grid, unsigned block, hipStream_t stream = nullptr,
                unsigned shared_bytes = 0) {
        ensure_compiled();
        std::vector<char> buf = pack_args(ptr_args, scalar_args);
        size_t arg_size = buf.size();
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, buf.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size, HIP_LAUNCH_PARAM_END};
        hip_check(hipModuleLaunchKernel(function_, grid[0], grid[1], grid[2], block, 1, 1,
                                        shared_bytes, stream, nullptr, config),
                  "hipModuleLaunchKernel");
    }

    // Pack the kernarg buffer per args_signature, honoring AMDGPU kernarg
    // alignment (each arg aligned to its size; segment aligned to max member).
    //
    // Defensive against a malformed manifest signature: an arg width is validated
    // to be one of the legal kernarg widths {1,2,4,8} before it is used both as
    // the field size and the alignment. align_up's `& ~(a-1)` form is only valid
    // for power-of-two `a`, and the scalar path memcpy's from a fixed 8-byte
    // uint64_t source -- a width > 8 (or a non-power-of-two width) from a
    // hand-edited / fuzzed manifest would otherwise corrupt alignment math and
    // read past the source. We throw a clear error instead.
    std::vector<char> pack_args(
        const std::unordered_map<std::string, void*>& ptr_args,
        const std::unordered_map<std::string, uint64_t>& scalar_args) const {
        auto arg_width = [](const ArgSpec& a) -> size_t {
            size_t al = static_cast<size_t>(a.width());
            // Pointers are always 8 (is_pointer() forces width()==8). Scalars must
            // pack into the 8-byte uint64_t value source, so cap at 8, and the
            // alignment must be a power of two for align_up to be correct.
            bool pow2 = al != 0 && (al & (al - 1)) == 0;
            if (!pow2 || al > 8)
                throw std::runtime_error("kernarg '" + a.name + "': illegal width " +
                                         std::to_string(al) + " (expected 1/2/4/8)");
            return al;
        };
        size_t off = 0, max_align = 1;
        // First pass: compute total size with alignment. width() derives the
        // byte width from the type when size_bytes is absent (attention).
        for (const auto& a : manifest_.args_signature) {
            size_t al = arg_width(a);
            max_align = std::max(max_align, al);
            off = align_up(off, al) + al;
        }
        size_t total = align_up(off, max_align);
        std::vector<char> buf(total, 0);
        off = 0;
        for (const auto& a : manifest_.args_signature) {
            size_t al = arg_width(a);
            off = align_up(off, al);
            // Belt-and-suspenders: the two-pass sizing above guarantees this, but
            // assert the write stays inside the buffer so a future signature/sizing
            // divergence can never become an out-of-bounds store.
            if (off + al > buf.size())
                throw std::runtime_error("kernarg pack overflow for '" + a.name + "'");
            if (a.is_pointer()) {
                auto it = ptr_args.find(a.name);
                if (it == ptr_args.end())
                    throw std::runtime_error("missing pointer arg '" + a.name + "'");
                void* p = it->second;
                std::memcpy(buf.data() + off, &p, sizeof(void*));
            } else {
                auto it = scalar_args.find(a.name);
                if (it == scalar_args.end())
                    throw std::runtime_error("missing scalar arg '" + a.name + "'");
                uint64_t v = it->second;
                std::memcpy(buf.data() + off, &v, al);  // little-endian low bytes
            }
            off += al;
        }
        return buf;
    }

    // GEMM grid helper: from block_m/n + grid_order (or grid_explicit).
    std::array<unsigned, 3> gemm_grid(long M, long N) const {
        if (manifest_.grid_explicit)
            return {static_cast<unsigned>((*manifest_.grid_explicit)[0]),
                    static_cast<unsigned>((*manifest_.grid_explicit)[1]),
                    static_cast<unsigned>((*manifest_.grid_explicit)[2])};
        unsigned m_tiles = ceil_div(M, manifest_.block_m);
        unsigned n_tiles = ceil_div(N, manifest_.block_n);
        // grid_order "MN": x<-M tiles, y<-N tiles; "NM": x<-N tiles, y<-M tiles.
        if (manifest_.grid_order == "NM") return {n_tiles, m_tiles, 1};
        return {m_tiles, n_tiles, 1};
    }

    // Direct access (serialize / cache / graph capture / timing).
    const std::vector<std::byte>& hsaco() const {
        return hsaco_;
    }
    const std::string& entry() const {
        return manifest_.kernel_name;
    }
    const Manifest& manifest() const {
        return manifest_;
    }
    std::string cache_key() const {
        return manifest_.id();
    }
    const std::string& llvm_ir() const {
        return llvm_ir_;
    }
    bool is_compiled() const {
        return function_ != nullptr;
    }

   private:
    Kernel() = default;
    void move_from(Kernel& o) {
        manifest_ = std::move(o.manifest_);
        hsaco_ = std::move(o.hsaco_);
        llvm_ir_ = std::move(o.llvm_ir_);
        isa_ = std::move(o.isa_);
        module_ = o.module_;
        function_ = o.function_;
        o.module_ = nullptr;
        o.function_ = nullptr;
    }
    static size_t align_up(size_t x, size_t a) {
        return (x + a - 1) & ~(a - 1);
    }
    static unsigned ceil_div(long x, int d) {
        return d > 0 ? static_cast<unsigned>((x + d - 1) / d) : 0u;
    }

    Manifest manifest_;
    std::vector<std::byte> hsaco_;
    std::string llvm_ir_;
    std::string isa_;
    hipModule_t module_ = nullptr;
    hipFunction_t function_ = nullptr;
};

}  // namespace ck_dsl
