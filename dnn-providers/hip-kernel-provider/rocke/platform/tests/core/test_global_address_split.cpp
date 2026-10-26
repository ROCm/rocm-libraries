// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/test_global_address_split.cpp -- the C++ engine's global address
 * split (cpp/core/lower_llvm/index_facts.cpp), the mirror of
 * tests/core/test_global_address_split.py.
 *
 * On gfx1151 a gather off a uniform pointer whose index is a uniform term plus
 * the thread id lowers to a GEP for the uniform term and the per-lane GEP on
 * it; inside a loop that varies the uniform term but not the thread id, the
 * uniform base passes through readfirstlane first. Other targets keep the
 * single `getelementptr inbounds`, and so does a kernel whose loop keeps more
 * vector state live than the target's VGPRs.
 *
 * Exit 0 = all pass.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"

namespace
{

const char* const kGather = R"IR(ckdsl.ir v1
kernel @split {
  params {
    %S : ptr<f16,global> { readonly = b:true }
    %D : ptr<f16,global>
    %stride : i32
  }
  region @entry {
    %tid1 : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %bid2 : i32 = gpu.block_id (  ) { axis = s:"x" }
    %mul3 : i32 = arith.mul ( %bid2, %stride )
    %add4 : i32 = arith.add ( %mul3, %tid1 )
    %gl5 : f16 = memref.global_load_typed ( %S, %add4 ) { align = i:1, elem_type = s:"f16" }
    memref.global_store_typed ( %D, %tid1, %gl5 ) { align = i:1, elem_type = s:"f16" }
    cf.return (  )
  }
}
)IR";

const char* const kDecl = "declare ptr addrspace(1) @llvm.amdgcn.readfirstlane.p1(ptr addrspace(1))";

/* A k loop carrying `accs` per-lane vec<f32x8> sums and reading the per-lane
 * vec<f32x8> %seed from outside, around the gather S[k * stride + tid]: its
 * live-through state is 8 * accs + 8 + 1 (the thread id) words. */
std::string budget_kernel(int accs)
{
    std::string s = "ckdsl.ir v1\nkernel @split_budget {\n  params {\n"
                    "    %S : ptr<f32,global> { readonly = b:true }\n"
                    "    %D : ptr<f32,global>\n    %stride : i32\n    %n : i32\n  }\n"
                    "  region @entry {\n"
                    "    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n"
                    "    %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n"
                    "    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n"
                    "    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n"
                    "    %off : i32 = arith.mul ( %tid, %c8 )\n"
                    "    %seed : vec<f32x8> = memref.global_load_vN ( %S, %off ) "
                    "{ align = i:32, elem_type = s:\"f32\", vec = i:8 }\n    ";
    std::string results, inits, meta, yields;
    for(int i = 0; i < accs; ++i)
    {
        const std::string sep = i ? ", " : "";
        results += sep + "%r" + std::to_string(i) + " : vec<f32x8>";
        inits += ", %seed";
        meta += sep + "{ name = s:\"%a" + std::to_string(i) + "\", type = s:\"vec<f32x8>\" }";
        yields += sep + "%y" + std::to_string(i);
    }
    s += results + " = scf.for ( %c0, %n, %c1" + inits
         + " ) { iter_args = l:[ " + meta + " ], iv = s:\"%k\", iv_type = s:\"i32\", num_iter_args = i:"
         + std::to_string(accs) + ", unroll = b:false }\n";
    s += "      region @body {\n"
         "        %row : i32 = arith.mul ( %k, %stride )\n"
         "        %idx : i32 = arith.add ( %row, %tid )\n"
         "        %v : f32 = memref.global_load_typed ( %S, %idx ) { align = i:4, elem_type = s:\"f32\" }\n"
         "        %splat : vec<f32x8> = vector.splat ( %v ) { vec = i:8 }\n"
         "        %step : vec<f32x8> = vector.add ( %splat, %seed )\n";
    for(int i = 0; i < accs; ++i)
        s += "        %y" + std::to_string(i) + " : vec<f32x8> = vector.add ( %a" + std::to_string(i)
             + ", %step )\n";
    s += "        scf.yield ( " + yields + " ) { num = i:" + std::to_string(accs) + " }\n      }\n";
    s += "    %sum : f32 = vector.sum ( %r0 )\n"
         "    memref.global_store_typed ( %D, %tid, %sum ) { align = i:4, elem_type = s:\"f32\" }\n"
         "    cf.return (  )\n  }\n}\n";
    return s;
}

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    failures += ok ? 0 : 1;
}

/* Lower `text` for `arch` at llvm22; empty on failure. */
std::string lower(const std::string& text, const char* arch)
{
    rocke_ir_builder_t b = {};
    rocke_ir_builder_init(&b, "split_test");
    rocke_kernel_def_t* k = nullptr;
    std::string out;
    if(rocke_ir_parse(text.c_str(), &b, &k) != ROCKE_OK || !k)
    {
        std::fprintf(stderr, "parse failed: %s\n", rocke_ir_builder_error(&b));
    }
    else
    {
        char* ll = nullptr;
        char err[512] = {0};
        if(rocke_lower_kernel_to_llvm_ex(k, ROCKE_LLVM_FLAVOR_LLVM22, arch, &ll, err, sizeof(err)) == ROCKE_OK
           && ll)
            out = ll;
        else
            std::fprintf(stderr, "lower failed for %s: %s\n", arch, err);
        std::free(ll);
    }
    rocke_ir_builder_free(&b);
    return out;
}

size_t count(const std::string& hay, const std::string& needle)
{
    size_t n = 0;
    for(size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + needle.size()))
        ++n;
    return n;
}

/* The SSA name defined by the line `  %name = <rhs>\n` of `ll`, or "". */
std::string defined_by(const std::string& ll, const std::string& rhs)
{
    const size_t p = ll.find(" = " + rhs + "\n");
    if(p == std::string::npos)
        return "";
    const size_t start = ll.rfind('%', p);
    return start == std::string::npos ? "" : ll.substr(start, p - start);
}

} // namespace

int main()
{
    const std::string ll = lower(kGather, "gfx1151");
    check(!ll.empty(), "gather lowers for gfx1151");
    const std::string term = defined_by(ll, "getelementptr half, ptr addrspace(1) %S, i32 %mul3");
    check(term.rfind("%gbase.", 0) == 0, "the uniform term indexes the pointer first");
    check(!term.empty()
              && ll.find("= getelementptr half, ptr addrspace(1) " + term + ", i32 %tid1\n") != std::string::npos,
          "the per-lane term indexes the uniform base");
    check(ll.find("readfirstlane") == std::string::npos, "no loop, no readfirstlane");
    check(ll.find("getelementptr inbounds half, ptr addrspace(1) %D, i32 %tid1") != std::string::npos,
          "a lane-only index keeps the single GEP");

    /* In the k loop the row k * stride varies and the thread id does not. */
    const std::string looped = lower(budget_kernel(2), "gfx1151");
    const std::string row = defined_by(looped, "getelementptr float, ptr addrspace(1) %S, i32 %row");
    const std::string uniform = defined_by(
        looped, "call ptr addrspace(1) @llvm.amdgcn.readfirstlane.p1(ptr addrspace(1) " + row + ")");
    check(row.rfind("%gbase.", 0) == 0 && uniform.rfind("%gbase.", 0) == 0,
          "a loop-varying uniform base goes through readfirstlane");
    check(!uniform.empty()
              && looped.find("= getelementptr float, ptr addrspace(1) " + uniform + ", i32 %tid\n")
                     != std::string::npos,
          "the per-lane term indexes the readfirstlane result");
    check(count(looped, std::string(kDecl) + "\n") == 1, "the readfirstlane declaration appears once");

    const std::string gfx950 = lower(kGather, "gfx950");
    check(!gfx950.empty() && gfx950.find("readfirstlane.p1") == std::string::npos
              && gfx950.find("getelementptr inbounds half, ptr addrspace(1) %S, i32 %add4")
                     != std::string::npos,
          "gfx950 keeps the single GEP");

    /* gfx1151 has 256 VGPRs: 16 sums keep 137 words live, 31 keep 257. */
    const std::string under = lower(budget_kernel(16), "gfx1151");
    check(!under.empty() && under.find("@llvm.amdgcn.readfirstlane.p1(ptr addrspace(1) %gbase.")
                                != std::string::npos,
          "a loop within the register file splits");
    const std::string over = lower(budget_kernel(31), "gfx1151");
    check(!over.empty() && over.find("%gbase.") == std::string::npos
              && over.find("getelementptr inbounds float, ptr addrspace(1) %S, i32 %idx")
                     != std::string::npos,
          "a loop past the register file keeps the single GEP");

    std::printf(failures ? ">> GLOBAL ADDRESS SPLIT FAILURES\n" : ">> ALL GLOBAL ADDRESS SPLIT CHECKS PASSED\n");
    return failures ? 1 : 0;
}
