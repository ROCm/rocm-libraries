/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "TestHelpers.hpp"

namespace stinkytofu {
namespace test {
// =================================================================
// 1. Iterated Dominance Frontier  (cascading PHIs)
//
//          Entry (v0 = x)
//            |
//            A
//           / \.
//          B   C (v0 = y)
//         / \   \.
//        D   E   F
//        |    \ /
//        |     G     <- PHI(v0) placed by DF(C)
//        |     |
//         \   /
//           H       <- PHI(v0) placed by iterated DF(G)
// =================================================================

struct IteratedDFCfg {
    BasicBlock *entry, *A, *B, *C, *D, *E, *F, *G, *H;

    StinkyInstruction* entryDef;
    StinkyInstruction* cDef;
    StinkyInstruction* gUse;
    StinkyInstruction* hUse;
};

inline IteratedDFCfg buildIteratedDFCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    IteratedDFCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");
    c.D = func.createBasicBlock("D");
    c.E = func.createBasicBlock("E");
    c.F = func.createBasicBlock("F");
    c.G = func.createBasicBlock("G");
    c.H = func.createBasicBlock("H");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.A, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.D);
    func.addEdge(c.B, c.E);
    func.addEdge(c.C, c.F);
    func.addEdge(c.E, c.G);
    func.addEdge(c.F, c.G);
    func.addEdge(c.D, c.H);
    func.addEdge(c.G, c.H);

    c.entryDef = createVAddInBlock(c.entry, arch, 0, 1, 2);
    c.cDef = createVAddInBlock(c.C, arch, 0, 3, 4);

    createVAddInBlock(c.A, arch, 40, 41, 42);
    createVAddInBlock(c.D, arch, 43, 44, 45);
    createVAddInBlock(c.E, arch, 46, 47, 48);
    createVAddInBlock(c.F, arch, 49, 50, 51);

    c.gUse = createVAddInBlock(c.G, arch, 52, 0, 53);
    c.hUse = createVAddInBlock(c.H, arch, 54, 0, 55);

    return c;
}

// =================================================================
// 2. Nested Loops  (PHIs at both loop headers)
//
//       Entry (v0 = x)
//         |
//         A  <--------+    (outer loop header)
//         |            |
//         B  <----+    |    (inner loop header)
//         |       |    |
//         C ------+    |    (v0 = y; inner back-edge C->B)
//         |            |
//         D -----------+    (uses v0; outer back-edge D->A)
// =================================================================

struct NestedLoopCfg {
    BasicBlock *entry, *A, *B, *C, *D;

    StinkyInstruction* entryDef;
    StinkyInstruction* cDef;
    StinkyInstruction* bUse;
    StinkyInstruction* dUse;
};

inline NestedLoopCfg buildNestedLoopCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    NestedLoopCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");
    c.D = func.createBasicBlock("D");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.A, c.B);
    func.addEdge(c.B, c.C);
    func.addEdge(c.C, c.B);
    func.addEdge(c.C, c.D);
    func.addEdge(c.D, c.A);

    c.entryDef = createVAddInBlock(c.entry, arch, 0, 1, 2);
    createVAddInBlock(c.A, arch, 40, 41, 42);
    c.bUse = createVAddInBlock(c.B, arch, 43, 0, 44);
    c.cDef = createVAddInBlock(c.C, arch, 0, 3, 4);
    c.dUse = createVAddInBlock(c.D, arch, 45, 0, 46);

    return c;
}

// =================================================================
// 3. Self-loop at a join point  (self-referential PHI)
//
//        Entry
//       /     \.
//      A       B
//    v0=x    v0=y
//       \     /
//         C <--+   (self-loop C->C; uses v0)
//         |    |
//         +----+
//         |
//         D  (uses v0)
// =================================================================

struct SelfLoopJoinCfg {
    BasicBlock *entry, *A, *B, *C, *D;

    StinkyInstruction* aDef;
    StinkyInstruction* bDef;
    StinkyInstruction* cUse;
    StinkyInstruction* dUse;
};

inline SelfLoopJoinCfg buildSelfLoopJoinCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    SelfLoopJoinCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");
    c.D = func.createBasicBlock("D");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.entry, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.C);
    func.addEdge(c.C, c.C);
    func.addEdge(c.C, c.D);

    c.aDef = createVAddInBlock(c.A, arch, 0, 1, 2);
    c.bDef = createVAddInBlock(c.B, arch, 0, 3, 4);
    c.cUse = createVAddInBlock(c.C, arch, 40, 0, 41);
    c.dUse = createVAddInBlock(c.D, arch, 42, 0, 43);

    return c;
}

// =================================================================
// 4. Irreducible CFG  (mutually recursive PHIs)
//
//        Entry
//       /     \.
//      A       B
//      |       |
//    v0=x    v0=y
//      |       |
//      C <---> D    (C->D and D->C, irreducible)
//       \     /
//         E  (uses v0)
// =================================================================

struct IrreducibleCfg {
    BasicBlock *entry, *A, *B, *C, *D, *E;

    StinkyInstruction* aDef;
    StinkyInstruction* bDef;
    StinkyInstruction* eUse;
};

inline IrreducibleCfg buildIrreducibleCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    IrreducibleCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");
    c.D = func.createBasicBlock("D");
    c.E = func.createBasicBlock("E");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.entry, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.D);
    func.addEdge(c.C, c.D);
    func.addEdge(c.D, c.C);
    func.addEdge(c.C, c.E);
    func.addEdge(c.D, c.E);

    c.aDef = createVAddInBlock(c.A, arch, 0, 1, 2);
    c.bDef = createVAddInBlock(c.B, arch, 0, 3, 4);
    c.eUse = createVAddInBlock(c.E, arch, 40, 0, 41);

    return c;
}

// =================================================================
// 5. Multiple registers needing PHIs at the same join
//
//        Entry
//       /     \.
//      A       B
//      |       |
//    v0=x    v0=y
//    v1=a    v1=b
//       \   /
//         C  (uses v0 and v1)
// =================================================================

struct MultiRegJoinCfg {
    BasicBlock *entry, *A, *B, *C;

    StinkyInstruction* aDefV0;
    StinkyInstruction* aDefV1;
    StinkyInstruction* bDefV0;
    StinkyInstruction* bDefV1;
    StinkyInstruction* cUseV0;
    StinkyInstruction* cUseV1;
};

inline MultiRegJoinCfg buildMultiRegJoinCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    MultiRegJoinCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.entry, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.C);

    c.aDefV0 = createVAddInBlock(c.A, arch, 0, 40, 41);
    c.aDefV1 = createVAddInBlock(c.A, arch, 1, 42, 43);
    c.bDefV0 = createVAddInBlock(c.B, arch, 0, 44, 45);
    c.bDefV1 = createVAddInBlock(c.B, arch, 1, 46, 47);
    c.cUseV0 = createVAddInBlock(c.C, arch, 50, 0, 51);
    c.cUseV1 = createVAddInBlock(c.C, arch, 52, 1, 53);

    return c;
}

// =================================================================
// 6. Chain of diamonds  (3 sequential merge points)
//
//        Entry (v0 = x)
//         / \.
//        A   B (v0 = y)
//         \ /
//          C       <- PHI(v0)
//         / \.
//        D   E (v0 = z)
//         \ /
//          F       <- PHI(v0)
//         / \.
//        G   H (v0 = w)
//         \ /
//          I       <- PHI(v0)   (uses v0)
// =================================================================

struct ChainOfDiamondsCfg {
    BasicBlock *entry, *A, *B, *C, *D, *E, *F, *G, *H, *I;

    StinkyInstruction* entryDef;
    StinkyInstruction* bDef;
    StinkyInstruction* eDef;
    StinkyInstruction* hDef;
    StinkyInstruction* iUse;
};

inline ChainOfDiamondsCfg buildChainOfDiamondsCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    ChainOfDiamondsCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");
    c.D = func.createBasicBlock("D");
    c.E = func.createBasicBlock("E");
    c.F = func.createBasicBlock("F");
    c.G = func.createBasicBlock("G");
    c.H = func.createBasicBlock("H");
    c.I = func.createBasicBlock("I");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.entry, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.C);
    func.addEdge(c.C, c.D);
    func.addEdge(c.C, c.E);
    func.addEdge(c.D, c.F);
    func.addEdge(c.E, c.F);
    func.addEdge(c.F, c.G);
    func.addEdge(c.F, c.H);
    func.addEdge(c.G, c.I);
    func.addEdge(c.H, c.I);

    c.entryDef = createVAddInBlock(c.entry, arch, 0, 1, 2);
    createVAddInBlock(c.A, arch, 40, 41, 42);
    c.bDef = createVAddInBlock(c.B, arch, 0, 3, 4);
    createVAddInBlock(c.D, arch, 43, 44, 45);
    c.eDef = createVAddInBlock(c.E, arch, 0, 5, 6);
    createVAddInBlock(c.G, arch, 46, 47, 48);
    c.hDef = createVAddInBlock(c.H, arch, 0, 7, 8);
    c.iUse = createVAddInBlock(c.I, arch, 49, 0, 50);

    return c;
}

// =================================================================
// 7. Dead register  (defined but never used — semi-pruned SSA)
//
//        Entry
//       /     \.
//      A       B
//      |       |
//    v0=x    v0=y    (v0 NEVER read anywhere)
//       \   /
//         C  (uses v5 only, NOT v0)
// =================================================================

struct DeadRegCfg {
    BasicBlock *entry, *A, *B, *C;

    StinkyInstruction* aDef;
    StinkyInstruction* bDef;
    StinkyInstruction* cUse;
};

inline DeadRegCfg buildDeadRegCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    DeadRegCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.entry, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.C);

    c.aDef = createVAddInBlock(c.A, arch, 0, 40, 41);
    c.bDef = createVAddInBlock(c.B, arch, 0, 42, 43);
    c.cUse = createVAddInBlock(c.C, arch, 50, 5, 6);

    return c;
}

// =================================================================
// 8. Re-definition within the same block  (lastDef shadows first)
//
//        Entry
//       /     \.
//      A       B
//      |       |
//    v0=x    v0=y
//    v0=w         (A defines v0 TWICE)
//       \   /
//         C  (uses v0)
// =================================================================

struct RedefSameBlockCfg {
    BasicBlock *entry, *A, *B, *C;

    StinkyInstruction* aDef1;
    StinkyInstruction* aDef2;
    StinkyInstruction* bDef;
    StinkyInstruction* cUse;
};

inline RedefSameBlockCfg buildRedefSameBlockCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    RedefSameBlockCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.entry, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.C);

    c.aDef1 = createVAddInBlock(c.A, arch, 0, 1, 2);
    c.aDef2 = createVAddInBlock(c.A, arch, 0, 3, 4);
    c.bDef = createVAddInBlock(c.B, arch, 0, 5, 6);
    c.cUse = createVAddInBlock(c.C, arch, 50, 0, 51);

    return c;
}

// =================================================================
// 9. Wide register with partial sub-register redefine (iterated DF)
//
// Entry defines v[0:3] (4 DWORDs, like buffer_load / ds_read_b128).
// C redefines only v0 (1 DWORD).
// G and H use individual DWORDs — only v0 needs PHIs.
//
//          Entry (v[0:3] = load)
//            |
//            A
//           / \.
//          B   C (v0 = vadd)
//         / \   \.
//        D   E   F
//        |    \ /
//        |     G     <- PHI(v0) placed by DF(C)
//        |     |          uses v0 (from PHI) and v2 (from Entry)
//         \   /
//           H       <- PHI(v0) placed by iterated DF(G)
//                        uses v0 (from PHI) and v1 (from Entry)
// =================================================================

struct WideRegPartialRedefCfg {
    BasicBlock *entry, *A, *B, *C, *D, *E, *F, *G, *H;

    StinkyInstruction* entryWideDef;  // ds_read_b128 v[0:3]
    StinkyInstruction* cPartialDef;   // v_add_f32 v0
    StinkyInstruction* gUse;          // v10 = v0 + v2
    StinkyInstruction* hUse;          // v11 = v0 + v1
};

inline WideRegPartialRedefCfg buildWideRegPartialRedefCfg(Function& func, GfxArchID arch) {
    setFunctionArch(func, arch);
    WideRegPartialRedefCfg c{};
    c.entry = func.createBasicBlock("entry");
    c.A = func.createBasicBlock("A");
    c.B = func.createBasicBlock("B");
    c.C = func.createBasicBlock("C");
    c.D = func.createBasicBlock("D");
    c.E = func.createBasicBlock("E");
    c.F = func.createBasicBlock("F");
    c.G = func.createBasicBlock("G");
    c.H = func.createBasicBlock("H");

    func.addEdge(c.entry, c.A);
    func.addEdge(c.A, c.B);
    func.addEdge(c.A, c.C);
    func.addEdge(c.B, c.D);
    func.addEdge(c.B, c.E);
    func.addEdge(c.C, c.F);
    func.addEdge(c.E, c.G);
    func.addEdge(c.F, c.G);
    func.addEdge(c.D, c.H);
    func.addEdge(c.G, c.H);

    // Entry: wide 4-DWORD load → v[0:3]
    c.entryWideDef = createDsReadB128InBlock(c.entry, arch, 0, 60);

    // C: partial redefine → only v0  (v30, v31 are undefined — not part of v[0:3])
    c.cPartialDef = createVAddInBlock(c.C, arch, 0, 30, 31);

    createVAddInBlock(c.A, arch, 40, 41, 42);
    createVAddInBlock(c.D, arch, 43, 44, 45);
    createVAddInBlock(c.E, arch, 46, 47, 48);
    createVAddInBlock(c.F, arch, 49, 50, 51);

    // G: uses v0 (conflicting defs → PHI) and v2 (only from Entry → no PHI)
    c.gUse = createVAddInBlock(c.G, arch, 10, 0, 2);

    // H: uses v0 (conflicting defs → PHI) and v1 (only from Entry → no PHI)
    c.hUse = createVAddInBlock(c.H, arch, 11, 0, 1);

    return c;
}

}  // namespace test
}  // namespace stinkytofu
