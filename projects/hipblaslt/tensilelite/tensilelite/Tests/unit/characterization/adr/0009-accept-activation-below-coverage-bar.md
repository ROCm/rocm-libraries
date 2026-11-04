# ADR 0009: Accept `Activation.py` below the coverage bar (asm codegen out of scope)

Status:  Accepted
Defect:  none — accepted coverage ceiling, not a bug
Commit:  9888ed6 (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/9888ed6f4779f7f03d63f02dbd29c56153457112 — landed on develop via PR #7989's squash merge, 74e4693

## Context
`Activation.py` is ~1037 statements. After pinning the pure config/type/
numeric layer, line coverage is 34.1% (up from a 16.8% starting point). The
remaining ~660 lines are rocisa **assembly codegen**: the `getXModule`
emitters (`getExp`/`getGelu`/`getSigmoid`/`getTanh`/`getDGelu`/`getSilu`/
`getSwish`/...), `CombineInstructions`/`FuseInstruction` and their iter
helpers, `replaceInst`/`removeOldInst`, `ConvertCoeffToHex`/`HolderToGpr`/
`createVgprIdxList`, and `ActivationInline`.

This codegen/asm/GPU layer is explicitly excluded from this characterization
effort's scope (see D0). In this environment most emitters also raise
immediately on their own — `NameError: 'SelectBit'`/`'VMaxF16'` (half paths
for sigmoid/exp/gelu/tanh/silu/swish/clamp) and `KeyError: 'TransOpWait'`
(single paths for gelu/sigmoid/exp/tanh/silu/swish/dgelu/geluscaling) —
because they are missing-symbol / ISA-map-dependent codegen paths that cannot
be exercised without the full `KernelWriter`/ISA context. Verifying emitted
assembly here would require building exactly the codegen harness the scope
excludes.

## Decision
Characterize only the pure layer plus the asm entry-points that run cleanly
with dummy vgprs. Do not attempt to drive the full asm codegen.

What is pinned (48 tests): `ActivationAvailable`,
`ActivationTypeRegister.typeAvailable`, the full `ActivationType` API
(construct/passActivation/getAdditionalArgNum/arg-strings/fitSupported/
getEnumIndex/getEnumStrList/state/repr/str/eq/lt/toEnum), `actCacheInfo.
isSame`, `getMagic`/`getMagicStr`/`HexToStr`/`addSpace`, and
`ActivationModule` defaults/setters/counters/vgprPrefix plus the working
`getModule` paths (abs/relu/none/clippedrelu/leakyrelu/clamp/drelu) and
`getAllGprUsage` for a single type.

## Consequences
`Activation.py` documents a coverage ceiling at 34.1% (1037 stmts, 683
missed). The asm-codegen emitters remain unpinned until the codegen surface
as a whole is brought into scope with a real `KernelWriter`/ISA context.

**Rejected alternatives:**
- Smoke-call every `getModule` type — most raise (see above); would only
  assert the raises, which pins environment breakage rather than behavior.
- Build a full rocisa register/ISA context and snapshot emitted asm — that
  is codegen characterization, out of scope and high-maintenance.
