# Known Issues and Limitations

This document describes known limitations of StinkyTofu that users and developers should be aware of.

---

## Def-Use Chain

**Status:** Resolved (cross-block support added)

**Description:** The def-use chain (`buildUseDefChain(Function&)`, `inst->getOperandDefs()`, `inst->getUsers()`) now supports **cross-block** tracking. Pseudo PHI instructions are inserted at block boundaries when a register is defined in multiple predecessor blocks. Call `buildUseDefChain(Function&)` for full cross-block chains; `buildUseDefChain(BasicBlock&)` for block-local only.

**Note:** Pseudo PHI instructions are not emitted to assembly. AsmEmitter ignores them.

---

## Other Limitations

- **DCE / RedundantMovElim**: Block-local only -- does not track dead stores or redundant movs across basic block boundaries.
- **Peephole patterns**: Limited to 2-3 instruction sequences within a basic block. Fixed set of constraints (HasOneUse, IsConstant, SameValue, DifferentValue).
- **WaitCntPass stores**: No address analysis -- stores use conservative "wait all" at barriers.
