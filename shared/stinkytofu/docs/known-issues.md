# Known Issues and Limitations

This document describes known limitations of StinkyTofu that users and developers should be aware of.

---

## Use-Def Chain: Cross-Block Not Tracked Properly

**Status:** Known limitation

**Description:** The use-def chain (`buildUseDefChain`, `inst->sources`, `inst->users`) is built **per basic block**. Cross-block def-use is tracked but **not tracked properly** -- definitions and uses that span block boundaries may be missing or incorrect.

**Current advice:** Treat each block edge as a side-effect. When a value flows across a block boundary (e.g., defined in block A, used in block B), conservatively assume it may be used and avoid optimizations that rely on accurate cross-block use-def information.

**Impact:**
- Peephole optimizations (e.g., Add+FMA fusion) only see definitions within the same block
- `getDefMapBefore()`, `getUseCountForDef()`, and similar APIs return results scoped to the current block
- Cross-block def-use links may be missing or unreliable

**Affected components:**
- [Peephole Pattern System](design/peephole-pattern-system.md) -- patterns cannot match def-use pairs spanning blocks
- [Dead Code Elimination](design/dead-code-elimination.md) -- operates per-block only
- [Redundant Mov Elimination](design/redundant-mov-elimination.md) -- operates per-block only
- DelayAluInsertionPass, DAG Scheduler -- use block-local use-def chains

---

## Other Limitations

Individual design documents describe further limitations:
- [Dead Code Elimination -- Limitations](design/dead-code-elimination.md#limitations)
- [Redundant Mov Elimination -- Limitations](design/redundant-mov-elimination.md#limitations)
- [Peephole Pattern System -- Limitations](design/peephole-pattern-system.md#limitations)
- [StinkyConfigurableWaitCntPass -- Limitations](design/stinky-configurable-waitcnt-pass.md#limitations)
