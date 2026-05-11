# ISA gap-class inventory + generalization design (rocm-libraries-s5g1, re-scope)

Investigation memo. No code changes. Originally s5g1 was scoped as "implement
Option 4 (post-009 dispatch table) for `_classify_edge_coverage`'s four-branch
fragility" — the narrow follow-up identified by `QUAD_CYCLE_DISPATCH_AUDIT.md`
(o0ei). The user has now asked for a wider inquiry: read the actual CDNA4 and
RDNA3.5 ISA documents, inventory **what classes of timing rules exist**,
audit how many of them the validator currently models, and design a
generalization that captures **classes of (producer, consumer) shape**
parameterized by per-arch data — not "MfmaF32_16x16x4F16, MfmaF32_32x32x4F16,
MfmaF32_16x16x16F16, ..." as one row each.

> Status: investigation only. The bead `rocm-libraries-s5g1` remains
> `in_progress` until the user reads this memo and decides on the re-scope.
> Section 5 proposes splitting s5g1 into ≥3 sibling beads.

---

## 1. Summary

The validator today models **four (4) quad-cycle gap rules** for CDNA4
(gfx950): standard-MFMA-finish, 4x4-PackMFMA-finish, CVT→MFMA settle,
4x4-PackMFMA→CVT settle. Plus an MFMA-type-switch +1 stall and a per-instruction
issue-cost (default 1, SNop+wait_state). All from CDNA4 ISA §7.6 and §4.5
collectively. The CDNA4 ISA itself defines **at least 18 distinct hardware
hazard classes** (CDNA4 §4.5 Table 11, §7.6 Table 38, §7.3 packed-convert
NOP rule, §3.x VCC-alias, §4.4 counter-wait semantics — full enumeration in
§2). The validator covers **roughly 4 of those 18 classes directly** (the
MFMA-related ones from §7.6) and treats counter-wait coverage (§4.4)
positionally rather than cycle-exactly. The other ~10–12 classes are NOT
modeled at all — they are either out-of-scope-by-design (validator only sees
schedules whose surrounding kernel-emitter is responsible for SCC/VCC/SMEM
hazards) or genuine gaps that would catch real bugs if the validator were
extended.

**Per-MFMA-opcode duplication is essentially zero.** The validator does NOT
key on MFMA opcode names. It uses ONE discriminator (`"_4x4x" in str(inst)`,
i.e. "is this a 4x4 PackMFMA family?") to split MFMAs into two buckets and
applies one finish-cycle constant per bucket. This is dramatically more
class-based than the bead's framing assumed. The interesting per-opcode
explosion lives **elsewhere**: in the ISA itself (§7.6 Table 38 has 5+
producer-class rows × 6+ consumer-class columns × per-MFMA-cycle-count
sub-buckets) and in **rocisa's own** `getMFMAIssueLatency()` C++ helper
(`mfma.hpp:34-66`) which already has per-(arch, dtype, B) issue-latency
rules the validator does not consume.

**Re-scope of s5g1 (top recommendation):**

The original o0ei memo (Option 4 dispatch-table refactor) stands as a
~50-LoC clean-up of the dispatch site. It does NOT need to wait for the
larger generalization. **Keep s5g1 narrow as o0ei specified it.** File three
new sibling beads:

1. `rdna35-arch-profile` — register a `_DEFAULT_RDNA35_ARCH_PROFILE` and the
   ~15 RDNA3.5-specific ISA-side hazard variants (S_DELAY_ALU vocabulary,
   VOPD-pair restrictions, WMMA scheduling). The current `ArchProfile`
   dataclass cannot represent any of these without new fields.
2. `mfma-finish-cycles-from-rocisa` — replace the `"_4x4x" in str(...)`
   string-parse in `_mfma_finish_cycles_for` with consumption of rocisa's
   `getMFMAIssueLatency` (or a Python-side mirror keyed on `(dtype,
   variant.B, isaVersion)`). This is the only place where there's a hidden
   per-opcode discriminator at all.
3. `gap-rule-table` — generalize `ArchProfile` from "four named scalar
   constants" to a `GapRuleTable: dict[(ProducerShape, ConsumerShape),
   GapRule]` keyed on `InstructionShape` (a refinement of
   `InstructionCategory`). Captures the four current constants as four
   table entries; opens the door to RDNA3.5's ~14 additional rules and to
   the ~6-12 currently-unmodeled CDNA4 hazards from §4.5 Table 11.

The `ArchProfile` struct as written today is honest about its scope —
"per-arch quad-cycle and issue-cycle constants" — but its fields encode
**4 constants that happen to all be MFMA-shaped**. Generalization is not
defensive: it's required as soon as the validator's coverage extends past
the MFMA-only carve-out.

**One unexpected finding** (surface separately to user, §6): RDNA3.5 ships
**S_DELAY_ALU** with a built-in dependency taxonomy (VALU_DEP_N,
TRANS32_DEP_N, SALU_CYCLE_N, FMA_ACCUM_CYCLE) — *the hardware* tells the
software which gap class is being inserted. This is a strong signal the
right abstraction for the validator is "named gap class with required
N", not "raw cycle count". The CDNA4 §7.6 table is just a denormalized
form of the same thing.

---

## 2. Task 1 — ISA gap inventory

### 2.1 Methodology

Both ISA `.txt` sidecars were greppable. The hazard / wait-state / NOP-rule
content is concentrated in:

- **CDNA4** §4.5 (Manually Inserted Wait States, Tables 11–12, p. 20),
  §7.6 (Dependency Resolution: Required Independent Instructions, Table 38,
  pp. 66–69), §7.3 (Packed Convert NOP rule, p. 58 — short paragraph,
  not a table), §3.x (VCC-alias hazard, p. 16), §4.4 (counter-wait
  semantics, p. 19).
- **RDNA3.5** §7.6 (Dual Issue VALU/VOPD restrictions, p. 70), §7.7 (DPP
  one-cycle delay, p. 71), §7.9.1 (WMMA Scheduling, p. 77), §10.8 (VMEM
  data dependencies, p. 109), §16.5 (S_DELAY_ALU SOPP, pp. 251–252),
  §12.2.1.2 (Parameter Load Data Hazard, p. 138 — VINTERP/LDS_PARAM_LOAD).

### 2.2 Inventory — CDNA4 (gfx950)

Each row: short-name, ISA cite (section + Table + page; or txt-line range
when it's prose), one-paragraph rule, validator coverage status.

| # | Short name | ISA cite | Rule | Coverage |
|---|-----------|----------|------|----------|
| C-1 | MFMA→MFMA same-VDST chained | §7.6 Table 38, p. 67 (XDL→XDL "Source C exactly same with 1st vDst") | Two consecutive MFMAs of the same family / passes / VDST start may chain on accumulator with **0 wait states** (the hardware forwards VDST→SrcC). | **Modeled in part.** `cumulative_issue_cycles` simulates `mfma_free_at = current_issue + 1 + finish_cycles` regardless of whether the second MFMA chains the accumulator. The "0 wait state for same-vDst chaining" optimization is NOT exploited; the simulator is conservative (treats every MFMA as needing the producer's full finish window even when chained). |
| C-2 | MFMA→MFMA different family + same vDst | §7.6 Table 38 ("the two V_MFMA must be the same number passes and vDst and vSrc start from the same offset" caveat) | When consecutive MFMAs differ in family AND share vDst, hardware does NOT chain — required wait varies by producer family (XDL: 2/4/6/10/18; SGEMM: 2/4/8/16; F64-16x16x4: 17/19; F64-4x4x4: 4/6). | **Modeled (loose).** Validator's `mfma_type_switch_threshold_from_standard=5` and `from_4x4=3` add a +1 stall when the **gap is below the threshold**; it does NOT use the actual table values from §7.6. |
| C-3 | MFMA→MFMA disjoint vDst | §7.6 Table 38 ("Source C overlapped with 1st vDst" rows) | Distinct accumulator regions — required waits also vary by producer family (XDL: 4/6/10/18; SGEMM: 2/4/8/16; F64: 0). | **Not modeled distinctly.** Validator does not look at vDst overlap; it applies the producer's finish-cycles as the floor regardless. |
| C-4 | MFMA→VALU same vDst (RAW on accumulator) | §7.6 Table 38 ("VALU read/write VGPR (RAW + WAW)" row) | XDL→VALU on accumulator: 5/8/12/20; SGEMM→VALU: 4/6/10/18; F64-16x16x4→VALU: 19; F64-4x4x4→VALU: 6. Per-MFMA-class cycle count. | **Not modeled.** The validator's MFMA-as-producer branch (`_quad_cycle_gap_ok`) returns `finish_cycles` (3 or 1) — much smaller than the 5–20 range above. The schedule typically hides this because LDS/global ops between MFMAs are wait-counter-gated, but a tight MFMA→VALU edge with no intervening counter would slip past. |
| C-5 | MFMA→VMEM/LDS/FLAT/Export read same vDst | §7.6 Table 38 ("VM, LDS, FLAT and Export Read VGPR overlapped with 1st vDst" rows) | XDL→memory-read: 5/8/12/20; SGEMM→memory-read: 4/6/10/18; F64-16x16x4→memory-read: 18; F64-4x4x4→memory-read: 9. | **Not modeled.** Same rationale as C-4. The MFMA→GR/LR/VECTOR_STORE edges in the dataflow graph fall into the "MFMA-as-producer" bucket and use the bare finish-cycles value. |
| C-6 | XDL/SMFMA SrcC read → VALU write VGPR (WAR) | §7.6 Table 38 p. 69 (last table) | "Co-Execution Anti-Dependency" — MFMA reads SrcC at internal stage S11; VALU write to overlapping VGPR needs 1/3/7/15 wait states depending on MFMA pass count. | **Not modeled.** No WAR edges are formed in the dataflow graph today. |
| C-7 | CVT→MFMA settle (TF32 emul) | §7.6 implied; also encoded in `_QUAD_CYCLES_CVT_BEFORE_MFMA = 2` | CVTPack (`v_cvt_pk_bf16_f32`) producing operand for MFMA needs 2 quad-cycles for the converted bf16 to be visible to the MFMA's read. | **Modeled.** `_cvt_to_mfma_gap_ok` with constant 2 from `cvt_before_mfma_quad_cycles`. |
| C-8 | 4x4 PackMFMA → CVT1 settle (TF32 emul) | §7.6 implied; also `_QUAD_CYCLES_MFMA_4X4_BEFORE_CVT1 = 5` | A 4x4 PackMFMA writing accumulator that a downstream CVT1 reads needs 5 quad-cycles for the accumulator to settle. | **Modeled.** `_mfma_pack_to_cvt_gap_ok` with constant 5. |
| C-9 | Non-DLops VALU writes VGPR → V_MFMA*/V_SMFMA* read | §7.6 Table 38, p. 66 (top row) | 2 wait states required between any VALU write of a VGPR and an MFMA reading that VGPR (no internal 4 & 8 cycle forwarding path). | **Not modeled distinctly.** Falls into the ALU-immediate exemption (`_is_alu_producer` returns True for the producer; the dispatch returns []`). For non-Pack ALU writers feeding MFMAs the validator skips the gap check entirely. |
| C-10 | DLops (V_DOT*) → DLops same opcode SrcC | §7.6 Table 38, p. 66 (DLops rows) | 0 wait states for back-to-back same-opcode DLops on SrcC; 3 wait states for different-opcode RAW/WAW; 3 for SrcA/B even same-opcode. | **Not modeled.** No DLops edges are formed today; the validator treats `V_DOT*` as ordinary VALU. |
| C-11 | F8 MFMA cycle-count doubling (gfx950) | §7.1.2 Table 28 (the V_MFMA_F32_*_F8F6F4 rows: "Larger cycle count if either matrix A or B is F8") | Cycle count for F8/F6/F4 mixed-format MFMAs depends on whether either input matrix is F8 (16↔32 or 32↔64). The rocisa-side `getMFMAIssueLatency` already has this branch (`mfma.hpp:60-65`) but the validator's `_mfma_finish_cycles_for` does not. | **Not modeled.** Validator's standard finish = 3 quad-cycles regardless of dtype. The rocisa helper already encodes this; the validator just doesn't read it. |
| C-12 | 16x16 vs 32x32 vs 4x4x16 finish cycles | §7.1.2 Table 28 (Cycles column) | 4x4: 8 cycles; 16x16: 16/32; 32x32: 32/64. Translates roughly to 1/4/8 quad-cycles for finish (4 cycles per quad-cycle on CDNA). | **Modeled coarsely.** Two buckets: 4x4 (1 quad-cycle) and "everything else" (3 quad-cycles). Misses the 16x16/32x32 distinction entirely; both round up to 3. |
| C-13 | VALU writes SGPR/VCC → V_{READ,WRITE}LANE | §4.5 Table 11, p. 20 | 4 wait states between VALU SGPR/VCC writer and VxxxLANE using that SGPR as lane select. | **Not modeled.** Out of scope for the dataflow graph (no SGPR-as-lane-select edges). |
| C-14 | VALU writes VCC → V_DIV_FMAS | §4.5 Table 11 | 4 wait states. | **Not modeled.** Out of scope. |
| C-15 | VALU writes VGPR → VALU DPP reads | §4.5 Table 11 | 2 wait states. | **Not modeled.** No DPP in the kernels we model. |
| C-16 | VALU writes EXEC → VALU DPP | §4.5 Table 11 | 5 wait states. | **Not modeled.** No DPP. |
| C-17 | VCC alias hazard (v_add*_i/u, v_cmp, v_div_scale*) → VALU reading VCC as constant | §4.5 Table 11 | 1 wait state if alias-named SGPR reused as constant. | **Not modeled.** Out of scope (kernel emitter handles). |
| C-18 | VALU writes SGPR → VMEM reads SGPR (address/offset) | §4.5 Table 11 | 5 wait states. | **Not modeled.** Out of scope; kernel emitter handles SGPR address setup. |
| C-19 | FLAT_STORE_X3/X4 etc. → VGPR write of writedata | §4.5 Table 11 | 1 wait state (writedata producer); 2 wait states (VALU write of VGPR holding writedata). | **Not modeled.** No FLAT writes in CMS bodies (categorized FLAT, finalize() guard). |
| C-20 | VALU Trans op → non-trans VALU consumer | §4.5 Table 11 | 1 wait state (no internal Trans→VALU forwarding). | **Not modeled.** No Trans ops in the kernels we model (no V_RCP_F32 etc. in MFMA bodies). |
| C-21 | OPSEL/SDWA bit-position-changing op → consumer | §4.5 Table 11 | 1 wait state. | **Not modeled.** OPSEL/SDWA opcodes do appear (PVCvtBF16toFP32 uses OP_SEL); the validator does not gap-check this. |
| C-22 | V_CMPX writes EXEC → V_PERMLANE* read | §4.5 Table 11 | 4 wait states. | **Not modeled.** Out of scope. |
| C-23 | VALU writes vdst → V_PERMLANE* reads vdst | §4.5 Table 11 | 2 wait states. | **Not modeled.** Out of scope. |
| C-24 | V_CMPX → V_MFMA* (EXEC mask forwarding) | §7.6 Table 38, p. 68 | 4 wait states (no EXEC mask forwarding to XDL/DGEMM). | **Not modeled.** Out of scope (no V_CMPX in MFMA bodies). |
| C-25 | Packed Convert split-register hazard | §7.3, p. 58 (prose) | "CVT_*_F32 instructions do not support 4-cycle forwarding... must insert NOP or instruction writing some other destination VREG between the conversions writing the low/high half or bytes of the same destination register." | **Not modeled.** The MIDDLE_PACK pair-interleaving check (`validate_middle_pack_pair_interleaving`) enforces a related-but-different invariant: that the two halves of a pair are interleaved with non-pair instructions. Does not enforce the "4-cycle gap or different-VREG between pair halves" wait-state rule. |
| C-26 | LDS bank conflict latency | §11.3.1, p. 96 | LDS read/write completes in 2–64 cycles depending on bank conflicts. | **Not modeled.** Wait-counter (`dscnt`) covers correctness at the producer-consumer level; bank-conflict latency is a performance concern the validator does not assess. |
| C-27 | Counter-wait semantics (vmcnt/lgkmcnt/vscnt + ordering) | §4.4, p. 19; §10.8, p. 109 (RDNA3.5 mirror) | Counters increment on issue, decrement on completion. Same-type ops complete in order; different-type ops complete out of order. SMEM reads can return out-of-order — only `S_WAITCNT 0` is legitimate after them. | **Modeled (partial).** Validator simulates per-counter FIFO queues for `dscnt`/`vlcnt` and verifies a covering `S_WAITCNT` exists between producer and consumer. SMEM out-of-order rule and the FLAT-vmcnt+lgkmcnt-simultaneous-decrement quirk are NOT modeled — they are excluded by the SMEM/FLAT finalize() guards in `InstructionCategory.py`. |
| C-28 | SCC RAW (carry chain) | §3.x (SCC is single-bit; SOPC ops set it) | An SCC writer must reach its SCC reader before another SCC writer clobbers it. | **Modeled.** Per-byte latest-writer in `build_dataflow_graph` forms SCC edges; the cross-body resolver clears them at body boundaries (rocm-libraries-theq). Treated as a structural data-flow edge, not a timing gap. |

**CDNA4 total: 28 distinct gap classes.** Modeled directly: 5 (C-2 loose,
C-7, C-8, C-12 coarse, C-27 partial, C-28). Modeled coarsely or by
related-but-not-equivalent rule: 4 (C-1, C-2, C-3, C-25). Not modeled but
in-scope by-design (would catch real bugs in current kernels): 6 (C-4, C-5,
C-6, C-9, C-11, C-12 finer-grained). Out-of-scope by-design (kernel emitter's
responsibility, not the schedule's): 13 (C-10, C-13, C-14, C-15, C-16, C-17,
C-18, C-19, C-20, C-21, C-22, C-23, C-24, C-26).

The rough split: **5 covered, 4 partial, 19 uncovered (6 of which would
catch real bugs)**.

### 2.3 Inventory — RDNA3.5 (gfx1151)

The whole RDNA3.5 ISA is **uncovered** because no `ArchProfile` is registered
for it (only CDNA4 today). Below is what would need to be modeled if/when
RDNA3.5 support is added. The shapes of the rules differ enough from CDNA4
that the current `ArchProfile` dataclass cannot represent any of them
without new fields.

| # | Short name | ISA cite | Rule | CDNA4 analog? |
|---|-----------|----------|------|---------------|
| R-1 | WMMA→WMMA chained vDst | §7.9.1, p. 77 | 1 V_NOP or independent VALU between two WMMAs if first's matrix-D overlaps second's A or B (correctness — not just performance). | C-1 (MFMA chained); but RDNA3.5 phrases it as "1 V_NOP" not "N quad-cycles". |
| R-2 | WMMA → WMMA same VGPR (D as C) | §7.9.1, p. 77 | Stall (perf-only) if not the same WMMA type or no IMOD on SRC2. | No direct CDNA4 analog. |
| R-3 | WMMA → VALU read of WMMA's matrix-D | §7.9.1, p. 77 | "Hardware may stall VALU instruction" (perf-only). | C-4 (MFMA→VALU). |
| R-4 | VOPD pair restriction: Src VGPR bank | §7.6, p. 70 | SRCX0 and SRCY0 must use different VGPR banks (banks indexed by `SRC[1:0]`); VSRCX1 and VSRCY1 must use different banks. **Hard rule — instruction does not function if violated.** | None — no CDNA4 dual-issue. |
| R-5 | VOPD pair restriction: Dst VGPR parity | §7.6, p. 70 | One Dst VGPR must be even, the other odd. **Hard rule.** | None. |
| R-6 | VOPD pair restriction: SRC2 even/odd | §7.6, p. 70 | If both ops use SRC2 (FMAMK_F32, DOT2ACC_F32_F16, etc.), one SRC2 must be even and the other odd. **Hard rule.** | None. |
| R-7 | VOPD pair restriction: Independence | §7.6, p. 70 | The two operations must be independent (no RAW/WAW between them — VOPD reads the *old* value if both touch same VGPR). | None. |
| R-8 | VOPD wave32-only | §7.6, p. 70 | Wave64 skips VOPD. **Hard rule.** | N/A — CDNA is wave64. |
| R-9 | DPP one-cycle delay | §7.7, p. 71 | "DPP instructions incur an extra cycle of delay to execute." | C-15/C-16 cover the producer side; CDNA4 does not call out an own-cycle delay. |
| R-10 | S_DELAY_ALU INSTID_VALU_DEP_N | §16.5, p. 251 | Insert N cycles' worth of delay for a previous VALU dependency, where N ∈ {1,2,3,4}. The instruction itself is the gap; the **encoding names the gap class**. | None; CDNA4 uses raw S_NOP. The fact that RDNA3.5 has *named* dependency classes is the strong signal that "gap-class enum" is the right validator abstraction. |
| R-11 | S_DELAY_ALU INSTID_TRANS32_DEP_N | §16.5, p. 252 | Same encoding for prior TRANS32 dependency, N ∈ {1,2,3}. | C-20 (Trans→non-trans). |
| R-12 | S_DELAY_ALU INSTID_SALU_CYCLE_N | §16.5, p. 252 | Same encoding for prior SALU dependency, N ∈ {1,2,3}. | None — CDNA4 doesn't enumerate SALU-dependency cycles separately. |
| R-13 | S_DELAY_ALU INSTID_FMA_ACCUM_CYCLE_1 | §16.5, p. 252 | "Single cycle penalty for FMA accumulation (reserved)." | C-1 (analog: MFMA chained). |
| R-14 | LDS_PARAM_LOAD WAW hazard with VALU read | §12.2.1.2, p. 138 | VALU read of VGPR followed by LDS_PARAM_LOAD overwrite of same VGPR — must not race. EXPcnt-based wait. | None — CDNA4 has no LDS_PARAM_LOAD shape. |
| R-15 | VINTERP "wait_EXPcnt" field | §12.2.1.2, p. 138 | VINTERP has an embedded wait_EXPcnt to avoid the R-14 hazard. | None. |
| R-16 | VMEM data-dep stall | §10.8, p. 109 | "Any ALU instruction that attempts to write [VMEM in-flight read VGPR] before it has been sent to the texture unit is stalled." Hardware-stall; correctness handled by VMCNT/VSCNT. | C-27 (counter-wait). |
| R-17 | Counter-wait: includes VSCNT | §10.8 + §16.5 (S_WAITCNT_VSCNT) | RDNA3.5 separates VSCNT (vector store) from VMCNT (vector memory) — CDNA4 only has VM_CNT. | C-27 (CDNA4 has VS_CNT bound but doesn't model it). |

**RDNA3.5 total: 17 distinct gap classes.** All uncovered — no
`_DEFAULT_RDNA35_ARCH_PROFILE` exists today. Would need an entirely
different `ArchProfile`-shaped data structure to capture R-4 through R-8
(VOPD pair-formation rules — these are not "wait N cycles", they are
"this pair encoding is illegal"). **Rules R-4 through R-8 are NOT
gap rules at all** — they are pair-formation invariants that the validator
would need a new validation pass for.

### 2.4 Cross-arch summary table

| Class of rule | CDNA4 examples | RDNA3.5 examples | Validator coverage |
|---------------|---------------|------------------|--------------------|
| MFMA→MFMA accumulator chaining | C-1, C-2, C-3 | R-1, R-13 | Coarse: 4x4-vs-standard finish + +1 stall on type-switch |
| MFMA→ALU/MEM read of accumulator | C-4, C-5 | R-3 | Falls into MFMA-as-producer bucket, uses finish-cycles only |
| ALU→MFMA forwarding gap | C-9 | (covered by S_DELAY_ALU.INSTID_VALU_DEP) | Skipped by ALU-immediate exemption |
| Format-conversion settle | C-7, C-8, C-25 | (none — no TF32 emul on RDNA) | Two of three modeled (C-7, C-8); C-25 not |
| MFMA WAR on accumulator | C-6 | (n/a — WMMA has no separate AccVGPRs) | Not modeled |
| Counter-wait semantics | C-27 | R-16, R-17 | dscnt/vlcnt FIFO modeled; vscnt not modeled; SMEM-out-of-order not modeled |
| SCC/VCC carry chain | C-17, C-28 | (similar) | C-28 modeled as data-flow edge, not gap; C-17 out of scope |
| Pair-formation invariants (VOPD) | (n/a) | R-4, R-5, R-6, R-7, R-8 | Not modeled — no VOPD on CDNA |
| Dual-issue / parallel encoding | (n/a) | R-4..R-8 | Not modeled |
| Dependency-class gap encoding | (none) | R-10, R-11, R-12, R-13 | Not modeled (the *idea* of a named gap class isn't there) |
| LDS bank-conflict latency | C-26 | (similar) | Not modeled (perf concern) |
| Long-latency VALU (Trans, V_PERMLANE) | C-20, C-22, C-23 | R-11 | Not modeled |
| OPSEL/SDWA bit-position hazard | C-21 | (similar) | Not modeled |
| MFMA cycle-count varies by dtype | C-11, C-12 | R-1 (WMMA always 1-NOP) | Coarse: only 4x4-vs-standard |

**Combined CDNA4+RDNA3.5 inventory: 28 + 17 = 45 distinct gap classes**
(some overlap; ~38 unique semantics across both archs). Validator currently
models, in some form, **about 4 of those 38 unique semantic rules**, all on
CDNA4. Most uncovered rules are out-of-scope-by-design (kernel emitter
responsibility). The genuinely-missing-but-in-scope set is roughly:

- **C-4 / C-5 / C-6 / C-9 / C-11 / C-12-finer** for CDNA4 (~6 rules).
- **R-1 / R-3 / R-9 / R-10 / R-11 / R-13** for RDNA3.5 (~6 rules) — once
  RDNA3.5 is supported at all.
- **R-4 through R-8** are a separate class of work (pair-formation
  invariants) that needs a non-`ArchProfile`-shaped infrastructure.

---

## 3. Task 2 — Current implementation audit

### 3.1 Per-MFMA-opcode entries: count + ratio

**The validator has zero per-MFMA-opcode tables.**

Grep evidence (in `CMSValidator.py`):

```
$ grep -nE "MfmaF32_|V_MFMA_|V_SMFMA_" CMSValidator.py
792:        v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_T0_I0+0:...], v[74:75], ...   # docstring example
```

The only mention of an MFMA opcode by name is one example in a docstring.
There is no dictionary keyed on MFMA opcode names, no `MFMA_QUAD_CYCLES =
{"V_MFMA_F32_16x16x4F16": ..., ...}`, no per-family carve-out beyond the
single `"_4x4x" in str(rocisa_inst)` discriminator at `CMSValidator.py:1459`.

**MFMA classification works via two buckets:**

```python
# CMSValidator.py:1455-1461
try:
    rendered = str(rocisa_inst)
except Exception:
    return p.standard_mfma_finish_cycles
if "_4x4x" in rendered:
    return p.mfma_4x4_finish_cycles
return p.standard_mfma_finish_cycles
```

So the validator's MFMA-finish-time table has shape `dict[bool, int]` with
two entries (4x4 = 1 quad-cycle, everything-else = 3 quad-cycles). The
ISA defines (per §7.1.2 Table 28) about **20 distinct MFMA opcode families**
(F32_F32 5 variants, F32_F16 5 variants, F32_BF16 5 variants, I32_I8 5,
F64 2, F8/BF8 mixes 4+, F8F6F4 4, SCALE 2). With cycle counts ranging from
8 (4x4) through 16 (smallest 16x16) through 64 (32x32x1_2B). The
**ratio is 2 entries / 20 distinct opcode families = 10%**, but the
**granularity loss** is significant: the 5-variant F32_F32 family has
cycle counts 8 / 32 / 64 / 64 / 32 — the validator collapses these
to just two values (1 quad-cycle for 4x4, 3 quad-cycles for everything
else). A 32x32x1_2B (64 hardware cycles) and a 16x16x4 (32 hardware cycles)
both get the same "3 quad-cycles" finish in the validator — a 2× error
on the 32x32x1_2B side, and a 0× error in the safe direction on the 16x16x4
(over-strict by 4 cycles).

### 3.2 ArchProfile field shape

`_DEFAULT_CDNA4_ARCH_PROFILE` (`CMSValidator.py:154-164`):

| Field | Value | What it represents |
|-------|-------|--------------------|
| `name` | `"CDNA4"` | Human-readable arch tag |
| `isa` | `(9, 5, 0)` | gfx950 ISA tuple |
| `standard_mfma_finish_cycles` | `3` | MFMA finish window for "everything but 4x4" |
| `mfma_4x4_finish_cycles` | `1` | MFMA finish window for "_4x4x" family |
| `cvt_before_mfma_quad_cycles` | `2` | CVTPack→MFMA settle |
| `mfma_4x4_before_cvt_quad_cycles` | `5` | 4x4-PackMFMA→CVT settle |
| `mfma_type_switch_threshold_from_standard` | `5` | Trigger for +1 stall when prev MFMA was standard-class |
| `mfma_type_switch_threshold_from_4x4` | `3` | Trigger for +1 stall when prev MFMA was 4x4-class |
| `default_issue_quad_cycles` | `1` | Per-instruction issue cost (overridden for SNop) |

That is **9 scalar fields, all MFMA-or-CVT-shaped** plus one global default
issue cost. There is no field that could express R-4 (VOPD pair bank
restrictions), R-9 (DPP one-cycle delay), or even C-4 (MFMA→VALU read of
accumulator). Adding RDNA3.5 doesn't fit by adding fields; it requires a
different shape.

### 3.3 Where the gap-helper functions live

Three pair-specific helpers:

- `_quad_cycle_gap_ok(producer, consumer, num_mfma_per_subiter, graph)` —
  uses `_mfma_finish_cycles_for(producer)` as required gap.
- `_cvt_to_mfma_gap_ok(producer, consumer, subj_graph)` — uses
  `cvt_before_mfma_quad_cycles` constant (2).
- `_mfma_pack_to_cvt_gap_ok(producer, consumer, subj_graph)` — uses
  `mfma_4x4_before_cvt_quad_cycles` constant (5).

All three call into `cumulative_issue_cycles(graph, producer, consumer)`
for the observed gap. The simulator inside `cumulative_issue_cycles`:

- Walks the captured stream from producer to consumer.
- For each instruction, adds `_min_issue_quad_cycles_for(inst, profile)` to
  `current_issue` (default 1; SNop adds wait_state).
- For MFMA instructions, applies `mfma_free_at = current_issue + 1 +
  finish_cycles` and the type-switch +1 stall.

This is a per-instruction cycle simulator. Its surface — the cost-per-
instruction and the MFMA-finish discriminator — is class-based already
(via `InstructionCategory` post-009). The MFMA discriminator is the only
opcode-shape dispatch; everything else keys on category.

### 3.4 Counter-wait coverage

`validate_edge_wait_coverage` and `_classify_edge_coverage` (Phase 2)
discriminate edge counter by category:

```python
# CMSValidator.py:497-502
if counter == "dscnt":
    v = inst.dscnt
elif counter == "vlcnt":
    v = inst.vlcnt
elif counter == "vscnt":
    v = inst.vscnt
```

`vscnt` is read but never used as a producer-counter today (no
VECTOR_STORE producer edges are formed because `LoopBodyCaptureBuilder.
finalize()` rejects bodies containing VECTOR_STORE / FLAT / SMEM via
the categorize-and-guard path).

### 3.5 Pair/encoding-rule infrastructure

**None.** The validator has no concept of "two adjacent instructions form
an illegal pair encoding." The CMS body produces individual instructions in
a stream; the only pair-shaped invariant is the MIDDLE_PACK pair-leader/
pair-consumer interleaving check (`validate_middle_pack_pair_interleaving`),
which is shape-checking for a TF32-emulation pattern and not a hardware
encoding rule.

### 3.6 Sum-up: per-MFMA duplication is essentially zero, but per-MFMA
*granularity* is missing

The bead's framing — "we're duplicating 'MfmaF32_16x16x4F16, ..., one by
one'" — is empirically not what's happening. The validator treats MFMAs
as ONE thing with ONE binary discriminator (4x4 vs standard). The
per-family granularity that EXISTS in the ISA (and in `getMFMAIssueLatency`)
is **lost** rather than duplicated. The generalization opportunity is
"add granularity, replace the one binary string-parse with a class-based
multi-value classifier" — not "collapse N duplicate entries into one
class entry."

---

## 4. Task 3 — Generalization design

### 4.1 Shape goals

The right abstraction (informed by the inventory in §2):

1. **Producer-class × consumer-class is the right key** — most rules in
   §7.6 Table 38 read as "X writes Y, Z reads it, required N." The shape
   of X and Z varies by ISA but the cardinality is small (~5–10 producer
   classes, ~5 consumer classes per arch).
2. **The required-N is sometimes a function of the producer's parameters**
   (cycle count for the MFMA family, dtype, B-count). It cannot be a
   single integer in all cases. A `Callable[[ProducerInstance, ConsumerInstance],
   int]` works but is heavy; a `dict[ProducerSubclass, int]` works for
   most cases and a callable for the F8F6F4 conditional.
3. **Per-arch** — each arch has its own table; CDNA4 and RDNA3.5 differ on
   which classes exist *and* on the semantics of "wait N" (CDNA4 = quad-
   cycles, RDNA3.5 = either S_DELAY_ALU enums or NOP counts). The
   validator's measurement of `observed` should also be per-arch.
4. **Pair-formation rules are not gap rules** and should NOT be in the
   gap table. R-4 through R-8 need a separate `validate_pair_encoding`
   pass with its own data structure (a `dict[InstructionPairKind,
   list[PairConstraint]]`).

### 4.2 Class taxonomy sketch

Extend `InstructionCategory` with finer-grained `InstructionShape` for
the producer and consumer roles. The current `InstructionCategory`
buckets MFMA into one value; the new shape splits MFMAs by family.

```python
# Tensile/Components/InstructionShape.py (new)
class InstructionShape(Enum):
    # Memory ops — same as InstructionCategory (LR/LW/GR/SMEM/FLAT/VECTOR_STORE).
    LR = "LR"
    LW = "LW"
    GR = "GR"
    SMEM = "SMEM"
    FLAT = "FLAT"
    VECTOR_STORE = "VECTOR_STORE"

    # Scalar control.
    SWAIT = "SWAIT"
    SBARRIER = "SBARRIER"
    SNOP = "SNOP"
    SSETPRIO = "SSETPRIO"
    SDELAY_ALU = "SDELAY_ALU"   # RDNA3.5 only
    SCC_WRITER = "SCC_WRITER"   # carry-chain producer; per InstructionCategory.SCC

    # MFMA / WMMA — split by hardware family (the K-dim, B, dtype tuple).
    MFMA_4x4   = "MFMA_4x4"      # 4x4xK family (CDNA: 1 quad-cycle finish; uses _4x4x discriminator)
    MFMA_16x16 = "MFMA_16x16"    # 16x16xK family (varies 16 or 32 hw cycles)
    MFMA_32x32 = "MFMA_32x32"    # 32x32xK family (varies 32 or 64 hw cycles)
    MFMA_F8    = "MFMA_F8"       # F8F6F4 family (CDNA4-only; cycle count depends on whether either input is F8)
    MFMA_F64   = "MFMA_F64"      # DGEMM family (separate timing rules per §7.6)
    SMFMA      = "SMFMA"         # sparse MFMA (CDNA4); uses srcC for index reads — distinct hazard
    DLOPS      = "DLOPS"         # V_DOT* dot-product instructions (CDNA4)
    WMMA       = "WMMA"          # RDNA3.5 only
    SWMMAC     = "SWMMAC"        # RDNA3.5 sparse WMMA

    # Format conversions (the TF32-emulation chain plus the broader CVT family).
    CVT_PACK    = "CVT_PACK"     # v_cvt_pk_bf16_f32 (TF32 emul intermediate); inst sub-shape OK
    MIDDLE_PACK = "MIDDLE_PACK"  # the BF16-error-term family
    CVT_GENERIC = "CVT_GENERIC"  # other CVT_*_F32 (the §7.3 NOP-rule family)

    # VALU — split by long-latency family per §4.5 Table 12.
    VALU_GENERIC = "VALU_GENERIC"
    VALU_TRANS   = "VALU_TRANS"  # V_RCP/RSQ/EXP/etc. (the Table 12 set); CDNA4 §4.5 needs +1 wait state
    VALU_DPP     = "VALU_DPP"    # one-cycle delay per RDNA3.5 §7.7
    VALU_DLOPS   = "VALU_DLOPS"  # V_DOT* per CDNA4 §7.6 (already as DLOPS above)
    VALU_PERMLANE = "VALU_PERMLANE"
    VALU_CMPX    = "VALU_CMPX"   # writes EXEC; gates V_PERMLANE / V_MFMA

    # VOPD pair (RDNA3.5 only — pair, not single instruction).
    VOPD_PAIR    = "VOPD_PAIR"   # treated as one logical instruction with pair-constraints elsewhere.

    OTHER = "OTHER"
```

Mapping `InstructionCategory` → `InstructionShape` is many-to-many (an
`MFMA`-categorized instance is one of `MFMA_4x4`/`MFMA_16x16`/`MFMA_32x32`/
`MFMA_F8`/`MFMA_F64`/`SMFMA` based on its variant + dtype). The map can
live in `InstructionCategory.py` next to the existing
`_CLASS_NAME_TO_CATEGORY` registry. For MFMAs the discriminator needs the
rocisa-side `variant` field (currently inaccessible via nanobind — the
existing `"_4x4x" in str(...)` parses the rendered assembly; the
generalization should go further into rocisa to expose `variant` properly).

### 4.3 Gap-rule data structure

```python
# Tensile/Components/GapRule.py (new)

@dataclass(frozen=True)
class GapRule:
    producer_shape: InstructionShape
    consumer_shape: InstructionShape

    # required_quad_cycles is either a constant (most cases) or a callable
    # over (producer_inst, consumer_inst) -> int for parameterized cases
    # (e.g. F8F6F4 mixed-format MFMA where the cycle count depends on
    # whether either input matrix is F8).
    required_quad_cycles: Union[int, Callable[[Any, Any], int]]

    # Optional discriminators — narrows the rule to a sub-shape:
    # for example, "MFMA→MFMA same-vDst" gets requires_overlap_kind="same_vdst";
    # "MFMA→MFMA disjoint" gets requires_overlap_kind="disjoint_vdst".
    requires_overlap_kind: Optional[str] = None
        # one of {"same_vdst", "overlap_vdst", "disjoint_vdst", "src_a_or_b",
        #         "src_c", "lane_select"}; default = no overlap discrimination.

    # Diagnostic — short name + ISA cite for failure messages.
    name: str = ""
    isa_cite: str = ""

    def evaluate(self, producer_inst, consumer_inst) -> int:
        if callable(self.required_quad_cycles):
            return self.required_quad_cycles(producer_inst, consumer_inst)
        return self.required_quad_cycles


@dataclass(frozen=True)
class ArchProfile:
    name: str
    isa: Tuple[int, int, int]

    # Per-instruction issue cost: keyed by InstructionShape (not by class
    # name), with a default fallback. Captures C-11/C-12 (MFMA cycle
    # counts vary by family) AND C-20 (Trans needs +1).
    issue_quad_cycles: Dict[InstructionShape, Union[int, Callable]]
    default_issue_quad_cycles: int = 1

    # MFMA finish-cycles — keyed by InstructionShape. Replaces the two
    # scalar fields (standard / 4x4) with a per-shape map.
    mfma_finish_cycles: Dict[InstructionShape, int] = field(default_factory=dict)

    # Pair-specific gap rules. Keyed on (producer_shape, consumer_shape);
    # value is the GapRule (or list of GapRules when multiple
    # overlap_kinds give different N).
    gap_rules: Dict[Tuple[InstructionShape, InstructionShape], List[GapRule]] = \
        field(default_factory=dict)

    # Counter-wait semantics. Per arch because RDNA3.5 has VSCNT separately
    # and SMEM out-of-order rules differ.
    counter_kinds: Tuple[str, ...] = ("dscnt", "vlcnt", "vscnt")
    smem_strict_waitcnt_zero: bool = False  # CDNA4 §4.4 SMEM out-of-order
    flat_decrements_both_counters: bool = False

    # RDNA3.5-specific S_DELAY_ALU vocabulary. Empty for CDNA4.
    delay_alu_vocab: Optional["DelayAluVocab"] = None
```

The CDNA4 profile would populate `mfma_finish_cycles` with ~6 entries
(MFMA_4x4 / MFMA_16x16 / MFMA_32x32 / MFMA_F8 / MFMA_F64 / SMFMA) and
`gap_rules` with ~10 entries from §7.6 Table 38. Today's "two MFMA
discriminators + four scalar gap constants" become ~16 table entries —
each with one ISA citation.

### 4.4 Dispatch flow

```python
def _classify_edge_coverage(edge, subj_graph) -> List[Failure]:
    p_node, c_node = edge.producer, edge.consumer
    p_shape = shape_of(p_node)   # InstructionCategory + sub-shape inference
    c_shape = shape_of(c_node)

    profile = subj_graph.arch_profile
    if profile is None:
        return []  # arch not supported; skip (per zkzw)

    rules = profile.gap_rules.get((p_shape, c_shape), [])
    for rule in rules:
        # If the rule is overlap-sensitive, check the overlap kind first.
        if rule.requires_overlap_kind is not None:
            if vgpr_overlap_kind(p_node, c_node) != rule.requires_overlap_kind:
                continue
        required = rule.evaluate(p_node.rocisa_inst, c_node.rocisa_inst)
        observed = cumulative_issue_cycles(subj_graph, p_node, c_node)
        if observed < required:
            return [TimingTooCloseFailure(
                producer=cms_node_label(p_node, ...),
                consumer=cms_node_label(c_node, ...),
                iter_delta=_cms_iter_delta(p_node, c_node),
                expected_quad_cycles=required,
                actual_quad_cycles=observed,
                rule_name=rule.name,
                isa_cite=rule.isa_cite,
            )]
        return []  # first matching rule passed; positive classification

    # No gap rule applies — fall through to wait coverage (Phase 2 unchanged).
    return _phase2_wait_coverage(p_node, c_node, subj_graph)
```

This is **one classifier** with **one branch** (rule lookup) instead of
the current four sequential `if`s with order-dependence. The dispatch
order issue identified by o0ei dissolves: a `(p_shape, c_shape)` tuple
maps to at most one rule list, and within the rule list overlap-kind
discrimination is explicit (not implicit in branch order).

### 4.5 LoC estimate vs current state

| Component | Current LoC | After generalization | Net |
|-----------|------------|---------------------|-----|
| `ArchProfile` dataclass | ~50 | ~120 (new fields + factory helpers) | +70 |
| `_DEFAULT_CDNA4_ARCH_PROFILE` data | ~12 | ~80 (one row per gap rule + per-shape issue costs) | +68 |
| `_DEFAULT_RDNA35_ARCH_PROFILE` data | 0 | ~100 (entire RDNA3.5 ruleset) | +100 |
| `_classify_edge_coverage` dispatch | ~75 | ~25 (one table lookup) | −50 |
| `diagnose_missing_edge` dispatch | ~70 | ~25 | −45 |
| `_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` / `_mfma_pack_to_cvt_gap_ok` | ~120 | merged into one helper, ~60 | −60 |
| `_mfma_finish_cycles_for` | ~40 | replaced by `profile.mfma_finish_cycles[shape]`, ~10 | −30 |
| `shape_of` (new) | 0 | ~80 (delegates to InstructionCategory + MFMA-family inference) | +80 |
| New `GapRule` dataclass + helpers | 0 | ~60 | +60 |
| `vgpr_overlap_kind` (new) | 0 | ~50 (operand-byte-set comparison) | +50 |
| Tests (new) | n/a | +200 (per-shape classifier tests, per-rule positive/negative tests) | +200 |
| Test fixture updates | n/a | ~50 (drop `num_mfma_per_subiter`, plumb shape through) | +50 |

**Net code change: roughly +500 LoC, of which ~+200 is tests.** This is
a large refactor, NOT a 50-LoC tweak. It replaces a 4-rule fragile dispatch
with a ~16-rule (CDNA4) + ~17-rule (RDNA3.5) data-driven matrix.

### 4.6 Test impact

- `test_dataflow_graph_register_gaps.py` (89 refs to the three gap helpers)
  becomes ~30 refs to the unified `gap_check(producer, consumer, graph)` —
  most fixtures reduce to "given a producer of shape X, consumer of shape
  Y, the rule fires/passes."
- `test_arch_profile_unregistered_isa.py` (12 refs) — unchanged in
  semantics; the behavior of "unknown ISA short-circuits timing checks" is
  preserved.
- New: `test_arch_profile_cdna4_gap_rules.py` — one test per rule entry in
  the CDNA4 profile. Per §2.2, that's ~16 tests for the modeled subset.
- New: `test_instruction_shape_classifier.py` — one test per shape value
  verifying the canonical rocisa class maps correctly. ~20 tests.
- New: `test_arch_profile_rdna35_gap_rules.py` — gated; landing if/when
  RDNA3.5 support is wanted.

Bead's auto-bundle threshold ("~50 LoC, no new conceptual surface, no
test fixture rewrites") is **massively exceeded.** This is a multi-PR
effort that should be split into beads as proposed in §5.

---

## 5. Task 4 — Re-scope of s5g1 (recommendation)

### 5.1 What s5g1 should remain

**The original o0ei narrow refactor.** Replace the four-branch dispatch in
`_classify_edge_coverage` and `diagnose_missing_edge` with a category-keyed
table or function (per `QUAD_CYCLE_DISPATCH_AUDIT.md` §6 — "Option 4").
~50 LoC plus drop `num_mfma_per_subiter`. That work is well-defined,
order-independent, and ships even if the larger generalization never
happens.

The generalization in §4 above is **a different, larger project**. It
should not block s5g1.

### 5.2 New sibling beads to file

Recommended (do NOT file in this commit; wait for user decision):

1. **`rdna35-arch-profile-stub`** (priority: medium)
   Scope: register a stub `_DEFAULT_RDNA35_ARCH_PROFILE` with `name="RDNA3.5"`
   and `isa=(11,5,1)` so kernels for gfx1151 stop hitting the
   "no ArchProfile registered" warning. Initial gap-rule set is empty
   (validator behaves as if the kernel says "skip timing-related
   validation"). Unblocks RDNA3.5 kernels from emitting the warning at
   runtime. ~30 LoC. **Should be done before any larger RDNA3.5 timing
   work.**

2. **`mfma-shape-from-rocisa`** (priority: high)
   Scope: replace the `"_4x4x" in str(rocisa_inst)` discriminator in
   `_mfma_finish_cycles_for` with consumption of rocisa's
   `getMFMAIssueLatency` (or a Python-side mirror keyed on
   `(dtype, variant.B, isaVersion)`). Either teach nanobind to expose
   `MFMAInstruction.variant` as a Python attribute, OR add a sibling
   `getMFMAFamily()` helper that returns an enum. Then `_mfma_finish_cycles_for`
   replaces the string-parse with an attribute read. **This is the
   single highest-leverage bead** — it eliminates the only opcode-name
   discriminator in the validator and immediately reduces the
   16x16/32x32 collapse. ~80 LoC + test updates.

3. **`gap-rule-table-cdna4`** (priority: medium-high; depends on #2)
   Scope: implement §4.2–4.4 above. Replace `ArchProfile`'s scalar fields
   with the `gap_rules: dict[(InstructionShape, InstructionShape),
   List[GapRule]]` map. Populate the CDNA4 profile with the ~16 rules
   from §7.6 Table 38. Adds rules C-4, C-5, C-9 from §2.2 (genuinely
   missing, in-scope) as new entries — the dispatch becomes
   table-driven so adding a new rule is a one-line table update.
   ~400 LoC + ~200 LoC tests.

4. **`vopd-pair-validator`** (priority: low; deferred until RDNA3.5 has
   real users)
   Scope: implement R-4 through R-8 as a new `validate_pair_encoding`
   pass. Independent of the gap-rule work — VOPD pair-formation
   invariants are not "wait N cycles" rules. ~150 LoC.

5. **`s_delay_alu-modeling`** (priority: low; deferred)
   Scope: model RDNA3.5 §16.5 S_DELAY_ALU as a first-class
   `InstructionShape.SDELAY_ALU` with a typed argument
   (`DelayAluKind` enum: `VALU_DEP_N`, `TRANS32_DEP_N`, etc.). Once
   modeled, the gap-rule table can encode "an S_DELAY_ALU(VALU_DEP_2)
   between producer and consumer satisfies the C-9 / R-10 gap." Big
   conceptual win because the hardware *literally names the gap class*
   in the instruction encoding; the validator gets to read it directly.
   ~120 LoC.

### 5.3 Recommended sequencing

```
o0ei (Option 4 dispatch refactor; current s5g1 scope, narrow)
   ↓
mfma-shape-from-rocisa (high-leverage; needed for any cycle-count fidelity)
   ↓
gap-rule-table-cdna4 (table refactor of ArchProfile + gap helpers)
   ↓
rdna35-arch-profile-stub (silences warnings on RDNA3.5 kernels)
   ↓
vopd-pair-validator + s_delay_alu-modeling (deferred until RDNA3.5 has
                                            real validation traffic)
```

Each step is independent enough to land alone; downstream steps benefit
from upstream simplifications but do not require them.

### 5.4 What s5g1's body should say after re-scope

(Suggested edit to `rocm-libraries-s5g1` body, for the user to apply if
they accept this re-scope.)

> Original scope (per o0ei memo, `QUAD_CYCLE_DISPATCH_AUDIT.md` §6):
> implement Option 4 — replace the four-branch dispatch in
> `_classify_edge_coverage` and `diagnose_missing_edge` with a category-
> keyed table. ~50 LoC.
>
> Confirmed scope after `ISA_GAP_GENERALIZATION_AUDIT.md`: this is the
> right narrow scope. The larger generalization (gap-rule table replacing
> `ArchProfile`'s scalar fields; per-MFMA-family finish cycles; RDNA3.5
> support) is split into 4 sibling beads (see audit memo §5.2). They do
> not block this bead.

---

## 6. Open questions for the user

1. **Accept the re-scope?** This audit recommends keeping s5g1 narrow
   (the original o0ei dispatch refactor) and filing 4 sibling beads for
   the larger generalization. Alternative: re-scope s5g1 to be the
   `gap-rule-table-cdna4` work itself, treating o0ei's dispatch refactor
   as a sub-step that lands inside it. The first option is incremental
   and shippable; the second is closer to "do the generalization
   properly the first time" but blocks for ~weeks.

2. **MFMA finish-cycles fidelity — is it worth the work?** The validator
   currently maps every non-4x4 MFMA to "3 quad-cycles finish." The ISA
   distinguishes 16x16 (~16 hw cycles = ~4 quad-cycles) from 32x32 (~32
   hw cycles = ~8 quad-cycles). The validator's current "3" is over-strict
   for 16x16 (catches bugs that aren't real) and under-strict for 32x32
   (misses bugs that are real). Are there known false-positive or
   false-negative reports today that the bead-bookkeeping might clarify?

3. **RDNA3.5 — actual user demand?** The whole RDNA3.5 column of the
   inventory is uncovered. Filing the four RDNA3.5 beads is wasted work
   if no kernels are shipping for gfx1151. What is the current
   gfx1151 validator traffic? If nonzero, `rdna35-arch-profile-stub` is
   urgent to silence the warning at minimum. If zero, all four beads
   defer.

4. **VOPD pair-formation — is this correctness or performance?** RDNA3.5
   §7.6 says "hardware does not function correctly if rules are not met"
   (R-4 through R-7 are explicit "**Hard rules — instruction does not
   function**"). If we ever generate VOPD instructions, this is a
   shipping-correctness validator, not a perf check. Does the kernel
   emitter ever generate VOPD? If not, `vopd-pair-validator` defers
   indefinitely; if yes, it should jump to high priority.

5. **The `_4x4x` string-parse hack: how committed is it?** The discriminator
   parses rendered assembly because `MFMAInstruction.variant` is not
   exposed via nanobind. If the user agrees to expose `variant` (or a
   `getFamily()` helper) on the C++ side, `mfma-shape-from-rocisa` lands
   cleanly. If nanobind exposure is off-limits, the bead has to mirror
   the C++ logic in Python — workable but more duplication. (The current
   `getMFMAIssueLatency` is also class-method-shaped; mirroring its
   `(arch, dtype, B)` branch in `InstructionCategory.py` is ~30 LoC.)

6. **Single source of truth for cycle counts: rocisa or validator?** Right
   now `getMFMAIssueLatency` (rocisa C++) and `_mfma_finish_cycles_for`
   (validator Python) are two independent encodings of similar — but not
   identical — knowledge. (rocisa returns `mi_divisor` and
   `miIssueLatency`; validator returns `finish_cycles`. The relationship
   is "finish_cycles ≈ K-pass / mi_divisor" for non-4x4 MFMAs, but the
   constants are stored separately and could drift.) Is the right move
   to have rocisa be the single source (validator imports the Python-
   bound helper), or the validator be the single source (rocisa
   factors into validator-style `ArchProfile` lookups)? This is a
   layering question more than a coding question — it affects 009's
   "rocisa is layer-violation-free" stance.

---

## 7. Self-review

A self-review pass before commit: did the audit miss anything?

**Possibly missed:**

- **Wave32-vs-wave64 timing differences** on RDNA3.5. Wave64 issues each
  VALU twice; this changes the *observed* gap simulator if the validator
  ever models wave64 RDNA3.5 schedules. Out of current scope (CDNA is
  wave64 always, RDNA is mixed); flagged but not enumerated.
- **S_GETREG / S_SETREG** wait states from §4.5 Table 11 (rows 1–3): 2
  wait states between an `S_SETREG` and its `S_GETREG`. Out of scope
  for the validator (kernel emitter handles).
- **Inter-wave / cross-SIMD effects.** The brief explicitly flagged these as
  probably out of scope. The ISAs do mention waveslot scheduling and
  WAVELIMITER but nothing the validator could model from a single-wave
  schedule capture. Confirmed out of scope.
- **L2/cache invalidation timing.** Tables on pp. 84–86 of CDNA4 cover
  `S_DCACHE_INV` and friends. The validator does not see these
  instructions in CMS bodies; out of scope.
- **MFMA Transpose Load (CDNA4 §11.4)** — a special LDS-load shape that
  reads matrix rows from LDS in a transposed layout. If used in the
  TF32 emul chain it could have its own gap rules; the brief did not
  mention any. Flagged for separate inquiry.
- **MIDDLE_PACK pair-interleaving** is an *invariant* (the validator has
  `validate_middle_pack_pair_interleaving`) but not a *timing gap*. The
  audit did not include it in the gap inventory because it is not a
  hardware hazard — it is a correctness check on the TF32 emulation
  software pattern. The line between "ISA hazard" and "software
  pattern invariant" is fuzzy here; the user may want both classes in
  the same `validate_*` infrastructure later.

**Honest about the framing:** the bead's framing assumed a per-MFMA-opcode
table explosion. This audit found instead that the validator is *already
class-based* (one binary discriminator and four scalar constants); the
generalization opportunity is to *add granularity* (split MFMA into
per-family shapes, encode gap rules as a table) rather than to
*deduplicate* (collapse N rows into 1). The §5 re-scope reflects this.
The user asked to be told if the existing code was already mostly
class-based; it is.

