# WIP: RFC 0017 / 0018 Alignment Plan

Working document. Not an RFC. Delete before the PR opens.

Scope: reconcile `0017_UniversalKernelDescriptor.md` (umbrella) and
`0018_UniversalMatchDescriptor.md` (the graph matcher) on branch `rfc-0018-umd`,
plus the collateral they drag in: `0020_UniversalEngineDescriptor.md` (merged on
develop) and `examples/0017_UniversalKernelDescriptor_WorkedExample.md`.

Supersedes `~/RFC_0018_0019_Alignment_Plan.md`, which was written against the
numbering that commit `ffad8ae03f6` undid. Its findings are carried forward and
re-verified; its Decisions 1-3 are re-opened, because all three assumed an
expression-language RFC that no longer exists.

## Cause

- `7990db65ff7` added the UMD RFC as 0018.
- `a976aecc25f` moved graph matching and symbol binding from the UMD to the UED.
- `db6690cb58f` split the expression language out as a new RFC 0018 and pushed
  the matcher to 0019.
- `ffad8ae03f6` deleted the expression-language RFC (deferred) and moved the
  matcher back to 0018, **without reverting the citations `db6690cb58f` had
  introduced**.
- `43eccca82cc` retitled the matcher doc to RFC 0018.

Everything below follows from the last two commits landing without a text pass.

The deleted expression-language RFC is preserved at
`~/0018_DescriptorExpressionLanguage.md`. It is the only place several contracts
0018 depends on were ever written down; see Decision C2.

---

## How to read this document

Findings are numbered in three classes, and the numbers are referenced
throughout, including from each other:

- **A1-A10** — mechanical debris. Dead links, stale labels, naming residue.
  No judgment; defined in the Class A table below.
- **B1-B3** — structural decisions. Ownership and scope of whole documents.
  Everything else depends on them.
- **C1-C10** — substantive technical contradictions between the RFCs.

A fix that reads "Per B1" means the decision register immediately below
determines what the text is rewritten *to*.

**Status legend.** **SETTLED** — decided, evidence recorded, no further
judgment. **OPEN** — needs a decision. **DONE** — applied.

---

## Decision register

Each entry states the question, the answer, why, and what was rejected. The
rejected options are recorded because the same three arguments will otherwise
be re-run at review time.

### B1 — Who owns the expression language? SETTLED

**Question.** `ffad8ae03f6` deleted the expression-language RFC as deferred
work. Its grammar, closed operator set, type system, three-valued semantics,
bounded interpreter, lowering parity, and conformance suite are cited 30 times
by 0018 and are now specified nowhere. Who owns them?

**Answer.** It stays an **unwritten follow-up**, named by role and not by
number, exactly as 0017 already treats its other ten follow-ups.
**RFC 0017 §5 is the interim authority** for the operator vocabulary.

**Why.** It preserves the deferral as an intentional act rather than an
accident of a reverted commit, and it preserves the reason `db6690cb58f` split
the language out in the first place: the UDD and UHD must be able to inherit
the language without inheriting the graph matcher. A matcher that owns a
contract three sibling formats depend on is the ownership bug the split fixed.

**Rejected.**
*0018 absorbs it as an interim normative appendix* — makes 0018 readable
standalone, but re-creates precisely that ownership bug.
*0018 absorbs it permanently and is retitled* — honest about where text lives,
but reverses a deliberate architectural decision from two commits ago.

**Consequences.** A1 and A3 rewrite to role-named prose with no link. **C2 is
the cost of this choice** and is not optional: 0017 §5 does not contain several
contracts 0018 leans on, so they need an interim home.

### B2 — Is RFC 0020 in scope for this branch? SETTLED

**Question.** 0017 and 0018 both define the UED as carrying a `nodes`
structural pattern. 0018 defers the format to "the UED follow-up", which is
RFC 0020 — already merged on develop, carrying no `nodes` field, closing its
field set with `additionalProperties: false`, and declaring "any unknown field
is a hard rejection". 0020 does not merely omit the pattern; it **rejects** it.
It also drops `sdk_version`, which 0017 §4 requires on a UED. Fix here, or
record and defer?

**Answer.** **Fix 0020 in this branch.**

**Why.** Otherwise the branch ships a series whose stage-one matcher rests on a
field the UED RFC actively rejects — a mechanical contradiction between two
documents in one directory, not an editorial one.

**Rejected.**
*Leave 0020 alone, record the gap* — smaller diff and separate reviewers, but
leaves develop self-contradictory for an unbounded period.
*Move the pattern spec into 0018* — smallest 0020 change, but contradicts
0017 §14.2's assignment of the pattern to the UED row.

**Consequences.** Enables **C8**. Forces the `ued/2.0` major bump under 0020's
own §11.2 rule. This is the single most expensive decision in the plan.

### B3 — Does 0017 §14.2 start naming RFC numbers? SETTLED

**Question.** 0017 references no follow-up by number anywhere; §14.2 is a
name-only table of eleven rows. 0018 meanwhile claims to be "the UMD + graph
matcher row" and calls 0020 "the UED + engine registry follow-up" — neither row
exists. Start numbering, or fix only the false claims?

**Answer.** **Stay unnumbered.** Fix only 0018's claims about which rows exist.

**Why.** The convention survives renumbering, which is exactly what just
happened: 0017 emerged from a two-commit renumber with zero stale references
precisely because it names nothing. Numbering would have made it a casualty.

**Rejected.**
*Number the landed ones and merge the two matcher rows* — a stable map, but
couples the umbrella to a series that has renumbered twice in a fortnight.
*Number all eleven rows now* — reserves numbers for documents nobody has
committed to writing.

**Scope of this rule.** B3 governs 0017's **forward-looking** rows. Landed
RFCs (0013, 0016, 0020) are still cited by number where 0018 already does so;
a reference to a document that exists is not the thing this rule guards
against.

---

## Class A: mechanical debris (SETTLED)

| # | Where | Problem | Fix |
|---|---|---|---|
| A1 | 0018, 30 links on 29 lines: 87, 101(x2), 335, 336, 337, 338, 418, 439, 472, 500, 561, 606, 616, 697, 723, 746, 800, 918, 1021, 1053, 1066, 1080, 1112, 1125, 1126, 1127, 1128, 1131, 1170 | Cites `[RFC 0018 §N](0018_DescriptorExpressionLanguage.md#...)`. File deleted; worse, the label is this document's own number, so cited "§6 Operators" resolves to 0018 §6 "Layout and Stride-Order Constraints" — a coherent-but-wrong self-reference, not a visibly dead link. | Per B1: role-named prose, no link. |
| A2 | 0018:12 | "The RFC number is provisional and is reconciled ... at PR-open time." | Delete. |
| A3 | example:13 | Dead link to the deleted expression-language RFC. | Per B1. |
| A4 | example:15 | `[RFC 0019: The Graph Matcher](../0019_UniversalMatchDescriptor.md)` — renumbered. | Retarget `../0018_UniversalMatchDescriptor.md`, relabel RFC 0018. |
| A5 | 0018:193 | `[§8](#9-...)` — anchor right, label wrong. | Label `§9`. |
| A6 | 0018:197 | `[§17](#18-risks)`. | Label `§18`. |
| A7 | 0018:300 | `[§11](0017_...#12-packaging-and-delivery)`. | Label `RFC 0017 §12`. |
| A8 | 0018:1287 | `[§19 ...](#18-risks)`. | Label `§18`. |
| A9 | 0018 TOC/headings 4, 6, 8 | "Constraint Vocabulary" / "...Constraints" is pre-split vocabulary. §4 itself disclaims the noun ("not a set of criterion *kinds* (there are none)"); 0017 §5, 0018's scope table, and 0018's glossary all say *criteria*. | Rename to Criteria; update TOC and ~10 inbound internal links. |
| A10 | 0018:207, 0018:597 | Jira IDs `AICK-1698` in prose. Workspace convention forbids Jira IDs; 0017 carries none. | Remove; keep the surrounding sentences. |

Recorded so it is not re-litigated: every cross-file `#anchor` into
0017/0018/0020 currently resolves. The renumber rotted filenames and RFC
numbers, not slugs.

---

## Class C: substantive contradictions

Two kinds live here, and the difference matters at review time:

- **Adjudicated by evidence — C3, C7, C9, C10.** One side was simply wrong
  against the source tree or against the other document. No alternative
  existed; the entry records the evidence, not a choice.
- **Decisions with real alternatives — C1, C2, C4, C5, C6, C8.** These carry a
  **Question** and a **Rejected** block for the same reason the B-entries do.

### SETTLED

**C3 — `head_size` and `mask_mode` are not attributes. 0017 is wrong.**
0017:702 makes `$sdpa_fwd.head_size` *the* attribute-namespace exemplar;
0017:834-835 put `$sdpa_fwd.head_size` and `$sdpa_fwd.mask_mode` in its SDPA
criteria; 0017:481 uses `$sdpa_fwd.head_size` as a UHD feature.
`schemas/sdpa_attributes.fbs:19-78` declares neither field. The real attributes
are `generate_stats`, `alibi_mask`, `padding_mask`, `causal_mask`,
`causal_mask_bottom_right`, `dropout_probability`, `attn_scale_value`,
`left_bound`, `right_bound`, `max_seq_len_kv`, `diagonal_alignment`,
`mma_core_mode`, `implementation`.
0018:155-157 ("head size, batch, and head count are **not** attributes") and the
worked example (`$q.dims[3]`, `$kernel.mask_mode`) are correct. 0017 §6 already
uses `$q.dims[3]` for the same quantity, so 0017 also contradicts itself.
**Fix in 0017:** :481, :702, :834, :835 → positional dims / `$kernel.mask_mode`.
Knock-on: :670 and :2190 justify `rsqrt` as `1/sqrt(head_size)`; reword to the
head extent read positionally, keeping the operator's justification intact.

**C7 — the stride-order convention is 0018's, not 0017's.**
0018:372 and A.4:1141-1142 attribute "an ordered list of logical dimension
indices, outermost (largest-stride) first" to RFC 0017 §5. 0017 contains no such
definition — only the literals `[0,1,2,3]` (:822) and `[0,2,3,1]` (:967, :969).
The literals are compatible; the definition is 0018's own.
**Fix in 0018:** state it as this RFC's definition; drop the false attribution.
0018's `extractStrideOrder` inverse-encoding reconciliation (:387-394) and the
layout aliases (A.4:1144-1149) are likewise 0018-only and stay, unattributed.

**C9 — three SVGs still draw the deleted design.**
`ukd_criteria_match.svg:56-58` (embedded in **0017** at :854) and
`umd_symbol_binding.svg:43-44` (embedded in 0018 at :265) render criteria using
the removed `shape` operator with named dims `[b, h, sq, d]`;
`umd_symbol_binding.svg:45` uses `$q.head_size` (dead per C3), `:57-58` shows
`$q.batch`/`$q.num_heads`/`$k.kv_heads` named dims;
`umd_binding_model.svg:64-65` labels a row `$q.seqlen_q, $q.head_size` /
"(named shape dims)" inside a box titled "Symbol table the UED publishes".
All three are half-updated: frame labels are post-change and correct, only the
payload is stale. `ukd_concepts.svg` and `umd_matcher_pipeline.svg` are clean —
do not touch.
These are hand-authored SVGs with absolute `x`/`y` and fixed-width Consolas
text, so replacing a criterion changes line count and width and the containing
`<rect>` geometry needs adjusting with it.
**Do all three payload fixes as one sweep** — it is the same vignette drawn
three times, and fixing one leaves the RFCs disagreeing about what a criterion
looks like.

**C1 — SETTLED. "Closed" means no extension point, not a frozen list.**
0017:654-655 ("The set is closed, so naming an unknown operator fails load
validation") and the deleted RFC's A.6 ("This table is exhaustive") were making
two different claims under one word. The series means the first: **there is no
registry, namespace, dotted key, or provider hook by which a descriptor
introduces an operator, and an unrecognized operation key fails load
validation.** It does *not* mean the vocabulary is frozen across spec revisions;
operators are added additively as consumers motivate them.
The exhaustive enumeration is documentation the expression-language follow-up
owns and will publish later; it will grow over time.
**Fix in 0017 §5:** relabel the :657-663 table a representative vocabulary, not
the normative enumeration, and reword :654-655 in the two-part form above.
**Wording hazard — the reason this is not simply "non-exhaustive":** a bare
"this list is not exhaustive" reads as licensing *unrecognized* operators at
load time, which contradicts fail-closed loading. The sentence must say both
halves: the set admits no descriptor-side extension and an unlisted key is
refused *at any given revision*, while the published list grows across
revisions. Never one half alone.
**Consequence:** 0018's `if` (:104, :272) is no longer a divergence and stays.
`abs`, `pow`, `log2` from the deleted RFC need no action — nothing in the tree
uses them today. 0018 needs no operator content change.

**Question.** 0017 §5's table has six rows and calls the set closed. 0018 uses
`if` and n-ary `not_present`; the deleted RFC's table carried `abs`, `pow`,
`log2`, `if` beyond 0017's and *also* called itself exhaustive. Two tables,
both self-declared closed, and they differ. Which is normative, and what does
"closed" actually mean?

**Rejected.**
*0017 §5 absorbs the superset and stays a genuinely exhaustive closed table* —
one table in the interim authority, and 0018 needs no change; rejected because
freezing an enumeration in the umbrella is what forces a 0017 edit every time
a consumer motivates an operator, and the enumeration belongs with the
language.
*0017 §5 keeps six rows; 0018 drops `if` and n-ary presence* — narrowest set,
nothing invented ahead of the language RFC; rejected because n-ary
`not_present` is load-bearing at 0018:873 and the mask classifier in the
worked example is an `if` chain written the long way.

**C2 — SETTLED. 0018 states the semantics locally, flagged as moving out.**
Under B1 the interim authority (0017 §5) is a table plus prose and does not
carry: three-valued unknown propagation and the `and`/`or` truth tables;
checked arithmetic (fail closed on overflow, on a zero divisor for `/`, `%`,
`ceil_div`, `false` for `divisible`, on a non-positive `log2`/`rsqrt`); the
`Ref` syntactic position; lowering parity. 0018 §5, §13, §15, A.2 and A.5 all
depend on them.
0018 states each as a **matcher-local rule marked "to move to the
expression-language follow-up"**. Source text to lift verbatim:
`~/0018_DescriptorExpressionLanguage.md` §7 (:327-360), §8 (:380-391),
A.2 (:726-737), A.3 per-operator rules (:767-789).
**This also settles the absent-operand contradiction** (previously tracked
separately), which is the same question asked twice:
- 0017:913-914 — a field check on an absent optional "neither passes nor fails,
  it simply does not run".
- 0018 A.2:1110-1117 — the reference "resolves to unknown ... so the match
  declines (fail closed)", and then *claims these are the same thing*.
  They are not, and 0018 contradicts itself three lines later at :1116 by
  depending on the propagating behaviour.
The correct rule, from the deleted RFC, distinguishes two levels:
  - **per-operand** — an unresolved reference yields unknown, which
    *propagates*; per the `or` table an `or` with a definite-`true` arm is
    `true`, **not** a decline. This is what makes "absent, or present and
    constrained" work.
  - **at the root** — a `Bool`-rooted criteria expression still holding unknown
    fails closed and declines.
**Fix:** state both levels in 0018 A.2, delete the false equivalence at :1114,
and reword 0017:913-914 to name the level it means. Out-of-range `dims[i]`
(0018:1118-1120) is unknown, uniform with an absent optional — one outcome for
all three producers.

**Question.** B1 makes 0017 §5 the interim authority, but §5 is a six-row
operator table plus prose. Where do the contracts it does not carry live until
the language RFC is written?

**Rejected.**
*Fold them into 0017 §5* — the interim authority would actually become
authoritative and one source would serve the UDD and UHD too, but it inflates
a framing section and the material has to move out again later.
*Restore the deleted RFC as a 0017 appendix or unnumbered companion* — the text
already exists, complete, at `~/0018_DescriptorExpressionLanguage.md`, so
nothing is lost and no number is reserved; rejected because it reverses B1's
deferral in substance while honouring it only in form.

**C4 — SETTLED. Presence operators only; arity one-or-more.**
`.present` as a field (0018:220, :228, :290, :867, :870, A.2:1102) is dropped;
`{"present": ["$x"]}` / `{"not_present": ["$x"]}` cover an optional operand and
an optional attribute alike, as 0017:907-911 already says. Arity is **one or
more**, not the deleted RFC's ">= 2", which would have made 0017:909, 0017:916,
0017:920 and the worked example's one-argument calls compile errors while
0018:873's three-operand call stayed legal.
**Fix in 0018:** rewrite the `.present` guard idiom
`{"or": [{"!": "$x.present"}, {"==": ["$x", 0.0]}]}` (:290, :867, :870) as
`{"or": [{"not_present": ["$x"]}, {"==": ["$x", 0.0]}]}`; drop `.present` from
the A.2 namespace table; keep :292's list form, now documented as one-or-more.
**Fix in 0017 §5:** state the arity explicitly at :907-911 — it is currently
implied by example only, which is how the conflict got in.

**Question.** Presence has two mechanisms and an arity conflict. 0017 uses
operators only, for operands and attributes alike. 0018 splits them: operators
for operands, a `.present` **field** for attributes. Separately the deleted
RFC typed the operators n-ary as ">= 2 arguments", which makes 0017's own
one-argument calls compile errors. One mechanism or two, and what arity?

**Rejected.**
*Keep both mechanisms* — `.present` does read naturally inside an `or` guard
and A.2 types it as a real Bool field, but it is two ways to ask one question
in a series that works hard elsewhere to avoid exactly that.
*`.present` field only* — most uniform, but rewrites 0017:907-930 wholesale,
and a field read on an absent operand is precisely the read that must **not**
propagate unknown, so it needs a carve-out anyway. The operator form has that
carve-out built in.

**C6 — SETTLED. Not a contradiction; different catalogs. Scope both claims.**
0017:869-870 says the escape hatch "is a last resort ... the validated catalog
of MIOpen CK convolution and rocKE SDPA applicability needed none."
0018 §17 (:831-836) has the SDPA-forward example needing two native matchers.
These are different engines: 0018's example is grounded on
`dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine`, which is not
rocKE (`hip-kernel-provider/rocke/`, gated by `HIPKERNELPROVIDER_ENABLE_ROCKE`)
and not MIOpen CK. So 0017's claim is true of the two families it studied and
0018 studied a third.
**Fix:** 0017:869-870 scopes the claim explicitly to the two families surveyed;
0018:831-836 notes the asm-SDPA builder sits outside that survey. Neither claim
is withdrawn — the rhetorical force of "needed none" is what needs bounding.

**Rejected.** *Withdraw 0017's claim* — the surveyed result is real evidence
about how often the hatch is needed and is worth keeping; the defect is that
it reads as a claim about every family. *Drop the two native matchers from
0018's example* — they are grounded in shipping code
(`wouldFwdByteStridesFitUint32`, `getMaskType`) and inventing a declarative
substitute would misrepresent the migration.

**C5 — SETTLED. Keep the names as this RFC's proposal; drop the attribution.**
`GraphMatcherFn`, `NativeRegistry`, `MatchDescriptor`, `matchSymbol`,
`BoundTokens`, and `MatchContext` (0018:421-423, :430, :445-448, :484, :919,
:956, :1026-1027) appear **nowhere** in the repo — swept `projects/hipdnn` and
`dnn-providers`, zero hits for all six. 0017:867-868 says only "the matcher
names a registered **native predicate** and carries typed arguments".

**Question.** These are invented names, attributed to a document that does not
use them, describing plumbing that does not exist. Keep, strip, or promote?

**Rejected.**
*Strip to 0017's "native predicate" vocabulary* — guts §7's cost analysis and
Open Question 1, which needs concrete types (`BoundTokens` being
`string -> int64_t`, too weak to carry a tensor) to say anything at all.
*Promote the names into 0017 §5 as well* — 0017 §5 is a framing section and
nowhere else names C++ types that do not yet exist.

They stay: §7's cost argument and Open Question 1 both need concrete types to
say anything. What goes is the false provenance.
**Fix in 0018:** stop citing `[RFC 0017 §5]` at :423 and :484 as the source of
these names; introduce them as the API *this RFC proposes*, and say plainly
that no such plumbing exists today. 0017 §5 keeps "native predicate" as the
concept; 0018 names the mechanism.
**Same family, mechanical:** 0018's grounded-case line citations have drifted
against the source. `getMaskType` is `SdpaFwdPlanBuilder.cpp:293`, cited as
`:276` (0018:466); `getKernelNameKey` is `:301`, cited as `:287` (0018:476).
Re-verify every `file:line` citation in §7 and §16 in one pass.

**C8 — SETTLED. The UED-row material moves into RFC 0020.**
0020 becomes the real "UED + graph matching" follow-up of 0017 §14.2, which is
what its own header already claims to be and what 0018's deferrals already
point at. This is the decision B2 makes possible.

**Question.** 0017 §14.2 draws a line: the "UED + graph matching" row owns the
`nodes` pattern, the matching semantics, the symbols matching publishes, and
the op-schema registry; the "UMD + applicability" row owns criteria over
already-published symbols, memoization and fail-prune, the native-matcher
hatch, and arbitration. 0018 specifies **both**, with roughly 450 lines on the
UED row's side. Where does that material belong?

**Rejected.**
*Keep it in 0018; retitle 0018 and re-cut 0017 §14.2 into one matcher row* —
honest about where the text already sits and the smallest 0020 change, but it
redraws the umbrella's row structure to match an accident of drafting order,
and leaves the UED's own RFC unable to explain its most important field.
*Split by layer: the `nodes` field and schema to 0020, its semantics to 0018* —
each document owns what it is already good at, but the pattern's story then
spans two RFCs, which is the single-row structure's whole purpose.

*Moves out of 0018 into 0020:*

| 0018 section | Lines | Why it is UED-row material |
|---|---|---|
| §2 The Matcher's Input: hipDNN's Graph Model | 112-172 | UID-edge reconstruction and "the matcher needs an op-schema registry" (:126-130) |
| §3 Symbol Binding: What the Engine Publishes | 174-265 | Literally "the symbols matching publishes" (0017:1793) |
| §9 stage one + the root-opcode index over engines | 530-544 | "the matching semantics that bind it against a graph" |
| §10 the pattern half of static-matcher parity | 599-611 | Pattern lowering |
| Appendix B Op-Schema Registry Generation | 1188-1343 | Named outright in 0017:1793 |
| Open Question 4 (alternative patterns per engine) | 969-976 | A `nodes`-block design question |

*0018 keeps* a **reader-side contract**: which namespace roots criteria may
reference and what each yields — not the full field tables, which belong with
the publisher. Drawing that line is the one judgment call left inside C8; take
the narrow reading (roots and types, not the per-field enumeration).

*0020 additionally gains, per B2:*

- `nodes` and `sdk_version` in §4.1's example, §4.2's field table, and the
  inline `ued/1.0.json` (including the `required` array, where `nodes` belongs).
- §9 registration: a **pattern-compile** step. 0017 §8.1 requires the compiled
  pattern be resolved and cached at registration; 0020 §9 step 2 binds only
  `heuristic`, `metadata`, and naming KDPs.
- §10.2 semantic validation: op/operand-name resolution against the op-schema
  registry, and the UED half of pair validation.
- §13 testing: pattern cases — registry resolution, optional-`?` binding, the
  published-symbol-set check, UMD/UDD/UHD validation against that set.
- §A.2/§A.3 appendix examples: A.2 claims to be "fully-populated" and has
  neither field; A.3's "minimal" UED needs `nodes` too.

*Statements in 0020 that become false and must be reworded:*
:132-135 ("exactly the members below", "The object has no logic"), :150,
:210-212 and :326-333 (`additionalProperties: false` / "any unknown field is a
hard rejection" — as written these **reject** the pattern the UED is defined to
carry), :35-38 and :33-34 (deliverable list omits the pattern), :79-80 ("No
other silent contradictions"), :251-259 (§7 membership omits the pattern from
what a pack inherits), :465-468 (glossary UED entry).

*Version consequence, do not skip:* 0020 §11.2 permits a minor bump only for
"additive, absence-safe changes ... where absence is well-defined as the
behavior before the field existed". A **required** `nodes` fails that test, so
this is a **major** bump to `ued/2.0` under 0020's own rule, not a table edit.

*Also fix while in 0020:* :6 calls it the "UED + engine registry" follow-up, a
§14.2 row that does not exist; :53-55 and :449-450 attribute the drop-in trust
deferral to 0017 §14, which says nothing about trust (it is 0017 §16 :2058-2060
and §17 Q1 :2171-2172); :510-511 vs the worked example — 0020 §A.2 and
`example:501` give the **same UUID** `7d4c2a9e-...` two different engine names,
and the example's `"attention_dense forward engine"` fails 0020's mandatory
`namespace:local` pattern.

**C10 — SETTLED. Smaller calls, recorded so they are not re-derived.**

- **Pair validation must name the pack — four sites, one atomic change.**
  0017:1551-1555 requires the load error to name "the matcher, the engine, the
  token, **and the pack that paired them**". 0018:526, :755, :792-794 and
  :1075 each state the rule and each omit the pack. Fix all four together or
  three contradict the fourth.
- **`$scale` is a type error in the worked example.** :377-378 and :412-413 use
  `{"value_or_default": ["$sdpa_fwd.attn_scale_value", "$scale"]}` — a tensor
  handle as the fallback for a Float attribute. 0017:875-879 makes
  `$q.value_f32` the way a compile-time scalar is read, and both
  `value_or_default` arms must be type-compatible (0017:675-676). Use
  `$scale.value_f32`. The `?`-bound operand `$scale` at :516 and the presence
  gates at :173-177 are correct as they stand and do not change.
- **Pattern syntax leaking into 0018.** `one_of` / `any` (:280, :946) and the
  `?` optional-operand suffix (:849-850, :887, :889, :1252) are UED pattern
  vocabulary; they travel with C8.
- **Glossary (0018:993-1041).** Add the `Escape hatch` umbrella entry
  0017:2334-2338 introduced; split `Catalog` from `Bound token state` per
  0017:2295-2306 (:1034-1037 still says "the matchers resolved", pre-split
  wording); the `JsonLogic` entry at :1020-1023 defines the term by deferring to
  the deleted RFC and must be rewritten per B1; the UED and Structural-pattern
  entries move with C8.
- **0018:5-12 blockquote** names two §14.2 rows that do not exist ("UMD + graph
  matcher", "UED + engine registry") and, at :10-11, inverts ownership by
  claiming 0020 "adopts the pattern block specified here" while :178-180 says
  0018 does not specify it. Rewrite against the real rows and C8's split.
- **Prior-art drift (0018:978-991 vs 0017:2204-2213).** Divergent and harmless;
  leave unless a reviewer raises it.

---

## Work packages

Ordered by dependency.

**WP1 — 0017 corrections.** C3 (`head_size`/`mask_mode`), C1 (operator-table
relabel), C4 (presence arity), C6 (scope the "needed none" claim), C2 (reword
:913-914 to name its level). Prerequisite: 0018 cannot be aligned to a 0017
that contradicts itself and the FlatBuffers schema.

**WP2 — the C8 move.** 0018 → 0020, as one unit. Largest internal link blast
radius in 0018 (20+ links to `#3-symbol-binding-...`, 10+ to `#9-...`), so
every other 0018 fix's target location depends on it.

**WP3 — 0020 completion.** The B2 field/registration/validation/testing
additions, the false-statement rewordings, the `ued/2.0` bump, and the §6/§53
mis-citations.

**WP4 — 0018 residual.** A1-A2, A5-A10, C2 local semantics, C4 rewrite, C5
attribution + citation re-verification, C10's pair-validation and glossary
items, the blockquote.

**WP5 — diagrams (C9).** All three payloads in one sweep, after WP1-WP4 settle
the vocabulary. `ukd_criteria_match.svg` is 0017-embedded, so it lands here
rather than WP1.

**WP6 — worked example and final sweep.** A3, A4, `$scale.value_f32`, and a
re-run of the anchor and section-label verification across all four documents.

## Rules that keep this from thrashing

1. **Never fix a pair-validation site in isolation** — four sites, one change.
2. **Settle §3's and §9's disposition (WP2) before anything cross-referencing
   them.**
3. **Stale section labels last** (A5-A8), after WP2's renumbering.
4. **SVG content before any SVG rename**, never the reverse.
5. **Decide unknown-vs-decline once** (C2), in 0018 — fixing A.2 alone just
   relocates the inconsistency.
6. **Do not churn what is verified clean:** no `shape` operator, named dims,
   `all`, `Vector`, or `contiguous` in any RFC prose; the umbrella-matcher rule
   correctly deleted in both; fusion-by-engine stated correctly; every
   cross-file anchor resolving.

---

## Applied changes

### WP1 — 0017 corrections. DONE

All line numbers below are pre-edit. Verified against
`projects/hipdnn/flatbuffers_sdk/schemas/sdpa_attributes.fbs:19-78` (the plan's
path `schemas/sdpa_attributes.fbs` is one level short of the real one).

- **C3.** :481 UHD feature `$sdpa_fwd.head_size` → `$q.dims[3]`. :702 attribute
  exemplar → `$sdpa_fwd.attn_scale_value`, plus a sentence stating that head
  size, batch, and head count are tensor extents, not attributes — matching
  0018:155-157. :834-835 → `{"==": ["$q.dims[3]", 128]}` and the mask pinned as
  the worked example pins it: `!causal_mask`, `!causal_mask_bottom_right`,
  `$kernel.mask_mode == "none"`.
  Knock-on: :670 and :2190 reword `1/sqrt(head_size)` to `1/sqrt` of the head
  extent read positionally; the operator's justification is unchanged.
  Second knock-on, not in the plan: :841-843's "every `$`-token the criteria
  read ... the engine's pattern published" became false once the example reads
  `$kernel.*`. Scoped to graph tokens, with the KMD token named and its
  per-kernel memoization noted (consistent with :892-893).
- **C1.** :654-655 reworded into the required two-part form — no descriptor-side
  extension point and an unlisted key refused at load *at this revision*, while
  the published set grows additively across revisions. The :657-663 table is
  labelled a representative vocabulary, not the normative enumeration.
- **C4.** :909-911 now states the arity: n-ary over one or more references,
  one-argument being the ordinary case.
- **C6.** :869-870 scoped — "across the two families whose applicability was
  validated for this design ... none needed one". The claim is bounded, not
  withdrawn.
- **C2 (0017 half).** :913-914's "neither passes nor fails, it simply does not
  run" replaced by the two-level rule: a field read on an absent operand yields
  unknown, unknown *propagates*, and the **root** is what fails closed. :926-928's
  justification paragraph rewritten to match — it argued from the old
  skip semantics and would have contradicted the new text.

Not touched, deliberately: :1217 and :1911 use `head_size` as a *builder/KMD*
field name, which is correct and unrelated to the attribute namespace.

### WP2 — the C8 move. DONE

All line numbers pre-edit. 0018 went 1343 → 1044 lines, 0020 634 → 1031.

**Moved out of 0018 into 0020**, per C8's table:

| Was | Is now |
|---|---|
| 0018 §2 The Matcher's Input (112-172) | 0020 § 5 The Graph Model the Pattern Matches |
| 0018 §3 Symbol Binding (174-265) | 0020 § 6 Symbol Binding: What the Pattern Publishes |
| 0018 A.2's namespace grammar + field table (1083-1109) | 0020 § 6.1 The published field set (normative) |
| 0018 §9 stage one + root-opcode index (530-543) | 0020 § 7 Pattern Matching: Stage One |
| 0018 §10 the pattern half of parity (601-611) | 0020 § 7, last paragraph |
| 0018 Appendix B (1188-1343) | 0020 Appendix B, verbatim but for re-targeted refs |
| 0018 Open Question 4 (969-974) | 0020 § 17 Open Questions (the section did not exist) |
| 0018 glossary UED / Structural-pattern / Auto-binding formula / Symbol lifecycle entries | folded into 0020 § 6; the UED entry survives in 0018 pointing at RFC 0020 |
| `umd_binding_model.svg`, `umd_symbol_binding.svg` | embedded in 0020 § 5 and § 6; `umd_matcher_pipeline.svg` stayed with 0018 §8 |

**0018 keeps a reader-side contract**, the narrow reading C8 asks for: new §2
"The Symbol Table Criteria Read" — the five roots, what each yields, who binds
each, and the three properties of the binding that criteria depend on
(`$kernel.*` is not pattern-bound, hence per-kernel re-evaluation; an
unresolvable reference is a load error). Types and per-field enumeration are
*not* restated; §2 and A.2 both point at 0020 § 6.1. A.2 keeps only the
reader's rules: reference syntax, pair-validation scope, and the absent-operand
outcome (which C2/WP4 rewrites in place).

**0018 renumbered**: §4-§21 became §3-§20 and A.2 was retitled "Variable
references and resolution". Every internal link, the TOC, and the §-labels were
rewritten with it; this incidentally settled **A5** (`[§8](#9-...)` — the
anchor's target is now genuinely §8) and **A6**.

**0020 renumbered**: old §5-§15 became §8-§18 to make room; new § 17 Open
Questions pushed the glossary and appendix to § 18/§ 19. Its `§ N` prose
references were shifted with it, RFC-qualified ones (`RFC 0017 § 14.2`) left
alone.

**Pattern vocabulary that travelled with C8** (C10): `one_of` / `any` and the
`?` optional-operand suffix are gone from 0018's prose — the constraint table's
opcode and optional-operand rows, §16's worked example, and §17's
engine-granularity risk now name the pattern facility and cite RFC 0020 § 4
rather than spelling its syntax.

**Reframed rather than moved:** 0018 §17's "Op-schema registry coupling" risk.
The mitigation is now the UED's and is cited, not restated; what stays is the
consequence for a *reader* — a criterion cannot detect registry drift itself,
which is why the published set is pair-validated at load.

**Also fixed, because the move made them false:** 0018's title (it no longer
specifies the UED's pattern) and its blockquote, which claimed the
non-existent "UMD + graph matcher" row and inverted ownership by saying 0020
"adopts the pattern block specified here" — it now claims the real "UMD +
applicability" row and defers the pattern to 0020. 0020:6's "UED + engine
registry" row claim likewise became "UED + graph matching", the row it now
actually fills.

Verified: every intra- and cross-file anchor across 0017/0018/0020/example
resolves, except the pre-existing A1/A3/A4 dead links to the deleted
expression-language RFC and the renumbered example link — WP4 and WP6's work.

Deliberately **not** done here, though C8 lists them: the 0020-side additions
that are WP3's (`nodes`/`sdk_version` in § 4's field table and inline schema,
the registration pattern-compile step, § 13.2 semantic validation of pattern
names, § 16 pattern test cases, the `ued/2.0` bump, and the appendix examples).
0020 § 6/§ 7 therefore forward-reference § 4 and § 13.2 for material WP3 adds.
