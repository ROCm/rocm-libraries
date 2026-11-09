# Flash-Next true-mono: the fast-math / Q-MMA / V-skip / MFMA-router stack on this branch (gfx950, MI355X)

This package reproduces a stack of four default-off rocKE knob families on top of the true-mono work in
`users/vanantha/rocke-monokernels`. It gives the commands to generate the code objects and to check them.
It gives **no performance numbers**. For the base stack (`c2s3b`), the checkout pins and the server environment,
read [../repro/REPRO.md](../repro/REPRO.md) first. This file only describes what is added.

## Status of this branch (read this first)

- The branch is a **rebase onto the branch tip `c58783fdf91`** of three small source changes. The stack was
  validated, in earlier work, on a tree built from the older commit `4478c174a13` plus Vanantha's patch 03
  (`../repro/patches/03-*.patch`) plus these same three changes.
- **Generation-level check (CPU, done):** the full served-function directory generated from this branch is, file
  for file, the same as the one generated from the validated tree, for each of the stacks C, D, E and F below.
  Method and result are in "Check 1".
- **Not re-validated on a GPU.** No layer oracle, accuracy run or serving run was done on this exact tree. Anything
  that needs a GPU below is the procedure to run it, not a result.
- Default-off: with none of the new knobs set, the generated LLVM IR and the sidecars of the existing builds are
  unchanged (Check 2).
- The knobs live in the Python library builders (`library/kernels/gfx950/qwen38_mono/`). **gfx950 and the Python
  engine only.** Nothing was done for other architectures or the C++ engine, and no arm of the stack uses them.
- The schedule string of stack D is **not rocKE code**. It is an `ATOM_Q38_SCHED` entry and lives on the ATOM side
  (see "The stacks").
- Which pieces need the unmerged base: all of it. `kernels/gfx950/qwen38_mono/`, `stages2/hc.py`, `qsa2/` and the
  fused MoE are on `users/vanantha/rocke-monokernels` and not on `develop`. The harness stage (`h-qk-*`) is the
  private frozen harness named in the base repro.

## The commits

| commit subject | files | knobs |
|---|---|---|
| `feat(rocke): add default-off fast-math knobs to the qwen38 mono stages` | `stages2/hc.py`, `qsa2/ctx.py`, `qsa2/knobs.py` | HC `invmath` (`f64` default, `rsq32`), HC `sigm` and QSA `sigm` (`acc` default, `hw`) |
| `feat(rocke): add default-off Q-MMA and V-skip knobs to qsa2` | `qsa2/knobs.py`, `qsa2/q567.py` | QSA `q5_mm`, `q7_mm` (`none` default, `mfma`), `q7_vskip` (`False` default); mutants `q5_mm_drop`, `q7_mm_drop`, `q7_vdrop` |
| `feat(rocke): add default-off MMA router knob (router_mma) to the fused MoE` | `moe.py`, `moe_fused.py`, `stage_moe_fused.py`, `tune/catalog.py` | `FusedKnobs.router_mma` (0 default, 1), mutant `router_mma_kdrop`; tune-catalog entry |

All of them are tolerance-class numerics knobs (see "Correctness"). `tests/` gains CPU tests for every new knob.

## The stacks

| name | what | picks / knob strings |
|---|---|---|
| C | fast math + Q-MMA + V-skip | HC `invmath=rsq32,sigm=hw` on every HC stage of the delta and qsa layer kinds; QSA2 `sigm=hw,q5_mm=mfma,q7_mm=mfma,q7_vskip=True` (global to Q3-Q7) |
| D | C + a schedule string | the qsa.r4s schedule group of the base recipe with `hold.q5` removed, `hold.q2=96`, `dyn.q7` removed. **ATOM-side string, not rocKE code**: the generated functions are C's |
| E | D + the MFMA router on delta.r4s | MoE pick `delta.r4s:router_mma=1` (picks key `moe.router_mma` in the cell's `canonical`) |
| F | E + the MFMA router on qsa | MoE picks `delta.r4s:router_mma=1;qsa.r4s:router_mma=1;qsa.r1:router_mma=1` |

The schedule group of D, as `sched.c2.txt` of the generated directory shows it, differs from C's only in the
`qsa.r4s` group:
```
C: qsa.r4s:dyn.q6=1,dyn.q7=1,head_sleep=128,hold.m1=32,hold.m3=64,hold.q2=192,hold.q5=96,prio.m1=2,prio.q2=2,qdense=3
D: qsa.r4s:dyn.q6=1,head_sleep=128,hold.m1=32,hold.m3=64,hold.q2=96,prio.m1=2,prio.q2=2,qdense=3
```
Stacks E and F keep D's string. The `router_mma` routers go in through the served picks path
(`lookback/gen_tuned.py` builds the MoE functions from the pick's `canonical`; `tune/catalog.py` has the new
tolerance knob `router_mma`).

Helper files in this directory:
- `recipes.sh`: `gen_stackC`, `gen_stackD`, `gen_stackE`, `gen_stackF`. They run the base recipe's `_c2_gen` with the
  knob strings above, check that the knobs reached the sidecars, and append the served-function hash lines to
  `expect_k3.txt`.
- `tools/mk_picks_fm.py`: adds the HC knobs to a copy of `WP2a-fix/picks_final.json`.
- `tools/mk_picks_moe.py`: adds the MoE canonical changes to a copy of `lookback/picksH.json`.
- `tools/wp2a_gen_fm.py`: runs the frozen `wp2a_gen.py` with the one change that the new HC fast-math knobs are
  not refused by its "tolerance-class knobs are not served" guard.

## Generate the code objects

The layout below is the one the harness expects (a job root with `stages/`). `$WORK` is any scratch directory.
```text
$WORK/job/stages/h-qk-<sfx>        the frozen harness stage (base repro, "What to check out")
$WORK/job/stages/rocke-stack       this branch's dnn-providers/hip-kernel-provider/rocke directory
$WORK/job/stages/avstack           recipes.sh and tools/ from this directory
/atom                              the ATOM-truemono tree (base repro)
```
Python needs torch (CPU is enough for generation), numpy and the other packages of the base repro's venv.
The assembler must be an LLVM 23 `llvm-as` (`ROCKE_LLVM_AS`); the qsa2 `.bc` files and the sidecars embed its path.

```bash
git -C <rocm-libraries> archive <this branch> dnn-providers/hip-kernel-provider/rocke \
  | tar -x -C $WORK/job/stages/rocke-stack --strip-components=3

H=/job/stages/h-qk-<sfx>
export QK_SFX=<sfx> H CS_GEN_PP=1 WP2B_KNOBS_JSON=picks_all.json WP2B_JSON_ONLY=1
export STACK_TOOLS=/job/stages/avstack/tools
export ARM_FILES="$H/truemono/RB/arms_rb.sh $H/truemono/WP2a-fix/arms_wp2a.sh $H/truemono/WP2b/arms_wp2b.sh /job/stages/avstack/recipes.sh"
export ROCKE_LLVM_AS=<llvm-as of LLVM 23>
export PYTHONPATH=/job/stages/rocke-tmbase/platform/python:/job/stages/rocke-tmbase/library   # the job's PYTHONPATH (hybrid rocKE stage, base repro); CS_GEN_PP=1 makes the recipes put the generating stage first
source $H/common/server.sh >/dev/null; source $H/truemono-quick/arms.sh
cd /job
gen_stackC rocke-stack /job/fns/stackC
gen_stackD rocke-stack /job/fns/stackD
gen_stackE rocke-stack /job/fns/stackE
gen_stackF rocke-stack /job/fns/stackF
```
`CS_GEN_PP=1` and the `WP2B_*` variables are the same pitfalls as in the base repro.

Each directory holds `sched.c2.txt` (the served `ATOM_Q38_SCHED`), `expect_k<K>.txt` (the build and hash lines the
server log must show), the HC, GEMV, MoE and QSA2 functions and `manifest.sha256`.

## Serve with ATOM

The server environment is the base arm `c2s3b` ([../repro/knobs/c2s3b.env](../repro/knobs/c2s3b.env)) with two
substitutions:
1. `$FN` is the stack's directory (`/job/fns/stackE`, ...);
2. `ATOM_Q38_SCHED` is that directory's `sched.c2.txt`.

`PYTHONPATH` is `/atom:$RK/platform/python:$RK/library` with `$RK` = this branch's rocKE directory, for generation and
for serving, so the same tree builds and serves. The rest of the environment (`ATOM_Q38_EXTERN_STAGES` template, tail,
draft vocabulary, round-boundary knobs) is unchanged. The server-log check of the harness (`quick.sh`) requires every
line of `expect_k3.txt` to appear, so a build that served a different function is caught.

ATOM side: the tree and patches of the base repro (`ATOM-truemono` = ATOM `352229d6` + `patches/01` + `patches/02`).
This stack adds no ATOM source change of its own; it adds only the environment strings above.
[INFERENCE] The validated runs used an `ATOM-truemono` snapshot of the same lineage. It was not diffed against
patches 01 + 02 for this branch.

## Check the result

### Check 1: generated functions equal the validated stack's (CPU, done for this branch)

For each of C, D, E and F, the directory generated from this branch was compared with the directory generated from
the validated tree (`4478c174a13` + patch 03 + the three changes), with the same harness, picks, assembler and job-root
layout. Both sides were generated here. For D, E and F the regenerated validated side was also compared with the stored
file lists of the validated runs: all 668 files equal. Result, per stack:
- both directories hold the same 668 files;
- 615 files are byte identical (every `.ll`, `.json`, `k/`, `hc/` and `qsa2/*.ll` file);
- the other 53 are path-only: the 47 `qsa2/*.bc` (they embed the directory path; their `.ll` is identical), and
  `manifest.sha256`, `qsa2.gen.json` and `expect_k0..3.txt` (they hold the directory path and the hashes of the
  `.bc` files). After removing paths and `.bc` hashes, they are equal.

The check was also run on the unpatched tip against the older base (default recipe, no knobs): same 668 files, same
615 byte identical, same path-only set. So the newer base commits do not change any served function of these
recipes. The branch's own default build matches the unpatched tip's in the same way.

### Check 2: default-off

With no new knob set, generated IR and sidecars are byte for byte the base's. Evidence: the golden hashes
`library/tests/golden/qwen38_moe_split_ir_sha256.json` (test below) and the regenerated default recipe above.
A sidecar records `sigm`, `q5_mm`, `q7_mm` and `q7_vskip` only when they are on.

### CPU tests

From `dnn-providers/hip-kernel-provider/rocke/library`, with `PYTHONPATH=../platform/python:.`,
`ROCKE_BACKEND=python`, an ATOM-truemono checkout in `ATOM_ROOT` / `ATOM_TRUEMONO_ROOT`, and torch available:
```bash
python -m pytest tests/test_qwen38_*.py -q -rs
```
New or changed tests: `test_qwen38_stages2_hc.py` (HC `invmath` / `sigm` act and default is unchanged),
`test_qwen38_qsa2.py` (three new mutant cells; each new knob changes its stage and the default is unchanged; knob
validation), `test_qwen38_p15_router.py` (`router_mma` changes the IR, default unchanged, the new mutant differs,
refusals). Tests that compile with COMGR or run on a device need a matching COMGR or a GPU and are skipped or fail
without one.

### Correctness on a GPU (procedure; not run on this tree)

1. Layer oracle: the stacked layer of each stack is compared with the layer of the stack without the knob, per cell
   (delta / delta_ple / qsa at r1 and r4s, qsa r4, long contexts), in the harness's C2 layer gate
   (`C2/c2_gates.sh`, as in the base repro's `gate` mode), with the stack's `$FN` and `PYTHONPATH`.
   Expected outcome:
   - D against C: bitwise equal (a schedule string changes only holds).
   - E / F against D: every tensor bitwise equal, except the router logits, which agree within 1 ulp; the top-10
     expert selection is identical.
   - V-skip against Q-MMA without it: bitwise equal.
   - Fast math and Q-MMA against the base stack: tolerance class, judged by the accuracy suite below.
2. Planted mutants must differ: `q5_mm_drop`, `q7_mm_drop`, `q7_vdrop` (QSA2 knob `mutate`) and
   `router_mma_kdrop` (`FusedKnobs.mutate`).
3. Accuracy suite: the base repro's T4 (GSM8K 5-shot chat, HumanEval, RULER 8K and 32K) with `t4_gate.py` against
   the registered hybrid pool. The stack passes when each suite is inside its registered tolerance.
4. Server-log checks of `quick.sh` on every arm-run, as in the base repro.

## Caveats

- Tolerance-class knobs: serve them only behind the accuracy gate above.
- Stacks E and F change only the MoE router functions that the picks name (the sidecars of `m3`, `m4`, `m5` of those
  cells record the knob).
- This package states no speed claim. Timing, if wanted, follows the base repro's `quick` procedure.
