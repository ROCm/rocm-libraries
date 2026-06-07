export const meta = {
  name: 'codegen-p4-round6',
  description: 'P4 Stage-2 round 6 (closing push to >=80%): arch-breadth rich configs + Solution-derivation breadth + mid-file targets; authoring+verify only, driver gates',
  phases: [{ title: 'Design' }, { title: 'Verify' }],
}

const SHARED = [
  'ENV (paths INSIDE the container):  CON=tl-char ; PROJ=/work/projects/hipblaslt/tensilelite',
  '  Host worktree: /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage',
  '  Host project: <root>/projects/hipblaslt/tensilelite (== $PROJ). Edit on HOST; RUN in container.',
  '  Container cp312 has pytest/coverage. NEVER run the whole -m unit suite; NEVER use a Monitor.',
  '',
  'ISOLATED MEASURE (one fresh process, own COVERAGE_FILE):',
  '  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.kept_6_<ID> -w $PROJ $CON \\',
  '    pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <TEST_NODE>',
  '  then: docker exec -e COVERAGE_FILE=$PROJ/.coverage.kept_6_<ID> -w $PROJ $CON coverage json -o /tmp/<ID>.json',
  '',
  'HARD RULES:',
  '  1. ADD-ONLY. NEW files ONLY under Tensile/Tests/unit/characterization/. NEVER modify/delete an',
  '     existing file. UNIQUE basename; test ONLY in its intended dir (stray dup at Tests/unit/ root',
  '     breaks collection). Clean up any temp .coverage.* you create in the project dir.',
  '  2. --cov=Tensile is a PATH (never Tensile.x). One COVERAGE_FILE per shard. pytestmark=pytest.mark.unit. CPU-only.',
  '  3. If you use a syrupy snapshot, SEED its .ambr with --snapshot-update ONCE then confirm it passes',
  '     WITHOUT it. NEVER leave a golden unseeded. Pure-assert tests need no snapshot.',
  '  4. Two-run determinism matters: if the TARGET file executed_lines vary run-to-run (coverage',
  '     concurrency=multiprocessing artifact), the verifier WILL reject — prefer assertions on stable',
  '     emitted text / derived state, keep configs small/deterministic.',
  '  5. Keep each input bounded (limit=8). NEVER push, NEVER commit. The driver gates + commits.',
  '',
  'TEMPLATE: copy _codegen/test_r2_store_char.py (emit via',
  '  `from config_harness import emit_kernels_from_config; emit_kernels_from_config(CFG, limit=8, arch=ARCH)`).',
  '  ARCH-BREADTH: base your per-arch config on the VALID curated logic shape for that arch under',
  '  _codegen/data/<arch>/*.yaml (these are known-valid for that arch — match its DataType/MI/wmma',
  '  format), then widen with a small ForkParameters (bias/activation/GSU/MI variants). RDNA archs',
  '  (gfx1100/gfx1201) use WMMA not MFMA — copy their curated logic format, do not force MFMA shapes.',
  '  Real shipped configs/logic: Tensile/Tests/common/** and the tuning tree.',
].join('\n')

const CANDS = [
  // ---- arch breadth: a rich multi-feature config per arch hits arch-gated asm-cap branches in KWA/KW ----
  { id: 'rich_gfx908', arch: 'gfx908', target: 'Tensile/KernelWriterAssembly.py', kind: 'arch',
    feat: 'gfx908 (CDNA1) rich GEMM: base on data/gfx908/*.yaml; widen with bias/activation + GSU + 2 MI shapes. gfx908 has distinct asm-caps (no some gfx94x features) -> exercises arch-gated KWA/KW arms.' },
  { id: 'rich_gfx90a', arch: 'gfx90a', target: 'Tensile/KernelWriterAssembly.py', kind: 'arch',
    feat: 'gfx90a (CDNA2) rich GEMM: base on data/gfx90a/*.yaml (BBS/HHS/DB/SB), widen with bias+activation+GSU+DTV fork. gfx90a-specific MFMA/cap arms.' },
  { id: 'rich_gfx950', arch: 'gfx950', target: 'Tensile/KernelWriterAssembly.py', kind: 'arch',
    feat: 'gfx950 (CDNA4) rich GEMM with MX/scale + bias+activation+GSU fork (base on data/gfx950/*.yaml). gfx950-specific scale/swizzle/MX arms beyond what R2-R5 hit.' },
  { id: 'rich_gfx1100', arch: 'gfx1100', target: 'Tensile/KernelWriter.py', kind: 'arch',
    feat: 'gfx1100 (RDNA3) WMMA GEMM: base on data/gfx1100/*.yaml (WMMA format, NOT MFMA), widen with bias/activation. RDNA wmma codegen arms differ from CDNA mfma -> distinct KW/KWA lines.' },
  { id: 'rich_gfx1201', arch: 'gfx1201', target: 'Tensile/KernelWriter.py', kind: 'arch',
    feat: 'gfx1201 (RDNA4) WMMA GEMM: base on data/gfx1201/*.yaml, widen modestly. RDNA4-specific wmma arms.' },
  { id: 'rich_gfx1250', arch: 'gfx1250', target: 'Tensile/KernelWriterAssembly.py', kind: 'arch',
    feat: 'gfx1250 rich GEMM: base on data/gfx1250/*.yaml, widen with bias/activation/cluster fork. gfx1250-specific arms (newest arch).' },
  // ---- Solution-derivation breadth (still 62%, the largest %-gap) ----
  { id: 'sol_breadth', arch: 'gfx942', target: 'Tensile/SolutionStructs/Solution.py', kind: 'emit',
    feat: 'Solution derivation breadth (1165 miss, 62%). A WIDE ForkParameters cartesian across the assignment/validity arms NOT yet hit: AssertSummationElementMultiple, AssertFree0/1ElementMultiple, GlobalReadVectorWidth, StoreVectorWidth, WorkGroupMapping, PersistentKernel, NumElementsPerBatchStore, LdsPadA/B, UnrollMajorLDS. Use solutions_from_config (derivation) and assert the derived/rejected states; many missing lines are validity branches.' },
  // ---- mid-file targeted (reliable emit) ----
  { id: 'activation3', arch: 'gfx942', target: 'Tensile/Activation.py', kind: 'emit',
    feat: 'Activation remaining (207, 76%). Fork ActivationType over functions + ActivationComputeDataType combos not yet covered (geluscaling, silu/swish, erf, exp, dgelu/gradient-activation if supported). Also ActivationFused/ActivationAlt variants.' },
  { id: 'gwb2', arch: 'gfx942', target: 'Tensile/Components/GlobalWriteBatch.py', kind: 'emit',
    feat: 'GlobalWriteBatch remaining (338, 77%). Fork edge (non-multiple free dims) x StoreVectorWidth[1,2,4] x bias+ScaleAlphaVec+activation-on-store + StoreRemapVectorWidth together so the remaining edge/remap/fused-store arms run.' },
  { id: 'asmstore2', arch: 'gfx942', target: 'Tensile/AsmStoreState.py', kind: 'emit',
    feat: 'AsmStoreState (155, 71%) + AsmAddressCalculation (163, 60%). Fork store configs with large/edge tensors + StoreRemap + atomic + 64-bit addressing so the remaining store-state and address-calc arms run.' },
  { id: 'gsu3', arch: 'gfx942', target: 'Tensile/Components/GSU.py', kind: 'emit',
    feat: 'GSU remaining (238, 77%). The reduction/workspace arms (442-588) still resist. READ Components/GSU.py to find the exact gate; try GlobalSplitUAlgorithm MultipleBufferSingleKernel / GSUC with a valid ProblemType. If the reduction needs a separate reduction kernel the single-config harness cannot trigger, report kept=false with the line reason (ceiling).' },
  { id: 'subtile3', arch: 'gfx950', target: 'Tensile/Components/Subtile/SubtileGREmit.py', kind: 'emit',
    feat: 'Subtile remaining (SubtileGREmit 143 + LogicalScheduler 151 + Kernel 130, gfx950). Extend subtile configs with more global-read/scheduler variants (different subtile shapes, scale, wave-split-k) so the remaining subtile GR-emit + logical-scheduler arms run.' },
  { id: 'driver3', arch: 'gfx942', target: 'Tensile/ClientWriter.py', kind: 'driver',
    feat: 'ClientWriter (158, 63%) + TensileCreateLibrary/Run (197, 63%) remaining. Drive MORE --cpu-only client/createlibrary scenarios (multiple solutions, different client flags, library-format variants, problem-size files) so the remaining client-config / run-orchestration arms run. Copy test_cpu_only_switch.py + ClientPath patterns.' },
  { id: 'librarylogic2', arch: 'gfx942', target: 'Tensile/LibraryLogic.py', kind: 'driver',
    feat: 'LibraryLogic (142, 83%) + BenchmarkProblems (111, 65%) remaining. Drive more analysis/selection scenarios (multiple sizes, efficiency variants, the granularity/rollup arms) via the LibraryLogic analysis path + BenchmarkProblems forked solutions.' },
]

const CAND = {
  type: 'object', additionalProperties: false,
  required: ['id', 'test_path', 'emitted', 'err', 'measured_marginal', 'kept', 'note'],
  properties: {
    id: { type: 'string' }, test_path: { type: 'string' },
    emitted: { type: 'integer' }, err: { type: 'integer' },
    measured_marginal: { type: 'integer' }, kept: { type: 'boolean' }, note: { type: 'string' },
  },
}
const GOLD = {
  type: 'object', additionalProperties: false, required: ['test_path', 'stable', 'reason'],
  properties: { test_path: { type: 'string' }, stable: { type: 'boolean' }, reason: { type: 'string' } },
}

phase('Design'); phase('Verify')
const worked = await pipeline(CANDS,
  (c) => agent(
    'ultracode: Author an add-only, CPU-only characterization test that ADDS whole-project coverage in\n' +
    c.target + ' (for arch-breadth candidates, via the arch-gated codegen arms reached by emitting on ' +
    c.arch + ').\nFOCUS: ' + c.feat + '\n\n' +
    'STEPS:\n' +
    '  1. ' + (c.kind === 'arch'
      ? 'Read the curated valid logic under _codegen/data/' + c.arch + '/ to learn this arch\'s valid\n' +
        '     DataType/MI(or WMMA)/feature shape, then author a rich designed config for ' + c.arch + ' that\n' +
        '     widens it (bias/activation/GSU/MI fork) while staying VALID for the arch.'
      : 'Read ' + c.target + ' to find the exact params/inputs gating the remaining arms, then author the\n' +
        '     minimal config/driver that turns them on.') + '\n' +
    '  2. Test at _codegen/test_r6_' + c.id + '_char.py (emit) or a NEW characterization/<Dir>/ test\n' +
    '     (driver/derivation). Pin ACTUAL behavior. Seed any snapshot once then confirm pass without it.\n' +
    '     pytestmark=pytest.mark.unit. Prefer assertions on stable emitted text / derived state.\n' +
    '  3. ISOLATED MEASURE into COVERAGE_FILE=$PROJ/.coverage.kept_6_' + c.id + ' ; coverage json ;\n' +
    '     measured_marginal = executed lines in ' + c.target + ' (and report total covered for arch kind).\n' +
    '  4. KEEP iff test passes (err==0, >=1 kernel for emit) AND measured_marginal >= 15. If genuinely\n' +
    '     not reachable CPU-only, kept=false + precise file:line reason (P5 ceiling evidence). Do NOT commit.\n\n' +
    SHARED,
    { label: 'design:' + c.id, phase: 'Design', schema: CAND, model: 'sonnet' }),
  (c) => c && c.kept
    ? agent(
        'ultracode: Adversarially verify the kept test at ' + c.test_path + '. Re-run that node INSIDE the\n' +
        'container TWICE (no --snapshot-update), each its own COVERAGE_FILE. Stable ONLY if both runs pass\n' +
        'identically AND any snapshot .ambr already EXISTS + is byte-identical. If the TARGET file\n' +
        'executed_lines differ between runs (concurrency=multiprocessing artifact), stable=false. Default\n' +
        'stable=false on any doubt. Return GOLD.\n\n' + SHARED,
        { label: 'verify:' + c.id, phase: 'Verify', schema: GOLD, model: 'sonnet' })
    : null)

const kept = worked.filter(Boolean)
return { kept_count: kept.length, candidates: CANDS.length, kept,
  note: 'Closing push. Driver runs the deterministic 4-process gate + commit, then P5 (>=80% gate or ceiling).' }
