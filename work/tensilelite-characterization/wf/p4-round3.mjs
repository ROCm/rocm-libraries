export const meta = {
  name: 'codegen-p4-round3',
  description: 'P4 Stage-2 round 3: deeper codegen + client/run path (switch-enabled); authoring+verify only, driver gates',
  phases: [{ title: 'Design' }, { title: 'Verify' }],
}

const SHARED = [
  'ENV (paths INSIDE the container):  CON=tl-char ; PROJ=/work/projects/hipblaslt/tensilelite',
  '  Host worktree: /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage',
  '  Host project: <root>/projects/hipblaslt/tensilelite (== $PROJ). Edit on HOST; RUN in container.',
  '  Container cp312 has pytest/coverage. NEVER run the whole -m unit suite; NEVER use a Monitor/',
  '  background — only your OWN new test node(s), bounded docker exec.',
  '',
  'ISOLATED MEASURE (one fresh process, own COVERAGE_FILE):',
  '  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.kept_3_<ID> -w $PROJ $CON \\',
  '    pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <TEST_NODE>',
  '  then:  docker exec -e COVERAGE_FILE=$PROJ/.coverage.kept_3_<ID> -w $PROJ $CON \\',
  '    coverage json -o /tmp/<ID>.json   (executed lines: files["Tensile/<target>.py"].executed_lines)',
  '',
  'HARD RULES:',
  '  1. ADD-ONLY. NEW files ONLY, under Tensile/Tests/unit/characterization/. NEVER modify/delete an',
  '     existing file. UNIQUE basename; never a stray copy outside the intended dir (a duplicate basename',
  '     at Tests/unit/ root breaks collection with "import file mismatch").',
  '  2. --cov=Tensile is a PATH (never Tensile.x => rocisa SIGABRT). One COVERAGE_FILE per shard.',
  '  3. Test module MUST set: import pytest ; pytestmark = pytest.mark.unit. CPU-only.',
  '  4. If your test uses a syrupy snapshot golden, you MUST seed it: run the node ONCE with',
  '     --snapshot-update so __snapshots__/<file>.ambr exists, THEN confirm it passes WITHOUT',
  '     --snapshot-update. A golden test with no .ambr always fails — that is the #1 R2 defect; do not',
  '     repeat it. (For pure behavior tests with plain asserts, no snapshot is needed.)',
  '  5. Keep each input bounded (limit kernels; tight params) — rocisa footprint per-process.',
  '  6. NEVER push, NEVER commit. The driver gates + commits.',
  '',
  'TWO TEST PATTERNS (pick the one matching your target):',
  '  (A) CODEGEN EMIT (KernelWriter*, Components/*, Solution derivation): copy a designed seed config',
  '      _codegen/data/test_data/_designed/<arch>/seed.yaml -> _designed/<arch>/<id>.yaml, change ONLY',
  '      ForkParameters to your knob sweep; test at _codegen/test_r3_<id>_char.py using',
  '      `from config_harness import emit_kernels_from_config; emit_kernels_from_config(CFG, limit=8,',
  '      arch=ARCH)`; assert >=1 kernel + err==0; {basename,err} snapshot (SEED it per rule 4). Copy',
  '      _codegen/test_r2_store_char.py as a known-good template.',
  '  (B) DRIVER / RUN PATH (ClientWriter, TensileCreateLibrary/Run, LibraryLogic): the --cpu-only switch',
  '      is present on this branch. Copy the pattern from Tensile/Tests/unit/test_cpu_only_switch.py',
  '      (it calls `Tensile.Tensile([cfg, out, "--cpu-only", "--gpu-targets", arch, ...])` end-to-end,',
  '      and sets env CU=304 for the GPU-less LibraryLogic CU probe). Put your NEW test in a NEW subdir',
  '      under characterization/ (e.g. characterization/ClientPath/test_<id>_char.py) with',
  '      pytestmark=pytest.mark.unit. Pin ACTUAL artifacts/return (a results .csv, a 3_LibraryLogic',
  '      yaml, a written client source) — no snapshot needed. These exercise the run-orchestration',
  '      CPU-only. Use a TINY config (1 small problem) to stay fast.',
  '',
  'NOTES: some exotic params yield 0 valid solutions (filtered) -> 0 kernels; try another valid value of',
  '  the SAME knob (cheapest-first) before giving up. Pin ACTUAL behavior; never edit source. Attribution:',
  '  _codegen/attribution-{gfx942,gfx950,gfx90a}.json. Real shipped logic/configs: Tensile/Tests/common/**',
  '  and the tuning tree.',
].join('\n')

// R3 candidates — fresh ranges from coverage/p4/master-baseline-R2.txt (the 72.53% baseline).
const CANDS = [
  { id: 'clientwriter', arch: 'gfx942', pat: 'B', target: 'Tensile/ClientWriter.py', miss: 221,
    ranges: '94-206,213-242,265-313,366-380', knob:
    'Drive the client-writer path CPU-only: Tensile.Tensile([cfg,out,"--cpu-only","--gpu-targets",gfx942,...]) on a tiny GEMM config (set env CU=304). ClientWriter.runClient/writeClientConfig/getClientExecutablePath run under the switch. Pin the written client artifacts.' },
  { id: 'createlibrun', arch: 'gfx942', pat: 'B', target: 'Tensile/TensileCreateLibrary/Run.py', miss: 275,
    ranges: '152-173,529-619,754-873,881-1086', knob:
    'Drive TensileCreateLibrary main CPU-only on a tiny in-tree logic YAML (the 754-1086 block is the main orchestration: codegen->cross-compile->package). Use --cpu-only. Pin the produced library artifacts. See test_cpu_only_switch.py keep-build-tmp/e2e tests for the invocation.' },
  { id: 'librarylogic', arch: 'gfx942', pat: 'B', target: 'Tensile/LibraryLogic.py', miss: 535,
    ranges: '112-169,419-455,477-538,552-632,671-758,782-1017,1024-1141,1215-1424', knob:
    'Drive LibraryLogic analysis/selection (the 782-1017 & 1024-1141 blocks = the solution-selection analysis). Run the LibraryLogic step on a small benchmark-data CSV + logic set (the addFromCSV / createLibraryLogic path), or Tensile.Tensile --cpu-only end-to-end which produces 3_LibraryLogic. Pin the analysis output.' },
  { id: 'solution', arch: 'gfx942', pat: 'A', target: 'Tensile/SolutionStructs/Solution.py', miss: 1328,
    ranges: '500-680,702-900,1024-1300', knob:
    'Solution derivation arms (R2 missed these). assignProblemIndependentDerivedParameters / validity branches. Fork DepthU:[8,16,32,64], StaggerU:[0,32], PrefetchAcrossPersistent:[0,1], 1LDSBuffer:[0,1], ExpandPointerSwap:[0,1], ScheduleIterAlg:[1,2,3] — combos that drive distinct derivation/validity paths (accept some rejections; the validity-reject code IS a target).' },
  { id: 'localread', arch: 'gfx942', pat: 'A', target: 'Tensile/Components/LocalRead.py', miss: 521,
    ranges: '117-151,217-225,265-299,362-365', knob:
    'LocalRead/DirectToLds (R2 attempt failed to emit). Find a VALID DirectToLds config: DirectToLds needs specific MatrixInstruction + transpose + DepthU. Try TLU combos with DirectToLdsA/B:[True] and UnrollMajorLDSA/B; if DTL rejects, sweep PrefetchLocalRead/ClusterLocalRead/LdsBlockSizePerPad which also drive LocalRead arms. Confirm >=1 kernel emits.' },
  { id: 'streamk', arch: 'gfx942', pat: 'A', target: 'Tensile/Components/StreamK.py', miss: 685,
    ranges: '78-153,202-291,316-409', knob:
    'Deeper StreamK (R2 got 883->685). Fork StreamK:[1,2,3], StreamKAtomic:[0,1], StreamKXCCMapping, and a two-tile grid (large K, small MT) so the StreamK partial-tile / fixup arms run.' },
  { id: 'globalwrite', arch: 'gfx942', pat: 'A', target: 'Tensile/Components/GlobalWriteBatch.py', miss: 518,
    ranges: '224-331,374-405,459-542', knob:
    'Deeper global write (R2 got 787->518). Fork StoreVectorWidth:[1,2,4], StoreRemapVectorWidth:[0,2,4], atomic/_GlobalAccumulation, edge stores (non-multiple free dims), bias+ScaleAlphaVec+activation on store together.' },
  { id: 'kwafeat', arch: 'gfx942', pat: 'A', target: 'Tensile/KernelWriterAssembly.py', miss: 3558,
    ranges: 'broad feature surface', knob:
    'Broad KWA feature breadth. Fork PrefetchAcrossPersistent:[0,1], PersistentKernel, ScheduleIterAlg:[1,2,3], 1LDSBuffer:[0,1], ExpandPointerSwap, StaggerU:[0,32], WorkGroupMapping, ScheduleGlobalRead/LocalWrite — combos that toggle distinct KWA emit arms. Aim for several distinct kernels.' },
  { id: 'kwfeat', arch: 'gfx942', pat: 'A', target: 'Tensile/KernelWriter.py', miss: 1664,
    ranges: 'broad', knob:
    'Broad KernelWriter breadth. Vary ProblemType to trigger beta-only + reduction + conversion helper kernels (mixed in/out dtype + GSU>1), and ScheduleIterAlg / PrefetchGlobalRead combos for the main KW arms. Use emit_kernels_from_config + emit_helpers if useful.' },
  { id: 'lra', arch: 'gfx942', pat: 'A', target: 'Tensile/Components/LraTileAssignment.py', miss: 279,
    ranges: '144-249,285-409,473-592,611-693', knob:
    'LRA tile assignment (R2 barely moved it). The 144-693 block = per-transpose tile-assignment algos. Fork TransposeA/B (all 4 TLU combos), LdsBlockSizePerPadA/B:[-1,128], UnrollMajorLDSA/B, MIWaveGroup/MIWaveTile shapes.' },
  { id: 'gsu', arch: 'gfx942', pat: 'A', target: 'Tensile/Components/GSU.py', miss: 257,
    ranges: '170-233,388-461,480-588', knob:
    'GSU arms (R2 missed). The 442-588 block needs GlobalSplitU>1 with the reduction/workspace path actually emitted. Fork GlobalSplitU:[2,4,8], GlobalSplitUAlgorithm:[SingleBuffer,MultipleBuffer], GlobalSplitUWorkGroupMappingRoundRobin, GlobalSplitUCoalesced — and a ProblemType where GSU reduction is valid.' },
  { id: 'subtile', arch: 'gfx950', pat: 'A', target: 'Tensile/Components/Subtile/SubtileLREmit.py', miss: 67,
    ranges: '490-535,613-655', knob:
    'Deeper Subtile (gfx950) covering SubtileLREmit + Subtile/Kernel + SubtileScaleEmit. Extend the R2 subtile config with scale/LR variants so the subtile local-read + scale-emit arms run. Target SubtileLREmit/SubtileScaleEmit/Kernel together.' },
  { id: 'addrstore', arch: 'gfx942', pat: 'A', target: 'Tensile/AsmAddressCalculation.py', miss: 182,
    ranges: 'addressing arms', knob:
    'Address calc + store state. Fork to drive edge/large-tensor addressing (big free dims, 64-bit addressing, multiple batch dims, GlobalAccess vector widths) so AsmAddressCalculation + AsmStoreState arms run. Pin >=1 kernel.' },
  { id: 'datamover', arch: 'gfx942', pat: 'A', target: 'Tensile/Components/TensorDataMover.py', miss: 158,
    ranges: 'DTV/DTL mover arms', knob:
    'TensorDataMover (158, 58%). Fork DirectToVgprA/B:[True], DirectToLds, and global-read vector widths so the alternate data-mover code paths run vs the default mover.' },
]

const CAND = {
  type: 'object', additionalProperties: false,
  required: ['id', 'test_path', 'pattern', 'emitted', 'err', 'measured_marginal', 'kept', 'note'],
  properties: {
    id: { type: 'string' }, test_path: { type: 'string' },
    pattern: { enum: ['A', 'B'] },
    emitted: { type: 'integer' },               // kernels emitted (pattern A) or 1 if driver flow completed (pattern B)
    err: { type: 'integer' },                    // pytest exit code of the isolated run (0=passed)
    measured_marginal: { type: 'integer' },      // count of target's previously-missing lines now executed
    kept: { type: 'boolean' },                   // true iff test passes (err==0) AND measured_marginal >= 15
    note: { type: 'string' },
  },
}
const GOLD = {
  type: 'object', additionalProperties: false, required: ['test_path', 'stable', 'reason'],
  properties: { test_path: { type: 'string' }, stable: { type: 'boolean' }, reason: { type: 'string' } },
}

phase('Design'); phase('Verify')
const worked = await pipeline(CANDS,
  (c) => agent(
    'ultracode: Author an add-only, CPU-only characterization test that EXECUTES the uncovered lines\n' +
    'of ' + c.target + ' (miss=' + c.miss + '; methodology-A missing ranges: ' + c.ranges + ').\n' +
    'Use TEST PATTERN ' + c.pattern + ' (A=codegen emit, B=driver/run path). Approach: ' + c.knob + '\n\n' +
    'STEPS:\n' +
    '  1. Read the target source ' + c.target + ' to see exactly which params/inputs gate the missing\n' +
    '     lines. For pattern A copy test_r2_store_char.py + a _designed/<arch>/' + c.id + '.yaml; for\n' +
    '     pattern B copy the test_cpu_only_switch.py invocation into a NEW characterization/<Dir>/ test.\n' +
    '  2. Pin ACTUAL behavior. If you use a syrupy snapshot, SEED its .ambr with --snapshot-update once,\n' +
    '     then confirm it passes WITHOUT --snapshot-update (rule 4 — do not leave a golden unseeded).\n' +
    '  3. ISOLATED MEASURE into COVERAGE_FILE=$PROJ/.coverage.kept_3_' + c.id + ' ; coverage json ;\n' +
    '     measured_marginal = count of line numbers in [' + c.ranges + '] of ' + c.target + ' now in\n' +
    '     executed_lines (docker exec python -c to intersect — report the REAL count).\n' +
    '  4. KEEP iff the test passes (err==0, and pattern A emitted >=1 kernel) AND measured_marginal >= 15.\n' +
    '     If a sweep emits nothing valid, try another valid value of the SAME knob first; if still nothing,\n' +
    '     kept=false + note (ceiling evidence). Do NOT commit. Return CAND with REAL numbers.\n\n' +
    SHARED,
    { label: 'design:' + c.id, phase: 'Design', schema: CAND, model: 'sonnet' }),
  (c) => c && c.kept
    ? agent(
        'ultracode: Adversarially verify the kept test at ' + c.test_path + '. Re-run that node INSIDE\n' +
        'the container TWICE (no --snapshot-update), each with its own COVERAGE_FILE. Stable ONLY if both\n' +
        'runs pass with identical outcome AND any syrupy snapshot is byte-identical both runs (and the\n' +
        '.ambr already EXISTS — a missing snapshot = NOT stable). Default stable=false on ANY doubt.\n' +
        'Return GOLD{test_path,stable,reason}.\n\n' + SHARED,
        { label: 'verify:' + c.id, phase: 'Verify', schema: GOLD, model: 'sonnet' })
    : null)

const kept = worked.filter(Boolean)
return { kept_count: kept.length, candidates: CANDS.length, kept,
  note: 'Driver runs the deterministic methodology-A gate + commit next; commit only kept[] with stable=true.' }
