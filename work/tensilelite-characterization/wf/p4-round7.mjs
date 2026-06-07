export const meta = {
  name: 'codegen-p4-round7',
  description: 'P4 Stage-2 round 7 (final): big reachable clusters LocalRead/StreamK/Run/ShiftVector/GSU/Solution-edges; relaxed pass/fail verify (coverage jitter tolerated); driver gates',
  phases: [{ title: 'Design' }, { title: 'Verify' }],
}

const SHARED = [
  'ENV (paths INSIDE the container):  CON=tl-char ; PROJ=/work/projects/hipblaslt/tensilelite',
  '  Host worktree: /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage',
  '  Host project: <root>/projects/hipblaslt/tensilelite (== $PROJ). Edit on HOST; RUN in container.',
  '  Container cp312 has pytest/coverage. NEVER run the whole -m unit suite; NEVER use a Monitor.',
  '',
  'ISOLATED MEASURE (one fresh process, own COVERAGE_FILE):',
  '  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.kept_7_<ID> -w $PROJ $CON \\',
  '    pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <TEST_NODE>',
  '  then: docker exec -e COVERAGE_FILE=$PROJ/.coverage.kept_7_<ID> -w $PROJ $CON coverage json -o /tmp/<ID>.json',
  '',
  'HARD RULES:',
  '  1. ADD-ONLY. NEW files ONLY under Tensile/Tests/unit/characterization/. NEVER modify/delete an',
  '     existing file. UNIQUE basename; test ONLY in its intended dir. Clean up temp .coverage.* you make.',
  '  2. --cov=Tensile is a PATH. One COVERAGE_FILE per shard. pytestmark=pytest.mark.unit. CPU-only.',
  '  3. If you use a syrupy snapshot, SEED its .ambr once with --snapshot-update then confirm pass without.',
  '     PREFER pure-assert tests (assert on emitted text / derived state), no snapshot — more robust to the',
  '     coverage concurrency=multiprocessing jitter.',
  '  4. Keep each input bounded (limit=8). NEVER push, NEVER commit. The driver gates + commits.',
  '',
  'NOTE ON COVERAGE JITTER: codegen runs through multiprocessing workers, so a target file\'s executed_lines',
  '  can vary run-to-run (a coverage.py artifact). This is FINE for this round — the driver\'s full-suite gate',
  '  combines all workers and counts any line covered. Your job: make a test that PASSES deterministically and',
  '  EXECUTES the target cluster at least once. Do not worry if the isolated line count jitters.',
  '',
  'TEMPLATE: copy _codegen/test_r2_store_char.py (emit via',
  '  `from config_harness import emit_kernels_from_config; emit_kernels_from_config(CFG, limit=8, arch=ARCH)`),',
  '  or test_cpu_only_switch.py for driver/run-path. Curated logic: _codegen/data/<arch>/*.yaml. READ the',
  '  target source at the cited lines FIRST to find the exact gating params. If a feature truly needs a',
  '  device (e.g. StreamK fixup needs a multi-WG grid), report kept=false + a precise file:line reason',
  '  (P5 ceiling evidence) rather than forcing it.',
].join('\n')

const CANDS = [
  { id: 'localread_dtl', arch: 'gfx942', target: 'Tensile/Components/LocalRead.py', miss: 489,
    ranges: '785-942, 1090-1135, 1164-1320, 1346-1445', feat:
    'LocalRead big clusters (489 miss, 54%). These are the DirectToLds / wide-LocalReadVectorWidth / transpose-xor / ds-read-conv arms. READ LocalRead.py 785-942 and 1164-1445 to find the gate (DirectToLds + UnrollMajorLDS + LocalReadVectorWidth>1 + transpose). Author 2-3 configs covering DTL on + wide-LRVW + the transpose variants so these arms execute. Pure-assert on emitted ds_read / lds instructions.' },
  { id: 'streamk_grid', arch: 'gfx942', target: 'Tensile/Components/StreamK.py', miss: 547,
    ranges: '630-784, 2915-3091, 3142-3215', feat:
    'StreamK big clusters (547 miss). READ StreamK.py 630-784 and 2915-3091 to see what gates them (the fixup/partial-tile/workspace-accumulate arms). Try StreamK:[2,3] with a small grid (ProblemSizes that force >1 tile per WG and partial tiles) + StreamKAtomic + the workspace path. If the fixup arm genuinely requires a runtime multi-WG device grid (cannot be reached by pure codegen emit), report kept=false with the exact file:line reason.' },
  { id: 'createlib_deep', arch: 'gfx942', target: 'Tensile/TensileCreateLibrary/Run.py', miss: 197,
    ranges: '152-173, 529-619, 754-873, 881-1086', feat:
    'TensileCreateLibrary/Run remaining (197, 63%). Drive more --cpu-only TensileCreateLibrary scenarios (different --code-object-version, --library-format yaml vs msgpack, multiple archs in one run, lazy-loading, separate-architectures, --no-merge-files vs merge) so the orchestration arms 754-1086 run. Copy test_cpu_only_switch.py / TensileCreateLibraryRun patterns. Pure-assert on produced artifacts.' },
  { id: 'shiftvec_full', arch: 'gfx942', target: 'Tensile/Components/ShiftVectorComponents.py', miss: 188,
    ranges: '47-200, 552-784', feat:
    'ShiftVector edge arms (188, 64%). The 47-200 and 552-784 blocks are per-(VectorWidth, MFMA-shape, edge) shift emits. Author a config sweeping VectorWidth:[1,2,4] x 2-3 MFMA shapes with free dims NON-multiple of the macro tile (forces edge shift) so many per-combo arms run.' },
  { id: 'gsu_reduce', arch: 'gfx942', target: 'Tensile/Components/GSU.py', miss: 238,
    ranges: '170-233, 388-461, 442-588', feat:
    'GSU remaining reduction/workspace arms (238, 77%). gsu3 (MultipleBufferSingleKernel) was redundant; try the OTHER GSU algorithms whose emit differs: GlobalSplitUAlgorithm SingleBuffer + MultipleBuffer with GlobalSplitU>1 and the workspace-accumulate / final-reduction emit. READ GSU.py 442-588 for the gate. If it needs a separate reduction kernel the harness cannot emit, report kept=false + reason.' },
  { id: 'asmaddr_64', arch: 'gfx942', target: 'Tensile/AsmAddressCalculation.py', miss: 163,
    ranges: '64-bit / multi-batch / edge addressing', feat:
    'AsmAddressCalculation (163, 60%). Author a config with multiple batch dims + large free dims (forces 64-bit / multi-dim address arms) + edge so the remaining address-calc branches run. Pure-assert on emitted address instructions.' },
  { id: 'sol_edges', arch: 'gfx942', target: 'Tensile/SolutionStructs/Solution.py', miss: 1121,
    ranges: '2282-2318, 3491-3532, 3697-3727, 3748-3792', feat:
    'Solution derivation edges (1121 miss, 63%). These specific clusters are distinct derivation/validity arms not yet hit. READ Solution.py at each range to find the gating param, then a ForkParameters set (or solutions_from_config with varied params) that triggers them. Assert on derived/rejected state. Many are validity-reject branches (err on reject is fine to pin).' },
  { id: 'kwa_remain', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 2767,
    ranges: '9921-9990, 10055-10108, 2970-3017', feat:
    'KWA remaining mid clusters. 9921-9990/10055-10108 are complex-conjugate global-read-increment / complex MAC follow-ons (extend the R5 complex test with ComplexConjugateA AND B + complex global-read-incs); 2970-3017 is extractPackedCoord1ToRowStart (UseE + packed multi-dim free index). Author configs to hit these.' },
  { id: 'benchproblems2', arch: 'gfx942', target: 'Tensile/BenchmarkProblems.py', miss: 111,
    ranges: '182-325, 431-452, 543-594', feat:
    'BenchmarkProblems remaining (111, 65%). Drive the BenchmarkProcess -> constructForkPermutations -> _generateForkedSolutions path with configs exercising the un-run derivation/cache/custom-kernel arms (multiple ProblemSizeGroups, custom kernels, the cache-write/compare path). Use config_harness.solutions_from_config breadth.' },
  { id: 'subtile_sched', arch: 'gfx950', target: 'Tensile/Components/Subtile/LogicalScheduler.py', miss: 151,
    ranges: 'scheduler arms', feat:
    'Subtile LogicalScheduler (151, 88%) + Subtile/Kernel (128). Extend the gfx950 subtile configs with more scheduler-exercising variants (different subtile shapes / wave-split-k / scheduling modes) so the remaining LogicalScheduler + Kernel arms run.' },
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
    'ultracode: Author an add-only, CPU-only characterization test that EXECUTES the uncovered cluster of\n' +
    c.target + ' (miss=' + c.miss + '; target ranges: ' + c.ranges + ').\n' +
    'FEATURE: ' + c.feat + '\n\n' +
    'STEPS:\n' +
    '  1. READ ' + c.target + ' at the cited ranges to find the exact gating params, then author the\n' +
    '     minimal config(s)/driver that turn the arm on.\n' +
    '  2. Test at _codegen/test_r7_' + c.id + '_char.py (emit) or a NEW characterization/<Dir>/ test\n' +
    '     (driver/derivation). PREFER pure-assert (robust to coverage jitter). pytestmark=pytest.mark.unit.\n' +
    '  3. ISOLATED MEASURE into COVERAGE_FILE=$PROJ/.coverage.kept_7_' + c.id + ' ; coverage json ;\n' +
    '     measured_marginal = count of target-range lines now in executed_lines (report the max you observe).\n' +
    '  4. KEEP iff the test PASSES (err==0, >=1 kernel for emit) AND measured_marginal >= 15. If the arm\n' +
    '     genuinely needs a device, kept=false + precise file:line reason (P5 ceiling evidence). Do NOT commit.\n\n' +
    SHARED,
    { label: 'design:' + c.id, phase: 'Design', schema: CAND, model: 'sonnet' }),
  (c) => c && c.kept
    ? agent(
        'ultracode: Verify the kept test at ' + c.test_path + '. Re-run that node INSIDE the container TWICE\n' +
        '(no --snapshot-update). RELAXED CRITERION: stable=true iff BOTH runs PASS with identical pass/fail\n' +
        'outcome (same tests pass) AND any syrupy .ambr exists + is byte-identical. DO NOT set stable=false\n' +
        'merely because the TARGET file executed_lines differ run-to-run (that is the known coverage\n' +
        'concurrency=multiprocessing artifact and is acceptable this round — the driver gate combines workers).\n' +
        'Only fail on real flakiness (a test that passes one run and fails the next, or a churning snapshot).\n' +
        'Return GOLD{test_path,stable,reason}.\n\n' + SHARED,
        { label: 'verify:' + c.id, phase: 'Verify', schema: GOLD, model: 'sonnet' })
    : null)

const kept = worked.filter(Boolean)
return { kept_count: kept.length, candidates: CANDS.length, kept,
  note: 'FINAL expansion round (relaxed pass/fail verify). Driver runs 4-process gate, drops post-gate 0-marginal, commits, then P5 (>=80% or ceiling).' }
