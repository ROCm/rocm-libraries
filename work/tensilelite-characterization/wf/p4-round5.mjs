export const meta = {
  name: 'codegen-p4-round5',
  description: 'P4 Stage-2 round 5 (final expansion): remaining reachable KWA/KW/Solution clusters (LSU-store / fp8-GR / complex / MFMA-pack / auto-LRVW); authoring+verify only, driver gates',
  phases: [{ title: 'Design' }, { title: 'Verify' }],
}

const SHARED = [
  'ENV (paths INSIDE the container):  CON=tl-char ; PROJ=/work/projects/hipblaslt/tensilelite',
  '  Host worktree: /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage',
  '  Host project: <root>/projects/hipblaslt/tensilelite (== $PROJ). Edit on HOST; RUN in container.',
  '  Container cp312 has pytest/coverage. NEVER run the whole -m unit suite; NEVER use a Monitor.',
  '',
  'ISOLATED MEASURE (one fresh process, own COVERAGE_FILE):',
  '  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.kept_5_<ID> -w $PROJ $CON \\',
  '    pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <TEST_NODE>',
  '  then: docker exec -e COVERAGE_FILE=$PROJ/.coverage.kept_5_<ID> -w $PROJ $CON coverage json -o /tmp/<ID>.json',
  '',
  'HARD RULES:',
  '  1. ADD-ONLY. NEW files ONLY under Tensile/Tests/unit/characterization/. NEVER modify/delete an',
  '     existing file. UNIQUE basename; test file ONLY in its intended dir (a stray dup at Tests/unit/',
  '     root breaks collection: "import file mismatch"). Clean up any temp .coverage.* you create.',
  '  2. --cov=Tensile is a PATH (never Tensile.x). One COVERAGE_FILE per shard.',
  '  3. Test module MUST set pytestmark = pytest.mark.unit. CPU-only.',
  '  4. If you use a syrupy snapshot, SEED its .ambr with --snapshot-update ONCE then confirm it passes',
  '     WITHOUT --snapshot-update. NEVER leave a golden unseeded. Pure-assert tests need no snapshot.',
  '  5. Keep each input bounded (limit=8). rocisa footprint per-process.',
  '  6. NEVER push, NEVER commit. The driver gates + commits.',
  '',
  'TEMPLATE: copy _codegen/test_r2_store_char.py (emit via',
  '  `from config_harness import emit_kernels_from_config; emit_kernels_from_config(CFG, limit=8, arch=ARCH)`)',
  '  + a designed config from _codegen/data/test_data/_designed/<arch>/seed.yaml. Real shipped logic/',
  '  configs: Tensile/Tests/common/** and the tuning tree. READ the target source at the cited lines',
  '  FIRST to find the exact ProblemType/Solution keys that gate the arm; if a feature is genuinely not',
  '  CPU-emittable (needs device), set kept=false with a precise file:line reason (P5 ceiling evidence).',
].join('\n')

const CANDS = [
  { id: 'lsu_store', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 2974,
    ranges: '13214-13353, 13442-13517 (localSplitUGlobalWriteIndices)', feat:
    'LOCALSPLITU STORE. localSplitUGlobalWriteIndices runs when LocalSplitU>1 (LSU reduces partial sums in LDS then one WG writes). Author a config with LocalSplitU:[2,4] (valid MT/threads) so the LSU global-write-index path emits. Combine with a normal bf16 GEMM.' },
  { id: 'f8gr', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 2974,
    ranges: '12115-12225, 12231-12293 (fp8 global-read conversion / toF8)', feat:
    'FP8 GLOBAL-READ CONVERSION. The toF8 arms convert during global read when MacDataTypeA/B isAnyFloat8 with specific GlobalReadVectorWidth (glvw==1 vs >1) and shiftGR. Author an fp8 (F8/F8/s) GEMM and fork GlobalReadVectorWidthA/B:[1,2,4] so both the glvw==1 single-element and the wide toF8 conversion arms run.' },
  { id: 'complex', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 2974,
    ranges: '9285-9355, 9918-9990 (ComplexConjugate / SingleComplex / DoubleComplex)', feat:
    'COMPLEX GEMM. The complex-conjugate MAC arms need DataType complex (SingleComplex "c" or DoubleComplex "z") with ComplexConjugateA/B. Author a complex GEMM (gfx942) so the ccVgprs / conjugate-negation arms emit. If complex is not supported in assembly CPU-only, report kept=false with the exact reject reason.' },
  { id: 'mfmapack', arch: 'gfx942', target: 'Tensile/KernelWriter.py', miss: 1472,
    ranges: '2155-2377 (MFMA local-read pack scheduling: instPerPack / packItems)', feat:
    'MFMA PACK SCHEDULING (222-line block). The pack-into-mfma-iter scheduler runs for MFMA kernels with packed local reads — gated by PrefetchLocalRead, the data type needing packing (f16/bf16/i8 with specific LocalReadVectorWidth), and MIInputPerThread. Fork PrefetchLocalRead:[1,2], LocalReadVectorWidth:[-1,4,8], and a packable dtype so the packItems/instPerPack scheduling arms run.' },
  { id: 'autolrvw', arch: 'gfx942', target: 'Tensile/SolutionStructs/Solution.py', miss: 1253,
    ranges: '819-882 (UseSubtileImpl storeD / VW force), 3064-3130 (isAutoLRVW auto LocalReadVectorWidth)', feat:
    'AUTO-LRVW + SUBTILE STORE DERIVATION. isAutoLRVW runs when LocalReadVectorWidth==-1 (auto), and the 819-882 block when UseSubtileImpl forces VectorWidth/BufferStore. Author configs with LocalReadVectorWidth:[-1] (auto) and UseSubtileImpl/Subtile on so these derivation arms run. Use solutions_from_config (derivation), assert the derived state.' },
  { id: 'lsu_emit2', arch: 'gfx942', target: 'Tensile/Components/LocalRead.py', miss: 494,
    ranges: '117-151, 217-299 (DirectToLds / wide LRVW)', feat:
    'LOCALREAD remaining (still 53%). READ LocalRead.py 117-151 to find the gating; construct a config that actually emits the DirectToLds OR the wide-LRVW (LocalReadVectorWidth>1 with UnrollMajorLDS) arms. Confirm >=1 kernel emits and the 117-299 lines run.' },
  { id: 'streamk_fixup', arch: 'gfx942', target: 'Tensile/Components/StreamK.py', miss: 547,
    ranges: '202-291, 316-409 (fixup / partial-tile)', feat:
    'STREAMK fixup/partial-tile arms (still 69%). Author StreamK:[2] (deterministic two-tile) with a grid forcing partial tiles + StreamKAtomic variants so the fixup-loop and partial-tile-store arms run. If the remaining StreamK arms need a device grid, report kept=false with the line reason.' },
  { id: 'gwb_atomic', arch: 'gfx942', target: 'Tensile/Components/GlobalWriteBatch.py', miss: 359,
    ranges: '224-331, 459-542 (atomic / edge / remap store)', feat:
    'GLOBAL WRITE atomic/edge remaining. Fork _GlobalAccumulation/atomic store + StoreRemapVectorWidth + edge (non-multiple free dims) together so the atomic-store and edge-remap arms run.' },
  { id: 'shiftvec3', arch: 'gfx942', target: 'Tensile/Components/ShiftVectorComponents.py', miss: 188,
    ranges: '47-200, 552-784', feat:
    'SHIFTVECTOR remaining edge arms (still 64.5%). Many small per-(VW,MI) edge arms. Fork over MORE (VectorWidth x MFMA-shape) combos with non-multiple free dims so additional per-combo edge-shift arms run.' },
  { id: 'subtile_kern', arch: 'gfx950', target: 'Tensile/Components/Subtile/Kernel.py', miss: 153,
    ranges: '151-253, 500-585 (subtile kernel arms)', feat:
    'SUBTILE Kernel arms (gfx950, still 75%). Extend the subtile config with more subtile/wave-split-k variants so the Subtile/Kernel.py 151-585 arms run.' },
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
    'ultracode: Author an add-only, CPU-only characterization test that EXECUTES a currently-uncovered\n' +
    'arm of ' + c.target + ' (miss=' + c.miss + '; target ranges: ' + c.ranges + ').\n' +
    'FEATURE: ' + c.feat + '\n\n' +
    'STEPS:\n' +
    '  1. READ ' + c.target + ' at the cited lines to learn the EXACT ProblemType/Solution keys that gate\n' +
    '     the arm, then author the minimal config that turns it on.\n' +
    '  2. Test at _codegen/test_r5_' + c.id + '_char.py (emit) or a NEW characterization/<Dir>/ test\n' +
    '     (derivation). Pin ACTUAL behavior. Seed any snapshot .ambr once, then confirm pass without it.\n' +
    '     pytestmark = pytest.mark.unit. Clean up temp .coverage.* you create.\n' +
    '  3. ISOLATED MEASURE into COVERAGE_FILE=$PROJ/.coverage.kept_5_' + c.id + ' ; coverage json ;\n' +
    '     measured_marginal = count of line numbers in the target ranges now in executed_lines (real).\n' +
    '  4. KEEP iff test passes (err==0) AND measured_marginal >= 15. If the feature genuinely needs a\n' +
    '     device / cannot emit CPU-only, set kept=false + a precise file:line reason (P5 ceiling evidence).\n' +
    '     Do NOT commit. Return CAND with REAL numbers.\n\n' +
    SHARED,
    { label: 'design:' + c.id, phase: 'Design', schema: CAND, model: 'sonnet' }),
  (c) => c && c.kept
    ? agent(
        'ultracode: Adversarially verify the kept test at ' + c.test_path + '. Re-run that node INSIDE the\n' +
        'container TWICE (no --snapshot-update), each its own COVERAGE_FILE. Stable ONLY if both runs pass\n' +
        'identically AND any snapshot .ambr already EXISTS + is byte-identical both runs. IMPORTANT: if the\n' +
        'TARGET file executed_lines differ between the two runs (coverage non-determinism under\n' +
        'concurrency=multiprocessing), set stable=false. Default stable=false on any doubt. Return GOLD.\n\n' + SHARED,
        { label: 'verify:' + c.id, phase: 'Verify', schema: GOLD, model: 'sonnet' })
    : null)

const kept = worked.filter(Boolean)
return { kept_count: kept.length, candidates: CANDS.length, kept,
  note: 'FINAL expansion round. Driver runs the deterministic 4-process gate + commit, then P5 (gate/ceiling).' }
