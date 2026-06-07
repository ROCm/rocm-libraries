export const meta = {
  name: 'codegen-p4-round4',
  description: 'P4 Stage-2 round 4: advanced codegen feature families (sparse / multi-index / UseE / XCC / int8 / fp8-MX) + Solution validity; authoring+verify only, driver gates',
  phases: [{ title: 'Design' }, { title: 'Verify' }],
}

const SHARED = [
  'ENV (paths INSIDE the container):  CON=tl-char ; PROJ=/work/projects/hipblaslt/tensilelite',
  '  Host worktree: /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage',
  '  Host project: <root>/projects/hipblaslt/tensilelite (== $PROJ). Edit on HOST; RUN in container.',
  '  Container cp312 has pytest/coverage. NEVER run the whole -m unit suite; NEVER use a Monitor.',
  '',
  'ISOLATED MEASURE (one fresh process, own COVERAGE_FILE):',
  '  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.kept_4_<ID> -w $PROJ $CON \\',
  '    pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <TEST_NODE>',
  '  then: docker exec -e COVERAGE_FILE=$PROJ/.coverage.kept_4_<ID> -w $PROJ $CON coverage json -o /tmp/<ID>.json',
  '  (executed lines: files["Tensile/<target>.py"].executed_lines)',
  '',
  'HARD RULES:',
  '  1. ADD-ONLY. NEW files ONLY, under Tensile/Tests/unit/characterization/. NEVER modify/delete an',
  '     existing file. UNIQUE basename; test file ONLY in its intended dir (a stray dup at Tests/unit/',
  '     root breaks collection: "import file mismatch").',
  '  2. --cov=Tensile is a PATH (never Tensile.x => rocisa SIGABRT). One COVERAGE_FILE per shard.',
  '  3. Test module MUST set: import pytest ; pytestmark = pytest.mark.unit. CPU-only.',
  '  4. If you use a syrupy snapshot golden, SEED its .ambr with --snapshot-update ONCE, then confirm',
  '     it passes WITHOUT --snapshot-update. NEVER leave a golden unseeded (a golden with no .ambr',
  '     always fails). Pure-assert tests need no snapshot.',
  '  5. Keep each input bounded (limit=8 kernels; tight params). rocisa footprint is per-process.',
  '  6. NEVER push, NEVER commit. The driver gates + commits.',
  '',
  'TEMPLATE: copy _codegen/test_r2_store_char.py (codegen emit via',
  '  `from config_harness import emit_kernels_from_config; emit_kernels_from_config(CFG, limit=8, arch=ARCH)`)',
  '  and a designed config from _codegen/data/test_data/_designed/<arch>/seed.yaml (change ProblemType +',
  '  ForkParameters for your feature). For driver/run-path features copy test_cpu_only_switch.py.',
  '  Real shipped configs/logic: Tensile/Tests/common/** and the tuning tree. Attribution:',
  '  _codegen/attribution-{gfx942,gfx950,gfx90a}.json. READ the target source at the cited lines first',
  '  to learn the EXACT ProblemType keys / params that gate the missing arm.',
].join('\n')

// R4 — advanced feature families (the remaining KWA/KW/Solution gaps are whole features, not knobs).
const CANDS = [
  { id: 'sparse', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 3383,
    ranges: '3048-3105 (graMetadataTileAssignment), 4102-4192 (computeMetaDataSrd)', feat:
    'STRUCTURED SPARSITY. The metadata SRD / metadata-tile-assignment arms are gated by structured sparsity (ProblemType "Sparse": 1 or 2, and DirectToVgprSparseMetadata). Author a sparse GEMM ProblemType (Sparse=1, a 2:4 structured-sparse A) for gfx942/gfx950 so computeMetaDataSrd + graMetadataTileAssignment + the metadata global-read arms emit. Read KWA around 3048 and 4102 for the exact gating keys.' },
  { id: 'multisum', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 3383,
    ranges: '5023-5088 (multi summation global-read increments)', feat:
    'MULTI-INDEX SUMMATION (tensor contraction). The "other summation" / NumIndicesSummation>1 arms (graIncrements for non-innermost summation loops) need a ProblemType with >1 summation index — a tensor contraction (e.g. indices like ProblemType with multiple ContractionDims / a 4D contraction). Author one so the multi-summation global-read-increment arms emit.' },
  { id: 'usee', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 3383,
    ranges: '2970-3017 (extractPackedCoord1ToRowStart), packed-coord + E', feat:
    'UseE / AUXILIARY OUTPUT + PACKED COORDINATES. Set ProblemType UseE:True (auxiliary E output tensor) with a packed/multi-dim free index so extractPackedCoord1ToRowStart + the E-output store arms run. Combine with Bias/Gradient as in the gfx942 seed. gfx942.' },
  { id: 'xccremap', arch: 'gfx942', target: 'Tensile/KernelWriterAssembly.py', miss: 3383,
    ranges: '2398-2453 (workgroup id cluster remap via ttmp)', feat:
    'WORKGROUP XCC CLUSTER REMAP. The "Init workgroup id from ttmp with cluster remap" arm is gated by WorkGroupMappingXCC>1 (and/or a cluster/XCC mapping param). Fork WorkGroupMappingXCC:[2,8] + WorkGroupMappingXCCGroup so the ttmp-based cluster-remap WG-init arm emits. gfx942 or gfx950.' },
  { id: 'int8', arch: 'gfx942', target: 'Tensile/KernelWriter.py', miss: 1555,
    ranges: 'int8 MAC/accumulate + conversion arms', feat:
    'INT8 GEMM. DataType=int8 (I8) with int32 compute drives distinct MAC/accumulate + conversion arms vs bf16/fp16. Author an I8 GEMM (gfx942) with HighPrecisionAccumulate; add a fork over GSU/bias for breadth.' },
  { id: 'fp8mx', arch: 'gfx950', target: 'Tensile/KernelWriterAssembly.py', miss: 3383,
    ranges: 'MX scale / swizzle arms (gfx950)', feat:
    'FP8 / MX SCALED (gfx950). MXFP data types with ScaleA/ScaleB/MXScale + HasMXScaleSwizzle drive gfx950-specific scale/swizzle emit arms. Author an MXFP8 GEMM with block scaling for gfx950 (see _designed/gfx950 + attribution-gfx950 MX/LSU_MX entries).' },
  { id: 'solvalidity', arch: 'gfx942', target: 'Tensile/SolutionStructs/Solution.py', miss: 1292,
    ranges: '500-680, 702-900, 1024-1300 (derivation + validity-reject arms)', feat:
    'SOLUTION VALIDITY / DERIVATION ARMS. Many missing lines are validity-REJECT paths (a param combo that the deriver rejects) and assignment branches. Author a config whose ForkParameters includes a MIX of valid AND knowingly-invalid combos (e.g. DepthU/VectorWidth/GSU/WorkGroupMapping combos that fail validity) so assignProblemIndependentDerivedParameters + the reject/return arms run. Pin the actual accepted/rejected outcome (err is fine for rejected). This is a derivation test, not necessarily a kernel emit — you can call the Solution path via config_harness.solutions_from_config and assert on the derived/rejected set.' },
  { id: 'kwreduce', arch: 'gfx942', target: 'Tensile/KernelWriter.py', miss: 1555,
    ranges: 'reduction + conversion + beta-only helper kernels', feat:
    'REDUCTION / CONVERSION / BETA-ONLY helper kernels. Mixed in/out dtype (e.g. fp8 in / fp32 out) + GlobalSplitU>1 (reduction) + UseBeta triggers KernelWriterReduction / KernelWriterConversion / KernelWriterBetaOnly emit. Use emit_helpers_from_logic or a config that needs these helpers; assert the helper kernels emit.' },
  { id: 'activation2', arch: 'gfx942', target: 'Tensile/Activation.py', miss: 279,
    ranges: '489-617, 639-718, 727-836, 852-936', feat:
    'REMAINING ACTIVATION FUNCTIONS. Fork ActivationType over the functions not yet covered (e.g. gelu, geluscaling, clippedrelu, leakyrelu, abs, sigmoid, tanh, erf, exp, silu, swish) individually + ActivationComputeDataType variants (f16/bf16/f32). Each unique function = a distinct emit arm.' },
  { id: 'shiftvector2', arch: 'gfx942', target: 'Tensile/Components/ShiftVectorComponents.py', miss: 188,
    ranges: '47-200, 552-784', feat:
    'SHIFT-VECTOR EDGE ARMS. Use free dims NOT a multiple of the macro-tile/vector-width across SEVERAL VectorWidth values [1,2,4] and MFMA shapes so the per-VW per-MI edge-shift arms (47-200 and 552-784) all run.' },
  { id: 'streamk3', arch: 'gfx942', target: 'Tensile/Components/StreamK.py', miss: 548,
    ranges: '78-153, 202-291, 316-409', feat:
    'STREAMK remaining arms. The fixup / partial-tile / two-tile arms still uncovered. Fork StreamK:[2,3] with a grid that forces partial tiles (K large, few WGs) + StreamKAtomic:[0,1]; vary the workspace/fixup params. gfx942.' },
  { id: 'localread3', arch: 'gfx942', target: 'Tensile/Components/LocalRead.py', miss: 497,
    ranges: '117-151, 217-299, 362-365', feat:
    'LOCALREAD DirectToLds (R2/R3 retries underdelivered). READ Components/LocalRead.py 117-151 to find the EXACT gating (DirectToLds + UnrollMajorLDS + specific MI/DepthU). Construct a VALID DirectToLds config that actually emits (confirm >=1 kernel). If DTL truly cannot emit CPU-only, sweep the non-DTL LocalRead arms (transpose/pad/PLR) instead and report which.' },
  { id: 'asmaddr2', arch: 'gfx942', target: 'Tensile/AsmAddressCalculation.py', miss: 166,
    ranges: '64-bit / edge / multi-batch addressing', feat:
    'ADDRESS CALC remaining arms. Large tensors (>2^31 elems → 64-bit addressing), multiple batch dims, and edge addressing drive the uncovered AsmAddressCalculation arms. Author a config with big free/batch dims so these run.' },
  { id: 'convbias', arch: 'gfx942', target: 'Tensile/KernelWriter.py', miss: 1555,
    ranges: 'gradient / conv-like / multi-bias arms', feat:
    'GRADIENT / MULTI-BIAS arms. Gradient=True with multiple Bias source dims + ActivationType + UseScaleAB drives KW arms not in current seeds. Author a backward/gradient GEMM with bias+scale+activation. gfx942.' },
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
    'ADVANCED FEATURE arm of ' + c.target + ' (miss=' + c.miss + '; target ranges: ' + c.ranges + ').\n' +
    'FEATURE: ' + c.feat + '\n\n' +
    'STEPS:\n' +
    '  1. READ ' + c.target + ' at the cited line ranges to learn the EXACT ProblemType keys / params\n' +
    '     that gate this arm (these are real Tensile features — sparsity, multi-summation, UseE, XCC,\n' +
    '     int8, MX-scale, validity-reject, etc.). Then author the minimal config that turns the feature ON.\n' +
    '  2. Test at _codegen/test_r4_' + c.id + '_char.py (emit) or characterization/<Dir>/ (driver). Pin\n' +
    '     ACTUAL behavior (emit OR a deliberate validity-reject is valid — assert it). Seed any snapshot\n' +
    '     (.ambr) once with --snapshot-update, then confirm it passes without. pytestmark=pytest.mark.unit.\n' +
    '  3. ISOLATED MEASURE into COVERAGE_FILE=$PROJ/.coverage.kept_4_' + c.id + ' ; coverage json ;\n' +
    '     measured_marginal = count of line numbers in the target ranges now in executed_lines (real count).\n' +
    '  4. KEEP iff the test passes (err==0) AND measured_marginal >= 15. If the feature genuinely cannot be\n' +
    '     turned on CPU-only (e.g. needs device), set kept=false + a precise note (this becomes P5 ceiling\n' +
    '     evidence with the file:line and the reason). Try valid variations before giving up. Do NOT commit.\n\n' +
    SHARED,
    { label: 'design:' + c.id, phase: 'Design', schema: CAND, model: 'sonnet' }),
  (c) => c && c.kept
    ? agent(
        'ultracode: Adversarially verify the kept test at ' + c.test_path + '. Re-run that node INSIDE the\n' +
        'container TWICE (no --snapshot-update), each its own COVERAGE_FILE. Stable ONLY if both runs pass\n' +
        'identically AND any snapshot .ambr already EXISTS and is byte-identical both runs. Default\n' +
        'stable=false on any doubt. Return GOLD{test_path,stable,reason}.\n\n' + SHARED,
        { label: 'verify:' + c.id, phase: 'Verify', schema: GOLD, model: 'sonnet' })
    : null)

const kept = worked.filter(Boolean)
return { kept_count: kept.length, candidates: CANDS.length, kept,
  note: 'Driver runs the deterministic methodology-A gate + commit next; kept[] with stable=true are ready.' }
