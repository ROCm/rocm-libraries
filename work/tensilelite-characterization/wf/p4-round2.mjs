export const meta = {
  name: 'codegen-p4-round2',
  description: 'P4 Stage-2 round 2: codegen emit widening (ForkParameters sweeps over uncovered codegen knobs); authoring+verify only, driver gates',
  phases: [{ title: 'Design' }, { title: 'Verify' }],
}

// Driver runs the methodology-A gate + commit (NOT this workflow) — workflow agents
// must NEVER run the full -m unit suite / a Monitor; they only author + isolated-measure + verify.
const SHARED = [
  'ENV (paths INSIDE the container):  CON=tl-char ; PROJ=/work/projects/hipblaslt/tensilelite',
  '  Host worktree: /home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage',
  '  Host project: <root>/projects/hipblaslt/tensilelite (== $PROJ). Edit on HOST; RUN in container.',
  '  Container cp312 has pytest/coverage. NEVER run the whole -m unit suite; NEVER use a Monitor/',
  '  background — only your OWN new test node, one bounded docker exec.',
  '',
  'ISOLATED MEASURE (one fresh process, own COVERAGE_FILE):',
  '  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.kept_2_<ID> -w $PROJ $CON \\',
  '    pytest -p no:cacheprovider -m unit --cov=Tensile --cov-config=pyproject.toml -q <TEST_NODE>',
  '  then:  docker exec -e COVERAGE_FILE=$PROJ/.coverage.kept_2_<ID> -w $PROJ $CON \\',
  '    coverage json -o /tmp/<ID>.json   (executed lines: files["Tensile/<target>.py"].executed_lines)',
  '',
  'HARD RULES:',
  '  1. ADD-ONLY. NEW files ONLY, under Tensile/Tests/unit/characterization/. NEVER modify/delete an',
  '     existing file. Put the test at _codegen/test_r2_<id>_char.py and the config at',
  '     _codegen/data/test_data/_designed/<arch>/<id>.yaml (the test_data path keeps it out of the',
  '     GPU findConfigs auto-discovery). Use a UNIQUE basename — never duplicate an existing test',
  '     basename, never write a test file anywhere except its _codegen home (a stray copy at',
  '     Tests/unit/ root breaks collection with "import file mismatch").',
  '  2. --cov=Tensile is a PATH (never Tensile.x => rocisa SIGABRT). One COVERAGE_FILE per shard.',
  '  3. Test module MUST set: import pytest ; pytestmark = pytest.mark.unit. CPU-only.',
  '  4. Keep each input bounded (limit=8 kernels; tight ForkParameters) — rocisa footprint per-process.',
  '  5. NEVER push, NEVER commit. The driver gates + commits.',
  '',
  'HARNESS / TEMPLATE (copy, do not reinvent):',
  '  - Emit path: from config_harness import emit_kernels_from_config ;',
  '      results = emit_kernels_from_config(CONFIG_PATH, limit=8, arch=ARCH)  -> [(basename, src, err)].',
  '  - COPY an existing designed seed YAML as your base and change ONLY the ForkParameters block to',
  '    your knob sweep (keep its GlobalParameters + a VALID dominant ProblemType):',
  '      _codegen/data/test_data/_designed/gfx942/seed.yaml  (bf16 BH+Bias+Activation, MI x DTVA fork)',
  '      _codegen/data/test_data/_designed/gfx950/seed.yaml , _designed/gfx90a/seed.yaml',
  '  - COPY an existing seed TEST as your base (assertion style + golden):',
  '      _codegen/test_seed_gfx942_char.py  (emit -> assert err==0 + real AMDGCN; snapshot {basename,err}).',
  '  - Attribution of which params move which lines: _codegen/attribution-{gfx942,gfx950,gfx90a}.json.',
  '  - Some exotic param values yield 0 valid solutions (constructForkPermutations/_generateForked-',
  '    Solutions filter invalid combos) -> 0 kernels. If your sweep emits 0 kernels OR err!=0 for all,',
  '    try a cheaper/valid value of the SAME knob family (cheapest-first); if still nothing emits,',
  '    set kept=false with a note (becomes ceiling evidence). Pin ACTUAL behavior; never edit source.',
].join('\n')

// R2 candidates: one codegen knob-family each, targeting a specific uncovered codegen file.
// missing_ranges are from the methodology-A term-missing receipt (coverage/p4/gap-by-miss.tsv).
const CANDS = [
  { id: 'streamk', arch: 'gfx942', target: 'Tensile/Components/StreamK.py',
    ranges: '202,216-243,254,272-291,316,326-409,464+', miss: 883,
    knob: 'StreamK family. Fork over StreamK:[0,1,2,3], StreamKAtomic:[0,1] (and StreamKXCCMapping if valid). StreamK splits the K loop across workgroups — drives the whole StreamK component. Pick a ProblemType where StreamK is valid (large K, single batch).' },
  { id: 'gsu', arch: 'gfx942', target: 'Tensile/Components/GSU.py',
    ranges: '170-233,246-301,388-461,480-588,646-698', miss: 257,
    knob: 'GlobalSplitU family. Fork over GlobalSplitU:[1,3,8], GlobalSplitUAlgorithm:[SingleBuffer,MultipleBuffer], GlobalSplitUWorkGroupMappingRoundRobin:[False,True]. Also lights GlobalWriteBatch GSU-store arms.' },
  { id: 'localread', arch: 'gfx942', target: 'Tensile/Components/LocalRead.py',
    ranges: '117-151,217-225,265-299,337,362-365', miss: 526,
    knob: 'LocalRead / DirectToLds. Fork over DirectToLds:[False,True], DirectToVgprA/B combos, PrefetchLocalRead:[1,2], ClusterLocalRead:[0,1], LdsBlockSizePerPadA/B.' },
  { id: 'wgm', arch: 'gfx942', target: 'Tensile/Components/WorkGroupMappingAlgos.py',
    ranges: '369-410,430-628,655-763,781-841,850-1037', miss: 364,
    knob: 'WorkGroupMapping algorithms (huge 369-1037 uncovered block = whole algos). Fork over WorkGroupMapping:[1,4,8], WorkGroupMappingXCC:[1,2,8], WorkGroupMappingXCCGroup, and any WGM-algorithm selector param. These choose the WG->tile mapping algo.' },
  { id: 'shiftvector', arch: 'gfx942', target: 'Tensile/Components/ShiftVectorComponents.py',
    ranges: '47-200,238,438-439,552-784', miss: 293,
    knob: 'Edge shift-vector path. Use NON-multiple free dims (e.g. M/N not a multiple of MacroTile) so edge shifting runs, with EdgeType/AssertFree0ElementMultiple:[1] vs unset, VectorWidth:[1,2,4]. The 47-200 & 552-784 blocks are the per-VW edge emit.' },
  { id: 'activation', arch: 'gfx942', target: 'Tensile/Activation.py',
    ranges: '489-617,639-718,727-836,852-936', miss: 302,
    knob: 'Activation variants. Fork ActivationType over INDIVIDUAL functions (gelu,relu,abs,sigmoid,tanh,clippedrelu,leakyrelu,geluscaling,...) instead of "all", plus ActivationComputeDataType variants. Each function emits a distinct code arm.' },
  { id: 'solution', arch: 'gfx942', target: 'Tensile/SolutionStructs/Solution.py',
    ranges: '500-680,702-900,1024-1300', miss: 1328,
    knob: 'Solution-derivation breadth (biggest non-KWA gap). A WIDE ForkParameters cartesian across assignment/validity arms: DepthU:[16,32,64], VectorWidth:[1,2,4], GlobalSplitU:[1,4], PrefetchGlobalRead:[0,1,2], PrefetchAcrossPersistent:[0,1], AssertSummationElementMultiple. Accept that some combos are rejected — the rejection + assignment code is the target. Keep limit modest.' },
  { id: 'store', arch: 'gfx942', target: 'Tensile/Components/GlobalWriteBatch.py',
    ranges: '287-331,374-405,459-542,568+', miss: 787,
    knob: 'Global write/store batch. Fork StoreVectorWidth:[1,2,4], _GlobalAccumulation/atomic, StoreRemapVectorWidth:[0,2,4], bias+ScaleAlphaVec on store, edge stores. Also lights AsmStoreState.' },
  { id: 'subtile', arch: 'gfx950', target: 'Tensile/Components/Subtile/SubtileGREmit.py',
    ranges: '359-411,528-639,658-834,856-903', miss: 313,
    knob: 'Subtile / WaveSplitK emit (gfx950). Enable the Subtile codegen path (Subtile/WaveSplitK/StreamK-subtile params per gfx950 seed). The 359-834 block is the subtile global-read/scale emit — needs the subtile feature ON. Consult attribution-gfx950.json (StreamK/MX entries).' },
  { id: 'kwconv', arch: 'gfx942', target: 'Tensile/KernelWriterConversion.py',
    ranges: 'beta-only + reduction + conversion arms', miss: 115,
    knob: 'Conversion/beta-only/reduction kernels. Drive a ProblemType needing a conversion kernel (mixed in/out dtype, e.g. fp8 in / bf16 out) and GSU>1 (reduction). emit_helpers_from_logic / beta-only path. Also lights KernelWriterBetaOnly + KernelWriterReduction.' },
  { id: 'lra', arch: 'gfx942', target: 'Tensile/Components/LraTileAssignment.py',
    ranges: '78-91,144-249,285-409,473-592,611-693', miss: 288,
    knob: 'LRA tile assignment. Fork transpose combos (TLU/TLDS), LdsBlockSizePerPad, MIWaveGroup/MIWaveTile shapes, UnrollMajorLDS. Different tile-assignment arms per transpose+pad.' },
  { id: 'mac', arch: 'gfx90a', target: 'Tensile/Components/MAC_F16_HPA.py',
    ranges: 'whole non-MFMA MAC path', miss: 78,
    knob: 'Non-MFMA MAC inner loop (fp16 HPA). Use a ProblemType WITHOUT MatrixInstruction (EnableMatrixInstruction:[False] / KernelLanguage Assembly, plain VALU MAC) so the MAC_F16_HPA / MAC_BF16_HPA / MAC_F16 source-emit arms run. gfx90a or gfx942.' },
]

const CAND = {
  type: 'object', additionalProperties: false,
  required: ['id', 'config_path', 'test_path', 'kernels', 'err', 'measured_marginal', 'kept', 'note'],
  properties: {
    id: { type: 'string' }, config_path: { type: 'string' }, test_path: { type: 'string' },
    kernels: { type: 'integer' }, err: { type: 'integer' },
    measured_marginal: { type: 'integer' },   // count of THIS target's previously-missing lines now executed
    kept: { type: 'boolean' },                  // true iff >=1 kernel, err==0, measured_marginal >= 15
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
    'ultracode: Author an add-only, CPU-only codegen-emit characterization test that EXECUTES the\n' +
    'uncovered branches of ' + c.target + ' (methodology-A missing ranges: ' + c.ranges + '; ' +
    c.miss + ' miss total) on ' + c.arch + '.\n' +
    'KNOB FAMILY to sweep via ForkParameters: ' + c.knob + '\n\n' +
    'STEPS:\n' +
    '  1. COPY _codegen/data/test_data/_designed/' + c.arch + '/seed.yaml to\n' +
    '     _codegen/data/test_data/_designed/' + c.arch + '/' + c.id + '.yaml ; keep GlobalParameters and a\n' +
    '     VALID dominant ProblemType; REPLACE the ForkParameters block with your knob sweep. Read the\n' +
    '     target source ' + c.target + ' to see which exact params/values gate the missing lines.\n' +
    '  2. Write _codegen/test_r2_' + c.id + '_char.py (copy test_seed_gfx942_char.py style): emit via\n' +
    '     emit_kernels_from_config(CONFIG, limit=8, arch="' + c.arch + '"); assert >=1 kernel and all\n' +
    '     err==0 (or pin the actual reject if that IS the target arm); add a {basename,err} snapshot\n' +
    '     golden test. pytestmark = pytest.mark.unit. UNIQUE basename; file ONLY in _codegen/.\n' +
    '  3. ISOLATED MEASURE into COVERAGE_FILE=$PROJ/.coverage.kept_2_' + c.id + ' ; coverage json ;\n' +
    '     measured_marginal = how many line numbers in [' + c.ranges + '] of ' + c.target + ' are now in\n' +
    '     executed_lines (run a short docker exec python -c to intersect — report the REAL count).\n' +
    '  4. KEEP iff >=1 kernel emitted AND err==0 AND measured_marginal >= 15. If a sweep emits nothing\n' +
    '     valid, try another valid value of the SAME knob (cheapest-first) before giving up; if still\n' +
    '     nothing, kept=false + note. Do NOT commit. Return CAND with REAL numbers.\n\n' +
    SHARED,
    { label: 'design:' + c.id, phase: 'Design', schema: CAND, model: 'sonnet' }),
  (c) => c && c.kept
    ? agent(
        'ultracode: Adversarially verify the kept codegen test at ' + c.test_path + '. Re-run that node\n' +
        'INSIDE the container TWICE (no --snapshot-update), each with its own COVERAGE_FILE. Stable ONLY\n' +
        'if both runs pass with identical outcome AND the {basename,err} snapshot is byte-identical both\n' +
        'runs (codegen emit is order-coupled via process-global scheduler state — watch for churn).\n' +
        'Default stable=false on ANY doubt. Return GOLD{test_path,stable,reason}.\n\n' + SHARED,
        { label: 'verify:' + c.id, phase: 'Verify', schema: GOLD, model: 'sonnet' })
    : null)

const kept = worked.filter(Boolean)
return {
  kept_count: kept.length,
  candidates: CANDS.length,
  kept,
  note: 'Driver runs the methodology-A gate + commit next. kept[] entries that also passed Verify (stable=true) are ready to commit.',
}
