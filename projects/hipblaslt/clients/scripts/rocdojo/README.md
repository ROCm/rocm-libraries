# rocDojo gfx1250 bf16 kernels: build, test, benchmark

The rocDojo custom kernels live in
`tensilelite/tensilelite/CustomKernels/rocdojo/` and their Prediction logic in
`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1250/gfx1250/Origami/*_rocdojo.yaml`
(see the custom-kernel README). The default build merges them into the stock
gfx1250 bf16 Origami libraries, so origami picks among stock and rocDojo
solutions per problem.

| hipBLASLt layout | logic file                                                     | rocDojo kernels | stock solutions |
| ---------------- | -------------------------------------------------------------- | --------------- | --------------- |
| NN               | `gfx1250_Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_rocdojo` | 129             | 425             |
| NT               | `gfx1250_Cijk_Ailk_Bjlk_BBS_BH_BiasSB_HAS_SAV_UserArgs_rocdojo` | 129             | 1136            |
| TN               | `gfx1250_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs_rocdojo` | 148             | 694             |
| TT               | `gfx1250_Cijk_Alik_Bjlk_BBS_BH_BiasSB_HAS_SAV_UserArgs_rocdojo` | 129             | none            |

rocDojo names layouts row-major, so kernel names disagree with hipBLASLt's:
`rocdojo_bf16_nt_*` kernels serve hipBLASLt TN (`--transA T --transB N`) and
`rocdojo_bf16_tn_*` serve hipBLASLt NT. NN and TT agree. Most kernels are
Stream-K with a parallel reduction (`_skp_` in the name), whose
`<kernel>_streamKReduce` post kernel lives in the same `.s`.

## Build

A full build for gfx1250 assembles the rocDojo kernels into the same code
objects as the stock ones:

```bash
cmake -B build -S . -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ -D CMAKE_C_COMPILER=$ROCM_PATH/bin/amdclang \
      -D CMAKE_PREFIX_PATH=$ROCM_PATH -D GPU_TARGETS=gfx1250
cmake --build build --parallel
```

To build only the gfx1250 bf16 Origami libraries (stock and rocDojo; much
faster than the whole gfx1250 library), add
`-D TENSILELITE_LOGIC_FILTER='gfx1250/Origami/gfx1250_Cijk_*_BBS_BH_*'`.
The library lands in `build/Tensile/library/gfx1250`.

## Three libraries from one tree

`create_library` builds a library from any subset of the logic, which gives
rocDojo-only and stock-only libraries to compare against the merged one.
From `projects/hipblaslt`, with the build's `rocisa` on the path:

```bash
B=$PWD/build T=$PWD/tensilelite L=$PWD/library/src/amd_detail/rocblaslt/src/Tensile/Logic
lib() {  # lib <name> <logic filter>
  mkdir -p /tmp/$USER/rocdojo-libs/$1.work && (cd /tmp/$USER/rocdojo-libs/$1.work &&
  PYTHONPATH=$T:$B/tensilelite/rocisa LD_LIBRARY_PATH=$B/tensilelite/rocisa/rocisa:$ROCM_PATH/lib \
  python3 -m tensilelite.create_library --architecture=gfx1250 --cxx-compiler=$ROCM_PATH/bin/amdclang++ \
    --jobs=16 --disable-asm-comments --logic-filter="$2" $L /tmp/$USER/rocdojo-libs/$1 HIP)
}
lib merged  'gfx1250/Origami/gfx1250_Cijk_*_BBS_BH_*'
lib stock   'gfx1250/Origami/gfx1250_Cijk_*_BBS_BH_*_UserArgs'
lib rocdojo '*_rocdojo'
```

The stock filter ends in `_UserArgs`, so it leaves out the `_UserArgs_rocdojo`
files. Point `HIPBLASLT_TENSILE_LIBPATH` at `/tmp/$USER/rocdojo-libs/<name>/library/gfx1250`
to use one of them; hipBLASLt takes that directory verbatim and loads nothing
else. These three hold only the bf16 Origami libraries: other data types find
no solution.

## Run

```bash
export LD_LIBRARY_PATH=$B/library:$B/tensilelite   # never /opt/rocm/lib first
export HIPBLASLT_TENSILE_LIBPATH=/tmp/$USER/rocdojo-libs/merged/library/gfx1250
$B/clients/hipblaslt-bench --api_method c -m 592 -n 8000 -k 540688 --transA N --transB T \
  --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r \
  --rotating 512 -i 200 -j 50 --print_kernel_info
```

The reported kernel is origami's pick (`rocdojo_*` or a stock `Cijk_*` name).
`--algo_method all --requested_solution -1` times every candidate;
`--algo_method index --solution_index N` forces one. For correctness add
`--verify` (norm and allclose against the CPU reference; set
`OMP_NUM_THREADS` to bound the host reference).

- `TENSILE_PREDICTION_LIB=1` skips the stock Equality rows, so every pick
  comes from the Prediction (origami) table. That is how the dashboard numbers
  were taken; without it, a tuned stock Equality row wins over origami.
- `HIPBLASLT_LOG_MASK=8 TENSILE_DB=0x8040` logs the code objects loaded and
  the kernels launched.
- The instruction-cache flush between launches can fault on gfx1250 (ROCr
  `LOADER_ENABLE_TRAMPOLINE`); it is not specific to these kernels. Benchmark
  with flush off: the CLI default, `flush: false` in a YAML list, or
  `LOADER_ENABLE_TRAMPOLINE=0`.

## Dashboard problems

`gfx1250_bf16_dashboard_bench.yaml` holds the 800 bf16 problems the rocDojo
library was tuned on: 160 unbatched and 40 batched problems per layout, with
adaptive timing, a 512 MiB rotating buffer and flush off. Run it once per
library and compare:

```bash
for lib in stock merged; do
  TENSILE_PREDICTION_LIB=1 HIPBLASLT_TENSILE_LIBPATH=/tmp/$USER/rocdojo-libs/$lib/library/gfx1250 \
    $B/clients/hipblaslt-bench --yaml clients/scripts/rocdojo/gfx1250_bf16_dashboard_bench.yaml \
    > /tmp/$USER/rocdojo-$lib.log 2>&1
done
python3 clients/scripts/rocdojo/compare.py /tmp/$USER/rocdojo-stock.log /tmp/$USER/rocdojo-merged.log
```

`compare.py` prints, per layout, the geometric mean of the TFLOPS ratio, the
problems won and lost by more than 2%, and how many picks are rocDojo kernels.
The stock-only library above has no TT table, so its TT problems find no
solution and show as `missing`; compare TT against a full stock build instead.
