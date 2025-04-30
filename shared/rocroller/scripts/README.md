# Scripts

## [trace_memory.py](trace_memory.py)

This script is intended to be used from within rocGDB.
e.g.:

```bash
rocgdb -ex "source trace_memory.py" -ex "gpu_memory_trace -h /work/tensile/Tensile/working/0_Build/client/tensile_client
```

### Using this Script:

1. The first step is to get the executable and inputs you're going to
   execute.  For Tensile this will look something like:

   ```bash
   ./working/0_Build/client/tensile_client --config-file ./working/1_BenchmarkProblems/Cijk_Ailk_Bjlk_HHS_BH_00/00_Final/source/ClientParameters.ini
   ```

   The kernel only needs to be launched once, so consider modifying
   the `num-*` parameters in the `ClientParameters.ini` file.  You can
   move the `ClientParameters.ini` to a more convenient place if you'd
   like.

2. The second step is to get the label for the kernel your working on,
   this is the first label in the kernel.  For Tensile kernels this
   will look something like:

       Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1

3. Next is to get the addresses matrices you are interested in
   tracing. To do this you can use the `get_to_kernel` option to this
   script to stop once the kernel is reached to enable inspection.

   Setting the environment variable `TENSILE_DB=0x40` will output the
   Tensile kernel arguments, including the addresses of all of the
   input matrices. For the A matrix it will have something like:

       [40..47] a: 00 00 a0 b0 f7 7f 00 00 (0x7ff7b0a00000)

4. Now we you are ready to collect the trace:

   ```bash
   rocgdb -ex "source /data/scripts/trace_memory.py" -ex "gpu_memory_trace full_memory_trace --kernel_label Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1 --instruction=load --workgroup=38,1,0 --base_address=0x7ff7b0a00000 --csv_file=/data/build/memory.csv --run_command=\"--config-file ClientParameters.ini\"" /work/tensile/Tensile/working/0_Build/client/tensile_client
   ```

5. Finally, you can visualize the trace using the `trace_to_png.py` script:

       ./trace_to_png.py -c -m 7680 -n 8448 -s 2 -w 8 -t wf memory.csv
