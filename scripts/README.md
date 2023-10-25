# Scripts

## [trace_memory.py](trace_memory.py)

This script is intended to be used from within rocGDB.
e.g.:

```bash
rocgdb -ex "source trace_memory.py" -ex "gpu_memory_trace -h /work/tensile/Tensile/working/0_Build/client/tensile_client
```

### Using this Script:

1. The first step is to get the executable and inputs you're going to execute. For Tensile this will look something like:
   ```bash
   ./working/0_Build/client/tensile_client --config-file ./working/1_BenchmarkProblems/Cijk_Ailk_Bjlk_HHS_BH_00/00_Final/build/../source/ClientParameters.ini
   ```

2. The second step is to get the label for the kernel your working on, this is the first label in the kernel. For Tensile kernels this will look something like: `Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1`

3. Next is to get the addresses of the loads/stores you are interested in tracing. To do this you can use the `get_to_kernel` option to this script to stop once the kernel is reached to enable inspection.
   Once the program has encountered the breakpoint in the kernel, the `layout asm` command can be used to inspect the kernel assembly.
   In the rocGDB `tui` you can search for the loads/stores you care about and make note of the instruction addresses and offset/buffer descriptor registers for each.
   e.g., for the following instruction, the address is `0x7ffff52992a4`, the offset register is `26`, the buffer descriptor register is `8`, and the width is 16 bytes.
   ```bash
   0x7ffff52992a4 <label_AlphaNonZero+556> buffer_load_dwordx4 v[30:33], v26, s[8:11], 0 offen
   ```
   Setting the environment variable `TENSILE_DB=0x40` will output the Tensile kernel arguments, including the addresses of all of the input matrices. For the A matrix it will have something like:
   ```bash
   [40..47] a: 00 00 a0 b0 f7 7f 00 00 (0x7ff7b0a00000)
   ```

4. Finally you can use all of this information to run the trace.
   ```bash
   rocgdb -ex "source /data/scripts/trace_memory.py" -ex "gpu_memory_trace full_memory_trace --kernel_label Cijk_Ailk_Bjlk_HHS_BH_MT128x256x16_MI32x32x8x1_SE_K1 --instruction_address 0x7ffff52992a4 0x7ffff52997c8 0x7ffff5299a10 --buffer_descriptor=8 --offset=26 --workgroup=(38,1,0) --base_address=0x7ff7b0a00000 --matrix_m=15360 --matrix_n=8192 --trace_count=2048 --image_file=/data/build/memoryA.png --csv_file=/data/build/memory.csv --run_command=\"--config-file /work/working/1_BenchmarkProblems/Cijk_Ailk_Bjlk_HHS_BH_00/00_Final/build/../source/ClientParameters.ini\"" /work/tensile/Tensile/working/0_Build/client/tensile_client
   ```

> It may take some experimentation to figure out the correct value to pass for `--trace_count`. You want this to be the number of times that the instructions are encountered in the given workgroup.
