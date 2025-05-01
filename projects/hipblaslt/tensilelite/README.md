* TensileLite

** How to Rebuild Object Codes Directly from Assembly

During the tuning process, it is of interest to modify an assembly file/s and rebuild the corresponding object file/s and then relink the corresponding co file. Currently, we generate additional source files and a script to provide this workflow. 

A new `Makefile` is added that manages rebuilding a co file during iterative development when tuning. One modifies an assembly file of interest, then runs `make` and make will detect what file/s changed and rebuild accordingly.

Assumptions:

- Each problem directory contains a library directory with one co file corresponding to one architecture

Example:

```
cd hipBLASLt/tensilelite
Tensile/bin/Tensile Tensile/Tests/gemm/fp16_use_e.yaml tensile-out
# modify an assembly file in tensile-out/1_BenchmarkProblems/Cijk_Ailk_Bjlk_DB_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly
make co TENSILE_OUT=tensile-out
# re-run the client
```

The Makefile will set the target based on the name of the co file and sets a default wavefront flag but each of these can be customized as follows:

For 64 wavefront size systems,

```
make co TENSILE_OUT=tensile-out ARCH="gfx942" WAVE=64
```

For 32 wavefront size systems,

```
make co TENSILE_OUT=tensile-out ARCH="gfx1100" WAVE=32
```

In addition, we provide `ASM_ARGS` and `LINK_ARGS` as additional customization points for the assemble and link step respectively. If the architecture cannot be detect corectly, you may need to manually add ``ARCH="gfx942:xnack-"`` to the ``make`` command.
