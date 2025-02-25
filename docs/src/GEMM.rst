GEMM
====

To explore GEMM workloads and do performance testing, rocRoller
provides a GEMM client.  The GEMM client is built alongside the
library by default.

To run the GEMM client from your build directory::

    client/gemm --help

The GEMM client can do a few different things::

1. Generate GEMM GPU solutions (which may be comprised of one or more
   GPU kernels).
2. Execute and validate GEMMs.
3. Execute and benchmark GEMMs.

These phases can be run individually, or together, in one invocation
of the gemm client.

Quick start
-----------

To generate a solution and validate it (which runs the solution on the
local GPU and compares it to a reference solution computed on the CPU
by OpenBLAS)::

    client/gemm --mac_m=128 --mac_n=128 --m 4096 --n 4096 --k 32 generate validate

Generating a solution
---------------------

To generate a solution and save it::

    client/gemm --mac_m=128 --mac_n=128 generate --save sgemm-current.yaml

The architecture can also be specified::

    client/gemm --mac_m=128 --mac_n=128 --hgemm generate --arch gfx90a --save hgemm-gfx90a.yaml

Generating a solution and validating it
---------------------------------------

To generate a solution, run it once, and compare the result to OpenBLAS::

    client/gemm --mac_m=128 --mac_n=128 generate validate --m 4096 --n 4096 --k 128

Generating a solution, modifying it, and validating it
------------------------------------------------------

To hack assembly manually::

    client/gemm --mac_m=128 --mac_n=128 generate --save my-sgemm.yaml
    emacs my-sgemm.s
    client/gemm --m 4096 --n 4096 --k 128 validate --load my-sgemm.yaml

Note that in the _generate_ invocation, the problem size does not need
to be specified.  Only _solution_ and and _type_ parameters are
required.

After modifying the assembly, you can then verify the solution using
the _validate_ invocation.  Only _problem_ parameters are required.

Benchmarking
------------

To benchmark a solution::

    client/gemm --mac_m=128 --mac_n=128 generate benchmark
