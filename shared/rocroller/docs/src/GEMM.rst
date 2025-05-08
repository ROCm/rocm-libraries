GEMM
====

To explore GEMM workloads and do performance testing, rocRoller
provides a GEMM client.  The GEMM client is built alongside the
library by default.

To run the GEMM client from your build directory::

    ./bin/client/rocRoller_gemm --help

The GEMM client can do a few different things::

1. Generate GEMM GPU solutions (which may be comprised of one or more
   GPU kernels).
2. Execute and validate GEMMs.
3. Execute and benchmark GEMMs.

These phases can be run individually, or together, in one invocation
of the gemm client.

Different GEMM kernels can be generated.  These are parameterized by
"solution parameters".  During kernel generation, the solution
parameters can be specified on the command line or in a configuration
file (YAML).

Quick start
-----------

To generate a solution and validate it (which runs the solution on the
local GPU and compares it to a reference solution computed on the CPU
by OpenBLAS)::

    ./bin/client/rocRoller_gemm --mac_m=128 --mac_n=128 --m 4096 --n 4096 --k 32 generate validate

Generating an example configuration
-----------------------------------

To generate a solution configuration file and save it::

    ./bin/client/rocRoller_gemm example getting-started.yaml

The `getting-started.yaml` file now contains some tuneable parameters
that can be used to generate different GEMM kernels.

Generating a solution
---------------------

The `generate` subcommand of the GEMM client can be used to generate a
GEMM kernel and save it.

To generate a solution, compile, and save it to a code-object::

    ./bin/client/rocRoller_gemm --mac_m=128 --mac_n=128 generate --co sgemm-current.co

The architecture can also be specified::

    ./bin/client/rocRoller_gemm --mac_m=128 --mac_n=128 --hgemm generate --arch gfx90a --co hgemm-gfx90a.co

You can also save the assembly if you'd like to modify it::

    ./bin/client/rocRoller_gemm --mac_m=128 --mac_n=128 generate --asm sgemm-current.s

If you have a configuration YAML file and want to generate a code-object for gfx90a::

    ./bin/client/rocRoller_gemm generate --config sgemm.yaml --arch gfx90a --co sgemm.co

Generating a solution and validating it
---------------------------------------

To generate a solution, run it once, and compare the result to OpenBLAS::

    ./bin/client/rocRoller_gemm --mac_m=128 --mac_n=128 generate validate --m 4096 --n 4096 --k 128

Generating a solution, modifying it, and validating it
------------------------------------------------------

To hack assembly manually::

    ./bin/client/rocRoller_gemm --mac_m=128 --mac_n=128 generate --asm my-sgemm.s
    emacs my-sgemm.s
    ./bin/client/rocRoller_gemm --m 4096 --n 4096 --k 128 validate --load my-sgemm.s

Note that in the _generate_ invocation, the problem size does not need
to be specified.  Only _solution_ and and _type_ parameters are
required.

After modifying the assembly, you can then verify the solution using
the _validate_ invocation.  Only _problem_ parameters are required.

Benchmarking
------------

To benchmark a solution::

    ./bin/client/rocRoller_gemm --mac_m=128 --mac_n=128 generate benchmark
