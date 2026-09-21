# rocALUTION

> [!NOTE]
> The published documentation is available at [rocALUTION](https://rocm.docs.amd.com/projects/rocALUTION/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

rocALUTION is a sparse linear algebra library that can be used to explore fine-grained parallelism on
top of the [ROCm](https://github.com/ROCm/ROCm) platform runtime and toolchains.
Based on C++ and [HIP](https://github.com/ROCm/HIP/), rocALUTION
provides a portable, generic, and flexible design that allows seamless integration with other scientific
software packages.

rocALUTION offers various backends for different (parallel) hardware:

* Host
* [OpenMP](http://www.openmp.org/): Designed for multi-core CPUs
* [HIP](https://github.com/ROCm/HIP/): Designed for ROCm-compatible devices
* [MPI](https://www.open-mpi.org/): Designed for multi-node clusters and multi-GPU setups

## Requirements

To use rocALUTION on GPU devices, you must first install the
[rocBLAS](https://github.com/ROCm/rocBLAS),
[rocSPARSE](https://github.com/ROCm/rocSPARSE), and
[rocRAND](https://github.com/ROCm/rocRAND) libraries. You can install these from
the ROCm repository, the GitHub 'releases' tab, or you can manually compile them.

## Documentation

Documentation for rocALUTION is available at
[https://rocm.docs.amd.com/projects/rocALUTION/en/latest/](https://rocm.docs.amd.com/projects/rocALUTION/en/latest/).

To build our documentation locally, use the following code:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build

You can compile rocALUTION using CMake 3.5 or later. Note that all compiler specifications are
determined automatically.

```bash
# Clone rocALUTION using git
git clone https://github.com/ROCm/rocALUTION.git

# Go to rocALUTION directory, create and change to build directory
cd rocALUTION; mkdir build; cd build

# Configure rocALUTION
# Build options:
#   SUPPORT_HIP         - build rocALUTION with HIP support (ON)
#   SUPPORT_OMP         - build rocALUTION with OpenMP support (ON)
#   SUPPORT_MPI         - build rocALUTION with MPI (multi-node) support (OFF)
#   BUILD_SHARED_LIBS   - build rocALUTION as shared library (ON, recommended)
#   BUILD_EXAMPLES      - build rocALUTION examples (ON)
cmake .. -DSUPPORT_HIP=ON -DROCM_PATH=/opt/rocm/

# Build
make
```

To test your installation, run a CG solver on a Laplacian matrix:

```bash
cd rocALUTION; cd build
wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
gzip -d gr_30_30.mtx.gz
./clients/staging/cg gr_30_30.mtx
```

## General information

rocALUTION is based on a generic and robust design that allows expansion in the direction of new
solvers and preconditioners with support for various hardware types. The library's design allows the
use of all solvers as preconditioners in other solvers. For example, you can define a CG solver with a
multi-elimination preconditioner, in which the last-block is preconditioned with another Chebyshev
iteration method that itself is preconditioned with a multi-colored symmetric Gauss-Seidel scheme.

### Iterative solvers

* Fixed-point iteration schemes: Jacobi, (Symmetric) Gauss-Seidel, SOR, SSOR
* Krylov subspace methods: CR, CG, BiCGStab, BiCGStab(*l*), GMRES, IDR, QMRCGSTAB,
  Flexible CG/GMRES
* Mixed-precision defect correction scheme
* Chebyshev iteration scheme
* Multigrid: Geometric and algebraic

### Preconditioners

* Matrix splitting schemes: Jacobi, (multi-colored) (symmetric) Gauss-Seidel, SOR, SSOR
* Factorization schemes: ILU(*0*), ILU(*p*) (based on levels), ILU(*p,q*) (power(*q*)-pattern method),
  multi-elimination ILU (nested/recursive), ILUT (based on threshold), IC(*0*)
* Approximate Inverses: Chebyshev matrix-valued polynomial, SPAI, FSAI, TNS
* Diagonal-based preconditioner for Saddle-point problems
* Block-type of sub-preconditioners/solvers
* Additive Schwarz (restricted)
* Variable type of preconditioners

### Sparse matrix formats

* Compressed Sparse Row (CSR)
* Modified Compressed Sparse Row (MCSR)
* Dense (DENSE)
* Coordinate (COO)
* ELL
* Diagonal (DIA)
* Hybrid ELL+COO (HYB)

## Portability

All code based on rocALUTION is portable and hardware-independent. It compiles and runs on any
supported platform. All solvers and preconditioners are based on a single source code implementation
that delivers portable results across all backends (note that variations are possible due to different
hardware rounding modes). The only visible difference between hardware is performance variation.
