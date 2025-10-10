# Origami: Analytical GEMM Solution Selection

The name "origami" still evokes the elegance of transforming a flat (2-D) sheet into intricate higher dimensional structures. In this context, however, Origami has evolved into a toolset for **GEMM solution selection and optimization**. Inspired by the art of paper folding, the library now enables users to explore a range of tiling and mapping configurations and to make informed decisions on data and computation mapping for high-performance GEMM operations.

Origami provides a rigorous methodology to analyze and select optimal GEMM parameters. It does so by evaluating both **compute** and **memory latencies** across a wide range of tile sizes. The framework computes essential metrics such as:
- **Matrix Instruction (MI) Tiling Counts** and the associated compute latencies.
- **Memory Load Latencies**, considering both L2 and main memory bandwidth constraints.
- **Active Compute Unit Occupancy and Wave Counts**, ensuring realistic mapping to the underlying hardware.
- **Tie-breaking Strategies** using arithmetic intensity and more refined L2 hit rate estimations to resolve candidate configuration ties.

This approach allows programmers to achieve near-optimal performance without manually exploring every possibility. While default (implicit) parameters provide "out-of-the-box" performance, expert users can dive deeper into the analytical model to tune for their specific hardware and problem sizes.

---

## Documentation

- **Getting Started**
- [Programming Abstraction]()
- [Hierarchical Tiling and GEMM Latency Calculation]()
- [Usage Examples]()

---

## Quick Start Guide – Origami

**Origami** provides an end-to-end analytical solution to GEMM parameter selection. It estimates performance by sweeping over candidate tile sizes and selecting the optimal configuration based on latency and arithmetic intensity.

### Building Origami

Assuming you are in the repository root, run:

```bash
# configure
cmake -S . -B build/ -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_INSTALL_PREFIX=/opt/rocm 
# build
cmake --build build/ --parallel
```

### Installing Origami

After configuring and building, run the following command to install:

```bash
# install
cmake --target install
```

### Testing Origami Python Bindings

To test the origami Python bindings:

```bash
# configure with python bindings and tests enabled 
cmake -S . -B build/ -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_INSTALL_PREFIX=/opt/rocm -D ORIGAMI ENABLE_PYTHON=ON -D ORIGAMI_BUILD_TESTING=ON

# build 
cmake --build build/ --parallel

# run tests
cd build/
ctest --output-on-failure
```

### Options
* `ORIGAMI_BUILD_SHARED_LIBS`: Enables building of shared libraries (default: `ON`)
* `ORIGAMI_ENABLE_PYTHON`: Enables generation of origami Python bindings (default: `OFF`)
* `ORIGAMI_BUILD_TESTING`: Build the Python binding tests (default: `OFF`)


### Building and Running Origami C++ Tests
Use CMake to configure the build system and compile Origami with testing enabled. From origami root `<rocm-libraries-root>/shared/origami`:

```bash
cmake -S . -B build/ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -D ORIGAMI_BUILD_TESTING=ON

cmake --build build/ --parallel
```

Once the build completes, navigate to the test directory and run the test suite:

```bash
cd build/tests/
./origami-tests
```