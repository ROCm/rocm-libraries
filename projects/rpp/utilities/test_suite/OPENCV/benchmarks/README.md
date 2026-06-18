# OpenCV vs RPP Host Benchmarking

This directory contains a comprehensive benchmarking suite that compares the performance of OpenCV operations against AMD's RPP (ROCm Performance Primitives) library on the host CPU.

## Overview

The `opencv_vs_rpp_host_benchmarking` performs side-by-side performance comparisons of 50+ image processing operations including:
- Color adjustments (brightness, contrast, gamma, exposure, hue, saturation)
- Filtering operations (box, median, gaussian, sobel filters)
- Image transformations (resize, crop, flip, rotate)
- Color space conversions
- Advanced operations (warp affine, LUT, histogram equalization)

Each operation is run for a configurable number of iterations (default: 100) on a batch of 128 1080p images to provide statistically meaningful performance metrics.

**Results are automatically exported to Excel files** with a filename format:
- `opencv_vs_rpp_benchmark_results_<N>threads.xlsx` - Where `<N>` is the number of threads used

Examples:
- `opencv_vs_rpp_benchmark_results_64threads.xlsx` - Results using 64 threads
- `opencv_vs_rpp_benchmark_results_32threads.xlsx` - Results using 32 threads
- `opencv_vs_rpp_benchmark_results_1threads.xlsx` - Single-threaded results

Each file contains:
- **Sheet 1:** System information (RPP version, OpenCV version, CPU, memory, OS)
- **Sheet 2:** Grayscale image benchmarks (OpenCV vs RPP HOST comparison with speedup)
- **Sheet 3:** RGB image benchmarks (OpenCV vs RPP HOST comparison with speedup)

## Directory Structure

```
OPENCV/
├── opencv_vs_rpp_host_benchmarking.cpp  # Main entry point and benchmark orchestration
├── opencv_benchmarks.cpp                # OpenCV benchmark implementations
├── rpp_benchmarks.cpp                   # RPP benchmark implementations
├── benchmarks_common.h                  # Shared declarations and data structures
├── benchmarks_utils.cpp                 # Utility functions (image loading, Excel export, system info)
├── dropout_helpers.cpp                  # Helper functions for dropout operations
├── CMakeLists.txt                       # CMake build configuration
├── generate_test_dataset.py             # Script to create synthetic test images
├── run_benchmarking.sh                  # One-command full setup and run
├── 1080p_128images_dataset/             # Test image directory (created by generate_test_dataset.py)
├── build_enabled_pthreads/              # Build output with parallel threads ON (created by run_benchmarking.sh)
└── build_disabled_pthreads/             # Build output with parallel threads OFF (created by run_benchmarking.sh)
```

## Code Organization

The benchmark suite is organized into modular files for better maintainability:

**Core Files:**
- `opencv_vs_rpp_host_benchmarking.cpp` - Main program that orchestrates benchmark execution
- `benchmarks_common.h` - Shared header with data structures, constants, and function declarations

**Implementation Files:**
- `opencv_benchmarks.cpp` - All OpenCV benchmark function implementations (~70+ operations)
- `rpp_benchmarks.cpp` - All RPP benchmark function implementations (~70+ operations)
- `benchmarks_utils.cpp` - Common utilities (image loading, Excel export, system information)
- `dropout_helpers.cpp` - Helper functions for dropout augmentation operations

This modular design separates concerns and makes it easier to:
- Add new benchmark operations
- Maintain and debug existing implementations
- Understand the codebase structure
- Reuse utility functions

## Prerequisites

### System Requirements
- Linux system (tested on Ubuntu 20.04+)
- CMake 3.10 or higher
- GCC with C++17 support
- Python 3 with Pillow (PIL) library (for dataset generation)
- Sudo privileges (for installing OpenCV)
- **Minimum 2GB RAM** (4GB+ recommended for 128 images × 1080p dataset)
  - Grayscale dataset: ~250MB RAM
  - RGB dataset: ~700MB RAM
  - Total with both: ~1GB for images + overhead

### Required Libraries
- **OpenCV 4.x** with components: core, imgproc, imgcodecs
- **OpenMP** for parallel processing
- **RPP (ROCm Performance Primitives)** - Automatically detected via CMake's `find_package(rpp)`
- **ROCm/HIP** - The RPP library requires HIP headers (platform is automatically defined in CMake)
- **libxlsxwriter** - For generating Excel output files

### Environment Setup

The build system uses the following priority for locating ROCm/RPP:
1. `ROCM_PATH` environment variable (if set)
2. CMake cache variable `ROCM_PATH` (if set)
3. Default path: `/opt/rocm`

To use a custom ROCm installation:

```bash
export ROCM_PATH=/path/to/your/rocm
# or
cmake -DROCM_PATH=/path/to/your/rocm ..
```

## Quick Start

### Option 1: Complete Automated Setup and Run

The easiest way to get started is to use the complete setup script:

```bash
cd utilities/test_suite/OPENCV
./run_benchmarking.sh
```

This script will:
1. Generate synthetic test dataset (if not present)
2. Install OpenCV and dependencies (if needed)
3. Build the benchmark executable
4. Run the complete benchmark suite

**Note:** The script will prompt for sudo password if dependencies need to be installed.

### Option 2: Step-by-Step Manual Setup

#### Step 1: Generate Test Dataset

Create 128 synthetic 1080p test images:

```bash
python3 generate_test_dataset.py
```

This generates various image patterns (gradients, checkerboards, circles, noise, stripes, etc.) in the `1080p_128images_dataset/` directory.

#### Step 2: Install Dependencies

Install OpenCV and build tools:

```bash
sudo apt-get update
sudo apt-get install -y libopencv-dev libgomp1 cmake build-essential libxlsxwriter-dev
pip3 install Pillow  # Required for generate_test_dataset.py
```

Verify OpenCV installation:

```bash
pkg-config --modversion opencv4  # or 'opencv' for older versions
```

#### Step 3: Build the Benchmark

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..
```

The executable will be created at `build/opencv_vs_rpp_host_benchmarking`

#### Step 4: Run the Benchmark

**Basic usage (auto-detect threads):**
```bash
./build/opencv_vs_rpp_host_benchmarking
```

**Command-line options:**
```bash
./build/opencv_vs_rpp_host_benchmarking [OPTIONS]

Options:
  -t, --threads <N>        Number of threads to use (default: auto-detect)
  -n, --num-runs <N>       Number of benchmark runs (default: 100)
  -g, --gray-path <PATH>   Path to grayscale images (default: 1080p_128images_dataset/)
  -r, --rgb-path <PATH>    Path to RGB images (default: 1080p_128images_dataset/)
  -h, --help               Display help message

Examples:
  ./build/opencv_vs_rpp_host_benchmarking                           # Auto-detect threads, 100 runs (default)
  ./build/opencv_vs_rpp_host_benchmarking --threads 64              # Use 64 threads
  ./build/opencv_vs_rpp_host_benchmarking -t 32 -n 50               # Use 32 threads with 50 runs
  ./build/opencv_vs_rpp_host_benchmarking -t 32 -g ./my_images/     # Use 32 threads with custom dataset
```

**Thread Configuration:**
- By default, the benchmark auto-detects the maximum available threads using OpenMP
- You can override this with the `-t` or `--threads` option
- Thread count only applies when built with `ENABLE_PARALLEL_THREADS=ON`
- When built with `ENABLE_PARALLEL_THREADS=OFF`, always uses 1 thread regardless of the argument


## Understanding the Output

### Console Output

The benchmark produces console output in the following format:

```
================================================
OpenCV vs RPP Benchmarking Script
================================================

System Information:
-------------------
CPU: AMD EPYC 7742 64-Core Processor
Kernel: Linux 6.17.0-22-generic
RPP Version: x.x.x
OpenCV Version: 4.x.x
Threads: 128
Batch Size: 128 images
Image Resolution: 1920x1080

Running Benchmarks (100 iterations each):
==========================================

benchmark_OpenCV_Brightness (Avg per run, 128 images, RGB): 45.2 ms
benchmark_RPP_Brightness    (Avg per run, 128 images, RGB): 32.1 ms

benchmark_OpenCV_GammaCorrection (Avg per run, 128 images, RGB): 52.3 ms
benchmark_RPP_GammaCorrection    (Avg per run, 128 images, RGB): 38.7 ms

...
```

Each operation shows:
- Operation name (OpenCV vs RPP)
- Number of images processed
- Color mode (RGB or Grayscale)
- Average execution time per run in milliseconds

### Excel Output File

Results are automatically saved to an Excel file with three sheets. The filename indicates the number of threads used:
- `opencv_vs_rpp_benchmark_results_<N>threads.xlsx` - Where `<N>` is the thread count

For example:
- `opencv_vs_rpp_benchmark_results_64threads.xlsx` - Results using 64 threads
- `opencv_vs_rpp_benchmark_results_1threads.xlsx` - Single-threaded baseline

**Sheet 1: System Information**
| Parameter | Value |
|-----------|-------|
| RPP Version | 3.1.2 |
| OpenCV Version | 4.6.0 |
| Operating System | Linux 6.17.0-22-generic |
| CPU | AMD EPYC 7742 64-Core Processor |
| Memory | 256.00 GB |
| Number of Threads | 128 |
| Number of Runs | 100 |

**Sheet 2: Grayscale Benchmarks**
| Operation | Parameters | OpenCV (ms) | RPP HOST (ms) | Speedup |
|-----------|------------|-------------|---------------|---------|
| Brightness | alpha=1.2, beta=20.0 | 45.20 | 32.10 | 1.41x |
| BoxFilter | kernel=3 | 26.02 | 26.04 | 1.00x |
| GammaCorrection | gamma=1.5 | 52.30 | 38.70 | 1.35x |
| ... | ... | ... | ... | ... |

**Sheet 3: RGB Benchmarks**
| Operation | Parameters | OpenCV (ms) | RPP HOST (ms) | Speedup |
|-----------|------------|-------------|---------------|---------|
| Brightness | alpha=1.2, beta=20.0 | 135.60 | 96.30 | 1.41x |
| Hue | hueDelta=10.0 | 78.90 | 52.40 | 1.51x |
| BoxFilter | kernel=3 | 79.60 | 79.60 | 1.00x |
| ... | ... | ... | ... | ... |

The Excel file can be opened in Excel, LibreOffice Calc, or Google Sheets for further analysis and visualization.

## Customizing the Benchmark

### Enabling/Disabling Parallel Threading

The benchmark supports toggling parallel threading at build time using the `ENABLE_PARALLEL_THREADS` CMake option. This allows you to compare performance with and without OpenMP parallelization.

**Build with parallel threads ENABLED (default):**
```bash
cd build
cmake ..
# or explicitly:
cmake -DENABLE_PARALLEL_THREADS=ON ..
make -j$(nproc)
```

**Build with parallel threads DISABLED:**
```bash
cd build
cmake -DENABLE_PARALLEL_THREADS=OFF ..
make -j$(nproc)
```

When parallel threads are disabled, all OpenMP `#pragma omp parallel for` directives are excluded from compilation, allowing you to measure single-threaded performance. The output Excel filename will automatically indicate which mode was used.

### Configuring Thread Count

The benchmark automatically detects the maximum available threads on your system. You can override this at runtime:

**Runtime configuration (recommended):**
```bash
# Use all available threads (auto-detect) with default 100 runs
./build/opencv_vs_rpp_host_benchmarking

# Use specific thread count
./build/opencv_vs_rpp_host_benchmarking --threads 64
./build/opencv_vs_rpp_host_benchmarking -t 32

# Use specific thread count with custom number of runs
./build/opencv_vs_rpp_host_benchmarking -t 32 -n 50
```

**Note:** Thread count configuration only applies when built with `ENABLE_PARALLEL_THREADS=ON`. When built with `ENABLE_PARALLEL_THREADS=OFF`, the benchmark always uses 1 thread regardless of command-line arguments.

### Configuring Number of Benchmark Runs

You can configure the number of benchmark runs (iterations) at runtime using the `-n` or `--num-runs` option:

**Runtime configuration (recommended):**
```bash
# Default 100 runs
./build/opencv_vs_rpp_host_benchmarking

# Custom number of runs
./build/opencv_vs_rpp_host_benchmarking --num-runs 50
./build/opencv_vs_rpp_host_benchmarking -n 200

# Combine with thread count
./build/opencv_vs_rpp_host_benchmarking -t 32 -n 50
```

**Alternative: Build-time configuration:**
You can also set a default value by editing `benchmarks_common.h`:

```cpp
extern int NUM_RUNS;  // Default is 100, but can be overridden via command-line
```

The runtime command-line option will always override the default value.

### Changing RPP Installation Path

The CMakeLists.txt automatically finds RPP using CMake's `find_package(rpp)`. If RPP is installed in a non-standard location, you can:

**Option 1: Set ROCM_PATH**
```bash
export ROCM_PATH=/path/to/rocm
cmake ..
```

**Option 2: Set RPP_DIR**
```bash
cmake -DRPP_DIR=/path/to/rpp/lib/cmake/rpp ..
```

**Option 3: Add to CMAKE_PREFIX_PATH**
```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/rpp ..
```

The build system will automatically locate the RPP headers and libraries.

### Using Custom Image Dataset

**Option 1: Command-line arguments (recommended):**
```bash
# Use custom dataset path for both grayscale and RGB
./build/opencv_vs_rpp_host_benchmarking --gray-path ./my_images/ --rgb-path ./my_images/

# Use different paths for grayscale and RGB
./build/opencv_vs_rpp_host_benchmarking -g ./gray_imgs/ -r ./rgb_imgs/
```

**Option 2: Replace default dataset:**
Replace the synthetic images in `1080p_128images_dataset/` with your own images.

**Supported image formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

The benchmark will automatically detect and process all valid images in the specified directory.

## Troubleshooting

### OpenCV Not Found

```bash
# Check if OpenCV is installed
pkg-config --modversion opencv4

# If not found, install it
sudo apt-get install libopencv-dev
```

### RPP Library Not Found

If CMake cannot find RPP, you'll see an error like:
```
-- opencv_benchmarking_script requires RPP. Install RPP or set RPP_DIR to the RPP installation path
```

**Solution 1: Set ROCM_PATH environment variable**
```bash
export ROCM_PATH=/path/to/rocm
```

**Solution 2: Set RPP_DIR when running cmake**
```bash
cmake -DRPP_DIR=/path/to/rpp/lib/cmake/rpp ..
```

**Verify RPP installation:**
```bash
# Check if RPP is installed
ls -la $ROCM_PATH/lib/librpp.so
ls -la $ROCM_PATH/include/rpp/

# Or check the default location
ls -la /opt/rocm/lib/librpp.so
ls -la /opt/rocm/include/rpp/
```

### libxlsxwriter Not Found

If CMake reports that libxlsxwriter is not found:
```
-- opencv_benchmarking_script requires libxlsxwriter for Excel output
```

**Solution:**
```bash
sudo apt-get install libxlsxwriter-dev
```

Then rebuild:
```bash
rm -rf build && mkdir build && cd build && cmake .. && make
```

### Build Errors - HIP Platform Not Defined

If you see errors like:
```
error: #error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
```

This means the HIP platform macro is not defined. The CMakeLists.txt automatically defines `__HIP_PLATFORM_AMD__` for AMD ROCm. If you still see this error, verify you have the latest CMakeLists.txt and rebuild:

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Other Build Errors

For general build issues, try a clean rebuild:

```bash
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Permission Denied

If you get permission errors when running scripts:

```bash
chmod +x run_benchmarking.sh
chmod +x generate_test_dataset.py
```

### No Images Found Error

Make sure the test dataset exists:

```bash
ls -la 1080p_128images_dataset/

# If empty or missing, generate it:
python3 generate_test_dataset.py
```

## Performance Notes

- **Thread Configuration:**
  - The benchmark auto-detects the maximum available threads by default
  - Override with `--threads N` command-line argument
  - Thread count only applies when built with `ENABLE_PARALLEL_THREADS=ON`
  - Uses OpenMP for parallel processing (can be disabled at build time with `-DENABLE_PARALLEL_THREADS=OFF`)

- **Benchmark Iterations:**
  - Each operation runs 100 times by default for stable average timings
  - Adjustable at runtime via `-n` or `--num-runs` command-line option (no rebuild required)
  - Example: `./build/opencv_vs_rpp_host_benchmarking -n 200` runs each benchmark 200 times

- **Performance Variability:**
  - Results vary based on CPU architecture, memory bandwidth, and system load
  - Close other resource-intensive applications for accurate comparisons
  - Test with different thread counts to find optimal performance for your system

- **Parallel vs Single-threaded Comparison:**
  - Build with `ENABLE_PARALLEL_THREADS=ON` and run with different `--threads` values
  - Build with `ENABLE_PARALLEL_THREADS=OFF` for single-threaded baseline
  - Compare the resulting Excel files to analyze scaling efficiency

## Tested Operations

The benchmark includes the following operation categories:

**Color Adjustments:**
- Brightness, Contrast, Gamma Correction, Exposure
- Hue, Saturation, Color Jitter
- Color Temperature, Vignette

**Filters:**
- Box Filter, Median Filter, Gaussian Filter
- Sobel Filter, Non-Max Suppression
- Erode, Dilate

**Transformations:**
- Resize (Nearest Neighbor, Bilinear, Bicubic, Lanczos)
- Crop, Flip (Horizontal, Vertical, Both)
- Rotate (90°, 180°, 270°)
- Warp Affine

**Color Conversions:**
- RGB to Grayscale, HSV, BGR
- LUT (Lookup Table)
- Histogram Equalization
- Channel Extract

## License

This benchmarking tool is part of the RPP (ROCm Performance Primitives) project.

## Support

For issues related to:
- **RPP library:** Refer to the main RPP repository
- **OpenCV:** Consult OpenCV documentation at https://docs.opencv.org/
- **This benchmark:** Open an issue in the RPP repository
