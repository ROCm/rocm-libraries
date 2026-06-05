# hipDNN Python Bindings

> [!CAUTION]
> **This is a POC of python bindings for hipdnn.  It likely has bugs and features missing.  Making this not a POC has been planned for a future date**


This project provides Python bindings for the hipDNN frontend library using the nanobind library. The bindings allow users to access the functionalities of the hipDNN library directly from Python, enabling seamless integration of deep learning operations.

## Project Structure

```
python
├── src
│   ├── module.cpp               # Main entry point for the nanobind module
│   ├── bindings.hpp             # Shared declarations for binding functions
│   ├── graph_bindings.cpp       # Bindings for the Graph class and its methods
│   ├── handle_bindings.cpp      # Bindings for handle management
│   ├── memory_bindings.cpp      # Bindings for device memory management
│   ├── tensor_bindings.cpp      # Bindings for tensor-related functionalities
│   ├── attributes_bindings.cpp  # Bindings for attribute classes
│   └── types_bindings.cpp       # Bindings for custom types and enums
├── hipdnn_frontend
│   ├── __init__.py              # Initializes the hipdnn_frontend package
│   ├── samples/                 # Sample scripts (conv_fprop, conv_dgrad, conv_wgrad, matmul)
│   └── test/                    # Tests for the Python bindings
├── CMakeLists.txt               # CMake configuration (scikit-build-core + subdirectory dual-mode)
├── pyproject.toml               # Python project configuration (scikit-build-core backend)
└── README.md                    # Project documentation
```

## Prerequisites

- CMake 3.18 or higher
- A C++ compiler with C++17 support (e.g. clang++)
- Python 3.9 or higher
- ROCm/HIP runtime and libraries

## Building

There are two ways to build the Python bindings:

### Subdirectory build (via parent hipDNN CMake)

Set `HIPDNN_BUILD_PYTHON_BINDINGS=ON` to compile the nanobind extension module alongside hipDNN and stage it for packaging. It is off by default because it requires Python development headers and fetches nanobind/tsl-robin-map as additional dependencies.

```bash
cmake -S projects/hipdnn -B build -GNinja -DHIPDNN_BUILD_PYTHON_BINDINGS=ON
cmake --build build
cmake --install build --prefix /path/to/install
```

The bindings are staged to `share/hipdnn/python/hipdnn_frontend/` under the install prefix.

### Standalone build (via pip)

hipDNN must already be built and installed (e.g. at `/opt/rocm`).

#### 1. Setting up a Python Virtual Environment

It's recommended to use a Python virtual environment to isolate the project dependencies:

```bash
# Create a virtual environment
python3 -m venv hipdnn_env

# Activate the virtual environment
# On Linux/Mac:
source hipdnn_env/bin/activate
# On Windows:
# hipdnn_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### 2. Building and Installing

```bash
# Navigate to the hipdnn python directory
cd python

pip install -v .
```

If hipDNN is installed somewhere other than `/opt/rocm`, pass the prefix:

```bash
pip install -v . -Ccmake.define.CMAKE_PREFIX_PATH=/path/to/hipdnn/install
```

#### 3. Development Installation

After C++ changes, uninstall and reinstall:

```bash
pip uninstall hipdnn-frontend -y
pip install -v .
```

## Running the Samples

Sample scripts are located in `hipdnn_frontend/samples/`:

```bash
python hipdnn_frontend/samples/conv_fprop.py
python hipdnn_frontend/samples/conv_dgrad.py
python hipdnn_frontend/samples/conv_wgrad.py
python hipdnn_frontend/samples/matmul.py
```
