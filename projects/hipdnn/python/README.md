# hipDNN Python Bindings

> [!CAUTION]
> **This is a POC of python bindings for hipdnn.  It likely has bugs and features missing.  Making this not a POC has been planned for a future date**

This project provides Python bindings for the hipDNN frontend library using the nanobind library. The bindings allow users to access the functionalities of the hipDNN library directly from Python, enabling seamless integration of deep learning operations.

## Project Structure

The project is organized as follows:

```
python
├── src
│   ├── module.cpp               # Main entry point for the nanobind module
│   ├── graph_bindings.cpp       # Bindings for the Graph class and its methods
│   ├── handle_bindings.cpp      # Bindings for handle management
│   ├── memory_bindings.cpp      # Bindings for device memory management
│   ├── tensor_bindings.cpp      # Bindings for tensor-related functionalities
│   ├── attributes_bindings.cpp  # Bindings for attribute classes
│   └── types_bindings.cpp       # Bindings for custom types and enums
├── hipdnn_frontend
│   ├── __init__.py              # Initializes the hipdnn_frontend package
│   └── samples
│       ├── bn_inference.py      # Batch normalization inference sample (DISABLED)
│       ├── conv_fprop.py        # Convolution forward propagation sample
│       ├── conv_dgrad.py        # Convolution backward data gradient sample
│       └── conv_wgrad.py        # Convolution backward weight gradient sample
├── CMakeLists.txt               # CMake configuration (scikit-build-core + subdirectory dual-mode)
├── pyproject.toml               # Python project configuration (scikit-build-core backend)
└── README.md                    # Project documentation
```

## Prerequisites

- CMake 3.18 or higher
- A C++ compiler with C++17 support (e.g. clang++)
- Python 3.9 or higher
- ROCm/HIP runtime and libraries
- hipDNN frontend library (built and installed)

## Getting Started

### 1. Setting up a Python Virtual Environment

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

### 2. Building and Installing the Python Bindings

The Python bindings use [scikit-build-core](https://scikit-build-core.readthedocs.io/) as the build backend, which drives CMake automatically through pip.

hipDNN must be installed (e.g. at `/opt/rocm`) before building the bindings:

```bash
# Navigate to the hipdnn python directory
cd python

pip install -v .
```

If hipDNN is installed somewhere other than `/opt/rocm`, pass the prefix:

```bash
pip install -v . -Ccmake.define.CMAKE_PREFIX_PATH=/path/to/hipdnn/install
```

### 3. Development Installation

After C++ changes, uninstall and reinstall:

```bash
pip uninstall hipdnn-frontend -y
pip install -v .
```

### 4. Running the Sample Applications

The repository includes several sample applications demonstrating different operations:

#### Convolution Forward Propagation
```bash
python conv_fprop.py
```

This sample demonstrates:
- Setting up a convolution forward pass
- Configuring padding, stride, and dilation parameters
- Executing the convolution and displaying results

#### Convolution Backward Data Gradient
```bash
python conv_dgrad.py
```

This sample demonstrates:
- Computing input gradients (dx) given output gradients (dy) and weights
- Used in backpropagation for training neural networks

#### Convolution Backward Weight Gradient
```bash
python conv_wgrad.py
```

This sample demonstrates:
- Computing weight gradients (dw) given output gradients (dy) and input (x)
- Used for updating convolution filter weights during training
