# hipDNN Python Bindings

> [!CAUTION]
> **This is a POC of python bindings for hipdnn.  It likely has bugs and features missing.  Making this not a POC has been planned for a future date**


This project provides Python bindings for the hipDNN frontend library using the nanobind library. The bindings allow users to access the functionalities of the hipDNN library directly from Python, enabling seamless integration of deep learning operations.

## Project Structure

The project is organized as follows:

```
hipdnn-python-bindings
├── src
│   ├── module.cpp               # Main entry point for the nanobind module
│   ├── graph_bindings.cpp       # Bindings for the Graph class and its methods
│   ├── handle_bindings.cpp      # Bindings for handle management
│   ├── memory_bindings.cpp      # Bindings for device memory management
│   ├── tensor_bindings.cpp      # Bindings for tensor-related functionalities
│   ├── attributes_bindings.cpp  # Bindings for attribute classes
│   └── types_bindings.cpp       # Bindings for custom types and enums
├── python
│   └── hipdnn_frontend
│       ├── __init__.py          # Initializes the hipdnn_frontend package
│       └── samples
│           ├── bn_inference.py   # Batch normalization inference sample(DISABLED)
│           ├── conv_fprop.py     # Convolution forward propagation sample
│           ├── conv_dgrad.py     # Convolution backward data gradient sample
│           └── conv_wgrad.py     # Convolution backward weight gradient sample
├── CMakeLists.txt               # CMake configuration file
├── pyproject.toml               # Python project configuration
├── setup.py                     # Packaging instructions for the Python module
└── README.md                    # Project documentation
```

## Prerequisites

- CMake 3.15 or higher
- A C++ compiler with C++17 support (e.g. clang++)
- Python 3.8 or higher
- ROCm/HIP runtime and libraries
- hipDNN frontend library (built and installed)
- NumPy (for running samples)

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

# Install required Python packages
pip install numpy
```

### 2. Building and Installing the Python Bindings

The Python bindings use scikit-build to handle the CMake build process automatically through pip:

```bash
# Navigate to the hipdnn-python-bindings directory
cd hipdnn-python-bindings

# Install the package with verbose output to see build progress
pip install -v .
```

If you encounter issues with finding the hipDNN library, you may need to set environment variables:

```bash
export CMAKE_PREFIX_PATH=/path/to/hipdnn/install:$CMAKE_PREFIX_PATH
pip install -v .
```

### 3. Development Installation

For development work where you want changes to Python files to take effect immediately:

#### Editable Installation (Recommended for Development)
```bash
# Install in editable/development mode
pip install -e .
```

This creates a link to your development directory so changes to Python files are immediately available without reinstalling.

#### Reinstalling After C++ Changes
```bash
# If you make changes to C++ bindings, rebuild and reinstall:
pip uninstall hipdnn-frontend -y
pip install -v .
```

#### Quick Rebuild for Testing
```bash
# For rapid testing during development:
pip uninstall hipdnn-frontend -y && pip install -e .
```

### 4. Running the Sample Applications

The repository includes several sample applications demonstrating different operations:

#### Batch Normalization Inference
```bash
cd python/hipdnn_frontend/samples
python bn_inference.py
```

This sample demonstrates:
- Building a batch normalization inference graph
- Executing the graph with test data
- Verifying results against expected values

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

## Usage Example

Here's a simple example of using the bindings in your own Python code:

```python
import numpy as np
import hipdnn_frontend as hipdnn

# Create a handle for backend operations
handle = hipdnn.Handle()

# Create a computation graph
graph = hipdnn.Graph()
graph.set_name("my_graph")
graph.set_io_data_type(hipdnn.DataType.FLOAT)

# Create tensors
input_tensor = hipdnn.Tensor.create([1, 3, 224, 224], hipdnn.DataType.FLOAT)
input_tensor.set_name("input")

# Build and execute your operations...
# See the samples directory for complete examples
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'hipdnn_frontend_python'**
   - Ensure the build completed successfully
   - Check that the module is in your PYTHONPATH or installed via pip

2. **Runtime errors about missing libraries**
   - Ensure ROCm/HIP libraries are in your LD_LIBRARY_PATH:
     ```bash
     export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
     ```

3. **CUDA/HIP device not found**
   - Verify your GPU is properly configured:
     ```bash
     rocm-smi  # Check if ROCm recognizes your GPU
     ```

4. **Memory allocation errors**
   - Ensure you have sufficient GPU memory available
   - Check that workspace allocation is properly handled (fixed in latest version)

## Development

### Running Tests

To run the sample tests as a validation suite:

```bash
# From the samples directory
python -m pytest . -v  # Requires pytest to be installed
```

### Debugging

For debugging Python bindings issues:

```bash
# Enable verbose output from hipDNN
export HIPDNN_LOG_LEVEL=5

# Run with Python debugger
python -m pdb conv_fprop.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

When adding new bindings:
- Update the appropriate binding file in `src/`
- Add corresponding Python tests/samples
- Update this README if adding new functionality

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Support

For issues and questions:
- Open an issue on the GitHub repository
- Check existing issues for solutions to common problems
- Provide detailed error messages and system configuration when reporting issues
