# hipDNN Python Bindings

This project provides Python bindings for the hipDNN frontend library using the nanobind library. The bindings allow users to access the functionalities of the hipDNN library directly from Python, enabling seamless integration of deep learning operations.

## Project Structure

The project is organized as follows:

```
hipdnn-python-bindings
├── src
│   ├── module.cpp               # Main entry point for the nanobind module
│   ├── graph_bindings.cpp       # Bindings for the Graph class and its methods
│   ├── tensor_bindings.cpp       # Bindings for tensor-related functionalities
│   ├── attributes_bindings.cpp   # Bindings for attribute classes
│   └── types_bindings.cpp        # Bindings for custom types and enums
├── python
│   └── hipdnn_frontend
│       ├── __init__.py          # Initializes the hipdnn_frontend package
│       └── samples
│           └── bn_inference.py   # Sample implementation of batch normalization inference
├── CMakeLists.txt               # CMake configuration file
├── pyproject.toml               # Python project configuration
├── setup.py                     # Packaging instructions for the Python module
└── README.md                    # Project documentation
```

## Getting Started

### Prerequisites

- CMake
- A C++ compiler (e.g., g++, clang++)
- Python 3.6 or higher
- The nanobind library

### Building the Project

1. Clone the repository:
   ```
   git clone <repository-url>
   cd hipdnn-python-bindings
   ```

2. Create a build directory and navigate into it:
   ```
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   make
   ```

### Installing the Python Package

After building the project, you can install the Python package using pip:

```
pip install .
```

### Usage

Once the package is installed, you can use the bindings in your Python scripts. For example, you can run the batch normalization inference sample:

```python
from hipdnn_frontend.samples.bn_inference import run_bn_inference

run_bn_inference()
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.