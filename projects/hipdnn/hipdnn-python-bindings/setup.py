from setuptools import setup, Extension
import numpy as np

# Define the extension module
ext_modules = [
    Extension(
        'hipdnn_frontend',
        sources=[
            'src/module.cpp',
            'src/graph_bindings.cpp',
            'src/tensor_bindings.cpp',
            'src/attributes_bindings.cpp',
            'src/types_bindings.cpp',
        ],
        include_dirs=[
            np.get_include(),  # Include NumPy headers
            'path/to/hipdnn_frontend/include',  # Adjust this path as necessary
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],  # Use C++17 standard
    ),
]

# Setup function
setup(
    name='hipdnn_frontend',
    version='0.1.0',
    description='Python bindings for hipDNN frontend',
    author='Advanced Micro Devices, Inc.',
    author_email='support@amd.com',
    packages=['hipdnn_frontend'],
    ext_modules=ext_modules,
    install_requires=[
        'numpy',  # Add any other dependencies here
    ],
    zip_safe=False,
)