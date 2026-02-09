# hipRAND

> [!NOTE]
> The published hipRAND documentation is available [here](https://rocm.docs.amd.com/projects/hipRAND/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

hipRAND is a RAND marshalling library with multiple supported backends. It sits between your
application and the backend RAND library, where it marshals inputs to the backend and results to the
application. hipRAND exports an interface that doesn't require the client to change, regardless of the
chosen backend.

hipRAND supports [rocRAND](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand).

## Requirements

You must have the following installed to use hipRAND:

* CMake (3.16 or later)
* For AMD GPUs:
  * AMD ROCm Software (5.0.0 or later)
  * rocRAND library

## Build and install

You can download pre-built packages by following the [ROCm Install Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html),
or by clicking the github releases tab (this option could have a newer version).

Once downloaded, use the following command to install hipRAND:

`sudo apt update && sudo apt install hiprand`

To build hipRAND, you can use the bash helper script (Ubuntu only) or build manually (for all
supported platforms):

* Bash helper build script:

  The helper script `install` is located in the root of the `projects/hiprand` folder. Note that this method doesn't take many
  options and hard-codes a configuration that you can specify by invoking CMake directly.

  A few commands in the script need sudo access, so it may prompt you for a password.

  * `./install -h`: Shows help
  * `./install -id`: Builds library, dependencies, and installs (the `-d` flag only needs to be passed once on
    a system)

* Manual build:

  For information on cloning and building the hipRAND library, see the
  [hipRAND installation documentation](https://rocm.docs.amd.com/projects/hipRAND/en/latest/install/installation.html) for version 7.0 or later. It has helpful information on how to configure CMake and build manually.


## Interface examples

The hipRAND interface is compatible with rocRAND and cuRAND-v2 APIs. Porting a CUDA application
that calls the cuRAND API to an application that calls the hipRAND API is relatively straightforward. For
example, to create a generator:

### Host API

```c
hiprandStatus_t
hiprandCreateGenerator(
  hiprandGenerator_t* generator,
  hiprandRngType_t rng_type
)
```

### Device API

Here is an example that generates a log-normally distributed float from a generator (these functions
are templated for all generators).

```c
__device__ double
hiprand_log_normal_double(
  hiprandStateSobol64_t* state,
  double mean,
  double stddev
)
```

## Building the documentation locally

### Requirements

#### Doxygen

The build system uses Doxygen [version 1.9.4](https://github.com/doxygen/doxygen/releases/tag/Release_1_9_4). You can try using a newer version, but that might cause issues.

After you have downloaded Doxygen version 1.9.4:

```shell
# Add doxygen to your PATH
echo 'export PATH=<doxygen 1.9.4 path>/bin:$PATH' >> ~/.bashrc

# Apply the updated .bashrc
source ~/.bashrc

# Confirm that you are using version 1.9.4
doxygen --version
```

#### Python

The build system uses Python version 3.10. You can try using a newer version, but that might cause issues.

You can install Python 3.10 alongside your other Python versions using [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):

```shell
# Install Python 3.10
pyenv install 3.10

# Create a Python 3.10 virtual environment
pyenv virtualenv 3.10 venv_hiprand

# Activate the virtual environment
pyenv activate venv_hiprand
```

### Building

After cloning this repository and navigating into its directory:

```shell
# Install Python dependencies
python3 -m pip install -r docs/sphinx/requirements.txt

# Build the documentation
python3 -m sphinx -T -E -b html -d docs/_build/doctrees -D language=en docs docs/_build/html
```

You can then open `docs/_build/html/index.html` in your browser to view the documentation.
