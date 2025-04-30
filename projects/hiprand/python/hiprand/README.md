# hipRAND Python Wrapper

## Requirements

* rocRAND
* Python 3.5
* pip (only for installing)
* NumPy (will be installed automatically as a dependency if necessary)

If rocRAND is built from sources but not installed or installed in non-standard
directory set `ROCRAND_PATH` environment variable, for example:

```
export ROCRAND_PATH=~/rocRAND/build/library/
```

## Installing

```
cd python/hiprand
pip install .
```

Run tests:

```
python tests/hiprand_test.py
```

It is also possible to test the wrapper without installing:

```
python setup.py test
```

Run examples:

```
python examples/pi.py
```

## Creating a source distribution

```
cd python/hiprand
python setup.py sdist
```

The package `hiprand-<version>.tar.gz` will be placed into `dist/`.
It can be installed later using this command:

```
pip install hiprand-<version>.tar.gz
```

## Building documentation

Install Sphinx (http://www.sphinx-doc.org/en/stable/index.html):

```
pip install Sphinx
```

Run:

```
cd python/hiprand
python setup.py build_sphinx
```

The documentation will be placed into `docs/build/html`.

Note: Sphinx requires that the module is properly loaded to generate
documentation from sources. Consider to install rocRAND first
or set `ROCRAND_PATH` if you see error messages like
"ImportError: libhiprand.so cannot be loaded..."
