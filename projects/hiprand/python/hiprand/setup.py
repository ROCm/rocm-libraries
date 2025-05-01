from setuptools import setup, find_packages
import sys


with open("README.md") as f:
    readme = f.read()

REQUIRED_PACKAGES = ["numpy"]

# Use a backport of weakref.finalizers if not supported by the standard library
if sys.version_info < (3, 4):
    REQUIRED_PACKAGES.append("backports.weakref")

version = "1.0"
release = "1.0.0"
setup(
    name="hiprand",
    version=release,
    description="hipRAND Python Wrapper",
    long_description=readme,
    author="Advanced Micro Devices, Inc.",
    url="https://github.com/ROCm/hipRAND",
    license="MIT",
    packages=["hiprand"],
    install_requires=REQUIRED_PACKAGES,
    command_options={
        "build_sphinx": {
            "version": ("setup.py", version),
            "release": ("setup.py", release)}},
)
