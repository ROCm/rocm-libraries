#!/bin/bash

set -eu

# Make this directory the PWD
cd "$(dirname "${BASH_SOURCE[0]}")"

# Build sphinx docs (Exhale will automatically run Doxygen)
make clean
make html
make latexpdf
