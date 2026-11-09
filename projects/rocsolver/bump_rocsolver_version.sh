#!/bin/sh

# run this script in develop after creating release-staging branch for feature-complete date
# Edit script to bump versions for new development cycle/release.

# for rocSOLVER version string
OLD_ROCSOLVER_VERSION="3\.39\.0"
NEW_ROCSOLVER_VERSION="3.40.0"
sed -i "s/${OLD_ROCSOLVER_VERSION}/${NEW_ROCSOLVER_VERSION}/g" CMakeLists.txt

# for rocSOLVER library name
OLD_ROCSOLVER_SOVERSION="0\.14"
NEW_ROCSOLVER_SOVERSION="0.15"
sed -i "s/${OLD_ROCSOLVER_SOVERSION}/${NEW_ROCSOLVER_SOVERSION}/g" library/CMakeLists.txt

# NOTE: build-time dependency fetches are pinned to immutable commits outside this script
# (cmake/get-rocm-cmake.cmake). If a release needs a newer pinned dep, grep "pinned-dep"
# and bump the commit by hand.
