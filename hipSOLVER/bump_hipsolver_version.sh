#!/bin/bash

# run this script in develop after creating release-staging branch for feature-complete date
# Edit script to bump versions for new development cycle/release.

# for hipSOLVER version string
OLD_HIPSOLVER_VERSION="2\.5\.0"
NEW_HIPSOLVER_VERSION="2.6.0"
sed -i "s/${OLD_HIPSOLVER_VERSION}/${NEW_HIPSOLVER_VERSION}/g" CMakeLists.txt

# for hipSOLVER library name
OLD_HIPSOLVER_SOVERSION="0\.5"
NEW_HIPSOLVER_SOVERSION="0.6"
sed -i "s/${OLD_HIPSOLVER_SOVERSION}/${NEW_HIPSOLVER_SOVERSION}/g" library/CMakeLists.txt

# for rocSOLVER package requirements
OLD_MINIMUM_ROCSOLVER_VERSION="3\.29\.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.30.0"
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" CMakeLists.txt
