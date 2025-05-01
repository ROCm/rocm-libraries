#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.43.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.44.0"

OLD_TENSILE_HASH="a544a93867733afa2d32674bd29ba7bc6d7a3122"
NEW_TENSILE_HASH="2b4b0577c3f83ca9792de84ead368f9b62c09b13"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt
