#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.40.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.41.0"

OLD_TENSILE_HASH="d2813593f97e892303c8870a3efb301d48ba52f3"
NEW_TENSILE_HASH="dbc2062dced66e4cbee8e0591d76e0a1588a4c70"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt
