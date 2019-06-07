#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

doxygen Doxyfile
