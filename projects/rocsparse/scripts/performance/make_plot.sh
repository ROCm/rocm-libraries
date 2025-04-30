#!/usr/bin/env bash

# ########################################################################
# Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################


if [ $# -eq 0 ]; then
    echo "Please supply a rocSPARSE performance logfile as command line argument";
    exit 1
fi

echo "Parsing $1 ..."

# Read all lines of the logfile
readarray log < $1

# Store if GFlop/s pattern has been found
found=0

# Create temporary file
tmpfile=$(mktemp /tmp/$1.XXXXXX)

i=0
for line in "${log[@]}"; do
    # Extract the matrix names
    matrix=$(echo $line | sed -n 's/.*Reading matrix .\/matrices\/// ; s/.csr.*//p')

    if [ ! "$matrix" = "" ]; then
        echo "Extracting performance data of matrix $matrix"
        mat=$matrix
    fi

    # Split line into its strings
    array=($(echo $line | tr " " "\n"))

    # Find GFlop/s slot
    j=0
    for slot in "${array[@]}"; do
        if [ "$slot" = "GFlop/s" ]; then
            idx=$j
        fi
        j=$((j+1))
    done

    # Extract GFlop/s on pattern match
    if [ $found -eq 1 ]; then
        # Add data to tmpfile
        echo "$i $mat ${array[$idx]}" >> $tmpfile
        i=$((i+1))
    fi

    # Check for GFlop/s pattern
    if [ "${array[$idx]}" = "GFlop/s" ]; then
        found=1
    else
        found=0
    fi
done

# Plot the bar chart
i=$((i-1))
gnuplot -persist <<-EOFMarker
    reset
    set grid
    set style fill solid 0.2
    set term postscript eps enhanced color font 'Helvetica,12'
    set output "$1.eps"
    set termoption noenhanced
    set ylabel "GFlop/s"
    set xrange [-0.5:$i.5]
    set yrange [0:*]
    set offsets 0.25, 0.25, 10, 0
    set xtics rotate by -45
    set boxwidth 0.5
    set size ratio 0.35
    unset key
    plot "$tmpfile" using 1:3:xtic(2) with boxes
EOFMarker

rm "$tmpfile"
