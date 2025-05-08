#!/usr/bin/env bash

# ########################################################################
# Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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


matrices=(circuit5M
          com-Orkut
          kron_g500-logn21
          kron_g500-logn20
          kron_g500-logn19
          bundle_adj
          SiO2
          mycielskian16
          mycielskian15
          mouse_gene
          av41092
          Maragal_8
          human_gene1
          human_gene2
          std1_Jac3
)

url=(https://sparse.tamu.edu/MM/Freescale
     https://sparse.tamu.edu/MM/SNAP
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/Mazaheri
     https://sparse.tamu.edu/MM/PARSEC
     https://sparse.tamu.edu/MM/Mycielski
     https://sparse.tamu.edu/MM/Mycielski
     https://sparse.tamu.edu/MM/Belcastro
     https://sparse.tamu.edu/MM/Vavasis
     https://sparse.tamu.edu/MM/NYPA
     https://sparse.tamu.edu/MM/Belcastro
     https://sparse.tamu.edu/MM/Belcastro
     https://sparse.tamu.edu/MM/VanVelzen
)

for i in {0..14}; do
    m=${matrices[${i}]}
    u=${url[${i}]}
    if [ ! -f ${m}.csr ]; then
        if [ ! -f ${m}.mtx ]; then
            if [ ! -f ${m}.tar.gz ]; then
                echo "Downloading ${m}.tar.gz ..."
                wget ${u}/${m}.tar.gz
            fi
            echo "Extracting ${m}.tar.gz ..."
            tar xf ${m}.tar.gz && mv ${m}/${m}.mtx . && rm -rf ${m}.tar.gz ${m}
        fi
        echo "Converting ${m}.mtx ..."
        ./convert ${m}.mtx ${m}.csr
        rm ${m}.mtx
    fi
done
