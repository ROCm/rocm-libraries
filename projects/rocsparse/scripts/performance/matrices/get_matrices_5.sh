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


matrices=(fv2
          bcsstk24
          nemsemm2
          jan99jac100
          skirt
          torsion1
          copter1
          man_5976
          benzene
          s1rmq4m1
          epb3
          TF17
          neos1
          nemeth17
          tomographic1
          wv2010
          bibd_17_8
          thermomech_TC
          brack2
          Andrews
          bundle1
          ss1
          sx-superuser
          crystk02
          ex11
          xenon1
)

url=(https://sparse.tamu.edu/MM/Norris
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/Meszaros
     https://sparse.tamu.edu/MM/Hollinger
     https://sparse.tamu.edu/MM/Pothen
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/GHS_psdef
     https://sparse.tamu.edu/MM/HB
     https://sparse.tamu.edu/MM/PARSEC
     https://sparse.tamu.edu/MM/Cylshell
     https://sparse.tamu.edu/MM/Averous
     https://sparse.tamu.edu/MM/JGD_Forest
     https://sparse.tamu.edu/MM/Mittelmann
     https://sparse.tamu.edu/MM/Nemeth
     https://sparse.tamu.edu/MM/Clark
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/JGD_BIBD
     https://sparse.tamu.edu/MM/Botonakis
     https://sparse.tamu.edu/MM/AG-Monien
     https://sparse.tamu.edu/MM/Andrews
     https://sparse.tamu.edu/MM/Lourakis
     https://sparse.tamu.edu/MM/VLSI
     https://sparse.tamu.edu/MM/SNAP
     https://sparse.tamu.edu/MM/Boeing
     https://sparse.tamu.edu/MM/FIDAP
     https://sparse.tamu.edu/MM/Ronis
)

for i in {0..25}; do
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
