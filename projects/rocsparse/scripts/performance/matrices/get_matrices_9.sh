#!/usr/bin/env bash

# ########################################################################
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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


matrices=(g7jac040
          lp_pds_10
          windtunnel_evap2d
          bloweybl
          baxter
          YCheng
          c-39
          graphics
          conf5_0-4x4-22
          foldoc
          bodyy4
          aft01
          cavity19
          stokes64
          jan99jac080
          Trec12
          hangGlider_5
          lowThrust_4
          nasa2910
          wang4
          rajat22
          world
          mario001
          cz20468
          internet
          mk12-b3
          ak2010
          lhr10
          spmsrtls
          struct4
          sinc12
          raefsky1
          Reuters911
          waveguide3D
)

url=(https://sparse.tamu.edu/MM/Hollinger
     https://sparse.tamu.edu/MM/LPnetlib
     https://sparse.tamu.edu/MM/Grueninger
     https://sparse.tamu.edu/MM/GHS_indef
     https://sparse.tamu.edu/MM/Meszaros
     https://sparse.tamu.edu/MM/psse2
     https://sparse.tamu.edu/MM/Schenk_IBMNA
     https://sparse.tamu.edu/MM/Sumner
     https://sparse.tamu.edu/MM/QCD
     https://sparse.tamu.edu/MM/Pajek
     https://sparse.tamu.edu/MM/Pothen
     https://sparse.tamu.edu/MM/Okunbor
     https://sparse.tamu.edu/MM/DRIVCAV
     https://sparse.tamu.edu/MM/GHS_indef
     https://sparse.tamu.edu/MM/Hollinger
     https://sparse.tamu.edu/MM/JGD_Kocay
     https://sparse.tamu.edu/MM/VDOL
     https://sparse.tamu.edu/MM/VDOL
     https://sparse.tamu.edu/MM/Nasa
     https://sparse.tamu.edu/MM/Wang
     https://sparse.tamu.edu/MM/Rajat
     https://sparse.tamu.edu/MM/Meszaros
     https://sparse.tamu.edu/MM/GHS_indef
     https://sparse.tamu.edu/MM/CPM
     https://sparse.tamu.edu/MM/Pajek
     https://sparse.tamu.edu/MM/JGD_Homology
     https://sparse.tamu.edu/MM/DIMACS10
     https://sparse.tamu.edu/MM/Mallya
     https://sparse.tamu.edu/MM/GHS_indef
     https://sparse.tamu.edu/MM/Rothberg
     https://sparse.tamu.edu/MM/Hohn
     https://sparse.tamu.edu/MM/Simon
     https://sparse.tamu.edu/MM/Pajek
     https://sparse.tamu.edu/MM/FEMLAB
)

for i in {0..33}; do
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
