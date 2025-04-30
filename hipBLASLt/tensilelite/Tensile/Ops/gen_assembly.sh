#!/bin/bash
################################################################################
#
# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

archStr=$1
dst=$2
venv=$3
build_id_kind=$4

rocm_path=/opt/rocm
if ! [ -z ${ROCM_PATH+x} ]; then
    rocm_path=${ROCM_PATH}
fi

toolchain=${rocm_path}/bin/amdclang++

. ${venv}/bin/activate

IFS=';' read -r -a archs <<< "$archStr"

for arch in "${archs[@]}"; do
    objs=()
    echo "Creating code object for arch ${arch}"
    for i in "256 4 1" "256 4 0"; do
        set -- $i
        s=$dst/L_$1_$2_$3_$arch.s
        o=$dst/L_$1_$2_$3_$arch.o
        python3 ./LayerNormGenerator.py -o $s -w $1 -c $2 --sweep-once $3 --arch $arch --toolchain $toolchain &
        objs+=($o)
    done
    for i in "16 16" "8 32" "4 64" "2 128" "1 256"; do
        set -- $i
        s=$dst/S_$1_$2_$arch.s
        o=$dst/S_$1_$2_$arch.o
        python3 ./SoftmaxGenerator.py -o $s -m $1 -n $2 --arch $arch --toolchain $toolchain &
        objs+=($o)
    done
    for i in "S S 256 4" "H H 256 4" "H S 256 4" "S H 256 4"; do
        set -- $i
        s=$dst/A_$1_$2_$3_$4_$arch.s
        o=$dst/A_$1_$2_$3_$4_$arch.o
        python3 ./AMaxGenerator.py -o $s -t $1 -d $2 -w $3 -c $4 --arch $arch --toolchain $toolchain &
        objs+=($o)
    done
    if [[ $arch =~ gfx94[0-9] ]]; then
        for i in "S S F8N 256 4" "S S B8N 256 4" "S H F8N 256 4" "S H B8N 256 4"; do
            set -- $i
            s=$dst/A_$1_$2_$3_$4_$5_$arch.s
            o=$dst/A_$1_$2_$3_$4_$5_$arch.o
            python3 ./AMaxGenerator.py --is-scale -o $s -t $1 -d $2 -s $3 -w $4 -c $5 --arch $arch --toolchain $toolchain &
            objs+=($o)
        done
    fi
    if [[ $arch =~ gfx950 ]]; then
        for i in "S S F8 256 4" "S S B8 256 4" "S H F8 256 4" "S H B8 256 4"; do
            set -- $i
            s=$dst/A_$1_$2_$3_$4_$5_$arch.s
            o=$dst/A_$1_$2_$3_$4_$5_$arch.o
            python3 ./AMaxGenerator.py --is-scale -o $s -t $1 -d $2 -s $3 -w $4 -c $5 --arch $arch --toolchain $toolchain &
            objs+=($o)
        done
    fi
    wait
    ${toolchain} -target amdgcn-amdhsa -Xlinker --build-id=$build_id_kind -o $dst/extop_$arch.co ${objs[@]}
    python3 ./ExtOpCreateLibrary.py --src=$dst --co=$dst/extop_$arch.co --output=$dst --arch=$arch
done

deactivate
