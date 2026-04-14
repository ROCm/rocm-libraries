#!/bin/bash
for i in nn tn; do
	for j in spec2 speccd; do
		k=${j}-${i}-gfx908
		touch ${k}.bgn
		../../../../Tensile/bin/Tensile ${k}.yaml ${k} > ${k}.out 2>&1
		touch ${k}.end
	done
done
