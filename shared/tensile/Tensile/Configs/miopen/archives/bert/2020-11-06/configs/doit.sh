#!/bin/bash
for i in nn nt tn; do
	../../Tensile/bin/Tensile ${i}.yaml ${i} > ${i}.out 2>&1
done
