#!/bin/bash
for i in nn nt tn; do
	p=bert-${i}
	touch ${p}.bgn
	../Tensile/bin/Tensile ${p}.yaml ${p} > ${p}.out 2>&1
	touch ${p}.end
done
