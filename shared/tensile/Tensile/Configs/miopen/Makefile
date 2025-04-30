################################################################################
#
# Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights reserved.
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

P=problems
DEEPBENCH_CONV_1x1=$P/nn/deepbench_conv_1x1_batchN.yml $P/nn/deepbench_conv_1x1_batch1.yml
RESNET=$P/nn/resnet_batch64_B.yml

# Override SCHED as vega10, vega20
SCHED=vega20

# commonly-used headers and footers:
HEADER=boiler/header.yml
FOOTER=boiler/library_logic_$(SCHED)_only.yml

# Override TYPE as sgemm, hgemm (hgemm_hpa, dgemm, etc in future)
TYPE=sgemm

# Override these to provide custom solution level or files
# Explore level 3,5,7 - by convention a higher level indicates more intensive search.
#  Only 3,5,7 currently supported.
#  Exact solution parms are defined in the specified solution file.
EXPLORE_LEVEL=5

# SOLUTION_LARGE is used for large matrix type
SOLUTION_LARGE=solutions/$(TYPE)_large_explore_$(EXPLORE_LEVEL).yml

# SOLUTION_SKINNY is used for problems with skinny M or N
SOLUTION_SKINNY=solutions/$(TYPE)_skinny_explore_$(EXPLORE_LEVEL).yml

# Targets specify different collections of solutions + problems

all: \
	$(TYPE)_resnet.yaml \
	$(TYPE)_resnet50_nn.yaml \
	$(TYPE)_resnet50_nt.yaml \
	$(TYPE)_resnet50_tn.yaml \
	$(TYPE)_deepbench_conv1x1.yaml \
	$(TYPE)_deepbench_gemm_nn.yaml \
	$(TYPE)_deepbench_gemm_nt.yaml \
	$(TYPE)_deepbench_gemm_tn.yaml

# Resnet
$(TYPE)_resnet.yaml: $(HEADER) types/$(TYPE)_nn.yml \
	$(SOLUTION_SKINNY) $(RESNET) \
	$(FOOTER)

# Resnet50
$(TYPE)_resnet50_nn.yaml: $(HEADER) types/$(TYPE)_nn.yml \
	$(SOLUTION_SKINNY) $P/nn/resnet50_all.yml \
	$(FOOTER)
$(TYPE)_resnet50_nt.yaml: $(HEADER) types/$(TYPE)_nt.yml \
	$(SOLUTION_SKINNY) $P/nt/resnet50_all.yml \
	$(FOOTER)
$(TYPE)_resnet50_tn.yaml: $(HEADER) types/$(TYPE)_tn.yml \
	$(SOLUTION_SKINNY) $P/tn/resnet50_all.yml \
	$(FOOTER)

# DeepBench Convolution:
$(TYPE)_deepbench_conv1x1.yaml: $(HEADER) types/$(TYPE)_nn.yml \
	$(SOLUTION_SKINNY) $(DEEPBENCH_CONV_1x1) \
	$(FOOTER)

# DeepBench GEMM
$(TYPE)_deepbench_gemm_nn.yaml: $(HEADER) types/$(TYPE)_nn.yml \
	$(SOLUTION_SKINNY) $P/nn/deepbench_gemm_skinny.yml \
	$(SOLUTION_LARGE) $P/nn/deepbench_gemm_large.yml \
	$(FOOTER)
$(TYPE)_deepbench_gemm_nt.yaml: $(HEADER) types/$(TYPE)_nt.yml \
	$(SOLUTION_LARGE) $P/nt/deepbench_gemm_large.yml \
	$(SOLUTION_SKINNY) $P/nt/deepbench_gemm_skinny.yml \
	$(FOOTER)
$(TYPE)_deepbench_gemm_tn.yaml: $(HEADER) types/$(TYPE)_tn.yml \
	$(SOLUTION_LARGE) $P/tn/deepbench_gemm_large.yml \
	$(SOLUTION_SKINNY) $P/tn/deepbench_gemm_skinny.yml \
	$(FOOTER)


.PHONY: clean
clean:
	rm -f *.yaml

# Common rule used for all 
%.yaml:
	cat $^ > $@
