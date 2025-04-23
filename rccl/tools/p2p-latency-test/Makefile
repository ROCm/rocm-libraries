# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
HIP_PATH ?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH = ../../..
endif
HIPCC = $(HIP_PATH)/bin/hipcc

all: p2p_latency_test ll_latency_test

CXXFLAGS = -g -O3
p2p_latency_test: p2p_latency_test.cpp
	$(HIPCC) $(CXXFLAGS) $^ -o $@
ll_latency_test: ll_latency_test.cpp
	$(HIPCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -f *.o $(EXE)
