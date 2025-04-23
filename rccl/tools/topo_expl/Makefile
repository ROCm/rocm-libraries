# Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
HIP_PATH ?= $(wildcard /opt/rocm)
ifeq (,$(HIP_PATH))
HIP_PATH = ../../..
endif
HIPCC = $(HIP_PATH)/bin/hipcc

EXE = topo_expl
CXXFLAGS = -g -ffunction-sections -fdata-sections -Wl,--gc-sections -fgpu-rdc -Iinclude -Ihipify_rccl/include -Ihipify_rccl/device/include -Ihipify_rccl/graph -I/opt/rocm/include/ -DTOPO_EXPL -DENABLE_TRACE -DENABLE_LL128 -DNVTX_NO_IMPL -DRCCL_EXPOSE_STATIC -lpthread

files = $(EXE).cpp model.cpp utils.cpp hipify_rccl/graph/topo.cc hipify_rccl/graph/rings.cc hipify_rccl/graph/paths.cc hipify_rccl/graph/trees.cc ../../src/misc/param.cc \
	hipify_rccl/graph/search.cc hipify_rccl/graph/connect.cc hipify_rccl/graph/tuning.cc hipify_rccl/graph/xml.cc ../../src/misc/nvmlwrap_stub.cc hipify_rccl/graph/rome_models.cc hipify_rccl/graph/archinfo.cc \
	hipify_rccl/collectives.cc hipify_rccl/enqueue.cc ../../src/rccl_wrap.cc

all: hipify $(EXE)

$(EXE): $(files)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

hipify:
	rm -rf hipify_rccl
	mkdir -p hipify_rccl/device/include hipify_rccl/include/network/unpack
	cp -a ../../src/include/ hipify_rccl/
	cp -a ../../src/graph/ hipify_rccl/
	cp -a ../../src/device/*.h hipify_rccl/device/include
	cp -a ../../src/device/network/unpack/*.h hipify_rccl/include/network/unpack
	cp -a ../../src/enqueue.cc hipify_rccl/
	cp -a ../../src/collectives.cc hipify_rccl/
	cp -a ../../src/misc/archinfo.cc hipify_rccl/graph/
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/*.h
	hipify-perl -inplace -quiet-warnings hipify_rccl/device/include/*.h
	sed -i "s/template<typename T, typename RedOp>/template<typename T, typename RedOp, int COLL_UNROLL>/g" "hipify_rccl/device/include/common.h"
	sed -i "s/\\(struct RunWorkBatch<ncclFunc[^>]*\\)>*/\\1, COLL_UNROLL>/" "hipify_rccl/device/include/common.h"
	hipify-perl -inplace -quiet-warnings hipify_rccl/graph/*
	hipify-perl -inplace -quiet-warnings hipify_rccl/include/network/unpack/*
	hipify-perl -inplace -quiet-warnings hipify_rccl/*.cc

clean:
	rm -rf hipify_rccl
	rm -f *.o $(EXE)