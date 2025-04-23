PREFIX ?= f2008

ifneq (,$(findstring sm_,$(CFLAGS))) # have CUDA arch flag
  TEST_COLLECTIONS = vecadd hipblas hipfft  
else
  TEST_COLLECTIONS = vecadd rocfft rocsparse rocsolver hipblas rocblas hipfft 
endif
RUN_TEST_COLLECTIONS   := $(addprefix run., $(TEST_COLLECTIONS))
CLEAN_TEST_COLLECTIONS := $(addprefix clean., $(TEST_COLLECTIONS))

.PHONY: build_all clean_all run_all build clean run $(TEST_COLLECTIONS) $(RUN_TEST_COLLECTIONS) $(CLEAN_TEST_COLLECTIONS)

build_all:
	+make -C . build  PREFIX=f2003
	+make -C . build  PREFIX=f2008
clean_all:
	+make -C . clean PREFIX=f2003
	+make -C . clean PREFIX=f2008
run_all:
	+make -C . run PREFIX=f2003
	+make -C . run PREFIX=f2008

$(TEST_COLLECTIONS):
	+make -C $(PREFIX)/$@ build

$(RUN_TEST_COLLECTIONS):
	+make -C $(PREFIX)/$(shell echo $@ | sed "s,run.,,g") run

$(CLEAN_TEST_COLLECTIONS):
	+make -C $(PREFIX)/$(shell echo $@ | sed "s,clean.,,g") clean

build: $(TEST_COLLECTIONS)

run: $(RUN_TEST_COLLECTIONS)

clean: $(CLEAN_TEST_COLLECTIONS)
