CXX = g++
HIPCC = hipcc
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
HIPFLAGS = -std=c++17 -O2 --offload-arch=gfx950

HEADERS = mxfp6_types.hpp mxfp6_preprocess.hpp mxfp6_reference.hpp mxfp6_asm_utils.hpp

test_reference: test_reference.cpp $(filter-out mxfp6_asm_utils.hpp, $(HEADERS))
	$(CXX) $(CXXFLAGS) -o $@ $<

test_lds_shuffle: test_lds_shuffle.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

test_single_mfma: test_single_mfma.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

bench_lds_shuffle: bench_lds_shuffle.cpp $(HEADERS)
	$(HIPCC) $(HIPFLAGS) $< -o $@

test_cpu: test_reference
	./test_reference

test_gpu: test_lds_shuffle test_single_mfma
	./test_lds_shuffle
	./test_single_mfma

clean:
	rm -f test_reference test_lds_shuffle test_single_mfma bench_lds_shuffle

.PHONY: test_cpu test_gpu clean
