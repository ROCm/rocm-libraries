CPP = hipcc
ROCWMMAPATH ?= /opt/rocm/include
CFLAGS = -fopenmp -std=c++17 -I$(ROCWMMAPATH) -lopenblas
all: streamK
streamK: LDSStreamK.cpp
	$(CPP) $(CFLAGS) -o streamK.exe LDSStreamK.cpp
clean:
	rm -f streamK.exe
