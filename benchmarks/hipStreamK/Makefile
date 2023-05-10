CPP = hipcc
ROCWMMAPATH ?= /opt/rocm/include
CFLAGS = -fopenmp -std=c++17 -I$(ROCWMMAPATH) -lopenblas
all: streamK
streamK: simpleStreamK.cpp
	$(CPP) $(CFLAGS) -o streamK.exe simpleStreamK.cpp
clean:
	rm -f streamK.exe
