.PHONY: build run clean

build: main

main: main.f03 hip_implementation.cpp
	hipfc $(CFLAGS) main.f03 hip_implementation.cpp -o main -lroctx64

run: main
	rocprof -d rocprof_out -o rocprof_out/results.csv --hip-trace --roctx-trace ./main

clean:
	rm -rf main *.o *.mod rocprof_out
