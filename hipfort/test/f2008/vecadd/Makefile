.PHONY: build run clean

build: main

main:
	hipfc $(CFLAGS) main.f03 hip_implementation.cpp -o main 

run: main
	./main

clean:
	rm -f main *.o *.mod
