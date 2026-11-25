CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall
NVCC     ?= nvcc
ARCH     ?= native

all: cpu

cpu: bin/v0_secuencial bin/v1_openmp

gpu: bin/v2_cuda

bin:
	mkdir -p bin

bin/v0_secuencial: src/v0_secuencial.cpp src/common.h | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin/v1_openmp: src/v1_openmp.cpp src/common.h | bin
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

bin/v2_cuda: src/v2_cuda.cu src/common.h | bin
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) $< -o $@

clean:
	rm -rf bin

.PHONY: all cpu gpu clean
