CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I. -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS= -arch=sm_89 -gencode arch=compute_89,code=sm_89 -O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/project

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/project: ./src/main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
