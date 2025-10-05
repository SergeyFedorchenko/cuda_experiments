# Usage:
#   make
#   make run SIZE=2048 BLOCK=32 ARCH=sm_86 SKIP_CPU=1
NVCC ?= nvcc
ARCH ?= sm_70
BLOCK ?= 32
CXXFLAGS := -O3 -std=c++17
NVFLAGS := -arch=$(ARCH) -Xcompiler="-fopenmp" --use_fast_math

BIN := build/matmul
SRC := src/matmul.cu

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p build
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -DBLOCK=$(BLOCK) -o $@ $<


run: $(BIN)
	@$(BIN) $(SIZE) $(SIZE) $(SIZE) $(SKIP_CPU)


#run: $(BIN)
#	@SIZE ?= 1024; \
#	SKIP_CPU ?= 0; \
#	$(BIN) $$SIZE $$SIZE $$SIZE $$SKIP_CPU

clean:
	rm -rf build

.PHONY: all run clean
