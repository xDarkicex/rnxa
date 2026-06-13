# Makefile for the rnxa CUDA shim (libcuda.so).
#
# Builds cuda.cu (which transitively links cuBLAS and cuDNN) into
# a shared library that the Go side loads via purego. Targets:
#
#   make           — build build/libcuda.so
#   make install   — copy to /usr/local/lib and run ldconfig
#   make test      — run the Go test suite with LD_LIBRARY_PATH set
#   make clean     — remove build/
#
# nvcc must be on $PATH (typically /usr/local/cuda/bin/nvcc, which
# the CUDA toolkit installer adds to .bashrc / .zshrc).

NVCC    ?= nvcc
CFLAGS  ?= -O3 -Xcompiler -fPIC
LDFLAGS ?= -shared -lcublas -lcudnn

BUILD_DIR = build
LIB       = $(BUILD_DIR)/libcuda.so
SRC       = cuda.cu
HDR       = cuda_ops.h

.PHONY: all install test clean

all: $(LIB)

$(LIB): $(SRC) $(HDR)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -I. $(SRC) -o $(LIB) $(LDFLAGS)

install: $(LIB)
	cp $(LIB) /usr/local/lib/
	ldconfig

test: $(LIB)
	LD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR):$(LD_LIBRARY_PATH) \
		go test -count=1 ./internal/compute/cuda/...

clean:
	rm -rf $(BUILD_DIR)
