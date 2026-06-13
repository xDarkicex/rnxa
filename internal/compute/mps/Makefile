# Makefile — build libmps.dylib from mps.mm for the rnxa MPS backend.
#
# Usage:
#   make            # build libmps.dylib into ./build
#   make install    # copy to /usr/local/lib (optional; for system-wide access)
#   make test       # build and run the Go tests
#
# Runtime discovery: the Go side loads `libmps.dylib` from the
# library search path. For local dev, either set
# `DYLD_LIBRARY_PATH=$(pwd)/build` or `make install` to copy into
# /usr/local/lib.

CC      = clang++
CFLAGS  = -O3 -fPIC -fobjc-arc -Wall
LDFLAGS = -dynamiclib -framework Metal -framework MetalPerformanceShaders -framework Foundation

BUILD_DIR = build
LIB      = $(BUILD_DIR)/libmps.dylib
SRC      = mps.mm mps_ops.h

.PHONY: all install test clean

all: $(LIB)

$(LIB): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I. mps.mm -o $(LIB) $(LDFLAGS)
	@echo "Built $(LIB)"
	@echo "Run with: DYLD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR) go test ./..."

install: $(LIB)
	cp $(LIB) /usr/local/lib/
	@echo "Installed to /usr/local/lib/"

test: $(LIB)
	DYLD_LIBRARY_PATH=$(PWD)/$(BUILD_DIR) go test -count=1 ./...

clean:
	rm -rf $(BUILD_DIR)
