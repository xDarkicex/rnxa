//go:build !linux
// +build !linux

// cuda_engine_other.go — non-linux stub for newCUDAEngine. The
// dispatcher in engine.go falls through to MPS/Metal/CPU on
// platforms where the CUDA .cu shim can't be built; this file
// exists purely so the symbol resolves at link time.

package rnxa

import "fmt"

func newCUDAEngine(device Device) (ComputeEngine, error) {
	return nil, fmt.Errorf("cuda: not supported on this platform")
}
