//go:build !darwin
// +build !darwin

// mps_engine_other.go — non-darwin stub for newMPSEngine. The
// dispatcher in engine.go falls through to Metal/CPU on platforms
// where the MPS .mm shim can't be built; this file exists purely so
// the symbol resolves at link time.

package rnxa

import "fmt"

func newMPSEngine(device Device) (ComputeEngine, error) {
	return nil, fmt.Errorf("mps: not supported on this platform")
}
