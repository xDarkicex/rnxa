//go:build !darwin
// +build !darwin

package rnxa

import "fmt"

// Stub implementation for non-Darwin systems
func newMetalEngine(device Device) (ComputeEngine, error) {
	return nil, fmt.Errorf("Metal support is only available on macOS")
}
