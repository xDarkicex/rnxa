// Package mps is the Metal Performance Shaders backend for rnxa.
//
// The darwin build of this package uses cgo to call into Apple's
// MPS primitives via an Objective-C++ shim (mps.mm). Non-darwin
// builds get a stub Backend whose New() returns an error.
//
// The Backend is FP32-only — float64 callers downcast on the way in
// and upcast on the way out. The precision roundtrip is documented
// in mps_ops.h.
//
// Layout:
//   mps.go         — public Go API + non-darwin stub (this file)
//   mps_darwin.go  — darwin cgo bridge; same dir as mps.mm
//   mps.mm         — Objective-C++ implementation
//   mps_test.go    — darwin-only tests
package mps

import "errors"

// Backend is the Metal Performance Shaders backend. The zero value is
// unusable; obtain one via New.
//
// The Backend holds lazy-initialised MPS device and command queue;
// the Go-side mps_darwin.go serialises calls through the engine's
// mutex, so the Backend itself is not thread-safe.
type Backend struct {
	// device and queue are opaque Go pointers to the Objective-C
	// MTLDevice and MTLCommandQueue. nil on non-darwin builds.
	device uintptr
	queue  uintptr
}

// New initialises the MPS backend. On non-darwin platforms it returns
// ("mps: not supported on this platform") so callers fall back to
// the pure-Go native CPU. On darwin with no Metal hardware it returns
// ("mps: no Metal device").
func New() (*Backend, error) {
	return newMPSBackend()
}

// Close releases the underlying device and queue. Safe to call
// multiple times. The Go-side engine owns the lifecycle, so callers
// should not call this directly.
func (b *Backend) Close() error {
	if b == nil {
		return nil
	}
	return closeMPSBackend(b)
}

// MatMul computes C = A @ B where A is M×K, B is K×N, C is M×N.
// Row-major. Returns an error if A, B, or C don't match the declared
// dimensions, or if the MPS kernel reports failure.
func (b *Backend) MatMul(A, B []float32, M, N, K int) ([]float32, error) {
	return mpsMatmul(b, A, B, M, N, K)
}

// ReLU applies max(0, x) elementwise. Returns a fresh slice.
func (b *Backend) ReLU(X []float32) ([]float32, error) {
	return mpsReLU(b, X)
}

// Sigmoid applies σ(x) = 1/(1+exp(-x)) elementwise.
func (b *Backend) Sigmoid(X []float32) ([]float32, error) {
	return mpsSigmoid(b, X)
}

// Tanh applies tanh(x) elementwise.
func (b *Backend) Tanh(X []float32) ([]float32, error) {
	return mpsTanh(b, X)
}

// Softmax applies softmax over the entire input. The input is treated
// as a flat 1D vector of length n. For tensor-axis softmax, reduce the
// input along the relevant axes before calling.
func (b *Backend) Softmax(X []float32) ([]float32, error) {
	return mpsSoftmax(b, X)
}

// ErrUnsupported is returned by New on non-darwin platforms.
var ErrUnsupported = errors.New("mps: not supported on this platform")
