//go:build !darwin
// +build !darwin

package mps

import "errors"

// Stub implementations of the package-level functions for non-darwin
// platforms. They satisfy the Go signatures in mps.go so the package
// compiles on linux/windows; the Backend returned by newMPSBackend is
// unusable (Close is a no-op, all kernels return ErrUnsupported).

func newMPSBackend() (*Backend, error) {
	return nil, ErrUnsupported
}

func closeMPSBackend(b *Backend) error { return nil }

func mpsMatmul(b *Backend, A, B []float32, M, N, K int) ([]float32, error) {
	return nil, errors.New("mps matmul: not supported on this platform")
}

func mpsReLU(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("mps relu: not supported on this platform")
}

func mpsSigmoid(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("mps sigmoid: not supported on this platform")
}

func mpsTanh(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("mps tanh: not supported on this platform")
}

func mpsSoftmax(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("mps softmax: not supported on this platform")
}
