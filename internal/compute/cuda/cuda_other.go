//go:build !linux
// +build !linux

package cuda

import "errors"

// Stub implementations of the package-level functions for non-linux
// platforms. They satisfy the Go signatures in cuda.go so the package
// compiles on darwin/windows; the Backend returned by newCudaBackend
// is unusable (Close is a no-op, all kernels return ErrUnsupported).

func newCudaBackend() (*Backend, error) {
	return nil, ErrUnsupported
}

func closeCudaBackend(b *Backend) error { return nil }

func cudaMatmul(b *Backend, A, B []float32, M, N, K int) ([]float32, error) {
	return nil, errors.New("cuda matmul: not supported on this platform")
}

func cudaEltwise(b *Backend, A, B []float32, n, op int) ([]float32, error) {
	return nil, errors.New("cuda eltwise: not supported on this platform")
}

func cudaReLU(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("cuda relu: not supported on this platform")
}

func cudaSigmoid(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("cuda sigmoid: not supported on this platform")
}

func cudaTanh(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("cuda tanh: not supported on this platform")
}

func cudaSoftmax(b *Backend, X []float32) ([]float32, error) {
	return nil, errors.New("cuda softmax: not supported on this platform")
}
