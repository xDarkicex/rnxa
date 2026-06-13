// Package cuda is the NVIDIA CUDA backend for rnxa.
//
// The linux build of this package uses purego to call into a thin
// shim (libcuda.so, built from cuda.cu) that wraps cuBLAS for
// matrix multiplication and elementwise ops, and cuDNN for
// activations and softmax. Non-linux builds get a stub Backend
// whose New() returns ErrUnsupported; the rnxa dispatcher falls
// through to other backends.
//
// The Backend is FP32-only — float64 callers downcast on the way
// in and upcast on the way out (matching the MPS backend). The
// precision roundtrip is documented in cuda_ops.h.
//
// Layout:
//   cuda.go         — public Go API + non-linux stub
//   cuda_linux.go   — linux purego bridge; same dir as cuda.cu
//   cuda.cu         — CUDA C++ implementation
//   cuda_ops.h      — C ABI header
//   Makefile        — nvcc build
//   cuda_test.go    — linux-only tests
package cuda

import "errors"

// Backend is the CUDA backend. The zero value is unusable;
// obtain one via New.
//
// The Backend holds the opaque C ABI context (a cublasHandle_t +
// cudnnHandle_t pair owned by the shim) and a copy of every bound
// function pointer so a partial build (e.g. cuDNN missing at
// compile time) leaves the relevant fn* slot nil and the
// corresponding op returns a clear "unavailable" error at call
// time. The shim itself is not thread-safe; the Go-side
// cudaEngine serialises calls through its own mutex.
type Backend struct {
	ctx int64

	fnDeviceCount func(int64) int32
	fnDeviceName  func(int64, int32, *byte, int32) int32
	fnDeviceMem   func(int64, int32) int64
	fnMatmul      func(int64, int32, *float32, *float32, *float32, int64, int64, int64) int32
	fnEltwise     func(int64, int32, int32, *float32, *float32, *float32, int64) int32
	fnReLU        func(int64, int32, *float32, *float32, int64) int32
	fnSigmoid     func(int64, int32, *float32, *float32, int64) int32
	fnTanh        func(int64, int32, *float32, *float32, int64) int32
	fnSoftmax     func(int64, int32, *float32, *float32, int64) int32
}

// New initialises the CUDA backend. On non-linux platforms it
// returns ErrUnsupported so callers fall back to the next-best
// backend in the rnxa ladder. On linux without a built
// libcuda.so on the loader path, it returns a clear "build
// via make" error. On linux with the shim built but no NVIDIA
// device, it returns a "no NVIDIA GPU detected" error.
//
// The error ladder is deliberately granular so the user can
// tell "no GPU" from "cuDNN missing" from "lib not built".
func New() (*Backend, error) {
	return newCudaBackend()
}

// Close releases the underlying cuBLAS / cuDNN handles. Safe to
// call multiple times. The Go-side engine owns the lifecycle, so
// callers should not call this directly.
func (b *Backend) Close() error {
	if b == nil {
		return nil
	}
	return closeCudaBackend(b)
}

// HasHandles reports whether the shim reported a valid context.
// Available() on the engine wrapper uses this to decide whether
// the dispatcher should consider this engine "live".
func (b *Backend) HasHandles() bool { return b != nil && b.ctx != 0 }

// DeviceCount returns the number of NVIDIA GPUs the driver sees.
// First cut of the wrapper binds device 0 unconditionally; the
// count is exposed for diagnostics.
func (b *Backend) DeviceCount() (int, error) {
	if b == nil || b.ctx == 0 {
		return 0, errors.New("cuda: backend not initialised")
	}
	return int(b.fnDeviceCount(b.ctx)), nil
}

// DeviceName fills buf with the name of device idx and returns
// the number of bytes written (excluding the NUL).
func (b *Backend) DeviceName(idx int, buf []byte) (int, error) {
	if b == nil || b.ctx == 0 {
		return 0, errors.New("cuda: backend not initialised")
	}
	return int(b.fnDeviceName(b.ctx, int32(idx), &buf[0], int32(len(buf)))), nil
}

// DeviceMemory returns the total global memory of device idx in
// bytes.
func (b *Backend) DeviceMemory(idx int) (int64, error) {
	if b == nil || b.ctx == 0 {
		return 0, errors.New("cuda: backend not initialised")
	}
	return b.fnDeviceMem(b.ctx, int32(idx)), nil
}

// MatMul computes C = A @ B where A is M×K, B is K×N, C is M×N.
// Row-major. Returns an error if A, B, or C don't match the
// declared dimensions, or if the kernel reports failure.
func (b *Backend) MatMul(A, B []float32, M, N, K int) ([]float32, error) {
	return cudaMatmul(b, A, B, M, N, K)
}

// Eltwise applies the chosen op: 0 = add, 1 = sub, 2 = mul.
// Returns a fresh slice of length n.
func (b *Backend) Eltwise(A, B []float32, n int, op int) ([]float32, error) {
	return cudaEltwise(b, A, B, n, op)
}

// ReLU applies max(0, x) elementwise. Returns a fresh slice.
func (b *Backend) ReLU(X []float32) ([]float32, error) {
	return cudaReLU(b, X)
}

// Sigmoid applies σ(x) = 1/(1+exp(-x)) elementwise.
func (b *Backend) Sigmoid(X []float32) ([]float32, error) {
	return cudaSigmoid(b, X)
}

// Tanh applies tanh(x) elementwise.
func (b *Backend) Tanh(X []float32) ([]float32, error) {
	return cudaTanh(b, X)
}

// Softmax applies softmax over the entire input. The input is
// treated as a flat 1D vector of length n. For tensor-axis
// softmax, reduce the input along the relevant axes before
// calling.
func (b *Backend) Softmax(X []float32) ([]float32, error) {
	return cudaSoftmax(b, X)
}

// ErrUnsupported is returned by New on non-linux platforms.
var ErrUnsupported = errors.New("cuda: not supported on this platform")
