//go:build linux
// +build linux

// cuda_engine_linux.go — ComputeEngine wrapper around the cuda
// subpackage. Adapts the `*rnxa.Tensor` API to the float32
// cuda.Backend primitives and supplies a single Eltwise op
// (op=0/1/2 → add/sub/mul) routed through the .cu's custom
// kernel.
//
// Mirrors mps_engine_darwin.go: the engine struct and its
// methods live in package rnxa at the repo root, while the
// low-level FFI lives in internal/compute/cuda. Splitting them
// this way keeps the import graph acyclic (cuda_engine imports
// internal/compute/cuda for the Backend; internal/compute/cuda
// does not import package rnxa).

package rnxa

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/xDarkicex/rnxa/internal/compute/cuda"
)

type cudaEngine struct {
	device Device
	mu     sync.Mutex
	closed bool
	b      *cuda.Backend
}

func newCUDAEngine(device Device) (ComputeEngine, error) {
	b, err := cuda.New()
	if err != nil {
		return nil, fmt.Errorf("cuda: %w (build libcuda.so via `make` in internal/compute/cuda)", err)
	}
	return &cudaEngine{device: device, b: b}, nil
}

func (e *cudaEngine) Device() Device { return e.device }
func (e *cudaEngine) Available() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.b != nil && !e.closed
}
func (e *cudaEngine) Memory() MemoryInfo { return MemoryInfo{} }

func (e *cudaEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	if e.b != nil {
		_ = e.b.Close()
	}
	return nil
}

func (e *cudaEngine) MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("cuda engine is closed")
	}
	aShape := A.Shape()
	if len(aShape) != 2 {
		return nil, fmt.Errorf("cuda matmul: A must be 2D, got %dD", len(aShape))
	}
	bShape := B.Shape()
	if len(bShape) != 2 {
		return nil, fmt.Errorf("cuda matmul: B must be 2D, got %dD", len(bShape))
	}
	M, K := aShape[0], aShape[1]
	N := bShape[1]
	if bShape[0] != K {
		return nil, fmt.Errorf("cuda matmul: A.K=%d, B.K=%d mismatch", K, bShape[0])
	}

	a32 := toFloat32(A)
	b32 := toFloat32(B)
	c32, err := e.b.MatMul(a32, b32, M, N, K)
	if err != nil {
		return nil, fmt.Errorf("cuda matmul: %w", err)
	}
	return cudaWrapFloat32(c32, []int{M, N}, A.DType()), nil
}

func (e *cudaEngine) VectorAdd(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	return e.eltwiseCuda(A, B, 0) // 0 = add
}

func (e *cudaEngine) VectorSub(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	return e.eltwiseCuda(A, B, 1) // 1 = sub
}

func (e *cudaEngine) VectorMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	return e.eltwiseCuda(A, B, 2) // 2 = mul
}

// eltwiseCuda runs the custom CUDA kernel via the Eltwise ABI
// slot. A and B are converted to float32 on the host; the kernel
// is launched on the device; the result is wrapped back into a
// Tensor. Float64 callers get an upcast result; float32 callers
// get a Tensor whose data is the float32 kernel output.
func (e *cudaEngine) eltwiseCuda(A, B *Tensor, op int) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("cuda engine is closed")
	}
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("cuda eltwise: size mismatch %d != %d", A.Size(), B.Size())
	}
	n := A.Size()
	a32 := toFloat32(A)
	b32 := toFloat32(B)
	c32, err := e.b.Eltwise(a32, b32, n, op)
	if err != nil {
		return nil, fmt.Errorf("cuda eltwise: %w", err)
	}
	return cudaWrapFloat32(c32, A.Shape(), A.DType()), nil
}

func (e *cudaEngine) ReLU(ctx context.Context, X *Tensor) (*Tensor, error) {
	return e.unary("relu", X, e.b.ReLU)
}

func (e *cudaEngine) Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error) {
	return e.unary("sigmoid", X, e.b.Sigmoid)
}

func (e *cudaEngine) Tanh(ctx context.Context, X *Tensor) (*Tensor, error) {
	return e.unary("tanh", X, e.b.Tanh)
}

// unary dispatches to the cuda subpackage's float32 primitive.
// We copy the input because the .cu function reads the buffer
// asynchronously, but the Go caller might mutate it after the
// call. The shim's cudaStreamSynchronize(0) at the end of every
// op makes this safe — by the time the function returns, the
// GPU is done — but the input buffer is shared with the
// caller's Tensor, so we still copy on the way in to keep the
// caller's data immutable.
func (e *cudaEngine) unary(op string, X *Tensor,
	cudaFn func([]float32) ([]float32, error)) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("cuda engine is closed")
	}
	x32 := toFloat32(X)
	y32, err := cudaFn(x32)
	if err != nil {
		return nil, fmt.Errorf("cuda %s: %w", op, err)
	}
	return cudaWrapFloat32(y32, X.Shape(), X.DType()), nil
}

func (e *cudaEngine) Softmax(ctx context.Context, X *Tensor) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("cuda engine is closed")
	}
	x32 := toFloat32(X)
	y32, err := e.b.Softmax(x32)
	if err != nil {
		return nil, fmt.Errorf("cuda softmax: %w", err)
	}
	return cudaWrapFloat32(y32, X.Shape(), X.DType()), nil
}

// Sum and Mean delegate to the rnxa package's axis-aware Go
// helpers (the .cu doesn't expose per-axis reduce in the first
// cut). Same trade-off the mpsEngine makes.
func (e *cudaEngine) Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return Sum(X, axis)
}

func (e *cudaEngine) Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return Mean(X, axis)
}

// cudaWrapFloat32 wraps a float32 result back into a Tensor. If
// the caller expected float64, upcast. Renamed (vs the MPS
// wrapper's wrapFloat32) to avoid a duplicate-declaration clash
// when both engines' wrappers are linked into the same binary
// (which is currently impossible — MPS is darwin-only, CUDA is
// linux-only — but the rename is a small future-proofing win).
func cudaWrapFloat32(data []float32, shape []int, wantDType DataType) *Tensor {
	if wantDType == Float64 {
		out := Zeros(shape...)
		dst := out.Data()
		for i, v := range data {
			dst[i] = float64(v)
		}
		return out
	}
	return NewTensorFromFloat32(data, shape...)
}
