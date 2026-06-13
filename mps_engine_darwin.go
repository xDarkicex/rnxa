//go:build darwin
// +build darwin

// mps_engine_darwin.go — ComputeEngine wrapper around the mps
// subpackage. Adapts the `*rnxa.Tensor` API to the float32
// mps.Backend primitives and supplies a pure-Go implementation of
// VectorAdd/Sub/Mul (the .mm doesn't expose them; they're trivial
// element-wise loops that the GPU path doesn't need to accelerate).
//
// Mirrors the metal_darwin.go layout: the engine struct and its
// methods live in package rnxa at the repo root, while the low-level
// FFI lives in internal/compute/mps. Splitting them this way keeps
// the import graph acyclic (mps_engine imports internal/compute/mps
// for the Backend; internal/compute/mps does not import package rnxa).

package rnxa

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/xDarkicex/rnxa/alloc"
	"github.com/xDarkicex/rnxa/internal/compute/mps"
)

type mpsEngine struct {
	device Device
	mu     sync.Mutex
	closed bool
	b      *mps.Backend
}

func newMPSEngine(device Device) (ComputeEngine, error) {
	b, err := mps.New()
	if err != nil {
		return nil, fmt.Errorf("mps: %w (build libmps.dylib via `make` in internal/compute/mps)", err)
	}
	return &mpsEngine{device: device, b: b}, nil
}

func (e *mpsEngine) Device() Device { return e.device }
func (e *mpsEngine) Available() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.b != nil && !e.closed
}
func (e *mpsEngine) Memory() MemoryInfo { return MemoryInfo{} }

func (e *mpsEngine) Close() error {
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

func (e *mpsEngine) MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("mps engine is closed")
	}
	aShape := A.Shape()
	if len(aShape) != 2 {
		return nil, fmt.Errorf("mps matmul: A must be 2D, got %dD", len(aShape))
	}
	bShape := B.Shape()
	if len(bShape) != 2 {
		return nil, fmt.Errorf("mps matmul: B must be 2D, got %dD", len(bShape))
	}
	M, K := aShape[0], aShape[1]
	N := bShape[1]
	if bShape[0] != K {
		return nil, fmt.Errorf("mps matmul: A.K=%d, B.K=%d mismatch", K, bShape[0])
	}

	a32 := toFloat32(A)
	b32 := toFloat32(B)
	c32, err := e.b.MatMul(a32, b32, M, N, K)
	if err != nil {
		return nil, fmt.Errorf("mps matmul: %w", err)
	}
	return wrapFloat32(c32, []int{M, N}, A.DType()), nil
}

func (e *mpsEngine) VectorAdd(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	return e.eltwise(A, B, "add")
}

func (e *mpsEngine) VectorSub(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	return e.eltwise(A, B, "sub")
}

func (e *mpsEngine) VectorMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	return e.eltwise(A, B, "mul")
}

// eltwise is a pure-Go implementation of elementwise binary ops.
// The MPS subpackage's .mm doesn't expose vector_add/sub/mul; for
// the bias-add and scale ops the relux compute layer needs, the
// host loop is plenty fast (these tensors are KB-sized).
func (e *mpsEngine) eltwise(A, B *Tensor, op string) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("mps engine is closed")
	}
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("mps %s: size mismatch %d != %d", op, A.Size(), B.Size())
	}
	n := A.Size()
	ad := A.Data()
	bd := B.Data()
	out := alloc.Float64(n)
	switch op {
	case "add":
		for i := 0; i < n; i++ {
			out[i] = ad[i] + bd[i]
		}
	case "sub":
		for i := 0; i < n; i++ {
			out[i] = ad[i] - bd[i]
		}
	case "mul":
		for i := 0; i < n; i++ {
			out[i] = ad[i] * bd[i]
		}
	default:
		return nil, fmt.Errorf("mps: unknown op %q", op)
	}
	return NewTensor(out, A.Shape()...), nil
}

func (e *mpsEngine) ReLU(ctx context.Context, X *Tensor) (*Tensor, error) {
	return e.unary("relu", X, e.b.ReLU)
}

func (e *mpsEngine) Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error) {
	return e.unary("sigmoid", X, e.b.Sigmoid)
}

func (e *mpsEngine) Tanh(ctx context.Context, X *Tensor) (*Tensor, error) {
	return e.unary("tanh", X, e.b.Tanh)
}

// unary dispatches to the mps subpackage's float32 primitive. We
// copy the input because the .mm function reads the buffer
// synchronously, but the Go caller might mutate it after the call.
func (e *mpsEngine) unary(op string, X *Tensor,
	mpsFn func([]float32) ([]float32, error)) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("mps engine is closed")
	}
	x32 := toFloat32(X)
	y32, err := mpsFn(x32)
	if err != nil {
		return nil, fmt.Errorf("mps %s: %w", op, err)
	}
	return wrapFloat32(y32, X.Shape(), X.DType()), nil
}

func (e *mpsEngine) Softmax(ctx context.Context, X *Tensor) (*Tensor, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("mps engine is closed")
	}
	x32 := toFloat32(X)
	y32, err := e.b.Softmax(x32)
	if err != nil {
		return nil, fmt.Errorf("mps softmax: %w", err)
	}
	return wrapFloat32(y32, X.Shape(), X.DType()), nil
}

// Sum and Mean delegate to the rnxa package's axis-aware Go
// helpers (mps.mm does activations as plain C and doesn't expose
// per-axis reduce).
func (e *mpsEngine) Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return Sum(X, axis)
}

func (e *mpsEngine) Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return Mean(X, axis)
}

// wrapFloat32 wraps a float32 result back into a Tensor. If the
// caller expected float64, upcast.
func wrapFloat32(data []float32, shape []int, wantDType DataType) *Tensor {
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
