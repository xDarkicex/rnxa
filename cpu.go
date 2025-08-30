package rnxa

import (
	"context"
	"fmt"
	"math"
)

type cpuEngine struct {
	device Device
}

func newCPUEngine() ComputeEngine {
	return &cpuEngine{
		device: Device{
			ID:       -1,
			Name:     "CPU",
			Type:     CPU,
			Platform: "CPU",
		},
	}
}

func (e *cpuEngine) MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if len(A.Shape()) != 2 || len(B.Shape()) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}

	M, K1 := A.Shape()[0], A.Shape()[1]
	K2, N := B.Shape()[0], B.Shape()[1]
	if K1 != K2 {
		return nil, fmt.Errorf("incompatible matrix dimensions")
	}

	result := Zeros(M, N)

	// Simple CPU matrix multiplication
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K1; k++ {
				sum += A.data[i*K1+k] * B.data[k*N+j]
			}
			result.data[i*N+j] = sum
		}
	}

	return result, nil
}

func (e *cpuEngine) VectorAdd(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match")
	}

	result := Zeros(A.Shape()...)
	for i := 0; i < A.Size(); i++ {
		result.data[i] = A.data[i] + B.data[i]
	}
	return result, nil
}

func (e *cpuEngine) VectorSub(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match")
	}

	result := Zeros(A.Shape()...)
	for i := 0; i < A.Size(); i++ {
		result.data[i] = A.data[i] - B.data[i]
	}
	return result, nil
}

func (e *cpuEngine) VectorMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match")
	}

	result := Zeros(A.Shape()...)
	for i := 0; i < A.Size(); i++ {
		result.data[i] = A.data[i] * B.data[i]
	}
	return result, nil
}

func (e *cpuEngine) ReLU(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)
	for i := 0; i < X.Size(); i++ {
		if X.data[i] > 0 {
			result.data[i] = X.data[i]
		} else {
			result.data[i] = 0
		}
	}
	return result, nil
}

func (e *cpuEngine) Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)
	for i := 0; i < X.Size(); i++ {
		result.data[i] = 1.0 / (1.0 + math.Exp(-X.data[i]))
	}
	return result, nil
}

func (e *cpuEngine) Tanh(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)
	for i := 0; i < X.Size(); i++ {
		result.data[i] = math.Tanh(X.data[i])
	}
	return result, nil
}

func (e *cpuEngine) Softmax(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	// Find max for numerical stability
	maxVal := X.data[0]
	for i := 1; i < X.Size(); i++ {
		if X.data[i] > maxVal {
			maxVal = X.data[i]
		}
	}

	// Compute exp and sum
	var sum float64
	for i := 0; i < X.Size(); i++ {
		result.data[i] = math.Exp(X.data[i] - maxVal)
		sum += result.data[i]
	}

	// Normalize
	for i := 0; i < X.Size(); i++ {
		result.data[i] /= sum
	}

	return result, nil
}

func (e *cpuEngine) Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	// Simple sum across all elements for now
	sum := 0.0
	for i := 0; i < X.Size(); i++ {
		sum += X.data[i]
	}
	return NewTensor([]float64{sum}), nil
}

func (e *cpuEngine) Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	sum := 0.0
	for i := 0; i < X.Size(); i++ {
		sum += X.data[i]
	}
	mean := sum / float64(X.Size())
	return NewTensor([]float64{mean}), nil
}

func (e *cpuEngine) Device() Device  { return e.device }
func (e *cpuEngine) Available() bool { return true }
func (e *cpuEngine) Memory() MemoryInfo {
	return MemoryInfo{Total: 8 * 1024 * 1024 * 1024, Available: 6 * 1024 * 1024 * 1024}
}
func (e *cpuEngine) Close() error { return nil }
