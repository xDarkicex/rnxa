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
	if A.DType() == Float32 && B.DType() == Float32 {
		return matMulFloat32CPU(ctx, A, B)
	}

	if len(A.Shape()) != 2 || len(B.Shape()) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}

	M, K1 := A.Shape()[0], A.Shape()[1]
	K2, N := B.Shape()[0], B.Shape()[1]
	if K1 != K2 {
		return nil, fmt.Errorf("incompatible matrix dimensions")
	}

	AData := A.float64Data()
	BData := B.float64Data()
	result := Zeros(M, N)

	// Simple CPU matrix multiplication
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K1; k++ {
				sum += AData[i*K1+k] * BData[k*N+j]
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

	AData := toFloat64(A)
	BData := toFloat64(B)
	result := Zeros(A.Shape()...)
	for i := 0; i < A.Size(); i++ {
		result.data[i] = AData[i] + BData[i]
	}
	return result, nil
}

func (e *cpuEngine) VectorSub(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match")
	}

	AData := toFloat64(A)
	BData := toFloat64(B)
	result := Zeros(A.Shape()...)
	for i := 0; i < A.Size(); i++ {
		result.data[i] = AData[i] - BData[i]
	}
	return result, nil
}

func (e *cpuEngine) VectorMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match")
	}

	AData := toFloat64(A)
	BData := toFloat64(B)
	result := Zeros(A.Shape()...)
	for i := 0; i < A.Size(); i++ {
		result.data[i] = AData[i] * BData[i]
	}
	return result, nil
}

func (e *cpuEngine) ReLU(ctx context.Context, X *Tensor) (*Tensor, error) {
	XData := toFloat64(X)
	result := Zeros(X.Shape()...)
	for i := 0; i < X.Size(); i++ {
		if XData[i] > 0 {
			result.data[i] = XData[i]
		} else {
			result.data[i] = 0
		}
	}
	return result, nil
}

func (e *cpuEngine) Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error) {
	XData := toFloat64(X)
	result := Zeros(X.Shape()...)
	for i := 0; i < X.Size(); i++ {
		result.data[i] = 1.0 / (1.0 + math.Exp(-XData[i]))
	}
	return result, nil
}

func (e *cpuEngine) Tanh(ctx context.Context, X *Tensor) (*Tensor, error) {
	XData := toFloat64(X)
	result := Zeros(X.Shape()...)
	for i := 0; i < X.Size(); i++ {
		result.data[i] = math.Tanh(XData[i])
	}
	return result, nil
}

func (e *cpuEngine) Softmax(ctx context.Context, X *Tensor) (*Tensor, error) {
	XData := toFloat64(X)
	result := Zeros(X.Shape()...)

	// Find max for numerical stability
	maxVal := XData[0]
	for i := 1; i < X.Size(); i++ {
		if XData[i] > maxVal {
			maxVal = XData[i]
		}
	}

	// Compute exp and sum
	var sum float64
	for i := 0; i < X.Size(); i++ {
		result.data[i] = math.Exp(XData[i] - maxVal)
		sum += result.data[i]
	}

	// Normalize
	for i := 0; i < X.Size(); i++ {
		result.data[i] /= sum
	}

	return result, nil
}

func (e *cpuEngine) Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return reduceSum(X, axis)
}

func (e *cpuEngine) Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return reduceMean(X, axis)
}

func (e *cpuEngine) Device() Device  { return e.device }
func (e *cpuEngine) Available() bool { return true }
func (e *cpuEngine) Memory() MemoryInfo {
	// TODO: query actual host RAM via OS-specific syscalls
	// (syscall.Sysctl("hw.memsize") on darwin, /proc/meminfo on linux).
	return MemoryInfo{Total: 8 * 1024 * 1024 * 1024, Available: 6 * 1024 * 1024 * 1024}
}
func (e *cpuEngine) Close() error { return nil }
