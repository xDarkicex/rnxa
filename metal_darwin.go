//go:build darwin
// +build darwin

package rnxa

import (
	"context"
	"fmt"
)

type metalEngine struct {
	device       Device
	metalDevice  interface{} // Store as interface{} to hold CGO type
	commandQueue interface{} // Store as interface{} to hold CGO type
}

func newMetalEngine(device Device) (ComputeEngine, error) {
	metalDevice := metalCreateDevice()
	if metalDevice == nil {
		return nil, fmt.Errorf("failed to create Metal device")
	}

	commandQueue := metalCreateCommandQueue(metalDevice)
	if commandQueue == nil {
		metalReleaseDevice(metalDevice)
		return nil, fmt.Errorf("failed to create Metal command queue")
	}

	return &metalEngine{
		device:       device,
		metalDevice:  metalDevice,  // Store CGO type directly
		commandQueue: commandQueue, // Store CGO type directly
	}, nil
}

// Helper function to get CGO device reference
func (e *metalEngine) getMetalDevice() interface{} {
	return e.metalDevice
}

// Helper function to get CGO command queue reference
func (e *metalEngine) getCommandQueue() interface{} {
	return e.commandQueue
}

// Matrix multiplication - Core MLP operation
func (e *metalEngine) MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if len(A.Shape()) != 2 || len(B.Shape()) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}

	M, K1 := A.Shape()[0], A.Shape()[1]
	K2, N := B.Shape()[0], B.Shape()[1]
	if K1 != K2 {
		return nil, fmt.Errorf("incompatible matrix dimensions: (%d,%d) × (%d,%d)", M, K1, K2, N)
	}

	C_result := Zeros(M, N)

	// Convert float64 to float32 for Metal
	A_f32 := make([]float32, len(A.data))
	B_f32 := make([]float32, len(B.data))
	C_f32 := make([]float32, len(C_result.data))

	for i, v := range A.data {
		A_f32[i] = float32(v)
	}
	for i, v := range B.data {
		B_f32[i] = float32(v)
	}

	result := metalMatrixMultiply(
		e.getMetalDevice(), e.getCommandQueue(),
		A_f32, M, K1,
		B_f32, K2, N,
		C_f32,
	)

	if result != 0 {
		return nil, fmt.Errorf("Metal matrix multiplication failed: %d", result)
	}

	// Convert back to float64
	for i, v := range C_f32 {
		C_result.data[i] = float64(v)
	}

	return C_result, nil
}

// Vector operations for bias addition, etc.
func (e *metalEngine) VectorAdd(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match: %d != %d", A.Size(), B.Size())
	}

	result := Zeros(A.Shape()...)

	// Convert to float32
	A_f32 := make([]float32, A.Size())
	B_f32 := make([]float32, B.Size())
	C_f32 := make([]float32, A.Size())

	for i, v := range A.data {
		A_f32[i] = float32(v)
	}
	for i, v := range B.data {
		B_f32[i] = float32(v)
	}

	ret := metalVectorAdd(
		e.getMetalDevice(), e.getCommandQueue(),
		A_f32, B_f32, C_f32, A.Size(),
	)

	if ret != 0 {
		return nil, fmt.Errorf("Metal vector add failed: %d", ret)
	}

	// Convert back to float64
	for i, v := range C_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) VectorSub(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match")
	}

	result := Zeros(A.Shape()...)

	A_f32 := make([]float32, A.Size())
	B_f32 := make([]float32, B.Size())
	C_f32 := make([]float32, A.Size())

	for i, v := range A.data {
		A_f32[i] = float32(v)
	}
	for i, v := range B.data {
		B_f32[i] = float32(v)
	}

	ret := metalVectorSub(
		e.getMetalDevice(), e.getCommandQueue(),
		A_f32, B_f32, C_f32, A.Size(),
	)

	if ret != 0 {
		return nil, fmt.Errorf("Metal vector sub failed: %d", ret)
	}

	for i, v := range C_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) VectorMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match")
	}

	result := Zeros(A.Shape()...)

	A_f32 := make([]float32, A.Size())
	B_f32 := make([]float32, B.Size())
	C_f32 := make([]float32, A.Size())

	for i, v := range A.data {
		A_f32[i] = float32(v)
	}
	for i, v := range B.data {
		B_f32[i] = float32(v)
	}

	ret := metalVectorMul(
		e.getMetalDevice(), e.getCommandQueue(),
		A_f32, B_f32, C_f32, A.Size(),
	)

	if ret != 0 {
		return nil, fmt.Errorf("Metal vector mul failed: %d", ret)
	}

	for i, v := range C_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

// Activation functions for MLP layers
func (e *metalEngine) ReLU(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	X_f32 := make([]float32, X.Size())
	Y_f32 := make([]float32, X.Size())

	for i, v := range X.data {
		X_f32[i] = float32(v)
	}

	success := metalReLU(
		e.getMetalDevice(), e.getCommandQueue(),
		X_f32, Y_f32, X.Size(),
	)

	if success != 0 {
		return nil, fmt.Errorf("Metal ReLU failed")
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	X_f32 := make([]float32, X.Size())
	Y_f32 := make([]float32, X.Size())

	for i, v := range X.data {
		X_f32[i] = float32(v)
	}

	success := metalSigmoid(
		e.getMetalDevice(), e.getCommandQueue(),
		X_f32, Y_f32, X.Size(),
	)

	if success != 0 {
		return nil, fmt.Errorf("Metal Sigmoid failed")
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) Tanh(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	X_f32 := make([]float32, X.Size())
	Y_f32 := make([]float32, X.Size())

	for i, v := range X.data {
		X_f32[i] = float32(v)
	}

	success := metalTanh(
		e.getMetalDevice(), e.getCommandQueue(),
		X_f32, Y_f32, X.Size(),
	)

	if success != 0 {
		return nil, fmt.Errorf("Metal Tanh failed")
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) Softmax(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	X_f32 := make([]float32, X.Size())
	Y_f32 := make([]float32, X.Size())

	for i, v := range X.data {
		X_f32[i] = float32(v)
	}

	success := metalSoftmax(
		e.getMetalDevice(), e.getCommandQueue(),
		X_f32, Y_f32, X.Size(),
	)

	if success != 0 {
		return nil, fmt.Errorf("Metal Softmax failed")
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

// Simple implementations for Sum/Mean
func (e *metalEngine) Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	// Use CPU fallback for now - focus on core MLP operations
	return newCPUEngine().Sum(ctx, X, axis)
}

func (e *metalEngine) Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	// Use CPU fallback for now - focus on core MLP operations
	return newCPUEngine().Mean(ctx, X, axis)
}

func (e *metalEngine) Device() Device  { return e.device }
func (e *metalEngine) Available() bool { return e.metalDevice != nil }

func (e *metalEngine) Memory() MemoryInfo {
	return MemoryInfo{
		Total:     metalGetTotalMemory(e.getMetalDevice()),
		Available: metalGetAvailableMemory(e.getMetalDevice()),
	}
}

func (e *metalEngine) Close() error {
	if e.commandQueue != nil {
		metalReleaseCommandQueue(e.getCommandQueue())
		e.commandQueue = nil
	}
	if e.metalDevice != nil {
		metalReleaseDevice(e.getMetalDevice())
		e.metalDevice = nil
	}
	return nil
}
