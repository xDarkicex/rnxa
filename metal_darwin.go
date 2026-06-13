//go:build darwin
// +build darwin

package rnxa

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

type metalEngine struct {
	device       Device
	metalDevice  interface{} // Store as interface{} to hold CGO type
	commandQueue interface{} // Store as interface{} to hold CGO type
	mu           sync.Mutex
	closed       bool
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

func (e *metalEngine) withResources(fn func(device, queue interface{}) error) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed || e.metalDevice == nil || e.commandQueue == nil {
		return errors.New("metal engine is closed")
	}

	return fn(e.metalDevice, e.commandQueue)
}

// Matrix multiplication - Core MLP operation
func (e *metalEngine) MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.DType() == Float32 && B.DType() == Float32 {
		return e.matMulFloat32(ctx, A, B)
	}

	if len(A.Shape()) != 2 || len(B.Shape()) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}

	M, K1 := A.Shape()[0], A.Shape()[1]
	K2, N := B.Shape()[0], B.Shape()[1]
	if K1 != K2 {
		return nil, fmt.Errorf("incompatible matrix dimensions: (%d,%d) × (%d,%d)", M, K1, K2, N)
	}

	C_result := Zeros(M, N)

	// Metal is a float32 API. Float64 inputs are converted to float32 for
	// the kernel and the result is upcast back. The conversion is lossy
	// (~7 decimal digits of precision); callers that need true float64
	// precision should use the CPU backend or pre-quantize their inputs.
	A_f32 := toFloat32(A)
	B_f32 := toFloat32(B)
	C_f32 := make([]float32, C_result.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		result := metalMatrixMultiply(
			device, queue,
			A_f32, M, K1,
			B_f32, K2, N,
			C_f32,
		)
		if result != 0 {
			return fmt.Errorf("Metal matrix multiplication failed: %d", result)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	// Convert back to float64 (lossy upcast, see precision note above).
	for i, v := range C_f32 {
		C_result.data[i] = float64(v)
	}

	return C_result, nil
}

func (e *metalEngine) matMulFloat32(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	_ = ctx

	if len(A.Shape()) != 2 || len(B.Shape()) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}

	M, K1 := A.Shape()[0], A.Shape()[1]
	K2, N := B.Shape()[0], B.Shape()[1]
	if K1 != K2 {
		return nil, fmt.Errorf("incompatible matrix dimensions: (%d,%d) × (%d,%d)", M, K1, K2, N)
	}

	AData := A.float32Data()
	BData := B.float32Data()
	resultTensor := ZerosFloat32(M, N)
	resultData := resultTensor.float32Data()
	if err := e.withResources(func(device, queue interface{}) error {
		result := metalMatrixMultiply(
			device, queue,
			AData, M, K1,
			BData, K2, N,
			resultData,
		)
		if result != 0 {
			return fmt.Errorf("Metal matrix multiplication failed: %d", result)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	return resultTensor, nil
}

// Vector operations for bias addition, etc.
func (e *metalEngine) VectorAdd(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("tensor sizes must match: %d != %d", A.Size(), B.Size())
	}

	result := Zeros(A.Shape()...)

	// See precision note in MatMul above.
	A_f32 := toFloat32(A)
	B_f32 := toFloat32(B)
	C_f32 := make([]float32, A.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		ret := metalVectorAdd(
			device, queue,
			A_f32, B_f32, C_f32, A.Size(),
		)
		if ret != 0 {
			return fmt.Errorf("Metal vector add failed: %d", ret)
		}
		return nil
	}); err != nil {
		return nil, err
	}

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

	// See precision note in MatMul above.
	A_f32 := toFloat32(A)
	B_f32 := toFloat32(B)
	C_f32 := make([]float32, A.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		ret := metalVectorSub(
			device, queue,
			A_f32, B_f32, C_f32, A.Size(),
		)
		if ret != 0 {
			return fmt.Errorf("Metal vector sub failed: %d", ret)
		}
		return nil
	}); err != nil {
		return nil, err
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

	// See precision note in MatMul above.
	A_f32 := toFloat32(A)
	B_f32 := toFloat32(B)
	C_f32 := make([]float32, A.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		ret := metalVectorMul(
			device, queue,
			A_f32, B_f32, C_f32, A.Size(),
		)
		if ret != 0 {
			return fmt.Errorf("Metal vector mul failed: %d", ret)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	for i, v := range C_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

// Activation functions for MLP layers
func (e *metalEngine) ReLU(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	// See precision note in MatMul above.
	X_f32 := toFloat32(X)
	Y_f32 := make([]float32, X.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		success := metalReLU(
			device, queue,
			X_f32, Y_f32, X.Size(),
		)
		if success != 0 {
			return fmt.Errorf("Metal ReLU failed")
		}
		return nil
	}); err != nil {
		return nil, err
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	// See precision note in MatMul above.
	X_f32 := toFloat32(X)
	Y_f32 := make([]float32, X.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		success := metalSigmoid(
			device, queue,
			X_f32, Y_f32, X.Size(),
		)
		if success != 0 {
			return fmt.Errorf("Metal Sigmoid failed")
		}
		return nil
	}); err != nil {
		return nil, err
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) Tanh(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	// See precision note in MatMul above.
	X_f32 := toFloat32(X)
	Y_f32 := make([]float32, X.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		success := metalTanh(
			device, queue,
			X_f32, Y_f32, X.Size(),
		)
		if success != 0 {
			return fmt.Errorf("Metal Tanh failed")
		}
		return nil
	}); err != nil {
		return nil, err
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

func (e *metalEngine) Softmax(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	// See precision note in MatMul above.
	X_f32 := toFloat32(X)
	Y_f32 := make([]float32, X.Size())

	if err := e.withResources(func(device, queue interface{}) error {
		success := metalSoftmax(
			device, queue,
			X_f32, Y_f32, X.Size(),
		)
		if success != 0 {
			return fmt.Errorf("Metal Softmax failed")
		}
		return nil
	}); err != nil {
		return nil, err
	}

	for i, v := range Y_f32 {
		result.data[i] = float64(v)
	}

	return result, nil
}

// Sum and Mean delegate to the CPU engine's axis-aware implementation;
// the Metal backend doesn't ship dedicated reduction kernels yet.
func (e *metalEngine) Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return newCPUEngine().Sum(ctx, X, axis)
}

func (e *metalEngine) Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error) {
	return newCPUEngine().Mean(ctx, X, axis)
}

func (e *metalEngine) Device() Device { return e.device }

func (e *metalEngine) Available() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return !e.closed && e.metalDevice != nil
}

func (e *metalEngine) Memory() MemoryInfo {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed || e.metalDevice == nil {
		return MemoryInfo{}
	}

	return MemoryInfo{
		Total:     metalGetTotalMemory(e.metalDevice),
		Available: metalGetAvailableMemory(e.metalDevice),
	}
}

func (e *metalEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}
	if e.commandQueue != nil {
		metalReleaseCommandQueue(e.commandQueue)
		e.commandQueue = nil
	}
	if e.metalDevice != nil {
		metalReleaseDevice(e.metalDevice)
		e.metalDevice = nil
	}
	e.closed = true
	return nil
}
