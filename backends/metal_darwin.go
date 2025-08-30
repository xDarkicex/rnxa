//go:build darwin
// +build darwin

package rnxa

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation

#include "../internal/metal/metal_ops.h"
*/
import "C"
import (
	"context"
	"fmt"
	"unsafe"
)

type metalEngine struct {
	device       Device
	metalDevice  C.MTLDeviceRef
	commandQueue C.MTLCommandQueueRef
}

func newMetalEngine(device Device) (ComputeEngine, error) {
	metalDevice := C.metal_create_device()
	if metalDevice == nil {
		return nil, fmt.Errorf("failed to create Metal device")
	}

	commandQueue := C.metal_create_command_queue(metalDevice)
	if commandQueue == nil {
		C.metal_release_device(metalDevice)
		return nil, fmt.Errorf("failed to create Metal command queue")
	}

	return &metalEngine{
		device:       device,
		metalDevice:  metalDevice,
		commandQueue: commandQueue,
	}, nil
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

	result := C.metal_matrix_multiply(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&A_f32[0])), C.int(M), C.int(K1),
		(*C.float)(unsafe.Pointer(&B_f32[0])), C.int(K2), C.int(N),
		(*C.float)(unsafe.Pointer(&C_f32[0])),
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

	ret := C.metal_vector_add(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&A_f32[0])),
		(*C.float)(unsafe.Pointer(&B_f32[0])),
		(*C.float)(unsafe.Pointer(&C_f32[0])),
		C.int(A.Size()),
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

	ret := C.metal_vector_sub(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&A_f32[0])),
		(*C.float)(unsafe.Pointer(&B_f32[0])),
		(*C.float)(unsafe.Pointer(&C_f32[0])),
		C.int(A.Size()),
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

	ret := C.metal_vector_mul(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&A_f32[0])),
		(*C.float)(unsafe.Pointer(&B_f32[0])),
		(*C.float)(unsafe.Pointer(&C_f32[0])),
		C.int(A.Size()),
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

	success := C.metal_relu(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&X_f32[0])),
		(*C.float)(unsafe.Pointer(&Y_f32[0])),
		C.int(X.Size()),
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

	success := C.metal_sigmoid(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&X_f32[0])),
		(*C.float)(unsafe.Pointer(&Y_f32[0])),
		C.int(X.Size()),
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

	success := C.metal_tanh(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&X_f32[0])),
		(*C.float)(unsafe.Pointer(&Y_f32[0])),
		C.int(X.Size()),
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

	success := C.metal_softmax(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&X_f32[0])),
		(*C.float)(unsafe.Pointer(&Y_f32[0])),
		C.int(X.Size()),
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
		Total:     uint64(C.metal_get_total_memory(e.metalDevice)),
		Available: uint64(C.metal_get_available_memory(e.metalDevice)),
	}
}

func (e *metalEngine) Close() error {
	if e.commandQueue != nil {
		C.metal_release_command_queue(e.commandQueue)
	}
	if e.metalDevice != nil {
		C.metal_release_device(e.metalDevice)
	}
	return nil
}
