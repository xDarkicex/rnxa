//go:build darwin
// +build darwin

package backends

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation

#include "metal_ops.h"
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

func (e *metalEngine) MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error) {
	// Validate inputs
	if len(A.Shape()) != 2 || len(B.Shape()) != 2 {
		return nil, fmt.Errorf("MatMul requires 2D tensors")
	}

	M, K1 := A.Shape()[0], A.Shape()[1]
	K2, N := B.Shape()[0], B.Shape()[1]
	if K1 != K2 {
		return nil, fmt.Errorf("incompatible matrix dimensions: (%d,%d) × (%d,%d)", M, K1, K2, N)
	}

	// Create result tensor
	C_result := Zeros(M, N)

	// Metal Performance Shaders matrix multiplication
	result := C.metal_matrix_multiply(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&A.data[0])), C.int(M), C.int(K1),
		(*C.float)(unsafe.Pointer(&B.data[0])), C.int(K2), C.int(N),
		(*C.float)(unsafe.Pointer(&C_result.data[0])),
	)

	if result != 0 {
		return nil, fmt.Errorf("Metal matrix multiplication failed: %d", result)
	}

	return C_result, nil
}

func (e *metalEngine) ReLU(ctx context.Context, X *Tensor) (*Tensor, error) {
	result := Zeros(X.Shape()...)

	success := C.metal_relu(
		e.metalDevice,
		e.commandQueue,
		(*C.float)(unsafe.Pointer(&X.data[0])),
		(*C.float)(unsafe.Pointer(&result.data[0])),
		C.int(X.Size()),
	)

	if success != 0 {
		return nil, fmt.Errorf("Metal ReLU failed")
	}

	return result, nil
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
