//go:build darwin
// +build darwin

package rnxa

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation

#include "internal/metal/metal_ops.h"
*/
import "C"
import (
	"unsafe"
)

// CGO wrapper functions - single point of integration
func metalCreateDevice() C.MTLDeviceRef {
	return C.metal_create_device()
}

func metalReleaseDevice(device interface{}) {
	C.metal_release_device(device.(C.MTLDeviceRef))
}

func metalCreateCommandQueue(device interface{}) C.MTLCommandQueueRef {
	return C.metal_create_command_queue(device.(C.MTLDeviceRef))
}

func metalReleaseCommandQueue(queue interface{}) {
	C.metal_release_command_queue(queue.(C.MTLCommandQueueRef))
}

func metalGetDeviceNameSafe(device interface{}) string {
	namePtr := C.get_device_name_safe(device.(C.MTLDeviceRef))
	return C.GoString(namePtr)
}

func metalGetDeviceCoresSafe(device interface{}) int {
	return int(C.get_device_cores_safe(device.(C.MTLDeviceRef)))
}

func metalGetTotalMemory(device interface{}) uint64 {
	return uint64(C.metal_get_total_memory(device.(C.MTLDeviceRef)))
}

func metalGetAvailableMemory(device interface{}) uint64 {
	return uint64(C.metal_get_available_memory(device.(C.MTLDeviceRef)))
}

func metalMatrixMultiply(device, queue interface{}, A []float32, M, K int, B []float32, K2, N int, C_result []float32) int {
	return int(C.metal_matrix_multiply(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&A[0])), C.int(M), C.int(K),
		(*C.float)(unsafe.Pointer(&B[0])), C.int(K2), C.int(N),
		(*C.float)(unsafe.Pointer(&C_result[0])),
	))
}

func metalVectorAdd(device, queue interface{}, A, B, C_result []float32, size int) int {
	return int(C.metal_vector_add(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_result[0])),
		C.int(size),
	))
}

func metalVectorSub(device, queue interface{}, A, B, C_result []float32, size int) int {
	return int(C.metal_vector_sub(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_result[0])),
		C.int(size),
	))
}

func metalVectorMul(device, queue interface{}, A, B, C_result []float32, size int) int {
	return int(C.metal_vector_mul(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&A[0])),
		(*C.float)(unsafe.Pointer(&B[0])),
		(*C.float)(unsafe.Pointer(&C_result[0])),
		C.int(size),
	))
}

func metalReLU(device, queue interface{}, input, output []float32, size int) int {
	return int(C.metal_relu(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(size),
	))
}

func metalSigmoid(device, queue interface{}, input, output []float32, size int) int {
	return int(C.metal_sigmoid(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(size),
	))
}

func metalTanh(device, queue interface{}, input, output []float32, size int) int {
	return int(C.metal_tanh(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(size),
	))
}

func metalSoftmax(device, queue interface{}, input, output []float32, size int) int {
	return int(C.metal_softmax(
		device.(C.MTLDeviceRef),
		queue.(C.MTLCommandQueueRef),
		(*C.float)(unsafe.Pointer(&input[0])),
		(*C.float)(unsafe.Pointer(&output[0])),
		C.int(size),
	))
}
