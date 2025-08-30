package rnxa

import (
	"context"
	"fmt"
)

// ComputeEngine provides hardware-accelerated tensor operations
// Universal interface - works with any Go ML framework
type ComputeEngine interface {
	// Core operations - immediate value for relux
	MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error)
	VectorAdd(ctx context.Context, A, B *Tensor) (*Tensor, error)
	VectorSub(ctx context.Context, A, B *Tensor) (*Tensor, error)
	VectorMul(ctx context.Context, A, B *Tensor) (*Tensor, error) // Element-wise

	// Activation functions - direct relux integration
	ReLU(ctx context.Context, X *Tensor) (*Tensor, error)
	Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error)
	Tanh(ctx context.Context, X *Tensor) (*Tensor, error)
	Softmax(ctx context.Context, X *Tensor) (*Tensor, error)

	// Reduction operations
	Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error)
	Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error)

	// Device management
	Device() Device
	Available() bool
	Memory() MemoryInfo
	Close() error
}

// Device represents compute hardware (M2, RTX 4090, etc.)
type Device struct {
	ID       int
	Name     string
	Type     DeviceType
	Memory   uint64 // Available memory in bytes
	Cores    int    // Compute units/cores
	Platform string // "Metal", "CUDA", "OpenCL", "CPU"
}

type DeviceType int

const (
	CPU DeviceType = iota
	GPU
	NPU // Neural Processing Unit (future)
)

type MemoryInfo struct {
	Total     uint64
	Available uint64
	Used      uint64
}

// NewEngine creates the best available compute engine
func NewEngine() (ComputeEngine, error) {
	// Auto-detect best device
	devices := DetectDevices()
	if len(devices) == 0 {
		return newCPUEngine(), nil // Always have CPU fallback
	}

	// Prioritize: Metal (M2) > CUDA > OpenCL > CPU
	for _, device := range devices {
		switch device.Platform {
		case "Metal":
			return newMetalEngine(device)
		case "CUDA":
			return newCUDAEngine(device)
		case "OpenCL":
			return newOpenCLEngine(device)
		}
	}

	return newCPUEngine(), nil
}

// NewEngineWithDevice creates engine for specific device
func NewEngineWithDevice(deviceID int) (ComputeEngine, error) {
	devices := DetectDevices()
	if deviceID >= len(devices) {
		return nil, fmt.Errorf("device %d not found", deviceID)
	}

	device := devices[deviceID]
	switch device.Platform {
	case "Metal":
		return newMetalEngine(device)
	case "CUDA":
		return newCUDAEngine(device)
	default:
		return newCPUEngine(), nil
	}
}

func newCUDAEngine(device Device) (ComputeEngine, error) {
	return nil, fmt.Errorf("CUDA support not implemented yet")
}

func newOpenCLEngine(device Device) (ComputeEngine, error) {
	return nil, fmt.Errorf("OpenCL support not implemented yet")
}
