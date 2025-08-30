package rnxa

import "fmt"

// Tensor represents n-dimensional arrays with hardware acceleration
type Tensor struct {
	data    []float64 // Host memory (always available)
	gpuData uintptr   // GPU memory handle (platform-specific)
	shape   []int     // Tensor dimensions [batch, height, width, channels]
	stride  []int     // Memory layout stride
	device  Device    // Which device owns this tensor
	dtype   DataType  // float32, float64, int32, etc.
}

type DataType int

const (
	Float32 DataType = iota
	Float64
	Int32
	Int64
)

// Creation functions
func NewTensor(data []float64, shape ...int) *Tensor {
	if len(shape) == 0 {
		shape = []int{len(data)} // 1D vector default
	}

	return &Tensor{
		data:   data,
		shape:  shape,
		stride: computeStride(shape),
		dtype:  Float64,
	}
}

func Zeros(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return NewTensor(make([]float64, size), shape...)
}

func Ones(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = 1.0
	}
	return NewTensor(data, shape...)
}

// Core tensor operations
func (t *Tensor) Shape() []int    { return t.shape }
func (t *Tensor) Size() int       { return len(t.data) }
func (t *Tensor) Data() []float64 { return t.data }
func (t *Tensor) Device() Device  { return t.device }

func (t *Tensor) Reshape(newShape ...int) *Tensor {
	// Verify compatible size
	oldSize, newSize := 1, 1
	for _, dim := range t.shape {
		oldSize *= dim
	}
	for _, dim := range newShape {
		newSize *= dim
	}

	if oldSize != newSize {
		panic(fmt.Sprintf("cannot reshape tensor of size %d to %d", oldSize, newSize))
	}

	return &Tensor{
		data:   t.data, // Same underlying data
		shape:  newShape,
		stride: computeStride(newShape),
		device: t.device,
		dtype:  t.dtype,
	}
}

// GPU memory management
func (t *Tensor) ToDevice(device Device) *Tensor {
	// Move tensor to specific device (implement per backend)
	// This is where Metal/CUDA-specific code lives
	return t
}

func (t *Tensor) ToHost() *Tensor {
	// Ensure data is in host memory
	return t
}

func computeStride(shape []int) []int {
	stride := make([]int, len(shape))
	if len(shape) == 0 {
		return stride
	}

	stride[len(stride)-1] = 1
	for i := len(stride) - 2; i >= 0; i-- {
		stride[i] = stride[i+1] * shape[i+1]
	}
	return stride
}
