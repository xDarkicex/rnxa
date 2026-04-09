package rnxa

import "fmt"

// Tensor represents n-dimensional arrays with hardware acceleration
type Tensor struct {
	data    []float64 // Host memory view for float64 consumers
	data32  []float32 // Optional float32 backing for fast paths
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

func NewTensorFromFloat32(data []float32, shape ...int) *Tensor {
	if len(shape) == 0 {
		shape = []int{len(data)}
	}

	return &Tensor{
		data32: data,
		shape:  shape,
		stride: computeStride(shape),
		dtype:  Float32,
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

func ZerosFloat32(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return NewTensorFromFloat32(make([]float32, size), shape...)
}

// Core tensor operations
func (t *Tensor) Shape() []int { return t.shape }
func (t *Tensor) Size() int {
	if t.dtype == Float32 && t.data32 != nil {
		return len(t.data32)
	}
	return len(t.float64Data())
}
func (t *Tensor) Data() []float64   { return t.float64Data() }
func (t *Tensor) Data32() []float32 { return t.float32Data() }
func (t *Tensor) Device() Device    { return t.device }
func (t *Tensor) DType() DataType   { return t.dtype }

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
		data:   t.data,
		data32: t.data32,
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

func (t *Tensor) float64Data() []float64 {
	if t.data != nil {
		return t.data
	}
	if t.data32 == nil {
		return nil
	}

	t.data = make([]float64, len(t.data32))
	for i, v := range t.data32 {
		t.data[i] = float64(v)
	}
	return t.data
}

func (t *Tensor) float32Data() []float32 {
	if t.data32 != nil {
		return t.data32
	}
	if t.data == nil {
		return nil
	}

	t.data32 = make([]float32, len(t.data))
	for i, v := range t.data {
		t.data32[i] = float32(v)
	}
	return t.data32
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
