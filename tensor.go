package rnxa

import (
	"fmt"

	"github.com/xDarkicex/rnxa/alloc"
)

// Tensor represents n-dimensional arrays with hardware acceleration.
// Data buffers are allocated from the off-heap pool (see the alloc
// package) so they do not contribute to the Go GC's working set.
type Tensor struct {
	data    []float64 // Host memory view for float64 consumers (off-heap)
	data32  []float32 // Optional float32 backing (off-heap)
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

// NewTensor wraps an existing float64 slice as a Tensor. The data is
// used as-is; it is NOT copied. For off-heap allocation use [Zeros] or
// pass a slice that was allocated via [alloc.Float64].
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

// NewTensorFromFloat32 wraps an existing float32 slice as a Tensor.
// See [NewTensor] for ownership semantics.
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

// Zeros returns a new zero-filled float64 tensor whose data lives in
// the off-heap pool.
func Zeros(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return NewTensor(alloc.Float64(size), shape...)
}

// Ones returns a new float64 tensor filled with 1.0s, allocated from
// the off-heap pool.
func Ones(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := alloc.Float64(size)
	for i := range data {
		data[i] = 1.0
	}
	return NewTensor(data, shape...)
}

// ZerosFloat32 returns a new zero-filled float32 tensor, off-heap.
func ZerosFloat32(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return NewTensorFromFloat32(alloc.Float32(size), shape...)
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

// ToDevice is currently a no-op: the ComputeEngine operates on host memory,
// and no GPU buffer management exists yet. Multi-device routing is tracked
// as a future enhancement. Returns the receiver unchanged.
func (t *Tensor) ToDevice(device Device) *Tensor { return t }

// ToHost is currently a no-op. Tensors always live in host memory; the
// ComputeEngine handles host<->device transfers internally. Returns the
// receiver unchanged.
func (t *Tensor) ToHost() *Tensor { return t }

// float64Data returns the tensor's float64 backing slice, or nil if the
// tensor was created as float32. This function never mutates the receiver:
// the conversion contract is "the requested dtype or nil" so callers can
// decide how to handle a dtype mismatch (typically via toFloat64).
func (t *Tensor) float64Data() []float64 {
	if t.data != nil {
		return t.data
	}
	return nil
}

// float32Data returns the tensor's float32 backing slice, or nil if the
// tensor was created as float64. Never mutates the receiver; see toFloat32
// for dtype-mismatch conversion.
func (t *Tensor) float32Data() []float32 {
	if t.data32 != nil {
		return t.data32
	}
	return nil
}

// toFloat64 returns x's data as a float64 slice. Allocates and converts if
// the tensor was created as float32; returns the backing slice directly if
// the tensor is already float64. Returns nil for an empty tensor.
func toFloat64(x *Tensor) []float64 {
	if x.data != nil {
		return x.data
	}
	if x.data32 == nil {
		return nil
	}
	out := make([]float64, len(x.data32))
	for i, v := range x.data32 {
		out[i] = float64(v)
	}
	return out
}

// toFloat32 returns x's data as a float32 slice. Allocates and converts if
// the tensor was created as float64; returns the backing slice directly if
// the tensor is already float32. Returns nil for an empty tensor.
func toFloat32(x *Tensor) []float32 {
	if x.data32 != nil {
		return x.data32
	}
	if x.data == nil {
		return nil
	}
	out := make([]float32, len(x.data))
	for i, v := range x.data {
		out[i] = float32(v)
	}
	return out
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
